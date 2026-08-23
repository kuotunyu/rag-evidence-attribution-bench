"""Offline FastAPI application for a single blind annotation package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import ValidationError

from rag_evidence.annotation.assignment import AssignmentManifest, AssignmentPackageV2
from rag_evidence.annotation.models import AnswerabilityAnnotation
from rag_evidence.annotation.store import AnnotationStore
from rag_evidence.annotation.workflow import AdjudicationStore
from rag_evidence.errors import ArtifactError
from rag_evidence.storage.artifacts import read_records


def _load_package(path: Path) -> AssignmentPackageV2:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to load annotation package {path}: {exc}") from exc
    return AssignmentPackageV2.model_validate(payload)


def create_annotation_app(package_path: Path, state_dir: Path) -> FastAPI:
    package = _load_package(package_path)
    store = AnnotationStore(state_dir, package)
    app = FastAPI(
        title="RAG evidence annotation console",
        description="Offline, blind, append-only human annotation tool.",
        version="annotation-ui-v2",
    )
    app.state.annotation_store = store

    def no_store(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    def artifact_error(exc: ArtifactError) -> HTTPException:
        message = str(exc)
        if "not assigned" in message:
            return HTTPException(status_code=404, detail=message)
        if "already submitted" in message:
            return HTTPException(status_code=409, detail=message)
        return HTTPException(status_code=400, detail=message)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(response: Response) -> str:
        no_store(response)
        return Path(__file__).with_name("ui.html").read_text(encoding="utf-8")

    @app.get("/api/tasks")
    def tasks(response: Response) -> dict[str, object]:
        no_store(response)
        return package.blind_export()

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str, response: Response) -> dict[str, Any]:
        no_store(response)
        for item in package.tasks:
            if item.annotation_task_id == task_id:
                return item.model_dump(mode="json")
        raise HTTPException(status_code=404, detail="task is not assigned in this package")

    @app.get("/api/drafts/{task_id}")
    def draft(task_id: str, response: Response) -> dict[str, Any] | None:
        no_store(response)
        try:
            return store.load_draft(task_id)
        except ArtifactError as exc:
            raise artifact_error(exc) from exc

    @app.put("/api/drafts/{task_id}")
    def save_draft(task_id: str, payload: dict[str, Any], response: Response) -> dict[str, Any]:
        no_store(response)
        try:
            return store.save_draft(task_id, payload)
        except ArtifactError as exc:
            raise artifact_error(exc) from exc

    @app.post("/api/submissions", status_code=status.HTTP_201_CREATED)
    def submit(payload: dict[str, Any], response: Response) -> dict[str, Any]:
        no_store(response)
        try:
            return store.submit(payload).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
        except ArtifactError as exc:
            raise artifact_error(exc) from exc

    @app.post("/api/amendments", status_code=status.HTTP_201_CREATED)
    def amend(payload: dict[str, Any], response: Response) -> dict[str, Any]:
        no_store(response)
        try:
            return store.amend(payload).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
        except ArtifactError as exc:
            raise artifact_error(exc) from exc

    @app.get("/api/progress")
    def progress(response: Response) -> dict[str, int]:
        no_store(response)
        return store.progress()

    @app.get("/api/export/{kind}.jsonl", response_class=PlainTextResponse)
    def export_jsonl(kind: Literal["submissions", "amendments"], response: Response) -> str:
        no_store(response)
        records = store.export_records(kind)
        if not records:
            return ""
        return (
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for record in records
            )
            + "\n"
        )

    @app.get("/api/export/{kind}")
    def export_json(
        kind: Literal["submissions", "amendments"], response: Response
    ) -> list[dict[str, Any]]:
        no_store(response)
        return store.export_records(kind)

    return app


def create_adjudication_app(
    manifest_path: Path,
    submissions_path: Path,
    state_dir: Path,
) -> FastAPI:
    try:
        manifest = AssignmentManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ArtifactError(f"failed to load assignment manifest {manifest_path}: {exc}") from exc
    submissions = tuple(
        AnswerabilityAnnotation.model_validate(payload)
        for payload in read_records(submissions_path)
    )
    store = AdjudicationStore(state_dir, manifest, submissions)
    tasks = {
        task.annotation_task_id: task for package in manifest.packages for task in package.tasks
    }
    app = FastAPI(
        title="RAG evidence adjudication console",
        description="Offline third-human disagreement resolution.",
        version="adjudication-ui-v1",
    )
    app.state.adjudication_store = store

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(response: Response) -> str:
        response.headers["Cache-Control"] = "no-store"
        return Path(__file__).with_name("adjudicator_ui.html").read_text(encoding="utf-8")

    @app.get("/api/disagreements")
    def disagreements(response: Response) -> list[dict[str, Any]]:
        response.headers["Cache-Control"] = "no-store"
        return [
            {
                **case.model_dump(mode="json"),
                "task": tasks[case.annotation_task_id].model_dump(mode="json"),
            }
            for case in store.cases.values()
        ]

    @app.get("/api/adjudications")
    def adjudications(response: Response) -> list[dict[str, Any]]:
        response.headers["Cache-Control"] = "no-store"
        return [record.model_dump(mode="json") for record in store.records()]

    @app.get("/api/status")
    def adjudication_status(response: Response) -> dict[str, int | bool]:
        response.headers["Cache-Control"] = "no-store"
        return store.status()

    @app.get("/api/export/adjudications.jsonl", response_class=PlainTextResponse)
    def export_adjudications_jsonl(response: Response) -> str:
        response.headers["Cache-Control"] = "no-store"
        try:
            return store.export_jsonl()
        except ArtifactError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/adjudications", status_code=status.HTTP_201_CREATED)
    def submit_adjudication(payload: dict[str, Any], response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            return store.submit(payload).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
        except ArtifactError as exc:
            code = 409 if "already adjudicated" in str(exc) else 400
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    return app
