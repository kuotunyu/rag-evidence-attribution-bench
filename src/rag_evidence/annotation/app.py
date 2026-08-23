"""Offline FastAPI application for a single blind annotation package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import ValidationError

from rag_evidence.annotation.assignment import AssignmentPackage
from rag_evidence.annotation.store import AnnotationStore
from rag_evidence.errors import ArtifactError


def _load_package(path: Path) -> AssignmentPackage:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to load annotation package {path}: {exc}") from exc
    return AssignmentPackage.model_validate(payload)


def create_annotation_app(package_path: Path, state_dir: Path) -> FastAPI:
    package = _load_package(package_path)
    store = AnnotationStore(state_dir, package)
    app = FastAPI(
        title="RAG evidence annotation console",
        description="Offline, blind, append-only human annotation tool.",
        version="annotation-ui-v1",
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
