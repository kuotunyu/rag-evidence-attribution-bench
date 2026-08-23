"""Annotation CLI exposes packaging and local serving without running either implicitly."""

from typer.core import TyperCommand, TyperGroup, TyperOption

from rag_evidence.cli import app


def test_annotation_cli_surface_is_explicit() -> None:
    root = app.registered_commands, app.registered_groups
    assert root
    command = TyperCommand(name="noop")
    del command
    from typer.main import get_command

    cli = get_command(app)
    assert isinstance(cli, TyperGroup)
    annotation = cli.commands["annotation"]
    assert isinstance(annotation, TyperGroup)
    assert set(annotation.commands) == {
        "adjudicate",
        "collect",
        "finalize-pilot",
        "package-pilot",
        "serve",
    }
    package = annotation.commands["package-pilot"]
    serve = annotation.commands["serve"]
    adjudicate = annotation.commands["adjudicate"]
    assert any(
        isinstance(param, TyperOption) and param.name == "config" for param in package.params
    )
    assert any(isinstance(param, TyperOption) and param.name == "package" for param in serve.params)
    assert {
        param.name for param in adjudicate.params if isinstance(param, TyperOption)
    } >= {"manifest", "effective", "state", "host", "port"}
