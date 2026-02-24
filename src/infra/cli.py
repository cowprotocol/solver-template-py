"""
Command line interface for the baseline solver.

This module provides CLI functionality using typer.
"""

import typer
from typing import Optional
import uvicorn
from .settings import settings
from .logging import setup_logging, log_startup, log_shutdown


app = typer.Typer(
    name="cow-solver-baseline",
    help="CoW Protocol Baseline Solver",
    add_completion=False,
)


@app.command()
def run(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Server host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Server port"),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l", help="Log level"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    access_log: bool = typer.Option(True, "--access-log", help="Enable access logging"),
):
    """
    Run the baseline solver server.
    """
    # Override with CLI args if provided
    if host:
        settings.host = host
    if port:
        settings.port = port
    if log_level:
        settings.log_level = log_level

    # Setup logging
    setup_logging()

    # Log startup
    log_startup("CoW Protocol Solver", "1.0.0", settings.host, settings.port)

    try:
        # Run the server
        uvicorn.run(
            "src.api.app:app",
            host=settings.host,
            port=settings.port,
            workers=workers,
            reload=reload,
            access_log=access_log,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        log_shutdown("CoW Protocol Solver")
    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def health():
    """
    Check the health of the solver service.
    """
    # TODO: Health check not implemented yet
    typer.echo("Health check not implemented yet")


@app.command()
def metrics():
    """
    Display solver metrics.
    """
    # TODO: Metrics display not implemented yet
    typer.echo("Metrics display not implemented yet")


@app.command()
def config():
    """
    Display current configuration.
    """
    typer.echo("Current configuration:")
    typer.echo(f"  Host: {settings.host}")
    typer.echo(f"  Port: {settings.port}")
    typer.echo(f"  Log Level: {settings.log_level}")
    typer.echo(f"  Default Solver Route: {settings.default_solver_route}")
    typer.echo("")
    typer.echo("Solver Configuration:")
    typer.echo(f"  Chain ID: {settings.solver.chain_id}")
    typer.echo(f"  WETH Address: {settings.solver.weth_address}")
    typer.echo(f"  Max Hops: {settings.solver.max_hops}")
    typer.echo(f"  Max Partial Attempts: {settings.solver.max_partial_attempts}")
    typer.echo(f"  Solution Gas Offset: {settings.solver.solution_gas_offset}")
    typer.echo(f"  Base Tokens: {len(settings.solver.base_tokens)} tokens")
    if settings.solver.uni_v3_quoter_address:
        typer.echo(f"  Uniswap V3 Quoter: {settings.solver.uni_v3_quoter_address}")
    else:
        typer.echo("  Uniswap V3 Quoter: Not configured")


@app.command()
def version():
    """
    Display solver version.
    """
    typer.echo("CoW Protocol Baseline Solver v1.0.0")


if __name__ == "__main__":
    app()
