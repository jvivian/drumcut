"""Rich UI components for drumcut pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class PipelineStep:
    """A single step in the pipeline."""

    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    substeps: list[str] = field(default_factory=list)


class PipelineUI:
    """Rich UI for pipeline execution."""

    def __init__(self, title: str = "drumcut Pipeline"):
        self.console = Console()
        self.title = title
        self.steps: list[PipelineStep] = []
        self.current_step_idx: int = -1
        self.live: Live | None = None

        # Progress for substeps
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.current_task: TaskID | None = None

    def add_step(self, name: str, description: str) -> int:
        """Add a step to the pipeline."""
        step = PipelineStep(name=name, description=description)
        self.steps.append(step)
        return len(self.steps) - 1

    def _get_status_icon(self, status: StepStatus) -> str:
        """Get icon for step status."""
        icons = {
            StepStatus.PENDING: "[dim]○[/]",
            StepStatus.RUNNING: "[cyan]◉[/]",
            StepStatus.COMPLETE: "[green]✓[/]",
            StepStatus.SKIPPED: "[yellow]⊘[/]",
            StepStatus.ERROR: "[red]✗[/]",
        }
        return icons.get(status, "○")

    def _render_steps(self) -> Table:
        """Render the steps table."""
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Icon", width=3)
        table.add_column("Step", width=20)
        table.add_column("Status", ratio=1)

        for step in self.steps:
            icon = self._get_status_icon(step.status)
            name = step.name

            if step.status == StepStatus.RUNNING:
                name = f"[bold cyan]{name}[/]"
                status = f"[cyan]{step.message or step.description}[/]"
            elif step.status == StepStatus.COMPLETE:
                name = f"[green]{name}[/]"
                status = f"[green]{step.message or 'Done'}[/]"
            elif step.status == StepStatus.SKIPPED:
                name = f"[dim]{name}[/]"
                status = f"[dim]{step.message or 'Skipped'}[/]"
            elif step.status == StepStatus.ERROR:
                name = f"[red]{name}[/]"
                status = f"[red]{step.message or 'Error'}[/]"
            else:
                name = f"[dim]{name}[/]"
                status = f"[dim]{step.description}[/]"

            table.add_row(icon, name, status)

        return table

    def _render_substeps(self) -> Table | None:
        """Render substeps for the current running step."""
        if self.current_step_idx < 0 or self.current_step_idx >= len(self.steps):
            return None

        step = self.steps[self.current_step_idx]
        if step.status != StepStatus.RUNNING or not step.substeps:
            return None

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Icon", width=3)
        table.add_column("Substep", ratio=1)

        # Show recent substeps with checkmarks for completed ones
        for i, substep in enumerate(step.substeps):
            is_last = i == len(step.substeps) - 1
            if is_last:
                # Current substep - spinner style
                table.add_row("[cyan]›[/]", f"[cyan]{substep}[/]")
            else:
                # Completed substep
                table.add_row("[green]✓[/]", f"[dim]{substep}[/]")

        return table

    def _render(self) -> Panel:
        """Render the full UI."""
        steps_table = self._render_steps()
        substeps_table = self._render_substeps()

        # Build content with optional sections
        parts = [steps_table]

        # Add separator and substeps if present
        if substeps_table is not None or self.current_task is not None:
            parts.append("[dim]─" * 50 + "[/]")

        if substeps_table is not None:
            parts.append(substeps_table)

        if self.current_task is not None:
            parts.append(self.progress)

        content = Group(*parts)

        return Panel(
            content,
            title=f"[bold]{self.title}[/]",
            border_style="blue",
            padding=(1, 2),
        )

    def __enter__(self) -> PipelineUI:
        """Start the live display."""
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        """Stop the live display."""
        if self.live:
            self.live.__exit__(*args)

    def _update(self):
        """Update the live display."""
        if self.live:
            self.live.update(self._render())

    def start_step(self, step_idx: int, message: str = ""):
        """Mark a step as running."""
        self.current_step_idx = step_idx
        self.steps[step_idx].status = StepStatus.RUNNING
        self.steps[step_idx].message = message
        self._update()

    def add_substep(self, message: str):
        """Add a substep message to current step."""
        if 0 <= self.current_step_idx < len(self.steps):
            self.steps[self.current_step_idx].substeps.append(message)
            self._update()

    def update_step(self, message: str):
        """Update the current step's message."""
        if 0 <= self.current_step_idx < len(self.steps):
            self.steps[self.current_step_idx].message = message
            self._update()

    def complete_step(self, step_idx: int, message: str = ""):
        """Mark a step as complete."""
        self.steps[step_idx].status = StepStatus.COMPLETE
        self.steps[step_idx].message = message
        self._update()

    def skip_step(self, step_idx: int, message: str = ""):
        """Mark a step as skipped."""
        self.steps[step_idx].status = StepStatus.SKIPPED
        self.steps[step_idx].message = message or "Skipped"
        self._update()

    def error_step(self, step_idx: int, message: str = ""):
        """Mark a step as errored."""
        self.steps[step_idx].status = StepStatus.ERROR
        self.steps[step_idx].message = message
        self._update()

    def start_progress(self, description: str, total: int | None = None) -> TaskID:
        """Start a progress bar for the current step."""
        self.current_task = self.progress.add_task(description, total=total)
        self._update()
        return self.current_task

    def advance_progress(self, amount: int = 1):
        """Advance the current progress bar."""
        if self.current_task is not None:
            self.progress.advance(self.current_task, amount)
            self._update()

    def complete_progress(self):
        """Complete and remove the current progress bar."""
        if self.current_task is not None:
            self.progress.remove_task(self.current_task)
            self.current_task = None
            self._update()

    def print(self, message: str):
        """Print a message below the UI."""
        self.console.print(message)


def create_pipeline_ui(
    session_folder: str,
    output_dir: str,
    session_id: int | None = None,
) -> PipelineUI:
    """Create a pre-configured pipeline UI."""
    ui = PipelineUI(title="🥁 drumcut Pipeline")

    ui.add_step("1. Merge Video", "Merge GoPro chapters")
    ui.add_step("2. Mix Audio", "Mix L/R/MIDI tracks")
    ui.add_step("3. Sync A/V", "Sync audio to video")
    ui.add_step("4. Segment", "Detect song boundaries")
    ui.add_step("5. Group", "Cluster similar segments")

    return ui
