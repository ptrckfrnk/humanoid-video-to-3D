#!/usr/bin/env python3
"""
Pre-download model checkpoints so the first run.py call is instant.

Run once after install.sh:
    python scripts/download_models.py
"""

import sys
from rich.console import Console

console = Console()

console.print()
console.rule("[bold]Pre-downloading model checkpoints[/bold]")
console.print()


def download_vggt() -> None:
    from huggingface_hub import snapshot_download
    console.print("[bold]1 / 2[/bold]  VGGT-1B (CVPR 2025 Best Paper)  [dim]~5 GB[/dim]")
    snapshot_download("facebook/VGGT-1B", repo_type="model")
    console.print("       [green]✓  Done[/green]")


def download_sam2() -> None:
    from huggingface_hub import hf_hub_download
    console.print("[bold]2 / 2[/bold]  SAM 2.1 Hiera-Small  [dim]~185 MB[/dim]")
    hf_hub_download("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt")
    console.print("       [green]✓  Done[/green]")


def download_vggt_omega() -> None:
    import torch
    if not torch.cuda.is_available():
        console.print("[dim]Skipping VGGT-Omega (no CUDA detected)[/dim]")
        return
    try:
        from huggingface_hub import hf_hub_download
        console.print("[bold]   +[/bold]  VGGT-Omega (CVPR 2026 Oral)  [dim]~5 GB[/dim]")
        hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.safetensors")
        console.print("       [green]✓  Done[/green]")
    except Exception as e:
        console.print(f"       [yellow]⚠  Skipped VGGT-Omega: {e}[/yellow]")


try:
    download_vggt()
    download_sam2()
    download_vggt_omega()
    console.print()
    console.rule("[bold green]All checkpoints ready[/bold green]")
    console.print()
except KeyboardInterrupt:
    console.print("\n[yellow]Interrupted.[/yellow]")
    sys.exit(1)
except Exception as e:
    console.print(f"\n[red]Error:[/red] {e}")
    console.print("Check your internet connection and HuggingFace token (if models are gated).")
    sys.exit(1)
