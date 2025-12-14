from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


def _wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for part in text.splitlines():
        part = part.strip()
        if not part:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fc: str = "#F7F7FB",
    title_fs: float = 12.0,
    body_fs: float = 10.5,
    wrap_width: int = 38,
    title_center: bool = False,
) -> None:
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor="#2F2F2F", facecolor=fc))
    if title_center:
        ax.text(x + w * 0.5, y + h * 0.88, title, fontsize=title_fs, fontweight="bold", va="top", ha="center")
    else:
        ax.text(x + w * 0.03, y + h * 0.86, title, fontsize=title_fs, fontweight="bold", va="top")
    ax.text(
        x + w * 0.03,
        y + h * 0.72,
        _wrap(body, wrap_width),
        fontsize=body_fs,
        va="top",
        linespacing=1.25,
    )


def _arrow(ax, x1, y1, x2, y2, *, text: str | None = None, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.2,
            color="#2F2F2F",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, fontsize=9.5, ha="center", va="bottom")


def _arrow_orth(ax, x1: float, y1: float, x_mid: float, y2: float, x2: float, *, lw: float = 1.2) -> None:
    """
    Orthogonal connector routed via (x_mid, y1) and (x_mid, y2), with arrow head on the final segment.
    This improves readability and avoids line crossings in dense diagrams.
    """
    ax.add_line(
        Line2D(
            [x1, x_mid, x_mid, x2],
            [y1, y1, y2, y2],
            linewidth=lw,
            color="#2F2F2F",
            solid_joinstyle="round",
            solid_capstyle="round",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (x_mid, y2),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=lw,
            color="#2F2F2F",
        )
    )


def _arrow_vertical(ax, x: float, y1: float, y2: float, *, lw: float = 1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x, y1),
            (x, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=lw,
            color="#2F2F2F",
        )
    )


def generate(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "fig_A_information_flow.svg"
    out_png = out_dir / "fig_A_information_flow.png"

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig = plt.figure(figsize=(14.5, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headers (centered over columns)
    ax.text(0.18, 0.94, "Входные данные (Stratasys Insight)", fontsize=12.5, fontweight="bold", ha="center")
    ax.text(0.50, 0.94, "Обработка (slice2solid)", fontsize=12.5, fontweight="bold", ha="center")
    ax.text(0.83, 0.94, "Выходы и применение", fontsize=12.5, fontweight="bold", ha="center")

    # Left column (inputs)
    _box(
        ax,
        0.05,
        0.68,
        0.26,
        0.20,
        "Placed STL",
        "Геометрия после размещения детали в Insight (координатный эталон)",
        fc="#EEF6FF",
        wrap_width=34,
        title_center=True,
    )
    _box(
        ax,
        0.05,
        0.44,
        0.26,
        0.20,
        "Экспорт траекторий печати (*.txt)",
        "Insight → Toolpaths → Simulation data export\nПоля: X, Y, Z; тип траектории; площадь сечения и др.",
        fc="#EEF6FF",
        wrap_width=34,
        title_center=True,
    )

    # Middle column (processing)
    _box(
        ax,
        0.36,
        0.74,
        0.28,
        0.14,
        "Импорт и интерпретация",
        "Согласование систем координат; фильтрация вспомогательных траекторий",
        fc="#F3F0FF",
        wrap_width=40,
        title_center=True,
    )
    _box(
        ax,
        0.36,
        0.56,
        0.28,
        0.14,
        "Реконструкция структуры",
        "Объёмное представление (воксели)\n→ извлечение поверхности (marching cubes)",
        fc="#F3F0FF",
        wrap_width=40,
        title_center=True,
    )
    _box(
        ax,
        0.36,
        0.38,
        0.28,
        0.14,
        "Оценка ориентаций для CAE",
        "Доминирующие направления по слоям; показатели достоверности оценки",
        fc="#F3F0FF",
        wrap_width=40,
        title_center=True,
    )

    # Right column (outputs)
    _box(
        ax,
        0.69,
        0.62,
        0.28,
        0.26,
        "Выходные файлы slice2solid",
        "preview_structure.stl — сеточная модель структуры\n"
        "ansys_layers.json / ansys_layers.csv — таблица ориентаций\n"
        "ansys_mechanical_import_layers.py — импорт в ANSYS Mechanical\n"
        "metadata.json — параметры и статистика обработки",
        fc="#FFF5E6",
        wrap_width=48,
        body_fs=10.2,
        title_center=True,
    )

    # Branches
    _box(
        ax,
        0.69,
        0.33,
        0.28,
        0.20,
        "CAD-ветвь",
        "Конвертация сетки в твердое тело (nTop или альтернативы)\n"
        "→ экспорт Parasolid (*.x_t) или STEP (*.stp)\n"
        "→ импорт в SolidWorks",
        fc="#FFF5E6",
        wrap_width=44,
        title_center=True,
    )
    _box(
        ax,
        0.69,
        0.08,
        0.28,
        0.20,
        "CAE-ветвь",
        "Импорт в ANSYS Mechanical\n"
        "→ назначение ориентированных свойств\n"
        "→ расчёт напряжений и деформаций",
        fc="#FFF5E6",
        wrap_width=44,
        title_center=True,
    )

    # Arrows: inputs -> processing
    _arrow(ax, 0.31, 0.78, 0.36, 0.81, rad=0.0)
    _arrow(ax, 0.31, 0.54, 0.36, 0.63, rad=0.0)
    _arrow(ax, 0.50, 0.74, 0.50, 0.70)
    _arrow(ax, 0.50, 0.56, 0.50, 0.52)

    # Processing -> outputs (orthogonal, rounded corners)
    bus_x = 0.665  # left of outputs box
    out_left_x = 0.69
    _arrow_orth(ax, 0.64, 0.81, bus_x, 0.80, out_left_x)
    _arrow_orth(ax, 0.64, 0.63, bus_x, 0.74, out_left_x)
    _arrow_orth(ax, 0.64, 0.45, bus_x, 0.68, out_left_x)

    # Outputs -> branches
    # CAD branch: direct vertical connector between boxes (no overlap with text).
    _arrow(ax, 0.83, 0.62, 0.83, 0.53)

    # CAE branch: route from the RIGHT side of outputs, go down along the outer margin,
    # then enter the CAE box from the right (as in the markup).
    outputs_right_x = 0.97
    outputs_mid_y = 0.62 + 0.26 * 0.5  # center of outputs box
    cae_right_x = 0.97
    cae_enter_y = 0.08 + 0.20 * 0.5  # center of CAE box
    outer_x = 0.995  # just outside the right edge of boxes
    ax.add_line(
        Line2D(
            [outputs_right_x, outer_x, outer_x, outer_x],
            [outputs_mid_y, outputs_mid_y, cae_enter_y, cae_enter_y],
            linewidth=1.2,
            color="#2F2F2F",
            solid_joinstyle="round",
            solid_capstyle="round",
        )
    )
    # Final segment with arrow head pointing left into the CAE box.
    ax.add_patch(
        FancyArrowPatch(
            (outer_x, cae_enter_y),
            (cae_right_x, cae_enter_y),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.2,
            color="#2F2F2F",
        )
    )

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)

    return out_svg, out_png


def main() -> int:
    svg, png = generate(Path("docs") / "figures")
    print(f"OK: wrote {svg}")
    print(f"OK: wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
