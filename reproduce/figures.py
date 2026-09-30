"""Render numerical manuscript figures from the frozen, source-linked evidence."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_author_revision_v15 as plot


def main():
    data, out = ROOT / "results/supporting", ROOT / "rerun/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none"})
    for function in (plot.ranking, plot.cases, plot.controls, plot.matched, plot.joint):
        function(data, out)
    grid = pd.read_csv(data / "endpoint_molecular_information_v10.csv").pivot(index="dataset", columns="molecular_axis_v10", values="normalized_absolute_mass").reindex(index=plot.ORDER, columns=plot.AXES).fillna(0) * 100
    loss = pd.read_csv(data / "model_family_deletion_by_endpoint.csv").pivot(index="dataset", columns="ablation_family", values="spearman_loss_when_removed").reindex(index=plot.ORDER, columns=["registry", "motif", "chemical_modification"])
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 7.25))
    fig.subplots_adjust(left=.25, right=.975, top=.94, bottom=.10, hspace=.60)
    for j, (ax, frame) in enumerate(zip(axs, [grid, loss])):
        limit = max(abs(frame.to_numpy()).max(), .01)
        ax.imshow(frame, cmap="YlGnBu" if j == 0 else "RdBu_r", vmin=0 if j == 0 else -limit, vmax=100 if j == 0 else limit, aspect="auto")
        ax.set_yticks(range(9), [s + (" †" if "hepatic" in k else "") for s,k in zip(plot.SHORT, plot.ORDER)])
        labels = (["Sequence", "Sugar", "Backbone", "Modified\nbases", "Wing–gap\nlayout", "Joint\ncontext", "Assay"] if j == 0 else ["Positional\ndescriptors", "Sequence-pattern\ndescriptors", "Chemical-summary\nand joint descriptors"])
        ax.set_xticks(range(len(labels)), labels)
        plot.panel(ax, "AB"[j])
        for row in range(9):
            for col in range(frame.shape[1]):
                v = frame.iloc[row,col]
                ax.text(col, row, f"{v:.1f}" if j == 0 else f"{v:+.3f}", ha="center", va="center", fontsize=8, color="white" if (v > 50 if j == 0 else abs(v) > limit*.6) else "#172b35")
    plot.finish(fig, out, "endpoint_molecular_information_v15")
    import json
    (out / "verification.json").write_text(json.dumps({"figures": plot.AUDIT,
        "scope": "Numerical figures from archived results; no fitting, new evidence review or causal inference."}, indent=2) + "\n")
    print(f"Rendered {len(plot.AUDIT)} numerical figures as editable SVG, PNG and PDF.")


if __name__ == "__main__":
    main()
