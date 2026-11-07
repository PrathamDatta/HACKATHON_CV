"""Draw docs/architecture.png. Box names are the real function / class names in depthup/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 7.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off")
GREEN, RED, AMBER, TEAL, PURPLE = ("#e9f7ef", "#2f855a"), ("#fdecea", "#c53030"), ("#fff4d6", "#b7791f"), ("#e6fffa", "#2c7a7b"), ("#f3f0ff", "#6b46c1")


def box(x, y, w, h, title, sub="", c=GREEN, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=c[0], ec=c[1], lw=2.4 if bold else 1.2))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.52, sub, ha="center", va="top", fontsize=7.6, color="#333", linespacing=1.35)


def line(pts, label=None, lpos=None, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color="#444", lw=1.4, solid_capstyle="butt") if len(pts) > 2 else None
    ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#444", shrinkA=0, shrinkB=0))
    if len(pts) > 2:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color="#444", lw=0)
    if label:
        lx, ly = lpos or ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2 + 0.14)
        ax.text(lx, ly, label, fontsize=7.4, ha="center", color="#222", bbox=dict(fc="white", ec="none", pad=0.6))


ax.text(7, 7.2, "EdgeGate hybrid depth upsampling: data flow  (module.function)", ha="center", fontsize=12.5, fontweight="bold")
box(0.2, 4.5, 1.9, 1.2, "LR depth", "+ validity mask\ndegrade.degrade\nio.read_depth", AMBER)
box(2.5, 4.5, 2.2, 1.2, "1  LR clean-up", "classical.clean_lr\nnormalized_fill, spikes,\nnoise-adaptive bilateral", GREEN)
box(5.1, 4.5, 2.4, 1.2, "3  Step detector + gate G", "classical.lr_discontinuity\nclassical.gate_map  (input: clean LR, σ̂)\n(steps yes, slopes no)", GREEN, True)
box(7.9, 4.5, 2.4, 1.2, "4  Gated tri-lateral JBU", "classical.jbu   ◆RGB\nsigma_c = f(G), depth term\n= expert E1", GREEN, True)
box(2.5, 2.7, 2.2, 1.1, "2  Bicubic base", "classical.bicubic_up\n= expert E0", GREEN)
box(7.9, 2.2, 2.4, 1.9, "5  HybridNet (U-Net)", "model.HybridNet   ◆RGB\n10 input channels:\nrgb, E0, E1, snap, G,\nedge map, conf, scale\n-> residual δ + blend β", RED, True)
box(11.2, 3.0, 2.6, 2.7, "6  Learned blend + residual", "out = E0 + β·(E1 − E0) + δ\nβ = per-pixel sigmoid in [0, 1]\nβ = 0: same form as deep-only\n(pipeline.run_hybrid)", RED, True)
box(11.2, 6.05, 2.6, 0.9, "High-res depth", "", TEAL)
box(0.2, 0.5, 3.1, 1.3, "RGB guidance (HR)", "io.read_rgb\n◆ = stage that reads it (4, 5, snap)", AMBER)
box(3.7, 0.1, 7.9, 1.3, "Other arms (pipeline.METHODS)", "nearest · bilinear · bicubic · JBU · guided · classical (= E1)\ndl_only / dl_only_L: RGB + bicubic only (0.22 M or 9 M params; large is best)\nauto (pipeline.run_auto): dl_only if pristine input, hybrid if noisy or incomplete", PURPLE)

line([(2.1, 5.1), (2.5, 5.1)]); line([(4.7, 5.1), (5.1, 5.1)]); line([(7.5, 5.1), (7.9, 5.1)], "G", (7.7, 5.28))
line([(10.3, 5.1), (11.2, 5.1)], "E1", (10.75, 5.28))
line([(3.6, 4.5), (3.6, 3.8)])
line([(4.7, 3.25), (7.9, 3.25)], "E0", (6.3, 3.4))
line([(6.3, 4.5), (6.3, 3.8), (7.9, 3.8)], "G, snap, edges", (6.9, 3.95))
line([(9.1, 4.5), (9.1, 4.1)], "E1", (9.35, 4.28))
line([(10.3, 3.4), (11.2, 3.4)], "δ, β", (10.75, 3.54))
line([(3.6, 2.7), (3.6, 1.55), (12.5, 1.55), (12.5, 3.0)], "E0", (11.9, 1.7))
line([(3.6, 5.7), (3.6, 6.5), (9.1, 6.5), (9.1, 5.7)], "clean LR + confidence", (6.4, 6.65))
line([(12.5, 5.7), (12.5, 6.05)])
fig.tight_layout()
fig.savefig("docs/architecture.png", dpi=140)
print("saved docs/architecture.png")
