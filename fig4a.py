import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

def sheaf_laplacian(G, alpha):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    
    B = np.zeros((m, n), dtype=complex)
    for e, (i, j) in enumerate(edges):
        # orientation: i -> j
        B[e, i] = 1
        B[e, j] = - np.exp(1j * alpha)
    
    L = B.conj().T @ B
    return L

# Color settings (change NODE_COLORMAP or set env var NODE_COLORMAP=plasma, viridis, etc.)
NODE_COLORMAP = os.getenv("NODE_COLORMAP", "plasma")
NODE_COLOR_SAMPLES = {
    2: 0.22,
    3: 0.26,
    4: 0.50,
    7: 0.54,
    5: 0.78,
    6: 0.82,
}


def build_node_colors(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    node_colors = {1: "#000000"}
    for node, sample in NODE_COLOR_SAMPLES.items():
        node_colors[node] = to_hex(cmap(sample))
    return node_colors

# Graph and parameters
n_nodes = 7

G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 6), (6, 0)])

alpha = np.pi/3 - 0.2
mode = 0

# Compute eigenvectors
L = sheaf_laplacian(G, alpha)
vals, vecs = np.linalg.eigh(L)
idx = np.argsort(vals)
vec = vecs[:, idx[mode]]

# Normalize phases
avg_angle = (np.angle(vec[3]) + np.angle(vec[6])) / 2
phase_fix = np.exp(-1j * avg_angle)
vec = vec * phase_fix
phases = vec / np.abs(vec)

# Convert phases to angles and radii
angles = np.angle(phases)
radii = np.ones_like(angles)

# Plot polar
fig, ax = plt.subplots(figsize=(2.8, 2.5), subplot_kw=dict(polar=True))

# Explicit node colors (keys are node labels 1..7)
node_colors = build_node_colors(NODE_COLORMAP)
colors = [node_colors[i + 1] for i in range(len(phases))]

ax.grid(True, color='gray', linestyle='-', linewidth=1, alpha=0.2)
ax.set_yticklabels([])

ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)

def format_pi_label(x):
    frac = x / np.pi
    if frac == 0:
        return "$0$"
    elif frac == 1:
        return r"$\pi$"
    elif frac == 2:
        return r"$2\pi$"
    elif frac.is_integer():
        return fr"${int(frac)}\pi$"
    else:
        # Format fraction like 1/2, 3/2, etc.
        from fractions import Fraction
        f = Fraction(frac).limit_denominator(8)
        if f.numerator == 1:
            return fr"$\frac{{\pi}}{{{f.denominator}}}$"
        else:
            return fr"$\frac{{{f.numerator}\pi}}{{{f.denominator}}}$"

tick_labels = [format_pi_label(t) for t in ticks]

ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)


# Plot eigenvector phases as arrows from origin
for i, angle in enumerate(angles):
    ax.annotate("",
                xy=(angle, 1),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5),
                zorder=3)
    

# Create legend with thin lines, no patches
legend_order = [1, 2, 3, 4, 7, 5, 6]
legend_lines = [Line2D([0], [0], color=node_colors[node], lw=2) for node in legend_order]
ax.legend(legend_lines, [f"Node {node}" for node in legend_order],
          bbox_to_anchor=(1.5, 1.2), frameon=False, fontsize=6)

ax.text(-0.15, 1.1, "a", fontsize=12, ha='center', va='center',
         transform=ax.transAxes, color='black', fontweight='bold', 
         fontname='DejaVu Sans')
ax.text(1.15, -0.15, "$\\alpha < \\alpha_c$", fontsize=12, ha='center', va='center',
         transform=ax.transAxes, color='black', fontweight='normal', 
         fontname='DejaVu Sans')

plt.tight_layout()
plt.savefig("fig4a.pdf", dpi=300)
