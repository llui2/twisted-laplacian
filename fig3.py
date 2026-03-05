import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
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

def order_parameter(v):
    phases = np.angle(v)
    return np.abs(np.mean(np.exp(1j * phases)))


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

# Graph definition
n_nodes = 7
G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 6), (6, 0)])

alphas = np.linspace(0, np.pi, 99)[1:-1]
mode_indices = [0]

R_values = {k: [] for k in mode_indices}
eigenvalues = {k: [] for k in mode_indices}

for alpha in alphas:
    L = sheaf_laplacian(G, alpha)
    vals, vecs = np.linalg.eigh(L)

    idxs = np.argsort(vals)
    vals = vals[idxs]
    vecs = vecs[:, idxs]

    for k in mode_indices:
        v = vecs[:, k]
        R_values[k].append(order_parameter(v))
        eigenvalues[k].append(vals[k].real)

# Dense alpha grid for eigenvalues only
alphas_dense = np.linspace(0, np.pi, 10000)[1:-1]
eigenvalues_dense = []

for alpha in alphas_dense:
    L = sheaf_laplacian(G, alpha)
    vals, _ = np.linalg.eigh(L)
    vals_sorted = np.sort(vals.real)
    eigenvalues_dense.append(vals_sorted[mode_indices[0]])

plt.rc('font', family='Helvetica', size=12)
plt.rc('mathtext', fontset='dejavusans')

# === Figure 1: Order parameter and eigenvalues ===
fig, ax = plt.subplots(figsize=(5, 3.4))
axb = ax.twinx()

# Dense eigenvalues (background)
eigen_line, = axb.plot(alphas_dense, eigenvalues_dense, linestyle='-', alpha=1,
                       color="#001aff", label=r'$\lambda_1$')

color = "#ff5900"
order_line, = ax.plot(
    alphas, R_values[mode_indices[0]],
    linestyle='-', linewidth=1.3, alpha=1,
    color=color, marker='o', markersize=5,
    markerfacecolor='none', markeredgecolor=color,
    label=r'$R_1$'
)

ax.set_zorder(axb.get_zorder() + 1)
ax.patch.set_visible(False)

# Remove top and right spines
for spine_ax in (ax, axb):
    spine_ax.spines['top'].set_visible(False)
    spine_ax.spines['right'].set_visible(False)

# Keep right spine of axb for the secondary y-axis tick marks
axb.spines['right'].set_visible(True)
axb.spines['top'].set_visible(False)

ax.set_xlabel("$\\alpha$", fontsize=15)
ax.set_ylabel("$R_1$", fontsize=15, labelpad=-5)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",
                    r"$\frac{3\pi}{4}$", r"$\pi$"])
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.3, 0.7, 1.0])
ax.set_yticklabels([r"$0$", r"$0.3$", r"$0.7$", r"$1$"])

axb.set_ylabel("$\\lambda_1$", fontsize=15)
axb.set_ylim(0, 0.4)
axb.set_yticks(np.arange(0, 0.6, 0.2))
axb.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$"])

ax.text(-0.15, 1.1, "b", fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='black', fontweight='bold', 
         fontname='DejaVu Sans')

# Legend
lines = [order_line, eigen_line]
labels_leg = [l.get_label() for l in lines]
ax.legend(lines, labels_leg, loc='center', bbox_to_anchor=(0.7, 0.9), frameon=False)

plt.tight_layout()
plt.savefig("fig3.pdf", dpi=300, bbox_inches='tight')


# === Figure 2: Graph structure ===
fig2, ax2 = plt.subplots(figsize=(3, 3))

pos = {
    0: np.array([0, 0]),
    1: np.array([-0.9, 0.8]),
    2: np.array([0.9, 0.8]),
}

pentagon_coords = [
    (-0.9, -0.8),
    (-0.5, -1.8),
    (0.5, -1.8),
    (0.9, -0.8),
    (0, -0.5)
]

for i, coord in enumerate(pentagon_coords, start=3):
    pos[i] = np.array(coord)

node_labels = {i: str(i + 1) for i in G.nodes()}
node_colors = build_node_colors(NODE_COLORMAP)
graph_node_colors = [node_colors[node + 1] for node in G.nodes()]

nx.draw_networkx_edges(G, pos=pos, ax=ax2, width=2.2, edge_color='black')
nx.draw_networkx_nodes(
    G,
    pos=pos,
    ax=ax2,
    node_color=graph_node_colors,
    node_size=400,
    linewidths=2,
    edgecolors='black',
)
for node, label in node_labels.items():
    label_color = "white" if node in {0, 1, 2, 3, 4, 5, 6, 7} else "black"
    x, y = pos[node]
    ax2.text(
        x,
        y,
        label,
        fontsize=10,
        color=label_color,
        ha='center',
        va='center',
        fontweight='black',
        fontname='DejaVu Sans',
    )


ax2.text(0, 0.9, "a", fontsize=12, ha='center', va='center',
         transform=ax2.transAxes, color='black', fontweight='bold', 
         fontname='DejaVu Sans')

ax2.axis('off')
ax2.set_aspect('equal')
ax2.margins(0.25)
for artist in ax2.get_children():
    artist.set_clip_on(False)

plt.tight_layout()
plt.savefig("fig3_graph.pdf", dpi=300, bbox_inches='tight')
