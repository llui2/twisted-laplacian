import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from matplotlib.colors import to_hex
from graph_drawing import draw_oriented_edges

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

# Dense eigenvalues (background)
eigen_line, = ax.plot(alphas_dense, eigenvalues_dense, linestyle='-', alpha=1,
                      color="#001aff", label=r'$\lambda_1$')

# Critical vertical line
ax.axvline(np.pi/3, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("$\\alpha$", fontsize=15)
ax.set_ylabel("$\\lambda_1$", fontsize=15)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi/4, np.pi/3, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{3}$",
                    r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

ax.set_ylim(0, 0.4)
ax.set_yticks(np.arange(0, 0.6, 0.2))
ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$"])

# === Inset: graph structure ===
ax_inset = fig.add_axes([0.35, 0.45, 0.6, 0.6])  # [left, bottom, width, height] in figure coords

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
draw_oriented_edges(G, pos=pos, ax=ax_inset, width=2, edge_color='black')
nx.draw_networkx_nodes(
    G,
    pos=pos,
    ax=ax_inset,
    node_color="#FFFFFF",
    node_size=300,
    linewidths=2,
    edgecolors='black',
)
nx.draw_networkx_labels(G, pos=pos, labels=node_labels, ax=ax_inset,
                        font_size=9, font_color='black', font_weight='bold')

ax_inset.axis('off')
ax_inset.set_aspect('equal')
ax_inset.margins(0.25)
for artist in ax_inset.get_children():
    artist.set_clip_on(False)

plt.tight_layout()
plt.savefig("fig3.pdf", dpi=300, bbox_inches='tight')
