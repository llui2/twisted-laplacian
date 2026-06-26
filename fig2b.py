import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_drawing import draw_oriented_edges

def sheaf_laplacian(G, alpha):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)

    B = np.zeros((m, n), dtype=complex)
    for e, (i, j) in enumerate(edges):
        B[e, i] = 1
        B[e, j] = -np.exp(1j * alpha)

    L = B.conj().T @ B
    return L

def order_parameter(v):
    phases = np.angle(v)
    return np.abs(np.mean(np.exp(1j * phases)))

# Graph definition
n_nodes = 3
G = nx.cycle_graph(n_nodes)
n_nodes = G.number_of_nodes()

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

# === Single figure ===
fig, ax = plt.subplots(figsize=(5, 3.4))

# Dense eigenvalues (background)
eigen_line, = ax.plot(alphas_dense, eigenvalues_dense, linestyle='-', alpha=1,
                      color="#001aff", label=r'$\lambda_1$')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("$\\alpha$", fontsize=15)
ax.set_ylabel("$\\lambda_1$", fontsize=15)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",
                    r"$\frac{3\pi}{4}$", r"$\pi$"])
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.3, 0.7, 1.0])
ax.set_yticklabels([r"$0$", r"$0.3$", r"$0.7$", r"$1$"])
ax.text(
    -0.14,
    1.06,
    "b",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    fontname="DejaVu Sans",
)

# === Inset: graph structure ===
ax_inset = fig.add_axes([0.15, 0.42, 0.45, 0.45])  # [left, bottom, width, height] in figure coords

pos = nx.circular_layout(G)
theta = 1.57
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
rotated_pos = {node: rotation_matrix @ pos[node] for node in G.nodes()}

node_labels = {i: str(i + 1) for i in G.nodes()}

draw_oriented_edges(G, pos=rotated_pos, ax=ax_inset, width=2, edge_color='black')
nx.draw_networkx_nodes(G, pos=rotated_pos, ax=ax_inset, node_color="#FFFFFF",
                       node_size=300, linewidths=2, edgecolors='black')
nx.draw_networkx_labels(G, pos=rotated_pos, labels=node_labels, ax=ax_inset,
                        font_size=9, font_color='black', font_weight='bold')
ax_inset.axis('off')
ax_inset.set_aspect('equal')
ax_inset.margins(0.25)  # prevent node cutoff at borders
for artist in ax_inset.get_children():
    artist.set_clip_on(False)

plt.tight_layout()
plt.savefig("fig2b.pdf", dpi=300, bbox_inches='tight')
