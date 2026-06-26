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
n_nodes = 5
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
alphas_dense = np.linspace(0, np.pi, 100000)[1:-1]
eigenvalues_dense = []

for alpha in alphas_dense:
    L = sheaf_laplacian(G, alpha)
    vals, _ = np.linalg.eigh(L)
    vals_sorted = np.sort(vals.real)
    eigenvalues_dense.append(vals_sorted[mode_indices[0]])

eigen_arr = np.array(eigenvalues_dense)
diff_eigen = np.abs(np.diff(eigen_arr))
idx_crit = np.argmax(diff_eigen)
alpha_crit = alphas_dense[idx_crit + 1]

print(f"Critical alpha from abrupt eigenvalue change: {alpha_crit}")

plt.rc('font', family='Helvetica', size=12)
plt.rc('mathtext', fontset='dejavusans')

# === Single figure ===
fig, ax = plt.subplots(figsize=(5, 3.4))

# Dense eigenvalues (background)
eigen_line, = ax.plot(alphas_dense, eigenvalues_dense, linestyle='-', alpha=1,
                      color="#001aff", label=r'$\lambda_1$')

# Critical vertical line
ax.axvline(alpha_crit, color='gray', linestyle='-', linewidth=1, alpha=0.5,
           label=f'Critical $\\alpha$ = {alpha_crit:.2f}')

# Remove top and right spines on both axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("$\\alpha$", fontsize=15)
ax.set_ylabel("$\\lambda_1$", fontsize=15)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi/4, np.pi/3, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{3}$",
                    r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
ax.set_ylim(0, 0.4)
ax.set_yticks([0, 0.2, 0.4])
ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$"])
ax.text(
    -0.14,
    1.06,
    "a",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    fontname="DejaVu Sans",
)

# === Inset: graph structure ===
# Place the inset in the lower-right region of the plot (axes-fraction coordinates)
ax_inset = fig.add_axes([0.45, 0.38, 0.45, 0.45])  # [left, bottom, width, height] in figure coords

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
plt.savefig("fig2a.pdf", dpi=300, bbox_inches='tight')
