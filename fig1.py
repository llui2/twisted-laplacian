import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

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
axb = ax.twinx()

# Dense eigenvalues (background)
eigen_line, = axb.plot(alphas_dense, eigenvalues_dense, linestyle='-', alpha=1,
                       color="#001aff", label=r'$\lambda_1$')

# Critical vertical line
axb.axvline(alpha_crit, color='gray', linestyle='-', linewidth=1, alpha=0.5,
            label=f'Critical $\\alpha$ = {alpha_crit:.2f}')

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

# Remove top and right spines on both axes
for spine_ax in (ax, axb):
    spine_ax.spines['top'].set_visible(False)
    spine_ax.spines['right'].set_visible(False)

# Keep right spine of axb for the secondary y-axis tick marks
axb.spines['right'].set_visible(True)
axb.spines['top'].set_visible(False)

ax.set_xlabel("$\\alpha$", fontsize=15)
ax.set_ylabel("$R_1$", fontsize=15, labelpad=-5)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi/4, alpha_crit, np.pi/3, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\alpha_{c}$",
                    r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$",
                    r"$\frac{3\pi}{4}$", r"$\pi$"])
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.3, 0.7, 1.0])
ax.set_yticklabels([r"$0$", r"$0.3$", r"$0.7$", r"$1$"])

axb.set_ylabel("$\\lambda_1$", fontsize=15)
axb.set_ylim(0, 0.4)
axb.set_yticks([0, 0.2, 0.4])
axb.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$"])

# Legend
lines = [order_line, eigen_line]
labels_leg = [l.get_label() for l in lines]
ax.legend(lines, labels_leg, loc='center', bbox_to_anchor=(0.7, 0.9), frameon=False)

# === Inset: graph structure ===
# Place the inset in the lower-right region of the plot (axes-fraction coordinates)
ax_inset = fig.add_axes([0.38, 0.38, 0.45, 0.45])  # [left, bottom, width, height] in figure coords

pos = nx.circular_layout(G)
theta = 1.57
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
rotated_pos = {node: rotation_matrix @ pos[node] for node in G.nodes()}

node_labels = {i: str(i + 1) for i in G.nodes()}

nx.draw_networkx_edges(G, pos=rotated_pos, ax=ax_inset, width=2, edge_color='black')
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
plt.savefig("fig1.pdf", dpi=300, bbox_inches='tight')
