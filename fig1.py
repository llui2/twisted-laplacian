import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from graph_drawing import draw_oriented_edges


RANDOM_SEED = 10 #np.random.randint(0, 20)
print(f"Random seed for graph generation: {RANDOM_SEED}")


def sheaf_laplacian(G, alpha):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)

    B = np.zeros((m, n), dtype=complex)
    for e, (i, j) in enumerate(edges):
        # orientation: i -> j
        B[e, i] = 1
        B[e, j] = -np.exp(1j * alpha)

    L = B.conj().T @ B
    return L


def order_parameter(v):
    phases = np.angle(v)
    return np.abs(np.mean(np.exp(1j * phases)))


def random_cactus_graph(seed=RANDOM_SEED, n_blocks=5):
    """
    Build a connected cactus by attaching cycles or bridge paths to one
    existing anchor vertex. Every new cycle uses fresh vertices except for
    its anchor, so no edge can belong to two different cycles.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0)

    next_node = 1
    cycle_blocks = 0

    for block_index in range(n_blocks):
        anchor = int(rng.choice(list(G.nodes())))
        must_add_cycle = cycle_blocks < 3 and block_index >= n_blocks - 3
        add_cycle = must_add_cycle or rng.random() < 0.72

        if add_cycle:
            cycle_length = int(rng.integers(3, 7))
            new_nodes = list(range(next_node, next_node + cycle_length - 1))
            next_node += cycle_length - 1

            cycle_nodes = [anchor] + new_nodes
            cycle_edges = list(zip(cycle_nodes, cycle_nodes[1:]))
            cycle_edges.append((cycle_nodes[-1], anchor))
            G.add_edges_from(cycle_edges)
            cycle_blocks += 1
        else:
            path_length = int(rng.integers(1, 4))
            previous = anchor
            for _ in range(path_length):
                G.add_edge(previous, next_node)
                previous = next_node
                next_node += 1

    return nx.convert_node_labels_to_integers(G)


def is_cactus_graph(G):
    cycles = nx.cycle_basis(G)
    cycle_edges = []

    for cycle in cycles:
        edges = {
            tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
            for i in range(len(cycle))
        }
        cycle_edges.append(edges)

    for i, edges_i in enumerate(cycle_edges):
        for edges_j in cycle_edges[i + 1:]:
            if edges_i & edges_j:
                return False

    return nx.is_connected(G)


def build_node_colors(G, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    n_nodes = G.number_of_nodes()
    samples = np.linspace(0.08, 0.88, n_nodes)
    node_colors = {0: "#000000"}

    for node, sample in zip(sorted(G.nodes())[1:], samples[1:]):
        node_colors[node] = to_hex(cmap(sample))

    return node_colors


def graph_layout(G):
    pos = nx.kamada_kawai_layout(G)
    return {node: np.asarray(coords) for node, coords in pos.items()}


# Graph definition
G = random_cactus_graph()
if not is_cactus_graph(G):
    raise RuntimeError("The generated graph is not a cactus graph.")

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

eigenvalues_dense = np.array(eigenvalues_dense)

plt.rc("font", family="Helvetica", size=12)
plt.rc("mathtext", fontset="dejavusans")

# === Figure 1: Order parameter and eigenvalues ===
fig, ax = plt.subplots(figsize=(5, 3.4))

# Dense eigenvalues (background)
eigen_line, = ax.plot(
    alphas_dense,
    eigenvalues_dense,
    linestyle="-",
    alpha=1,
    color="#001aff",
    label=r"$\lambda_1$",
)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"$\alpha$", fontsize=15)
ax.set_ylabel(r"$\lambda_1$", fontsize=15)
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax.set_xticklabels([
    r"$0$",
    r"$\frac{\pi}{4}$",
    r"$\frac{\pi}{2}$",
    r"$\frac{3\pi}{4}$",
    r"$\pi$",
])

lambda_ylim = max(0.05, 1.08 * float(np.max(eigenvalues_dense)))
ax.set_ylim(0, lambda_ylim)
ax.set_yticks([0, lambda_ylim / 2, lambda_ylim])
ax.set_yticklabels([r"$0$", f"{lambda_ylim / 2:.2f}", f"{lambda_ylim:.2f}"])

# === Inset: graph structure ===
ax_inset = fig.add_axes([0.2, 0.6, 0.99, 0.99])  # [left, bottom, width, height] in figure coords
pos = graph_layout(G)
node_labels = {i: str(i + 1) for i in G.nodes()}

draw_oriented_edges(G, pos=pos, ax=ax_inset, width=2, edge_color="black")
nx.draw_networkx_nodes(
    G,
    pos=pos,
    ax=ax_inset,
    node_color="#FFFFFF",
    node_size=300,
    linewidths=2,
    edgecolors="black",
)
nx.draw_networkx_labels(G, pos=pos, labels=node_labels, ax=ax_inset,
                        font_size=9, font_color="black", font_weight="bold")

ax_inset.axis("off")
ax_inset.set_aspect("equal")
ax_inset.margins(0.25)
for artist in ax_inset.get_children():
    artist.set_clip_on(False)

plt.tight_layout()
plt.savefig("fig1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
