import numpy as np
import networkx as nx


def draw_oriented_edges(
    G,
    pos,
    ax,
    width=2,
    edge_color="black",
    arrowsize=14,
    arrow_fraction=0.12,
):
    edges = list(G.edges())
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=edges,
        ax=ax,
        width=width,
        edge_color=edge_color,
        arrows=False,
    )

    arrow_edges = []
    arrow_pos = {}
    for index, (source, target) in enumerate(edges):
        midpoint = 0.5 * (np.asarray(pos[source]) + np.asarray(pos[target]))
        offset = arrow_fraction * (np.asarray(pos[target]) - np.asarray(pos[source]))
        tail = ("arrow_tail", index)
        head = ("arrow_head", index)
        arrow_pos[tail] = midpoint - offset
        arrow_pos[head] = midpoint + offset
        arrow_edges.append((tail, head))

    nx.draw_networkx_edges(
        nx.DiGraph(arrow_edges),
        pos=arrow_pos,
        edgelist=arrow_edges,
        ax=ax,
        width=width,
        edge_color=edge_color,
        arrows=True,
        arrowstyle="->",
        arrowsize=arrowsize,
        node_size=0,
    )
