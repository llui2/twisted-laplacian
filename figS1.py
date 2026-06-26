import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from scipy.optimize import root


alpha_pi = np.pi / 3.0

edges = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 0),
]


def locked_equations(xy, alpha):
    x, y = xy

    f1 = (
        np.sin(y + alpha)
        - 2.0 * np.sin(x) * np.cos(alpha)
        - np.sin(alpha)
    )

    f2 = (
        2.0 * np.sin(x - alpha)
        + 2.0 * np.sin(y - alpha)
        + np.sin(x + alpha)
        + np.sin(alpha)
    )

    return np.array([f1, f2])


def M_perp(x, y, alpha):
    return np.array(
        [
            [
                -np.cos(y + alpha) - np.cos(x - alpha),
                np.cos(x - alpha),
            ],
            [
                np.cos(x + alpha),
                -np.cos(x + alpha) - 2.0 * np.cos(alpha),
            ],
        ]
    )


def lambda_eta(x, alpha):
    return -(np.cos(x + alpha) + 2.0 * np.cos(alpha))


def branch_phases(x, y):
    z = x + y
    return np.array([0.0, x, x, y, z, z, y])


def full_jacobian(phases, alpha):
    J = np.zeros((7, 7))

    for i, j in edges:
        d_ij = np.cos(phases[j] - phases[i] - alpha)
        d_ji = np.cos(phases[i] - phases[j] - alpha)

        J[i, j] = d_ij
        J[j, i] = d_ji
        J[i, i] -= d_ij
        J[j, j] -= d_ji

    return J


def sorted_eigenvalues(matrix):
    values = np.linalg.eigvals(matrix)
    values = np.real_if_close(values, tol=1000)
    order = np.argsort(np.real(values))[::-1]
    return values[order]


def threshold_equations(variables):
    x, y, alpha = variables
    f1, f2 = locked_equations((x, y), alpha)
    determinant = np.linalg.det(M_perp(x, y, alpha))
    return np.array([f1, f2, determinant])


# Stability threshold
solution = root(threshold_equations, np.array([0.06, 0.97, 1.02]))
if not solution.success:
    raise RuntimeError("Threshold solver failed: " + solution.message)

x_c, y_c, alpha_c = solution.x
z_c = x_c + y_c

M_perp_c = M_perp(x_c, y_c, alpha_c)
lambda_eta_c = lambda_eta(x_c, alpha_c)

phases_c = branch_phases(x_c, y_c)
J_c = full_jacobian(phases_c, alpha_c)

eigenvalues_M_perp = sorted_eigenvalues(M_perp_c)
eigenvalues_J = sorted_eigenvalues(J_c)


# Locked branch
alphas = np.linspace(0.0, alpha_pi + 0.10, 600)

x_values = []
y_values = []
z_values = []
largest_eigenvalue = []

guess = np.array([0.0, 0.0])

for alpha in alphas:
    solution = root(lambda xy: locked_equations(xy, alpha), guess)
    if not solution.success:
        raise RuntimeError(
            f"Branch continuation failed at alpha={alpha:.8f}: "
            + solution.message
        )

    x, y = solution.x
    z = x + y
    eigenvalues = np.linalg.eigvals(M_perp(x, y, alpha))

    x_values.append(x)
    y_values.append(y)
    z_values.append(z)
    largest_eigenvalue.append(np.max(eigenvalues.real))
    guess = solution.x


# Numerical values used in the text
np.set_printoptions(precision=10, suppress=True, linewidth=140)

print("Loss of transverse stability")
print("--------------------------------")
print(f"alpha_c    = {alpha_c:.10f}")
print(f"alpha_pi   = {alpha_pi:.10f}")
print(f"x_c        = {x_c:.10f}")
print(f"y_c        = {y_c:.10f}")
print(f"z_c        = {z_c:.10f}")
print(f"lambda_eta = {lambda_eta_c:.10f}")
print("M_perp(alpha_c) =")
print(M_perp_c)
print("eigenvalues(M_perp) =", eigenvalues_M_perp)
print("Full Jacobian J(alpha_c), rows/columns ordered as phi_1,...,phi_7 =")
print(J_c)
print("eigenvalues(J) =", eigenvalues_J)


# Figure
plt.rc('font', family='Helvetica', size=10)
plt.rc('mathtext', fontset='dejavusans')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.0))

colors = ["#648FFF", "#DC267F", "#FFB000"]

ax1.plot(alphas, x_values, color=colors[0], linewidth=1.8, label=r"$x$")
ax1.plot(alphas, y_values, color=colors[1], linewidth=1.8, label=r"$y$")
ax1.plot(alphas, z_values, color=colors[2], linewidth=1.8, label=r"$z=y+x$")
ax1.scatter(
    [alpha_pi, alpha_pi, alpha_pi],
    [0.0, alpha_pi, alpha_pi],
    color=[colors[0], colors[1], colors[2]],
    s=18,
    zorder=4,
)
ax1.axvline(alpha_c, color="0.45", linestyle="--", linewidth=1.1, label=r"$\alpha_c$")
ax1.axvline(alpha_pi, color="0.45", linestyle=":", linewidth=1.4, label=r"$\alpha_{\pi}$")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(0.0, alphas[-1])
ax1.set_xticks([0.0, np.pi / 9.0, 2.0 * np.pi / 9.0, alpha_pi])
ax1.set_xticklabels(
    [r"$0$", r"$\frac{\pi}{9}$", r"$\frac{2\pi}{9}$", r"$\frac{\pi}{3}$"]
)
ax1.set_xlabel(r"$\alpha$", fontsize=13)
ax1.set_ylabel("Phase offsets", fontsize=11)
ax1.tick_params(axis="x", labelsize=9)
ax1.tick_params(axis="y", labelsize=9)
ax1.legend(loc="upper left", frameon=False, fontsize=9)
ax1.text(
    -0.14,
    1.06,
    "a",
    transform=ax1.transAxes,
    fontsize=12,
    fontweight="bold",
    fontname="DejaVu Sans",
)

ax2.plot(
    alphas,
    largest_eigenvalue,
    color="red",
    linewidth=1.8,
    label=r"$\max \operatorname{Re}\sigma(M_{\perp})$",
)
ax2.axhline(0.0, color="0.2", linewidth=1.0)
ax2.scatter([alpha_c], [0.0], color="red", s=18, zorder=4)
ax2.axvline(alpha_c, color="0.45", linestyle="--", linewidth=1.1, label=r"$\alpha_c$")
ax2.axvline(alpha_pi, color="0.45", linestyle=":", linewidth=1.4, label=r"$\alpha_{\pi}$")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlim(0.0, alphas[-1])
ax2.set_xticks([0.0, np.pi / 9.0, 2.0 * np.pi / 9.0, alpha_pi])
ax2.set_xticklabels(
    [r"$0$", r"$\frac{\pi}{9}$", r"$\frac{2\pi}{9}$", r"$\frac{\pi}{3}$"]
)
ax2.set_xlabel(r"$\alpha$", fontsize=13)
ax2.set_ylabel("Transverse growth rate", fontsize=11)
ax2.tick_params(axis="x", labelsize=9)
ax2.tick_params(axis="y", labelsize=9)
ax2.legend(loc="upper left", frameon=False, fontsize=9)
ax2.text(
    -0.14,
    1.06,
    "b",
    transform=ax2.transAxes,
    fontsize=12,
    fontweight="bold",
    fontname="DejaVu Sans",
)

plt.tight_layout(w_pad=2.0)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
full_bbox = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
split_x = 0.5 * (full_bbox.x0 + full_bbox.x1)

left_bbox = Bbox.from_extents(full_bbox.x0, full_bbox.y0, split_x, full_bbox.y1)
right_bbox = Bbox.from_extents(split_x, full_bbox.y0, full_bbox.x1, full_bbox.y1)

fig.savefig("figS1a.pdf", dpi=300, bbox_inches=left_bbox, pad_inches=0.0)
fig.savefig("figS1b.pdf", dpi=300, bbox_inches=right_bbox, pad_inches=0.0)
