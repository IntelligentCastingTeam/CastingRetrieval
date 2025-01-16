# %%
import numpy as np
import skfem
import skfem as fem
from skfem.helpers import dot, grad


# %%
@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))


@fem.LinearForm
def l(v, w):
    x, y = w.x
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v


# %%
mesh = fem.MeshTri().refined(3)
mesh
# %%
Vh = fem.Basis(mesh, fem.ElementTriP1())
Vh
# %%
A = a.assemble(Vh)
b = l.assemble(Vh)
print(A.shape)
print(b.shape)
# %%
D = Vh.get_dofs()
D
# %%
x = fem.solve(*fem.condense(A, b, D=D))
print(x.shape)


# %%
@fem.Functional
def error(w):
    x, y = w.x
    uh = w["uh"]
    u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2.0 * np.pi**2)
    return (uh - u) ** 2


print(round(error.assemble(Vh, uh=Vh.interpolate(x)), 9))
# %%

from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models.poisson import laplace, mass

halfwidth = np.array([1.0, 1.0])
ncells = 2**3
diffusivity = 5.0

mesh = MeshQuad.init_tensor(
    np.linspace(-1, 1, 2 * ncells) * halfwidth[0],
    np.linspace(-1, 1, 2 * ncells) * halfwidth[1],
)

elements = ElementQuad2()
basis = Basis(mesh, elements)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = 0.01
theta = 0.5
L0, M0 = penalize(L, M, D=basis.get_dofs())
A = M0 + theta * L0 * dt
b = M0 - (1 - theta) * L0 * dt

backsolve = splu(A.T).solve
u_init = np.cos(np.pi * basis.doflocs / 2 / halfwidth[:, None]).prod(0)


def exact(t: float) -> np.ndarray:
    return np.exp(-diffusivity * np.pi**2 * t / 4 * sum(halfwidth**-2)) * u_init


def evolve(t: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    while np.linalg.norm(u, np.inf) > 2**-3:
        t, u = t + dt, backsolve(B @ u)
        yield t, u


probe = basis.probes(np.zeros((mesh.dim(), 1)))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skfem.visuals.matplotlib import plot

ax = plot(mesh, u_init[basis.nodal_dofs.flatten()], shading="gouraud")
title = ax.set_title("t = 0.00")
field = ax.get_children()[0]  # vertex-based temperature-colour
fig = ax.get_figure()
fig.colorbar(field)


def update(event):
    t, u = event

    u0 = {"skfem": (probe @ u)[0], "exact": (probe @ exact(t))[0]}
    print(
        "{:4.2f}, {:5.3f}, {:+7.4f}".format(t, u0["skfem"], u0["skfem"] - u0["exact"])
    )

    title.set_text(f"$t$ = {t:.2f}")
    field.set_array(u[basis.nodal_dofs.flatten()])


animation = FuncAnimation(
    fig,
    update,
    evolve(0.0, u_init),
    repeat=False,
    interval=50,
)

plt.show()

# %%
print(L)
# %%
print(M)

# %%
temp = asm(laplace, basis)
print(temp.shape)
# %%
x, y = np.meshgrid(range(temp.shape[0]), range(temp.shape[1]))
z = temp.toarray()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z)
plt.show()
# %%
model = np.load("../data/voxelModel/tutai01.npy")
print(model.shape)

# %%
pos = np.argwhere(model > 0)
pos = pos.T
# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pos[0, :], pos[1, :], pos[2, :], marker="o")
plt.show()
# %%
print(pos)
# %%
import skfem

hex_mesh = skfem.MeshHex()
hex_mesh
# %%

import matplotlib.pyplot as plt
import numpy as np

L = 1
T = 1
Nx = 100
Nt = 1000
alpha = 0.01

x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)
dx = x[1] - x[0]
dt = t[1] - t[0]

u = np.zeros([Nt + 1, Nx + 1])
u[0, :] = np.sin(np.pi * x)

u[:, 0] = 0
u[:, Nx] = 0

for n in range(Nt):
    for i in range(1, Nx):
        u[n + 1, i] = u[n, i] + (
            alpha * dt / dx**2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
        )


plt.imshow(u, extent=[0, L, 0, T], aspect="auto", cmap="hot")
plt.show()

# %%
import numpy as np

model = np.load("../data/voxelModel/ErZhouBi.npy")
print(np.max(model))
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
points = np.argwhere(model == 1).T
ax.scatter(points[0, :], points[1, :], points[2, :])
plt.show()
# %%
import numpy as np
from scipy.ndimage import convolve

model = np.load("../data/voxelModel/1411731.npy")
print(model.shape)
kernel = np.zeros([3, 3, 3])
kernel[1, 1, 1] = -6
kernel[1, 1, 0] = 1
kernel[1, 0, 1] = 1
kernel[0, 1, 1] = 1
kernel[1, 1, 2] = 1
kernel[2, 1, 1] = 1
kernel[1, 2, 1] = 1

alpha = 0.1

Temp = model.astype(np.float32) * 1000
points = np.argwhere(model > 0)
values = []

for i in range(32):
    Temp = Temp + alpha * convolve(Temp, weights=kernel)
    value = [Temp[x, y, z] for x, y, z in points]
    values.append(value)
    Temp = Temp * model
print(values)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# points = np.argwhere(Temp > 0).T
# ax.scatter(points[0, :], points[1, :], points[2, :], c=values, cmap="viridis")
# print(np.max(values))

# plt.show()

# %%
values = np.array(values)
plt.imshow(np.array(values), extent=[0, values.shape[1], 0, values.shape[0]])
plt.savefig("../data/temp.png")
# %%
print(values.shape)
# %%
values.T.shape
# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, k_means, kmeans_plusplus
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 4000
n_components = 4

X, y_true = make_blobs(
    n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
)
X = X[:, ::-1]

# Calculate seeds from k-means++
centers_init, indices = kmeans_plusplus(X, n_clusters=4, random_state=0)

# Plot init seeds along side sample data
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

kmeans = KMeans(n_clusters=4, random_state=0, init="k-means++", n_init="auto").fit(X)

plt.figure(1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="b", s=50)
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])
plt.show()

# %%
# %%
values = np.array(values).T
print(np.min(values))
# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=32, random_state=0, init="k-means++", n_init="auto").fit(
    values
)
# %%
print(values.shape)
# %%
result = kmeans.predict(values)
print(np.max(result), np.min(result))
# %%
x, bins = np.histogram(result, bins=np.arange(-1, 32))
print(x.astype(np.float32) / np.sum(x))
print(x.shape)
# %%
data = np.random.rand(5, 5)
cor = [(0, 0), (1, 2)]
ex = [data[x, y] for x, y in cor]
print(ex)
# %%
x = np.bincount(result, minlength=0)
print(x, x.shape)
x = x.astype(np.float32) / np.sum(x)
x = x.flatten()
print(x)
print(x.shape)
# %%
