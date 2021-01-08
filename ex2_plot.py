from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# x1 = np.arange(-1, 0, 0.01)
# x2 = np.arange(-1, 1, 0.01)
# x3 = np.arange(0, 1, 0.01)
#
# xs1, ys1 = np.meshgrid(x1, x2)
# xs2, ys2 = np.meshgrid(x3, x1)
#
# z1 = np.sqrt(xs1 ** 2 + ys1 ** 2)
# z2 = np.sqrt(xs2 ** 2 + ys2 ** 2)
# z3 = np.sqrt(xs2 ** 2 + ys2 ** 2) * np.nan
# z = np.concatenate((z1, np.concatenate((z2, z3), 0)), 1)
# ax = plt.subplot(1, 1, 1)
# h = plt.imshow(z, interpolation='nearest', cmap='rainbow',
#                extent=[-1, 1, -1, 1],
#                origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(h, cax=cax)
# plt.imshow(z, interpolation='nearest', cmap='rainbow',
#                extent=[0, 1, -1, 0],
#                origin='lower', aspect='auto')

err_drm = np.load('ex2_drm_err.npy')
err_dgm = np.load('ex2_dgm_err.npy')
fig = plt.figure()
h2 = plt.imshow(err_dgm, interpolation='nearest', cmap='rainbow',
                extent=[-1, 1, -1, 1],
                origin='lower', aspect='auto')
plt.colorbar(h2)
plt.title('DGM')
plt.figure()
h1 = plt.imshow(err_drm, interpolation='nearest', cmap='rainbow',
                extent=[-1, 1, -1, 1],
                origin='lower', aspect='auto')

plt.colorbar(h1)
plt.title('DRM')
plt.show()
