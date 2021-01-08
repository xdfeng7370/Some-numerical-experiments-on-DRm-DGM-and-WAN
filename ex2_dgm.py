import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from math import pi
import numpy as np
import sobol_seq as sobol

"""
-\Delta u = 0 in \Omega
u = r^{1/2}sin(theta/2) on Gamma
\Omega = (-1, 1)^2 \ (0,1)^2
exact solution: u = r^{1/2}sin(theta/2)
"""
class Block(nn.Module):
    """
    Implementation of the block used in the Deep Ritz
    Paper
    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network
    Implements a network with the architecture used in the
    deep ritz method paper
    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, depth=4):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def get_interior_points(N=128):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    x1 = sobol.i4_sobol_generate(2, N) - np.array([1, 1])
    x2 = sobol.i4_sobol_generate(2, N) - np.array([1, 0])
    x3 = sobol.i4_sobol_generate(2, N) - np.array([0, 1])
    return torch.from_numpy(np.concatenate((x1, x2, x3), 0)).float()


def get_boundary_points(N=33):
    index = sobol.i4_sobol_generate(1, N)
    index1 = sobol.i4_sobol_generate(1, N) * 2 - 1
    xb1 = np.concatenate((index, np.zeros_like(index)), 1)
    xb2 = np.concatenate((index1, - np.ones_like(index1)), 1)
    xb3 = np.concatenate((index - 1, np.ones_like(index)), 1)
    xb4 = np.concatenate((np.zeros_like(index), index), 1)
    xb5 = np.concatenate((np.full_like(index1, -1), index1), 1)
    xb6 = np.concatenate((np.ones_like(index), index - 1), 1)
    xb = torch.from_numpy(np.concatenate((xb1, xb2, xb3, xb4, xb5, xb6), 0)).float()

    return xb

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

# def function_u_exact(x):
#     r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1: 2] ** 2)
#     theta = torch.where(x[:, 0: 1]==0, torch.ones_like(x[:, 0: 1]) * pi / 2, torch.atan(x[:, 1: 2] / x[:, 0: 1]))
#     theta = torch.where(((x[:, 0: 1]==0) & (x[:, 1: 2] < 0)), - torch.ones_like(x[:, 0: 1]) * pi / 2, theta)
#     return torch.sqrt(r) * torch.sin(theta / 2)

def function_u_exact(x):
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1: 2] ** 2)
    sin_theta_2 = torch.where(r==0, torch.zeros_like(r), torch.sqrt((1-x[:, 0:1] / r) / 2))
    return torch.sqrt(r) * sin_theta_2

def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]


def main():

    epochs = 1000

    in_N = 2
    m = 40
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    print(model)

    best_loss, best_epoch = 1000, 0
    tt = time.time()
    xr = get_interior_points(1000)
    xb = get_boundary_points(N=200)
    xr = xr.to(device)
    xb = xb.to(device)
    for epoch in range(epochs+1):

        # generate the data set
        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        exact_b = function_u_exact(xb)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads2 = gradients(xr, grads[:, 0: 1])[:, 0: 1] + gradients(xr, grads[:, 1: 2])[:, 1: 2]
        loss_r = torch.mean(torch.square(grads2))
        loss_r = torch.mean(loss_r)
        loss_b = 50 * torch.mean(torch.abs(output_b - exact_b))
        # loss_b = 50 * torch.mean(torch.square(output_b - exact_b))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        if epoch % 100 == 0:

            print(time.time() - tt, 'epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(), 'loss_b:', loss_b.item())
            tt = time.time()

    with torch.no_grad():
        N0 = 1000
        x1 = np.linspace(-1, 0, N0 + 1)
        x2 = np.linspace(-1, 1, 2 * N0 + 1)
        x3 = np.linspace(0, 1, N0 + 1)

        xs1, ys1 = np.meshgrid(x1, x2)
        xs2, ys2 = np.meshgrid(x3, x1)
        Z1 = torch.from_numpy(np.concatenate((xs1.flatten()[:, None], ys1.flatten()[:, None]), 1)).float()
        Z2 = torch.from_numpy(np.concatenate((xs2.flatten()[:, None], ys2.flatten()[:, None]), 1)).float()
        pred1 = torch.reshape(model(Z1), [2 * N0 + 1, N0 + 1])
        pred2 = torch.reshape(model(Z2), [N0 + 1, N0 + 1])
        pred1 = pred1.cpu().numpy()
        pred2 = pred2.cpu().numpy()
        pred3 = pred2 * np.nan
        pred = np.concatenate((pred1, np.concatenate((pred2, pred3[1:, :]), 0)), 1)
        exact1 = torch.reshape(function_u_exact(Z1), [2 * N0 + 1, N0 + 1]).cpu().numpy()
        exact2 = torch.reshape(function_u_exact(Z2), [N0 + 1, N0 + 1]).cpu().numpy()
        exact = np.concatenate((exact1, np.concatenate((exact2, exact2[1:, :] * np.nan), 0)), 1)
        np.save('ex2_dgm_err.npy', np.abs(pred-exact))
    err = np.sqrt((np.sum(np.square(exact1 - pred1)) + np.sum(np.square(exact2 - pred2))) / (np.sum(np.square(exact1)) + np.sum(np.square(exact2))))
    print("Error:", err)

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[-1, 1, -1, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)

    plt.figure()
    ax2 = plt.subplot(1, 1, 1)
    h1 = plt.imshow(exact, interpolation='nearest', cmap='rainbow',
                    extent=[-1, 1, -1, 1],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h1, cax=cax)

    plt.figure()
    ax3 = plt.subplot(1, 1, 1)
    h2 = plt.imshow(exact-pred, interpolation='nearest', cmap='rainbow',
                    extent=[-1, 1, -1, 1],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h2, cax=cax)

    plt.show()


if __name__ == '__main__':
    main()
