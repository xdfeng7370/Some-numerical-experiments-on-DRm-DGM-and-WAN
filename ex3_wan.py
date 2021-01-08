import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from math import pi
import numpy as np
import sobol_seq as sobol

"""
-Delta u = f in \Omega
u = g  on \partial \Omega
\Omega=(0,1)^2.
exact solution: u(x,y) = min{x^2,(1-x)^2}
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
    x1 = sobol.i4_sobol_generate(2, N)

    return torch.from_numpy(x1).float()


def get_boundary_points(N=33):
    index = sobol.i4_sobol_generate(1, N)
    xb1 = np.concatenate((index, np.zeros_like(index)), 1)
    xb2 = np.concatenate((index, np.ones_like(index)), 1)
    xb4 = np.concatenate((np.zeros_like(index), index), 1)
    xb6 = np.concatenate((np.ones_like(index), index), 1)
    xb = torch.from_numpy(np.concatenate((xb1, xb2, xb4, xb6), 0)).float()

    return xb


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def function_u_exact(x):
    value = torch.where(x[:, 0: 1] > 0.5, (1 - x[:, 0: 1]) ** 2, x[:, 0: 1] ** 2)
    return value


def function_l_exact(x):
    return x[:, 0: 1] * x[:, 1: 2] * (1 - x[:, 0: 1]) * (1 - x[:, 1: 2])


def function_f():
    return -2


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

def error_l2(x, y):
    """
    :param x: predicted value
    :param y: exact value
    :return: L^2 error
    """
    return torch.norm(x - y) / torch.norm(y)


def main():

    epochs = 30000

    in_N = 2
    m = 40
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    model.apply(weights_init)
    model2 = drrnn(in_N, m, out_N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-2)
    StepLR1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    StepLR2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1000, gamma=0.5)
    print(model)

    best_loss, best_epoch = 1000, 0
    tt = time.time()
    f_value = function_f()
    xr = get_interior_points(1000)
    xb = get_boundary_points(N=200)
    xr = xr.to(device)
    xb = xb.to(device)
    for epoch in range(epochs+1):

        # # generate the data set
        # xr = get_interior_points()
        # xb = get_boundary_points()
        #
        # xr = xr.to(device)
        # xb = xb.to(device)

        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        output_phi_r = model2(xr) * function_l_exact(xr)
        exact_b = function_u_exact(xb)
        grads_u = gradients(xr, output_r)
        grads_phi = gradients(xr, output_phi_r)
        loss_r = torch.square(torch.mean(torch.sum(grads_u * grads_phi, dim=1) - f_value * output_phi_r)) / torch.mean(torch.square(output_phi_r))
        loss_b = 1000000 * torch.mean(torch.abs(output_b - exact_b))
        loss1 = loss_r + loss_b

        loss2 = - loss_r + torch.square(torch.mean(torch.square(output_phi_r)) - 1)
                # + 10 * torch.square(torch.mean(torch.square(output_phi_r)) - 1)
        if epoch % 3 == 2:
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            StepLR2.step()
        else:
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            StepLR1.step()

        if epoch % 100 == 0:
            err = error_l2(model(xr), function_u_exact(xr))
            print('epoch:', epoch, 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'error', err.item(), 'time', time.time() - tt)
            tt = time.time()

    with torch.no_grad():
        N0 = 1000
        x1 = np.linspace(0, 1, N0 + 1)

        xs1, ys1 = np.meshgrid(x1, x1)
        Z1 = torch.from_numpy(np.concatenate((xs1.flatten()[:, None], ys1.flatten()[:, None]), 1)).float()
        pred1 = torch.reshape(model(Z1), [N0 + 1, N0 + 1])
        pred = pred1.cpu().numpy()
        exact = torch.reshape(function_u_exact(Z1), [N0 + 1, N0 + 1]).cpu().numpy()
    err = np.sqrt(np.sum(np.square(exact - pred)) / np.sum(np.square(exact)))
    print("Error:", err)

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)

    plt.figure()
    ax2 = plt.subplot(1, 1, 1)
    h1 = plt.imshow(exact, interpolation='nearest', cmap='rainbow',
                    extent=[0, 1, 0, 1],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h1, cax=cax)

    plt.figure()
    ax2 = plt.subplot(1, 1, 1)
    h1 = plt.imshow(exact-pred, interpolation='nearest', cmap='rainbow',
                    extent=[0, 1, 0, 1],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h1, cax=cax)
    plt.show()


if __name__ == '__main__':
    main()
