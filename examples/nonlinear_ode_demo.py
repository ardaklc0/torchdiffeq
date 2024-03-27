import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


true_y0 = torch.tensor([2.]).to(device) # initial condition; t(0) = 0
t = torch.linspace(1., 5.-1e-6, args.data_size).to(device) # time step t_0 to t_N for (--data_size) in our case 1000
true_dy = torch.tensor([1.]).to(device) # y' = 1 * e^y 


class Nonhomogenous(nn.Module):
    def forward(self, t, y):
        # print(y, torch.exp(y) * true_dy[0])
        return (y / (2 * t)) + (t ** 2 / (2 * y))  * true_dy[0]

with torch.no_grad():
    true_y = odeint(Nonhomogenous(), true_y0, t, method='dopri5')


def visualize(true_y, pred_y, odefunc, itr):
    plt.clf()
    plt.title('Trajectories')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.plot(t.cpu().numpy(), true_y.cpu().numpy(), label='True Trajectory', color='blue')
    plt.plot(t.cpu().numpy(), pred_y.cpu().numpy(), '--', label='Predicted Trajectory', color='red')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 10)
    plt.grid(True)
    plt.axvline(0)
    plt.axhline(0)
    plt.draw()
    plt.pause(0.001)


class ODEFunc (nn. Module ):
    def __init__ (self, y_dim =1, n_hidden =64) :
        super ( ODEFunc, self ). __init__ ()
        self.net = nn. Sequential (
            nn.Linear(y_dim, n_hidden ),
            nn.ReLU(),
            nn.Linear(n_hidden, y_dim )
        )
    def forward (self , t, y):
        return self.net(y)

s = torch.from_numpy(
np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
batch_y0 = true_y[s]  # (M, D)
batch_t = t[:args.batch_time]  # (T)
batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)

if __name__ == '__main__':
    ii = 0
    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = F.mse_loss(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = F.mse_loss(pred_y, true_y)
                visualize(true_y, pred_y, func, ii)
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            ii += 1
