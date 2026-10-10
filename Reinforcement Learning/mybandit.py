import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def reward(a, q, rng):
    return q[a] + 2 * rng.next() - 1

def softmax(exps):
    exps = np.exp(exps - np.max(exps, axis=1, keepdims=True))
    return exps / np.sum(exps, axis = 1, keepdims=True)

def run_greedy(R, k, T, eps, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((R, k))
    Q = np.zeros((R, k)) #+ 5
    N = np.zeros((R, k))
    idx = np.arange(R)
    avg_rew = np.zeros(T)
    pct_opt = np.zeros(T, dtype=float)
    for t in range(1, T + 1):
        u = rng.random(R)
        iddx = rng.random(R)
        a = np.where(u < eps, (iddx * k).astype(int), np.argmax(Q, axis = 1))
        pct_opt[t - 1] = float(np.where(a == np.argmax(q, axis = 1), 1, 0).sum()) / R
        rew = q[idx, a] + rng.standard_normal(R)    #shape(R, )
        avg_rew[t - 1] = rew.mean()
        N[idx, a] += 1
        Q[idx, a] += (rew - Q[idx, a]) / N[idx, a]
    return avg_rew, pct_opt

def run_ucb(R, k, T, c, seed, q_true):
    rng = np.random.default_rng(seed)
    q_true = np.array(q_true)
    N = np.zeros((R, k))
    Q = np.zeros((R, k))
    sum = np.zeros(R)
    idx = np.arange(R)
    for t in range(T):
        if t < k:
            a = np.full((R), t, dtype=int)
        else:
            temp_Q = Q + (c * (math.log(t + 1) / N)**0.5)
            a = np.argmax(temp_Q, axis=1)
        vl = rng.random(R)
        rew = q_true[a] + 2 * vl - 1
        sum += rew
        N[idx, a] += 1
        Q[idx, a] += (rew - Q[idx, a]) / N[idx, a]
    return sum.mean(), (Q.argmax(axis = 1) == np.argmax(q_true)).mean()

def run_grad(R, k, T, alpha, seed, q_true):
    rng = np.random.default_rng(seed)
    sum = np.zeros((R))
    H = np.zeros((R, k))
    pi = softmax(H)
    rb = np.zeros((R))
    idx = np.arange(R)
    q_true = np.array(q_true)
    for t in range(1, T + 1):
        u = rng.random(R)
        a = (np.cumsum(pi, axis=1) < u[:, None]).sum(axis = 1)
        a = np.minimum(a, k - 1)
        vl = rng.random(R)
        rew = q_true[a] + 2 * vl - 1
        # print(rew)
        sum += rew
        rb = rb + (rew - rb) / t
        # print("Part 2: done")
        onehot = np.eye(k)[a]                      # shape (R, k)
        H += alpha * (rew - rb)[:, None] * (onehot - pi)
        # print("Part 3: done")
        pi = softmax(H)
    return sum.mean(), (pi.argmax(axis = 1) == np.argmax(q_true)).mean()

def plot_curves(curves, path="bandit_curves_10000.png"):
    # curves: {label: (avg_reward, pct_opt)}, pct_opt as a fraction 0..1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for label, (ar, po) in curves.items():
        steps = np.arange(1, len(ar) + 1)
        ax1.plot(steps, ar, label=label)
        ax2.plot(steps, po * 100, label=label)
    ax1.set_ylabel("Average reward")
    ax2.set_ylabel("% optimal action")
    ax2.set_xlabel("Steps")
    ax1.legend(); ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)

model = input()
# R, K, T, var, seed = list(input().split())

res = []

if model == "e-greedy":
    # R, K, T, eps, seed = int(R), int(K), int(T), float(var), int(seed)
    # res = run_greedy(R, K, T, eps, seed)
    curves = {f"eps={e}": run_greedy(20000, 10, 1000, e, 0) for e in (0, 0.01, 0.1)}
    plot_curves(curves)
    
elif model == "ucb":
    R, K, T, c, seed = int( ), int(K), int(T), float(var), int(seed)
    res = run_ucb(R, K, T, c, seed)

else:
    R, K, T, alpha, seed = int(R), int(K), int(T), float(var), int(seed)
    res = run_grad(R, K, T, alpha, seed)