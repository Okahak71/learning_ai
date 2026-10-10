import numpy as np
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_greedy(R, k, T, eps, alpha, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((R, k))
    #every q starts same and differ overtime
    Q = np.zeros((R, k))
    N = np.zeros((R, k))
    avg_rew = np.zeros(T)
    pct_opt = np.zeros(T)
    idx = np.arange(R)
    for t in range(1, T + 1):
        q += (0.01 if t < 10000 else 0) * rng.standard_normal((R, k))
        u = rng.random(R)
        rndm = rng.random(R)
        a = np.where(u < eps, (rndm * k).astype(int), np.argmax(Q, axis = 1))
        pct_opt[t - 1] = np.where(a == np.argmax(q, axis = 1), 1, 0).sum() / R
        rew = q[idx, a] + rng.standard_normal(R)
        avg_rew[t - 1] = rew.mean()
        N[idx, a] += 1
        Q[idx, a] += (alpha if alpha != None else 1 / N[idx, a]) * (rew - Q[idx, a])
    return avg_rew, pct_opt


def plot_curves(curves, path="bandit_ex25.png"):
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


curves = {
    "sample average (1/n)": run_greedy(2000, 10, 20000, 0.1, None, 0),
    "constant alpha=0.1":   run_greedy(2000, 10, 20000, 0.1, 0.1, 0),
}
plot_curves(curves)
