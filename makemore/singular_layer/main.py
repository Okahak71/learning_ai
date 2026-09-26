import torch as t
import torch.nn.functional as F

words = open("names.txt", 'r').read().splitlines()
print(words[:10])
print(len(words))

cnt = t.zeros((27, 27), dtype = t.int32)

st = sorted(list(set(''.join(words))))

stoi = {s:i + 1 for i, s in enumerate(st)}
stoi["."] = 0

itos = {i : s for s, i in stoi.items()}

mp = {}

for wr in words:
    w = ["."] + list(wr) + ["."]
    for i in range(len(w) - 1):
        pr = (w[i], w[i + 1])
        mp[pr] = mp.get(pr, 0) + 1

for (w1, w2), vl in mp.items():
    cnt[stoi[w1], stoi[w2]] = vl

rng = t.Generator().manual_seed(2147483647)

p = (cnt + 1).float() / t.einsum("ij -> i", (cnt + 1).float()).unsqueeze(1)


# 1 2 3      [6, 9] -> (divide)  (2, 3) / (2, 1) -> [[6, 6, 6], [9, 9, 9]]
# 2 3 4 --> 

# id = 0

# for i in range(5):
#     s = []
#     while True:
#         id = t.multinomial(p[id], num_samples=1, replacement=True, generator=rng).item()
#         s.append(itos[id])
#         if id == 0:
#             break
#     print(''.join(s))

# -----------predictor level 1----------

# avg = [0, 0]

# for s in ["andrej"]:
#     ch = ['.'] + list(s) + ['.']
#     for i in range(len(ch) - 1):
#         id1 = stoi[ch[i]]
#         id2 = stoi[ch[i + 1]]
#         prob = p[id1, id2]
#         lg = t.log(prob)
#         avg = [avg[0] + lg, avg[1] + 1]
#         print(f"{ch[i]}{ch[i + 1]} -> {prob:.4f} , {lg:.4f}")

# avg[0] = -avg[0]

# nll = avg[0] / avg[1]

# print(f"{nll:.10f}")

# ---------------------perdictor level 2 // neural network -------------------

xs, ys = [], []

for s in words:
    ch = ['.'] + list(s) + ['.']
    for i in range(len(ch) - 1):
        id1 = stoi[ch[i]]
        id2 = stoi[ch[i + 1]]
        xs.append(id1)
        ys.append(id2)

xs = t.tensor(xs)
ys = t.tensor(ys)

# print(f'xs : {xs} \n ------ys : {ys}')

xx = F.one_hot(xs, num_classes=27).float()

# print(f'xx ----> {xx}')

w = t.randn((27, 27), generator=rng)

logits = (xx @ w) # log counts

count = logits.exp() # equals to cnt

prob =  count / count.sum(1, keepdims = True)
print(prob.shape, cnt.shape)

# ch = ['.'] + list(s) + ['.']
# for i in range(len(ch) - 1):
#     x = stoi[ch[i]]
#     y = stoi[ch[i + 1]]
#     print(f'      {x} ---> {y}')
#     # print(f"probability of y after x is {prob[i, y]}")
#     # for not going too small use LOG
#     lg = t.log(prob[i, y])
#     # print(f"the logarithm of prob is {-lg}")
#     nll += -lg
#     sz += 1

learning_rate = -75

sz = xs.nelement()

rng = t.Generator().manual_seed(2147483647)
w = t.randn((27, 27), generator=rng, requires_grad=True)

for i in range(1000):
    #forward pass
    xx = F.one_hot(xs, num_classes=27).float()
    logits = xx @ w
    count = (logits + 1).exp()
    prob = count / count.sum(1, keepdim=True)
    loss = -prob[t.arange(sz), ys].log().mean()

    print(loss.data)
    #backward
    w.grad = None
    loss.backward()

    w.data += learning_rate * w.grad

# 100 epoch && learning_rate = -50 ---> loss = 2.4729

# 1000 epoch && learning_rate = -5 ---> loss = 2.4628

for i in range(5):
    s = []
    id = 0
    while True:
        lg_t = w[id].unsqueeze(0)
        cc = lg_t.exp()
        probb = cc / cc.sum(1, keepdim=True)

        id = t.multinomial(probb, num_samples=1, replacement=True, generator=rng).item()
        s.append(itos[id])
        if id == 0:
            break
    print(''.join(s))

#min loss ~ 2.46
