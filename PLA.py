# perceptron learning algorithm

# data generation
import random
import numpy as np
import copy as cp
import math
target = [random.random() - 0.5, random.random() - 0.5]

data = [[random.random() - 0.5, random.random() - 0.5, 0] for _ in range(10000)]

for x in data:
    if np.dot(target, x[0:2]) > 0:
        x[2] = 1
    else:
        x[2] = -1
print(data)

# unit data

udata = cp.deepcopy(data)
for x in udata:
    div = math.sqrt(x[0] * x[0] + x[1] * x[1])
    x[0] = x[0] / div
    x[1] = x[1] / div

# calculate
w = udata[0][0:2]
print(w)
stop = len(udata)
print("stop: {}".format(stop))
idx = -1
cnt = 0
while stop > 0:
    idx = (idx + 1) % len(udata)
    ans = np.dot(w, udata[idx][0:2])
    cnt = cnt + 1
    if ans * udata[idx][2] <= 0:
        stop = len(udata)
        if ans * udata[idx][2] < 0 and not w == udata[idx][0:2]:
            for i in range(len(udata[0]) - 1):
                w[i] = w[i] + udata[idx][i] * udata[idx][2]
        else:
            temp = -w[0]
            w[0] = w[1]
            w[1] = temp
        print("w{}".format(w))
        print("{} target {}".format(-1 * w[0] / w[1], target))
    stop = stop - 1
print(w)
print("Finish! Iterations: {} Final: {} Target: {}".format(cnt, -1 * w[0] / w[1], -1 * target[0] / target[1]))
