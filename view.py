import numpy as np
import os
from matplotlib import pyplot as plt

datasets = "cifar10"
# datasets = "mnist"
# datasets = "cifar100"
# datasets = "F-MNIST"
# datasets = "SVHN"

noise_type="symmetric"
# noise_type="asymmetric"
# noise_type="pairflip"
# noise_type="tridiagonal"

noise_rate = 0.5

folder = "results/" + datasets + "/coteaching/"

file = open(folder + "cifar10_coteaching_symmetric_0.5_resPam_resCoor.txt")

model1 = []
model2 = []

flag = 0
while 1:
    flag = flag + 1

    line = file.readline()
    if flag < 2:
        continue
    if not line:
        break
    temp = list(filter(None, line.split(" ")))
    print(temp)

    model1.append(float(temp[3]))

    model2.append(float(temp[4]))

i = len(model2)
sum1 = 0
sum2 = 0
for j in range(10):
    t = i - j - 1
    sum1 = sum1 + model1[t]
    sum2 = sum2 + model2[t]

average1 = sum1 / 10
average1 = round(average1, 2)

average2 = sum2 / 10
average2 = round(average2, 2)

variance1 = 0
variance2 = 0

for j in range(10):
    t = i - j - 1
    variance1 = variance1 + (model1[t] - average1) * (model1[t] - average1)
    variance2 = variance2 + (model2[t] - average2) * (model2[t] - average2)

variance1 = round(variance1, 2)
variance2 = round(variance2, 2)

plt.plot(range(len(model1)), model1, color="green", label="model1")
plt.plot(range(len(model2)), model2, color="red", label="model2")
plt.title(datasets + "_" + noise_type + "_" + str(noise_rate))
plt.text(8, 50, "model1 accuracy: " + str(average1) + "±" + str(variance1))
plt.text(8, 60, "model2 accuracy: " + str(average2) + "±" + str(variance2))
plt.legend()

plt.show()
