import math

# example output from outuput layer of nnetwork, softmax prob distribution
softmax_output = [0.7, 0.1, 0.2]
# ground truth
target_output = [1, 0, 0]

loss = 0
for idx, y_hat in enumerate(softmax_output):
    y = target_output[idx]
    loss -= y * math.log(y_hat)

print('loss', loss)