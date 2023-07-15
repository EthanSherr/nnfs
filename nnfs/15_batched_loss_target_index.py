import numpy as np

# probabilities for 3 samples
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])



# 0 index is dog, 
# 1 index is cat,
# 2 index is huma
# samples: first is dog, second is cat, third is cat
class_targets = [0, 1, 1]

print('softmax_outputs[[0, 1, 2], class_targets]', softmax_outputs[[0, 1, 2], class_targets])


print('range(len(softmax_outputs))', range(len(softmax_outputs)))
grabbed_indices = softmax_outputs[range(len(softmax_outputs)), class_targets]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)
print('neg_log', neg_log)
print('avg_loss', avg_loss)