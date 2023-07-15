import numpy as np

softmax_outputs = [
    [0.7, 0.2, 0.1],
    [0.5, 0.1, 0.4],
    [0.02, 0.9, 0.08]
]

class_targets = np.array([0, 1, 1])

# get index of max in 2nd axis, 
predictions = np.argmax(softmax_outputs, axis=1)
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

print('softmax_outputs', softmax_outputs)

print('np.argmax(softmax_outputs, axis=1)', np.argmax(softmax_outputs, axis=1))
print('predictions === class_targets', predictions == class_targets)
accuracy = np.mean(predictions == class_targets)
print('accuracy', accuracy)