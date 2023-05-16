import matplotlib.pyplot as plt

# Evaluation metrics
eval_metrics = [{'eval_loss': 0.5180476903915405, 'eval_accuracy': 0.8226495726495726, 'eval_runtime': 3.8701, 'eval_samples_per_second': 241.854, 'eval_steps_per_second': 7.752, 'epoch': 1.0} ,
                {'eval_loss': 0.43801945447921753, 'eval_accuracy': 0.8910256410256411, 'eval_runtime': 3.9384, 'eval_samples_per_second': 237.662, 'eval_steps_per_second': 7.617, 'epoch': 2.0} ,
                {'eval_loss': 0.58245450258255, 'eval_accuracy': 0.8792735042735043, 'eval_runtime': 3.9106, 'eval_samples_per_second': 239.35, 'eval_steps_per_second': 7.671, 'epoch': 3.0},
                {'eval_loss': 0.5886743068695068, 'eval_accuracy': 0.8856837606837606, 'eval_runtime': 3.6434, 'eval_samples_per_second': 256.906, 'eval_steps_per_second': 8.234, 'epoch': 4.0},
                {'eval_loss': 0.5752162933349609, 'eval_accuracy': 0.9017094017094017, 'eval_runtime': 3.6443, 'eval_samples_per_second': 256.841, 'eval_steps_per_second': 8.232, 'epoch': 5.0},
                {'eval_loss': 0.6967033743858337, 'eval_accuracy': 0.9038461538461539, 'eval_runtime': 3.6143, 'eval_samples_per_second': 258.972, 'eval_steps_per_second': 8.3, 'epoch': 6.0},
                {'eval_loss': 0.774192214012146, 'eval_accuracy': 0.9049145299145299, 'eval_runtime': 3.6144, 'eval_samples_per_second': 258.962, 'eval_steps_per_second': 8.3, 'epoch': 7.0},
                {'eval_loss': 0.723403811454773, 'eval_accuracy': 0.9123931623931624, 'eval_runtime': 3.6436, 'eval_samples_per_second': 256.888, 'eval_steps_per_second': 8.234, 'epoch': 8.0},
                {'eval_loss': 0.7567483186721802, 'eval_accuracy': 0.9102564102564102, 'eval_runtime': 3.7814, 'eval_samples_per_second': 247.527, 'eval_steps_per_second': 7.934, 'epoch': 9.0},
                {'eval_loss': 0.7439692616462708, 'eval_accuracy': 0.9134615384615384, 'eval_runtime': 3.7995, 'eval_samples_per_second': 246.351, 'eval_steps_per_second': 7.896, 'epoch': 10.0},
                ]

# Training metrics
train_metrics = [{'loss': 0.3468, 'learning_rate': 5.726495726495726e-05, 'epoch': 4.27},
                 {'loss': 0.0361, 'learning_rate': 1.4529914529914531e-05, 'epoch': 8.55},
                 {'train_runtime': 496.3807, 'train_samples_per_second': 75.345, 'train_steps_per_second': 2.357, 'train_loss': 0.1641991087514111, 'epoch': 10.0}]

# Extract the relevant metrics
eval_loss = [metrics.get('eval_loss', None) for metrics in eval_metrics]
eval_accuracy = [metrics.get('eval_accuracy', None) for metrics in eval_metrics]
epochs = [metrics.get('epoch', None) for metrics in eval_metrics]

train_loss = [metrics.get('loss', None) for metrics in train_metrics]
learning_rate = [metrics.get('learning_rate', None) for metrics in train_metrics]
train_epochs = [metrics.get('epoch', None) for metrics in train_metrics]

# Plotting the evaluation metrics
fig, ax= plt.subplots(figsize=(8, 6))

ax.plot(epochs, eval_loss, label='Eval Loss')
ax.plot(epochs, eval_accuracy, label='Eval Accuracy')

ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Evaluation Metrics')
ax.legend()

plt.show()

# Plotting the training metrics
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(train_epochs, train_loss, label='Train Loss')
ax.plot(train_epochs, learning_rate, label='Learning Rate')

ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Training Metrics')
ax.legend()

plt.show()
