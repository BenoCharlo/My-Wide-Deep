# Name the configurations.
NAME = None  # Override in sub-classes

# Embedding size
EMBEDDING_SIZE = 50

# Number of training steps per epoch
# This doesn't need to match the size of the training set. Tensorboard
# updates are saved at the end of each epoch, so setting this to a
# smaller number means getting more frequent TensorBoard updates.
# Validation stats are also calculated at each epoch end and they
# might take a while, so don't set this too small to avoid spending
# a lot of time on validation stats.
STEPS_PER_EPOCH = 1000

# Number of validation steps to run at the end of every training epoch.
# A bigger number improves accuracy of validation stats, but slows
# down the training.
VALIDATION_STEPS = 50

# Number of classification classes (including background)
NUM_CLASSES = 2  # Override in sub-classes

# Batch size
BATCH_SIZE = 8

# Learning rate and momentum
LEARNING_RATE = 0.001
LEARNING_MOMENTUM = 0.9

# Weight decay regularization
WEIGHT_DECAY = 0.0001
