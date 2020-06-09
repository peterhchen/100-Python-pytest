import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
# data preprocessed
train_labels = []
train_samples = []

for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

train_labels, train_samples = shuffle(train_labels, train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

test_labels = []
test_samples = []
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)
#scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))
# for i in scaled_train_samples:
#     print(i)
print('\nscaled_train_samples[:5]:')
print(scaled_train_samples[:5])
print('train_labels[:5]:', train_labels[:5])

# Set up Cuda GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set up modle
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
# model summary
model.summary()

# model compile
model.compile(optimizer=Adam(learning_rate=0.0001), \
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model fit
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, \
    batch_size=10, epochs=30, verbose=2)

# Prediction [experienced, no expereinced]
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
print('\ntest_samples[:5]:')
print(test_samples[:5])
print('\nscaled_test_samples[:5]:')
print(scaled_test_samples[:5])
print('\ntest_labels[:5]:')
print(test_labels[:5])
print('\npredictions[:5]:')
print(predictions[:5])

rounded_predictions = np.argmax(predictions, axis=-1)
#rounded_predictions = np.argmax(predictions, axis=1)
print ('\nrounded_predictions[:5]:')
print (rounded_predictions[:5])
# axis=1 or -1 have same results [1 0 1 0 0 ...] in row direction. 
# axis=0 get [66 30] which is column direction