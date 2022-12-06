import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load the breast cancer dataset
data = keras.datasets.breast_cancer()
(x_train, y_train), (x_test, y_test) = data

# Build the model with 10 hidden layers
model = keras.Sequential()
model.add(Dense(64, input_shape=(30,), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

for _ in range(9):
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model with TensorBoard callback
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, epochs=10, validation_data=(
    x_test, y_test), callbacks=[tensorboard_callback])

# Evaluate the model and show the results
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', results[0])
print('Test accuracy:', results[1])

# Plot the results with TensorBoard
%tensorboard - -logdir logs/fit
