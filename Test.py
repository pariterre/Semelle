# Setting up Python
from pyomeca import data
from pyomeca.math import matrix
from pyomeca.show.vtk import Model
from pyomeca.show.vtk import Window
import numpy
import math
import time
import h5py
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# Make a constant seed
seed = numpy.random.seed(42)

# Configuration variables
trainingFileName = "TestOrthese02.c3d"
reconstructFileName = "TestOrthese03.c3d"
modelName = "Orthese"
nTriPod = 6
nMarkOnInsole = 54
forceRetrainingModel = False


# load dataSet
# split into input (X) and output (Y) variables
dataSet = data.load_marker_data(trainingFileName)
X = matrix.reshape_3d_to_2d_matrix(dataSet[:, 0:nTriPod*3, :])
Y = matrix.reshape_3d_to_2d_matrix(dataSet[:, nTriPod*3:nTriPod*3 + nMarkOnInsole, :])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)


# define and train the model
def baseline_model():
    # create model
    m = Sequential()
    m.add(Dense(3 * (72 - nTriPod * 3), input_dim=nTriPod * 3 * 3, kernel_initializer='normal', activation='relu'))
    m.add(Dense(3 * (72 - nTriPod * 3), kernel_initializer='normal', activation='relu'))
    m.add(Dense(3 * (72 - nTriPod * 3), kernel_initializer='normal'))
    # Compile model
    m.compile(loss='mean_squared_error', optimizer='adam')
    m.fit(X_train, y_train, epochs=1000, batch_size=2, verbose=2)
    return m

if forceRetrainingModel:
    model = baseline_model()
    model.save(modelName + ".h5")
else:
    model = load_model(modelName + ".h5")

# Show model performance on training and
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

# Reconstruct kinematics
# load dataSet
dataSet = data.load_marker_data(trainingFileName)  # reconstructFileName)
# split into input (X) and output (Y) variables
X_data = matrix.reshape_3d_to_2d_matrix(dataSet[:, 0:nTriPod*3, :])
y_data = matrix.reshape_3d_to_2d_matrix(dataSet[:, nTriPod*3:nTriPod*3+nMarkOnInsole, :])
t = time.time()
y_recons = model.predict(X_data)
print(X_data.shape)
print(time.time() - t)

# View it.
# Convert back points matrix to Vectors3d
TReal = matrix.reshape_2d_to_3d_matrix(y_data)
TPred = matrix.reshape_2d_to_3d_matrix(y_recons)

# Create a figure
fig = Window(background_color=(.5, .5, .5))

# Add models to the figure
h_real = Model(fig, markers_color=(1, 0, 0), markers_size=5.0, markers_opacity=1)
h_real.new_marker_set(TReal[:, :, 0])
h_pred = Model(fig, markers_color=(0, 0, 0), markers_size=10.0, markers_opacity=.5)
h_pred.new_marker_set(TPred[:, :, 0])

# Loop and show
for i in range(TReal.shape[2]):
    if not fig.is_active:
        break

    # Print on console where we are
    if i % 100 == 0:
        print(i)

    h_real.update_markers(TReal[:, :, i])
    h_pred.update_markers(TReal[:, :, i])

    # Update graphics
    fig.update_frame()
