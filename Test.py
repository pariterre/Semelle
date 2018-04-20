from pyomeca import fileio
from pyomeca.math import matrix
from pyomeca.show.vtk import Model
from pyomeca.show.vtk import Window
import numpy as np
import math
import time
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

# Make a constant seed
seed = np.random.seed(42)

# Configuration variables
trainingFileName = "data/subject1/training.c3d"
reconstructFileName = "data/subject1/static.c3d"
saveReconstructionFileName = "result/static.csv"
name_markers = ['TriadeRightFrontF', 'TriadeRightFrontT', 'TriadeRightFrontB',
                'TriadeRightBackF', 'TriadeRightBackT', 'TriadeRightBackB',
                'TriadeBackR', 'TriadeBackT', 'TriadeBackL',
                'TriadeLeftBackB', 'TriadeLeftBackT', 'TriadeLeftBackF',
                'TriadeLeftMiddleB', 'TriadeLeftMiddleT', 'TriadeLeftMiddleF',
                'TriadeLeftFrontB', 'TriadeLeftFrontT', 'TriadeLeftFrontF', 'ContourLeftFront', 'ContourFront1',
                'ContourFront2', 'ContourFront3', 'ContourFront4', 'ContourRightFront', 'ContourRight1',
                'ContourRight2', 'ContourRight3', 'ContourRight4', 'ContourRight5', 'ContourRight6', 'ContourRight7',
                'ContourRight8', 'ContourBack', 'ContourLeft8', 'ContourLeft7', 'ContourLeft6', 'ContourLeft5',
                'ContourLeff4', 'ContourLeft3', 'ContourLeft2', 'ContourLeft1',
                'Interior1', 'Interior2', 'Interior3', 'Interior4', 'Interior5', 'Interior6', 'Interior7',
                'Interior8', 'Interior9', 'Interior10', 'Interior11', 'Interior12', 'Interior13', 'Interior14',
                'Interior15', 'Interior16', 'Interior17', 'Interior18','Interior19', 'Interior20', 'Interior21',
                'Interior22', 'Interior23','Interior24', 'Interior25', 'Interior26', 'Interior27', 'Interior28',
                'Interior29', 'Interior30','Interior31', 'Interior32']
modelName = "Orthese"
nTriPod = 6
nMarkOnInsole = 55
forceRetrainingModel = False
rtm = [6, 7, 8]  # idx of the markers in training file of the Tripod at the back of the insole [left, top, right]

# load dataSet
dataSet, meta = fileio.read_c3d(trainingFileName, get_metadata=True, names=name_markers)
# dataSet = dataSet[:, :, 900:-1]

# Get markers in a known reference frame
rt = matrix.define_axes(dataSet, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)
dataSet = dataSet.rotate(rt.transpose())

# split into input (X) and output (Y) variables
X = matrix.reshape_3d_to_2d_matrix(dataSet[:, 0:nTriPod*3, :])
Y = matrix.reshape_3d_to_2d_matrix(dataSet[:, nTriPod*3:nTriPod*3 + nMarkOnInsole, :])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)


# define and train the model
def baseline_model():
    # create model
    m = Sequential()
    m.add(Dense(6 * nMarkOnInsole, input_dim=nTriPod * 3 * 3, kernel_initializer='normal', activation='relu'))
    m.add(Dense(3 * nMarkOnInsole, kernel_initializer='normal'))
    # Compile model
    m.compile(loss='mean_squared_error', optimizer='adam')
    m.fit(X_train, y_train, epochs=10, batch_size=2, verbose=2)
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
dataSetAll = fileio.read_c3d(reconstructFileName, names=['TriadeRightFrontF', 'TriadeRightFrontT',
                                                            'TriadeRightFrontB', 'TriadeRightBackF',
                                                            'TriadeRightBackT', 'TriadeRightBackB',
                                                            'TriadeBackR', 'TriadeBackT',
                                                            'TriadeBackL', 'TriadeLeftBackB',
                                                            'TriadeLeftBackT', 'TriadeLeftBackF',
                                                            'TriadeLeftMiddleB', 'TriadeLeftMiddleT',
                                                            'TriadeLeftMiddleF', 'TriadeLeftFrontB',
                                                            'TriadeLeftFrontT', 'TriadeLeftFrontF'])
dataSetTripod = fileio.read_c3d(reconstructFileName, names=['TriadeRightFrontF', 'TriadeRightFrontT',
                                                            'TriadeRightFrontB', 'TriadeRightBackF',
                                                            'TriadeRightBackT', 'TriadeRightBackB',
                                                            'TriadeBackR', 'TriadeBackT',
                                                            'TriadeBackL', 'TriadeLeftBackB',
                                                            'TriadeLeftBackT', 'TriadeLeftBackF',
                                                            'TriadeLeftMiddleB', 'TriadeLeftMiddleT',
                                                            'TriadeLeftMiddleF', 'TriadeLeftFrontB',
                                                            'TriadeLeftFrontT', 'TriadeLeftFrontF'])

# Get markers in a known reference frame
rt = matrix.define_axes(dataSetTripod, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)
dataSetTripod = dataSetTripod.rotate(rt.transpose())
dataSetAll = dataSetAll.rotate(rt.transpose())
# rt = matrix.define_axes(dataSetTripod, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)

# split into input (X) and output (Y) variables
X_data = matrix.reshape_3d_to_2d_matrix(dataSetTripod[:, 0:nTriPod*3, :])
y_data = matrix.reshape_3d_to_2d_matrix(dataSetAll)
t = time.time()
y_recons = model.predict(X_data)

print(X_data.shape)
print(time.time() - t)

# View it.
# Convert back points matrix to Vectors3d
TReal = dataSetAll  # matrix.reshape_2d_to_3d_matrix(X_data)
TPred = matrix.reshape_2d_to_3d_matrix(y_recons)
fileio.write_csv(saveReconstructionFileName, TPred)

# Put back the markers in global frame
# TReal = TReal.rotate(rt)
# TPred = TPred.rotate(rt)

# Create a figure
fig = Window(background_color=(.9, .9, .9))

# Add models to the figure
h_real = Model(fig, markers_color=(1, 0, 0), markers_size=5.0, markers_opacity=1)
h_pred = Model(fig, markers_color=(0, 0, 0), markers_size=10.0, markers_opacity=.5)

# Calculate total error
# error = (TReal[0:3, nTriPod*3:nTriPod*3 + nMarkOnInsole, :] - TPred[0:3, :, :]).norm()
# mean_rmse = np.mean(error)
# mean_std = np.mean(np.std(error, axis=0))
# print("Mean RSME is " + str(mean_rmse) + " ± " + str(mean_std) + " mm")

# Loop and show
for i in range(TReal.shape[2]):
    if not fig.is_active:
        break

    # Print on console where we are
    if i % 100 == 0:
        print(i)

    h_real.update_markers(TReal[:, :, i])
    h_pred.update_markers(TPred[:, :, i])
    # h_real.update_rt(rt[:, :, i])

    # Update graphics
    fig.update_frame()
