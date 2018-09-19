import math
import time
import os.path
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

from pyomeca import Markers3d, RotoTrans
from pyomeca.show.vtk import VtkModel, VtkWindow

# Make a constant seed
seed = np.random.seed(42)

# Configuration variables
F_path = "/home/laboratoire/mnt/F/"
insole_type = "soft"  # rigid

trainingFileName = f"{F_path}Data/Footi/Data/2018-07-30/test_validation/{insole_type}_othosis/{insole_type}_alldirections.c3d"
reconstructFileName = f"{F_path}Data/Footi/Data/2018-07-30/test_validation/{insole_type}_othosis/{insole_type}_flexion.c3d"
# reconstructFileName = f"{F_path}Data/Footi/Data/Healthy/2018-05-10/GaDe/Deformation_filledGaps/Walking_2.c3d"
saveReconstructionFileName = f"result/{insole_type}_inv_eve.csv"
model_file_path = f"result/{insole_type}.h5"

name_markers = ['TriadeRightFrontF', 'TriadeRightFrontT', 'TriadeRightFrontB',
                'TriadeRightBackF', 'TriadeRightBackT', 'TriadeRightBackB',
                'TriadeBackR', 'TriadeBackT', 'TriadeBackL',
                'TriadeLeftBackB', 'TriadeLeftBackT', 'TriadeLeftBackF',
                'TriadeLeftMiddleB', 'TriadeLeftMiddleT', 'TriadeLeftMiddleF',
                'TriadeLeftFrontB', 'TriadeLeftFrontT', 'TriadeLeftFrontF',
                'ContourLeftFront', 'ContourFront1',
                'ContourFront2', 'ContourFront3', 'ContourFront4', 'ContourRightFront', 'ContourRight1',
                'ContourRight2', 'ContourRight3', 'ContourRight4', 'ContourRight5', 'ContourRight6', 'ContourRight7',
                'ContourRight8', 'ContourBack', 'ContourLeft8', 'ContourLeft7', 'ContourLeft6', 'ContourLeft5',
                'ContourLeff4', 'ContourLeft3', 'ContourLeft2', 'ContourLeft1',
                'Interior1', 'Interior2', 'Interior3', 'Interior4', 'Interior5', 'Interior6', 'Interior7',
                'Interior8', 'Interior9', 'Interior10', 'Interior11', 'Interior12', 'Interior13', 'Interior14',
                'Interior15', 'Interior16', 'Interior17', 'Interior18','Interior19', 'Interior20', 'Interior21',
                'Interior22', 'Interior23','Interior24', 'Interior25', 'Interior26', 'Interior27', 'Interior28',
                'Interior29', 'Interior30','Interior31', 'Interior32']
nTriPod = 6
nMarkOnInsole = 55
forceRetrainingModel = False
rtm = [6, 7, 8]  # idx of the markers in training file of the Tripod at the back of the insole [left, top, right]
show_in_global = True
calculate_error = True
new_marker_tripod_names = False

# load dataSet
dataSet = Markers3d.from_c3d(trainingFileName, names=name_markers).low_pass(freq=100, order=4, cutoff=10)

# Get markers in a known reference frame
# rt = RotoTrans.define_axes(dataSet, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)
rt = RotoTrans.define_axes(dataSet, [0, 4*3], [[0, 5*3], [2*3+1, 2*3+1]], "xz", "z", [0, 2*3+1, 5*3])
dataSet = dataSet.rotate(rt.transpose())

# split into input (X) and output (Y) variables
X = dataSet[:, 0:nTriPod*3, :].to_2d()
Y = dataSet[:, nTriPod*3:nTriPod*3 + nMarkOnInsole, :].to_2d()
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


if not os.path.isfile(model_file_path) or forceRetrainingModel:
    model = baseline_model()
    model.save(model_file_path)
else:
    model = load_model(model_file_path)

# Show model performance on training and
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

# Reconstruct kinematics
# load dataSet
if new_marker_tripod_names:
    markers_tripod = ['TriadeLateralFrontF', 'TriadeLateralFrontT', 'TriadeLateraltFrontB',
                      'TriadeLateralBackF', 'TriadeLateralBackT', 'TriadeLateralBackB',
                      'TriadeBackL', 'TriadeBackT', 'TriadeBackM',
                      'TriadeMedialBackB', 'TriadeMedialBackT', 'TriadeMedialBackF',
                      'TriadeMedialMiddleB', 'TriadeMedialMiddleT', 'TriadeMedialMiddleF',
                      'TriadeMedialFrontB', 'TriadeMedialFrontT', 'TriadeMedialFrontF']
else:
    markers_tripod = ['TriadeRightFrontF', 'TriadeRightFrontT', 'TriadeRightFrontB',
                      'TriadeRightBackF', 'TriadeRightBackT', 'TriadeRightBackB',
                      'TriadeBackR', 'TriadeBackT', 'TriadeBackL',
                      'TriadeLeftBackB', 'TriadeLeftBackT', 'TriadeLeftBackF',
                      'TriadeLeftMiddleB', 'TriadeLeftMiddleT', 'TriadeLeftMiddleF',
                      'TriadeLeftFrontB', 'TriadeLeftFrontT', 'TriadeLeftFrontF']
markers_all = markers_tripod
if calculate_error:
    markers_all = markers_all + \
                  ['ContourLeftFront', 'ContourFront1',
                   'ContourFront2', 'ContourFront3', 'ContourFront4', 'ContourRightFront', 'ContourRight1',
                   'ContourRight2', 'ContourRight3', 'ContourRight4', 'ContourRight5', 'ContourRight6', 'ContourRight7',
                   'ContourRight8', 'ContourBack', 'ContourLeft8', 'ContourLeft7', 'ContourLeft6', 'ContourLeft5',
                   'ContourLeff4', 'ContourLeft3', 'ContourLeft2', 'ContourLeft1',
                   'Interior1', 'Interior2', 'Interior3', 'Interior4', 'Interior5', 'Interior6', 'Interior7',
                   'Interior8', 'Interior9', 'Interior10', 'Interior11', 'Interior12', 'Interior13', 'Interior14',
                   'Interior15', 'Interior16', 'Interior17', 'Interior18','Interior19', 'Interior20', 'Interior21',
                   'Interior22', 'Interior23','Interior24', 'Interior25', 'Interior26', 'Interior27', 'Interior28',
                   'Interior29', 'Interior30','Interior31', 'Interior32']
dataSetAll = Markers3d.from_c3d(reconstructFileName, names=markers_all).low_pass(freq=100, order=4, cutoff=10)
dataSetTripod = Markers3d.from_c3d(reconstructFileName, names=markers_tripod).low_pass(freq=100, order=4, cutoff=10)

# Get markers in a known reference frame
# rt = RotoTrans.define_axes(dataSetTripod, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)
rt = RotoTrans.define_axes(dataSetTripod, [0, 4*3], [[0, 5*3], [2*3+1, 2*3+1]], "xz", "z", [0, 2*3+1, 5*3])
dataSetTripod = dataSetTripod.rotate(rt.transpose())
dataSetAll = dataSetAll.rotate(rt.transpose())
# rt = matrix.define_axes(dataSetTripod, [rtm[2], rtm[0]], [[rtm[0], rtm[1]], [rtm[2], rtm[1]]], "xz", "z", rtm)

# split into input (X) and output (Y) variables
X_data = dataSetTripod[:, 0:nTriPod*3, :].to_2d()
y_data = dataSetAll.to_2d()
t = time.time()
y_recons = model.predict(X_data)
print(f"Prediction done in {time.time() - t} seconds")

# View it.
# Convert back points matrix to Vectors3d
TReal = dataSetAll  # Markers3d(X_data)
TPred = Markers3d(y_recons)
TPred.to_csv(saveReconstructionFileName)

# Put back the markers in global frame
if show_in_global:
    TReal = TReal.rotate(rt)
    TPred = TPred.rotate(rt)

# Create a figure
fig = VtkWindow(background_color=(.9, .9, .9))

# Add models to the figure
h_real = VtkModel(fig, markers_color=(1, 0, 0), markers_size=5.0, markers_opacity=1)
h_pred = VtkModel(fig, markers_color=(0, 0, 0), markers_size=10.0, markers_opacity=.5)

# Calculate total error [For the paper]
if calculate_error:
    error = (TReal[0:3, nTriPod*3:nTriPod*3 + nMarkOnInsole, :] - TPred[0:3, :, :]).norm()
    mean_rmse = np.mean(error)
    mean_std = np.mean(np.std(error, axis=0))
    print("Mean RSME is " + str(mean_rmse) + " ± " + str(mean_std) + " mm")

# Loop and show
for i in range(TReal.shape[2]):
    if not fig.is_active:
        break

    # Print on console where we are
    if i % 100 == 0:
        print(i)

    h_real.update_markers(TReal[:, :, i])
    h_pred.update_markers(TPred[:, :, i])
    if show_in_global:
        h_real.update_rt(rt[:, :, i])

    # Update graphics
    fig.update_frame()
