import numpy as np
import laspy
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  # If needed for future expansions

# Function to reshape and convert data
def prepare_data(data_attribute):
    return np.array(data_attribute).reshape(-1, 1)

# Load the LAS file
test_data = laspy.read("vall_small.las")

# Prepare data attributes
intensity = prepare_data(test_data.intensity)
x_coord = prepare_data(test_data.x)
y_coord = prepare_data(test_data.y)
z_coord = prepare_data(test_data.z)
return_number = prepare_data(test_data.return_number)
number_of_returns = prepare_data(test_data.number_of_returns)
scan_angle = prepare_data(test_data.scan_direction_flag)
classification = prepare_data(test_data.classification)
red = prepare_data(test_data.red)
green = prepare_data(test_data.green)
blue = prepare_data(test_data.blue)

# Identify points belonging to a specified class
target_class = 14
is_target_class = (classification == target_class).astype(int)
print(is_target_class)
print("NUMBER_OF_POINTS_IN_CLASS", np.sum(is_target_class))

# Combine relevant data attributes
data = np.column_stack((x_coord, y_coord, z_coord, red, green, blue))
test_model_data = np.column_stack((z_coord, intensity, return_number, number_of_returns, red, green, blue))

# Split and scale the data
row, column = test_model_data.shape
split_index = 2
# Example of splitting, not used further but kept for future reference
test_model_1, test_model_2 = test_model_data[:split_index, :], test_model_data[split_index:, :]

print(row)
print(row / 3)

# Load and predict with the model
filepath = 'MODEL'
model = load_model(filepath, compile=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_model = scaler.fit_transform(test_model_data)
test_prediction = model.predict(scaled_test_model)

# Save predictions to file
np.savetxt('TEST_PREDICTION.txt', test_prediction, delimiter=',')

# Analyze predictions
prediction_threshold = 0.7
prediction_classes = [1 if prob > prediction_threshold else 0 for prob in np.ravel(test_prediction)]
print(test_prediction)

# Find positions of correctly predicted points
correct_points_positions = np.nonzero(prediction_classes)
correct_class_positions = np.nonzero(is_target_class)
print("POSITION_OF_CORRECT_POINTS", correct_points_positions)
print("SUM_OF_CLASSIFIED_POINTS", np.sum(prediction_classes))
print("POSITION_OF_CORRECT_POINTS_CLASS_14", correct_class_positions)
print("NUMBER_OF_POINTS_IN_CLASS_14", np.sum(correct_class_positions))

# Extract and save points from the specified class
chosen_points_from_class = data[correct_points_positions]
np.savetxt('TEST_DATA_POINT_POSITION_TEST_DATA_CLASS.txt', chosen_points_from_class, delimiter=',')
test_model_unscaled = test_model_data[correct_points_positions]
print(test_model_unscaled)
print(test_model_unscaled.shape)
np.savetxt('TEST_MODEL_NON_SCALE.txt', test_model_unscaled, delimiter=',')




