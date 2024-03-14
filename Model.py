import numpy as np
import pandas as pd
import tensorflow as tf
import laspy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping

# Set numpy print options and matplotlib plot parameters for better readability
np.set_printoptions(suppress=True)
plt.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Function to load and prepare data from a .las file
def load_and_prepare_data(filename):
    data = laspy.read(filename)
    attributes = ['intensity', 'x', 'y', 'z', 'return_number', 'number_of_returns',
                  'scan_direction_flag', 'classification', 'red', 'green', 'blue']
    prepared_data = {}
    for attr in attributes:
        prepared_data[attr] = np.array(getattr(data, attr)).reshape(-1, 1)
    return prepared_data, list(data.point_format.dimension_names)

# Load training and additional data
train_data, train_heading = load_and_prepare_data("TRAIN_DATA.las")
train_data_2, train_heading_2 = load_and_prepare_data("vall_small.las")

# Combine data from both sources
for key in train_data:
    train_data[key] = np.vstack((train_data[key], train_data_2[key]))

# Choose class for model learning
target_class = 14
train_data_true = (train_data['classification'] == target_class).astype(int)
print(f"NUMBER_OF_POINTS in class {target_class}:", np.sum(train_data_true))

# Prepare data for model
features = np.column_stack((train_data['z'], train_data['intensity'], train_data['return_number'],
                            train_data['number_of_returns'], train_data['red'], train_data['green'], train_data['blue']))
labels = train_data_true

# Convert to DataFrame for easy manipulation
data_df = pd.DataFrame(features, columns=['z', 'intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue'])
data_df['target'] = labels

# Split data
train_df, test_df = train_test_split(data_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Prepare features and labels
train_labels = train_df.pop('target').values
val_labels = val_df.pop('target').values
test_labels = test_df.pop('target').values
train_features = train_df.values
val_features = val_df.values
test_features = test_df.values

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Define model metrics
metrics = [
    TruePositives(name='tp'), FalsePositives(name='fp'),
    TrueNegatives(name='tn'), FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'), Precision(name='precision'),
    Recall(name='recall'), AUC(name='auc'), AUC(name='prc', curve='PR')
]

# Define and compile the model
def make_model(metrics=metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(train_features.shape[-1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)
    return model

model = make_model()
model.summary()

# Train the model
EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max', restore_best_weights=True)
model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping],
          validation_data=(val_features, val_labels), verbose=1)

# Save the trained model
model.save('model_trained')

# Prediction and evaluation
train_predictions = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions = model.predict(test_features, batch_size=BATCH_SIZE)

# Further analysis and saving results can continue from here
