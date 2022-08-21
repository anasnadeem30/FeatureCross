import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import feature_column
from tensorflow import keras
from keras import layers

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

train_df=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

scale_factor=1000.0
train_df['median_house_value']/=scale_factor
test_df['median_house_value']/=scale_factor

train_df= train_df.reindex(np.random.permutation(train_df.index))

feature_columns=[]

latitude=tf.feature_column.numeric_column("latitude")
feature_columns.append(latitude)

longitude=tf.feature_column.numeric_column("longitude")
feature_columns.append(longitude)

fp_feature_layer=tf.keras.layers.DenseFeatures(feature_columns)

def create_model(my_learning_rate, feature_layer):
    model= tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def train_model(model, dataset, epochs, batch_size, label_name):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True)

  # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
  
  # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse   


def plot_the_loss_curve(epochs, rmse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])
  plt.show()  

# learning_rate = 0.05
# epochs = 30
batch_size = 100
label_name = 'median_house_value'
# my_model = create_model(learning_rate, fp_feature_layer)
# epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

# plot_the_loss_curve(epochs, rmse)
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
# my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

resolution_in_degrees=0.4
feature_columns=[]
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), 
                                     int(max(train_df['latitude'])), 
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, 
                                               latitude_boundaries)
feature_columns.append(latitude)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), 
                                      int(max(train_df['longitude'])), 
                                      resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, 
                                                longitude_boundaries)
feature_columns.append(longitude)

# Convert the list of feature columns into a layer that will ultimately become
# part of the model. Understanding layers is not important right now.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will later be fed into
# the model. 
feature_cross_feature_layer = keras.layers.DenseFeatures(feature_columns)
learning_rate = 0.04
epochs = 35

# Build the model, this time passing in the feature_cross_feature_layer: 
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)