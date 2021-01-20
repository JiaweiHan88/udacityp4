import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def generate_training_data(image_paths, measurements, batch_size=32):
    image_paths, measurements = shuffle(image_paths, measurements)
    X,y = ([],[])
    while True:
        for i in range(len(measurements)):
            img = cv2.imread(image_paths[i])
            measurement = measurements[i]
            img = cv2.cvtColor(cv2.resize(img[70:135, :, :], (200, 66), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2YUV)
            X.append(img)
            y.append(measurement)
            img = np.fliplr(img)
            measurement *= -1
            X.append(img)
            y.append(measurement)
            if len(X) >= batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])



lines = []
with open('data_custom/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

image_paths = []
images = []
measurements = []
lines.pop(0)
for line in lines:
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.3  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # add images and angles to data set
    image_paths.append(line[0])
    image_paths.append(line[1])
    image_paths.append(line[2])
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    #image = cv2.imread(current_path)
    #crop_img = cv2.cvtColor(cv2.resize(image[70:135, :, :], (200, 66), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2YUV)
    #image_flipped = np.fliplr(crop_img)
    #measurement = float(line[3])
    #measurement_flipped = -measurement
    #images.append(crop_img)
    #measurements.append(measurement)
    #images.append(image_flipped)
    #measurements.append(measurement_flipped)



image_paths = np.array(image_paths)
measurements = np.array(measurements)
print('Before:', image_paths.shape, measurements.shape)
num_bins = 31
avg_samples_per_bin = len(measurements)/num_bins
hist, bins = np.histogram(measurements, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

keep_probs = []
target = 2500
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        #keep_probs.append(1.)
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(measurements)):
    for j in range(num_bins):
        if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
measurements = np.delete(measurements, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(measurements, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()
print('After:', image_paths.shape, measurements.shape)
if True:
    X_train = np.array(images)
    y_train = np.array(measurements)
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Conv2D, Cropping2D, Lambda, Dropout
    from keras.optimizers import Adam
    from keras.regularizers import l2

    image_paths, measurements = shuffle(image_paths, measurements)

    image_paths_train, image_paths_valid, angles_train, angles_valid = train_test_split(image_paths, measurements, test_size=0.2, random_state=1)
    train_gen = generate_training_data(image_paths_train, angles_train, batch_size=64)
    val_gen = generate_training_data(image_paths_valid, angles_valid, batch_size=64)

    reg = l2(0.001)
    model = Sequential()
    model.add(Lambda(lambda x: (x - 128.0) / 128.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer =reg))
    model.add(Dropout(0.25))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer =reg))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer =reg))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer =reg))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer =reg))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(100, kernel_regularizer =reg))
    #model.add(Dropout(0.25))
    model.add(Dense(50, kernel_regularizer =reg))
    #model.add(Dropout(0.25))
    model.add(Dense(10, kernel_regularizer =reg))
    model.add(Dense(1, kernel_regularizer =reg))
    opt = Adam(learning_rate=0.0005)
    model.compile(loss='mse', optimizer=opt)
    history = model.fit(train_gen, validation_data=val_gen, epochs=5, steps_per_epoch=10000, validation_steps=len(angles_valid) * 2 / 64)
    model.save('model.h5')
    print(model.summary())
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)