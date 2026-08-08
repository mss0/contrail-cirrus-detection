import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import shutil
import numpy as np
import keras
import os

from PIL import Image
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Folders named after the class their files belong in are found in /training_data
# In our case /training_data/cirrus and /training_data/not cirrus

data_dir = 'training_data'

batch_size = 8
img_width = 684
img_height = 513 


# Dataset functions


# Load the photos and their respective labels, as inferred from the directory names
def load_data(data_dir):

    label_names = []
    data = []
    labels = []

    for label, dir_name in enumerate(os.listdir(data_dir)):

        label_names.append(dir_name)

        curr_path = os.path.join(data_dir, dir_name)
        for image in os.listdir(curr_path):

            labels.append(label)
            
            img = Image.open(os.path.join(curr_path, image))
            data.append(np.asarray(img.resize((img_width, img_height), Image.Resampling.LANCZOS)).astype('float32'))

    label_no = len(label_names)

    print('Finished loading data')

    return data, labels, label_no, label_names


AUTOTUNE = tf.data.AUTOTUNE


def configure_for_performance(ds):

    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)

    return ds



data, labels, no_classes, class_names = load_data(data_dir)

data = np.asarray(data).astype('float32')
labels = np.asarray(labels).astype('int')

# The dataset was severely imbalanced initially, leading the model to put all photos in the majority class
# More photos were selected to correct that imbalance
# (all images were taken from https://www.flickr.com/photos/raspberrypi/albums)


# Create the model


# The model follows a typical CNN structure and has been limited to only just a few layers due to overfitting
# Among the solutions to that problem are data augmentation and dropout layers, the latter of which is implemented below
# We tried adding more layers, in a manner inspired by the VGG16 model,
# However we found most of the improvement from changing optimizers and their parameters

def create_model(no_classes):

    model = Sequential([

        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(no_classes, name="outputs")

    ])

    optimizer = keras.optimizers.SGD(learning_rate = 0.0001, weight_decay = 0.1)
    
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model


# K Fold cross validation 

# Does not return anything, as this is a method solely used for model validation
# It involves training in multiple folds, each of which uses a diferent portion of the data for training and validation
# It helps with telling how well the model is able to generalize


no_splits = 10


# Fucntion that generates plots used in diagnosing the model
def visualization(history, epochs):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def run_fold(index, no_classes, training_indices, validation_indices, accuracies, losses):

    print(f'Fold {index + 1} out of {no_splits}')

    training_dataset = tf.data.Dataset.from_tensor_slices((data[training_indices], labels[training_indices]))
    validation_dataset = tf.data.Dataset.from_tensor_slices((data[validation_indices], labels[validation_indices]))

    training_dataset = configure_for_performance(training_dataset)
    validation_dataset = configure_for_performance(validation_dataset)

    print('Loaded training and testing datasets')

    model = None
    model = create_model(no_classes)

    print('Traning start')

    epochs = 50

    history = model.fit(

        training_dataset,
        validation_data = validation_dataset,
        batch_size = batch_size,
        epochs = epochs
    )

    accuracies[index] = history.history['val_accuracy'][-1]
    losses[index] = history.history['val_loss'][-1]

    visualization(history, epochs)


# The folds are run in separate processes due to memory issues
# Function collects final accuracies and losses and displays them
def run_kfold(): 

    # Stratified ensures class distribution in selection matches class distribution in dataset
    cross_validator = StratifiedKFold(n_splits = no_splits, shuffle = True)
    
    accuracies = multiprocessing.Array('d', no_splits)
    losses = multiprocessing.Array('d', no_splits)

    for index, (training_indices, validation_indices) in enumerate(cross_validator.split(data, labels)):

        process = multiprocessing.Process(target = run_fold, args = (index, no_classes, training_indices, validation_indices, accuracies, losses))
        process.start()
        process.join()

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(accuracies)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {losses[i]} - Accuracy: {accuracies[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(accuracies)} (+- {np.std(accuracies)})')
    print(f'> Loss: {np.mean(losses)}')
    print('------------------------------------------------------------------------')


# If satisfied with performance in KFold, train and save the model


saved_model = 'saved_model/my_model'


def train():

    training_data, validation_data, training_labels, validation_labels = train_test_split(data, labels, test_size = 0.2, stratify=labels)

    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))

    training_dataset = configure_for_performance(training_dataset)
    validation_dataset = configure_for_performance(validation_dataset)

    print('Loaded training and validation datasets')

    model = create_model(no_classes)

    print('Training start')

    epochs = 65

    history = model.fit(

        training_dataset,
        validation_data = validation_dataset,
        epochs = epochs
    )

    visualization(history, epochs)

    model.save(saved_model)  


# Load model and classify data


def classify(model_path):

    model = tf.keras.models.load_model(model_path)

    # Images to classify go in to_classify folder

    samples_to_predict = []
    samples_path = 'to_classify'
    end_path ='classified'

    for image in os.listdir(samples_path):

        img = Image.open(os.path.join(samples_path, image))
        samples_to_predict.append(np.asarray(img.resize((img_width, img_height))).astype('float32')) 

    samples_to_predict = np.asarray(samples_to_predict).astype('float32')

    predictions = model.predict(samples_to_predict)

    for index, image in enumerate(os.listdir(samples_path)):

        score = tf.nn.softmax(predictions[index])

        # Move photos from to_classify in the folder corresponding to the predicted label
        
        if class_names[np.argmax(score)] == 'cirrus':
            shutil.move(os.path.join(samples_path, image), os.path.join(end_path, 'with_cirrus', image))
        else:
            shutil.move(os.path.join(samples_path, image), os.path.join(end_path, 'without_cirrus', image))


# run_kfold()
train()
classify(saved_model)
