
"""
Title: Digit Recognizer (Kaggle Competition)
Created on Thu Jan 23 16:04:27 2020

@author: Saeed Mohajeryami, PhD
"""
## Basic Libraries
import numpy as np
import pandas as pd

## Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

## Keras Libraries
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing import image  # for ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

## Model evaluation tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools

np.random.seed(123)


## Import the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.iloc[:,1:]
y_train = train.iloc[:,0:1]

## describe the training set
g = sns.countplot(y_train)
y_train.value_counts()
## the plot shows that we have a balanced dataset.

## Check for missing values
X_train.isnull().any().describe()
y_train.isnull().any().describe()
## There is no missing values in the train and test dataset. 

## Normalization
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
test = scaler.transform(test)

# Reshape image in 3 dimensions (height = 28, width = 28 , channels = 1)
X_train = X_train.reshape(X_train.shape[0],28,28,1)   #or   X_train.reshape(-1,28,28,1)
test = test.reshape(test.shape[0],28,28,1)            #or   test.reshape(-1,28,28,1)

# Note: Keras requires an extra dimension in the end which correspond to
# channels. MNIST images are gray scaled so it use only one channel. For RGB
# images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3
# 3D matrices.

## label enconding
y_train = to_categorical(y_train, num_classes = 10)

## split training/validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

## Note: Due to balanced data, random sampling works. If not balanced, then use
## use stratify = True option

## show some example
g = plt.imshow(X_train[3][:,:,0])





# ----------------------Convolutional Neural Network-----------------------
## Initializing the CNN
classifier = Sequential()

## Step 1 - Convolution
## 32 is the number of feature detectors
## (3,3) is the dimension of feature detectors
classifier.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same',
                      input_shape = (28,28,1), activation='relu'))
## Step 2 - MaxPooling
#classifier.add(MaxPool2D(pool_size=(2,2)))

## Step 3 - Dropout
#classifier.add(Dropout(0.2))

## Adding a second convolution layer
## note: ReLU function is for breaking the linearity.
classifier.add(Conv2D(32,(3,3), padding='Same', activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

## Adding a third convolution layer
classifier.add(Conv2D(64,(3,3), padding='Same', activation='relu'))
#classifier.add(MaxPool2D(pool_size=(2,2)))

## Adding a fourth convolution layer
classifier.add(Conv2D(64,(3,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
classifier.add(Dropout(0.2))

## step 4 - Flatten
## This layer converts the final feature maps into a one single 1D vector. 
## It combines all the (found) local features of the previous convolutional
## layers.
classifier.add(Flatten())

## step 5 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

## compiling the CNN 
classifier.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

# ----------------------------------------------------------------------

# Set a learning rate annealer
# The LR is the step by which the optimizer adjust the weights. The higher LR
# lead to more radical actions and quicker convergence. However, the faster speed
# could probably lead the algorithm into a local minima.
# The better way is to start with a high LR and decrease it during the training
# to reach efficiently the global minimum of the loss function. It is possible
# the LR dynamically every X steps (epochs) if necessary (when accuracy is not
# improved). Below, I reduce LR by half if the accuracy is not improved after 3
# epochs.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

# Without image augmentation and without learning rate reduction
history = classifier.fit(X_train, y_train, batch_size = batch_size,
                         epochs = epochs, validation_data = (X_val, y_val),
                         verbose = 2)
# accuracy: 0.9817

## Evaluate the model (Training and validation curves)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

### Image Augmentation:
# Approaches that alter the training data in ways that change the array
# representation while keeping the label the same are known as data
# augmentation techniques. Some popular augmentations people use are
# grayscales, horizontal flips, vertical flips, random crops, color jitters,
# translations, rotations, and much more.

# By applying just a couple of these transformations to our training data, 
# we can easily double or triple the number of training examples and create a 
# very robust model.


datagen = image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images (by 10%) in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image (by 10%)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images - set to false to avoid misclassify 6 and 9
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

# Fit the model
history = classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# Note: why do we use classifier.fit_generator instead of classifier.fit?
# fit is used when the entire training dataset can fit into the memory and no
# data augmentation is applied. . fit_generator is used when either we have a
# huge dataset to fit into our memory or when data augmentation needs to be
# applied.



# Plotting the confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
y_pred = classifier.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# Display some error results 
# Errors are difference between predicted labels and true labels
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)




# ----------------------------------Results and Submission ------------------
# predict results
results = classifier.predict(test)

# select the indices with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
