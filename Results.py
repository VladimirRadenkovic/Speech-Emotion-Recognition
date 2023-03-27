import keras.models
import numpy as np
import pandas as pd
import sklearn.utils
from keras import layers
import pickle
from keras import models
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import itertools
import matplotlib.pyplot as plt
from model import earlystopping, learning_rate_reduction, f1_m
TrainData = pd.read_csv('TrainDataDownsampled.csv')
TestData = pd.read_csv('TestDataDownsampled.csv')
X_train = TrainData.drop(labels="labels", axis=1)
y_train = TrainData["labels"]
X_test = TestData.drop(labels="labels", axis=1)
y_test = TestData["labels"]
lb = LabelEncoder()
lb.fit(y_train)
y_train = np_utils.to_categorical(lb.transform(y_train))
y_test = np_utils.to_categorical(lb.transform(y_test))
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
#X_train,y_train = sklearn.utils.shuffle(X_train,y_train,random_state=0)
EPOCHS = 25
batch_size = 64
model = models.load_model('model.h5', custom_objects={'f1_m': f1_m})
print("Accuracy of our model on test data : " , model.evaluate(X_test,y_test)[1]*100 , "%")


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_check = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_check, y_pred=y_pred)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #plt.subplots(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix on test data')
plt.show()
with open('history.pkl', 'rb') as f:
   history = pickle.load(f)
fig , ax = plt.subplots(1,2)
train_acc = history['acc']
train_loss = history['loss']
test_acc = history['val_acc']
test_loss = history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(train_loss, label = 'Training Loss')
ax[0].plot(test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(train_acc, label = 'Training Accuracy')
ax[1].plot(test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

y_pred = model.predict(X_train)
y_pred = np.argmax(y_pred, axis=1)
y_check = np.argmax(y_train, axis=1)


cm = confusion_matrix(y_true=y_check, y_pred=y_pred)
cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix on training data')
plt.show()


