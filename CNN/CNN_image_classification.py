## Using Keras to ceate from the ground up a Image classification model
## The model recognizes planes from cars.
## Several tests and evaluations to experiment with the omdel architecture, training data...
## The dataset is not mine, which means it is not included on my GitHub page.

## This code was originally a notebook, and some parts of code might not work due to retrocompatibility issues with Keras.

from keras.preprocessing.image import load_img
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D ,Flatten
from keras import Input
from keras.utils import plot_model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory

import matplotlib.pyplot as plt
import numpy as np


import graphviz
import pydot

img_width = 224
img_height = img_width

image = load_img("v_data/train/planes/73.jpg")
img = np.array(image) / 255.0
img = img.reshape(1,img_width, img_height,3)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height,img_width)    
else:
    input_shape = (img_height,img_width,3)



## Model architecture

model = Sequential()
model.add(Input(shape=input_shape)) #couche d'entrée (input)

# partie "features"

model.add(Conv2D(32, (3, 3)))#, input_shape=(100, 150, 3)))   # 100 lignes et 150 colonnes (notation matricielle)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# à ce stade, on sort des caractéristiques 3D 

# partie "classifier"

model.add(Flatten())  #  Ceci transforme les caractéristique 3D en une "colonne" de neurones d'entrée, comme dans les MLP classiques
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))  
model.add(Activation('sigmoid'))

# on précise à présent le loss à optimiser
# ainsi qu'une métrique à afficher (score de classification) et un optimiseur (rmsprop adapte le taux d'apprentissage automatiquement)

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
#model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')



# résumé du modèle précédemment défini
# affichage graphique (si graphviz et pydot correctement installés)
plot_model(model,  show_shapes=True)

## Compilation and training

## Validation data
"""X_val = image_dataset_from_directory('v_data/test', image_size=(224, 224))
#print(X_val)
validation_data = X_val.map(lambda X,y: (X/255, y))"""

rescale_generator= ImageDataGenerator(rescale=1./255)
validation_data = rescale_generator.flow_from_directory(
    'v_data/test/',
    target_size=(img_width, img_height),
    class_mode='binary'
)


## Training data
train_aug = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range = [1, 1.5],
)

#generate the augmented training data with the train_aug data generator
augmented_train_data = train_aug.flow_from_directory(
    'v_data/train/',
    target_size=(img_width, img_height),
    batch_size = 10,
    class_mode='binary'
)


history = model.fit(
    augmented_train_data,
    epochs=15,
    validation_data=(validation_data)
)


model.save('model', overwrite=True)


## Evaluation

#fonction pour afficher un certain nombre d'images. Inspiré du cours d'apprentissage auto. Encore un peu expérimental.

def affichage_images(verbose=False):
    plt.figure(figsize=[10,12])   
    for n in range(1,21): #cars: 20 first images from the test dataset

        image=load_img(f'v_data/test/cars/{n}.jpg')
        img = np.array(image) / 255.0

        plt.subplot(10,10,n+1,xticks=[],yticks=[])
        plt.imshow(img,cmap='gray_r')


        img = img.reshape((1, 224, 224, 3))
        predictions = model.predict(img,verbose=verbose)
        #score = float(keras.ops.sigmoid(predictions[0][0]))
        #print(f"This image is {100 * (1 - score):.2f}% car and {100 * score:.2f}% plane.")
        
        if np.round(predictions)==0:
            plt.text(0.1,0.1,str(0)+' / '+str(0),fontsize=6,bbox=dict(facecolor='white', alpha=1))    
        else:
            plt.text(0.1,0.1,str(1)+' / '+str(0),fontsize=6,bbox=dict(facecolor='red', alpha=1))    
    for n in range(1,21): #planes: 20 first images from the test dataset

        image=load_img(f'v_data/test/planes/{n}.jpg')
        img = np.array(image) / 255.0

        plt.subplot(10,10,n+20+1,xticks=[],yticks=[])
        plt.imshow(img,cmap='gray_r')

        img = img.reshape((1, 224, 224, 3))
        predictions = model.predict(img,verbose=verbose)
        #score = float(keras.ops.sigmoid(predictions[0][0]))
        #print(f"This image is {100 * (1 - score):.2f}% car and {100 * score:.2f}% plane.")
        
        if np.round(predictions)==1:
            plt.text(0.1,0.1,str(1)+' / '+str(1),fontsize=6,bbox=dict(facecolor='white', alpha=1))    
        else:
            plt.text(0.1,0.1,str(0)+' / '+str(1),fontsize=6,bbox=dict(facecolor='red', alpha=1))    
    plt.suptitle('classe prédite / classe réelle')
    plt.show();


affichage_images()


#plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## Predictions 

#1
image=load_img('v_data/test/cars/16.jpg')
img = np.array(image) / 255.0
size_images=(img_width,img_height)

plt.imshow(img)
img = img.reshape((1, 224, 224, 3))
print(f'prediction: {np.round(model.predict(img))}')

#2
image=load_img('v_data/test/planes/16.jpg')
img = np.array(image) / 255.0
size_images=(img_width,img_height)

plt.imshow(img)
img = img.reshape((1, 224, 224, 3))
print(f'prediction: {np.round(model.predict(img))}')







## Testing if augmenting training data helps accuracy:

nb_iter = 11 #nombre de modèles qui seront entraînés pour le test

#scaler pour le dataset de test (base de test)
rescale_generator= ImageDataGenerator(
    rescale=1./255)

test_dataset = rescale_generator.flow_from_directory(
    'v_data/test',
    target_size=size_images,
    color_mode='rgb',
    classes=['cars','planes'],
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=1234,
    subset='training',
    interpolation='nearest',
    keep_aspect_ratio=False
)

#puis on itère sur le nombre d'images dans la base de test (ne correspond pas à i mais c'est l'idée)
accuracy=[] #liste contenant les accuracies de training
val_accuracy=[] #liste contenant les accuracies de validation
for i in range(nb_iter): #nb_iter = nombre de modèles qui seront entraînés
    rescale_generator= ImageDataGenerator(
    rescale=1./255,
    validation_split=0.5 - i*0.05, #à chaque itération, on réduit le validation_split en faveur de la base de test.
    )

    training_dataset = rescale_generator.flow_from_directory(
    'v_data/train',
    target_size=size_images,
    color_mode='rgb',
    classes=['cars','planes'],
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=1234,
    subset='training',
    interpolation='nearest',
    keep_aspect_ratio=False
)

    history = model.fit(
        training_dataset,
        epochs=10,
        validation_data=test_dataset
    )
    accuracy.append(history.history['accuracy'][-1])
    val_accuracy.append(history.history['val_accuracy'][-1])

#affichages graphiques
plt.plot(np.arange(200,420,20), accuracy)
plt.plot(np.arange(200,420,20), val_accuracy)
plt.title("accuracy du modèle en fonction du nombre d'images utilisé pour l'entraînement.")
plt.ylabel('accuracy')
plt.xlabel("nombre d'images utilisé pour l'entraînement")
plt.legend(['train', 'test'], loc='upper left')
plt.show()


