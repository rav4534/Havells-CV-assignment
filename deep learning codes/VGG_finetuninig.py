import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(720, 720, 3))

for layer in base_model.layers[:-5]: 
    layer.trainable = False

    
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x) #adding a fully connected layer
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)  #sigmoid as the output is binary classification


model = Model(inputs=base_model.input, outputs=x)


model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    '/Users/ravichandra/Documents/ML/havells/assignment/human-and-non-human/training_set/training_set/train_resize' ,
    target_size=(720, 720),
    batch_size=32,
    class_mode='binary'
)


model.fit(
    train_generator,
    epochs=10,  
    steps_per_epoch=len(train_generator),
)
