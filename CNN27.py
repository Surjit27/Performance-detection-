from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

train_data_dir = '/home/surjit/Documents/lingu/Potato/Train'
validation_data_dir = '/home/surjit/Documents/lingu/Potato/Valid'
test_data_dir = '/home/surjit/Documents/lingu/Potato/Test'

img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)  # Grayscale images have 1 channel

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='rgb',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='rgb',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

test_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    color_mode='rgb',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # Don't shuffle test data

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))  # 3 classes for your case
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator,
                                          steps=test_generator.samples // test_generator.batch_size)
print("Test Accuracy:", test_accuracy)

model.save('dam_file_1.h5')
