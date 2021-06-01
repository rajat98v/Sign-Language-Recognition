import cv2 
import numpy as np 
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


# TRAINING_DIR = "/home/rajatv/Downloads/Machine/archives/Train/"
# image_generator = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
#
# train_generator = image_generator.flow_from_directory(
#         TRAINING_DIR,
#         target_size=(100,100),
#         class_mode='categorical',
#         batch_size=32
# )
# label_map = (train_generator.class_indices)
#
# validation_generator = image_generator.flow_from_directory(
#         TRAINING_DIR,
#         target_size=(100,100),
#         class_mode='categorical',
#   batch_size=32
# )
# def md():
#     import tensorflow as tf
#
#     model = tf.keras.models.Sequential([
#         # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(100, 100, 3)),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2), 
#         # tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
#         # tf.keras.layers.MaxPooling2D(2,2),
#         # tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
#         # tf.keras.layers.MaxPooling2D(2,2),
#         # Flatten the results to feed into a DNN
#         tf.keras.layers.Flatten(), 
#         # 512 neuron hidden layer
#         tf.keras.layers.Dense(514, activation='relu'), 
#         tf.keras.layers.Dense(214, activation='relu'), 
#         # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
#         tf.keras.layers.Dense(29, activation='softmax')  
#     ])
#
#     model.summary()
#
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     history = model.fit(train_generator,
#                                   validation_data=validation_generator,
#                                   steps_per_epoch=1700,
#                                   epochs=5,
#                               validation_steps=50,
#                               verbose=1)
#
#     model.save("mymodel2")
#
# md()

model = keras.models.load_model("mymodel2")

stream=cv2.VideoCapture(0)

def get_key(val):
    for key, value in label_map.items():
         if val == value:
             return key
 
    return "key doesn't exist"
 

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=1)
  # resize the image to the desired size
  return tf.image.resize(img, [28, 28])


backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv2.createBackgroundSubtractorKNN()
while True:
    ret, frame=stream.read()

    cv2.rectangle(frame, (100, 100), (500, 500), (0, 255,0),6)
    frame2=cv2.flip(frame, 1)
    cv2.imshow('video', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    hand_frame = frame[100:500, 100:500]
    fgMask = backSub.apply(hand_frame)

    cv2.imshow('Video', hand_frame)
    cv2.imshow('fg mask', fgMask)

    # image_np = np.array(frame)
    # img = decode_img(image_np)
    inp = cv2.resize(hand_frame, (100 , 100 ))
    # asdf = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
    asdf = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(asdf, dtype=tf.float32)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)



    classes = model.predict(rgb_tensor, steps=1)
    # print(classes)
    # print(classes[0])
    # print(np.argmax(classes[0]))
    print(chr(np.argmax(classes[0])+97))
    # print(get_key(np.argmax(classes[0])))


stream.release()
