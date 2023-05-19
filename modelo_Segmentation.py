import cv2
import keras.models as kr
import tensorflow as tf
import numpy as np


model = kr.load_model("./Modelos/model_segmentatitn.h5",compile=False)
model.compile(optimizer='adam',loss='binary_crossentropy')
#Read the images
img=cv2.imread('assets\planeta.jpg')
image_resize = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)
# predict the image
prediction = model.predict(np.array([image_resize]))
print(prediction.shape)

# Mostrar la imagen en una ventana
cv2.imshow('Imagen', image_resize)
cv2.imshow('Mask', prediction[0])

# Esperar hasta que el usuario presione una tecla para salir
cv2.waitKey(0)

# Cerrar todas las ventanas
cv2.destroyAllWindows()
