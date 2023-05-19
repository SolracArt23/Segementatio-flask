from flask import Flask,render_template,request
import os
import keras as kr
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# importar flask
app = Flask(__name__)
# Subir imagenes
app.config["UPLOAD_FOLDER"] ='./images'

# funciones ocultas
def Prediccion(model,img):
    
    #Cargar el modelo
    try:    
        Modelo=kr.models.load_model(f'./Modelos/{model}')
    except:
            
        Modelo=kr.models.load_model(f'./Modelos/{model}',compile=False)
        Modelo.compile(optimizer='adam',loss='binary_crossentropy')

    # Tipo de entrada
    input_shape = Modelo.layers[0].input.shape

    # Cargar imagen y redimencionarla
    imagen = Image.open(f"./static/images/{img}")
    imagen = imagen.resize(input_shape[1:-1]) 
    numpy_img = np.array(imagen)
    #procesar Salida
    y_predict=Modelo.predict(np.array([numpy_img]))
    if y_predict.shape[-1] == 1:
        y_predict=y_predict.reshape(y_predict.shape[1:-1])

    #Guardar resultado
    print(y_predict.shape)
    result_ruta = "result/result.jpg"
    plt.imsave(f"static/{result_ruta}",y_predict)
    
    return result_ruta


def Cargar_modelos():
    result =[]
    for _,_,modelos in os.walk('./Modelos'):
        for modelo in modelos:
            result += [modelo]
    return result

@app.route('/',methods=['GET','POST'])
def main ():
    # varible
    img_o =None
    img_pre = None
    # En caso de que ya se subio la imagen
    if 'imagen' in request.files:
        image = request.files["imagen"]
        image.save('static/images/'+image.filename)
        nombre_imagen = image.filename
        print(nombre_imagen)
        print(request.form.get('opciones'))
        # Predecir
        img_o = f"images/{image.filename}"
        img_pre=Prediccion(request.form.get('opciones'), nombre_imagen)


    # Cargar los modelos
    lista_modelos=Cargar_modelos()
    return render_template('index.html',lista=lista_modelos,img_origin=img_o,img_pre=img_pre)

# Comenzar
if __name__ == '__main__':
    app.run('0.0.0.0',port=5000,debug=True)