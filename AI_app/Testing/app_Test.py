import numpy as np
from keras.engine.saving import model_from_json

# cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# cargar json y crear el modelo
json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights("Model/model.h5")
print("Cargado modelo desde disco")

# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# Predict
print (loaded_model.predict(training_data).round())

print("=========== TEST ===========")
