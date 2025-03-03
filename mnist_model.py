import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar os valores dos pixels (de 0-255 para 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Criar o modelo de rede neural
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  
    keras.layers.Dense(256, activation='relu'),  
    keras.layers.Dropout(0.2),  # Evita overfitting
    keras.layers.Dense(128, activation='relu'),  
    keras.layers.Dense(10, activation='softmax')  
])


# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
print("Treinando a IA...")
model.fit(x_train, y_train, epochs=5)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc:.4f}")

# Fazer previsões no conjunto de teste
predictions = model.predict(x_test)

# Função para exibir uma previsão da IA
def mostrar_previsao(index):
    predicted_label = np.argmax(predictions[index])  # Pega a classe com maior probabilidade
    plt.imshow(x_test[index], cmap='gray')  # Exibe a imagem
    plt.title(f"IA Prediz: {predicted_label}")  # Mostra a previsão da IA
    plt.show()

# Testar uma imagem específica (mude o número para testar outra)
index = 10  # Escolha um número entre 0 e 9999
mostrar_previsao(index)

