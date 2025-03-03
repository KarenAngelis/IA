import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Para processar imagens desenhadas pelo usuário
import os

# 🔹 Verificar se temos uma GPU disponível para acelerar o treino
print("TensorFlow rodando em:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# 🔹 Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 🔹 Normalizar os valores dos pixels (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 🔹 Expandir dimensões para usar em uma CNN (28x28 → 28x28x1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 🔹 Definir o modelo mais avançado com camadas convolucionais (CNN)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Camada Conv2D
    keras.layers.MaxPooling2D((2, 2)),  # Redução de dimensionalidade
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 classes (0-9)
])

# 🔹 Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 🔹 Verificar se já existe um modelo treinado salvo
modelo_salvo = "mnist_modelo_avancado.h5"

if os.path.exists(modelo_salvo):
    print("\n🔹 Carregando modelo salvo...")
    model = keras.models.load_model(modelo_salvo)
else:
    print("\n🔹 Treinando o modelo...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    model.save(modelo_salvo)  # Salvar o modelo após o treinamento

# 🔹 Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Acurácia no conjunto de teste: {test_acc:.4f}")

# 🔹 Fazer previsões no conjunto de teste
predictions = model.predict(x_test)

# Função para exibir uma previsão da IA
def mostrar_previsao(index):
    predicted_label = np.argmax(predictions[index])
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"IA Prediz: {predicted_label}")
    plt.show()

# 🔹 Testar uma imagem aleatória do conjunto de teste
index = np.random.randint(0, len(x_test))
mostrar_previsao(index)

# 🔹 Função para testar uma imagem desenhada pelo usuário
def prever_imagem_desenhada(caminho_imagem):
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza
    imagem = cv2.resize(imagem, (28, 28))  # Redimensionar para 28x28
    imagem = 255 - imagem  # Inverter cores (se necessário)
    imagem = imagem / 255.0  # Normalizar
    imagem = np.expand_dims(imagem, axis=(0, -1))  # Ajustar formato para a IA

    predicao = model.predict(imagem)
    label_predito = np.argmax(predicao)

    plt.imshow(imagem.reshape(28, 28), cmap='gray')
    plt.title(f"IA Prediz: {label_predito}")
    plt.show()

# 🔹 Para testar uma imagem desenhada, coloque o caminho do arquivo abaixo:
# prever_imagem_desenhada("meu_desenho.png")  # Descomente essa linha e adicione o caminho da imagem

