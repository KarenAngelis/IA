# IA
Python

### **📌 Resumo da IA que Estamos Criando**  

Estamos desenvolvendo uma **Inteligência Artificial (IA) avançada para reconhecimento de números escritos à mão**, baseada no conjunto de dados **MNIST**.  

---

## **🔥 O que essa IA faz?**
✅ **Treina uma Rede Neural Convolucional (CNN)** para reconhecer números (0 a 9).  
✅ **Salva o modelo treinado** para evitar re-treinamento desnecessário.  
✅ **Testa previsões** com imagens do conjunto de teste MNIST.  
✅ **Permite carregar imagens desenhadas pelo usuário** e fazer previsões.  

---

## **🔍 Como funciona?**
1. **Carrega o conjunto de dados MNIST** (60.000 imagens para treino, 10.000 para teste).  
2. **Normaliza as imagens** (convertendo os valores de pixels para o intervalo 0-1).  
3. **Cria um modelo CNN** (com camadas convolucionais para melhor desempenho).  
4. **Treina o modelo** usando o otimizador **Adam** e função de perda **Sparse Categorical Crossentropy**.  
5. **Salva o modelo** para evitar treinamentos repetidos.  
6. **Avalia a IA** no conjunto de teste e exibe a precisão.  
7. **Faz previsões automáticas** mostrando imagens e os números que a IA reconhece.  
8. **Permite testar imagens desenhadas pelo usuário** convertendo-as para o formato correto.  

---

## **📊 Resultados esperados**
🔹 **Precisão esperada acima de 97%** no conjunto de teste.  
🔹 **Rede CNN mais eficiente** que uma rede densa simples.  
🔹 **Capacidade de reconhecer números desenhados pelo usuário**.  

---

## **🚀 Próximos Passos**
🔹 Criar uma **interface gráfica** para desenhar números em tempo real.  
🔹 Melhorar o modelo **com mais camadas convolucionais** para maior precisão.  
🔹 Usar **redes neurais mais avançadas**, como EfficientNet ou Vision Transformers.  

---

### **Quer seguir para a interface gráfica ou aprimorar ainda mais a precisão? 😃**