import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

modelo = tf.keras.models.load_model('models/modelo_classificador.h5')
print(modelo.summary())

# carregar modelo
modelo = tf.keras.models.load_model('models/modelo_classificador.h5')

# carregar dataset
df = pd.read_csv('data/processed/treino_rotulado.csv')

X = df["texto"].values
y = df["classe"].values

# recriar encoder
encoder = LabelEncoder()
encoder.fit(y)

# recriar vetorização
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=30
)
vectorizer.adapt(X)

# texto de teste
texto = ["Abertura de processo licitatório para contratação de empresa especializada em saúde pública."]

texto_vetorizado = vectorizer(texto)

pred = modelo.predict(texto_vetorizado)

classe_predita = encoder.inverse_transform([pred.argmax()])

print("Classe prevista:", classe_predita[0])