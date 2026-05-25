
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_csv("../data/processed/treino_rotulado.csv")

X = df["texto"].values
y = df["classe"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

vectorizer = tf.keras.layers.TextVectorization(max_tokens=5000, output_sequence_length=30)
vectorizer.adapt(X)

X_vec = vectorizer(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec.numpy(), y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(set(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("../models/modelo_classificador.h5")
