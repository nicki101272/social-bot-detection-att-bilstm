# Social Bot Detection using Att-BiLSTM and GAN

This project is part of my master's thesis research and explores contextual social bot detection using deep learning methods.

## 🔍 Summary

- Uses an attention-based BiLSTM model to classify social bots and humans based on tweet content.
- Compares performance with BERT-based models.
- Applies Seq-GAN for data augmentation to address class imbalance.
- Evaluated on the Cresci dataset (2017).

## 📁 Contents

- `main_model.py`: Implementation of Att-BiLSTM
- `seq_gan.py`: GAN-based data augmentation module
- `augment_and_train.py`: Integration of GAN-generated data into Att-BiLSTM training
- `preprocessing/`: Scripts for text preprocessing
- `results/`: Output metrics and performance plots
- `report.pdf`: Final manuscript

## 📊 Dataset

We used the publicly available [Cresci Bot Dataset (2017)](https://github.com/dfreelon/cresci-2017).

## 🧠 Model Highlights

- Attention layer improves interpretability
- Seq-GAN significantly enhances classifier performance
- Outperforms classical ML baselines and matches BERT in certain settings

## 📄 Paper

You can read the full paper [here](./report.pdf)

## 📫 Contact

**Nicki Sadeghi**  
nicki.sadeqi@gmail.com

---

## 📦 `main_model.py`: Att-BiLSTM Implementation (Keras)

```python
# ... [unchanged: Att-BiLSTM code] ...
```

---

## 📦 `seq_gan.py`: Seq-GAN for Data Augmentation (simplified prototype)

```python
# ... [unchanged: Generator/Discriminator code] ...
```

---

## 🔁 `augment_and_train.py`: Combine GAN-generated data with real data for model training

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from main_model import model
from seq_gan import generator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

X_real = np.load('preprocessed/X_real.npy', allow_pickle=True)
y_real = np.load('preprocessed/y_real.npy', allow_pickle=True)

def generate_synthetic_data(generator, num_samples=1000, sequence_length=20):
    noise = tf.random.uniform((num_samples, sequence_length), minval=0, maxval=5000, dtype=tf.int32)
    generated = generator(noise)
    sampled = tf.argmax(generated, axis=-1)
    return sampled.numpy()

X_fake = generate_synthetic_data(generator, num_samples=1000)
y_fake = np.ones((1000,))

X_combined = np.concatenate([X_real, X_fake], axis=0)
y_combined = np.concatenate([y_real, y_fake], axis=0)

X_combined = pad_sequences(X_combined, maxlen=100)

history = model.fit(X_combined, y_combined, batch_size=64, epochs=10, validation_split=0.2)

y_pred = model.predict(X_combined)
y_pred_labels = (y_pred > 0.5).astype(int)
print(classification_report(y_combined, y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_combined, y_pred_labels))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history.get('accuracy', []), label='Train Acc')
plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history.get('loss', []), label='Train Loss')
plt.plot(history.history.get('val_loss', []), label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/training_curves.png")
plt.show()
```

> This script shows how to generate fake samples using the trained GAN, integrate them with real data, train the Att-BiLSTM model, and visualize performance metrics.
