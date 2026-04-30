import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model = tf.keras.models.load_model('./models/mnist_cnn.keras')

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_processed = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

loss, accuracy = model.evaluate(x_test_processed, y_test, verbose=0)
print(f"Test Loss    : {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = model.predict(x_test_processed)
y_pred_classes = np.argmax(y_pred, axis=1)

#print(classification_report(y_test, y_pred_classes))

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()