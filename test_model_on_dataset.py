import os
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("tic_tac_toe_cnn.h5")

class_names = ["O", "X", "empty"]

test_dir = "test_processed"

correct = 0
total = 0

for true_label in os.listdir(test_dir):
    class_path = os.path.join(test_dir, true_label)

    if not os.path.isdir(class_path):
        continue

    for fname in os.listdir(class_path):
        img_path = os.path.join(class_path, fname)

        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = class_names[predicted_class]

        total += 1
        if predicted_label == true_label:
            correct += 1
        else:
            print(f"❌ Wrong: {fname} | True: {true_label} | Predicted: {predicted_label}")

accuracy = correct / total * 100
print("\n Test Accuracy on processed dataset:", accuracy, "%")