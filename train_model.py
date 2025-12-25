import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================
# DATA DIRECTORIES
# =========================
train_dir = "train_processed"
valid_dir = "valid_processed"
test_dir  = "test_processed"

img_size = (128, 128)
batch_size = 32

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)
print("Class Indices:", train_gen.class_indices)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# =========================
# MODEL ARCHITECTURE
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(3, activation="softmax")  # X / O / EMPTY
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10
)

# =========================
# EVALUATE
# =========================
test_loss, test_acc = model.evaluate(test_gen)
print("\n✅ Test Accuracy:", test_acc)

# =========================
# SAVE MODEL
# =========================
model.save("tic_tac_toe_cnn.h5")
print("✅ Model saved as tic_tac_toe_cnn.h5")

# =========================
# ACCURACY
# =========================
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()