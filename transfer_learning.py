import tensorflow as tf
from tf.keras import layers
import matplotlib.pyplot as plt

# Set the path to your dataset
dataset_path = '/amazon_forest_dataset'

# Parameters
batch_size = 16  # Consider reducing the batch size if memory limits are exceeded
img_height = 512
img_width = 512

# Load the training images
train_image_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + '/images',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode=None,  # Since we'll be using a separate mask as labels
    color_mode='rgb'
)

# Load the training masks
train_mask_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + '/masks',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode=None,
    color_mode='grayscale'  # Masks are typically grayscale
)

# Preprocess input images according to ResNet50 requirements
train_image_ds = train_image_ds.map(lambda x: tf.keras.applications.resnet50.preprocess_input(x))

# Combine images and masks into a single dataset
train_ds = tf.data.Dataset.zip((train_image_ds, train_mask_ds))

# Load the ResNet50 model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)

# Freeze the base model
base_model.trainable = False

# Create a new model on top
model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Increase the layer size for better feature learning
    layers.Dropout(0.5),  # Adding Dropout for regularization
    layers.Dense(1, activation='sigmoid')  # Output layer for binary segmentation
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    epochs=10,
    steps_per_epoch=len(train_ds)
)

# Plot training loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()