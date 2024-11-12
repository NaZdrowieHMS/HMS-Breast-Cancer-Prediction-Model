import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from constants import MODEL_PATH, BREAST_USG_MODEL_H5, TEST_DIRECTORY, TRAIN_DIRECTORY, VALIDATION_DIRECTORY

# Directories for training, validation, and test sets
train_dir = TRAIN_DIRECTORY
val_dir = VALIDATION_DIRECTORY
test_dir = TEST_DIRECTORY

# Image Data Generators for preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Wczytanie modelu ResNet50 bez warstw klasyfikacyjnych
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze some initial layers and allow fine-tuning for deeper layers
for layer in base_model.layers[:140]:  # Freezing initial 140 layers
    layer.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)  # Increase size of Dense layer for more capacity
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)  # 3 classes: benign, malignant, normal

model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a learning rate scheduler
initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate)

# Learning rate scheduler to reduce the learning rate if the validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with additional callbacks
history = model.fit(
    train_generator,
    epochs=45,
    validation_data=val_generator,
    callbacks=[lr_scheduler, early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the model in both formats
model.save(BREAST_USG_MODEL_H5)  # Save in HDF5 format
# model.export(MODEL_PATH)  # Export to SavedModel format
