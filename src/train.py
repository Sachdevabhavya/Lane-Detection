import tensorflow as tf
from model import unet_model, dice_coef
from dataset import load_data

# Load Data
train_image_dir = "../data/processed/train/images"
train_mask_dir = "../data/processed/train/masks"
val_image_dir = "../data/processed/val/images"
val_mask_dir = "../data/processed/val/masks"

train_dataset, val_dataset = load_data(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir)

# Build & Compile Model
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", dice_coef])

# Train Model
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("../models/unet_road_segmentation_v0.0.1.h5", save_best_only=True),
]

model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=callbacks)
model.save("../models/final_unet_road_segmentation_model_v0.0.1.h5")
