"""
Hybrid CNN Model for Vegetable Freshness Classification
========================================================
This script implements a hybrid deep learning model combining:
- MobileNetV2 backbone (pretrained on ImageNet)
- Spatial Attention Module
- Multi-task classification for freshness prediction

Optimized for Kaggle GPU training (Tesla P100/T4)

To use on Kaggle:
1. Upload your dataset to Kaggle
2. Create a new notebook and copy this code
3. Update DATA_DIR to match your Kaggle dataset path
4. Enable GPU accelerator and run all cells
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Sklearn for evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for all hyperparameters and paths"""
    
    # Data paths - UPDATE THIS FOR KAGGLE
    # For Kaggle: DATA_DIR = '/kaggle/input/your-dataset-name/Dataset/Vegetables'
    # For local: DATA_DIR = 'Dataset/Vegetables'
    DATA_DIR = '/kaggle/input/vegetable-freshness/Dataset/Vegetables'
    
    # Image settings
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3
    INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 10  # Train only classification head
    EPOCHS_PHASE2 = 20  # Fine-tune with backbone
    INITIAL_LR = 0.001
    FINE_TUNE_LR = 0.0001
    
    # Model settings
    DROPOUT_RATE_1 = 0.5
    DROPOUT_RATE_2 = 0.3
    L2_REG = 0.01
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Classes
    FRESHNESS_CLASSES = ['Fresh', 'Rotten']
    VEGETABLE_CLASSES = ['Bellpepper', 'Carrot', 'Cucumber', 'Potato', 'Tomato']
    
    # Output paths
    MODEL_SAVE_PATH = 'vegetable_freshness_model.h5'
    TFLITE_SAVE_PATH = 'vegetable_freshness_model.tflite'
    
config = Config()

# Check GPU availability
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# ============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_dataset(data_dir):
    """
    Load dataset from directory structure:
    Dataset/Vegetables/
        ├── FreshBellpepper/
        ├── RottenBellpepper/
        ├── FreshCarrot/
        ├── RottenCarrot/
        └── ...
    
    Returns:
        DataFrame with columns: image_path, freshness_label, vegetable_type
    """
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    data = []
    
    # Iterate through all folders
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder name to get freshness and vegetable type
        folder_lower = folder_name.lower()
        
        # Determine freshness
        if folder_lower.startswith('fresh'):
            freshness = 'Fresh'
            vegetable = folder_name[5:]  # Remove 'Fresh' prefix
        elif folder_lower.startswith('rotten'):
            freshness = 'Rotten'
            vegetable = folder_name[6:]  # Remove 'Rotten' prefix
        else:
            print(f"  Skipping unknown folder: {folder_name}")
            continue
        
        # Normalize vegetable name
        vegetable = vegetable.capitalize()
        
        # Get all images in folder
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_path = os.path.join(folder_path, filename)
                data.append({
                    'image_path': image_path,
                    'freshness': freshness,
                    'vegetable': vegetable
                })
    
    df = pd.DataFrame(data)
    
    # Print dataset statistics
    print(f"\nTotal images: {len(df)}")
    print("\nFreshness distribution:")
    print(df['freshness'].value_counts())
    print("\nVegetable distribution:")
    print(df['vegetable'].value_counts())
    print("\nCross-tabulation:")
    print(pd.crosstab(df['vegetable'], df['freshness']))
    
    return df


def create_data_generators(df, config):
    """
    Create train, validation, and test data generators with augmentation
    """
    print("\n" + "=" * 60)
    print("CREATING DATA GENERATORS")
    print("=" * 60)
    
    # Split data into train, val, test
    train_df, temp_df = train_test_split(
        df, 
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=df['freshness'],
        random_state=SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=temp_df['freshness'],
        random_state=SEED
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='freshness',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='freshness',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='freshness',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nClass indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator, train_df, val_df, test_df


# ============================================================================
# SECTION 4: CUSTOM ATTENTION LAYER
# ============================================================================

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module
    
    Focuses the model on regions that are most relevant for freshness detection
    (e.g., areas of decay, discoloration, texture changes)
    """
    
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Learnable attention weights
        self.conv1 = layers.Conv2D(
            filters=input_shape[-1] // 4,
            kernel_size=1,
            activation='relu',
            padding='same'
        )
        self.conv2 = layers.Conv2D(
            filters=1,
            kernel_size=7,
            activation='sigmoid',
            padding='same'
        )
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Generate attention map
        attention = self.conv1(inputs)
        attention = self.conv2(attention)
        
        # Apply attention to input
        return inputs * attention
    
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        return config


class ChannelAttention(layers.Layer):
    """
    Channel Attention Module (Squeeze-and-Excitation style)
    
    Recalibrates channel-wise feature responses
    """
    
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, channels))
        super(ChannelAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Squeeze
        attention = self.gap(inputs)
        
        # Excitation
        attention = self.fc1(attention)
        attention = self.fc2(attention)
        attention = self.reshape(attention)
        
        # Scale
        return inputs * attention
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


# ============================================================================
# SECTION 5: HYBRID MODEL ARCHITECTURE
# ============================================================================

def build_hybrid_model(config, trainable_backbone=False):
    """
    Build the hybrid CNN model with:
    - MobileNetV2 backbone
    - Spatial and Channel Attention modules
    - Classification head for freshness prediction
    
    Args:
        config: Configuration object
        trainable_backbone: Whether to allow training of backbone weights
    
    Returns:
        Compiled Keras model
    """
    print("\n" + "=" * 60)
    print("BUILDING HYBRID MODEL")
    print("=" * 60)
    
    # Input layer
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    
    # =========== BACKBONE ===========
    # Load MobileNetV2 without top layers
    backbone = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = trainable_backbone
    
    # Get features from backbone
    x = backbone(inputs)
    
    print(f"Backbone output shape: {x.shape}")
    print(f"Backbone trainable: {trainable_backbone}")
    print(f"Total backbone layers: {len(backbone.layers)}")
    
    # =========== ATTENTION MODULES ===========
    # Apply Channel Attention
    x = ChannelAttention(reduction_ratio=16, name='channel_attention')(x)
    
    # Apply Spatial Attention
    x = SpatialAttention(name='spatial_attention')(x)
    
    # =========== CLASSIFICATION HEAD ===========
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers with dropout
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=l2(config.L2_REG),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(config.DROPOUT_RATE_1, name='dropout_1')(x)
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(config.L2_REG),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(config.DROPOUT_RATE_2, name='dropout_2')(x)
    
    # Output layer (Fresh/Rotten classification)
    outputs = layers.Dense(
        len(config.FRESHNESS_CLASSES),
        activation='softmax',
        name='freshness_output'
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='VegetableFreshnessClassifier')
    
    # Print model summary
    print("\n" + "-" * 40)
    print("MODEL SUMMARY")
    print("-" * 40)
    model.summary()
    
    # Count parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    
    return model, backbone


def compile_model(model, learning_rate):
    """Compile model with optimizer and loss function"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# SECTION 6: TRAINING
# ============================================================================

def create_callbacks(phase_name='phase1'):
    """Create training callbacks"""
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=f'best_model_{phase_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, train_gen, val_gen, epochs, phase_name='phase1'):
    """Train the model"""
    
    print("\n" + "=" * 60)
    print(f"TRAINING - {phase_name.upper()}")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=create_callbacks(phase_name),
        verbose=1
    )
    
    return history


def fine_tune_model(model, backbone, config, num_layers_to_unfreeze=30):
    """
    Fine-tune the model by unfreezing top layers of backbone
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING BACKBONE")
    print("=" * 60)
    
    # Unfreeze backbone
    backbone.trainable = True
    
    # Freeze all layers except the last N
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in backbone.layers if layer.trainable])
    print(f"Unfrozen {trainable_count} layers in backbone")
    
    # Recompile with lower learning rate
    model = compile_model(model, config.FINE_TUNE_LR)
    
    return model


# ============================================================================
# SECTION 7: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, test_gen, class_names):
    """Evaluate model on test set"""
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    # Classification report
    print("\n" + "-" * 40)
    print("CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return predictions, predicted_classes, true_classes, cm


def plot_training_history(history, phase_name='training'):
    """Plot training and validation accuracy/loss curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{phase_name} - Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'{phase_name} - Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{phase_name}_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 14}
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, test_gen, class_names, num_samples=12):
    """Visualize sample predictions"""
    
    # Get a batch
    test_gen.reset()
    images, labels = next(test_gen)
    predictions = model.predict(images)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # Display image
        ax.imshow(images[i])
        
        # Get prediction
        pred_class = np.argmax(predictions[i])
        true_class = np.argmax(labels[i])
        confidence = predictions[i][pred_class] * 100
        
        # Set title color based on correctness
        color = 'green' if pred_class == true_class else 'red'
        
        ax.set_title(
            f'True: {class_names[true_class]}\n'
            f'Pred: {class_names[pred_class]} ({confidence:.1f}%)',
            color=color,
            fontsize=10
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# SECTION 8: MODEL EXPORT
# ============================================================================

def save_model(model, config):
    """Save model in multiple formats"""
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save as HDF5
    model.save(config.MODEL_SAVE_PATH)
    print(f"✓ Saved HDF5 model: {config.MODEL_SAVE_PATH}")
    
    # Save as SavedModel
    model.save('saved_model/', save_format='tf')
    print("✓ Saved TensorFlow SavedModel: saved_model/")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(config.TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = os.path.getsize(config.TFLITE_SAVE_PATH) / (1024 * 1024)
    print(f"✓ Saved TFLite model: {config.TFLITE_SAVE_PATH} ({tflite_size:.2f} MB)")
    
    return config.MODEL_SAVE_PATH, config.TFLITE_SAVE_PATH


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 60)
    print("VEGETABLE FRESHNESS CLASSIFICATION - HYBRID CNN MODEL")
    print("=" * 60)
    
    # Step 1: Load dataset
    df = load_dataset(config.DATA_DIR)
    
    # Step 2: Create data generators
    train_gen, val_gen, test_gen, train_df, val_df, test_df = create_data_generators(df, config)
    
    # Step 3: Build model (backbone frozen)
    model, backbone = build_hybrid_model(config, trainable_backbone=False)
    model = compile_model(model, config.INITIAL_LR)
    
    # Step 4: Phase 1 Training (frozen backbone)
    print("\n" + "*" * 60)
    print("PHASE 1: Training Classification Head Only")
    print("*" * 60)
    
    history1 = train_model(
        model, train_gen, val_gen,
        epochs=config.EPOCHS_PHASE1,
        phase_name='phase1'
    )
    plot_training_history(history1, 'Phase 1 - Frozen Backbone')
    
    # Step 5: Fine-tune backbone
    model = fine_tune_model(model, backbone, config)
    
    # Step 6: Phase 2 Training (fine-tuning)
    print("\n" + "*" * 60)
    print("PHASE 2: Fine-tuning Backbone")
    print("*" * 60)
    
    history2 = train_model(
        model, train_gen, val_gen,
        epochs=config.EPOCHS_PHASE2,
        phase_name='phase2'
    )
    plot_training_history(history2, 'Phase 2 - Fine-tuning')
    
    # Step 7: Evaluate on test set
    class_names = list(train_gen.class_indices.keys())
    predictions, pred_classes, true_classes, cm = evaluate_model(model, test_gen, class_names)
    
    # Step 8: Visualizations
    plot_confusion_matrix(cm, class_names)
    visualize_predictions(model, test_gen, class_names)
    
    # Step 9: Save model
    h5_path, tflite_path = save_model(model, config)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to:")
    print(f"  - {h5_path}")
    print(f"  - {tflite_path}")
    print("\nYou can now use these models for inference!")
    
    return model, history1, history2


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    model, history1, history2 = main()
