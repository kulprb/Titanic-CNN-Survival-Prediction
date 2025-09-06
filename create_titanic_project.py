#!/usr/bin/env python3
"""
Complete project creator for Titanic CNN Professional project.
Run this script in your D:/projects/titanic/ directory to create the entire project.
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project structure and files."""
    
    # Create base directory
    base_dir = Path("titanic_cnn_professional")
    base_dir.mkdir(exist_ok=True)
    
    # Create directory structure
    directories = [
        "configs",
        "data/raw", "data/processed", "data/external",
        "src/data", "src/models", "src/utils", "src/visualization",
        "models/saved_models", "models/checkpoints",
        "reports/figures",
        "logs",
        "scripts",
        "notebooks/exploratory", "notebooks/modeling"
    ]
    
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    # Create requirements.txt
    requirements_content = '''# Core ML Libraries
tensorflow==2.17.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4

# Visualization
matplotlib==3.9.2
seaborn==0.13.2

# Utilities
tqdm==4.66.5
joblib==1.4.2
pyyaml==6.0.2

# Web Framework
flask==3.0.3

# Additional utilities
requests==2.32.3
'''
    
    with open(base_dir / "requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt created")
    
    # Create simple setup.py
    setup_content = '''#!/usr/bin/env python3
"""Simple setup script for Titanic CNN project."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("Setting up Titanic CNN project...")
    
    # Create virtual environment
    if not Path("venv").exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    
    # Install dependencies
    pip_path = "venv\\\\Scripts\\\\pip" if os.name == 'nt' else "venv/bin/pip"
    print("Installing dependencies...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed")
    
    print("\\n🎉 Setup complete!")
    print("\\nNext steps:")
    if os.name == 'nt':
        print("1. venv\\\\Scripts\\\\activate")
    else:
        print("1. source venv/bin/activate")
    print("2. python scripts/simple_titanic_cnn.py")

if __name__ == "__main__":
    main()
'''
    
    with open(base_dir / "setup.py", "w") as f:
        f.write(setup_content)
    
    print("✅ setup.py created")
    
    # Create simplified CNN training script
    simple_cnn_content = '''#!/usr/bin/env python3
"""
Simplified Titanic CNN implementation for quick start.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import requests
from pathlib import Path

class SimpleTitanicCNN:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def download_data(self):
        """Download Titanic dataset."""
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        data_path = Path("data/raw/train.csv")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not data_path.exists():
            print("Downloading Titanic dataset...")
            response = requests.get(url)
            with open(data_path, 'w') as f:
                f.write(response.text)
            print("✅ Dataset downloaded")
        
        return pd.read_csv(data_path)
    
    def preprocess_data(self, df):
        """Preprocess the data."""
        # Handle missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Create features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Encode categorical variables
        for col in ['Sex', 'Embarked']:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])
        
        # Select features
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
        X = df[features].values
        y = df['Survived'].values if 'Survived' in df.columns else None
        
        return X, y
    
    def build_model(self, input_shape):
        """Build 1D CNN model."""
        model = keras.Sequential([
            keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            
            # 1D Convolutional layers
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Complete training pipeline."""
        print("🚢 Starting Titanic CNN Training...")
        
        # Load and preprocess data
        df = self.download_data()
        print(f"Dataset shape: {df.shape}")
        
        X, y = self.preprocess_data(df)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        print("\\nModel Architecture:")
        self.model.summary()
        
        # Train model
        print("\\n🔥 Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        # Evaluate model
        print("\\n📊 Evaluating model...")
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\\n🎯 Test Accuracy: {accuracy:.4f}")
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot training history
        self.plot_history(history)
        
        # Save model
        model_path = Path("models/saved_models")
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path / "simple_titanic_cnn.h5")
        print(f"\\n💾 Model saved to {model_path / 'simple_titanic_cnn.h5'}")
        
        return accuracy
    
    def plot_history(self, history):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        fig_path = Path("reports/figures")
        fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Training plots saved to {fig_path / 'training_history.png'}")

def main():
    """Main function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create and train model
    titanic_cnn = SimpleTitanicCNN()
    accuracy = titanic_cnn.train_model()
    
    print("\\n" + "="*50)
    print("🎉 TITANIC CNN TRAINING COMPLETED!")
    print("="*50)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\\nFiles created:")
    print("📁 data/raw/train.csv - Dataset")
    print("💾 models/saved_models/simple_titanic_cnn.h5 - Trained model")
    print("📈 reports/figures/training_history.png - Training plots")
    print("="*50)

if __name__ == "__main__":
    main()
'''
    
    with open(base_dir / "scripts/simple_titanic_cnn.py", "w") as f:
        f.write(simple_cnn_content)
    
    print("✅ Simplified CNN script created")
    
    # Create __init__.py files
    init_content = '"""Titanic CNN Professional Project"""'
    
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py"
    ]
    
    for init_file in init_files:
        with open(base_dir / init_file, "w") as f:
            f.write(init_content)
    
    print("✅ __init__.py files created")
    
    # Create README
    readme_content = '''# Titanic Survival Prediction - CNN Implementation

## 🚀 Quick Start

1. **Setup Environment:**
