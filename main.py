"""
Movie Review Sentiment Analysis using TensorFlow
A fun NLP project that analyzes the sentiment of movie reviews using deep learning
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud
from tqdm import tqdm
import os
import ssl

# Handle SSL certificate issues for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SentimentAnalyzer:
    def __init__(self, vocab_size=10000, max_length=250, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self):
        """Load and prepare the IMDB movie reviews dataset"""
        print("📂 Loading IMDB movie reviews dataset...")
        
        # Load the dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=self.vocab_size
        )
        
        # Get word index for text decoding
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # Function to decode reviews
        def decode_review(encoded_review):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
        
        # Pad sequences to same length
        x_train = pad_sequences(x_train, maxlen=self.max_length)
        x_test = pad_sequences(x_test, maxlen=self.max_length)
        
        print(f"✅ Dataset loaded:")
        print(f"   Training samples: {len(x_train)}")
        print(f"   Test samples: {len(x_test)}")
        print(f"   Sequence length: {self.max_length}")
        print(f"   Vocabulary size: {self.vocab_size}")
        
        # Store some sample reviews for visualization
        self.sample_reviews = []
        self.sample_labels = []
        for i in range(10):
            self.sample_reviews.append(decode_review(x_train[i]))
            self.sample_labels.append("Positive" if y_train[i] == 1 else "Negative")
        
        return x_train, y_train, x_test, y_test
    
    def build_model(self):
        """Build LSTM neural network model"""
        print("🧠 Building LSTM model...")
        
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Dropout(0.2),
            layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
            layers.LSTM(32, dropout=0.5, recurrent_dropout=0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        """Train the model with validation"""
        print(f"🏋️ Training model for {epochs} epochs...")
        
        # Create validation split
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001
        )
        
        # Train model
        self.history = self.model.fit(
            x_train_split, y_train_split,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        
        return self.history
    
    def plot_training_history(self):
        """Visualize training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, x_test, y_test):
        """Detailed model evaluation"""
        print("📊 Generating detailed evaluation...")
        
        # Predictions
        y_pred_prob = self.model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_pred_prob
    
    def show_sample_predictions(self, x_test, y_test, num_samples=8):
        """Show sample predictions with text"""
        print("🔍 Showing sample predictions...")
        
        # Get word index for decoding
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        def decode_review(encoded_review):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
        
        # Random sample indices
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        predictions = self.model.predict(x_test[indices])
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            review = decode_review(x_test[idx])
            true_label = "Positive" if y_test[idx] == 1 else "Negative"
            pred_prob = predictions[i][0]
            pred_label = "Positive" if pred_prob > 0.5 else "Negative"
            
            # Truncate review for display
            display_review = review[:200] + "..." if len(review) > 200 else review
            
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].text(0.05, 0.95, f"True: {true_label}", transform=axes[i].transAxes, 
                        fontsize=12, fontweight='bold', va='top')
            axes[i].text(0.05, 0.85, f"Predicted: {pred_label} ({pred_prob:.2%})", 
                        transform=axes[i].transAxes, fontsize=12, fontweight='bold', 
                        va='top', color=color)
            axes[i].text(0.05, 0.75, f"Review: {display_review}", transform=axes[i].transAxes, 
                        fontsize=10, va='top', wrap=True)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].axis('off')
            axes[i].add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                                          edgecolor=color, linewidth=2))
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='sentiment_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"💾 Model saved as {filepath}")


def main():
    print("🎬 Movie Review Sentiment Analysis with TensorFlow")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(vocab_size=10000, max_length=250, embedding_dim=128)
    
    # Prepare data
    x_train, y_train, x_test, y_test = analyzer.prepare_data()
    
    # Build model
    model = analyzer.build_model()
    
    # Train model
    history = analyzer.train_model(x_train, y_train, x_test, y_test, epochs=10)
    
    # Visualize training
    print("\n📈 Generating training visualizations...")
    analyzer.plot_training_history()
    
    # Evaluate model
    y_pred, y_pred_prob = analyzer.evaluate_model(x_test, y_test)
    
    # Show sample predictions
    analyzer.show_sample_predictions(x_test, y_test)
    
    # Save model
    analyzer.save_model()
    
    # Final summary
    final_accuracy = analyzer.model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"\n🎯 Final Results:")
    print(f"   Test Accuracy: {final_accuracy:.2%}")
    print(f"   Model saved: sentiment_model.h5")
    print(f"\n🎉 Training completed successfully!")
    print("📊 Check out the generated visualizations:")
    print("   - training_history.png")
    print("   - confusion_matrix.png") 
    print("   - sample_predictions.png")


if __name__ == "__main__":
    main()