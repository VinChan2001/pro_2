"""
Model utilities for sentiment analysis
Includes functions for model evaluation, comparison, and analysis
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import pandas as pd
from config import MODEL_CONFIG, PATHS, VIZ_CONFIG
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or PATHS['model_file']
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            logger.error(f"Error loading model: {e}")
            
    def evaluate_on_test_data(self):
        """Comprehensive evaluation on IMDB test data"""
        print("📊 Evaluating model on IMDB test data...")
        
        # Load test data
        (_, _), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=MODEL_CONFIG['vocab_size']
        )
        
        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        x_test = pad_sequences(x_test, maxlen=MODEL_CONFIG['max_length'])
        
        # Get predictions
        y_pred_prob = self.model.predict(x_test, verbose=1)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(f"\n🎯 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # ROC curve
        self.plot_roc_curve(y_test, y_pred_prob)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='detailed_confusion_matrix.png'):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'{cm[i,j]/total:.2%}', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        print(f"💾 Confusion matrix saved to {save_path}")
        
    def plot_roc_curve(self, y_true, y_pred_prob, save_path='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        print(f"💾 ROC curve saved to {save_path}")
        return roc_auc
    
    def analyze_prediction_distribution(self, y_pred_prob, save_path='prediction_distribution.png'):
        """Analyze the distribution of prediction probabilities"""
        plt.figure(figsize=(12, 4))
        
        # Histogram of probabilities
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_prob, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence distribution
        plt.subplot(1, 2, 2)
        confidence = np.where(y_pred_prob > 0.5, y_pred_prob, 1 - y_pred_prob)
        plt.hist(confidence, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        print(f"💾 Prediction distribution saved to {save_path}")
        
        # Print statistics
        high_confidence = np.sum(confidence > 0.8)
        medium_confidence = np.sum((confidence > 0.6) & (confidence <= 0.8))
        low_confidence = np.sum(confidence <= 0.6)
        total = len(confidence)
        
        print(f"\n📊 Confidence Analysis:")
        print(f"   High confidence (>80%): {high_confidence}/{total} ({high_confidence/total:.1%})")
        print(f"   Medium confidence (60-80%): {medium_confidence}/{total} ({medium_confidence/total:.1%})")
        print(f"   Low confidence (≤60%): {low_confidence}/{total} ({low_confidence/total:.1%})")
    
    def model_summary_report(self):
        """Generate comprehensive model summary"""
        if not self.model:
            print("❌ No model loaded")
            return
        
        print("🔍 Model Architecture Summary")
        print("=" * 40)
        self.model.summary()
        
        # Count parameters by layer type
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        print(f"\n📈 Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Layer information
        print(f"\n🧠 Layer Information:")
        for i, layer in enumerate(self.model.layers):
            print(f"   {i+1}. {layer.name} ({layer.__class__.__name__})")
            if hasattr(layer, 'units'):
                print(f"      Units: {layer.units}")
            if hasattr(layer, 'rate'):
                print(f"      Dropout rate: {layer.rate}")


def compare_models(model_paths, model_names=None):
    """Compare multiple models side by side"""
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results = []
    
    for i, (path, name) in enumerate(zip(model_paths, model_names)):
        try:
            evaluator = ModelEvaluator(path)
            result = evaluator.evaluate_on_test_data()
            result['model_name'] = name
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
    
    if not results:
        print("❌ No models could be evaluated")
        return
    
    # Create comparison table
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Model': result['model_name'],
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'Test Loss': f"{result['test_loss']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n📊 Model Comparison:")
    print(df.to_string(index=False))
    
    return results


def main():
    """Main function for model evaluation"""
    print("🔍 Model Evaluation Toolkit")
    print("=" * 35)
    
    evaluator = ModelEvaluator()
    
    print("\nChoose evaluation option:")
    print("1. Full evaluation on test data")
    print("2. Model summary report")
    print("3. Quick accuracy check")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        results = evaluator.evaluate_on_test_data()
        evaluator.analyze_prediction_distribution(results['y_pred_prob'])
    elif choice == '2':
        evaluator.model_summary_report()
    elif choice == '3':
        # Quick evaluation without detailed plots
        (_, _), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=MODEL_CONFIG['vocab_size']
        )
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        x_test = pad_sequences(x_test, maxlen=MODEL_CONFIG['max_length'])
        
        test_loss, test_accuracy = evaluator.model.evaluate(x_test, y_test, verbose=1)
        print(f"\n🎯 Quick Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()