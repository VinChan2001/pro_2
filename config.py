"""
Configuration file for the Sentiment Analysis project
"""

# Model configuration
MODEL_CONFIG = {
    'vocab_size': 10000,
    'max_length': 250,
    'embedding_dim': 128,
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units': 32,
    'dropout_rate': 0.5,
    'recurrent_dropout': 0.5,
    'embedding_dropout': 0.2,
    'dense_dropout': 0.5
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    'patience': 3,  # Early stopping patience
    'reduce_lr_patience': 2,  # Learning rate reduction patience
    'min_lr': 0.0001
}

# File paths
PATHS = {
    'model_file': 'sentiment_model.h5',
    'training_history': 'training_history.png',
    'confusion_matrix': 'confusion_matrix.png',
    'sample_predictions': 'sample_predictions.png',
    'word_clouds': 'word_clouds.png',
    'custom_wordclouds': 'custom_wordclouds.png',
    'prediction_analysis': 'prediction_analysis.png'
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

# Demo configuration
DEMO_CONFIG = {
    'sample_reviews': [
        "This movie was absolutely fantastic! Great acting, amazing plot, and beautiful cinematography. I loved every minute of it.",
        "Terrible movie. Waste of time and money. Poor acting and boring storyline. Would not recommend to anyone.",
        "An okay film. Nothing spectacular but decent entertainment for a weekend. Could have been better.",
        "Outstanding performance by the lead actor! This film will definitely win awards. Masterpiece of modern cinema.",
        "I fell asleep halfway through. Very slow paced and confusing plot. Not worth watching.",
        "Brilliant direction and script. This movie made me laugh and cry. Truly emotional journey.",
        "Worst movie I've ever seen. Terrible special effects and laughable dialogue. Complete disaster.",
        "Pretty good movie overall. Solid 7/10. Good for a date night or casual viewing.",
        "This film changed my perspective on life. Incredible storytelling and deep meaningful themes.",
        "Boring and predictable. The ending was obvious from the first scene. Very disappointing."
    ]
}