from pathlib import Path

# Model configuration
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 5,  # Number of pose classes
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Paths
BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / 'models' / 'saved_models'