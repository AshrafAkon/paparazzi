from pathlib import Path

# Dataset and DataLoader
BASE_DIR = Path(__file__).absolute().parent / "data"
# Parameters
train_dir = BASE_DIR / "train"  # Set the path to your dataset
val_dir = BASE_DIR / "validation"
BATCH_SIZE = 64
learning_rate = 0.00001
num_epochs = 30
NUM_CLASSES = 100
MODEL_PATH = BASE_DIR / "models" / "trained.pth"
CHECKPOINT_PATH = BASE_DIR / "models" / "trained.checkpoint"
