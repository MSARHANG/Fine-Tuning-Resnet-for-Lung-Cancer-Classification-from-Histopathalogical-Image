DEVICE = 'cuda'
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-4
NUM_CLASSES = 3
IMG_SIZE = 224
IN_CHANNELS = 3
DROPOUT = 0.2
HIDDEN_DIM = 1024
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),         
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
