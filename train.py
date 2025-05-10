import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ultralytics import YOLO
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_yolo_seg(config_path='Config/config.yaml', model_size='n'):
    """
    Train YOLOv11 segmentation model
    
    Args:
        config_path (str): Path to config file
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        For tree instance segmentation, we use 'n' model size
        We can use 's', 'm', 'l', 'x' model size for other tasks
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Initialize YOLO model
    model = YOLO(f'yolov8{model_size}-seg.pt')
    logger.info(f"Initialized YOLOv8{model_size}-seg model")
    
    # Training configuration
    training_args = {
        'data': 'dataset.yaml',  # Path to your dataset configuration
        'epochs': config['MODEL']['EPOCHS'],
        'imgsz': config['MODEL']['IMAGE_SIZE'],
        'batch': config['MODEL']['BATCH_SIZE'],
        'lr0': config['MODEL']['LEARNING_RATE'],
        'weight_decay': config['MODEL']['WEIGHT_DECAY'],
        'momentum': config['MODEL']['MOMENTUM'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': 'runs/train',
        'name': config['NAME'],
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'save': True,  # Save best model
        'save_period': 10,  # Save every 10 epochs
        'patience': 50,  # Early stopping patience
        'plots': True,  # Generate training plots
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for last 10 epochs
    }
    
    try:
        logger.info("Starting training...")
        results = model.train(**training_args)
        
        # Save the final model
        model.save('best.pt')
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to 'best.pt'")
        
        # Print training results
        metrics = results.results_dict
        logger.info("\nTraining Results:")
        logger.info(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        logger.info(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        logger.info(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
        return model, results
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

def validate_model(model, data_yaml):
    """
    Validate the trained model
    
    Args:
        model: Trained YOLO model
        data_yaml (str): Path to dataset configuration
    """
    try:
        results = model.val(data=data_yaml)
        return results
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = 'Config/config.yaml'
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Check model type and train accordingly
    if config['MODEL']['NAME'] == 'YOLO':
        logger.info("Starting YOLO segmentation training...")
        model, results = train_yolo_seg(config_path=config_path, model_size='n')
        logger.info("Starting model validation...")
        validation_results = validate_model(model, 'dataset.yaml')
    else:
        logger.error(f"Unsupported model type: {config['MODEL']['NAME']}")
        raise ValueError(f"Model type {config['MODEL']['NAME']} is not supported. Expected 'YOLO'")
