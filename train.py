import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ultralytics import YOLO
import logging
from models.MaskRCNN import get_backbone
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

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
    model = YOLO(f'yolov11{model_size}-seg.pt')
    logger.info(f"Initialized YOLOv11{model_size}-seg model")
    
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

def validate_model(model, data_yaml, model_type):
    """
    Unified validation function for both YOLO and Mask R-CNN models
    
    Args:
        model: Trained model (YOLO or Mask R-CNN)
        data_yaml (str): Path to dataset configuration
        model_type (str): Type of model ('YOLO' or 'MASKRCNN')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if model_type == 'YOLO':
            results = model.val(data=data_yaml)
            return results
        elif model_type == 'MASKRCNN':
            model.eval()
            total_loss = 0
            
            with torch.no_grad():
                for images, targets in data_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    total_loss += sum(loss for loss in loss_dict.values())
                    
            return total_loss / len(data_loader)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        raise

def train_mask_rcnn(config_path='Config/config.yaml'):
    """
    Train Mask R-CNN model for instance segmentation
    
    Args:
        config_path (str): Path to config file
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize backbone
    backbone = get_backbone(
        backbone_name=config['MODEL']['MASKRCNN_BACKBONE'],
        pretrained=True
    )
    
    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Initialize Mask R-CNN model
    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,  # Background + foreground
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=100,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        mask_min_size=28,
        mask_score_thresh=0.5
    )
    
    model = model.to(device)
    logger.info("Initialized Mask R-CNN model")
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['MODEL']['LEARNING_RATE'],
        weight_decay=config['MODEL']['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['MODEL']['EPOCHS']
    )
    
    # Training loop
    try:
        logger.info("Starting training...")
        for epoch in range(config['MODEL']['EPOCHS']):
            model.train()
            total_loss = 0
            lr_scheduler.step()
            logger.info(f"Epoch {epoch+1}/{config['MODEL']['EPOCHS']}, Loss: {total_loss:.4f}")     
        # Save the final model
        torch.save(model.state_dict(), 'mask_rcnn_model.pth')
        logger.info("Training completed successfully!")
        logger.info("Model saved to 'mask_rcnn_model.pth'")
        
        return model
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config_path = 'Config/config.yaml'
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    if config['MODEL']['NAME'] == 'YOLO':
        yolo_config_path = 'Config/YOLO/yolov11-seg.yaml'
        logger.info("Starting YOLO segmentation training...")
        model, results = train_yolo_seg(config_path=config_path, model_size='n')
        logger.info("Starting model validation...")
        validation_results = validate_model(model, 'dataset.yaml', 'YOLO')
    elif config['MODEL']['NAME'] == 'MASKRCNN':
        logger.info("Starting Mask R-CNN training...")
        model = train_mask_rcnn(config_path=config_path)
        logger.info("Starting model validation...")
        validation_results = validate_model(model, 'dataset.yaml', 'MASKRCNN')
    else:
        logger.error(f"Unsupported model type: {config['MODEL']['NAME']}")
        raise ValueError(f"Model type {config['MODEL']['NAME']} is not supported. Expected 'YOLO' or 'MASKRCNN'")
