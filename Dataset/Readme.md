# Dataset Organization Guide 


### Dataset Structure
```
Dataset/
├── images/                 # Original images
│   ├── train/             # Training images
│   └── val/               # Validation images
├── annotations/           # Annotation files
│   ├── train/            # Training annotations
│   └── val/              # Validation annotations

```

### File Formats
- Images: `.jpg`
- Annotations: `.json` format (Labelme format)


### Data Organization
1. **Images Directory**
   - Contains original input images
   - Organized into train and validation sets
   - Maintain consistent naming convention: `image_XXXX.jpg`

2. **Annotations Directory**
   - Contains corresponding annotation files
   - Follows Labelme format for instance segmentation
   - Each annotation file corresponds to an image
   - Naming convention: `image_XXXX.json`



