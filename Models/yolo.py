from ultralytics import YOLO

def YOLO_model(model_path):
    model = YOLO(model_path)
    return model

def train_yolo(model_path, data_path, epochs=100, batch_size=16, img_size=640):
    model = YOLO_model(model_path)
    model.train(data=data_path, epochs=epochs, batch=batch_size, img_size=img_size)
