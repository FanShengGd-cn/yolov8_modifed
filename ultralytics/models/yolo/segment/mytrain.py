from ultralytics import YOLO
import os
if __name__ == '__main__':

      
    model = YOLO('yolov8-seg.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    model.train(epochs=100, data='my2class.yaml')