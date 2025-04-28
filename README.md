# Vehicle Damage Claim Detection

This project uses a YOLOv8 model trained with a vehicle damage dataset to detect damages and assist in claims processing. It is built with Flask for the backend server and Ultralytics for model training and inference.

## Project Structure

```
project/
├── app.py                 # Flask application
├── model/
│   └── best.pt            # YOLOv11 trained model
├── dataset/               # Dataset for future training
│   ├── yolo11m.pt   
│   ├── requirements.txt   
│   ├── runs/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml       # Dataset configuration
|   └── train.py           # For training dataset
|   └── predict.py         # For testing the model
├── static/
|   ├── result/
|   └── uploads/  
├── templates/
|   └── index.html  
├── requirements.txt
├── car_brand_models.csv   # CSV file for car brands and models
├── prices.csv             # CSV file for claims amount
├── requirements.txt
└── README.md
```

## Running the Application

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the Flask server:

```
python app.py
```

3. Open your browser at:

```
http://localhost:5000
```

## Model Information

- Model: YOLOv8
- Framework: Ultralytics
- Model file: `model/best.pt`

## Re-Training the Model

To update the model with a new dataset:

1. Place your new dataset inside the `dataset/` folder following this structure:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

2. Edit `dataset.yaml` with your new classes. Example:

```yaml
train: dataset/images/train
val: dataset/images/val

nc: 4
names: ['scratch', 'dent', 'broken_glass', 'bumper_damage']
```

3. Train the model using the existing weights:

```
yolo detect train data=dataset/dataset.yaml model=model/best.pt epochs=100 imgsz=640
```

4. After training, replace the old `model/best.pt` with the new one found at:

```
runs/detect/train/weights/best.pt
```

## Requirements

- Python 3.8+
- Flask
- Ultralytics

Install manually if needed:

```
pip install flask ultralytics
```

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Roboflow](https://roboflow.com/)

Ready for training, detection, and deployment!

