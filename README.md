## SAFETY HELMET DETECTION(Using YOLOv8) 👷‍♂️

In industrial and construction settings, worker safety is paramount. This project develops a real-time safety helmet detection system using the YOLOv8 deep learning model, known for its efficiency and accuracy. The main objective of this project is to create an automated system that can accurately identify whether workers are wearing safety helmets in real-time video streams and detect the people who are not wearing safety helmets.

## DATASET:🗂️

The image dataset for training this model was collected from various sources and converted to the appropriate file type. It was then passed through several preprocessing and augmentation steps (e.g., horizontal flipping) using Roboflow. The final dataset contains nearly 28,000 images and is available at [Dataset](https://www.kaggle.com/datasets/archisman24/new-dataset?select=new+dataset). The dataset is divided into 3 parts, training, validation and test sets and contained 3 classes : Helmet, No Helmet, and Worker (with varying levels of representation in the dataset).
For more information on the dataset, visit my [website](https://archismansengupta.site/safetyhelmetdetection).


## STEPS INVOLVED:✅

1. Install all the necessary packages used for this project, using the command : ` pip install -r requirements.txt`.

2. The model was trained for 3 different epochs, i.e. for 30, 50 and 100 epochs, to achieve the best performance from the model. The model was trained using CLI commands in Jupyter Notebook as follows:

```python
# Training YOLOv8 model for 30 epochs:
!yolo task=detect mode=train model=yolov8n.pt data="../new dataset/data.yaml" epochs=30 imgsz=640 batch=16

# Training YOLOv8 model for 50 epochs:
!yolo task=detect mode=train model=yolov8n.pt data="../new dataset/data.yaml" epochs=50 imgsz=640 batch=16

# Training YOLOv8 model for 100 epochs:
!yolo task=detect mode=train model=yolov8n.pt data="../new dataset/data.yaml" epochs=100 imgsz=640 batch=16
```

 Here’s a breakdown of each part:

- `!yolo`: This exclamation mark indicates the start of a shell command in some environments, like Jupyter notebooks.
- `task=detect`: Specifies that the task is object detection.
- `mode=train`: Indicates that the model should be in training mode.
- `model=yolov8n.pt`: Specifies the model to use for training, in this case, the YOLOv8 nano model, which is a lightweight version. This model is trained on the COCO dataset and is the fastest among the other pre-trained models. We can also train our model from scratch if we want.
- `data="../new dataset/data.yaml"`: Points to the dataset configuration file (`data.yaml`) that contains information about the dataset, such as paths to the training and validation images and class labels.
- `epochs=30`: Specifies the number of training epochs, which is the number of times the training algorithm will work through the entire training dataset.
- `imgsz=640`: Sets the image size to 640 pixels. This means images will be resized to 640x640 pixels before being fed into the model.
- `batch=16`: Sets the batch size to 16, meaning 16 images will be processed in each iteration of training.

3. After training the model, we look at the different model metrics to get an idea of the model’s performance. The model metrics are stored in the `runs` folder and can be viewed in the Jupyter Notebook. We can also perform a model validation using a similar command as mentioned below:

```python
# Validating the model on the best.pt model:
!yolo val model='../new dataset/data.yaml" imgsz=640 batch=16 conf=0.4
```

4. After training and validating the model, the next step is to test the model on the test dataset. Now, we can do this in two ways:

```python
# Using CLI:
!yolo task=detect mode=predict model='../best.pt' conf=0.5 source='Sample media/sample_video.mp4' save=True

# Using Python command:
from ultralytics import YOLO
model = YOLO('../best.pt')
model.predict('sample/sample_video_1.mp4', save=True, conf=0.55, agnostic_nms=True)
```

The above steps can be performed to make predictions on the test dataset or on any media file. Using the [crop image](modules/crop%20image.ipynb) file, we can also extract the cropped images from the predictions based on the classes specified.

5. The model is now ready for deployment and used for other applications. [Safety Helmet Detection Using Webcam](Safety%20Helmet%20Detection%20using%20Webcam.py) is designed to detect safety helmets in real-time video streams using a pre-trained YOLOv8 model. The script captures video from a webcam, processes each frame to detect whether workers are wearing safety helmets, and displays the results with annotations. In case, it finds workers without a safety helmet, it displays a warning message on the screen.


**Note:**
- In case, you want to train the model on your own custom dataset, along with following the above steps mentioned, you can use [xml2yolo](modules/xml2yolo.ipynb) file to convert the files to yolo format in case the source dataset is in xml format and can use [split dataset](modules/split_dataset.ipynb) and [Changing names](modules/Changing%20names.py) to split the dataset into train, test and validation datasets and changing the names of the data files respectively.

- The complete code for training and validating the model can be found in my [Kaggle Notebook](https://www.kaggle.com/code/archisman24/safety-helmet-detection-using-yolov8/notebook).

## RESULTS:📈

Many benchmark results and confusion matrices are automatically created by the YOLOv8 model on completion of training and validation. These results can be found in the `runs` folder. We will look and analyze these benchmarks in the following section :

1. **Confusion Matrix :** The normalised confusion matrices for the models trained on different epochs are shown below: (shown for validation of models)
<div style="display: flex; justify-content: space-around; align-items: center; text-align: center;">

<div style="margin: 10px;">
    <img src="runs/detect/val/30_Epochs/confusion_matrix_normalized.png" alt="Confusion Matrix for 30 Epochs" width="500" title="Confusion Matrix for 30 Epochs">
    <h3>Confusion Matrix for 30 Epochs</h3>
</div>

<div style="margin: 10px;">
    <img src="runs/detect/val/50_Epochs/confusion_matrix_normalized.png" alt="Confusion Matrix for 50 Epochs" width="500" title="Confusion Matrix for 50 Epochs">
    <h3>Confusion Matrix for 50 Epochs</h3>
</div>

<div style="margin: 10px;">
    <img src="runs/detect/val/100_Epochs/confusion_matrix_normalized.png" alt="Confusion Matrix for 100 Epochs" width="500" title="Confusion Matrix for 100 Epochs">
    <h3>Confusion Matrix for 100 Epochs</h3>
</div>

</div>






## AREAS OF APPLICATIONS:🔨