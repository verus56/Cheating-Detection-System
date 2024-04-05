# YOLO v8 Cheating Detection System 

This repository contains Python code for real-time Cheating Detection System  using YOLO (You Only Look Once) v8 deep learning algorithm. The code is specifically tailored for implementing a Cheating Detection System.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy

## Usage
1. Clone this repository to your local machine.
2. Download the YOLO v8 model configuration file (`model.cfg`) and pre-trained weights file (`model.weights`) from the official YOLO website or other trusted sources.
3. Create a file named `labels.names` containing the class labels corresponding to the trained model.
4. Place the `model.cfg`, `model.weights`, and `labels.names` files in the same directory as the Python script.
5. Modify the `video_path` variable in the script to specify the path to the video file you want to perform object detection on.
6. Run the Python script.

```
python object_detection.py
```

## Parameters
- `minconf`: Minimum confidence threshold for object detection. Default value is 0.4.
- `minthresh`: Minimum threshold for non-maxima suppression to suppress weak, overlapping bounding boxes. Default value is 0.3.

## Functionality
- This code utilizes YOLO v8 for real-time object detection, specifically for implementing a Cheating Detection System.
- The system can be used for monitoring cheating behaviors in various contexts such as exams, games, or any scenario where cheating detection is required.
- Detected objects are labeled with bounding boxes and confidence scores.
- Press 'q' key to exit the application.

## References
- YOLO: You Only Look Once - [Paper](https://arxiv.org/abs/1506.02640)
- OpenCV Documentation - [OpenCV](https://docs.opencv.org/)
- NumPy Documentation - [NumPy](https://numpy.org/doc/)

Feel free to contribute to this repository by improving the code or adding new features. If you encounter any issues, please report them in the "Issues" section.
