# AI-Based Yoga Pose Detection and Correction
This project is a Python-based implementation of an AI system that can detect and correct yoga poses using landmark information extracted from images or videos. The system is built using the MediaPipe library, which provides a pre-trained machine learning model for pose estimation. The goal of this project is to provide a tool that can help yoga practitioners improve their form and alignment during their practice.

## Features
Landmark extraction: The system uses the MediaPipe Pose model to extract landmarks from images or videos of yoga poses.
Pose detection: The system compares the extracted landmarks to a database of correct yoga poses to identify any errors or discrepancies in the pose.
Pose correction: Once errors are identified, the system provides real-time feedback on how to correct the pose.
Visualization: The system provides visual feedback by drawing lines and dots on the image or video to show the detected landmarks and correct alignment.
Installation

## To run this project, you will need to install the following dependencies:
Python 3.x \
MediaPipe library \
OpenCV library \
Numpy library \
Flask library \
Tensorflow 

## Limitations
It's important to note that the accuracy of the system depends on the quality of the input image or video and the accuracy of the landmark extraction process. Additionally, while the system can detect basic errors in the pose, it may not be able to detect more subtle errors or provide nuanced feedback on the correct alignment of the body during a yoga pose. Therefore, it's important to use the system in conjunction with feedback from a yoga instructor or other sensor-based systems.

## Acknowledgements
This project was inspired by the work of various researchers and developers in the field of AI-based yoga pose detection and correction. Special thanks to the MediaPipe development team for providing a powerful and easy-to-use library for pose estimation.




