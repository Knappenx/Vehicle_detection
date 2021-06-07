# Vehicle_detection
This is the final project I made for my Master's degree class of Image Processing. You can see it in action in the link below.

https://youtu.be/B4WH9g6KwDA

## Prerequisites
- Python 3.7 (NOTE: Check numpy and opencv compatibility for other Python versions)
- Numpy 1.19.5
- OpenCV 4.5.1.48

## Project Description

The main goal of this project was to count vehicles on the road using image processing techniques only, even though there might be easier and more accurate approaches such as trained AI.

Counting all the vehicles is difficult with this approach as there are different factors that complicate the project, such as shadows, cars with similar colours to the road and background movement.

The region of interest allows the user to choose an area where the vehicles would be viewed correctly. Only objects going through this region will be counted.

The mask selection helps to keep the image comparison within the selected area, while trying to avoid background noise such as trees or shadows.

The result shows an approximate number of objects going through the selected region of interest.
