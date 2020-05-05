# CS 445
## CS 445 Computational Photography - Final project

### Purpose
The project captures user's image from the webcam using face detection.
It automatically converts the captured image into a cartoonized image.
As the image is converted it will be saved at each step during the conversion process.
#
### Execution Steps
You can run this by downloading the repository and running the `imageToCartoonConverter.py` file by using the command

`python imageToCartoonConverter.py
`

To capture the screen you can hit/hold "c" on the keyboard to terminate the program and it will capture
the image and convert to cartoonize image as the program terminates.
The captured and converted images will be saved in the same directory.

To terminate the program you can hit/hold "q" on the keyboard and it will terminate the program without
saving any image.

#
Individual cartoonization programs can be run by executing the main block in each of the 
programs or by executing `python imageCartoonizer.py` and `python imageCartoonizerKMeans.py` command.
They will take existing `capturedRawImage.jpg` image as input from the same source code folder.

#
### Results
The image captured using face detection is stored as `capturedRawImage.jpg`

The program uses 2 different approaches to cartoonize the image. Approach 1 final image is 
stored as `imageCartoonizer-ConvertedImage.jpg`.  Approach 2 final image is stored as 
`imageCartoonizerKMeans-ConvertedImage.jpg`
During the workflow at each step intermediate results are saved in the same folder, they are named according
to the step being executed.
