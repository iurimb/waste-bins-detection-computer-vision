# waste-bins-detection-computer-vision

## Source video used
https://www.youtube.com/watch?v=39ysPwVOetg 

Custom dataset was extracted from the first 30 minutes of video and testing was done in the rest of it. 

## Objectives and general comments  
Waste bin detection project. Detecting, tracking and classifying bins into "in conformity" and "out of conformity". Custom dataset. Using Yolov8, ByteTrack and Supervision. Generate 2 PDF's with information on waste bins disconformities along the ride. 1 PDF with information about the points of video where disconformities were found and another with pictures as proof. 

The project started, at first, merely as waste-bin detection. This proved to be easily achievable with a relatively small dataset. It then progressed to find out the waste bins that are in disconformity, that is, waste bins which have its content coming out of its top, with unclosed lids etc. **This is a more challenging and interesting task**, given that the original video does not have many instances of disconform waste bins. So we have class imbalance coupled with a small dataset. I used my previous trained network to annotate the first 30 minutes of the source video, extracting one frame every 10 frames. I then revised the dataset and added the "disconform" label to the project.

The source code provided aims, then, to extract the information about disconform waste bins from the video and generate, at the end, two PDF files with the main findings. 

**You can make the model class agnostic to have a regular waste-bin-detection network**

## Disclaimers 
The code is functionable, but not yet organized, so I also couldn't yet organize the python environment. You might note (until I solve this) that there are many commented lines in the codes, and even parts of it that are 'outdated' (for example, some polygon shapes not being used). 

The dataset used for the entirety of this project is the video in "Source Video Used", above.

## Folders 

- disconformity_detection: folder containing one python script to detect, track and count disconform waste bins. The network has the capacity to detect the conform waste-bins, but this class is being filtered out. There's a development to be made, if the conform waste-bin class is readded to the mix, regarding utilizing different confidence thresholds for the two classes (example: 85% conf for conform waste bins and 25% conf for disconform)
- results: folder with a short video representing the output at a given situation. 
- training_loop: script to train a yolo model for object detection.
- Weights: trained model weights for detecting conform and disconform waste-bins. 
- line_zone: Line_Zone code I use that contains a modified function for annotating info. You could substitute the original line_zone file with mine or only copy the modified method code into yours ("_annotate_anything_count"). Alternatevely, you can give it a different name and import it. 

## About the task and its development

The difficulty lies mainly in the fact that the class imbalance is steep and there are not many examples in the original video of disconform waste bins. Without having a larger dataset, the job of annotating the dataset becomes paramount. There are project decisions to be made, such as, how many conform examples to leave on, were to start detecting waste bins and, mainly, how to deal with the annotation of disconform examples in case of occlusion, for example. Also, there's an argument to be made that focusing in the waste bins LIDS might be better for disconformity detection, since that's the part where the difference really lies in. Maybe three classes would suffice: "waste bin", "conform waste bin lid" and "disconform waste bin lid". I haven't tried that approach yet, but it's been in my mind and it's worth mentioning. 

I have also tried cropping out detected waste bins (without distinction) and classifying those bins in conformity and disconformity classes afterwards. Even though it showed promise, the resolution of cropped images was an issue. Besides, whenever you are doing detection on a video, but classification on a photo extracted from it, you create an issue, since the same object is detected in multiple frames under different conditions. That requires either that the classification dataset has to be really representative (many pictures from many angles), or some sort of filtering of detections to select the cropped photos to evaluate. So I moved on from that idea in this project for now.  

Many developments were made until we ended up with this particular project and solution, so I find useful to share those. 


## Running the codes
Main codes are found in the "training_loop" and "disconformity_detection" folders. 
The training_loop code is easily executed by just inserting your 'data.yaml' file_path into the "data" variable. 

To execute disconformity_detection code, plug in the model_weight and the video_path (I downloaded it from youtube).

- model_weight = "INSERT_WEIGHTS_PATH"
- model = YOLO('model_weight')
- VIDEO_PATH = "INSERT_VIDEO_PATH"

That should do it. 


### Some final observations 
#### 1) I used a modified version of the annotator method in "LineZoneAnnotator" to display information on the videos. I uploaded my "Line_Zone" file together with the code. 
#### 2) I did not have the time to adequately organize the code, I reckon it is a bit messy and there's work to be done there. I'm currently working on refactoring the personal projects I upload to github. 
