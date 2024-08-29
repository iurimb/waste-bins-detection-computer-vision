import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from supervision import Position
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
import os
#from ultralytics import YOLOv10
import pandas as pd
# Importing datetime 
import datetime 
import time
from time import perf_counter
from timeit import default_timer as cronometro
from fpdf import FPDF

# importing whole module
#from tkinter import *
#from tkinter.ttk import *
import datetime 
# importing strftime function to
# retrieve system's time
from time import strftime
#import time
from datetime import timedelta
import math 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = os.getcwd()
annotator = sv.LineZoneAnnotator()
model_weights = "YOUR_MODEL_WEIGHTS_PATH"
model = YOLO(model_weights)
VID_TEST = 'YOUR_VIDEO_PATH'
classes_dict = {0: 'disconformity', 1: 'conformity'}
df_id_list = []
df_start_frame_list = []
df_end_frame_list = []

desconforme_dataframe_dict = {'ID': df_id_list, 'START_FRAME': df_start_frame_list, 'END_FRAME': df_end_frame_list}
desconforme_images_list = []
desconforme_images_id = []

print(model.names)



frames_generator = sv.get_video_frames_generator(VID_TEST, start = 55200, stride=10)


frame_counter = 0
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
lost_track_buffer = 10
minimum_matching_threshold = 0.95
track_act_thresh = 0.25
min_consec_frames = 2
vcap = cv2.VideoCapture(VID_TEST)
width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vcap.get(cv2.CAP_PROP_FPS)
tracker = sv.ByteTrack(track_activation_threshold=track_act_thresh, lost_track_buffer=lost_track_buffer, frame_rate=fps, minimum_matching_threshold=minimum_matching_threshold, minimum_consecutive_frames=min_consec_frames)

#input(fps)
 # choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#video = cv2.VideoWriter('CONF_DISCONF_DEF.mp4', fourcc, fps, (int(width), int(height)))
heat_map_annotator = sv.HeatMapAnnotator()
COLORS = sv.ColorPalette.DEFAULT
length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
#input(length)

polygon = np.array([
    [0, 211], #top_left
    [168, 216], #top_right
    [168, 480], #bottom_right
    [0, 480] #bottom_left
])

points = np.array([
    [0, 211], #top_left
    [168, 216], #top_right
    [168, 480], #bottom_right
    [0, 480] #bottom_left
], np.int32)

polygon_special = Polygon([[0, 211], [168, 216], [168, 480], [0, 480]])

detections_list = []

polygon = sv.PolygonZone(polygon=polygon)
lost_tracks_list = []
out_count = 0 
in_count = 0
#index_count = 1
zone_annotator = sv.PolygonZoneAnnotator(zone=polygon, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)
smoother = sv.DetectionsSmoother()

center_and_ids_list = []
last_id_in_zone = None

for frame in frames_generator:
    frame_counter = frame_counter + 1
    print(frame_counter)
    #result = model.track(frame, persist=True)[0]
    #result = model.predict(frame, conf=0.6, classes=[0,1], save_crop=True)[0]
    result = model.predict(frame, classes=[0])[0]
    print("REMOVED", tracker.removed_tracks)
    #print("TRACKED", tracker.tracked_tracks)
    #print('tracked, lost, removed', tracker.tracked_tracks, tracker.lost_tracks, tracker.removed_tracks)
    #print("REMOVED", tracker.removed_tracks)
    #print("aqui", result.boxes.conf, result.boxes.cls)  # confidence scores
        # class
    #print('ola?', result.probs)
    #keypoints_2d = result.keypoints.xy.int().numpy()
    #im_array = result.plot(labels=False, boxes=False, probs=False) 

    #labels (bool): Whether to plot the label of bounding boxes.
            
    #IF WANNA PLOT
    # plot a BGR numpy array of predictions
    #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    ######

    #input("Parou")
    #tensor / [number_of_detections, number_of_keypoints, keypoints_XY_coordinates] / keypoints 2d tensor.
    #print(type(keypoints_2d), keypoints_2d.shape, keypoints_2d)
    
    #print(keypoints_2d[0, 0].tolist())
    #print(type(keypoints_2d[0, 0].tolist()))
    
    #input("PAROU")
    detections = sv.Detections.from_ultralytics(result)
    #detections = detections[detections.confidence > 0.5 if detections.class_id == 0 else detections.confidence >]
    detections = tracker.update_with_detections(detections) 

    #save original_img to crop detections
    orig_img = result.orig_img
    #print("SEU TESTE AQUI", len(detections))
    #detections = smoother.update_with_detections(detections)
    
    #input(detections)
#results = model.track(source=CCTV, persist=True, stream=True, show=True)
    if detections.tracker_id.any():
        labels_tracker = [
        f"#{tracker_id} {class_name} {confidence:.2f}"
        for tracker_id, class_name, confidence
        in zip(detections.tracker_id, detections['class_name'], detections.confidence)
        ]

    if detections:
        labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
        ]
        
    
    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    
    #cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)
    #annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=points, color=COLORS.colors[1])
    #annotated_frame = zone_annotator.annotate(scene=annotated_frame)
    #annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=polygon, color=COLORS.colors[0])
    #input(type(detections))
    track_ids_buffer = []
    #tracked_tracks; lost_tracks and removed_tracks -> get from tracker 
    #get from tracker self.track_id, self.start_frame, self.end_frame
    
    if len(detections) > 0 and len(labels_tracker) == len(detections): #and len(detections) is not None:#len(detections.tracker_id):
        #print("aqui", detections.pred)
        #print(len(labels_tracker), len(detections.tracker_id), len(detections))
        #tracked_tracks = input(tracker.tracked_tracks)
        #input(type(detections.tracker_id))
        #self.track_id, self.start_frame, self.end_frame
        print(detections)
        #input('see')
        
        boxes = result.boxes.xyxy.cpu()
        track_ids = detections.tracker_id.tolist()
        clss = result.boxes.cls.cpu().tolist() 
        confs = result.boxes.conf.float().cpu().tolist() 
        print("aqui", clss, confs)

        for box, track_id, cls in zip(boxes, track_ids, clss):
            #input("aqui?")
            #assumindo que só tem uma
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
            center_id_tuple = (bbox_center, track_id)
            center_and_ids_list.append(center_id_tuple)

            if track_id not in desconforme_images_id:
                desconforme_images_id.append(track_id)
                desconforme_images_list.append(Image.fromarray(np.array(annotated_frame)))
        #for bbox_center, center_id in center_and_ids_list:
        #    if polygon_special.contains(Point((bbox_center[0], bbox_center[1]))):
        #        annotator._annotate_anything_count(annotated_frame, sv.Point(120, 150), f"IN ZONE: 1, ID {center_id}")
            
        #center_and_ids_list.clear()
        #last_id_in_zone = center_id
        #display keypoints
        #para cada pessoa detectada
        
        #VER ISSO
        
        #print(len(labels_tracker), len(detections.tracker_id), len(detections))
        
        '''
        for i in range(keypoints_2d.shape[0]):
            #para cada keypoint
            #for j in range(keypoints_2d.shape[1]):
            annotated_frame = cv2.drawKeypoints(annotated_frame, keypoints_2d[i], 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
  


        for i in range(keypoints_2d.shape[0]):
            #para cada keypoint
            for j in range(keypoints_2d.shape[1]):

            #get points as ints
                keypoint_x = round(keypoints_2d[i, j, 0].item())
                keypoint_y = round(keypoints_2d[i, j, 1].item())
                keypoint = [keypoint_x, keypoint_y]

            #desenha o keypoint
            #round(a), round(b)
                #keypoints_2d[i, j] = [(int(x), int(y)) for x,y in keypoints_2d[i,j]]
                annotated_frame = cv2.circle(annotated_frame, keypoint, 1, color=(255, 0, 0), thickness=1)
        '''
        '''
        for track in tracker.tracked_tracks:
            
        #for track in detections.tracker_id:
            #print(tracker.tracked_tracks, tracker.lost_tracks, tracker.removed_tracks, tracker.frame_id) #== track 
            #input("Para")
        #    if 
            #tracker.tracked_tracks[0].track_id, tracker.tracked_tracks[0].start_frame, tracker.tracked_tracks[0].end_frame,
            #time_detected = track.end_frame - track.start_frame
            if track.track_id not in detections_list and track.end_frame - track.start_frame > 30 and track.track_id not in track_ids_buffer: #descarta as menores
                if track.track_id == 2:
                    print("CHECA AQUI", track.end_frame, track.start_frame)
                #if track.track_id not in track_ids_buffer:
                track_ids_buffer.append(track.track_id)
                #ele aparece mais de uma vez e isso buga, tem que pegar só uma aparicao 
                #detections_list.append(track.track_id) 
                detections_list.append(track.track_id)
                #input(type(track.track_id))
                index_count = len(detections_list)
        '''
        

        track_ids_buffer = []
        '''
        for detected in detections_list:
            if detected not in detections.tracker_id and detected not in lost_tracks_list :
                out_count = out_count+1
                lost_tracks_list.append(detected)
        '''
        #if detections.tracker_id not in detections_list:
        #    detections_list.append(detections.tracker_id)
            #input(detections.tracker_id)
        #in_count = len(detections_list)
        #input(detections.tracker_id)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_tracker)
        #polygon.trigger(detections=detections)
        
        boxes = result.boxes.xyxy.cpu()
        track_ids = detections.tracker_id.tolist()
        clss = result.boxes.cls.cpu().tolist() 
    
        for box, track_id, cls in zip(boxes, track_ids, clss):
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

            #if polygon_special.contains(Point((bbox_center[0], bbox_center[1]))):
    elif(len(detections)>0 and len(detections.tracker_id) == 0):
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        #for box, cls in zip(boxes, clss):
        #    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
        #if polygon_special.contains(Point((bbox_center[0], bbox_center[1]))):
            #presumo que nesse caso o que está na zona é o mesmo que estava na zona
        #    annotator._annotate_anything_count(annotated_frame, sv.Point(120, 150), f"IN ZONE: 1, ID POTENCIAL: {last_id_in_zone}")

    #else:
        #input("aqui???")
        #annotator._annotate_anything_count(annotated_frame, sv.Point(120, 150), f"IN ZONE: 0")    
    #if tracker.lost_tracks:
    #if tracker.removed_tracks:
    #    input(tracker.lost_tracks[0])
    #    input((tracker.lost_tracks[0]))
    #    for removed in tracker.removed_tracks:
            #if track not in detections_list and track.end_frame - track.start_frame > 30:
                #detections_list.append(track.track_id) 

    #        if removed.track_id not in lost_tracks_list and removed.end_frame - removed.start_frame > 30:  #and losttrack.end_frame - losttrack.start_frame > lost_track_buffer:
                #print("OLHA PRA CA:", losttrack.end_frame, losttrack.start_frame)
    #            lost_tracks_list.append(removed.track_id)
    
    #out_count = len(lost_tracks_list)
    
    #annotator._annotate_anything_count(annotated_frame, sv.Point(120, 20), f"Entraram: {in_count}")
    #annotator._annotate_anything_count(annotated_frame, sv.Point(120, 50), f"Saida {out_count}")
    #annotator._annotate_anything_count(annotated_frame, sv.Point(120, 80), f"Entraram_IDS {detections_list}")
    #annotator._annotate_anything_count(annotated_frame, sv.Point(120, 110), f"Sairam_IDS {lost_tracks_list}")
        
    if tracker.removed_tracks: 
        for track in tracker.removed_tracks:
            if track.external_track_id not in df_id_list:
                df_id_list.append(track.external_track_id)
                print("EXTERNAL APPENDED", track.external_track_id)
                df_start_frame_list.append(track.start_frame)
                df_end_frame_list.append(track.end_frame)
    #video.write(annotated_frame)

                #input(df)
                #internal_track_id, self.start_frame, self.end_frame, int(self.class_ids)
 
    #cv2.imwrite("frame.jpg", frame)
    #break
    cv2.imshow("frame", annotated_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

df = pd.DataFrame(desconforme_dataframe_dict)
fig, ax = plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
pp = PdfPages("desconf_vid_information.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()

images_pdf = FPDF()
for image in desconforme_images_list:
    images_pdf.add_page()
    images_pdf.image(image)
images_pdf.output("desconf_vid_images.pdf")
#pdf_path = r'C:\Users\noz\Documents\Projeto\RESIDUOS\desconf_vid_images.pdf'
    
#desconforme_images_list[0].save(pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=desconforme_images_list[1:])
