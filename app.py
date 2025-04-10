import cv2
import argparse
import json
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO
import torch
# Before processing
torch.cuda.empty_cache()

CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11]
CLASS_NAMES = ['2-axis', '3and4-axis', '5-axis', '6andmore-axis', 'BRT', 'bicycle', 'bus', 'car', 'microbus', 'motorbike', 'scooter', 'sitp']
MODEL_CONF = 0.493
MODEL_IOU = 0.5
VIDEOPATH=""
LINES = []
model = YOLO("best.pt")
tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()
smoother =sv.DetectionsSmoother()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=60)
lineZones =[]
lineZoneAnnotators = []


def readJsonFile(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def processIntersection(data, currentIntersection, currentCamera,video_file_path):
    intersections = data.get("intersections",[])
    intersectionProcessed = False
    for intersection in intersections:
        name = intersection.get("name",'')
        if currentIntersection in name:
            intersectionProcessed = True
            return processCameras(intersection, currentCamera, video_file_path)    
    if intersectionProcessed == False:
        processFrameAnnotator(
            POLYGON=np.array([]),
            lineZones=np.array([]),
            lineZoneAnnotators=np.array([]),
            video_file_path=video_file_path)
        return []       
            
def processCameras(intersection, currentCamera, video_file_path):
    cameras = intersection.get("cameras",[])
    cameraProcessed = False
    for camera in cameras:
        cameraName = camera.get("name",'')
        if currentCamera in cameraName:
            cameraProcessed = True
            return processLines(camera, video_file_path)            
    if cameraProcessed == False:
        processFrameAnnotator(
            POLYGON=np.array([]),
            lineZones=np.array([]),
            lineZoneAnnotators=np.array([]),
            video_file_path=video_file_path)
        return []
            
def processLines(camera,video_file_path):
    POLYGON = np.array(camera.get("polygon",[]), dtype= np.int32)  

    lines = np.array(camera.get("lines",[]))
    
    for line in lines:
        lineStart = sv.Point(line.get('start1',0),line.get('start2',0))
        lineEnd = sv.Point(line.get('end1',0),line.get('end2',0))
        lineZone = sv.LineZone(
                lineStart,
                lineEnd,
                triggering_anchors= (sv.Position.CENTER,)
                )

        lineZoneAnnotator = sv.LineZoneAnnotator(
                                text_scale=0.8,
                                text_orient_to_line=True,
                                display_in_count= line.get('in_enabled', True),
                                custom_in_text= line.get('custom_in_text', 'in'),
                                display_out_count= line.get('out_enabled', True),
                                custom_out_text=line.get('custom_out_text', 'out')
                            )
        lineZones.append(lineZone)
        lineZoneAnnotators.append({"lineCounter": lineZone,"lineZones":[] ,"lineZoneAnnotator": lineZoneAnnotator})
    lineZoneAnnotators.append({"lineCounter":{},"lineZones":lineZones ,"lineZoneAnnotator": sv.LineZoneAnnotatorMulticlass(
                                    text_scale=0.5,
                                    text_thickness=2,
                                    table_margin=20,                                    
                                )})
    processFrameAnnotator(POLYGON, lineZones, lineZoneAnnotators, video_file_path)
    return lines

    
def processFrameAnnotator(POLYGON, lineZones, lineZoneAnnotators, video_file_path):
    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)
    for frame in frame_generator:
        result = model(frame, device="cuda", verbose= False, conf=MODEL_CONF, iou = MODEL_IOU)[0]
         
        detections = sv.Detections.from_ultralytics(result)

        if POLYGON.any():
            polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors= (sv.Position.CENTER,))   
            detections = detections[polygon_zone.trigger(detections)]

        #detections = detections[np.isin(detections.class_id, CLASSES)]
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)    
        

        labels = [
            f"{CLASS_NAMES[class_id]} #{tracker_id} {confidence:.2f}"
            for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]

        lineZone:sv.LineZone
        for lineZone in lineZones:
            lineZone.trigger(detections= detections)

        annotated_frame = frame.copy()

        if POLYGON.any():
            annotated_frame = sv.draw_polygon(
                scene= annotated_frame,
                polygon=POLYGON,
                color= sv.Color.RED,
                thickness= 2
                )
        annotated_frame = box_annotator.annotate(
            scene= annotated_frame,
            detections = detections
        )
        annotated_frame = label_annotator.annotate(
            scene= annotated_frame,
            detections = detections,
            labels= labels
        )
        annotated_frame = trace_annotator.annotate(
            scene= annotated_frame,
            detections = detections
        )
        for  lineZoneAnnotator in lineZoneAnnotators:
            if lineZoneAnnotator.get("lineZones",[]) != []:
                zoneAnottator:sv.LineZoneAnnotatorMulticlass = lineZoneAnnotator.get("lineZoneAnnotator",{})
                lineZones =  lineZoneAnnotator.get("lineZones",[])
                annotated_frame =  zoneAnottator.annotate(
                    annotated_frame,
                    line_zones = lineZones
                )
            else:        
                zoneAnottator:sv.LineZoneAnnotator = lineZoneAnnotator.get("lineZoneAnnotator",{})       
                lineCounter: sv.LineZone = lineZoneAnnotator.get("lineCounter",{})
                annotated_frame =  zoneAnottator.annotate(
                    annotated_frame,
                    line_counter = lineCounter
                )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def main(video_file_path):
    data = readJsonFile("intersections.json")
    currentCamera = Path(video_file_path).parent.name
    currentIntersection= Path(video_file_path).parent.parent.name
    return processIntersection(data, currentIntersection,currentCamera, video_file_path)

def processJsonResults(lines:np.array):
    if not lines.any():
        return "The configuration for the video was not found."
    result = {}
    lineZoneCounter = 0
    for lineZone in lineZones:        
        currentLine = lines[lineZoneCounter]
        if currentLine.get("in_enabled", True):
            result[currentLine.get("custom_in_text", "in")] ={}
        if currentLine.get("out_enabled", True):
            result[currentLine.get("custom_out_text", "out")] ={}
        for Class in CLASSES:                     
            if currentLine.get("in_enabled", True) and Class in lineZone.in_count_per_class.keys():
                modelName = str(lineZone.class_id_to_name[Class])
                result[currentLine.get("custom_in_text", "in")][modelName] = lineZone.in_count_per_class[Class]
            if currentLine.get("out_enabled", True) and Class in lineZone.out_count_per_class.keys():
                modelName = str(lineZone.class_id_to_name[Class])
                result[currentLine.get("custom_out_text", "out")][modelName] = lineZone.out_count_per_class[Class]
        lineZoneCounter= lineZoneCounter +1
    return json.dumps(result)
       

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--video_file_path")
    args = parser.parse_args()

    lines = main(args.video_file_path)
    print(processJsonResults(lines))
    