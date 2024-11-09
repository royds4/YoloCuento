import cv2
import argparse
import json
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO

CLASSES = [1,2,3,5,7]
VIDEOPATH=""
model = YOLO("yolo11x.pt")
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
    for intersection in intersections:
        name = intersection.get("name",'')
        if currentIntersection in name:
            processCameras(intersection, currentCamera, video_file_path)
            return
            

def processCameras(intersection, currentCamera, video_file_path):
    cameras = intersection.get("cameras",[])
    for camera in cameras:
        cameraName = camera.get("name",'')
        if currentCamera in cameraName:
            processLines(camera, video_file_path)
            return
            
def processLines(camera,video_file_path):
    POLYGON = np.array(camera.get("polygon",[]), dtype= np.int32)  

    lines = camera.get("lines",[])
    
    for line in lines:
        lineStart = sv.Point(line.get('start1',0),line.get('start2',0))
        lineEnd = sv.Point(line.get('end1',0),line.get('end2',0))
        lineZone = sv.LineZone(
                lineStart,
                lineEnd,
                triggering_anchors= (sv.Position.BOTTOM_CENTER,)
                )

        lineZoneAnnotator = sv.LineZoneAnnotator(
                                text_scale=0.8,
                                text_orient_to_line=True,
                                display_in_count= True
                            )
        lineZones.append(lineZone)
        lineZoneAnnotators.append({"lineCounter": lineZone,"lineZones":[] ,"lineZoneAnnotator": lineZoneAnnotator})
    lineZoneAnnotators.append({"lineCounter":{},"lineZones":lineZones ,"lineZoneAnnotator": sv.LineZoneAnnotatorMulticlass(
                                    text_scale=0.8,
                                    text_thickness=2,
                                    table_margin=20
                                )})
    processFrameAnnotator(POLYGON, lineZones, lineZoneAnnotators, video_file_path)

    
def processFrameAnnotator(POLYGON, lineZones, lineZoneAnnotators, video_file_path):
    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)
    for frame in frame_generator:
        result = model(frame, device="cuda", verbose= False, imgsz = 1920, conf=0.3, iou = 0.7 )[0]
        polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors= (sv.Position.CENTER,))    
        detections = sv.Detections.from_ultralytics(result)
        detections[polygon_zone.trigger(detections)]
        detections = detections[np.isin(detections.class_id, CLASSES)]
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)    
        

        labels = [
            f"#{tracker_id}"
            for tracker_id 
            in detections.tracker_id
        ]

        lineZone:sv.LineZone
        for lineZone in lineZones:
            lineZone.trigger(detections= detections)

        annotated_frame = frame.copy()
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def main(video_file_path):
    data = readJsonFile("intersections.json")
    currentCamera = Path(video_file_path).parent.name
    currentIntersection= Path(video_file_path).parent.parent.name
    processIntersection(data, currentIntersection,currentCamera, video_file_path)

def processJsonResults():
    result = {}
    lineZoneCounter = 0
    for lineZone in lineZones:        
        lineZoneId = 'lineZone'+ str(lineZoneCounter) #TODO: Map lineZoneId with Movement Name
        result[lineZoneId] = {}
        for Class in CLASSES:                     
            if Class in lineZone.in_count_per_class.keys():
                modelName = str(lineZone.class_id_to_name[Class])
                result['lineZone'+ str(lineZoneCounter)][modelName] = lineZone.in_count_per_class[Class]
        lineZoneCounter= lineZoneCounter +1
    return json.dumps(result)
       

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--video_file_path")
    args = parser.parse_args()

    main(args.video_file_path)
    print(processJsonResults())
    