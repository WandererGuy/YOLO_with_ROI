import time 
import uuid
import yaml
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os 
import configparser
from datetime import datetime
import cv2
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_directory)
static_crop_folder = os.path.join(parent_dir,"static", "crop")
os.makedirs(static_crop_folder, exist_ok=True)

def load_model(custom_checkpoint_path):
    # custom_yaml_path = "1_686\YOLODataset\dataset.yaml"
    model = YOLO(custom_checkpoint_path)  # load a pretrained model (recommended for training)
    return model

with open("config/config.yaml", "r") as file:
    data = yaml.safe_load(file)
    CUSTOM_CHECKPOINT_PATH = data["CUSTOM_CHECKPOINT_PATH"]
YOLO_MODEL = load_model(CUSTOM_CHECKPOINT_PATH)
with open("config/ship_class.yaml", "r") as file:
    SHIP_CLASS_DICT = yaml.safe_load(file)

def save_crop_image(xyxy, image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to open: {image_path}")
    x1, y1, x2, y2 = xyxy
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite(save_path, cropped_image)
    
def extract_objects(batch_yolo_result, image_path):
    all_objects = []
    single_image_yolo_result = batch_yolo_result[0]
    for single_object_result in single_image_yolo_result: # loop through objects in 1 image 
        single_object_result_box = single_object_result.boxes
        xyxy = single_object_result_box.xyxy.cpu().numpy()[0]
        conf = single_object_result_box.conf.item() 
        object_cls = single_object_result_box.cls.cpu().tolist()[0]
        single_object = ItemYolo(xyxy = xyxy, 
                                 cls = object_cls, 
                                 conf = conf, 
                                 image_path = image_path)
        all_objects.append(single_object)

    return all_objects


def check_box_in_polygon(poly_coor_ls:list[tuple], xyxy):
    from matplotlib import path

    xmin, ymin, xmax, ymax = xyxy
    t = [(xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin)]
    p = path.Path(poly_coor_ls) 
    if False in p.contains_points(t):
        return False
    else:
        return True

# import matplotlib.pyplot as plt

# def visualize_polygon(image_path, poly_coor_ls):
#     # Load the image from the file path
#     image = plt.imread(image_path)
    
#     # Create a figure and display the image
#     plt.imshow(image)
    
#     # Splits the list of tuples into two separate tuples for x and y coordinates.
#     x, y = zip(*poly_coor_ls)
    
#     # Overlay the polygon on the image with a red edge and no fill
#     plt.fill(x, y, edgecolor='red', fill=False, linewidth=2)
    
#     # Ensure the axes have an equal scale
#     plt.axis('equal')
    
#     # Show the plot
#     plt.show()

def visualize_polygon(image_path, poly_coor_ls):
    image = cv2.imread(image_path)
    # Polygon corner points coordinates
    pts = np.array(poly_coor_ls,
                np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color = (255, 0, 0)
    thickness = 2
    image = cv2.polylines(image, [pts], 
                        isClosed, color, thickness)
    # Displaying the image
    cv2.imshow("Polygon", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


class ItemYolo():
    def __init__(self,
                 xyxy, 
                 cls, 
                 conf, 
                 image_path):
        self.xyxy = xyxy
        self.cls = SHIP_CLASS_DICT[int(cls)]
        self.conf = round(conf*100, 2)
        self.image_path = image_path

class Camera():
    def __init__(self, camera_id, poly_coor_ls):
        self.camera_id = camera_id 
        self.poly_coor_ls = poly_coor_ls

class ItemReturn():
    def __init__(self, 
                 type_ship = None, 
                 arrival_state = None, 
                 camera_id = None, 
                 confidence = None, 
                 image_crop_path = None,
                 image_path = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
        self.type_ship = type_ship
        self.arrival_state = arrival_state
        self.camera_id = camera_id
        self.confidence = confidence
        self.image_crop_path = image_crop_path
        self.image_path = image_path

def annotate_image(image_path, poly_coor_ls, object_ls):
    image = visualize_polygon(image_path, poly_coor_ls)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for object in object_ls:
        xyxy = object.xyxy
        obj_class = object.cls
        conf = object.conf
        x1, y1, x2, y2 = xyxy

        final_pred_class = obj_class
        # Place text label at the top
        text_top = final_pred_class + " " + str(conf)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(image, text_top, (int(x1), int(y1) - 10), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Polygon", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from ROI_draw import draw_polygon

if __name__ == "__main__":
    # poly_coor_ls = [(135,256),(386,401),(305,493),(43,436)]
    # poly_coor_ls = [(401,334), (402,394), (763,447), (749,217)]
    image_path = r"D:\WORK\SHIP_project\ship_infer\unknown 2.jpg"
    poly_coor_ls = draw_polygon(image_path)
    # visualize_polygon(image_path, poly_coor_ls)
    camera = Camera(camera_id = 1, poly_coor_ls = poly_coor_ls)

    batch_yolo_result = YOLO_MODEL.predict(source=image_path, 
                                            save=False, 
                                            imgsz=640, 
                                            conf=0.30, 
                                            verbose = False, 
                                            iou=0.7)
    all_objects = extract_objects(batch_yolo_result, image_path)
    accept_objects = []
    final_res = []
    for object in all_objects:
        if check_box_in_polygon(poly_coor_ls, object.xyxy):
            save_path = os.path.join(static_crop_folder, str(uuid.uuid4()) + ".jpg")
            save_crop_image(object.xyxy, image_path, save_path)
            item = ItemReturn()
            item.type_ship = object.cls
            item.arrival_state = "arrived"
            item.camera_id = camera.camera_id
            item.confidence = object.conf
            item.image_crop_path = save_path
            item.image_path = image_path
            final_res.append(item)
            accept_objects.append(object)

    for item in final_res:
        print ("----------------------------------")
        for key, value in item.__dict__.items():
            print(key, value)

    annotate_image(image_path, camera.poly_coor_ls, accept_objects)