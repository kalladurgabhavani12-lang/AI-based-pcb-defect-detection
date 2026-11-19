import os
import shutil
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# Defect classes matching your abstract & PKU PCB dataset
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

RAW_IMAGES_DIR = os.path.join("PCB_DATASET", "images")
RAW_ANNOTATIONS_DIR = os.path.join("PCB_DATASET", "Annotations")
YOLO_DIR = "yolo_pcb_data"

def convert_xml_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip().lower()
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES.index(cls_name)
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines

def prepare_yolo_dataset():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    pairs = []
    for root_dir, _, files in os.walk(RAW_IMAGES_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root_dir, f)
                rel_folder = os.path.basename(root_dir)
                xml_name = os.path.splitext(f)[0] + ".xml"
                xml_path = os.path.join(RAW_ANNOTATIONS_DIR, rel_folder, xml_name)
                if os.path.exists(xml_path):
                    pairs.append((img_path, xml_path))

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    splits = {"train": pairs[:split_idx], "val": pairs[split_idx:]}

    for split_name, items in splits.items():
        for img_path, xml_path in items:
            tree = ET.parse(xml_path)
            size = tree.getroot().find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            lines = convert_xml_to_yolo(xml_path, w, h)
            if not lines:
                continue

            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(YOLO_DIR, "images", split_name, base_name))
            label_file = os.path.splitext(base_name)[0] + ".txt"
            with open(os.path.join(YOLO_DIR, "labels", split_name, label_file), "w") as out:
                out.write("\n".join(lines))

    yaml_content = f"""path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open("pcb_data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Prepared {len(pairs)} annotated PCB images for YOLOv8 training.")

if __name__ == "__main__":
    prepare_yolo_dataset()
    model = YOLO("yolov8n.pt")
    model.train(data="pcb_data.yaml", epochs=30, imgsz=640, batch=16, name="pcb_yolov8")
    best_weights = os.path.join("runs", "detect", "pcb_yolov8", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "best_pcb_yolo.pt")
        print("Training complete! Saved model as 'best_pcb_yolo.pt'.")
