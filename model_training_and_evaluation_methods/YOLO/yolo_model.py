from ultralytics import YOLO

def main():
    data = "../../datasets/dataset1/yolo_format/post_quality_check/data.yaml"
    model = YOLO("yolo11n-seg.pt")
    model.train(data=data, epochs=200, imgsz=1600, max_det=800,
                patience=50, cos_lr=True, freeze=1, box=3, cls=1, agnostic_nms=True,
                dropout=0.5, project="model_weights", name="post_qc_yolo_model_1", exist_ok=True)

if __name__ == "__main__":
    main()