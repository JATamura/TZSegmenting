from ultralytics import YOLO

if __name__ == "__main__":
    # data = "../../datasets/yolo/pre_qc/data.yaml"
    # model = YOLO("yolo11s-seg.pt")
    # model.train(data=data, epochs=100, imgsz=1600, max_det=800,
    #             patience=20, cos_lr=True, freeze=1, box=3, cls=1, agnostic_nms=True,
    #             dropout=0.5, name="pre_qc_yolo_model_11", exist_ok=True)

    # data = "../../datasets/yolo/post_qc/data.yaml"
    # model = YOLO("yolo11n-seg.pt")
    # model.train(data=data, epochs=100, imgsz=2000, max_det=800,
    #             patience=20, cos_lr=True, freeze=1, box=3, cls=1, agnostic_nms=True,
    #             dropout=0.5, name="post_qc_yolo_model_11", exist_ok=True)

    # data = "../../datasets/yolo/pre_qc/data.yaml"
    # model = YOLO("yolo12n-seg.pt")
    # model.train(data=data, epochs=100, imgsz=1600, max_det=800,
    #             patience=20, cos_lr=True, freeze=1, box=3, cls=1, agnostic_nms=True,
    #             dropout=0.5, name="pre_qc_yolo_model_12", exist_ok=True)
    #
    data = "../../datasets/dataset1/yolo/post_qc/data.yaml"
    model = YOLO("yolo11n-seg.pt")
    model.train(data=data, epochs=200, imgsz=1600, max_det=800,
                patience=50, cos_lr=True, freeze=1, box=3, cls=1, agnostic_nms=True,
                dropout=0.5, name="post_qc_yolo_model_1", exist_ok=True)