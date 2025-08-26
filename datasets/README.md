Before running prepare_data.py:
````
├───datasets
│   ├───dataset1
│   │   ├───all_images
│   │   ├───coco_format
│   │   │   ├───pre_quality_check
│   │   │   │   ├───raw_data
│   │   │   ├───post_quality_check
│   │   │   │   ├───raw_data
````

Final directories:
````
├───datasets
│   ├───dataset1
│   │   ├───all_images
│   │   ├───coco_format
│   │   │   ├───pre_quality_check
│   │   │   │   ├───analysis_duplicate_1.json
│   │   │   │   ├───analysis_duplicate_2.json
│   │   │   │   ├───base_dataset.json
│   │   │   │   ├───model_data
│   │   │   │   │   ├───test.json
│   │   │   │   │   ├───train.json
│   │   │   │   │   ├───val.json
│   │   │   │   ├───raw_data
│   │   │   ├───post_quality_check
│   │   │   │   ├───analysis_duplicate_1.json
│   │   │   │   ├───analysis_duplicate_2.json
│   │   │   │   ├───base_dataset.json
│   │   │   │   ├───model_data
│   │   │   │   │   ├───test.json
│   │   │   │   │   ├───train.json
│   │   │   │   │   ├───val.json
│   │   │   │   ├───raw_data
│   │   └───yolo_format
│   │       ├───post_quality_check
│   │       │   ├───images
│   │       │   │   ├───test
│   │       │   │   ├───train
│   │       │   │   └───val
│   │       │   └───labels
│   │       │       ├───test
│   │       │       ├───train
│   │       │       └───val
│   │       └───pre_quality_check
│   │           ├───images
│   │           │   ├───test
│   │           │   ├───train
│   │           │   └───val
│   │           └───labels
│   │               ├───test
│   │               ├───train
│   │               └───val
````