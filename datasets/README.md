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
│   │   └───seed_stats
│   │   │   ├───pre_quality_check
│   │   │   │   ├───analysis_duplicate_1_stats.csv
│   │   │   │   ├───analysis_duplicate_2_stats.csv
│   │   │   │   ├───base_dataset_stats.csv
│   │   │   │   ├───model_data
│   │   │   │   │   ├───test_stats.csv
│   │   │   │   │   ├───train_stats.csv
│   │   │   │   │   ├───val_stats.csv
│   │   │   ├───post_quality_check
│   │   │   │   ├───analysis_duplicate_1_stats.csv
│   │   │   │   ├───analysis_duplicate_2_stats.csv
│   │   │   │   ├───base_dataset_stats.csv
│   │   │   │   ├───model_data
│   │   │   │   │   ├───test_stats.csv
│   │   │   │   │   ├───train_stats.csv
│   │   │   │   │   ├───val_stats.csv
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