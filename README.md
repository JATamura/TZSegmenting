# TZSegmenting

This repository contains the scripts, data and outputs related to instance segmentation of images of stained orchid seeds to count the number of
viable, non-viable and empty seeds. The protocol for taking images compatible with this model is available
on [GitHub](https://pgomba.github.io/orchid_protocol/).

### Model Overview

We trained and evaluated a Mask R-CNN model to perform instance segmentation on our human-annotated OrchAId TZ viability dataset in order to identify
and count viable, non-viable and empty orchid seeds.

To develop and evaluate the models, the data was split into 60-20-20 train-validation-test sets. The data was split in this way by stratifying based
on the number of viable seeds and average seed size in the images. This was to help ensure both validation and test sets remained representative of
the dataset as a whole.

We compared a variety of architectures and parameters, and selected the best performing model based on evaluation on the validation data.

The final model, developed using the detectron2 framework, is based on the Cascade Mask R-CNN architecture and was trained on the training+validation
data and evaluated on the test set.
We hope to release full details of the development process in a publication soon.

The final trained model is publicly available on [HuggingFace](https://huggingface.co/TZProject/final_tz_segmentor), along with
an [app](https://huggingface.co/spaces/TZProject/TZSeedApp).

#### Results

A summary of model performance on the final test set can be found in this repository
at [metrics_readable_summary.csv](model_weights/final_tz_segmentor/metrics_readable_summary.csv).

To briefly summarise these results, the model achieves a mAP of 31.61. AP indicates model performance across in both segmenting and classifying object
instances. This is a difficult metric to interpret, but is the focus for improving and comparing models.
As a reference, on COCO test-dev, the winners of the COCO 2015 and 2016 segmentation challenges achieved AP=24.6 and 29.2, repespectively; while a
version of Mask R-CNN achieed 37.1 in 2017 \cite{he_mask_2017}.

We also calculated mean absolute error for the models in order to provide an indication of the amount of over or under counting of seeds that one may
expect from the model. We can see that in the test set there is an average of 116.9 seeds per image, and the model on average over or under counts
seeds in total by 5.7. We distinguish the over and undercounting asscoiated to each class as well, as a proportion of the total number we see that
viable seeds have the most error (MAE = 1.1 compared to 4.9 viable seeds per image).

### Acknowledgements

The OrchAId TZ viability dataset was developed by the Royal Botanic Gardens, Kew, Silo National des Graines Forestieres, Madagascar, the Ministry of
Agriculture, Lands, Housing and Environment, Monsterrat, Instituto de Investigação Agrária de Moçambique, Mozambique, Departmento de Recursos
Naturales y Ambientales, Puerto Rico & the National Parks Trust of the Virgin Islands.

### Licence

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg