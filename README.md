# TZSegmenting

This repository contains the scripts, data and outputs related to instance segmentation of images of stained orchid seeds to count the number of
viable, non-viable and empty seeds. You can find a project
overview [here](https://www.kew.org/science/our-science/projects/machine-learning-to-improve-orchid-viability-testing).

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
an [app](https://huggingface.co/spaces/TZProject/OrchAId) for uploading images and running the model.

#### Dataset

The dataset for training, validation and testing was built by first collecting and imaging stained orchid seeds. The protocol for taking these images
is available [here](https://pgomba.github.io/orchid_protocol/). Next, images were labelled by experts using [CVAT](https://www.cvat.ai/) to segment
viable, non-viable and
empty seeds in the images. A subset of the images were annotated by multiple annotators in order to analyse the agreement of annotators on seed
classifications in TZ tests. We find that there is a significant level of disagreement in these decisions, indicating a level of ambiguity in TZ tests
that we hope to provide a more detailed analysis of in a publication soon.

#### Results

A summary of model performance on the final test set can be found in this repository
at [metrics_readable_summary.csv](model_weights/final_tz_segmentor/metrics_readable_summary.csv).

To briefly summarise these results, first we focus on the AP score. AP indicates model performance in both segmenting and classifying object
instances. This is a difficult metric to interpret, but is the focus for improving and comparing models. Our model achieves an AP of 31.61 in the
final evaluation.
As a reference, on the COCO test-dev dataset, the winners of the COCO 2015 and 2016 segmentation challenges achieved AP=24.6 and 29.2, respectively;
while an early version of Mask R-CNN achieved 37.1 in
2017 [(He et al.)](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html).

We also calculated mean absolute error (the absolute counting error, averaged across images) for the models in order to provide an indication of the
number of over or undercounting of seeds that one may expect from the model on each image. We can see that in the test set there is an average of
116.9 seeds per image, and the model on average over or under counts
seeds in total by 5.7. We distinguish the over and undercounting associated with each class as well. As a proportion of the total number we see that
viable seeds have the most error (MAE = 1.1 compared to 4.9 viable seeds per image), while non-viable seeds have a relatively small error (MAE = 6.23
compared to 94.35 non-viable seeds per image).

### Acknowledgements

The OrchAId TZ viability dataset was developed by the Royal Botanic Gardens, Kew, Silo National des Graines Forestieres, Madagascar, the Ministry of
Agriculture, Lands, Housing and Environment, Monsterrat, Instituto de Investigação Agrária de Moçambique, Mozambique, Departmento de Recursos
Naturales y Ambientales, Puerto Rico & the National Parks Trust of the Virgin Islands.

The developers acknowledge Research Computing at the James Hutton Institute for providing computational resources and technical support for the 'UK’s
Crop Diversity Bioinformatics HPC' (BBSRC grants BB/S019669/1 and BB/X019683/1), use of which has contributed to the development of the model used in
this analysis.

### Licence

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg