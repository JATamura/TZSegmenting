import json
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import REPO_PATH


def make_graph_from_final_training():
    raise NotImplementedError('Iteration data is not well formatted.')
    metrics = ['total_loss', 'segm_cls_agn_nms/AP']

    metric_iteration_dict = {}

    with open(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'metrics.json'), 'r') as f:
        lines = f.read().splitlines()
    copies = []
    for l in lines:
        line_dict = json.loads(l)
        if line_dict['iteration'] ==19:
            copies.append(line_dict)

    done_iters = []
    for l in lines:
        line_dict = json.loads(l)
        line_results = []
        for m in metrics:

            try:
                result = line_dict[m]
            except KeyError:
                result = np.nan
            line_results.append(result)
        iteration = line_dict['iteration']
        if iteration in done_iters:
           print(f'Duplicate iteration {iteration}')
        done_iters.append(iteration)
        try:
            line_results.append(line_dict['data_time'])
        except KeyError:
            line_results.append(np.nan)
        metric_iteration_dict[iteration] = line_results

    # make a dataframe from metric_iteration_dict
    plot_df = pd.DataFrame(metric_iteration_dict).T
    plot_df.columns =metrics + ['data_time']

    plot_df['iteration'] = plot_df.index
    # make a seaborn plot of the total_loss
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=plot_df, x='iteration', y='total_loss')
    plt.savefig(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'training_total_loss.png'))
    print(metric_iteration_dict)

def make_tables_from_results_json(metrics_json_path: str, test_dataset_summary_path: str):
    # base_/nms_ prefixes indicate outputs without/with non-max suppression
    # _raw_values indicate mean number of counts per image
    # _maes indicate mean absolute error per image
    # bbox/segm prefix indicates bbox/segmentation metrics
    # bbox and segm metrics with/without _cls_agn_nms suffix indicate use of non-max suppression.
    # for both segm and bbox with and without nms we have the following metrics:
    # - AP
    # - AP-Empty/nonviable/viable
    # - AP50
    # - AP75
    # - APl
    # - APm
    # - APs

    # These are based on COCO metrics described in https://cocodataset.org/#detection-eval
    # The terminology is annoying because 'AP' is used in a self-referential way.
    # Average precision is computed as the area under the precision recall curve (based on model confidence), for a given IoU threshold and given category.
    # The COCO 'AP' is the average precision, averaged over all categories (what is usually called mAP) and
    # averaged across multiple IoU thresholds (0.5:0.05:0.95)
    # (IoU indicates whether a prediction is a true positive or not, and higher thresholds are more strict).
    # Where the category is specified, this is just AP (averaged over IOUs as before) for the given category.
    # Where the IoU is specified, this is the average precision for a given IoU threshold averaged across categories.
    #  APl/m/s are APs for varied object sizes, which we won't pay much attention to here.

    # We also evaluate AP for a 'seed_class', which uses NMS and then evaluates the model as if all labels are just 'seed'.

    # AP indicates model across in both segmenting and classifying object instances.
    # This is a difficult metric to interpret, but is the focus for improving and comparing models.
    # As a reference, on COCO test-dev, the winners of the COCO 2015 and 2016 segmentation challenges achieved AP=24.6 and 29.2 repespectively.
    # While a version of Mask R-CNN achieved 37.1 in 2017 \cite{he_mask_2017}

    # Furthermore, we provide metrics to indicate how this translates to the actual seed counts. (MAE).
    # When compared to the counts in the underlying population, this provides a useful measure of the average under and over counting of seeds.

    with open(metrics_json_path, 'r') as f:
        data = json.load(f)

    ap_scores = [data['segm_cls_agn_nms']['AP-'+c] for c in ['Viable', 'Non-Viable', 'Empty']] + [data['segm_seed_class']['AP']]
    mae_scores = [data['nms_maes'][c] for c in ['viable', 'non_viable', 'empty', 'total']]

    out_df = pd.DataFrame([ap_scores, mae_scores]).T
    out_df.index = ['Viable', 'Non-viable', 'Empty', 'Seed class' ]
    out_df.columns = ['AP', 'MAE']

    summary_df = pd.read_csv(test_dataset_summary_path, index_col=0)
    out_df['Mean actual count per image'] = [summary_df.loc['mean', c] for c in ['viable', 'nonviable', 'empty', 'total']]
    out_df = out_df.round(2)
    out_df.loc['Overall (Avg across classes)'] = [round(data['segm_cls_agn_nms']['AP'],2), 'Not calculated', None]
    out_df.to_csv(metrics_json_path.replace('.json', '_readable_summary.csv'))


def main():
    # make_tables_from_results_json(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'final_evaluation_metrics.json'),
    #                               os.path.join(REPO_PATH, 'datasets', 'dataset1', 'seed_stats', 'post_quality_check', 'model_data',
    #                                            'test_stats_summary.csv'))
    make_graph_from_final_training()

if __name__ == '__main__':
    main()