import json
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample
import seaborn as sns

from utils import REPO_PATH

_test_dataset_summary_path = os.path.join(REPO_PATH, 'datasets', 'dataset1', 'seed_stats', 'post_quality_check', 'model_data',
                                          'test_stats_summary.csv')
_metrics_json_path = os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'final_evaluation_metrics.json')

_individual_image_results_path = os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor',
                                              'test_results_on_individual_images.csv')
_test_seed_stats_path = os.path.join(REPO_PATH, 'datasets/dataset1/seed_stats/post_quality_check/model_data/test_stats.csv')


def make_graph_from_final_training():
    raise NotImplementedError('Iteration data is not well formatted.')
    metrics = ['total_loss', 'segm_cls_agn_nms/AP']

    metric_iteration_dict = {}

    with open(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'metrics.json'), 'r') as f:
        lines = f.read().splitlines()
    copies = []
    for l in lines:
        line_dict = json.loads(l)
        if line_dict['iteration'] == 19:
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
    plot_df.columns = metrics + ['data_time']

    plot_df['iteration'] = plot_df.index
    # make a seaborn plot of the total_loss
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=plot_df, x='iteration', y='total_loss')
    plt.savefig(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'training_total_loss.png'))
    print(metric_iteration_dict)


def make_tables_from_results_json():
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

    with open(_metrics_json_path, 'r') as f:
        data = json.load(f)

    ap_scores = [round(data['segm_cls_agn_nms']['AP-' + c], 2) for c in ['Viable', 'Non-Viable', 'Empty']] + [round(data['segm_seed_class']['AP'], 2)]

    # This is just a sanity check, should be the same as the per-image values calculated below.
    mae_scores = [round(data['nms_maes'][c], 2) for c in ['viable', 'non_viable', 'empty', 'total']]

    per_img_metrics = pd.read_csv(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor',
                                               'test_results_on_individual_images.csv'), index_col=0)

    def bootstrap_mean_ci(column):
        values = per_img_metrics[column].tolist()
        stats = []
        for i in range(1000):
            samples = resample(values)
            stats.append(np.nanmean(samples, ))
        # confidence intervals
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = np.percentile(stats, p)
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = np.percentile(stats, p)

        mean = np.nanmean(values)
        return round(mean, 2), round(lower, 2), round(upper, 2)

    per_img_ap_results = []
    for metric in ['AP-Viable', 'AP-Non-Viable', 'AP-Empty', 'AP-Seed']:
        mean, lower, upper = bootstrap_mean_ci(metric)
        per_img_ap_results.append(f'{mean} ({lower} - {upper})')

    per_img_mae_results = []
    for metric in ['MAE-Viable', 'MAE-Non-Viable', 'MAE-Empty', 'MAE-Total']:
        mean, lower, upper = bootstrap_mean_ci(metric)
        per_img_mae_results.append(f'{mean} ({lower} - {upper})')

    out_df = pd.DataFrame([ap_scores, per_img_ap_results, per_img_mae_results, mae_scores]).T
    out_df.index = ['Viable', 'Non-viable', 'Empty', 'Seed class']
    out_df.columns = ['Overall AP', 'Per-Image AP (95% CI)', 'Per-Image MAE (95% CI)', 'MAE']

    summary_df = pd.read_csv(_test_dataset_summary_path, index_col=0)
    out_df['Per-Image Mean in GT'] = [round(summary_df.loc['mean', c], 2) for c in ['viable', 'nonviable', 'empty', 'total']]

    mean_ap, lower_ap, upper_ap = bootstrap_mean_ci('AP')
    out_df.loc['Overall (Avg across classes)'] = [round(data['segm_cls_agn_nms']['AP'], 2),
                                                  f'{mean_ap} ({lower_ap} - {upper_ap})',
                                                  'Not calculated', 'Not calculated', None]
    out_df.to_csv(_metrics_json_path.replace('.json', '_readable_summary.csv'))


def plot_AP_vs_seed_count():
    per_img_metrics = pd.read_csv(_individual_image_results_path, index_col=0, usecols=['image_name', 'AP'])
    per_img_metrics = per_img_metrics.rename(columns={'AP': 'mAP'})

    summary_df = pd.read_csv(_test_seed_stats_path, index_col=0, usecols=['file_names', 'total', 'viability_ratio'])
    summary_df = summary_df.rename(columns={'total': 'Seed Count'})
    analysis_df = pd.merge(per_img_metrics, summary_df, left_on='image_name', right_on='file_names', how='inner')
    assert len(analysis_df) == len(per_img_metrics)
    print(analysis_df)
    sns.set_theme(style="whitegrid")
    sns.regplot(data=analysis_df, x='Seed Count', y='mAP', lowess=True, ci=95, scatter=True)
    plt.savefig(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'mAP_vs_seed_count.png'), dpi=300)
    plt.close()

    small_number_of_seeds_ap = round(analysis_df[analysis_df['Seed Count'] < 200]['mAP'].mean(), 2)
    large_number_of_seeds_ap = round(analysis_df[analysis_df['Seed Count'] >= 200]['mAP'].mean(), 2)
    out_df = pd.DataFrame([small_number_of_seeds_ap, large_number_of_seeds_ap], index=['<200 seeds', '>200 seeds'], columns=['mAP'])
    out_df.to_csv(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'mAP_vs_seed_count_summary.csv'))


def plot_over_under_counts():
    per_img_metrics = pd.read_csv(_individual_image_results_path, index_col=0,
                                  usecols=['image_name', 'Predicted Viable', 'Predicted Non-Viable', 'Predicted Empty', 'Predicted Total'])
    summary_df = pd.read_csv(_test_seed_stats_path, index_col=0, usecols=['file_names', 'viable', 'nonviable', 'empty', 'total'])

    analysis_df = pd.merge(per_img_metrics, summary_df, left_on='image_name', right_on='file_names', how='inner')
    assert len(analysis_df) == len(per_img_metrics)
    print(analysis_df)

    analysis_df['viable error'] = analysis_df['Predicted Viable'] - analysis_df['viable']
    analysis_df['nonviable error'] = analysis_df['Predicted Non-Viable'] - analysis_df['nonviable']
    analysis_df['empty error'] = analysis_df['Predicted Empty'] - analysis_df['empty']
    analysis_df['total error'] = analysis_df['Predicted Total'] - analysis_df['total']
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    sns.kdeplot(data=analysis_df, x="viable error", color='green', label='Viable', ax=ax)
    sns.kdeplot(data=analysis_df, x="nonviable error", color='red', label='Non-Viable', ax=ax)
    sns.kdeplot(data=analysis_df, x="empty error", color='blue', label='Empty', ax=ax)
    sns.kdeplot(data=analysis_df, x="total error", color='black', label='Total', ax=ax)
    plt.legend(loc='upper left')
    plt.xlabel('Error in predicted counts')
    plt.savefig(os.path.join(REPO_PATH, 'model_results_and_final_weights', 'final_tz_segmentor', 'over_under_counts.png'), dpi=300)
    plt.close()


def main():
    # make_tables_from_results_json()
    # plot_AP_vs_seed_count()
    plot_over_under_counts()


if __name__ == '__main__':
    main()
