import pandas as pd

from agreement_analysis.analysis import get_agreement_analysis_annotations, annotations_to_polygons, compare_annotations_3, compare_annotations_2


def get_counts_for_annotator(annotator_idx: int, agreement_analysis_annotations, image_name: str):
    viable_count = 0
    nonviable_count = 0
    empty_count = 0

    for a in agreement_analysis_annotations[image_name][annotator_idx]:
        if a['category_id'] == 1:
            viable_count += 1
        if a['category_id'] == 2:
            nonviable_count += 1
        if a['category_id'] == 3:
            empty_count += 1

    return viable_count, nonviable_count, empty_count


def main(pre_or_post: str):
    switch = {
        0: 'Undetected',
        1: 'Viable',
        2: 'Non-Viable',
        3: 'Empty'
    }
    agreement_analysis_image_names, agreement_analysis_annotations, output_path = get_agreement_analysis_annotations(pre_or_post)
    print(agreement_analysis_image_names)

    total_counts_in_final_data = {'viable': 0, 'nonviable': 0, 'empty': 0, 'seeds': 0}
    seed_aes = []
    viable_aes = []
    nonviable_aes = []
    empty_aes = []
    image_count = 0
    for image_name in agreement_analysis_image_names:
        if agreement_analysis_annotations.get(image_name) is None:
            print(image_name)
            continue
        print(image_name)
        image_count += 1
        original_viable_count, original_nonviable_count, original_empty_count = get_counts_for_annotator(0, agreement_analysis_annotations,
                                                                                                         image_name)
        total_counts_in_final_data['viable'] += original_viable_count
        total_counts_in_final_data['nonviable'] += original_nonviable_count
        total_counts_in_final_data['empty'] += original_empty_count
        total_counts_in_final_data['seeds'] += original_viable_count + original_nonviable_count + original_empty_count

        annotator_1_viable_count, annotator_1_nonviable_count, annotator_1_empty_count = get_counts_for_annotator(1, agreement_analysis_annotations,
                                                                                                                  image_name)
        annotator_2_viable_count, annotator_2_nonviable_count, annotator_2_empty_count = get_counts_for_annotator(2, agreement_analysis_annotations,
                                                                                                                  image_name)
        annotator1_seed_ae = abs((original_viable_count + original_nonviable_count + original_empty_count) - (
                annotator_1_viable_count + annotator_1_nonviable_count + annotator_1_empty_count))
        annotator1_viable_ae = abs((original_viable_count - annotator_1_viable_count))
        annotator1_nonviable_ae = abs((original_nonviable_count - annotator_1_nonviable_count))
        annotator1_empty_ae = abs((original_empty_count - annotator_1_empty_count))

        annotator2_seed_ae = abs((original_viable_count + original_nonviable_count + original_empty_count) - (
                annotator_2_viable_count + annotator_2_nonviable_count + annotator_2_empty_count))
        annotator2_viable_ae = abs((original_viable_count - annotator_2_viable_count))
        annotator2_nonviable_ae = abs((original_nonviable_count - annotator_2_nonviable_count))
        annotator2_empty_ae = abs((original_empty_count - annotator_2_empty_count))

        # treat annotator 1 and 2 instances as like a new image, to average over later

        seed_aes.append(annotator1_seed_ae)
        seed_aes.append(annotator2_seed_ae)

        viable_aes.append(annotator1_viable_ae)
        viable_aes.append(annotator2_viable_ae)

        nonviable_aes.append(annotator1_nonviable_ae)
        nonviable_aes.append(annotator2_nonviable_ae)

        empty_aes.append(annotator1_empty_ae)
        empty_aes.append(annotator2_empty_ae)

    seed_mae = sum(seed_aes) / len(seed_aes)
    viable_mae = sum(viable_aes) / len(viable_aes)
    nonviable_mae = sum(nonviable_aes) / len(nonviable_aes)
    empty_mae = sum(empty_aes) / len(empty_aes)

    out_df = pd.DataFrame(
        {'MAE': [viable_mae, nonviable_mae, empty_mae, seed_mae],
         'Mean actual count per image': [total_counts_in_final_data['viable']/image_count, total_counts_in_final_data['nonviable']/image_count,
                                         total_counts_in_final_data['empty']/image_count, total_counts_in_final_data['seeds']/image_count]},
        index=['Viable', 'Non-Viable', 'Empty', 'Seed'])
    out_df.to_csv(
        output_path + '/MAEs.csv')


if __name__ == '__main__':
    main('pre')
    main('post')
    raise NotImplementedError(
        'add some sanity checks e.g. that mean counts per image match summaries. And think about which images are included in this part of analysis compared to model testing.')
