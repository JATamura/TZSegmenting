import os

import pandas as pd

def get_img_resolution(img_path):
    from PIL import Image
    img = Image.open(img_path)
    return img.size

def main():
    meta_data_input = pd.read_excel('00_Dataset1_metadata.xlsx')
    filenme_translation = pd.read_excel('Translation.xlsx')

    def convert_int_to_filename(int_filename):
        if int_filename < 10:
            return '00' + str(int_filename) + '.jpg'
        elif int_filename < 100:
            return '0' + str(int_filename) + '.jpg'
        else:
            return str(int_filename) + '.jpg'

    filenme_translation['filename'] = filenme_translation['Photo_ID'].apply(convert_int_to_filename)
    filenme_translation.rename(columns={'Serial_no': 'Serial_number'}, inplace=True)

    def get_img_resolution_from_filename(filename):
        return get_img_resolution(os.path.join('all_images',filename))

    filenme_translation['width'], filenme_translation['height'] = zip(*filenme_translation['filename'].apply(get_img_resolution_from_filename))
    filenme_translation['resolutions'] = filenme_translation[['width','height']].apply(lambda x: '_'.join([str(i) for i in x]), axis=1)

    # All taken images
    all_images_df = pd.merge(filenme_translation,meta_data_input,how='left' , on='Serial_number')
    assert len(all_images_df) == 522

    # Images used after removing 4 problem images
    images_with_issues = pd.read_csv('image_metadata/images_with_issues.csv')
    final_images = all_images_df[~all_images_df['filename'].isin(images_with_issues['file_name'])]

    all_images_df.to_csv('image_metadata/all_metadata.csv',index=False)
    final_images.to_csv('image_metadata/final_image_metadata.csv',index=False)

    print(final_images['resolutions'].unique())

    final_images.describe(include='all').to_csv('image_metadata/final_image_metadata_summary.csv')
    all_images_df.describe(include='all').to_csv('image_metadata/all_image_metadata_summary.csv')


if __name__ == '__main__':
    main()
