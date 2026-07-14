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

    out_df = pd.merge(filenme_translation,meta_data_input,how='left' , on='Serial_number')
    assert len(out_df) == 522
    out_df.to_csv('metadata.csv',index=False)


if __name__ == '__main__':
    main()
