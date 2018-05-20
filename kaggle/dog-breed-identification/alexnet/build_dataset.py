import argparse
import random
import os
import util
import image_preprocessing_util as iputil

from PIL import Image
from tqdm import tqdm

SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Images',
                    help="Directory with the Images for Breed dataset")
parser.add_argument('--output_dir', default='data/256x256_Images', help="Where to write the new data")

parser.add_argument('--annotation_dir', default='data/Annotation',
                    help="Directory with the Annotations for Breed dataset")

def crop_and_resize_and_save(filename, breed_name, output_dir,
                             annotation_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method

    size_info = iputil.get_orginal_size_info(breed_name,
                                             filename, annotation_dir)
    target_image = image.resize(
                        (size_info["width"], size_info["height"]),
                        Image.BILINEAR)

    bounding_info = iputil.get_bounding_size_info(
                        breed_name, filename, annotation_dir)
    image_list = []
    for i, box in enumerate(bounding_info):
        try:
            cropped_image = target_image.crop((box[0], box[1], box[2], box[3]))
            resized_image = cropped_image.resize(
                        (size, size), Image.BILINEAR)
            target_filename = str(i)+"_"+filename.split('/')[-1]
            target_filename = os.path.join(output_dir, target_filename)
            resized_image.save(target_filename)
        except OSError as e:
            print(e)
            if os.path.exists(target_filename):
                os.remove(target_filename)
                print("{0} file is removed".format(target_filename))



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    for breed_name in os.listdir(args.data_dir):
        breed_dir = os.path.join(args.data_dir, breed_name)
        filenames = os.listdir(breed_dir)
        filenames = [os.path.join(breed_dir, f)
                     for f in filenames if f.endswith('.jpg')]
        random.seed(230)
        filenames.sort()
        random.shuffle(filenames)

        train_split = int(0.6 * len(filenames))
        dev_split = train_split + int(0.2 * len(filenames))

        train_filenames = filenames[:train_split]
        dev_filenames = filenames[train_split:dev_split]
        test_filenames = filenames[train_split:dev_split]

        filenames = {'train': train_filenames,
                     'dev': dev_filenames,
                     'test': test_filenames}

        util.check_and_make_directory(args.output_dir)

        # Preprocess train, dev and test
        for split in ['train', 'dev', 'test']:
            util.check_and_make_directory(os.path.join(
                args.output_dir, split))
            output_dir_split = os.path.join(
                            args.output_dir, split, breed_name)
            util.check_and_make_directory(output_dir_split)

            print("Processing {} data, saving preprocessed data to {}".format(
                                split, output_dir_split))
            for filename in tqdm(filenames[split]):
                crop_and_resize_and_save(filename, breed_name,
                                        output_dir_split,
                                        args.annotation_dir, size=SIZE)

    print("Done building dataset")
