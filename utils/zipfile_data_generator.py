import io
import zipfile
import numpy as np
import random
import cv2
from PIL import Image
from keras.utils import Sequence, to_categorical


class FileData:
    def __init__(self, label, file_path, zipfile_name):
        self.label = label
        self.file_path = file_path
        self.zipfile_name = zipfile_name


class DataHolder:
    def __init__(self, zipfile_paths):
        self.zipfile_paths = zipfile_paths
        self.zipfiles = self.open_zipfiles()
        self.class_dict = self.map_class_labels()
        self.files = self.map_file_structures()

    def get_zipfile(self, key):
        return self.zipfiles[key]

    def get_class_num(self):
        return len(self.class_dict)

    def get_files(self, subset):
        if subset not in ['training', 'validation']:
            raise ValueError('Invalid subset name: {}'.format(subset))

        files = []
        found_labels = set()
        for f in self.files:
            if subset == 'training':
                if f.label in found_labels:
                    files.append(f)
                else:
                    found_labels.add(f.label)
            else:
                if f.label not in found_labels:
                    files.append(f)
                    found_labels.add(f.label)

        return files

    def map_class_labels(self):
        class_dict = dict()
        class_num = 0
        for key, value in self.zipfiles.items():
            cleaned_fnames = [x for x in value.namelist() if x.endswith('.jpg')]

            for filename in cleaned_fnames:
                class_label = self.get_parent_dir(filename)
                if class_label not in class_dict:
                    class_dict[class_label] = class_num
                    class_num += 1

        print('Found {} classes total'.format(len(class_dict)))
        return class_dict

    def map_file_structures(self):
        files = []
        for key, value in self.zipfiles.items():
            cleaned_fnames = [x for x in value.namelist() if x.endswith('.jpg')]

            for filename in cleaned_fnames:
                class_label = self.get_parent_dir(filename)
                files.append(FileData(self.class_dict[class_label], filename, key))

        print('Found {} files total'.format(len(files)))
        return files

    def open_zipfiles(self):
        zipfiles = dict()
        for zf_path in self.zipfile_paths:
            zipfiles[zf_path] = zipfile.ZipFile(zf_path, 'r')
        return zipfiles

    def get_parent_dir(self, filepath):
        path = filepath.split('/')
        return path[len(path) - 2]


class DataGenerator(Sequence):
    def __init__(self, subset, data_holder, batch_size=128):
        if subset not in ['training', 'validation']:
            raise ValueError('Invalid subset name: {}'.format(subset))

        self.subset = subset
        self.data_holder = data_holder
        self.batch_size = batch_size
        self.files = self.data_holder.get_files(self.subset)
        random.shuffle(self.files)
        print('{} subset length: {}'.format(self.subset, len(self.files)))

    def __getitem__(self, idx):
        locations = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.empty((self.batch_size, 128, 128, 1))
        batch_y = np.empty(self.batch_size)

        for i, loc in enumerate(locations):
            batch_y[i] = loc.label

            zf = self.data_holder.get_zipfile(loc.zipfile_name)
            with zf.open(loc.file_path) as imgfile:
                imgdata = imgfile.read()
            x_i = self.preprocess_image(imgdata)
            batch_x[i] = x_i

        batch_y = to_categorical(batch_y, self.data_holder.get_class_num())
        return batch_x, batch_y

    def __len__(self):
        return len(self.files) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.files)

    def preprocess_image(self, imgdata):
        image = np.array(Image.open(io.BytesIO(imgdata)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32) / 255
        image = image.reshape(image.shape[0], image.shape[1], 1)
        image = self.crop_image(image, crop_method=('random' if self.subset == 'training' else 'center'))
        if self.subset == 'training' and random.uniform(0, 1) < 0.5:
            image = np.fliplr(image)
        return image

    def crop_image(self, image, crop_size=(128, 128), crop_method='random'):
        if crop_method not in ['random', 'center']:
            raise ValueError('Invalid crop method: {}'.format(crop_method))

        x, y, z = image.shape
        if crop_method == 'random':
            a = random.randint(0, x - crop_size[0])
            b = random.randint(0, y - crop_size[1])
        else:
            a = (x - crop_size[0]) // 2
            b = (y - crop_size[1]) // 2
        return image[a:a + crop_size[0], b:b + crop_size[1]]
