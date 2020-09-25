import dlib
import io
import os
import argparse
import datetime
import zipfile
import matplotlib.image
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--zipfiles', default='', type=str, help='comma separated list of zipfiles containing face dataset')

detector = dlib.cnn_face_detection_model_v1('resources/mmod_human_face_detector.dat')
sp = dlib.shape_predictor('resources/shape_predictor_5_face_landmarks.dat')


def main():
    global args
    args = parser.parse_args()

    zipfiles = [x for x in args.zipfiles.split(',')]
    if len(zipfiles) < 1:
        raise ValueError('Please specify at least one zipfile')

    i = 0
    temp_path = os.path.join(os.getcwd(), 'temp.jpg')
    global_t, t = datetime.datetime.now(), datetime.datetime.now()

    for zf in zipfiles:
        index = zf.rfind('.')
        new_zf = zf[:index] + '-aligned' + zf[index:]
        with zipfile.ZipFile(zf, 'r') as read_zf, zipfile.ZipFile(new_zf, 'a') as write_zf:
            for filename in read_zf.namelist():
                imgdata = read_zf.read(filename)
                image = np.array(Image.open(io.BytesIO(imgdata)))
                aligned = align_face(image)
                matplotlib.image.imsave(temp_path, aligned)
                write_zf.write(temp_path, arcname=filename)

                i += 1
                if i % 50_000 == 0:
                    print('Aligned {} faces'.format(i))
                    print('Time elapsed: {}'.format(datetime.datetime.now() - t))
                    t = datetime.datetime.now()

        print('TOTAL TIME: {}'.format(datetime.datetime.now() - global_t))


def align_face(image, crop_size=144):
    dets = detector(image, 1)
    if len(dets) < 1:
        return dlib.resize_image(image, crop_size, crop_size)
    max_confidence = 0
    for det in dets:
        if det.confidence > max_confidence:
            face = sp(image, det.rect)
            max_confidence = det.confidence
    return dlib.get_face_chip(image, face, crop_size)


if __name__ == '__main__':
    main()
