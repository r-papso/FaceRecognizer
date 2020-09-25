import argparse
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TerminateOnNaN, CSVLogger

from utils.zipfile_data_generator import DataHolder, DataGenerator
from cnn import cnn_model as cm

parser = argparse.ArgumentParser()
parser.add_argument('--zipfiles', default='', type=str, help='comma separated list of zipfiles containing face dataset')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--init_epoch', default=0, type=int, help='initial epoch number')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--actual_lr', default=0.001, type=float, help='actual learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--weight_decay_fc2', default=5e-3, type=float, help='weight decay of last fully connected layer')
parser.add_argument('--dropout_rate', default=0.7, type=float, help='dropout rate')
parser.add_argument('--load_path', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--save_path', default='', type=str, help='path to save checkpoint')
parser.add_argument('--log_path', default='', type=str, help='path to log training statistics into csv format')


def main():
    global args
    args = parser.parse_args()

    zipfiles = [x for x in args.zipfiles.split(',')]
    if len(zipfiles) < 1:
        raise ValueError('Please specify at least one zipfile')

    holder = DataHolder(zipfiles)
    train_gen = DataGenerator(subset='training', data_holder=holder)
    val_gen = DataGenerator(subset='validation', data_holder=holder)
    x, y = train_gen.__getitem__(0)

    model = cm.get_model(weights_path=args.load_path,
                         input_shape=x[0].shape,
                         num_classes=len(y[0]),
                         dropout_rate=args.dropout_rate,
                         weight_decay=args.weight_decay,
                         weight_decay_fc2=args.weight_decay_fc2)

    callbacks = []
    callbacks.append(LearningRateScheduler(lr_schedule))
    if args.save_path:
        callbacks.append(ModelCheckpoint(args.save_path, save_weights_only=True))
    if args.log_path:
        callbacks.append(CSVLogger(args.log_path, append=True))
    callbacks.append(TerminateOnNaN())

    model.compile_model(actual_lr=args.actual_lr, momentum=args.momentum)
    model.train_model(train_gen, val_gen, args.epochs, callbacks, args.init_epoch)


def lr_schedule(epoch, lr):
    return args.init_lr * (0.65 ** (epoch // 10))


if __name__ == '__main__':
    main()
