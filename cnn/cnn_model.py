import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Add, Flatten, MaxPooling2D, Dropout, Lambda
from keras.regularizers import l2
from keras import backend as K
# use this import instead of keras.models due to known issue #13353 (https://github.com/keras-team/keras/issues/13353)
from tensorflow.keras.models import load_model


def get_model(**kwargs):
    defaults = {
        'model_path': None,
        'weights_path': None,
        'input_shape': (128, 128, 1),
        'num_classes': 10_575,
        'dropout_rate': 0.7,
        'weight_decay': 5e-4,
        'weight_decay_fc2': 5e-3
    }

    if 'model_path' in kwargs:
        return TrainedModel(kwargs['model_path'])
    else:
        return TrainableModel(weights_path=kwargs.get('weights_path', defaults['weights_path']),
                              input_shape=kwargs.get('input_shape', defaults['input_shape']),
                              num_classes=kwargs.get('num_classes', defaults['num_classes']),
                              dropout_rate=kwargs.get('dropout_rate', defaults['dropout_rate']),
                              weight_decay=kwargs.get('weight_decay', defaults['weight_decay']),
                              weight_decay_fc2=kwargs.get('weight_decay_fc2', defaults['weight_decay_fc2']))


class TrainedModel:
    def __init__(self, model_path):
        self.model = load_model(model_path, custom_objects={'tf': tf})
        # alternative way to obtain model output
        # self.predictor = K.function(self.model.input, self.model.output)

    def get_feature_vector(self, face):
        return self.model.predict(face)[0]
        # alternative way to obtain model output
        # return self.predictor(face)[0]


class TrainableModel:
    def __init__(self, weights_path=None, input_shape=(128, 128, 1), num_classes=10_575, dropout_rate=0.7, weight_decay=5e-4, weight_decay_fc2=5e-3):
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.weight_decay_fc2 = weight_decay_fc2
        self.model = self.create_model(input_shape, num_classes)
        if weights_path:
            self.model.load_weights(weights_path)

    def compile_model(self, actual_lr=0.001, momentum=0.9):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(lr=actual_lr, momentum=momentum),
                           metrics=['accuracy'])

    def train_model(self, train_gen, val_gen, epochs, callbacks, init_epoch):
        history = self.model.fit_generator(generator=train_gen,
                                           validation_data=val_gen,
                                           epochs=epochs,
                                           verbose=1,
                                           workers=4,
                                           callbacks=callbacks,
                                           initial_epoch=init_epoch)
        return history

    def create_model(self, input_shape, num_classes):
        model_input = Input(shape=input_shape)
        x = Conv2D(filters=96, kernel_size=5, padding='same', kernel_regularizer=l2(self.weight_decay))(model_input)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = MaxPooling2D(pool_size=2, padding='same')(x)

        x = self.resnet_block(x, 96)
        x = Conv2D(filters=96, kernel_size=1, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = Conv2D(filters=192, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = MaxPooling2D(pool_size=2, padding='same')(x)

        for i in range(2):
            x = self.resnet_block(x, 192)
        x = Conv2D(filters=192, kernel_size=1, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = Conv2D(filters=384, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = MaxPooling2D(pool_size=2, padding='same')(x)

        for i in range(3):
            x = self.resnet_block(x, 384)
        x = Conv2D(filters=384, kernel_size=1, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)

        for i in range(4):
            x = self.resnet_block(x, 256)
        x = Conv2D(filters=256, kernel_size=1, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = MaxPooling2D(pool_size=2, padding='same')(x)

        x = Flatten()(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Dense(units=512, kernel_regularizer=l2(self.weight_decay))(x)
        x = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(self.weight_decay_fc2))(x)

        model = Model(model_input, x)
        return model

    def resnet_block(self, block_input, num_filters):
        block = Conv2D(filters=num_filters, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(block_input)
        block = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(block)
        block = Conv2D(filters=num_filters, kernel_size=3, padding='same', kernel_regularizer=l2(self.weight_decay))(block)
        block = Lambda(function=self.max_feature_map, output_shape=self.MFM_output_shape)(block)
        block = Add()([block, block_input])
        return block

    def max_feature_map(self, x):
        shape = x.shape
        if len(shape) == 2:
            output = tf.split(x, 2, 1)
        elif len(shape) == 4:
            output = tf.split(x, 2, 3)
        else:
            raise ValueError('Invalid tensor dimension: {}'.format(len(shape)))
        return K.maximum(output[0], output[1])

    def MFM_output_shape(self, input_shape):
        shape = list(input_shape)
        if len(input_shape) == 2:
            shape[1] //= 2
        elif len(input_shape) == 4:
            shape[3] //= 2
        else:
            raise ValueError('Invalid input shape: {}'.format(shape))
        return tuple(shape)
