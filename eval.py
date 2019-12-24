import os
import tensorflow as tf
from tensorflow.keras import backend as K

from retrain import species_num, learning_rate, load_data, data_path


def main():

    # 加载MobileNet模型
    # init_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3),
    #                                                        include_top=False,
    #                                                        weights='imagenet',
    #                                                        pooling='avg')
    # x = tf.keras.layers.Dense(512, activation='relu')(init_model.output)
    # output = tf.keras.layers.Dense(species_num, activation='softmax')(x)
    # model = tf.keras.Model(inputs=init_model.input, outputs=output)

    # 加载先前训练好的模型
    model = tf.keras.models.load_model('./models/my_model.h5')

    # optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True, decay=0.0003)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.load_weights('./models/checkpoints/model')

    test_dataset, test_num = load_data(data_path, 'test')
    test_dataset = test_dataset.batch(1)
    print('data prepared! test:%d' % (test_num))
    # 测试模型
    result = model.evaluate(test_dataset, steps=test_num, verbose=0)
    print('test loss:%0.4f, test acc:%0.4f' % (result[0], result[1]))


if __name__ == '__main__':
    main()
