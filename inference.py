import os
import time

import tensorflow as tf


from retrain import species_num


def main(path):
    # species = []
    train_path = 'dataset/bjfu_plants'
    species = []
    for spe in sorted(os.listdir(train_path)):
        species.append(spe)
    # init_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3),
    #                                                        include_top=False,
    #                                                        pooling='avg')
    # x = tf.keras.layers.Dense(128, activation='relu')(init_model.output)
    # output = tf.keras.layers.Dense(species_num, activation='softmax')(x)
    # model = tf.keras.Model(inputs=init_model.input, outputs=output)
    # model.load_weights('./models/checkpoints/model')

    # 加载先前训练好的模型
    model = tf.keras.models.load_model('./models/my_model.h5')

    image_string = tf.io.read_file(path)
    image_string = tf.image.decode_image(image_string)
    image_string.set_shape([224, 224, 3])
    image = tf.cast(image_string,
                    dtype=tf.float32) / tf.constant(255, dtype=tf.float32)
    image_decoded = tf.expand_dims(image, axis=0)
    start = time.time()
    result = model.predict(image_decoded, steps=1)[0]
    print('Evaluation time for 1 image:%0.2f s' % (time.time() - start))
    for species, conf in zip(species, result):
        print('\t%s:%s0.4f' % (species, conf))


if __name__ == '__main__':
    img_path = 'PlantsData/test/meigui/MEIGUItest.jpg'
    main(img_path)
