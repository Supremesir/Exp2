import os
import tensorflow as tf
from tensorflow.keras import backend as K

species_num = 5  # 植物种类
learning_rate = 0.001  # 学习率
batch_size = 32  # 每个batch的大小
nb_epoch = 50  # 迭代次数
data_path = './PlantsData'  # 数据集路径


# 加载数据
def load_data(path, set):
    def _parse_function(filename, label):

        image_string = tf.io.read_file(filename)
        image_string = tf.image.decode_image(image_string)
        image_string.set_shape([224, 224, 3])
        image_decoded = tf.cast(image_string,
                                dtype=tf.float32) / tf.constant(255, dtype=tf.float32)
        label = tf.one_hot(label, species_num)
        return image_decoded, label

    file_name = []
    labels = []
    for inds, species in enumerate(sorted(os.listdir(os.path.join(path, set)))):
        for img in os.listdir(os.path.join(path, set, species)):
            file_name.append(os.path.join(path, set, species, img))
            labels.append(inds)
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_name), tf.constant(labels)))
    dataset = dataset.map(_parse_function)
    return dataset, len(file_name)


def main():
    train_dataset, train_num = load_data(data_path, 'train')
    train_dataset = train_dataset.shuffle(train_num).batch(batch_size).repeat()
    valid_dataset, valid_num = load_data(data_path, 'valid')
    valid_dataset = valid_dataset.batch(1).repeat()

    print('data prepared! train:%d,valid:%d' % (train_num, valid_num))
    # 加载MobileNet模型
    init_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3),
                                                           include_top=False,
                                                           weights='imagenet',
                                                           pooling='avg')
    # 在网络顶层进行fine-tune操作
    x = tf.keras.layers.Dense(512, activation='relu')(init_model.output)
    output = tf.keras.layers.Dense(species_num, activation='softmax')(x)
    model = tf.keras.Model(inputs=init_model.input, outputs=output)
    for layer in init_model.layers:
        layer.trainable = False
    # 优化算法
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True, decay=0.0003)
    # 编译模型
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # 运行模型
    model.fit(train_dataset, epochs=nb_epoch, steps_per_epoch=int(train_num / batch_size + 1),
              shuffle=True, validation_data=valid_dataset, validation_steps=valid_num)
    if not os.path.exists('./models/'):
        os.makedirs('./models/')  # 新建文件夹
    # model.save_weights('./models/checkpoints/model')  # 保存模型
    # model.save_weights('./models/checkpoints/model_weights_keras.h5', save_format='h5')  # 保存模型为.h5模型
    model.save('./models/my_model.h5')


if __name__ == '__main__':
    main()
