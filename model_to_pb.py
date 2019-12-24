import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


def main():
    species_num = 5
    tf.keras.backend.set_learning_phase(0)

    model = tf.keras.models.load_model('./models/my_model.h5')

    sess = tf.compat.v1.keras.backend.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               [model.outputs[0].name.split('s')[0]])
    graph_io.write_graph(constant_graph, './models/my_model.h5', './models/my_model.pb', as_text=False)
    print(constant_graph)


if __name__ == '__main__':
    main()
