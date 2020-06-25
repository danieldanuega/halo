import tensorflow as tf
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.keras.models import load_model
from tensorflow.python.platform import gfile
import tensorflow.contrib.tensorrt as trt
import helper

# Save into tensorflow model
tf.compat.v1.disable_eager_execution()
model = load_model('./models/keras_deepface.h5')
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.keras.backend.get_session()
save_path = saver.save(sess, './models/tfdeepface')

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
    # import the meta graph of the tensorflow model
    #saver = tf.train.import_meta_graph("./model/tensorflow/big/model1.meta")
    saver = tf.compat.v1.train.import_meta_graph("./models/tfdeepface.meta")
    # then, restore the weights to the meta graph
    #saver.restore(sess, "./model/tensorflow/big/model1")
    saver.restore(sess, "./models/tfdeepface")
    
    # specify which tensor output you want to obtain 
    # (correspond to prediction result)
    your_outputs = ["F7/Relu"]
    
    # convert to frozen model
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, # session
        tf.compat.v1.get_default_graph().as_graph_def(),# graph+weight from the session
        output_node_names=your_outputs)
    #write the TensorRT model to be used later for inference
    with gfile.GFile("./models/frozen_deepface.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")
    
    # Load frozen model from file
    frozen_graph = helper.loadPbGraph('./models/frozen_deepface.pb')

    # convert (optimize) frozen model to TensorRT model
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,# frozen model
        outputs=your_outputs,
        max_batch_size=1,# specify your max batch size
        max_workspace_size_bytes=2*(10**9),# specify the max workspace
        precision_mode="FP16") # precision, can be "FP32" (32 floating point precision) or "FP16"

    #write the TensorRT model to be used later for inference
    with gfile.GFile("./models/deepface_tensorrt.pb", 'wb') as f:
        f.write(trt_graph.SerializeToString())
    print("TensorRT model is successfully stored!")
