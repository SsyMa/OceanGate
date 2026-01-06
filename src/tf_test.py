import tensorflow as tf

'''
    Script to test if tensorflow can detect GPU correctly.
'''

def test():
    print(f"\nTensorflow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {gpus}")
    print("\nDetailed GPU info:")
    for i, gpu in enumerate(gpus):
        print(f"[{i}]: {tf.config.experimental.get_device_details(gpu)}")



if __name__ == "__main__":
    test()