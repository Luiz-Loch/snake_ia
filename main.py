import tensorflow as tf

if __name__ == "__main__":
    # Verifique se o TensorFlow está reconhecendo as GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(tf.__version__)
    if gpus:
        print("GPUs encontradas:", gpus)
    else:
        print("Nenhuma GPU encontrada.")
