from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np 


def getAudio():
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    @tf.keras.utils.register_keras_serializable()
    class CTCLoss(tf.keras.losses.Loss):
        def __init__(self, name="ctc_loss",**kwargs):
            super(CTCLoss, self).__init__(name=name , **kwargs)
        def get_config(self):
            config = super(CTCLoss, self).get_config()
            return config
        def call(self,y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf. cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            return loss
    
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    frame_length = 256
    frame_step = 160
    fft_length = 384

    def transcribe_single_voice(audio_file_path):
        file = tf.io.read_file(audio_file_path)
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        tmodel = load_model("modelKaggle.h5")
        audio = tf.cast(audio, tf.float32)
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        spectrogram = tf.expand_dims(spectrogram, 0)
        predictions = tmodel.predict(spectrogram)
        transcription = decode_batch_predictions(predictions)[0]
        return transcription

    custom_voice_path = "temp.wav"
    transcription = transcribe_single_voice(custom_voice_path)
    return transcription 