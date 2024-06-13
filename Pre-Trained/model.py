
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import json

def getAudio():
    FRAME_RATE = 16000
    CHANNELS=1
    model = Model(model_name="vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)
    mp3 = AudioSegment.from_wav("temp.wav")
    mp3 = mp3.set_channels(CHANNELS)
    mp3 = mp3.set_frame_rate(FRAME_RATE)

    rec.AcceptWaveform(mp3.raw_data)
    result = rec.Result()

    text = json.loads(result)["text"]


    return text