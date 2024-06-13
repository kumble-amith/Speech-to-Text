from flask import Flask, render_template, request
import wave
import io
from model import getAudio
app = Flask(__name__ ,template_folder='templates')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('index2.html', error='No file part')

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return render_template('index2.html', error='No selected file')

        if audio_file and allowed_file(audio_file.filename):
            audio_data = audio_file.read()

            result = process_audio(audio_data)  

            return render_template('index2.html', result=result)

        return render_template('index2.html', error='File format not supported')

    return render_template('index2.html')


def process_audio(audio_data):
    with wave.open(io.BytesIO(audio_data), 'rb') as audio_wave:
        params = audio_wave.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_wave.rewind()
        audio_frames = audio_wave.readframes(n_frames)

    temp_wav_path = 'temp.wav'
    with wave.open(temp_wav_path, 'wb') as temp_wave:
        temp_wave.setnchannels(n_channels)
        temp_wave.setsampwidth(sampwidth)
        temp_wave.setframerate(framerate)
        temp_wave.writeframes(audio_frames)
    text = getAudio()
    return text


if __name__ == '__main__':
    app.run(debug=True)
