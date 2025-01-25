import whisper
from pydub import AudioSegment
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = whisper.load_model('medium')
model = model.to(device)

audio_name1 = "datasets/test_samples/NewJeans-OMG"
audio_data1 = AudioSegment.from_file(audio_name1 + ".mp4", format='mp4')
audio_data1.export(audio_name1 + ".mp3", format='mp3')

audio_name2 = "datasets/test_samples/moaning1"
# audio_data2 = AudioSegment.from_file(audio_name2 + ".mp4", format='mp4')
# audio_name2.export(audio_name2 + ".mp3", format='mp3')

result = model.transcribe(audio_name2 + '.mp3')
print(result['text'])