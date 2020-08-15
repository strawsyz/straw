import sys
import wave

import pyaudio

#  播放一个wav文件
#  例如输入：python pyaudioexample.py F.wav
#  就会播放F.wav文件
CHUNK = 1024
if len(sys.argv) < 2:
    print("Plays  a wave file")
    sys.exit(-1)
wf = wave.open(sys.argv[1], 'rb')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                rate=wf.getframerate(), output=True)
data = wf.readframes(CHUNK)
while data != b'':
    stream.write(data)
    data = wf.readframes(CHUNK)
    print(data)

stream.stop_stream()
stream.close()
p.terminate()
