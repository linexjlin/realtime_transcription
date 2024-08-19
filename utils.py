import scipy.io.wavfile as wavfile
import numpy as np
import lameenc
import io

def wav_to_mp3(wav_io, bitrate=128):
    # 读取WAV文件
    wav_io.seek(0)  # Ensure we are at the start of the BytesIO object
    sample_rate, samples = wavfile.read(wav_io)
    
    # 确保样本是int16类型
    if samples.dtype != np.int16:
        samples = (samples * 32767).astype(np.int16)
    
    # 初始化LAME编码器
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(2 if len(samples.shape) > 1 else 1)
    encoder.set_quality(2)  # 2是高质量设置
    
    # 编码
    if len(samples.shape) == 1:
        mp3_data = encoder.encode(samples)
    else:
        mp3_data = encoder.encode(samples.T[0], samples.T[1])
    mp3_data += encoder.flush()
    
    # 创建并返回包含MP3数据的io.BytesIO对象
    mp3_io = io.BytesIO()
    mp3_io.write(mp3_data)
    mp3_io.seek(0)  # Reset the position to the start
    return mp3_io

# 使用示例
#wav_to_mp3("rec_13-84.wav", "output.mp3")