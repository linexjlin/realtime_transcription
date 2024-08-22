import requests
import io
import os
from utils import wav_to_mp3
from dotenv import load_dotenv
load_dotenv()

def transcription(wav_io,prompt=""):
    if os.getenv("GROQ_KEY"):
        #print("GROQ>>>")
        return groq_api(wav_io,prompt)
    
    if os.getenv("SILICONFLOW_KEY"):
        #print("SILICONFLOW>>>")
        #return siliconflow_api_wav(wav_io)
        mp3_io = wav_to_mp3(wav_io=wav_io)
        return siliconflow_api_mp3(mp3_io)


def siliconflow_api_mp3(mp3_io, timeout=3): # mp3
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"

    audio_content = mp3_io.read()

    files = {
        'model': (None, 'iic/SenseVoiceSmall'),
        'file': ('audio.mp3', audio_content, 'audio/mpeg')
    }

    kauthorization_key = os.getenv("SILICONFLOW_KEY")

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {kauthorization_key}"
    }

    try:
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the response as a JSON object
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def siliconflow_api_wav(audio_io, timeout=3): # mp3
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"

    audio_content = audio_io.read()

    files = {
        'model': (None, 'iic/SenseVoiceSmall'),
        'file': ('audio.wav', audio_content, 'audio/wav')
    }

    kauthorization_key = os.getenv("SILICONFLOW_KEY")

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {kauthorization_key}"
    }

    try:
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the response as a JSON object
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def groq_api(audio_file, prompt="",timeout=3):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    if isinstance(audio_file, str):
        with open(audio_file, 'rb') as f:
            audio_content = f.read()
    elif isinstance(audio_file, io.BytesIO):
        audio_file.seek(0)  # Ensure we're at the start of the BytesIO buffer
        audio_content = audio_file.read()
    else:
        raise ValueError("audio_file must be either a file path (str) or io.BytesIO object.")

    files = {
        'model': (None, 'whisper-large-v3'),
        'file': ('audio.wav', audio_content, 'audio/wav'),
        'temperature': (None, '1'),
        'response_format': (None, 'json'),
        'prompt': (None, prompt),
    }
    authorization_key = os.getenv("GROQ_KEY")
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {authorization_key}"
    }

    try:
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the response as a JSON object
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
#print(groq_send_audio_to_api('xzs-sample.wav'))