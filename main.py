import os
import yt_dlp
import whisper
from google.cloud import speech, storage
from pydub import AudioSegment
import pykakasi  # Add this import

# ВАЖЛИВО! Вказати шлях до JSON-файлу з ключем Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jp-ua-karaoke-subtitles-2e9b708a8ad6.json"

# 🔧 Константи
YOUTUBE_URL = "https://www.youtube.com/watch?v=WPl10ZrhCtk"
BUCKET_NAME = "jp-to-ua-audio-sub-bucket"
AUDIO_FILE = "audio.wav"


# Функція для завантаження аудіо з YouTube
def download_audio(youtube_url, output_path="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print("✅ Аудіо завантажено")


# Конвертація в WAV (Google Speech-to-Text вимагає WAV)
def convert_to_wav(mp3_path="audio.mp3", wav_path="audio.wav"):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    print("✅ Аудіо конвертовано в WAV")


# Завантаження у GCS
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Створюємо bucket, якщо він не існує
    if not bucket.exists():
        bucket.create(location="us")
        print(f"✅ Bucket `{bucket_name}` створено!")

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"✅ Файл {source_file_name} завантажено у gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"


# Функція розпізнавання мови
def transcribe_audio_gcs(gcs_uri):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ja-JP"
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("🕒 Обробка аудіо, зачекайте...")

    response = operation.result(timeout=600)  # Чекаємо до 10 хвилин
    transcript = "\n".join([result.alternatives[0].transcript for result in response.results])

    return transcript.strip()


# Транслітерація японського тексту в romaji (використовуємо pykakasi замість Whisper)
def japanese_to_romaji(japanese_text):
    # Ініціалізація kakasi конвертера
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')  # Hiragana to ascii
    kakasi.setMode('K', 'a')  # Katakana to ascii
    kakasi.setMode('J', 'a')  # Japanese to ascii
    converter = kakasi.getConverter()

    print("Text before converting to romaji:", japanese_text)
    romaji_text = converter.do(japanese_text)
    print("✅ Romaji:", romaji_text)
    return romaji_text


# Покращена таблиця транслітерації romaji → українська
ROMAJI_TO_UA = {
    "ka": "ка", "ki": "кі", "ku": "ку", "ke": "ке", "ko": "ко",
    "ga": "ґа", "gi": "ґі", "gu": "ґу", "ge": "ґе", "go": "ґо",
    "sa": "са", "shi": "ші", "su": "су", "se": "се", "so": "со",
    "za": "дза", "ji": "джі", "zu": "дзу", "ze": "дзе", "zo": "дзо",
    "ta": "та", "chi": "чі", "tsu": "цу", "te": "те", "to": "то",
    "da": "да", "di": "ді", "du": "ду", "de": "де", "do": "до",
    "na": "на", "ni": "ні", "nu": "ну", "ne": "не", "no": "но",
    "ha": "ха", "hi": "хі", "fu": "фу", "he": "хе", "ho": "хо",
    "ba": "ба", "bi": "бі", "bu": "бу", "be": "бе", "bo": "бо",
    "pa": "па", "pi": "пі", "pu": "пу", "pe": "пе", "po": "по",
    "ma": "ма", "mi": "мі", "mu": "му", "me": "ме", "mo": "мо",
    "ya": "я", "yu": "ю", "yo": "йо",
    "ra": "ра", "ri": "рі", "ru": "ру", "re": "ре", "ro": "ро",
    "wa": "ва", "wo": "во", "n": "н",
    "a": "а", "i": "і", "u": "у", "e": "е", "o": "о",
    " ": " ", ".": ".", ",": ",", "!": "!", "?": "?"
}


# Функція транслітерації romaji → українська
def romaji_to_ukrainian(romaji_text):
    # Покращений алгоритм конвертації
    result = ""
    i = 0
    while i < len(romaji_text):
        # Перевіряємо спочатку двосимвольні комбінації
        if i < len(romaji_text) - 1:
            digraph = romaji_text[i:i + 2].lower()
            if digraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[digraph]
                i += 2
                continue

        # Якщо не знайшли двосимвольну комбінацію, перевіряємо один символ
        char = romaji_text[i].lower()
        if char in ROMAJI_TO_UA:
            result += ROMAJI_TO_UA[char]
        else:
            # Якщо символ не знайдено в таблиці, залишаємо його як є
            result += romaji_text[i]
        i += 1

    print("✅ Транслітерація:", result)
    return result


# Основна функція
def main():
    download_audio(YOUTUBE_URL)
    convert_to_wav()

    # Завантажуємо у GCS та отримуємо URI
    gcs_uri = upload_to_gcs(BUCKET_NAME, AUDIO_FILE, AUDIO_FILE)

    # Розпізнаємо мову через GCS
    jp_text = transcribe_audio_gcs(gcs_uri)
    print("Japanese text:", jp_text)

    # Транслітерація
    romaji_text = japanese_to_romaji(jp_text)
    ukrainian_text = romaji_to_ukrainian(romaji_text)

    # Збереження субтитрів у файл
    with open("karaoke_subtitles.srt", "w", encoding="utf-8") as f:
        f.write(ukrainian_text)

    print("✅ Субтитри збережені в karaoke_subtitles.srt")


if __name__ == "__main__":
    main()