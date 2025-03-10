import os
import yt_dlp
import whisper
from google.cloud import speech, storage
from pydub import AudioSegment
import pykakasi
import re

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


# Оновлена функція транслітерації японського тексту в romaji з додаванням пробілів
def japanese_to_romaji(japanese_text):
    print("Text before converting to romaji:", japanese_text)

    # Використовуємо новий API для pykakasi
    kks = pykakasi.kakasi()
    result = kks.convert(japanese_text)

    # Збираємо romaji з пробілами між словами
    romaji_parts = []
    for item in result:
        # Використовуємо hepburn romanization
        romaji_parts.append(item['hepburn'])

    # З'єднуємо з пробілами
    romaji_text = " ".join(romaji_parts)

    # Спеціальна обробка для "engrish" (англійських слів у японській)
    english_pattern = re.compile(r'[a-zA-Z]+')
    romaji_text = english_pattern.sub(lambda m: m.group(0), romaji_text)

    print("✅ Romaji з пробілами:", romaji_text)
    return romaji_text


# Покращена таблиця транслітерації romaji → українська з підтримкою багатосимвольних комбінацій
ROMAJI_TO_UA = {
    # Базові японські склади
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

    # Додаткові комбінації
    "kya": "кя", "kyu": "кю", "kyo": "кьо",
    "gya": "ґя", "gyu": "ґю", "gyo": "ґьо",
    "sha": "ша", "shu": "шу", "sho": "шьо",
    "ja": "джя", "ju": "джю", "jo": "джьо",
    "cha": "чя", "chu": "чю", "cho": "чьо",
    "nya": "ня", "nyu": "ню", "nyo": "ньо",
    "hya": "хя", "hyu": "хю", "hyo": "хьо",
    "bya": "бя", "byu": "бю", "byo": "бьо",
    "pya": "пя", "pyu": "пю", "pyo": "пьо",
    "mya": "мя", "myu": "мю", "myo": "мьо",
    "rya": "ря", "ryu": "рю", "ryo": "рьо",

    # Подвоєні приголосні
    "kk": "кк", "ss": "сс", "tt": "тт", "pp": "пп",

    # Маленька "tsu" для подвоєних приголосних
    "っ": "っ",

    # Довгі голосні
    "aa": "аа", "ii": "іі", "uu": "уу", "ee": "ее", "oo": "оо",
    "ou": "оу", "ei": "ей",

    # Окремі голосні
    "a": "а", "i": "і", "u": "у", "e": "е", "o": "о",

    # Розділові знаки та спеціальні символи
    " ": " ", ".": ".", ",": ",", "!": "!", "?": "?", "-": "-",
    "'": "'", "\"": "\"", "(": "(", ")": ")"
}


# Функція транслітерації romaji → українська
def romaji_to_ukrainian(romaji_text):
    result = ""
    i = 0
    romaji_text = romaji_text.lower()  # Переводимо текст у нижній регістр для обробки

    while i < len(romaji_text):
        # Спершу перевіряємо тризначні комбінації (для японських приголосних з "ya", "yu", "yo")
        if i < len(romaji_text) - 2:
            trigraph = romaji_text[i:i + 3]
            if trigraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[trigraph]
                i += 3
                continue

        # Потім перевіряємо двозначні комбінації
        if i < len(romaji_text) - 1:
            digraph = romaji_text[i:i + 2]
            if digraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[digraph]
                i += 2
                continue

        # Якщо не знайшли багатосимвольну комбінацію, перевіряємо один символ
        char = romaji_text[i]
        if char in ROMAJI_TO_UA:
            result += ROMAJI_TO_UA[char]
        else:
            # Перевірка на латинські букви (для "engrish")
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                # Якщо це латинська буква, залишаємо її як є (для англійських слів)
                result += char
            else:
                # Інакше залишаємо символ як є
                result += char
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