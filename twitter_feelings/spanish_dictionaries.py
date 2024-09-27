from deep_translator import GoogleTranslator
import time
import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
    return chardet.detect(raw_data)["encoding"]


def read_file_with_encoding(file_path):
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return [line.strip() for line in file if line.strip()]
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as file:
            return [line.strip() for line in file if line.strip()]


def translate_words(input_file, output_file):
    translator = GoogleTranslator(source="en", target="es")
    translated_words = []

    words = read_file_with_encoding(input_file)

    for word in words:
        try:
            # Traducir la palabra al español
            translation = translator.translate(word)
            translated_words.append(translation.lower())

            # Esperar un poco para evitar sobrecargar la API
            time.sleep(0.5)
        except Exception as e:
            print(f"Error traduciendo '{word}': {str(e)}")

    # Guardar las palabras traducidas
    with open(output_file, "w", encoding="utf-8") as file:
        for word in translated_words:
            file.write(word + "\n")

    print(f"Traducción completada. Resultados guardados en '{output_file}'")


# Uso del script
translate_words("archive/negative-words.txt", "archive/palabras_negativas.txt")
translate_words("archive/positive-words.txt", "archive/palabras_positivas.txt")
