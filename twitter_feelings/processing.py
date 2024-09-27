import spacy
import pandas as pd
import re

# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")


# Cargar los diccionarios desde archivos .txt
def load_word_list(file_path):
    """Carga una lista de palabras desde un archivo .txt, una palabra por línea."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


positive_words = load_word_list("archive/positive-words-utf8.txt")
negative_words = load_word_list("archive/negative-words-utf8.txt")


# Función para tokenizar y lematizar el texto usando spaCy
def lemmatize_text(text):
    """Tokeniza y lematiza el texto usando spaCy."""
    if not isinstance(text, str):
        return []
    doc = nlp(text.lower())  # Procesar el texto con spaCy
    lemmas = [token.lemma_ for token in doc]  # Extraer lemas
    return lemmas


# Función para contar palabras de un diccionario
def count_words_from_dict(text, word_list):
    """Cuenta cuántas palabras de un diccionario aparecen en el texto lematizado."""
    lemmas = lemmatize_text(text)
    return sum(1 for lemma in lemmas if lemma in word_list)


# Función para contar palabras positivas
def count_positive_words(text):
    return count_words_from_dict(text, positive_words)


# Función para contar palabras negativas
def count_negative_words(text):
    return count_words_from_dict(text, negative_words)


# Función para contar el número de palabras en el texto
def count_words(text):
    """Cuenta el número total de palabras en un texto."""
    if not isinstance(text, str):
        return 0
    words = re.findall(r"\b\w+\b", text)  # Usa regex para contar palabras
    return len(words)


# Función para contar el número de caracteres en el texto
def count_characters(text):
    """Cuenta el número total de caracteres en un texto."""
    if not isinstance(text, str):
        return 0
    return len(text)


# Función para detectar signos de interrogación
def has_question(text):
    """Verifica si el texto tiene signos de interrogación."""
    if not isinstance(text, str):
        return 0
    return 1 if "?" in text or "¿" in text else 0


# Función para detectar signos de exclamación
def has_exclamation(text):
    """Verifica si el texto tiene signos de exclamación."""
    if not isinstance(text, str):
        return 0
    return 1 if "!" in text or "¡" in text else 0


# Cargar el archivo CSV
df = pd.read_csv("twitter_feelings/csv/cleaned.csv")

# Reemplazar los valores NaN en 'content' con cadenas vacías
df["content"] = df["content"].fillna("")

# Aplicar las funciones para contar palabras, caracteres, y detectar signos
df["num_words"] = df["content"].apply(count_words)
df["num_characters"] = df["content"].apply(count_characters)
df["has_question"] = df["content"].apply(has_question)
df["has_exclamation"] = df["content"].apply(has_exclamation)
df["num_positive_words"] = df["content"].apply(count_positive_words)
df["num_negative_words"] = df["content"].apply(count_negative_words)

# Guardar el DataFrame actualizado en un nuevo archivo CSV
df.to_csv("output_file.csv", index=False)

print("Análisis completado y guardado en 'output_file.csv'")
