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


# Función para detectar puntos suspensivos (tanto ... como ..)
def has_ellipsis(text):
    """Verifica si el texto tiene puntos suspensivos (.. o ...)."""
    if not isinstance(text, str):
        return 0
    return 1 if re.search(r"\.{2,}", text) else 0


# Longitud promedio de las palabras en el texto
def average_word_length(text):
    words = re.findall(r"\b\w+\b", text)  # Tokenizar las palabras
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)


# Función para contar palabras completamente en mayúsculas
def count_uppercase_words(text):
    words = re.findall(r"\b\w+\b", text)
    return sum(1 for word in words if word.isupper())


# Función para contar el número de adjetivos en el texto usando spaCy
def count_adjectives(text):
    if not isinstance(text, str):
        return 0
    doc = nlp(text.lower())  # Procesar el texto con spaCy
    return sum(1 for token in doc if token.pos_ == "ADJ")  # Contar adjetivos


# Función para calcular la densidad de signos de puntuación
def punctuation_density(text):
    """Calcula la densidad de signos de puntuación en relación al total de caracteres."""
    if not isinstance(text, str) or len(text) == 0:
        return 0
    # Contar cuántos caracteres son signos de puntuación
    punctuation_count = len(
        re.findall(r"[^\w\s]", text)
    )  # Caracteres que no son alfanuméricos ni espacios
    return punctuation_count / len(text)


def count_negation_words_spacy(text):
    """Cuenta cuántas palabras de negación hay en el texto usando spaCy."""
    if not isinstance(text, str):
        return 0
    doc = nlp(text)  # Procesar el texto con spaCy
    return sum(1 for token in doc if token.dep_ == "neg")


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
df["has_ellipsis"] = df["content"].apply(
    has_ellipsis
)  # Nueva característica para puntos suspensivos
df["average_word_length"] = df["content"].apply(average_word_length)
# df["num_uppercase_words"] = df["content"].apply(count_uppercase_words)
df["num_adjectives"] = df["content"].apply(count_adjectives)
df["punctuation_density"] = df["content"].apply(punctuation_density)
# df["num_negation_words"] = df["content"].apply(count_negation_words_spacy)
df = df.drop(columns=["content"])


# Guardar el DataFrame actualizado en un nuevo archivo CSV
df.to_csv("output_file.csv", index=False)

print("Análisis completado y guardado en 'output_file.csv'")
