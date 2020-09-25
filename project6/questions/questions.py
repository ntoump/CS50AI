import nltk
import sys
import os
import string
import math
# from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    mapper = {}

    # Loop through (txt) files
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        # Open and read file
        with open(path, "r", encoding="utf-8") as reader:
            r = reader.read()
            mapper[file] = r
    # Return dict in this format {name: file reader}
    return mapper


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    new_words = []

    # Get punctuation and stopwords variables
    punctuation = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")

    # Tokenize document
    words = nltk.word_tokenize(document)

    # Loop through tokenized words and make them lowercase
    for word in words:
        word = word.lower()

        # Append word to new_words list if it's valid
        if word not in punctuation and word not in stopwords:
            new_words.append(word)

    # Return list
    return new_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Initialize useful variables
    pairs, all_words = {}, []
    number_of_docs = len(documents)
    all_texts = documents.keys()

    # Flatten all words in all_words
    for sublist in documents.values():
        for item in sublist:
            all_words.append(item)

    # Loop thourgh all words
    for word in all_words:

        # Initialize variable to 0 with each word
        docs_with_word = 0

        # Loop through all texts in "documents"
        for text in all_texts:

            # If word in text, increase docs_with_word and move on to next text
            if word in documents.get(text):
                docs_with_word += 1
                continue

        # Insert {word: inverse document frequency (idf) value} to dict pairs
        pairs[word] = math.log(number_of_docs/docs_with_word)

    # Return dict
    return pairs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    totals = {}
    returning_list = []

    # Loop through files
    for text in files:
        # Initialize variable to 0 with each text
        values = 0

        # Loop through query
        for word in query:
            # Initialize variable to 0 with each word
            counter, value = 0, 0

            # Check if word in corresponding list of words of text
            if word in files.get(text):

                # Lister is not necessary, as files.get(text) is already a list // Used to indicate that further
                lister = list(files.get(text))

                # Count word frequency in list
                counter = lister.count(word)

                # Calculate the "term frequency (tf) - idf" value
                value = idfs.get(word) * counter

            # Add value to values
            values += value

        # Insert {text: values} to dict totals
        totals[text] = values

    # Sort in an ascending order and save in sorts
    sorts = sorted([value, key] for key, value in totals.items())

    # Loop through (1, n+1)
    for i in range(1, n+1):

        # Append key of last entry
        returning_list.append(sorts[-i][1])

    # Return list
    return returning_list


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    totals = {}
    returning_list = []

    # Loop through sentences
    for sentence in sentences:
        summer, counter = 0, 0

        # Through words
        for word in query:

            # Add to variables "counter" and "summer"
            if word in sentences.get(sentence):
                counter += sentences.get(sentence).count(word)
                summer += (idfs.get(word) if idfs.get(word) is not None else 0)

        # Prevent NoneType errors
        if counter > 0:

            # Get density & insert {sentence: (summer, density} to totals
            density = counter / len(sentences.get(sentence))
            totals[sentence] = (summer, density)

    # Sort in ascending order and save in sorts
    sorts = sorted((value, key) for key, value in totals.items())

    # Loop through (1, n+1)
    for i in range(1, n + 1):

        # Append key of last entry
        returning_list.append(sorts[-i][1])

    # Return list
    return returning_list


if __name__ == "__main__":
    main()
