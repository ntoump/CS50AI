import nltk
# nltk.download('all')
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | VP
NP -> N | Adj NP | Det N | Det Adj NP | Adj Conj Adj N | Det Adj Conj Adj N | Det N NP | P NP | P Det N | P Det N Adv\
    | N NP | P NP
VP -> V | V NP | Adv V | V Adv | Adv V NP | V P NP
"""
# Fully functional version; KEEP IT AS A reminder
# S -> NP VP | S Conj S | VP
# NP -> N | Adj NP | Det N | Det Adj NP | Adj Conj Adj N | Det Adj Conj Adj N | Det N NP | P N | P Det N | P Det N Adv\
#     | N NP | P Det NP
# VP -> V | V NP | Adv V | V Adv | Adv V NP | V P NP

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # Initialize useful variables
    words, counter = [], 0

    # Tokenize sentence and include tokens in list
    words.extend(nltk.word_tokenize(sentence))

    # Loop through the tokens
    for word in words:

        # For each letter that is an alphabetic char, add 1 to counter
        for letter in word:
            counter += (1 if letter.isalpha() else 0)

        # If counter == 0, that word doesn't have any alphabetic chars
        if counter == 0:

            # Remove unfit word from list
            words.remove(word)

        # Recalibrate counter variable
        counter = 0

    # Enfoce lowercase
    words = [w.lower() for w in words]

    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    # Loop through all subtrees that meet the filter criteria
    for subtree in tree.subtrees(lambda t: t.label().endswith("NP")):
        pass

    # Return last subtree as a nltk.tree
    return subtree


if __name__ == "__main__":
    main()
