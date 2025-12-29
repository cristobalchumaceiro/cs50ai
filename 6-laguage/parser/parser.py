import nltk
import sys
from nltk.tree import Tree

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
S -> NP VP
S -> NP Conj NP VP
S -> NP VP Conj NP VP
S -> NP VP Conj VP

NP -> N | Det N | Det AP N | N PP | NP PP
VP -> V | V NP | V NP PP | AV | V PP
AP -> Adj | Adj AP
VP -> Adv VP | VP Adv | Adv
PP -> P NP
"""

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

    return [token.lower() for token in nltk.word_tokenize(sentence) if token.isalnum()]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_list = []
    
    # Iterates through all subtrees in tree
    for subtree in tree.subtrees():

        # Initialising flag to check if an NP is truly a 'chunk'
        chunk = True

        # If the current subtree is an NP and it does not contain any children with
        # the label NP or PP (therefore being a superset containing another NP) we
        # can then add it to the list of chunks.
        if (subtree.label() == "NP"):
            for child in subtree:
                if child.label() in ["NP", "PP"]:
                    chunk = False
            if chunk == True:
                np_list.append(subtree)

    return np_list


if __name__ == "__main__":
    main()
