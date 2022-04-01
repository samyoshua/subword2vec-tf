"""
A MWE of generating a corpus to feed into the model.
"""

import re

from nltk.corpus import brown, stopwords
from subword2vec import SubWord2Vec

STOPWORDS = set(stopwords.words("english"))


def brown_corpus(stopwords=STOPWORDS):
    """The Brown corpus is already tokenized."""
    for tokens in brown.sents():
        # remove puncation and stopwords\n",
        tokens = [
            re.sub(r"[\\!\"#$%&\\*+,-./:;<=>?@^_`()|~=]", "", token.lower())
            for token in tokens
            if token.lower() not in stopwords
        ]
        tokens = [token for token in tokens if token]  # remove empty tokens (e.g. \"''\")\n",
        if len(tokens) > 2:  # we don't want short sentences\n",
            yield tokens


w2v = SubWord2Vec(brown_corpus, epochs=5)

print(w2v.similarity.get_similarity("school", "superintendent"))
