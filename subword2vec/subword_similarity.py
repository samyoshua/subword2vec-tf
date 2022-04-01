import numpy as np
from annoy import AnnoyIndex


class SubwordSimilarity:
    def __init__(self, tok2id, id2tok, embeddings, embedding_size, get_character_ngrams):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.embedding_size = embedding_size
        self.get_character_ngrams = get_character_ngrams

        self.wordvecs = self.get_word_vectors(embeddings)
        self.subwordvecs = self.get_subword_vectors(embeddings)
        self.build_indexes()

    def build_indexes(self):
        """build separate indexes for whole words and subwords"""
        self.word_index = AnnoyIndex(self.embedding_size, "angular")
        for i, vec in self.wordvecs.items():
            self.word_index.add_item(i, vec)
        self.word_index.build(10)

        self.subword_index = AnnoyIndex(self.embedding_size, "angular")
        for i, vec in self.subwordvecs.items():
            self.subword_index.add_item(i, vec)
        self.subword_index.build(10)

    def get_word_vectors(self, embeddings):
        """gets word vectors as sum of n-grams"""
        word_vectors = {}
        for word, i in self.tok2id.items():
            if word == "<UNK>":
                continue
            if word.startswith("<") and word.endswith(">"):
                ngrams = self.get_character_ngrams(word.lstrip("<").rstrip(">"))
                subvectors = [
                    embeddings[self.tok2id[tok], :] for tok in ngrams if self.tok2id.get(tok)
                ]
                if len(subvectors) > 1:
                    word_vectors[i] = np.sum(np.array(subvectors), axis=0)
                else:
                    word_vectors[i] = subvectors[0]
        return word_vectors

    def get_subword_vectors(self, embeddings):
        """gets individual n-gram vectors"""
        word_vectors = {}
        for word, i in self.tok2id.items():
            if word == "<UNK>":
                continue
            if not (word.startswith("<") and word.endswith(">")):
                word_vectors[i] = embeddings[i, :]
        return word_vectors

    def get_nearest_words(self, word, n=5):
        word = f"<{word}>"
        return [self.id2tok[i] for i in self.word_index.get_nns_by_item(self.tok2id[word], n=n)]

    def get_nearest_subwords(self, word, n=5):
        word = f"<{word}>"
        return [
            self.id2tok[i]
            for i in self.subword_index.get_nns_by_vector(self.wordvecs[self.tok2id[word]], n=n)
        ]

    def get_similar_words(self, positive: list, negative: list = None, n: int = 5):
        negative = negative or []

        vector = np.zeros(
            self.embedding_size,
        )
        for pos in positive:
            vector += self.wordvecs[self.tok2id[f"<{pos}>"]]
        for neg in negative:
            vector -= self.wordvecs[self.tok2id[f"<{neg}>"]]
        return [self.id2tok[i] for i in self.word_index.get_nns_by_vector(vector, n=n)]

    def get_similarity(self, word1, word2):
        vec1 = self.wordvecs[self.tok2id[f"<{word1}>"]]
        vec2 = self.wordvecs[self.tok2id[f"<{word2}>"]]
        return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
