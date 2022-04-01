import itertools
import random
from collections import Counter
from inspect import isgeneratorfunction

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dot, Embedding, Input, Lambda
from tensorflow.keras.models import Model
from tqdm import tqdm

from .subword_similarity import SubwordSimilarity


class Corpus:
    """A simple wrapper to allow corpora to be either lists or
    generator function and be resused within SubWord2Vec"""

    def __init__(self, corpus: iter) -> None:
        self.corpus = corpus
        self.is_generator = isgeneratorfunction(self.corpus)

    def __iter__(self) -> str:
        yield from self.corpus() if self.is_generator else self.corpus

    def __len__(self) -> int:
        if self.is_generator:
            return sum(1 for _ in self.__iter__())
        else:
            return len(self.corpus)


class SubWord2Vec:
    def __init__(
        self,
        corpus: iter = None,  # a list or generator function, if passed will train
        embedding_size: int = 300,  # size of word embeddings, 300 is common.
        ngram_range: tuple = (3, 6),  # based on paper
        window: int = 4,  # window to get contexts
        ns: int = 5,  # negative sampling ratio, set to 0 for no negative samples
        min_token_count: int = 1,  # must have >= this to be included in dictionary
        max_vocab_size: int = None,  # if set can cap dictionary size
        batch_size: int = 128,  # minibatch size used by tf.data
        buffer_size: int = 25000,  # buffer to store used by tf.data
        custom_loss: bool = False,  # variant of the model architecture
        epochs: int = 5,  # epochs to train for
    ):
        self.embedding_size = embedding_size
        self.ngram_range = ngram_range
        self.window = window
        self.ns = ns
        self.custom_loss = custom_loss

        if corpus:
            corpus = Corpus(corpus)
            self.build_dictionary(
                corpus, min_token_count=min_token_count, max_vocab_size=max_vocab_size
            )
            _ = self.train(corpus, batch_size=batch_size, buffer_size=buffer_size, epochs=epochs)
            self.build_similarity_indexes()

    def build_dictionary(
        self, corpus: Corpus, min_token_count: int = 1, max_vocab_size: int = None
    ):
        """Takes a corpus and generates (word and character n-gram) token mappings"""
        print("Building dictionaries...")
        # get counts
        self.subword_counts = Counter()
        self.word_counts = Counter()
        self.corpus_size = len(corpus)

        for sentence in tqdm(corpus, desc="processing corpus"):
            self.word_counts.update(sentence)
            self.subword_counts.update(
                itertools.chain(*[self.get_character_ngrams(token) for token in sentence])
            )

        # make our token <-> id mappings
        self.tok2id, self.id2tok = ({"<UNK>": 1}, {1: "<UNK>"})
        token_iter = tqdm(
            self.subword_counts.most_common(max_vocab_size), desc="subword dictionary"
        )
        for i, (token, count) in enumerate(token_iter):
            if count < min_token_count:
                continue
            self.tok2id[token] = i + 2
            self.id2tok[i + 2] = token

        # make our word <-> id mappings
        self.word2id, self.id2word = ({"<UNK>": 1}, {1: "<UNK>"})
        for i, word in enumerate(tqdm(self.word_counts, desc="word dictionary")):
            self.word2id[word] = i + 2
            self.id2word[i + 2] = word

        # get dictionary sizes
        self.vocab_size = len(self.tok2id)
        self.context_vocab_size = len(
            self.word_counts
        )  # for context embeddings to not have subwords

    def generate_training_data(self, corpus: Corpus, window: int = 5, ns: int = 5):
        """creates target/context pairs with optional negative sampling"""
        if not hasattr(self, "tok2id"):
            self.build_dictionary(corpus)
        targets, contexts, labels = ([], [], [])
        target_size = 0

        for tokens in tqdm(corpus, desc="generating training data"):
            t, c, l, ts = self.generate_skipgram_pairs(tokens, window=window, ns=ns)
            if t:
                targets.extend(t)
                contexts.extend(c)
                labels.extend(l)
                target_size = max(target_size, ts)

        print(f"We now have {len(targets):,d} training examples")
        return targets, contexts, labels, target_size

    def train(
        self,
        corpus: Corpus,
        window: int = None,
        ns: int = None,
        batch_size: int = 128,
        buffer_size: int = 25000,
        custom_loss: bool = False,
        rebuild_model: bool = False,
        epochs: int = 5,
    ):
        if not hasattr(self, "tok2id"):  # make sure we have vocab sizes
            self.build_dictionary(corpus)

        window = window or self.window
        ns = ns or self.ns
        custom_loss = custom_loss or custom_loss

        targets, contexts, labels, target_size = self.generate_training_data(
            corpus, window=window, ns=ns
        )

        if rebuild_model or not hasattr(self, "model"):  # build model if not already present
            self.build_model(target_size, custom_loss=custom_loss)

        def get_padded_training_data():
            """generator function to add padding and format for tensorflow.data"""
            for target, context, label in zip(targets, contexts, labels):
                target = target + [0] * (target_size - len(target))
                yield {"target_input": target, "context_input": context}, label

        training_data_signature = (
            {
                "target_input": tf.TensorSpec(shape=(target_size,), dtype=tf.int32),
                "context_input": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        dataset = tf.data.Dataset.from_generator(
            get_padded_training_data, output_signature=training_data_signature
        )
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        print("Num Batches:", int(len(targets) / buffer_size))

        self.history = self.model.fit(dataset, epochs=epochs)
        return self.history

    def build_model(self, target_size, custom_loss: bool = False):
        # to make model more readable
        embedding_size = self.embedding_size
        vocab_size = self.vocab_size + 1
        context_vocab_size = self.context_vocab_size + 1

        # target has target_size tokens (i.e. word and character n-grams/subwords)
        target_tokens = Input(shape=(target_size,), dtype="int32", name="target_input")
        target_subvectors = Embedding(vocab_size, embedding_size, name="target_embedding")(
            target_tokens
        )

        # context -  not we don't use subwords here - so use a different embedding size of just the whole words
        context_token = Input(shape=(1,), dtype="int32", name="context_input")  # vector id
        context_vector = Embedding(context_vocab_size, embedding_size, name="context_embedding")(
            context_token
        )

        # output loss is equivalent to the binary cross entropy of (sum subvectors) dotted with the context vector
        output = Dot(axes=2, name="dot_target_context")([target_subvectors, context_vector])
        if not custom_loss:
            output = Lambda(lambda z: tf.keras.backend.sum(z, axis=1), name="sum")(output)
            output = Dense(1, activation="sigmoid", name="output")(output)  # as with skipgram

        self.model = Model(inputs=[target_tokens, context_token], outputs=output)
        loss = self.subword_loss if custom_loss else "binary_crossentropy"
        self.model.compile(loss=loss, optimizer="adam")

    @staticmethod
    def subword_loss(y_true, dot_product):
        y_true = tf.cast(y_true, tf.bool)
        s = tf.reduce_sum(dot_product, axis=1, keepdims=True)

        L_pos = tf.boolean_mask(s, y_true, axis=0)
        L_pos = tf.math.log(tf.add(1.0, tf.math.exp(-L_pos)))

        L_neg = tf.boolean_mask(s, tf.math.logical_not(y_true), axis=0)
        L_neg = tf.math.log(tf.add(1.0, tf.math.exp(L_neg)))

        return tf.add(tf.reduce_sum(L_pos), tf.reduce_sum(L_neg))

    def get_character_ngrams(self, word, ngram_range=None):
        ngram_range = ngram_range or self.ngram_range
        word = f"<{word}>"  # add start/end character
        z = [word]
        for n in range(*ngram_range):
            z.extend(word[i : i + n] for i in range(len(word) - n + 1))
        return z

    def generate_skipgram_pairs(self, tokens: list, ns: int = None, window: int = None) -> list:
        window = window or self.window
        ns = ns or self.ns

        targets, contexts, labels = ([], [], [])
        max_target_size = 0
        context_nc_ids = range(2, self.context_vocab_size)  # do not want <UNK>

        for i in range(len(tokens)):
            target = tokens[i]
            context = [
                c
                for c in tokens[i - window : i] + tokens[i + 1 : i + window]
                if self.tok2id.get(c, 0) != 0
            ]
            target_chargrams = self.get_character_ngrams(target)
            subtargets = [
                self.tok2id.get(chargram)
                for chargram in target_chargrams
                if self.tok2id.get(chargram, 0) != 0
            ]
            max_target_size = max(max_target_size, len(subtargets))

            for c in context:
                targets.append(subtargets)
                contexts.append(self.word2id.get(c))
                labels.append(1)
            if ns > 0:
                for nc in random.sample(context_nc_ids, ns):
                    targets.append(subtargets)
                    contexts.append(nc)
                    labels.append(0)
        return targets, contexts, labels, max_target_size

    def get_embeddings(self):
        return self.model.get_layer("target_embedding").get_weights()[0]

    def build_similarity_indexes(self):
        self.similarity = SubwordSimilarity(
            self.tok2id,
            self.id2tok,
            self.get_embeddings(),
            self.embedding_size,
            self.get_character_ngrams,
        )
