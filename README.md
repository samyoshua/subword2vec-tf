# Subword2Vec

A quick reimplementation of "[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)" by P. Bojanowski et al in Tensorflow. Note that this code is for reference to understand the model and is not recommended for practical use as it does not implement hash embedding, so the extra subword tokens can make this model too large with realistic corpora.

`SubWord2Vec` is the base class. Provided with a corpus it will build the dictionaries, train a model and build indexes to find nearest whole words or nearest subwords. Can be instantiated with the following options.

```python
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
```

### TODO:

- Replace more Keras code with TensorFlow (for hash embeddings)
- Improve documentation
- Add tests
