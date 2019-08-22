# ALWE
## Usage
to train word vectors:
```
word2vec.py [-h] -train FI -model FO [-cbow CBOW] [-negative NEG]
            [-dim DIM] [-alpha ALPHA] [-window WIN]
            [-min-count MIN_COUNT] [-processes NUM_PROCESSES]
            [-binary BINARY]

required arguments:
  -train FI                 Training file
  -model FO                 Output model file

optional arguments:
  -h, --help                show this help message and exit
  -cbow CBOW                1 for CBOW, 0 for skip-gram
  -negative NEG             Number of negative examples (>0) for negative sampling, 
                            0 for hierarchical softmax
  -dim DIM                  Dimensionality of word embeddings
  -alpha ALPHA              Starting learning rate
  -window WIN               Max window length
  -min-count MIN_COUNT      Min count for words used to learn <unk>
  -processes NUM_PROCESSES  Number of processes
  -binary BINARY            1 for output model in binary format, 0 otherwise
  ```
  
  # Implementation Details
  Written in Python 2.7.6 and NumPy 1.9.1.
