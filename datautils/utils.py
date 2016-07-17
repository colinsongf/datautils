import numpy

veclen = numpy.vectorize(len)

def padded_batch_fromstring(state, x, y, return_dict=False):
    """A function used to transform data in suitable format

    :type x: numpy.array
    :param x: the array is a batch of sentences in some source language
        - each sentence must be a list of words (str)

    :type y: numpy.array
    :param y: the array is a batch of sentences in some target language
        - each sentence must be a list of words (str)

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
    """
    # Ditch long text pairs
    if state['trim_batches']:
        valid_inputs = numpy.logical_and(
                veclen(x) <= state['maxlen_source'],
                veclen(y) <= state['maxlen_target']
            )

        if not numpy.any(valid_inputs == True):
            return None

        x = x[valid_inputs]
        y = y[valid_inputs]

    # Actual max lengths
    mx = max([len(xx) for xx in x])+1
    my = max([len(yy) for yy in y])+1

    # Batch size
    n = x.shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    dictX = state['dict_source']
    for idx in xrange(len(x)):
        # Insert sequence idx in a column of matrix X
        X[:len(x[idx]), idx] = map(lambda w: dictX.get(w, dictX['<unk>']), x[idx][:mx])
        # Mark the end of sentence
        if len(x[idx]) < mx:
            X[len(x[idx]):, idx] = dictX['<eos>']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[idx]), idx] = 1.
        if len(x[idx]) < mx:
            Xmask[len(x[idx]), idx] = 1. # only the fisrt '<eos>' matters

    # Fill Y and Ymask
    dictY = state['dict_target']
    for idx in xrange(len(y)):
        # Insert sequence idx in a column of matrix Y
        Y[:len(y[idx]), idx] = map(lambda w: dictY.get(w, dictY['<unk>']), y[idx][:my])
        # Mark the end of sentence
        if len(y[idx]) < my:
            Y[len(y[idx]):, idx] = dictY['<eos>']

        # Initialize Ymask column with ones in all positions that
        # were just set in Y
        Ymask[:len(y[idx]), idx] = 1.
        if len(y[idx]) < my:
            Ymask[len(y[idx]), idx] = 1. # only the fisrt '<eos>' matters

    # Unknown words
    # X[X >= state['n_sym_source']] = dictX['<unk>']
    # Y[Y >= state['n_sym_target']] = dictY['<unk>']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask


def padded_batch_frombinary(state, x, y, return_dict=False):
    """A function used to transform data in suitable format

    :type x: numpy.array of objects (list)
    :param x: the array is a batch of sentences in some source language
        - each sentence must be a list of integers (int)

    :type y: numpy.array of objects (list)
    :param y: the array is a batch of sentences in some target language
        - each sentence must be a list of integers (int)

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
    """

    # Ditch long text pairs
    if state['trim_batches']:
        valid_inputs = numpy.logical_and(
                veclen(x) <= state['maxlen_source'],
                veclen(y) <= state['maxlen_target']
            )

        if not numpy.any(valid_inputs == True):
            return None

        x = x[valid_inputs]
        y = y[valid_inputs]

    # Actual max lengths
    mx = max([len(xx) for xx in x])+1
    my = max([len(yy) for yy in y])+1

    # Batch size
    n = x.shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x)):
        # Insert sequence idx in a column of matrix X
        X[:len(x[idx]), idx] = x[idx][:mx]
        # Mark the end of sentence
        if len(x[idx]) < mx:
            X[len(x[idx]):, idx] = dictX['<eos>']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[idx]), idx] = 1.
        if len(x[idx]) < mx:
            Xmask[len(x[idx]), idx] = 1.

    # Fill Y and Ymask
    for idx in xrange(len(y)):
        # Insert sequence idx in a column of matrix Y
        Y[:len(y[idx]), idx] = y[idx][:my]
        # Mark the end of sentence
        if len(y[idx]) < my:
            Y[len(y[idx]):, idx] = dictY['<eos>']

        # Initialize Ymask column with ones in all positions that
        # were just set in Y
        Ymask[:len(y[idx]), idx] = 1.
        if len(y[idx]) < my:
            Ymask[len(y[idx]), idx] = 1.

    # Unknown words
    # X[X >= state['n_sym_source']] = dictX['<unk>']
    # Y[Y >= state['n_sym_target']] = dictY['<unk>']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask