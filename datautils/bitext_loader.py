import numpy

import sys
import os
import mmap

import threading
import Queue

import itertools
import operator

import utils

def to_byte(size):
    if size.endswith('K'):
        return float(size.rstrip('K')) * (2 << 10)
    elif size.endswith('M'):
        return float(size.rstrip('M')) * (2 << 20)
    elif size.endswith('G'):
        return float(size.rstrip('G')) * (2 << 30)
    else:
        return float(size)

def to_disp(size):
    size = float(size)
    if size > (2 << 30):
        return '%.2fG' % (size / (2 << 30))
    elif size > (2 << 20):
        return '%.2fM' % (size / (2 << 20))
    elif size > (2 << 10):
        return '%.2fK' % (size / (2 << 10))
    else:
        return '%dB' % (int(size))

class BitextFetcher(threading.Thread):
    def __init__(self, parent, process):
        threading.Thread.__init__(self)
        self.parent = parent
        self.process = process
        self.can_fit = False

    def __del__(self):
        if not self.can_fit:
            self._smm.close()
            self._tmm.close()

            self._sf.close()
            self._tf.close()

    def scan_textfile(self):
        diter = self.parent

        source_size = os.stat(diter.source_file).st_size
        target_size = os.stat(diter.target_file).st_size

        if source_size + target_size <= to_byte(diter.max_mem):
            self.can_fit = True

        if self.can_fit:
            print 'Load data (%s) into memory' % to_disp(source_size + target_size)

            self.sources = []
            self.targets = []

            with open(diter.source_file, 'r') as _sf:
                for line in _sf:
                    self.sources.append(line)

            with open(diter.target_file, 'r') as _tf:
                for line in _tf:
                    self.targets.append(line)

            assert len(self.sources) == len(self.targets), 'Number of lines do not match'

            self.data_len = len(self.sources)
        else:
            print 'Memory-map data (%s)' % to_disp(source_size + target_size)
            self.soffsets = []
            self.toffsets = []

            self._sf = open(diter.source_file, 'r+b')
            self._tf = open(diter.target_file, 'r+b')

            self._smm = mmap.mmap(self._sf.fileno(), 0, access = mmap.ACCESS_READ)
            while True:
                pos = self._smm.tell()
                line = self._smm.readline()
                if line == '':
                    break
                self.soffsets.append(pos)

            self._tmm = mmap.mmap(self._tf.fileno(), 0, access = mmap.ACCESS_READ)
            while True:
                pos = self._tmm.tell()
                line = self._tmm.readline()
                if line == '':
                    break
                self.toffsets.append(pos)

            assert len(self.soffsets) == len(self.toffsets), 'Number of lines do not match'

            self.data_len = len(self.soffsets)

        self.reset()

    def reset(self):
        self.shuf_idx = 0
        self.shuf_indices = numpy.random.permutation(self.data_len)

    def fetch_data(self):
        data_idx = self.shuf_indices[self.shuf_idx]

        if self.can_fit:
            source = self.process(self.sources[data_idx])
            target = self.process(self.targets[data_idx])
        else:
            soffset, toffset = self.soffsets[data_idx], self.toffsets[data_idx]
            source = self.process(self.read_mmdata(soffset, self._smm))
            target = self.process(self.read_mmdata(toffset, self._tmm))

        self.shuf_idx += 1

        return source, target

    def read_mmdata(self, offset, mm):
        try:
            mm.seek(offset)
            line = mm.readline()
            return line
        except:
            print 'Error at location: %d' % offset
            raise

    def run(self):
        self.scan_textfile()

        diter = self.parent

        while not diter.exit_flag:
            last_batch = False
            source_sents = []
            target_sents = []

            while len(source_sents) < diter.batch_size:
                if self.shuf_idx == self.data_len:
                    self.reset()
                    last_batch = True
                    break

                source, target = self.fetch_data()                

                if len(source) == 0 or len(source) > diter.max_len \
                    or len(target) == 0 or len(target) > diter.max_len:
                    print soffset, toffset
                    continue

                source_sents.append(source)
                target_sents.append(target)

            if len(source_sents) > 0:
                diter.queue.put([source_sents, target_sents])
            
            if last_batch:
                # use None to signal end of an epoch
                diter.queue.put(None)

                # return if not to use infinite loop
                if not diter.use_infinite_loop:
                    return

class BitextIterator(object):

    def __init__(self,
                 batch_size,
                 target_file,
                 source_file,
                 dtype="int64",
                 max_mem="2G",
                 queue_size=1000,
                 shuffle=True,
                 use_infinite_loop=True,
                 max_len=1000):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.exit_flag = False

    def start(self, process):
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = BitextFetcher(self, process)
        self.gather.setDaemon(True)
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        batch = self.queue.get()

        if not batch:
            raise StopIteration()

        return batch[0], batch[1]

class HomogenousBitextIterator(BitextIterator):

    def __init__(self, k_batches, *args, **kwargs):
        super(HomogenousBitextIterator, self).__init__(*args, **kwargs)
        self.k_batches = k_batches
        self.batch_iter = None

    def get_homogenous_batch_iter(self):
        while True:
            last_batch = False
            
            # load k batches for length-based bucketing
            data = []
            for k in range(self.k_batches):
                batch = self.queue.get()
                if batch is None:
                    last_batch = True
                    break
                data.append(batch)

            # cast to nparray: each element in the nparray is a list of words
            s = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
            t = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
            
            # argsort by length (the larger one between source and target)
            lens = numpy.asarray([map(len, s), map(len, t)])
            order = numpy.argsort(lens.max(axis=0)) if self.k_batches > 1 \
                    else numpy.arange(len(s))

            # yield batch with similar lengths
            for k in range(len(data)):
                indices = order[k * self.batch_size:(k+1) * self.batch_size]
                yield (s[indices], t[indices])
            
            if last_batch:
                yield None

    def next(self):
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter()

        batch = next(self.batch_iter)

        if not batch:
            raise StopIteration

        return batch