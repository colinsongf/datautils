import multiprocessing
from multiprocessing.pool import Pool
import numpy
import io
import os
from nltk import word_tokenize
from collections import Counter, OrderedDict
import cPickle as pkl

class Preprocessor(multiprocessing.Process):
  _processors = {
                 'lowercase': lambda x: x.lower(), 
                 'tokenize': lambda x: word_tokenize(x)
                }

  def __init__(self, task_queue, counter_queue, options):
    super(Preprocessor, self).__init__()
    self.task_queue = task_queue
    self.counter_queue = counter_queue

    self.validate_options(options)

  def validate_options(self, options):
    # Get is used for shared options
    self.output_dir = options.get('output_dir', './')

    # Encoding for reading and writing files
    self.encoding = options.get('encoding', 'utf8')

    # Whether to save each column into separate files
    self.separate_columns = options.get('separate_columns', False)
    
    # Expected number of columns per text line
    if self.separate_columns:
      self.separate_dict = options.get('separate_dict', False)
      self.num_columns = options.get('num_columns')

    # How to process each column of a text line
    self.processors = []
    for processor in options['processors']:
      assert processor in self._processors or callable(processor), \
        'Processor neither recognizable nor callable: %s' % (processor)
      # each processor is a callable function now
      self.processors.append(processor if callable(processor) else self._processors.get(processor))

    # How to split each text line into columns
    self.delimiter = options.get('delimiter', u'\t')

    # Whether to do word / symbol count
    self.count_sym = options.get('count_sym', False)
    if self.count_sym:
      assert self.counter_queue is not None, \
        'Cannot count symbols without a counter Queue'

  def preprocess(self, file_path):
    # Create counter(s)
    if self.count_sym:
      if self.separate_dict:
        counters = []
        for cidx in xrange(self.num_columns):
          counters.append(Counter())
      else:
        counter = Counter()

    with io.open(file_path, 'r', encoding=self.encoding) as fi:
      # open output file(s)
      basename = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
      if self.separate_columns:
        fos = [io.open(os.path.join(self.output_dir, '%s.out.%d' % (basename, cidx)), 
                                    'w', encoding=self.encoding) 
               for cidx in xrange(self.num_columns)]
      else:
        fo = io.open(os.path.join(self.output_dir, '%s.out' % (basename)), 
                     'w', encoding=self.encoding)
      
      # process each line and write to corresponding file(s)
      for lidx, line in enumerate(fi):
        columns = line.strip().split(self.delimiter)
        if len(columns) != self.num_columns:
            continue
        assert len(columns) == self.num_columns, 'Number of columns does not match'
        for cidx, column in enumerate(columns):
          # process the column
          processed = column
          for p in self.processors:
            processed = p(processed)

          # update counter if needed
          if self.count_sym:
            if self.separate_dict:
              for cidx in xrange(self.num_columns):
                counters[cidx].update(processed)
            else:
              counter.update(processed)
          
          # write to file(s)
          if self.separate_columns:
            fos[cidx].write(u' '.join(processed) + u'\n')
          else:
            fo.write(u' '.join(processed))
            fo.write(u'\n' if (not self.separate_columns) or cidx+1 == self.num_columns else self.delimiter)

      # close output file(s)
      if self.separate_columns:
        for fo in fos:
          fo.close()
      else:
        fo.close()

    if self.count_sym:
      if self.separate_dict:
        for cidx in xrange(self.num_columns):
          self.counter_queue[cidx].put(counters[cidx])
      else:
        self.counter_queue.put(counter)

  def run(self):
    proc_name = self.name
    while True:
      file_path = self.task_queue.get()
      if file_path is None:
        print '%s: Exiting' % proc_name
        self.task_queue.task_done()
        break

      print '%s: %s' % (proc_name, file_path)

      self.preprocess(file_path = file_path)
      self.task_queue.task_done()
    
    return

class Binarizer(multiprocessing.Process):
  def __init__(self, task_queue, sym2idx_dict, options):
    super(Binarizer, self).__init__()
    self.task_queue = task_queue
    self.sym2idx_dict = sym2idx_dict

    self.validate_options(options)

  def validate_options(self, options):
    # Encoding for reading and writing files
    self.encoding = options.get('encoding', 'utf8')

    # Whether to delete intermediate files
    self.clean_files = options.get('clean_files', True)

  def binarize(self, file_path):
    unk = self.sym2idx_dict['<unk>']
    
    with io.open(file_path, 'r', encoding=self.encoding) as fi:
      with io.open(file_path + '.bin', 'w', encoding=self.encoding) as fo:
        for lidx, line in enumerate(fi):
          indices = map(lambda x: self.sym2idx_dict.get(x, unk), line.strip().split())
          binary = u' '.join([unicode(x) for x in indices])
          fo.write(binary + u'\n')

    if self.clean_files:
      os.remove(file_path)

  def run(self):
    proc_name = self.name
    while True:
      file_path = self.task_queue.get()
      if file_path is None:
        print '%s: Exiting' % proc_name
        self.task_queue.task_done()
        break

      print '%s: %s' % (proc_name, file_path)

      self.binarize(file_path = file_path)
      self.task_queue.task_done()


class PreprocessPipeline(object):

  def __init__(self, file_list, options):
    self.file_list = file_list
    self.num_file = len(file_list)
    self.validate_options(options)

  def share_options(self, options):
    # Encoding for reading and writing files
    self.encoding = options.get('encoding', 'utf8')

    # Whether to save each column into separate files
    self.separate_columns = options.get('separate_columns', False)
    
    # Expected number of columns per text line
    if self.separate_columns:
      self.separate_dict = options.get('separate_dict', False)
      self.num_columns = options.get('num_columns')
      print self.separate_dict

    # Whether to delete intermediate files
    self.clean_files = options.get('clean_files', True)

  def validate_options(self, options):
    # Get is used for shared options
    self.output_dir = options.get('output_dir', './')

    # Number of processes to use
    #     - Default: min(num_file, cpu_count * 2), assuming hyper-threading
    self.num_process = min(self.num_file, options.get('num_process', multiprocessing.cpu_count() * 2))
    assert self.num_process > 0, 'Number of processes %d no larger than 0' % (self.num_process)

    # Whether to create dictionary from processed files
    self.create_dict = options.get('create_dict', True)
    self.special_sym = options.get('special_sym', [u'<unk>', u'<eos>'])
    
    # Whether to created binarized files (transform symbols to indices)
    self.binarize = options.get('binarize', False)
    if self.binarize:
      self.create_dict = True

    # Whether to combine files into a single large file
    self.combine_files = options.get('combine_files', False)

    # After master-only options, the rest should be worker options
    self.worker_options = options

    # In order to create dictionary, workers must count symbols
    if self.create_dict:
      self.worker_options['count_sym'] = True

    # If combine files, share some additional options with workers
    if self.combine_files:
      self.share_options(self.worker_options)

  def combine(self, cidx=None):
    suffix = '.out' if cidx is None else '.out.%d' % (cidx)
    if self.binarize:
      suffix += '.bin'

    with io.open(os.path.join(self.output_dir, 'combined'+suffix), \
                 'w', encoding=self.encoding) as fo:
      for file_path in self.file_list:
        basename = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        part_file_path = os.path.join(self.output_dir, basename+suffix)
        with io.open(part_file_path, 'r', encoding=self.encoding) as fi:
          for lidx, line in enumerate(fi):
            fo.write(line)

        if self.clean_files:
          os.remove(part_file_path)

  def create_dictionary(self, cidx=None):
    # Get corresponding queue
    if cidx is not None:
      counter_queue = self.counter_queue[cidx]
    else:
      counter_queue = self.counter_queue

    # Empty counter
    counter = Counter()
    for idx in xrange(len(self.file_list)):
      counter += counter_queue.get()

    sym2idx_dict = OrderedDict()
    for sym in self.special_sym:
      sym2idx_dict[sym] = len(sym2idx_dict)
      
    for sym, count in counter.most_common():
      sym2idx_dict[sym] = len(sym2idx_dict)

    idx2sym_dict = OrderedDict()
    for sym, idx in sym2idx_dict.iteritems():
      idx2sym_dict[idx] = sym

    if cidx is not None:
      suffix = '.%d.pkl' % (cidx)
    else:
      suffix = '.pkl'

    pkl.dump(sym2idx_dict, file(os.path.join(self.output_dir, 'dict.sym2idx%s' % (suffix)), 'w'))
    pkl.dump(idx2sym_dict, file(os.path.join(self.output_dir, 'dict.idx2sym%s' % (suffix)), 'w'))

  def para_process(self):
    # Create task Queue and put tasks
    self.task_queue = multiprocessing.JoinableQueue()
    for file_path in self.file_list:
      self.task_queue.put(file_path)

    # Create counter Queue is needed
    if self.create_dict:
      if self.separate_dict:
        self.counter_queue = []
        for cidx in xrange(self.num_columns):
          self.counter_queue.append(multiprocessing.Queue())
      else:
        self.counter_queue = multiprocessing.Queue()
    else:
      self.counter_queue = None

    # Spawn workers
    self.workers = []
    for widx in xrange(self.num_process):
      worker = Preprocessor(self.task_queue, self.counter_queue, self.worker_options)
      worker.start()
      self.workers.append(worker)
      # use None to signal end of tasks
      self.task_queue.put(None)

    # Wait for all tasks to finish
    self.task_queue.join()

  def para_binarize(self, cidx=None):
    if cidx is not None:
      suffix = '.%d.pkl' % (cidx)
    else:
      suffix = '.pkl'
    
    sym2idx_dict = pkl.load(file(os.path.join(self.output_dir, 'dict.sym2idx%s' % (suffix)), 'r'))

    # Create task Queue and put tasks
    self.task_queue = multiprocessing.JoinableQueue()
    for file_path in self.file_list:
      basename = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
      part_file_path = os.path.join(self.output_dir, basename+'.out')
      # Separate dict for each column file
      if cidx is not None:
        self.task_queue.put('%s.%d' % (part_file_path, cidx))
      else:
        # Shared dict for each column file
        if self.separate_columns:
          for cidx in xrange(self.num_columns):
            self.task_queue.put('%s.%d' % (part_file_path, cidx))
        # Shared dict for one whole file
        else:
          self.task_queue.put(part_file_path)

    # Spawn workers
    self.workers = []
    for widx in xrange(self.num_process):
      worker = Binarizer(self.task_queue, sym2idx_dict, self.worker_options)
      worker.start()
      self.workers.append(worker)
      # use None to signal end of tasks
      self.task_queue.put(None)

    # wait for all tasks to finish
    self.task_queue.join()

  def start(self):
    # Create output directory
    if not os.path.exists(self.output_dir):
      os.mkdir(self.output_dir)

    # Parallel process
    self.para_process()

    # create dictionary if needed
    if self.create_dict:
      if self.separate_dict:
        for cidx in xrange(self.num_columns):
          self.create_dictionary(cidx)
      else:
        self.create_dictionary()

    # Parallel binarize
    if self.binarize:
      if self.separate_dict:
        for cidx in xrange(self.num_columns):
          self.para_binarize(cidx)
      else:
        self.para_binarize()

    # reduce
    if self.combine_files:
      if self.separate_columns:
        for cidx in xrange(self.num_columns):
          self.combine(cidx)
      else:
        self.combine()
