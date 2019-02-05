import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from .sampler import SequentialSampler, RandomSampler
import collections
import math
import sys
import traceback
import threading
from pprint import pprint
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)
import time

import _pickle as pickle


_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _data_loop(preprocess, raw_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = raw_queue.get()
        if r is -1:
            # raw_queue.put(-1)
            break
        idx, samples = r
        batch = collate_fn([preprocess(raw) for raw in samples])
        # print('worker', raw_queue.qsize)
        data_queue.put((idx, batch))

        # try:
        #     batch = collate_fn([preprocess.transform(raw) for raw in samples])
        # except Exception:
        #     data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        # else:
        #     data_queue.put((idx, batch))

def _raw_loop(dataset, index_queue, raw_queue, num_workers):
    torch.set_num_threads(1)
    dataset_loop = dataset
    while True:
        r = index_queue.get()
        if r is None:
            for _ in range(num_workers):
                raw_queue.put(-1)
        else:
        # send_idx, [indices]
            idx, batch_indices = r
            # Flag for reconfiguring dataset
            if idx == -1:
                batch_indices.data_cache = dataset_loop.data_cache
                dataset_loop = batch_indices
            else:
                samples = [dataset_loop[i] for i in batch_indices]
                raw_queue.put((idx, samples))
            # try:
            #     samples = [dataset[i] for i in batch_indices]
            # except Exception:
            #     raw_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            # else:
            #     raw_queue.put((idx, samples))

def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch

class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.preprocess = loader.preprocess
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.drop_last = loader.drop_last
        self.done_event = threading.Event()

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)  

        if self.num_workers > 0:
            if self.loader.process_data is None:
                self.index_queue = multiprocessing.Queue()
                self.raw_queue = multiprocessing.Queue()
                self.dataset_process = multiprocessing.Process(
                    target=_raw_loop, 
                    args=(self.dataset, self.index_queue, self.raw_queue, self.num_workers))
                self.dataset_process.daemon = True
                self.dataset_process.start()
            else:
                self.index_queue = self.loader.process_data[0]
                self.raw_queue = self.loader.process_data[1]
                self.dataset_process = self.loader.process_data[2]
                self.index_queue.put((-1, self.dataset))
            self.data_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_data_loop,
                    args=(self.preprocess, self.raw_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]
            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop    
            for _ in range(2 * self.num_workers):
                self._put_indices()


    # number of batches
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            if self.drop_last and self.samples_remaining < self.batch_size:
                raise StopIteration
            if self.samples_remaining == 0:
                raise StopIteration
            indices = self._next_indices()
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch

    # puts [indices] into index_queue, relies on sr and dl, increases bo, si 
    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        if self.samples_remaining > 0:
            if self.samples_remaining < self.batch_size and self.drop_last:
                self._next_indices()
            else:
                self.index_queue.put((self.send_idx, self._next_indices()))
                self.batches_outstanding += 1

                self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            self.index_queue.put(None)
            self.loader.process_data = self._get_process_data()
            for w in self.workers:
                w.join()

    def _get_process_data(self):
        return [self.index_queue, self.raw_queue, self.dataset_process]

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()




class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional)
        pin_memory (bool, optional)
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=10,
                 collate_fn=default_collate, pin_memory=False, drop_last=False):
        if num_workers == 0 :
            raise NotImplementedError('numworkes must be more than 0')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.process_data = None

        self.reconfigure_dataset(dataset)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def reconfigure_dataset(self, dataset):
        self.dataset = dataset
        self.preprocess = self.dataset.preprocess
        self._set_sampler()

    def reconfigure_batch_size(self, batch_size):
        self.batch_size = batch_size

    def reconfigure_shuffle(self, shuffle):
        self.shuffle = shuffle

    def _set_sampler(self):  
        if self.shuffle is True:
            self.sampler = RandomSampler(self.dataset)
        elif self.shuffle is False:
            self.sampler = SequentialSampler(self.dataset)
        else:
            self.sampler = np.linspace(0, len(self.dataset)-1, self.sampler).astype(np.int)



# class DataLoaderIter(object):
#     "Iterates once over the DataLoader's dataset, as specified by the sampler"

#     def __init__(self, loader):
#         self.dataset = loader.dataset
#         self.batch_size = loader.batch_size
#         self.collate_fn = loader.collate_fn
#         self.sampler = loader.sampler
#         self.num_workers = loader.num_workers
#         self.pin_memory = loader.pin_memory
#         self.drop_last = loader.drop_last
#         self.done_event = threading.Event()

#         self.samples_remaining = len(self.sampler)
#         self.sample_iter = iter(self.sampler)    


#         if self.num_workers > 0:
#             self.index_queue = multiprocessing.SimpleQueue()
#             self.data_queue = multiprocessing.SimpleQueue()
#             self.raw_queue = multiprocessing.SimpleQueue()
#             self.batches_outstanding = 0
#             self.shutdown = False
#             self.send_idx = 0
#             self.rcvd_idx = 0
#             self.reorder_dict = {}

#             self.batch_idx = 0
#             while self.samples_remaining != 0:
#                 self.index_queue.put(self.batch_idx, self._next_indices())
#                 self.batch_idx += 1

#             self.workers = [
#                 multiprocessing.Process(
#                     target=_worker_loop,
#                     args=(self.dataset, self.raw_queue, self.data_queue, self.collate_fn))
#                 for _ in range(self.num_workers)]

#             for w in self.workers:
#                 w.daemon = True  # ensure that the worker exits on process exit
#                 w.start()

#             if self.pin_memory:
#                 in_data = self.data_queue
#                 self.data_queue = queue.Queue()
#                 self.pin_thread = threading.Thread(
#                     target=_pin_memory_loop,
#                     args=(in_data, self.data_queue, self.done_event))
#                 self.pin_thread.daemon = True
#                 self.pin_thread.start()

#             # prime the prefetch loop
#             for _ in range(2 * self.num_workers):
#                 self._put_indices()




#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size

#     def __next__(self):
#         if self.num_workers == 0:  # same-process loading
#             if self.drop_last and self.samples_remaining < self.batch_size:
#                 raise StopIteration
#             if self.samples_remaining == 0:
#                 raise StopIteration
#             indices = self._next_indices()
#             batch = self.collate_fn([self.dataset[i] for i in indices])
#             if self.pin_memory:
#                 batch = pin_memory_batch(batch)
#             return batch

#         # check if the next sample has already been generated
#         if self.rcvd_idx in self.reorder_dict:
#             batch = self.reorder_dict.pop(self.rcvd_idx)
#             return self._process_next_batch(batch)

#         if self.batches_outstanding == 0:
#             self._shutdown_workers()
#             raise StopIteration

#         while True:
#             assert (not self.shutdown and self.batches_outstanding > 0)
#             idx, batch = self.data_queue.get()
#             self.batches_outstanding -= 1
#             if idx != self.rcvd_idx:
#                 # store out-of-order samples
#                 self.reorder_dict[idx] = batch
#                 continue
#             return self._process_next_batch(batch)

#     next = __next__  # Python 2 compatibility

#     def __iter__(self):
#         return self

#     # def _next_indices(self):
#     #     batch_size = min(self.samples_remaining, self.batch_size)
#     #     batch = [next(self.sample_iter) for _ in range(batch_size)]
#     #     self.samples_remaining -= len(batch)
#     #     return batch

#     def _next_raw(self):
#         raw = self.raw_queue.get()

#         batch_size = min(self.samples_remaining, self.batch_size)
#         batch = [next(self.sample_iter) for _ in range(batch_size)]
#         self.samples_remaining -= len(batch)
#         return batch

#     def _put_raw(self):
#         assert self.batches_outstanding < 2 * self.num_workers
#         if self.samples_remaining > 0:
#             if self.samples_remaining < self.batch_size and self.drop_last:
#                 self._next_raw()
#             else:
#                 self.raw_queue.put((self.send_idx, self._next_raw()))
#                 self.batches_outstanding += 1
#                 self.send_idx += 1

#     def _process_next_batch(self, batch):
#         self.rcvd_idx += 1
#         self._put_indices()
#         if isinstance(batch, ExceptionWrapper):
#             raise batch.exc_type(batch.exc_msg)
#         return batch

#     def __getstate__(self):
#         # TODO: add limited pickling support for sharing an iterator
#         # across multiple threads for HOGWILD.
#         # Probably the best way to do this is by moving the sample pushing
#         # to a separate thread and then just sharing the data queue
#         # but signalling the end is tricky without a non-blocking API
#         raise NotImplementedError("DataLoaderIterator cannot be pickled")



#     def _shutdown_workers(self):
#         if not self.shutdown:
#             self.shutdown = True
#             self.done_event.set()
#             for _ in self.workers:
#                 self.raw_queue.put(None)

#     def _shutdown_raw(self):
#         if not self.shutdown:
#             self.shutdown_raw = True  
#             self.index_queue.put(None)      

#     def __del__(self):
#         if self.num_workers > 0:
#             self._shutdown_workers()


# class DataLoader(object):
#     """
#     Data loader. Combines a dataset and a sampler, and provides
#     single- or multi-process iterators over the dataset.

#     Arguments:
#         dataset (Dataset): dataset from which to load the data.
#         batch_size (int, optional): how many samples per batch to load
#             (default: 1).
#         shuffle (bool, optional): set to ``True`` to have the data reshuffled
#             at every epoch (default: False).
#         sampler (Sampler, optional): defines the strategy to draw samples from
#             the dataset. If specified, the ``shuffle`` argument is ignored.
#         num_workers (int, optional): how many subprocesses to use for data
#             loading. 0 means that the data will be loaded in the main process
#             (default: 0)
#         collate_fn (callable, optional)
#         pin_memory (bool, optional)
#         drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
#             if the dataset size is not divisible by the batch size. If False and
#             the size of dataset is not divisible by the batch size, then the last batch
#             will be smaller. (default: False)
#     """

#     def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
#                  collate_fn=default_collate, pin_memory=False, drop_last=False):
#         assert (num_workers > 0 and num_workers < 1000), 'num_workers should be over 0'
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.collate_fn = collate_fn
#         self.pin_memory = pin_memory
#         self.drop_last = drop_last

#         if sampler is not None:
#             self.sampler = sampler
#         elif shuffle:
#             self.sampler = RandomSampler(dataset)
#         elif not shuffle:
#             self.sampler = SequentialSampler(dataset)

#     def __iter__(self):
#         return DataLoaderIter(self)

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size


