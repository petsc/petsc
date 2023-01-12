#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:59:46 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import os
import enum
import time
import queue
import multiprocessing as mp
import multiprocessing.queues
import clang.cindex as clx
import petsclinter  as pl

from ..util._timeout import timeout, TimeoutError

from ._diag   import DiagnosticManager
from ._linter import Linter

# directory names to exclude from processing, case sensitive
exclude_dir_names = {
  'tests', 'tutorials', 'output', 'input', 'python', 'benchmarks', 'docs', 'binding', 'contrib',
  'fsrc', 'f90-mod', 'f90-src', 'f90-custom', 'ftn-auto', 'ftn-custom', 'f2003-src', 'ftn-kernels',
}
# directory suffixes to exclude from processing, case sensitive
exclude_dir_suffixes  = ('.dSYM', '.DS_Store')
# file extensions to process, case sensitve
allow_file_extensions = ('.c', '.cpp', '.cxx', '.cu', '.cc', '.h', '.hpp', '.inc')

class WorkerPool(mp.queues.JoinableQueue):
  __slots__ = (
    'parallel', 'error_queue', 'return_queue', 'lock', 'workers', 'num_workers', 'timeout',
    'verbose', 'prefix', 'warnings', 'errors_left', 'errors_fixed', 'patches', 'linter'
  )

  class QueueSignal(enum.IntEnum):
    """
    Various signals to indicate return type on the data queue from child processes
    """
    WARNING      = enum.auto()
    UNIFIED_DIFF = enum.auto()
    ERRORS_LEFT  = enum.auto()
    ERRORS_FIXED = enum.auto()
    EXIT_QUEUE   = enum.auto()

  def __init__(self, num_workers, timeout=2, verbose=False, prefix=None, **kwargs):
    if num_workers < 0:
      num_workers = max(mp.cpu_count() - 1, 1)

    if prefix is None:
      prefix = '[ROOT]'

    super().__init__(num_workers, **kwargs, ctx=mp.get_context())
    self.num_workers  = num_workers
    self.timeout      = timeout
    self.verbose      = verbose
    self.prefix       = prefix
    self.warnings     = []
    self.errors_left  = []
    self.errors_fixed = []
    self.patches      = []
    if num_workers in {0, 1}:
      self.__print(f'Number of worker processes ({num_workers}) too small, disabling multiprocessing')
      self.parallel     = False
      self.error_queue  = None
      self.return_queue = None
      self.lock         = None
    else:
      self.__print(f'Number of worker processes ({num_workers}) sufficient, enabling multiprocessing')
      self.parallel     = True
      self.error_queue  = mp.Queue()
      self.return_queue = mp.Queue()
      self.lock         = mp.RLock()
    return

  def __del__(self):
    try:
      self.__shutdown()
    except TimeoutError:
      pass
    try:
      super().__del__()
    except AttributeError:
      pass
    return

  @timeout(seconds=10)
  def __shutdown(self, *args, **kwargs):
    if getattr(self, 'parallel', False):
      for worker in getattr(self, 'workers', []):
        try:
          worker.terminate()
        except:
          pass
    return

  def __print(self, *args, **kwargs):
    if self.verbose:
      if args or kwargs:
        pl.sync_print(self.prefix, *args, **kwargs)
    return

  def __consume_results(self):
    if not self.parallel:
      return

    self.check()
    return_q = self.return_queue
    try:
      qsize = return_q.qsize()
    except NotImplementedError:
      # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
      #
      # Note that this may raise NotImplementedError on Unix platforms like macOS where
      # sem_getvalue() is not implemented.
      qsize = '0' if return_q.empty() else 'unknown (not implemented on platform)'
    self.__print('Estimated number of results:', qsize)

    while not return_q.empty():
      try:
        packet = return_q.get(timeout=1)
      except queue.Empty:
        # this should never really happen since this thread is the only consumer of the
        # queue, but it's here just in case
        break
      for signal, data in packet:
        if signal == self.QueueSignal.ERRORS_LEFT:
          self.errors_left.extend(data)
        elif signal == self.QueueSignal.ERRORS_FIXED:
          self.errors_fixed.extend(data)
        elif signal == self.QueueSignal.UNIFIED_DIFF:
          self.patches.extend(data)
        elif signal == self.QueueSignal.WARNING:
          self.warnings.append(data)
        else:
          raise ValueError(f'Unknown data returned by return_queue {signal}, {data}')
    return

  def setup(self, compiler_flags, clang_lib=None, clang_options=None, werror=False):
    if clang_lib is None:
      assert clx.conf.loaded, 'Must initialize libClang first'
      clang_lib = clx.conf.get_filename()

    if clang_options is None:
      clang_options = pl.util.base_clang_options

    if self.parallel:
      from ..queue_main       import queue_main
      from ..checks._register import check_function_map, classid_map

      self.workers = [
        mp.Process(
          target=queue_main,
          args=(
            clang_lib, check_function_map, classid_map, DiagnosticManager, compiler_flags,
            clang_options, self.verbose, werror, self.error_queue, self.return_queue, self, self.lock
          ),
          name=f'[{i}]'
        )
        for i in range(self.num_workers)
      ]

      for worker in self.workers:
        worker.start()
    else:
      self.linter = Linter(
        compiler_flags,
        clang_options=clang_options, prefix=self.prefix, verbose=self.verbose, werror=werror
      )
    return self

  def walk(self, src_path, exclude_dirs=None, exclude_dir_suff=None, allow_file_suff=None):
    if exclude_dirs is None:
      exclude_dirs = exclude_dir_names
    if exclude_dir_suff is None:
      exclude_dir_suff = exclude_dir_suffixes
    if allow_file_suff is None:
      allow_file_suff = allow_file_extensions

    if src_path.is_file():
      self.put(src_path)
    else:
      _, initial_dirs, _ = next(os.walk(src_path))
      initial_dirs = [d for d in initial_dirs if d not in exclude_dirs]
      initial_dirs = {str(src_path/d) for d in initial_dirs if not d.endswith(exclude_dir_suff)}
      for root, dirs, files in os.walk(src_path):
        self.__print('Processing directory', root)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dirs[:] = [d for d in dirs if not d.endswith(exclude_dir_suff)]
        for filename in (os.path.join(root, f) for f in files if f.endswith(allow_file_suff)):
          self.put(filename)
        # Every time we reach another top-level node we consume some of the results. This
        # makes the eventual consume-until-empty loop much faster since the queue is not
        # as backed up
        if root in initial_dirs:
          self.__consume_results()
    return self

  def put(self, filename):
    if self.parallel:
      # continuously put files onto the queue, if the queue is full we block for
      # queueTimeout seconds and if we still cannot insert to the queue we check
      # children for errors. If no errors are found we try again.
      while 1:
        try:
          super().put(filename, True, self.timeout)
        except queue.Full:
          # we don't want to join here since a child may have encountered an error!
          self.check()
        else:
          # only get here if put is successful
          break
    else:
      err_left, err_fixed, warnings, patches = self.linter.parse(filename).diagnostics()
      self.errors_left.extend(err_left)
      self.errors_fixed.extend(err_fixed)
      self.warnings.append(warnings)
      self.patches.extend(patches)
    return

  def check(self):
    if not self.parallel:
      return

    stop_multiproc = False
    timeout_it     = 0
    max_timeouts   = 3
    while not self.error_queue.empty():
      # while this does get recreated for every error, we do not want to needlessly
      # reinitialize it when no errors exist. If we get to this point however we no longer
      # care about performance as we are about to crash everything.
      try:
        exception = self.error_queue.get(timeout=.5)
      except queue.Empty:
        # Queue is not empty (we were in the loop), but we timed out on the get. Should
        # not happen yet here we are. Try a few couple more times, otherwise bail
        timeout_it += 1
        if timeout_it > max_timeouts:
          break
        continue

      err_bars = ''.join(['[ERROR]', 85 * '-', '[ERROR]\n'])
      try:
        err_mess = f'{err_bars}{str(exception)}{err_bars}'
      except:
        err_mess = exception

      print(err_mess, flush=True)
      stop_multiproc = True
      timeout_it     = 0
    if stop_multiproc:
      raise RuntimeError('Error in child process detected')
    return

  def finalize(self):
    if self.parallel:
      # join here to colocate error messages if needs be
      self.join()
      self.check()
      # send stop-signal to child processes
      for _ in self.workers:
        self.put(self.QueueSignal.EXIT_QUEUE)

      self.__consume_results()
      # If there is a lot of data being sent from the child processes, they may not fully
      # flush in time for us to catch their output in the first consume_results() call.
      #
      # We need to spin (and continue consuming results) until all processes have have
      # exited, since they will only fully exit once they flush their pipes.
      max_timeout = 60 # seconds
      start       = time.time()
      while time.time() - start <= max_timeout:
        live_list = [w.is_alive() for w in self.workers]
        for worker, alive in zip(self.workers, live_list):
          self.__print(
            'Checking whether process', worker.name, 'has finished:', 'no' if alive else 'yes'
          )
          if alive:
            worker.join(timeout=1)
          self.__consume_results()
        if sum(live_list) == 0:
          break
      else:
        alive = '\n'.join(
          f'{worker.name}: {"alive" if alive else "terminated"}' for worker, alive in zip(self.workers, live_list)
        )
        self.__shutdown()
        raise RuntimeError(f'Timed out! Workers failed to terminate:\n{alive}')

      self.error_queue.close()
      self.return_queue.close()

    self.errors_left  = [e for e in self.errors_left  if e] # remove any None's
    self.errors_fixed = [e for e in self.errors_fixed if e]
    self.warnings     = [w for w in self.warnings     if w]
    self.patches      = [p for p in self.patches      if p]
    return self.warnings, self.errors_left, self.errors_fixed, self.patches
