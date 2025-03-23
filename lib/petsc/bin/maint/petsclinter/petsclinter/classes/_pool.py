#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:59:46 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import os
import abc
import enum
import queue
import multiprocessing as mp
import multiprocessing.synchronize
import petsclinter     as pl

from .._typing import *

from ..util._timeout import timeout, TimeoutError

from ._linter import Linter
from ._path   import Path

_T = TypeVar('_T')

# directory names to exclude from processing, case sensitive
exclude_dir_names = {
  'tests', 'tutorials', 'output', 'input', 'python', 'benchmarks', 'docs', 'binding', 'contrib',
  'ftn-mod', 'ftn-src', 'ftn-custom', 'ftn-kernels',
  'perfstubs', 'yaml'
}
# directory suffixes to exclude from processing, case sensitive
exclude_dir_suffixes  = ('.dSYM', '.DS_Store')
# file extensions to process, case sensitve
allow_file_extensions = ('.c', '.cpp', '.cxx', '.cu', '.cc', '.h', '.hpp', '.inc')

class WorkerPoolBase(abc.ABC):
  __slots__ = ('verbose', 'warnings', 'errors_left', 'errors_fixed', 'patches')

  verbose: int
  warnings: list[CondensedDiags]
  errors_left: list[CondensedDiags]
  errors_fixed: list[CondensedDiags]
  patches: list[PathDiffPair]

  class QueueSignal(enum.IntEnum):
    """
    Various signals to indicate return type on the data queue from child processes
    """
    FILE_PATH  = enum.auto()
    EXIT_QUEUE = enum.auto()

  def __init__(self, verbose: int) -> None:
    r"""Construct a `WoekerPoolBase`

    Parameters
    ----------
    verbose :
      whether to print verbose logging output (at level)
    """
    super().__init__()
    self.verbose      = verbose
    self.warnings     = []
    self.errors_left  = []
    self.errors_fixed = []
    self.patches      = []
    return

  def _vprint(self: PoolImpl, *args, **kwargs) -> None:
    r"""Prints output, but only in verbose mode"""
    if self.verbose:
      pl.sync_print(*args, **kwargs)
    return

  @abc.abstractmethod
  def _setup(self: PoolImpl, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    return

  @abc.abstractmethod
  def _consume_results(self) -> None:
    return

  @abc.abstractmethod
  def _finalize(self) -> None:
    return

  @abc.abstractmethod
  def put(self: PoolImpl, item: PathLike) -> None:
    raise NotImplementedError

  def setup(self: PoolImpl, compiler_flags: list[str], clang_lib: Optional[PathLike] = None, clang_options: Optional[CXTranslationUnit] = None, clang_compat_check: bool = True, werror: bool = False) -> PoolImpl:
    r"""Set up a `WorkerPool` instance

    Parameters
    ----------
    compiler_flags :
      the list of compiler flags to pass to the `Linter`
    clang_lib : optional
      the path to libclang
    clang_options: optional
      the options to pass to the `Linter`, defaults to `petsclinter.util.base_clang_options`
    clang_compat_check: optional
      whether to do compatibility checks (if this initializes libclang)
    werror:
      whether to treat warnings as errors

    Returns
    -------
    self:
      the `WorkerPool` instance
    """
    if clang_lib is None:
      import clang.cindex as clx # type: ignore[import]
      assert clx.conf.loaded, 'Must initialize libClang first'
      clang_lib = clx.conf.get_filename()

    if clang_options is None:
      clang_options = pl.util.base_clang_options

    self._setup(compiler_flags, clang_lib, clang_options, clang_compat_check, werror)
    return self

  def walk(self: PoolImpl, src_path_list: Sequence[PathLike], exclude_dirs: Optional[Collection[str]] = None, exclude_dir_suff: Optional[tuple[str, ...]] = None, allow_file_suff: Optional[tuple[str, ...]] = None) -> PoolImpl:
    r"""Walk `src_path_list` and process it

    Parameters
    ----------
    src_path_list :
      a list of paths to process
    exclude_dirs : optional
      a list or set to exclude from processing
    exclude_dir_suff : optional
      a set of suffixes to ignore
    allow_file_suff : optional
      a list of suffixes to explicitly allow

    Returns
    -------
    self :
      the `WorkerPool` instance
    """
    if exclude_dirs is None:
      exclude_dirs = exclude_dir_names
    if exclude_dir_suff is None:
      exclude_dir_suff = exclude_dir_suffixes
    if allow_file_suff is None:
      allow_file_suff = allow_file_extensions

    for src_path in src_path_list:
      if src_path.is_file():
        self.put(src_path)
        continue

      _, dirs, _   = next(os.walk(src_path))
      dir_gen      = (d for d in dirs if d not in exclude_dirs)
      initial_dirs = {str(src_path / d) for d in dir_gen if not d.endswith(exclude_dir_suff)}
      for root, dirs, files in os.walk(src_path):
        self._vprint('Processing directory', root)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dirs[:] = [d for d in dirs if not d.endswith(exclude_dir_suff)]
        for filename in (os.path.join(root, f) for f in files if f.endswith(allow_file_suff)):
          self.put(Path(filename))
        # Every time we reach another top-level node we consume some of the results. This
        # makes the eventual consume-until-empty loop much faster since the queue is not
        # as backed up
        if root in initial_dirs:
          self._consume_results()
    return self

  def finalize(self: PoolImpl) -> tuple[list[CondensedDiags], list[CondensedDiags], list[CondensedDiags], list[PathDiffPair]]:
    r"""Finalize the queue and return the results

    Returns
    -------
    warnings :
      the list of warnings
    errors_left :
      the remaining (unfixed) errors
    errors_fixed :
      the fixed errors
    patches :
      the generated patches

    Notes
    -----
    If running in parallel, and workers fail to finalize in time, calls `self.__crash_and_burn()`
    """
    def prune(container: list[_T]) -> list[_T]:
      return [item for item in container if item]

    self._finalize()
    warnings     = prune(self.warnings)
    errors_left  = prune(self.errors_left)
    errors_fixed = prune(self.errors_fixed)
    patches      = prune(self.patches)
    return warnings, errors_left, errors_fixed, patches

class ParallelPool(WorkerPoolBase):
  __slots__ = ('input_queue', 'error_queue', 'return_queue', 'lock', 'workers', 'num_workers')

  class SendPacket(NamedTuple):
    type: WorkerPoolBase.QueueSignal
    data: Optional[PathLike]

  class ReturnPacket(NamedTuple):
    patches: List[PathDiffPair]
    errors_left: CondensedDiags
    errors_fixed: CondensedDiags
    warnings: CondensedDiags

  # use Union to get around not having TypeAlias until 3.10
  CommandQueueType: TypeAlias = 'mp.JoinableQueue[SendPacket]'
  ErrorQueueType: TypeAlias   = 'mp.Queue[str]'
  ReturnQueueType: TypeAlias  = 'mp.Queue[ReturnPacket]'
  LockType: TypeAlias         = mp.synchronize.Lock
  input_queue: CommandQueueType
  error_queue: ErrorQueueType
  return_queue: ReturnQueueType
  lock: LockType
  workers: list[mp.Process]
  num_workers: int

  def __init__(self, num_workers: int, verbose: int) -> None:
    r"""Construct a `ParallelPool`

    Parameters
    ----------
    num_workers :
      how many worker processes to spawn
    verbose :
      whether to print verbose output
    """
    super().__init__(verbose)
    self.input_queue  = mp.JoinableQueue()
    self.error_queue  = mp.Queue()
    self.return_queue = mp.Queue()
    lock              = mp.Lock()
    self.lock         = lock
    self.workers      = []
    self.num_workers  = num_workers

    old_sync_print = pl.sync_print
    def lock_sync_print(*args, **kwargs) -> None:
      with lock:
        old_sync_print(*args, **kwargs)
      return
    pl.sync_print = lock_sync_print
    return

  @timeout(seconds=10)
  def __crash_and_burn(self, message: str) -> NoReturn:
    r"""Forcefully annihilate the pool and crash the program

    Parameters
    ----------
    message :
      an informative message to print on crashing

    Raises
    ------
    RuntimeError :
      raises a RuntimeError in all cases
    """
    for worker in self.workers:
      if worker is not None:
        try:
          worker.terminate()
        except:
          pass
    raise RuntimeError(message)

  def _consume_results(self) -> None:
    r"""Consume pending results from the queue

    Raises
    ------
    ValueError :
      if an unknown QueueSignal is returned from the pipe
    """
    self.check()
    return_q = self.return_queue
    try:
      qsize_mess = str(return_q.qsize())
    except NotImplementedError:
      # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
      #
      # Note that this may raise NotImplementedError on Unix platforms like macOS where
      # sem_getvalue() is not implemented.
      qsize_mess = '0' if return_q.empty() else 'unknown (not implemented on platform)'
    self._vprint('Estimated number of results:', qsize_mess)

    while not return_q.empty():
      try:
        packet = return_q.get(timeout=1)
      except queue.Empty:
        # this should never really happen since this thread is the only consumer of the
        # queue, but it's here just in case
        break
      if not isinstance(packet, self.ReturnPacket):
        try: # type: ignore[unreachable]
          self.__crash_and_burn('')
        except RuntimeError:
          pass
        raise ValueError(type(packet))
      self.errors_left.append(packet.errors_left)
      self.errors_fixed.append(packet.errors_fixed)
      self.patches.extend(packet.patches)
      self.warnings.append(packet.warnings)
    return

  def _setup(self, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    r"""Setup a `ParallelPool`

    Parameters
    ----------
    compiler_flags :
      the list of general compiler flags to parse the files with
    clang_lib :
      the path to libclang
    clang_options :
      the translation unit options to parse the files with
    clang_compat_check :
      whether to do compatibility checks for libclang
    werror :
      whether to consider warnings as errors

    Notes
    -----
    This routine is what actually spawns the processes
    """
    from ..queue_main       import queue_main
    from ..checks._register import check_function_map, classid_map
    from ._diag             import DiagnosticManager

    # in case we are double-calling this
    self._flush_workers()
    assert len(self.workers) == 0
    for i in range(self.num_workers):
      worker = mp.Process(
        target=queue_main,
        args=(
          clang_lib, clang_compat_check, check_function_map, classid_map, DiagnosticManager,
          compiler_flags, clang_options, self.verbose, werror, self.error_queue, self.return_queue,
          self.input_queue, self.lock
        ),
        name=f'[{i}]'
      )
      worker.start()
      self.workers.append(worker)
    return

  def _flush_workers(self) -> None:
    r"""Reap any existing workers

    Notes
    -----
    Tries to gracefully terminate any existing workers, but will crash if it is not able to. Does
    nothing if no workers active.
    """
    import time

    @timeout(seconds=3)
    def timeout_join() -> None:
      self.input_queue.join()
      return

    # join here to colocate error messages if needs be
    try:
      timeout_join()
    except TimeoutError:
      # OK if we timeout, likely there is a problem so this join is a deadlock
      pass
    self.check()
    # send stop-signal to child processes
    for _ in self.workers:
      self.put_packet(self.SendPacket(type=self.QueueSignal.EXIT_QUEUE, data=None))

    self._consume_results()
    # If there is a lot of data being sent from the child processes, they may not fully
    # flush in time for us to catch their output in the first consume_results() call.
    #
    # We need to spin (and continue consuming results) until all processes have have
    # exited, since they will only fully exit once they flush their pipes.
    @timeout(seconds=60)
    def reap_workers() -> None:
      while 1:
        live_list = [w.is_alive() for w in self.workers]
        for worker, alive in zip(self.workers, live_list):
          self._vprint(
            'Checking whether process', worker.name, 'has finished:', 'no' if alive else 'yes'
          )
          if alive:
            worker.join(timeout=1)
            self._consume_results()
        if sum(live_list) == 0:
          break
      return

    try:
      reap_workers()
    except TimeoutError:
      mess = '\n'.join(
        f'{worker.name}: {"alive" if worker.is_alive() else "terminated"}' for worker in self.workers
      )
      self.__crash_and_burn(f'Timed out! Workers failed to terminate:\n{mess}')
    self.workers = []
    return

  def _finalize(self) -> None:
    r"""Finalize a `ParallelPool`

    Notes
    -----
    In addition to reaping all workers, this also closes all queues
    """
    self._flush_workers()
    self.error_queue.close()
    self.return_queue.close()
    return

  def check(self) -> None:
    r"""Check for errors from the queue

    Notes
    -----
    Calls `self.__crash_and_burn()` if any errors are detected, but does nothing if running in
    serial
    """
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
      self.__crash_and_burn('Error in child process detected')
    return

  def put_packet(self, packet: SendPacket) -> None:
    r"""Put a `SendPacket` onto the queue

    Parameters
    ----------
    packet :
      the packet
    """
    # continuously put files onto the queue, if the queue is full we block for
    # queueTimeout seconds and if we still cannot insert to the queue we check
    # children for errors. If no errors are found we try again.
    assert isinstance(packet, self.SendPacket)
    while 1:
      try:
        self.input_queue.put(packet, True, 2)
      except queue.Full:
        # we don't want to join here since a child may have encountered an error!
        self.check()
        self._consume_results()
      else:
        # only get here if put is successful
        break
    return

  def put(self, path: PathLike) -> None:
    r"""Put an filepath into the `ParallelPool`s processing queue

    Parameters
    ----------
    path :
      the path to be processed
    """
    return self.put_packet(self.SendPacket(type=self.QueueSignal.FILE_PATH, data=path))

class SerialPool(WorkerPoolBase):
  __slots__ = ('linter',)

  linter: Linter

  def __init__(self, verbose: int) -> None:
    r"""Construct a `SerialPool`

    Parameters
    ----------
    verbose :
      whether to print verbose output
    """
    super().__init__(verbose)
    return

  def _setup(self, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    r"""Setup a `SerialPool`

    Parameters
    ----------
    compiler_flags :
      the list of general compiler flags to parse the files with
    clang_lib :
      the path to libclang
    clang_options :
      the translation unit options to parse the files with
    clang_compat_check :
      whether to do compatibility checks for libclang
    werror :
      whether to consider warnings as errors

    Notes
    -----
    This routine is what constructs the `Linter` instance
    """
    self.linter = Linter(
      compiler_flags, clang_options=clang_options, verbose=self.verbose, werror=werror
    )
    return

  def _consume_results(self) -> None:
    r"""Consume results for a `SerialPool`, does nothing"""
    return

  def _finalize(self) -> None:
    r"""Finalize a `SerialPool`, does nothing"""
    return

  def put(self, item: PathLike) -> None:
    r"""Put an item into a `SerialPool`s processing queue

    Parameters
    ----------
    item :
      the item to process
    """
    err_left, err_fixed, warnings, patches = self.linter.parse(item).diagnostics()
    self.errors_left.append(err_left)
    self.errors_fixed.append(err_fixed)
    self.warnings.append(warnings)
    self.patches.extend(patches)
    return

def WorkerPool(num_workers: int, verbose: int) -> Union[SerialPool, ParallelPool]:
  r"""Construct a `WorkerPool`

  Parameters
  ----------
  num_workers :
    the number of worker threads the pool should own
  verbose :
    what level to print verbose output at

  Returns
  -------
  pool :
    the pool instance

  Notes
  -----
  If `num_workers` is < 0, then the number of processes is automatically determined, usually equal to
  the number logical cores for the current machine.
  """
  if num_workers < 0:
    # take number of cores - 1, up to a maximum of 16 as not to overload big machines
    num_workers = min(max(mp.cpu_count() - 1, 1), 16)

  if num_workers in (0, 1):
    if verbose:
      pl.sync_print(f'Number of worker processes ({num_workers}) too small, disabling multiprocessing')
    return SerialPool(verbose)
  else:
    if verbose:
      pl.sync_print(f'Number of worker processes ({num_workers}) sufficient, enabling multiprocessing')
    return ParallelPool(num_workers, verbose)
