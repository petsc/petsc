#!/usr/bin/env python3
"""
# Created: Tue Jun 21 09:44:08 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import petsclinter     as pl

from .classes._diag   import DiagnosticManager
from .classes._path   import Path
from .classes._pool   import WorkerPoolBase, ParallelPool
from .classes._linter import Linter

from .util._timeout import timeout
from .util._utility import traceback_format_exception

from ._error  import BaseError
from ._typing import *

class MainLoopError(BaseError):
  """
  Thrown by child processes when they encounter an error in the main loop
  """
  def __init__(self, filename: str, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._main_loop_error_filename = filename
    return

@timeout(seconds=5)
def __handle_error(error_prefix: str, filename: str, error_queue: ParallelPool.ErrorQueueType, file_queue: ParallelPool.CommandQueueType, base_e: ExceptionKind) -> None:
  try:
    # attempt to send the traceback back to parent
    exception_trace = ''.join(traceback_format_exception(base_e))
    error_message   = f'{error_prefix} {filename}\n{exception_trace}'
    if not error_message.endswith('\n'):
      error_message += '\n'
    error_queue.put(error_message)
  except Exception as send_e:
    send_exception_trace = ''
    try:
      # if this fails then I guess we really are screwed
      send_exception_trace = ''.join(traceback_format_exception(send_e))
    except Exception as send_e2:
      send_exception_trace = str(send_e) + '\n\n' + str(send_e2)
    error_queue.put(f'{error_prefix} {filename}\n{send_exception_trace}\n')
  finally:
    try:
      # in case we had any work from the queue we need to release it but only after
      # putting our exception on the queue
      file_queue.task_done()
    except ValueError:
      # task_done() called more times than get(), means we threw before getting the
      # filename
      pass
  return

def __main_loop(cmd_queue: ParallelPool.CommandQueueType, return_queue: ParallelPool.ReturnQueueType, linter: Linter) -> None:
  try:
    while 1:
      ret = cmd_queue.get()
      assert isinstance(ret, ParallelPool.SendPacket)
      if ret.type == WorkerPoolBase.QueueSignal.EXIT_QUEUE:
        break
      if ret.type == WorkerPoolBase.QueueSignal.FILE_PATH:
        filename = ret.data
        assert isinstance(filename, Path)
      else:
        raise ValueError(f'Don\'t know what to do with Queue signal: {ret.type} -> {ret.data}')

      errors_left, errors_fixed, warnings, patches = linter.parse(filename).diagnostics()
      return_queue.put(
        ParallelPool.ReturnPacket(
          patches=patches,
          errors_left=errors_left,
          errors_fixed=errors_fixed,
          warnings=warnings
        )
      )
      cmd_queue.task_done()
  except Exception as exc:
    raise MainLoopError(str(filename)) from exc
  return

class LockPrinter:
  __slots__ = ('_verbose', '_print_prefix', '_lock')

  _verbose: bool
  _print_prefix: str
  _lock: ParallelPool.LockType

  def __init__(self, verbose: bool, print_prefix: str, lock: ParallelPool.LockType) -> None:
    r"""Construct a `LockPrinter`

    Parameters
    ----------
    verbose :
      whether to print at all
    print_prefix :
      the prefix string to prepend to all print output
    lock :
      the lock to acquire before printing
    """
    self._verbose      = verbose
    self._print_prefix = print_prefix
    self._lock         = lock
    return

  def __call__(self, *args, flush: bool = True, **kwargs) -> None:
    r"""Print stuff

    Parameters
    ----------
    args : optional
      the positional stuff to print
    flush : optional
      whether to flush the stream after printing
    kwargs : optional
      additional keyword arguments to send to `print()`

    Notes
    -----
    If called empty (i.e. `args` and `kwargs`) this does nothing
    """
    if self._verbose:
      if args or kwargs:
        with self._lock:
          print(self._print_prefix, *args, flush=flush, **kwargs)
    return

def queue_main(
    clang_lib: PathLike,
    clang_compat_check: bool,
    updated_check_function_map: dict[str, FunctionChecker],
    updated_classid_map: dict[str, str],
    updated_diagnostics_mngr: DiagnosticsManagerCls,
    compiler_flags: list[str],
    clang_options: CXTranslationUnit,
    verbose: bool,
    werror: bool,
    error_queue: ParallelPool.ErrorQueueType,
    return_queue: ParallelPool.ReturnQueueType,
    file_queue: ParallelPool.CommandQueueType,
    lock: ParallelPool.LockType
) -> None:
  """
  main function for worker processes in the queue, does pretty much the same thing the
  main process would do in their place
  """
  def update_globals() -> None:
    from .checks import _register

    _register.check_function_map = copy.deepcopy(updated_check_function_map)
    _register.classid_map        = copy.deepcopy(updated_classid_map)
    DiagnosticManager.disabled   = copy.deepcopy(updated_diagnostics_mngr.disabled)
    return

  # in case errors are thrown before setup is complete
  error_prefix = '[UNKNOWN_CHILD]'
  filename     = 'QUEUE SETUP'
  printbar     = 15 * '='
  try:
    # initialize the global variables
    proc         = mp.current_process().name
    print_prefix = proc + ' --'[:len('[ROOT]') - len(proc)]
    error_prefix = f'{print_prefix} Exception detected while processing'

    update_globals()
    # removing the type: ignore would require us to type-annotate sync_print in
    # __init__.py. However, __init__.py does a version check so we cannot put stuff (like
    # type annotations) that may require a higher version of python to even byte-compile.
    pl.sync_print = LockPrinter(verbose, print_prefix, lock) # type: ignore[assignment]
    pl.sync_print(printbar, 'Performing setup', printbar)
    # initialize libclang, and create a linter instance
    pl.util.initialize_libclang(clang_lib=clang_lib, compat_check=clang_compat_check)
    linter = Linter(compiler_flags, clang_options=clang_options, verbose=verbose, werror=werror)
    pl.sync_print(printbar, 'Entering queue  ', printbar)

    # main loop
    __main_loop(file_queue, return_queue, linter)
  except Exception as base_e:
    try:
      if isinstance(base_e, MainLoopError):
        filename = base_e._main_loop_error_filename
    except:
      pass
    try:
      __handle_error(error_prefix, str(filename), error_queue, file_queue, base_e)
    except:
      pass
  try:
    pl.sync_print(printbar, 'Exiting queue   ', printbar)
  except:
    pass
  return
