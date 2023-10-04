#!/usr/bin/env python3
"""
# Created: Mon Jun 20 16:50:07 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import copy
import enum
import inspect
import contextlib

from .._typing import *

_T = TypeVar('_T')

from ..util._color import Color

from ._src_pos import SourceLocation, SourceRange

class DiagnosticMapProxy:
  __slots__ = '_diag_map', '_mro'

  def __init__(self, diag_map: DiagnosticMap, mro: tuple[type, ...]) -> None:
    self._diag_map = diag_map
    self._mro      = mro
    return

  def _fuzzy_get_attribute(self, in_diags: dict[str, str], in_attr: str) -> tuple[bool, str]:
    try:
      return True, in_diags[in_attr]
    except KeyError:
      pass
    attr_items = [v for k, v in in_diags.items() if k.endswith(in_attr)]
    if len(attr_items) == 1:
      return True, attr_items[0]
    return False, ''

  def __getattr__(self, attr: str) -> str:
    diag_map = self._diag_map
    try:
      return TYPE_CAST(str, getattr(diag_map, attr))
    except AttributeError:
      pass
    diag_map_diags = diag_map._diags
    for cls in self._mro:
      try:
        sub_diag_map = diag_map_diags[cls.__qualname__]
      except KeyError:
        continue
      success, ret = self._fuzzy_get_attribute(sub_diag_map, attr)
      if success:
        return ret
    raise AttributeError(attr)

class DiagnosticMap:
  r"""
  A dict-like object that allows 'DiagnosticMap.my_diagnostic_name' to return 'my-diagnostic-name'
  """
  __slots__ = ('_diags',)

  _diags: dict[str, dict[str, str]]

  @staticmethod
  def _sanitize_input(input_it: Iterable[str]) -> dict[str, str]:
    return {attr.replace('-', '_') : attr for attr in input_it}

  def __init__(self) -> None:
    self._diags = {'__general' : {}}
    return

  def __getattr__(self, attr: str) -> str:
    diags = self._diags['__general']
    try:
      return diags[attr]
    except KeyError:
      attr_items = [v for k, v in diags.items() if k.endswith(attr)]
      if len(attr_items) == 1:
        return attr_items[0]
    raise AttributeError(attr)

  def __get__(self, obj: Any, objtype: Optional[type] = None) -> DiagnosticMapProxy:
    r"""We need to do MRO-aware fuzzy lookup. In order to do that we need know about the calling class's
    type, which is not passed to the regular __getattr__(). But type information *is* passed to
    __get__() (which is called on attribute access), so the workaround is to create a proxy object
    that ends up calling our own __getattr__().

    The motivating example is as follows. Suppose you have:

    @DiagnosticsManager.register('some-diag', ...)
    class Foo:
      def baz():
        diag = self.diags.some_diag

    @DiagnosticsManager.register('some-diag', ...)
    class Bar:
      def baz():
        diag = self.diags.some_diag

    In doing this we effectively want to create bar-some-diag and foo-some-diag, which works as
    intended. However in Foo.baz() when it searches for the some_diag (transformed to 'some-diag')
    attribute, it will fuzzy match to 'bar-some-diag'.

    So we need to first search our own classes namespace, and then search each of our base classes
    namespaces before finally considering children.
    """
    assert objtype is not None
    return DiagnosticMapProxy(self, inspect.getmro(objtype))

  def update(self, obj: Any, other: Iterable[str], **kwargs) -> None:
    if not isinstance(other, (list, tuple)) or inspect.isgenerator(other):
      raise ValueError(type(other))

    dmap = self._sanitize_input(other)
    self._diags['__general'].update(dmap, **kwargs)
    qual_name = obj.__qualname__
    if qual_name not in self._diags:
      self._diags[qual_name] = {}
    self._diags[qual_name].update(dmap, **kwargs)
    return

class DiagnosticsManagerCls:
  __slots__                   = 'disabled', 'flagprefix'
  _registered: dict[str, str] = {}

  disabled: set[str]
  flagprefix: str

  @classmethod
  def registered(cls) -> dict[str, str]:
    r"""Return the registered diagnostics

    Returns
    -------
    registered :
      the set of registered diagnostics
    """
    return cls._registered

  @staticmethod
  def _expand_flag(flag: Union[Iterable[str], str]) -> str:
    r"""Expand a flag

    Transforms `['foo', 'bar', 'baz']` into `'foo-bar-baz'`

    Parameters
    ----------
    flag :
      the flag parts to expand

    Returns
    -------
    flag :
      the expanded flag

    Raises
    ------
    ValueError
      if flag is an iterable, but cannot be joined
    """
    if not isinstance(flag, str):
      try:
        flag = '-'.join(flag)
      except Exception as ex:
        raise ValueError(type(flag)) from ex
    return flag

  @classmethod
  def flag_prefix(cls, obj: object) -> Callable[[str], str]:
    r"""Return the flag prefix

    Parameters
    ----------
    obj :
      a class instance which may or may implement `__diagnostic_prefix__(flag: str) -> str`

    Returns
    -------
    prefix :
      the prefix

    Notes
    -----
    Implementing `__diagnostic_prefix__()` is optional, in which case this routine returns an identity
    lambda
    """
    return getattr(obj, '__diagnostic_prefix__', lambda f: f)

  @classmethod
  def check_flag(cls, flag: str) -> str:
    r"""Check a flag for validity and expand it

    Parameters
    ----------
    flag :
      the flag to expand

    Returns
    -------
    flag :
      the expanded flag

    Raises
    ------
    ValueError
      if the flag is not registered with the `DiagnosticManager`
    """
    flag = cls._expand_flag(flag)
    if flag not in cls._registered:
      raise ValueError(f'Flag \'{flag}\' is not registered with {cls}')
    return flag

  @classmethod
  def _inject_diag_map(cls, symbol: _T, diag_pairs: Iterable[tuple[str, str]]) -> _T:
    r"""Does the registering and injecting of the `DiagnosticMap` into some symbol

    Parameters
    ----------
    symbol :
      the symbol to inject the `DiagnosticMap` into
    diag_pairs :
      an iterable of pairs of flag - description which should be injected

    Returns
    -------
    symbol :
      the symbol with the injected map

    Notes
    -----
    This registeres the flags in `diag_pairs` will be registered with the `DiagnosticsManager`. After
    this returns `symbol` will have a member `diags` through which the diagnostics can be accessed. So
    if do
    ```
    DiagnosticManager.register('foo-bar-baz', 'check a foo, a bar, and a baz')
    def MyClass:
      ...
      def some_func(self, ...):
        ...
        diag = self.diags.foo_bar_baz # can access by replacing '-' with '_' in flag
    ```

    This function appears to return a `_T` unchanged. But what we really want to do is
    ```
    _T = TypeVar('_T')

    class HasDiagMap(Protocol):
      diags: DiagnosticMap

    def _inject_diag_map(symbol: _T, ...) -> Intersection[HasDiagMap, [_T]]:
      ...
    ```
    I.e. the returned type is *both* whatever it was before, but it now also obeys the diag-map
    protocol, i.e. it has a member `diags` which is a `DiagnosticMap`. But unfortunately Python has
    no such 'Intersection' type yet so we need to annotate all the types by hand...
    """
    diag_attr          = 'diags'
    symbol_flag_prefix = cls.flag_prefix(symbol)
    expanded_diags     = [
      (cls._expand_flag(symbol_flag_prefix(d)), h.casefold()) for d, h in diag_pairs
    ]
    if not hasattr(symbol, diag_attr):
      setattr(symbol, diag_attr, DiagnosticMap())
    getattr(symbol, diag_attr).update(symbol, [d for d, _ in expanded_diags])
    cls._registered.update(expanded_diags)
    return symbol

  @classmethod
  def register(cls, *args: tuple[str, str]) -> Callable[[_T], _T]:
    def decorator(symbol: _T) -> _T:
      return cls._inject_diag_map(symbol, args)
    return decorator

  def __init__(self, flagprefix: str = '-f') -> None:
    r"""Construct the `DiagnosticManager`

    Parameters
    ----------
    flagprefix : '-f', optional
      the base flag prefix to prepend to all flags
    """
    self.disabled   = set()
    self.flagprefix = flagprefix if flagprefix.startswith('-') else '-' + flagprefix
    return

  def disable(self, flag: str) -> None:
    r"""Disable a flag

    Parameters
    ----------
    flag :
      the flag to disable
    """
    self.disabled.add(self.check_flag(flag))
    return

  def enable(self, flag: str) -> None:
    r"""Enable a flag

    Parameters
    ----------
    flag :
      the flag to enable
    """
    self.disabled.discard(self.check_flag(flag))
    return

  def set(self, flag: str, value: bool) -> None:
    r"""Set enablement of a flag

    Parameters
    ----------
    flag :
      the flag to set
    value :
      True to enable, False to disable
    """
    if value:
      self.enable(flag)
    else:
      self.disable(flag)
    return

  def disabled_for(self, flag: str) -> bool:
    r"""Is `flag` disabled?

    Parameters
    ----------
    flag :
      the flag to check

    Returns
    -------
    disabled :
      True if `flag` is disabled, False otherwise
    """
    return self.check_flag(flag) in self.disabled

  def enabled_for(self, flag: str) -> bool:
    r"""Is `flag` enabled?

    Parameters
    ----------
    flag :
      the flag to check

    Returns
    -------
    enabled :
      True if `flag` is enabled, False otherwise
    """
    return not self.disabled_for(flag)

  def make_command_line_flag(self, flag: str) -> str:
    r"""Build a command line flag

    Parameters
    ----------
    flag :
      the flag to build for

    Returns
    -------
    ret :
      the full command line flag
    """
    return f'{self.flagprefix}{self.check_flag(flag)}'

  @contextlib.contextmanager
  def push_from(self, dict_like: Mapping[str, Collection[re.Pattern[str]]]):
    r"""Temporarily enable or disable flags based on `dict_like`

    Parameters
    ----------
    dict_like :
      a dictionary of actions to take

    Yields
    ------
    self :
      the object

    Raises
    ------
    ValueError
      if an unknown key is encountered
    """
    if dict_like:
      dispatcher   = {
        'disable' : self.disabled.update,
        'ignore'  : self.disabled.update
      }
      reg          = self.registered().keys()
      old_disabled = copy.deepcopy(self.disabled)
      for key, values in dict_like.items():
        mod_flags = [f for f in reg for matcher in values if matcher.match(f)]
        try:
          dispatcher[key](mod_flags)
        except KeyError as ke:
          raise ValueError(
            f'Unknown pragma key \'{key}\', expected one of: {list(dispatcher.keys())}'
          ) from ke
    try:
      yield self
    finally:
      if dict_like:
        self.disabled = old_disabled

DiagnosticManager = DiagnosticsManagerCls()

@enum.unique
class DiagnosticKind(enum.Enum):
  ERROR   = enum.auto()
  WARNING = enum.auto()

  def color(self) -> str:
    if self == DiagnosticKind.ERROR:
      return Color.bright_red()
    elif self == DiagnosticKind.WARNING:
      return Color.bright_yellow()
    else:
      raise ValueError(str(self))

class Diagnostic:
  FLAG_SUBST = r'%DIAG_FLAG%'
  Kind       = DiagnosticKind
  __slots__  = 'flag', 'message', 'location', 'patch', 'clflag', 'notes', 'kind'

  flag: str
  message: str
  location: SourceLocation
  patch: Optional[Patch]
  clflag: str
  notes: list[tuple[SourceLocationLike, str]]
  kind: DiagnosticKind

  def __init__(self, kind: DiagnosticKind, flag: str, message: str, location: SourceLocationLike, patch: Optional[Patch] = None, notes: Optional[list[tuple[SourceLocationLike, str]]] = None) -> None:
    r"""Construct a `Diagnostic`

    Parameters
    ----------
    kind :
      the kind of `Diagnostic` to create
    flag :
      the flag to attribute the diagnostic to
    message :
      the informative message
    location :
      the location to attribute the diagnostic to
    patch :
      a patch to automatically fix the diagnostic
    notes :
      a list of notes to initialize the diagnostic with
    """
    if notes is None:
      notes = []

    self.flag     = DiagnosticManager.check_flag(flag)
    self.message  = str(message)
    self.location = SourceLocation.cast(location)
    self.patch    = patch
    self.clflag   = f' [{DiagnosticManager.make_command_line_flag(self.flag)}]'
    self.notes    = notes
    self.kind     = kind
    return

  @staticmethod
  def make_message_from_formattable(message: str, crange: Optional[Formattable] = None, num_context: int = 2, **kwargs) -> str:
    r"""Make a formatted error message from a formattable object

    Parameters
    ----------
    message :
      the base message
    crange : optional
      the formattable object, which must have a method `formatted(num_context: int, **kwargs) -> str`
      whose formatted text is optionally appended to the message
    num_context : optional
      if crange is given, the number of context lines to append
    **kwargs : optional
      if crange is given, additional keyword arguments to pass to `SourceRange.formatted()`

    Returns
    -------
    mess :
      the error message
    """
    if crange is None:
      return message
    return f'{message}:\n{crange.formatted(num_context=num_context, **kwargs)}'

  @classmethod
  def from_source_range(cls, kind: DiagnosticKind, diag_flag: str, msg: str, src_range: SourceRangeLike, patch: Optional[Patch] = None, **kwargs) -> Diagnostic:
    r"""Construct a `Diagnostic` from a source_range

    Parameters
    ----------
    kind :
      the `DiagnostiKind`
    diag_flag :
      the diagnostic flag to display
    msg :
      the base message text
    src_range :
      the source range to generate the message from
    patch : optional
      the patch to create a fixit form
    **kwargs :
      additional keyword arguments to pass to `src_range.formatted()`

    Returns
    -------
    diag :
      the constructed `Diagnostic`

    Notes
    -----
    This is the de-facto standard factory for creating `Diagnostic`s as it ensures that the messages
    are all similarly formatted and displayed. The vast majority of `Diagnostic`s are created via this
    function
    """
    src_range = SourceRange.cast(src_range)
    return cls(
      kind, diag_flag,
      cls.make_message_from_formattable(msg, crange=src_range, **kwargs),
      src_range.start,
      patch=patch
    )

  def __repr__(self) -> str:
    return f'<flag: {self.clflag}, patch: {self.patch}, message: {self.message}, notes: {self.notes}>'

  def formatted_header(self) -> str:
    r"""Return the formatted header for this diagnostic, suitable for output

    Returns
    -------
    hdr :
      the formatted header
    """
    return f'{self.kind.color()}{self.location}: {self.kind.name.casefold()}:{Color.reset()} {self.format_message()}'

  def add_note(self, note: str, location: Optional[SourceLocationLike] = None) -> Diagnostic:
    r"""Add a note to a diagnostic

    Parameters
    ----------
    note :
      a useful additional message
    location : optional
      a location to attribute the note to, if not given, the location of the diagnostic is used

    Returns
    -------
    self :
      the diagnostic object
    """
    if location is None:
      location = self.location
    else:
      location = SourceLocation.cast(location)

    self.notes.append((location, note))
    return self

  def format_message(self) -> str:
    r"""Format the diagnostic

    Returns
    -------
    ret :
      the formatted diagnostic message, suitable for display to the user
    """
    message = self.message
    clflag  = self.clflag
    if self.FLAG_SUBST in message:
      message = message.replace(self.FLAG_SUBST, clflag.lstrip())
    else:
      sub = ':\n'
      pos = message.find(sub)
      if pos == -1:
        message += clflag
      else:
        assert not message[pos - 1].isdigit(), f'message[pos - 1] (pos = {pos}) -> {message[pos - 1]} is a digit when it should not be'
        message = message.replace(sub, clflag + sub, 1)

    if self.notes:
      notes_tmp = '\n\n'.join(f'{loc} Note: {note}' for loc, note in self.notes)
      message   = f'{message}\n\n{notes_tmp}'
    assert not message.endswith('\n')
    return message

  def disabled(self) -> bool:
    r"""Is the flag for this diagnostic disabled?

    Returns
    -------
    disabled :
      True if this diagnostic is disabled, False otherwise
    """
    return DiagnosticManager.disabled_for(self.flag.replace('_', '-'))
