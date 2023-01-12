#!/usr/bin/env python3
"""
# Created: Mon Jun 20 16:50:07 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import inspect
import functools

from ._patch   import Patch
from ._src_pos import SourceLocation

class DiagnosticMapProxy:
  __slots__ = '__diag_map', '__mro'

  def __init__(self, diag_map, mro):
    self.__diag_map = diag_map
    self.__mro      = mro
    return

  def __fuzzy_get_attribute__(self, in_diags, in_attr):
    try:
      return True, in_diags[in_attr]
    except KeyError:
      pass
    attr_items = [v for k, v in in_diags.items() if k.endswith(in_attr)]
    if len(attr_items) == 1:
      return True, attr_items[0]
    return False, None

  def __getattr__(self, attr):
    diag_map = self.__diag_map
    try:
      return getattr(diag_map, attr)
    except AttributeError:
      pass
    diag_map_diags = diag_map._diags
    for cls in self.__mro:
      try:
        sub_diag_map = diag_map_diags[cls.__qualname__]
      except KeyError:
        continue
      success, ret = self.__fuzzy_get_attribute__(sub_diag_map, attr)
      if success:
        return ret
    raise AttributeError(attr)

class DiagnosticMap:
  """
  A dict-like object that allows 'DiagnosticMap.my_diagnostic_name' to return 'my-diagnostic-name'
  """
  __slots__ = ('_diags',)

  @staticmethod
  def __sanitize_input(input_it):
    return {attr.replace('-', '_') : attr for attr in input_it}

  def __init__(self):
    self._diags = {'__general' : {}}
    return

  def __getattr__(self, attr):
    diags = self._diags['__general']
    try:
      return diags[attr]
    except KeyError:
      attr_items = [v for k, v in diags.items() if k.endswith(attr)]
      if len(attr_items) == 1:
        return attr_items[0]
    raise AttributeError(attr)

  def __get__(self, obj, objtype=None):
    """
    We need to do MRO-aware fuzzy lookup. In order to do that we need know about the calling class's
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
    return DiagnosticMapProxy(self, inspect.getmro(objtype))

  def update(self, obj, other, **kwargs):
    if isinstance(other, dict):
      dmap = self.__sanitize_input(other.keys())
    elif isinstance(other, (list, tuple)) or inspect.isgenerator(other):
      dmap = self.__sanitize_input(other)
    else:
      raise ValueError(type(other))
    self._diags['__general'].update(dmap, **kwargs)
    qual_name = obj.__qualname__
    if qual_name not in self._diags:
      self._diags[qual_name] = {}
    return self._diags[qual_name].update(dmap, **kwargs)

class _DiagnosticsManager:
  __slots__   = 'disabled', 'flagprefix'
  _registered = {}

  @classmethod
  def registered(cls):
    return cls._registered

  @staticmethod
  def __expand_flag(flag):
    if not isinstance(flag, str):
      try:
        flag = '-'.join(flag)
      except Exception as ex:
        raise ValueError(type(flag)) from ex
    return flag

  @classmethod
  def flag_prefix(cls, obj):
    return getattr(obj, '__diagnostic_prefix__', lambda f: f)

  @classmethod
  def check_flag(cls, flag):
    flag = cls.__expand_flag(flag)
    if flag not in cls._registered:
      raise ValueError(f'Flag \'{flag}\' is not registered with {cls}')
    return flag

  @classmethod
  def register(cls, *args):
    def decorator(symbol):
      if inspect.isclass(symbol):
        wrapper = symbol
      else:
        @functools.wraps(symbol)
        def wrapper(*args, **kwargs):
          return symbol(*args, **kwargs)

      symbol_flag_prefix = cls.flag_prefix(wrapper)
      diag_list          = [(cls.__expand_flag(symbol_flag_prefix(d)), h.casefold()) for d, h in args]
      if not hasattr(wrapper, 'diags'):
        wrapper.diags = DiagnosticMap()
      wrapper.diags.update(wrapper, [d for d, _ in diag_list])
      cls._registered.update(diag_list)
      return wrapper
    return decorator

  def __init__(self, flagprefix='-f'):
    self.disabled   = set()
    self.flagprefix = flagprefix if flagprefix.startswith('-') else '-' + flagprefix
    return


  def disable(self, flag):
    self.disabled.add(self.check_flag(flag))
    return

  def enable(self, flag):
    self.disabled.discard(self.check_flag(flag))
    return

  def set(self, flag, value):
    return self.enable(flag) if value else self.disable(flag)

  def disabled_for(self, flag):
    return self.check_flag(flag) in self.disabled

  def enabled_for(self, flag):
    return not self.disabled_for(flag)

  def make_command_line_flag(self, flag):
    return f'{self.flagprefix}{self.check_flag(flag)}'

DiagnosticManager = _DiagnosticsManager()

class Diagnostic:
  FLAG_SUBST = r'%DIAG_FLAG%'
  __slots__  = 'flag', 'message', 'location', 'patch', 'clflag', 'notes'

  def __init__(self, flag, message, location, patch=None, notes=None):
    if notes is None:
      notes = []
    else:
      assert isinstance(notes, list)

    if patch is not None:
      assert isinstance(patch, Patch)

    self.flag     = DiagnosticManager.check_flag(flag)
    self.message  = str(message)
    self.location = SourceLocation.cast(location)
    self.patch    = patch
    self.clflag   = f' [{DiagnosticManager.make_command_line_flag(self.flag)}]'
    self.notes    = notes
    return

  def __repr__(self):
    return f'<flag: {self.clflag}, patch: {self.patch}, message: {self.message}, notes: {self.notes}>'

  def add_note(self, note, location = None):
    if location is None:
      location = self.location
    else:
      location = SourceLocation.cast(location)

    self.notes.append((location, note))
    return self

  def format_message(self):
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
    return message

  def disabled(self):
    return DiagnosticManager.disabled_for(self.flag.replace('_', '-'))
