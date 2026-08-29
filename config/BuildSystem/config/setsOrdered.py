'''An insertion-ordered set

This is the part of the old Python 2 "sets" module that BuildSystem still uses: construction,
add(), clear(), and iteration in the order the elements were added. Dictionaries have preserved
insertion order since Python 3.7, so the values are ignored.
'''

class Set:
  '''A set that iterates in the order elements were added'''
  def __init__(self, iterable = None):
    self._data = {} if iterable is None else dict.fromkeys(iterable)

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

  def __contains__(self, element):
    return element in self._data

  def __repr__(self):
    return 'Set(['+', '.join(map(repr, self._data))+'])'

  def __getstate__(self):
    '''Kept in the format used before this class replaced the vendored sets module, so that a
       configure cache pickled by an older BuildSystem still loads'''
    return self._data,

  def __setstate__(self, state):
    self._data, = state

  def add(self, element):
    self._data[element] = None

  def clear(self):
    self._data.clear()
