from __future__ import print_function
try:
  import readline
except ImportError: pass

class Arg(object):
  '''This is the base class for all objects contained in RDict. Access to the raw argument values is
provided by getValue() and setValue(). These objects can be thought of as type objects for the
values themselves. It is possible to set an Arg in the RDict which has not yet been assigned a value
in order to declare the type of that option.

Inputs which cannot be converted to the correct type will cause TypeError, those failing validation
tests will cause ValueError.
'''
  def __init__(self, key, value = None, help = '', isTemporary = False, deprecated = False):
    self.key         = key
    self.help        = help
    self.isTemporary = isTemporary
    self.deprecated  = False
    if not value is None:
      self.setValue(value)
    self.deprecated  = deprecated
    return

  def isValueSet(self):
    '''Determines whether the value of this argument has been set'''
    return hasattr(self, 'value')

  def getTemporary(self):
    '''Retrieve the flag indicating whether the item should be persistent'''
    return self.isTemporary

  def setTemporary(self, isTemporary):
    '''Set the flag indicating whether the item should be persistent'''
    self.isTemporary = isTemporary
    return

  def parseValue(arg):
    '''Return the object represented by the value portion of a string argument'''
    # Should I replace this with a lexer?
    if arg: arg = arg.strip()
    if arg and arg[0] == '[' and arg[-1] == ']':
      if len(arg) > 2: value = arg[1:-1].split(',')
      else:            value = []
    elif arg and arg[0] == '{' and arg[-1] == '}':
      value = {}
      idx = 1
      oldIdx = idx
      while idx < len(arg)-1:
        if arg[oldIdx] == ',':
          oldIdx += 1
        while not arg[idx] == ':': idx += 1
        key = arg[oldIdx:idx]
        idx += 1
        oldIdx = idx
        nesting = 0
        while not (arg[idx] == ',' or arg[idx] == '}') or nesting:
          if arg[idx] == '[':
            nesting += 1
          elif arg[idx] == ']':
            nesting -= 1
          idx += 1
        value[key] = Arg.parseValue(arg[oldIdx:idx])
        oldIdx = idx
    else:
      value = arg
    return value
  parseValue = staticmethod(parseValue)

  def parseArgument(arg, ignoreDouble = 0):
    '''Split an argument into a (key, value) tuple, stripping off the leading dashes. Return (None, None) on failure.'''
    start = 0
    if arg and arg[0] == '-':
      start = 1
      if arg[1] == '-' and not ignoreDouble:
        start = 2
    if arg.find('=') >= 0:
      (key, value) = arg[start:].split('=', 1)
    else:
      if start == 0:
        (key, value) = (None, arg)
      else:
        (key, value) = (arg[start:], '1')
    return (key, Arg.parseValue(value))

  parseArgument = staticmethod(parseArgument)

  def findArgument(key, argList):
    '''Locate an argument with the given key in argList, returning the value or None on failure
       - This is generally used to process arguments which must take effect before canonical argument parsing'''
    if not isinstance(argList, list): return None
    # Reverse the list so that we preserve the semantics which state that the last
    #   argument with a given key takes effect
    l = argList[:]
    l.reverse()
    for arg in l:
      (k, value) = Arg.parseArgument(arg)
      if k == key:
        return value
    return None
  findArgument = staticmethod(findArgument)

  def processAlternatePrefixes(argList):
    '''Convert alternate prefixes to our normal form'''
    for l in range(0, len(argList)):
      name = argList[l]
      if name.find('enable-') >= 0:
        argList[l] = name.replace('enable-','with-')
        if name.find('=') == -1: argList[l] = argList[l]+'=1'
      if name.find('disable-') >= 0:
        argList[l] = name.replace('disable-','with-')
        if name.find('=') == -1: argList[l] = argList[l]+'=0'
        elif name.endswith('=1'): argList[l].replace('=1','=0')
      if name.find('without-') >= 0:
        argList[l] = name.replace('without-','with-')
        if name.find('=') == -1: argList[l] = argList[l]+'=0'
        elif name.endswith('=1'): argList[l].replace('=1','=0')
    return
  processAlternatePrefixes = staticmethod(processAlternatePrefixes)

  def __str__(self):
    if not self.isValueSet():
      return 'Empty '+str(self.__class__)
    value = self.value
    if isinstance(value, list):
      return str(list(map(str, value)))
    return str(value)

  def getKey(self):
    '''Returns the key. SHOULD MAKE THIS A PROPERTY'''
    return self.key

  def setKey(self, key):
    '''Set the key. SHOULD MAKE THIS A PROPERTY'''
    self.key = key
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      raise KeyError('Could not find value for key '+str(self.key))
    return self.value

  def checkKey(self):
    if self.deprecated:
      if isinstance(self.deprecated, str):
        raise KeyError('Deprecated option '+self.key+' should be '+self.deprecated)
      raise KeyError('Deprecated option '+self.key)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    self.value = value
    return

class ArgBool(Arg):
  '''Arguments that represent boolean values'''
  def __init__(self, key, value = None, help = '', isTemporary = 0, deprecated = False):
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    try:
      if   value == 'no':    value = 0
      elif value == 'yes':   value = 1
      elif value == 'true':  value = 1
      elif value == 'false': value = 0
      elif value == 'True':  value = 1
      elif value == 'False': value = 0
      else:                  value = int(value)
    except:
      raise TypeError('Invalid boolean value: '+str(value)+' for key '+str(self.key))
    self.value = value
    return

class ArgFuzzyBool(Arg):
  '''Arguments that represent boolean values of an extended set'''
  def __init__(self, key, value = None, help = '', isTemporary = 0, deprecated = False):
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def valueName(self, value):
    if value == 0:
      return 'no'
    elif value == 1:
      return 'yes'
    elif value == 2:
      return 'ifneeded'
    return str(value)

  def __str__(self):
    if not self.isValueSet():
      return 'Empty '+str(self.__class__)
    elif isinstance(self.value, list):
      return str(map(self.valueName, self.value))
    return self.valueName(self.value)

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    try:
      if   value == '0':        value = 0
      elif value == '1':        value = 1
      elif value == 'no':       value = 0
      elif value == 'yes':      value = 1
      elif value == 'false':    value = 0
      elif value == 'true':     value = 1
      elif value == 'maybe':    value = 2
      elif value == 'ifneeded': value = 2
      elif value == 'client':   value = 2
      elif value == 'server':   value = 3
      else:                     value = int(value)
    except:
      raise TypeError('Invalid fuzzy boolean value: '+str(value)+' for key '+str(self.key))
    self.value = value
    return

class ArgInt(Arg):
  '''Arguments that represent integer numbers'''
  def __init__(self, key, value = None, help = '', min = -2147483647, max = 2147483648, isTemporary = 0, deprecated = False):
    self.min = min
    self.max = max
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    try:
      value = int(value)
    except:
      raise TypeError('Invalid integer number: '+str(value)+' for key '+str(self.key))
    if value < self.min or value >= self.max:
      raise ValueError('Number out of range: '+str(value)+' not in ['+str(self.min)+','+str(self.max)+')'+' for key '+str(self.key))
    self.value = value
    return

class ArgReal(Arg):
  '''Arguments that represent floating point numbers'''
  def __init__(self, key, value = None, help = '', min = -1.7976931348623157e308, max = 1.7976931348623157e308, isTemporary = 0, deprecated = False):
    self.min = min
    self.max = max
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    try:
      value = float(value)
    except:
      raise TypeError('Invalid floating point number: '+str(value)+' for key '+str(self.key))
    if value < self.min or value >= self.max:
      raise ValueError('Number out of range: '+str(value)+' not in ['+str(self.min)+','+str(self.max)+')'+' for key '+str(self.key))
    self.value = value
    return

class ArgDir(Arg):
  '''Arguments that represent directories'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    # Should check whether it is a well-formed path
    if not isinstance(value, str):
      raise TypeError('Invalid directory: '+str(value)+' for key '+str(self.key))
    value = os.path.expanduser(value)
    value = os.path.abspath(value)
    if self.mustExist and value and not os.path.isdir(value):
      raise ValueError('Nonexistent directory: '+str(value)+' for key '+str(self.key))
    self.value = value
    return

class ArgDirList(Arg):
  '''Arguments that represent directory lists'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    if not isinstance(value, list):
      value = value.split(':')
    # Should check whether it is a well-formed path
    nvalue = []
    for dir in value:
      if dir:
        nvalue.append(os.path.expanduser(dir))
    value = nvalue
    for dir in value:
      if self.mustExist and not os.path.isdir(dir):
        raise ValueError('Invalid directory: '+str(dir)+' for key '+str(self.key))
    self.value = value
    return

class ArgFile(Arg):
  '''Arguments that represent a file'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    # Should check whether it is a well-formed path
    if not isinstance(value, str):
      raise TypeError('Invalid file: '+str(value)+' for key '+str(self.key))
    value = os.path.expanduser(value)
    value = os.path.abspath(value)
    if self.mustExist and value and not os.path.isfile(value):
      raise ValueError('Nonexistent file: '+str(value)+' for key '+str(self.key))
    self.value = value
    return

class ArgFileList(Arg):
  '''Arguments that represent file lists'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    if not isinstance(value, list):
      value = value.split(':')
    # Should check whether it is a well-formed path
    nvalue = []
    for file in value:
      if file:
        nvalue.append(os.path.expanduser(file))
    value = nvalue
    for file in value:
      if self.mustExist and not os.path.isfile(file):
        raise ValueError('Invalid file: '+str(file)+' for key '+str(self.key))
    self.value = value
    return

class ArgLibrary(Arg):
  '''Arguments that represent libraries'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    # Should check whether it is a well-formed path and an archive or shared object
    if self.mustExist:
      if not isinstance(value, list):
        value = value.split(' ')
    self.value = value
    return

class ArgExecutable(Arg):
  '''Arguments that represent executables'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0, deprecated = False):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      return Arg.getValue(self)
    return self.value

  def checkExecutable(self, dir, name):
    import os
    prog = os.path.join(dir, name)
    return os.path.isfile(prog) and os.access(prog, os.X_OK)

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    # Should check whether it is a well-formed path
    if self.mustExist:
      index = value.find(' ')
      if index >= 0:
        options = value[index:]
        value   = value[:index]
      else:
        options = ''
      found = self.checkExecutable('', value)
      if not found:
        for dir in os.environ['PATH'].split(os.path.pathsep):
          if self.checkExecutable(dir, value):
            found = 1
            break
      if not found:
        raise ValueError('Invalid executable: '+str(value)+' for key '+str(self.key))
      value += options
    self.value = value
    return

class ArgString(Arg):
  '''Arguments that represent strings satisfying a given regular expression'''
  def __init__(self, key, value = None, help = '', regExp = None, isTemporary = 0, deprecated = False):
    self.regExp = regExp
    if self.regExp:
      import re
      self.re = re.compile(self.regExp)
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.checkKey()
    if self.regExp and not self.re.match(value):
      raise ValueError('Invalid string '+str(value)+'. You must give a string satisfying "'+str(self.regExp)+'"'+' for key '+str(self.key))
    self.value = value
    return

class ArgDownload(Arg):
  '''Arguments that represent software downloads'''
  def __init__(self, key, value = None, help = '', isTemporary = 0, deprecated = False):
    Arg.__init__(self, key, value, help, isTemporary, deprecated)
    return

  def valueName(self, value):
    if value == 0:
      return 'no'
    elif value == 1:
      return 'yes'
    return str(value)

  def __str__(self):
    if not self.isValueSet():
      return 'Empty '+str(self.__class__)
    elif isinstance(self.value, list):
      return str(map(self.valueName, self.value))
    return self.valueName(self.value)

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    self.checkKey()
    try:
      if   value == '0':        value = 0
      elif value == '1':        value = 1
      elif value == 'no':       value = 0
      elif value == 'yes':      value = 1
      elif value == 'false':    value = 0
      elif value == 'true':     value = 1
      elif not isinstance(value, int):
        value = str(value)
    except:
      raise TypeError('Invalid download value: '+str(value)+' for key '+str(self.key))
    if isinstance(value, str):
      from urllib import parse as urlparse_local
      if not urlparse_local.urlparse(value)[0] and not os.path.exists(value):
        raise ValueError('Invalid download location: '+str(value)+' for key '+str(self.key))
    self.value = value
    return
