try:
  import readline
except ImportError: pass

class Arg(object):
  '''This is the base class for all objects contained in RDict. Access to the raw argument values is
provided by getValue() and setValue(). These objects can be thought of as type objects for the
values themselves. It is possible to set an Arg in the RDict which has not yet been assigned a value
in order to declare the type of that option.'''
  def __init__(self, key, value = None, help = '', isTemporary = 0):
    self.key = key
    if not value is None:
      self.setValue(value)
    self.help        = help
    self.isTemporary = isTemporary
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
      if len(arg) > 2:
        for item in arg[1:-1].split(','):
          entry = item.split(':')
          if len(entry) > 1:
            value[entry[0]] = entry[1]
          else:
            value[entry[0]] = None
    else:
      value = arg
    return value
  parseValue = staticmethod(parseValue)

  def parseArgument(arg, ignoreDouble = 0):
    '''Split an argument into a (key, value) tuple, stripping off the leading dashes. Return (None, None) on failure.'''
    if arg[0] == '-':
      start = 1
      if arg[1] == '-' and not ignoreDouble:
        start = 2
      if arg.find('=') >= 0:
        (key, value) = arg[start:].split('=', 1)
      else:
        (key, value) = (arg[start:], '1')
      return (key, Arg.parseValue(value))
    return (None, None)
  parseArgument = staticmethod(parseArgument)

  def findArgument(key, argList):
    '''Locate an argument with the given key in argList, returning the value or None on failure
       - This isgenerally used to process arguments which must take effect before canonical argument parsing'''
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

  def __str__(self):
    if not self.isValueSet():
      return 'Empty '+str(self.__class__)
    elif isinstance(self.value, list):
      return str(map(str, self.value))
    return str(self.value)

  def getEntryPrompt(self):
    return 'Please enter value for '+str(self.key)+': '

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
      if self.help: print self.help
      while 1:
        try:
          self.setValue(Arg.parseValue(raw_input(self.getEntryPrompt())))
          break
        except KeyboardInterrupt:
          raise KeyError('Could not find value for key '+str(self.key))
        except TypeError, e:
          print str(e)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    self.value = value
    return

class ArgBool(Arg):
  '''Arguments that represent boolean values'''
  def __init__(self, key, value = None, help = '', isTemporary = 0):
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def getEntryPrompt(self):
    return 'Please enter boolean value for '+str(self.key)+': '

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    try:
      if   value == 'no':    value = 0
      elif value == 'yes':   value = 1
      elif value == 'true':  value = 1
      elif value == 'false': value = 0
      else:                  value = int(value)
    except:
      raise TypeError('Invalid boolean value: '+str(value))
    self.value = value
    return

class ArgInt(Arg):
  '''Arguments that represent integer numbers'''
  def __init__(self, key, value = None, help = '', min = -2147483647L, max = 2147483648L, isTemporary = 0):
    self.min = min
    self.max = max
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def getEntryPrompt(self):
    return 'Please enter integer value for '+str(self.key)+': '

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    try:
      value = int(value)
    except:
      raise TypeError('Invalid integer number: '+str(value))
    if value < self.min or value >= self.max:
      raise TypeError('Number out of range: '+str(value)+' not in ['+str(self.min)+','+str(self.max)+')')
    self.value = value
    return

class ArgReal(Arg):
  '''Arguments that represent floating point numbers'''
  def __init__(self, key, value = None, help = '', min = -1.7976931348623157e308, max = 1.7976931348623157e308, isTemporary = 0):
    self.min = min
    self.max = max
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def getEntryPrompt(self):
    return 'Please enter floating point value for '+str(self.key)+': '

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    try:
      value = float(value)
    except:
      raise TypeError('Invalid floating point number: '+str(value))
    if value < self.min or value >= self.max:
      raise TypeError('Number out of range: '+str(value)+' not in ['+str(self.min)+','+str(self.max)+')')
    self.value = value
    return

class ArgDir(Arg):
  '''Arguments that represent directories'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def getEntryPrompt(self):
    return 'Please enter directory for '+str(self.key)+': '

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      try:
        import GUI.FileBrowser
        import SIDL.Loader
        db = GUI.FileBrowser.FileBrowser(SIDL.Loader.createClass('GUI.Default.DefaultFileBrowser'))
        if self.help: db.setTitle(self.help)
        else:         db.setTitle('Select the directory for '+self.key)
        db.setMustExist(self.exist)
        self.value = db.getDirectory()
      except Exception:
        return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    # Should check whether it is a well-formed path
    if self.mustExist and not os.path.isdir(value):
      raise TypeError('Invalid directory: '+str(value))
    self.value = value
    return

class ArgLibrary(Arg):
  '''Arguments that represent libraries'''
  def __init__(self, key, value = None, help = '', mustExist = 1, isTemporary = 0):
    self.mustExist = mustExist
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def getEntryPrompt(self):
    return 'Please enter library for '+str(self.key)+': '

  def getValue(self):
    '''Returns the value. SHOULD MAKE THIS A PROPERTY'''
    if not self.isValueSet():
      try:
        import GUI.FileBrowser
        import SIDL.Loader
        db = GUI.FileBrowser.FileBrowser(SIDL.Loader.createClass('GUI.Default.DefaultFileBrowser'))
        if self.help: db.setTitle(self.help)
        else:         db.setTitle('Select the library for '+self.key)
        db.setMustExist(self.exist)
        self.value = db.getFile()
      except Exception:
        return Arg.getValue(self)
    return self.value

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    import os
    # Should check whether it is a well-formed path and an archive or shared object
    if self.mustExist and not os.path.isfile(value):
      raise TypeError('Invalid library: '+str(value))
    self.value = value
    return

class ArgString(Arg):
  '''Arguments that represent strings satisfying a given regular expression'''
  def __init__(self, key, value = None, help = '', regExp = None, isTemporary = 0):
    self.regExp = regExp
    if self.regExp:
      import re
      self.re = re.compile(self.regExp)
    Arg.__init__(self, key, value, help, isTemporary)
    return

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    if self.regExp and not self.re.match(value):
      raise TypeError('Invalid string '+str(value)+'. You must give a string satisfying "'+str(self.regExp)+'".')
    self.value = value
    return
