#
#   Provides the BS build system dictionary
#
import os
import re
import sys
try:
import readline
except ImportError: pass

import RDict

#===============================================================================
def parseValue(arg):
  if arg and arg[0] == '[' and arg[-1] == ']':
    if len(arg) > 2: arg = arg[1:-1].split(',')
    else:            arg = []
  if arg and arg[0] == '{' and arg[-1] == '}':
    d = {}
    if len(arg) > 2:
      for item in arg[1:-1].split(','):
        entry = item.split(':')
        d[entry[0]] = entry[1]
    arg = d
  return arg

def parseArgument(arg, ignoreDouble = 0):
  if arg[0] == '-':
    start = 1
    if arg[1] == '-' and not ignoreDouble:
      start = 2
    if arg.find('=') >= 0:
      (key, value) = arg[start:].split('=')
    else:
      (key, value) = (arg[start:], '1')
    return (key, parseValue(value))
  return (None, None)

def findArgument(arg, argList):
  if not isinstance(argList, list): return None
  for a in argList:
    (key, value) = parseArgument(a)
    if key == arg:
      return value
  return None

#===============================================================================
#  The base class of objects that are stored in the ArgDict
# The dictionary actual contains objects, the actual option is
# is obtained from the object with getValue(), this allows
# us to provide properties of the option before the option is set
class ArgEmpty:
  def __str__(self):
    if not hasattr(self,'value'):
      return ''
    if isinstance(self.value, list):
      return str(map(str, self.value))
    return str(self.value)

  def convertValue(self, value):
    return value

  def getValue(self,key):
    return (0,None)

class Arg(ArgEmpty):
  def __init__(self,value):
    self.value = value
    return

  def getValue(self,key):
    #   First argument = 0 indicates already contains value
    #                    1 indicates got the value, needs to be put back in dictionary server
    return (0,self.value)

#  Objects that are stored in the ArgDict that represent directories
class ArgDir(ArgEmpty):
  def __init__(self,dirmustexist = 1, help = None):
    self.exist = dirmustexist
    self.help  = help
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      try:
        import GUI.FileBrowser
        import SIDL.Loader
        db = GUI.FileBrowser.FileBrowser(SIDL.Loader.createClass('GUI.Default.DefaultFileBrowser'))
        if self.help: db.setTitle(self.help)
        else:         db.setTitle('Select the directory for '+key)
        db.setMustExist(self.exist)
        self.value = db.getDirectory()
        return (1,self.value)
      except Exception:
        # default to getting directory as string
        if not hasattr(self,'value'):
           if self.help: print self.help
           try: self.value = parseValue(raw_input('Please enter value for '+key+': '))
           except KeyboardInterrupt: sys.exit(1)
           return (1,self.value)
        else: return (0,self.value)
    else: return (0,self.value)

#  Objects that are stored in the ArgDict that represent libraries
class ArgLibrary(ArgEmpty):
  def __init__(self, mustExist = 1, help = None):
    self.exist = mustExist
    self.help  = help
    
  def getValue(self,key):
    if not hasattr(self, 'value'):
      try:
        import GUI.FileBrowser
        import SIDL.Loader
        db = GUI.FileBrowser.FileBrowser(SIDL.Loader.createClass('GUI.Default.DefaultFileBrowser'))
        if self.help: db.setTitle(self.help)
        else:         db.setTitle('Select the file for '+key)
        db.setMustExist(self.exist)
        self.value = db.getFile()
        # TODO: Should verify that it is a library here
        return (1, self.value)
      except Exception:
        # default to getting library as string
        if not hasattr(self, 'value'):
           if self.help: print self.help
           try: self.value = parseValue(raw_input('Please enter value for '+key+': '))
           except KeyboardInterrupt: sys.exit(1)
           return (1, self.value)
        else: return (0, self.value)
    else: return (0, self.value)
      
#  Objects that are stored in the ArgDict that are strings
class ArgString(ArgEmpty):
  def __init__(self,help = None):
    self.help  = help
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      if self.help: print self.help
      try:                      self.value = parseValue(raw_input('Please enter value for '+key+': '))
      except KeyboardInterrupt:	sys.exit(1)
      return (1,self.value)
    else: return (0,self.value)

#  Objects that are stored in the ArgDict that are integers
class ArgInt(ArgEmpty):
  def __init__(self,help = None,min = -1000000, max = 1000000):
    self.help = help
    self.min  = min
    self.max  = max

  def convertValue(self, value):
    try:
      value = int(value)
    except:
      value = self.min
    return value
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      if self.help: print self.help
      while 1:
        try: self.value = parseValue(raw_input('Please enter integer value for '+key+': '))
        except KeyboardInterrupt: sys.exit(1)
        try: self.value = int(self.value)
        except: self.value = self.min
        if self.value > self.min and self.value < self.max: return (1,self.value)
    else: return (0,self.value)

#  Objects that are stored in the ArgDict that are booleans
class ArgBool(ArgEmpty):
  def __init__(self, help = None):
    self.help  = help
    return

  def convertValue(self, value):
    try:
      value = int(value)
    except:
      value = 0
    if value:
      value = 1
    return value

  def getValue(self,key):
    if not hasattr(self, 'value'):
      if self.help: print self.help
      try:
        self.value = parseValue(raw_input('Please enter boolean value for '+key+': '))
      except KeyboardInterrupt:
        sys.exit(1)
      return (1, self.convertValue(self.value))
    else:
      return (0, self.value)

#  Objects that are stored in the ArgDict that are floating point numbers
class ArgReal(ArgEmpty):
  def __init__(self, help = None, min = -1.7976931348623157e308, max = 1.7976931348623157e308):
    self.help = help
    self.min  = min
    self.max  = max

  def convertValue(self, value):
    try:
      value = float(value)
    except:
      value = self.min
    return value
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      if self.help: print self.help
      while 1:
        try: self.value = parseValue(raw_input('Please enter real value for '+key+': '))
        except KeyboardInterrupt: sys.exit(1)
        try: self.value = float(self.value)
        except: self.value = self.min
        if self.value > self.min and self.value < self.max: return (1,self.value)
    else: return (0,self.value)

#=======================================================================================================
#   Dictionary of actual command line arguments, etc.
#
class ArgDict (RDict.RArgs):
  def __init__(self, name = "ArgDict", argList = None, localDict = 0, addr = None):
    if argList is None:
      argList = []
    if localDict or '-localDict' in argList:
      self.purelocal = 1
    else:
      self.purelocal = 0
    RDict.RArgs.__init__(self, name, addr = addr, purelocal = self.purelocal)
    self.local  = {}
    self.target = ['default']
    #  the list of targets is always local
    #  These should not be listed here, but need to be set
    #  before the command line is parse, hence put here
    self.setLocalType('install',ArgInt())
    self.setLocalType('fileset',ArgString())
    self.insertArgs(argList)
    return

  def __setitem__(self,key,value):
    if self.local.has_key(key):
      self.local[key].value = value
    elif self.purelocal:
      self.local[key] = Arg(value)
    else:
      # set the value into the remote dictionary
      RDict.RArgs.__setitem__(self,key,Arg(value))

  def __getitem__(self,key):
    if self.local.has_key(key):
      rval = self.local[key]
      return rval.getValue(key)[1]
    elif self.purelocal:
      self.local[key] = ArgString()
      return self.local[key].getValue(key)[1]
    else:
      # get the remote object
      try: rval = RDict.RArgs.__getitem__(self,key)
      except: rval = ArgString()
      # get the value from it
      tup  = rval.getValue(key)
      # if needed, save the value back to the server
      if tup[0]: RDict.RArgs.__setitem__(self,key,rval)
      return tup[1]

  def __delitem__(self,key):
    if self.local.has_key(key): del self.local[key]
    elif not self.purelocal: RDict.RArgs.__delitem__(self,key)
    
  def has_key(self,key):
    if self.local.has_key(key):
      return hasattr(self.local[key],'value')
    elif not self.purelocal:
      if RDict.RArgs.has_key(self, key):
        # get the remote object
        rval = RDict.RArgs.__getitem__(self, key)
        return hasattr(rval, 'value')
      else:
        return 0
    return 0

  def keys(self):
    return self.local.keys()+RDict.RArgs.keys(self)

  def hasType(self, key):
    if self.local.has_key(key):
      return 1
    elif not self.purelocal:
      if RDict.RArgs.has_key(self, key):
        return 1
    return 0

  def getType(self, key):
    if not self.purelocal:
      if RDict.RArgs.has_key(self,key):
        return RDict.RArgs.__getitem__(self, key)
    if self.local.has_key(key):
      return self.local[key]
    return None

  def setLocalType(self,key,arg):
    # sets properties of the option that will be requested later
    if not self.local.has_key(key):
      self.local[key] = arg
    return

  def setType(self,key,arg):
    # sets properties of the option that will be requested later
    if not self.purelocal:
      if not RDict.RArgs.has_key(self,key): RDict.RArgs.__setitem__(self,key,arg)
    else:
      self.setLocalType(key,arg)
    return

  def insertArg(self, key, value, arg):
    if not key is None:
      if self.hasType(key):
        self[key] = self.getType(key).convertValue(value)
      else:
        self[key] = value
    else:
      if not self.target == ['default']:
        self.target.append(arg)
      else:
        self.target = [arg]
    return

  def insertArgs(self, args):
    if isinstance(args, list):
      for arg in args:
        (key, value) = parseArgument(arg)
        self.insertArg(key, value, arg)
    elif isinstance(args, dict):
      for key in args:
        value = parseValue(args[key])
        self.insertArg(key, value, arg)
    return

if __name__ ==  '__main__':
  print "Entries in BS build system options dictionary"
  try:
    keys = ArgDict("ArgDict",sys.argv[1:]).keys()
    if keys:
      for k in keys:
        print str(k)
  except Exception, e:
    print 'ERROR: '+str(e)
    sys.exit(1)
  sys.exit(0)
  
