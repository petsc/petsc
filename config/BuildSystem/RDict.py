'''A typed, persistent dictionary

    RDict is a typed, persistent dictionary intended to manage
    all arguments or options for a program. The interface remains exactly the
    same as dict, but the storage is more complicated.

    Argument typing is handled by wrapping all values stored in the dictionary
    with nargs.Arg or a subclass. A user can call setType() to set the type of
    an argument without any value being present. Whenever __getitem__() or
    __setitem__() is called, values are extracted or replaced in the wrapper.
    These wrappers can be accessed directly using getType(), setType(), and
    types().

    The persistence mechanism is a pickle file, RDict.db, written whenever an
    argument is changed. A timer thread is created after an initial change, so
    that many rapid changes do not cause many writes. Each time a dictionary is
    created, the current directory is searched for an RDict.db file, and if
    found the contents are loaded into the dictionary.

    This script also provides some default actions on the dictionary in the
    current directory:

      - client
        Lists the contents.

      - clear
        Clears the contents.

      - insert <key> <value>
        Inserts the key-value pair.

      - remove <key>
        Removes the given key.
'''
import nargs

import pickle
import os
import sys
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads is None:
  useThreads = 0 # workaround issue with parallel configure
elif useThreads == 'no' or useThreads == '0':
  useThreads = 0
elif useThreads == 'yes' or useThreads == '1':
  useThreads = 1
else:
  raise RuntimeError('Unknown option value for --useThreads ',useThreads)

class RDict(dict):
  '''An RDict is a typed dictionary. All elements derive from the
Arg class, which wraps the usual value.'''
  def __init__(self, load = 1, autoShutdown = 1, readonly = False):
    import atexit

    self.logFile      = None
    self.setupLogFile()
    self.target       = ['default']
    self.saveTimer    = None
    self.saveFilename = 'RDict.db'
    self.readonly     = readonly
    self.writeLogLine('Greetings')
    if load: self.load()
    if autoShutdown and useThreads:
      atexit.register(self.shutdown)
    return

  def __getstate__(self):
    '''Remove the log file from the dictionary before pickling'''
    self.writeLogLine('Pickling RDict')
    d = self.__dict__.copy()
    if 'saveTimer' in d: del d['saveTimer']
    if '_setCommandLine' in d: del d['_setCommandLine']
    del d['logFile']
    return d

  def __setstate__(self, d):
    '''Reopen the log file after unpickling'''
    self.logFile  = open('RDict.log', 'a')
    self.writeLogLine('Unpickling RDict')
    self.__dict__.update(d)
    return

  def setupLogFile(self, filename = 'RDict.log'):
    if not self.logFile is None:
      self.logFile.close()
    if os.path.isfile(filename) and os.stat(filename).st_size > 10*1024*1024:
      if os.path.isfile(filename+'.bkp'):
        os.remove(filename+'.bkp')
      os.rename(filename, filename+'.bkp')
      self.logFile = open(filename, 'w')
    else:
      self.logFile = open(filename, 'a')
    return

  def writeLogLine(self, message):
    '''Writes the message to the log along with the current time'''
    import time
    self.logFile.write('('+str(os.getpid())+')('+str(id(self))+')'+message+' ['+time.asctime(time.localtime())+']\n')
    self.logFile.flush()
    return

  def __len__(self):
    return dict.__len__(self)

  def getType(self, key):
    '''Returns the Arg object for the key, or None if not found'''
    try:
      value = dict.__getitem__(self, key)
      self.writeLogLine('getType: Getting local type for '+key+' '+str(value))
      return value
    except KeyError:
      pass
    return None

  def dict_has_key(self, key):
    """Utility to check whether the key is present in the dictionary without RDict side-effects."""
    return key in dict(self)

  def __getitem__(self, key):
    '''Returns the value of the Arg for the key
       - If the value has not been set, the user will be prompted for input'''
    if self.dict_has_key(key):
      self.writeLogLine('__getitem__: '+key+' has local type')
    else:
      self.writeLogLine('__getitem__: Setting local type for '+key)
      dict.__setitem__(self, key, nargs.Arg(key))
      #self.save()
    self.writeLogLine('__getitem__: Setting local value for '+key)
    return dict.__getitem__(self, key).getValue()

  def setType(self, key, value):
    '''Sets the type for this key
       - If a value for the key already exists, it is converted to the new type'''
    if not isinstance(value, nargs.Arg):
      raise TypeError('An argument type must be a subclass of Arg')
    value.setKey(key)
    if self.dict_has_key(key):
      v = dict.__getitem__(self, key)
      if v.isValueSet():
        try:
          value.setValue(v.getValue())
        except TypeError:
          print(value.__class__.__name__[3:])
          print('-----------------------------------------------------------------------')
          print('Warning! Incorrect argument type specified: -'+str(key)+'='+str(v.getValue())+' - expecting type '+value.__class__.__name__[3:]+'.')
          print('-----------------------------------------------------------------------')
          pass
    dict.__setitem__(self, key, value)
    #self.save()
    return

  def __setitem__(self, key, value):
    '''Sets the value of the Arg for the key'''
    if not self.dict_has_key(key):
      dict.__setitem__(self, key, nargs.Arg(key))
    dict.__getitem__(self, key).setValue(value)
    self.writeLogLine('__setitem__: Set value for '+key+' to '+str(dict.__getitem__(self, key)))
    #self.save()
    return

  def __delitem__(self, key):
    '''Deletes the Arg for the key completely'''
    if self.dict_has_key(key):
      dict.__delitem__(self, key)
      #self.save()
    return

  def clear(self):
    '''Clears the dictionary'''
    if dict.__len__(self):
      dict.clear(self)
      #self.save()
    return

  def __contains__(self, key):
    '''Checks for the key, then checks whether its value has been set'''
    if self.dict_has_key(key):
      if dict.__getitem__(self, key).isValueSet():
        self.writeLogLine('has_key: Have value for '+key)
      else:
        self.writeLogLine('has_key: Do not have value for '+key)
      return dict.__getitem__(self, key).isValueSet()
    return 0

  def get(self, key, default=None):
    if key in self:
      return self.__getitem__(key)
    else:
      return default

  def hasType(self, key):
    '''Checks whether a type has been set for the key'''
    if self.dict_has_key(key):
      return 1
    return 0

  def items(self):
    '''Return a list of all items, as (key, value) pairs.'''
    return dict.items(self)

  def localitems(self):
    '''Return a list of all the items stored locally, as (key, value) pairs.'''
    return dict.items(self)

  def keys(self):
    '''Returns the list of keys whose value has been set'''
    return [key for key in dict.keys(self) if dict.__getitem__(self, key).isValueSet()]

  def types(self):
    '''Returns the list of keys for which types are defined'''
    return dict.keys(self)

  def update(self, d):
    '''Update the dictionary with the contents of d'''
    for k in d:
      self[k] = d[k]
    return

  def updateTypes(self, d):
    '''Update types locally, which is equivalent to the dict.update() method'''
    return dict.update(self, d)

  def insertArg(self, key, value, arg):
    '''Insert a (key, value) pair into the dictionary. If key is None, arg is put into the target list.'''
    if not key is None:
      self[key] = value
    else:
      if not self.target == ['default']:
        self.target.append(arg)
      else:
        self.target = [arg]
    return

  def insertArgs(self, args):
    '''Insert some text arguments into the dictionary (list and dictionaries are recognized)'''

    if isinstance(args, list):
      for arg in args:
        (key, value) = nargs.Arg.parseArgument(arg)
        self.insertArg(key, value, arg)
    elif hasattr(args, 'keys'):
      for key in args.keys():
        if isinstance(args[key], str):
          value = nargs.Arg.parseValue(args[key])
        else:
          value = args[key]
        self.insertArg(key, value, None)
    elif isinstance(args, str):
        (key, value) = nargs.Arg.parseArgument(args)
        self.insertArg(key, value, args)
    return

  def load(self):
    '''Load the saved dictionary'''
    self.saveFilename = os.path.abspath(self.saveFilename)
    if os.path.exists(self.saveFilename):
      try:
        dbFile = open(self.saveFilename, 'rb')
        data   = pickle.load(dbFile)
        self.updateTypes(data)
        dbFile.close()
        self.writeLogLine('Loaded dictionary from '+self.saveFilename)
      except Exception as e:
        self.writeLogLine('Problem loading dictionary from '+self.saveFilename+'\n--> '+str(e))
    else:
      self.writeLogLine('No dictionary to load in this file: '+self.saveFilename)
    return

  def save(self, force = 1):
    '''Save the dictionary after 5 seconds, ignoring all subsequent calls until the save
       - Giving force = True will cause an immediate save'''
    if self.readonly: return
    if force:
      self.saveTimer = None
      # This should be a critical section
      dbFile = open(self.saveFilename, 'wb')
      data   = dict([i for i in self.localitems() if not i[1].getTemporary()])
      pickle.dump(data, dbFile)
      dbFile.close()
      self.writeLogLine('Saved local dictionary to '+os.path.abspath(self.saveFilename))
    elif not self.saveTimer:
      import threading
      self.saveTimer = threading.Timer(5, self.save, [], {'force': 1})
      self.saveTimer.daemon = True
      self.saveTimer.start()
    return

  def shutdown(self):
    '''Shutdown the dictionary, writing out any changes'''
    if self.saveTimer:
      self.saveTimer.cancel()
      self.save(force = 1)
    self.writeLogLine('Shutting down')
    self.logFile.close()
    return

if __name__ ==  '__main__':
  import sys
  try:
    if len(sys.argv) < 2:
      print('RDict.py [client | cacheClient | stampClient | clear | insert <key> <value> | remove <key>]')
    else:
      action = sys.argv[1]
      if action == 'client':
        print('Entries in dictionary')
        rdict = RDict()
        for key in rdict.types():
          if not key.startswith('cacheKey') and not key.startswith('stamp-'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'cacheClient':
        print('Cache entries in dictionary')
        rdict = RDict()
        for key in rdict.types():
          if key.startswith('cacheKey'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'stampClient':
        print('Stamp entries in dictionary')
        rdict = RDict()
        for key in rdict.types():
          if key.startswith('stamp-'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'clear':
        print('Clearing dictionary')
        rdict = RDict()
        rdict.clear()
        rdict.save()
      elif action == 'insert':
        rdict = RDict()
        rdict[sys.argv[2]] = sys.argv[3]
        rdict.save()
      elif action == 'remove':
        rdict = RDict()
        del rdict[sys.argv[2]]
        rdict.save()
      else:
        sys.exit('Unknown action: '+action)
  except Exception as e:
    import traceback
    print(traceback.print_tb(sys.exc_info()[2]))
    sys.exit(str(e))
  sys.exit(0)
