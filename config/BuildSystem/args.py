from __future__ import absolute_import
class ArgumentProcessor(object):
  '''This class provides interaction with an RDict object, which by default is shared'''
  defaultDB = None

  def __init__(self, clArgs = None, argDB = None):
    '''Setup the argument database'''
    self.argDB = self.createArgDB(argDB)
    if clArgs is None:
      import sys

      self.clArgs = sys.argv[1:]
    else:
      self.clArgs = clArgs
    return

  def __getstate__(self):
    '''We do not want to pickle the default RDict'''
    d = self.__dict__.copy()
    if '_argDB' in d:
      if d['_argDB'] is ArgumentProcessor.defaultDB:
        del d['_argDB']
      else:
        d['_argDB'] = None
    return d

  def __setstate__(self, d):
    '''We must create the default RDict'''
    self.__dict__.update(d)
    if not '_argDB' in d:
      self.argDB = self.createArgDB(None)
    return

  def getArgDB(self):
    return self._argDB
  def setArgDB(self, argDB):
    self._argDB = argDB
    return
  argDB = property(getArgDB, setArgDB, doc = 'The RDict argument database')

  def createArgDB(self, initDB):
    '''Create an argument database unless initDB is provided, and insert the command line arguments'''
    if not initDB is None:
      argDB = initDB
    else:
      if ArgumentProcessor.defaultDB is None:
        import RDict
        import os
        import sys

        # Changed this to assume RDict is independent
        ArgumentProcessor.defaultDB = RDict.RDict(load = 0, autoShutdown = 0)
      argDB = ArgumentProcessor.defaultDB
    return argDB

  def setupArguments(self, argDB):
    '''Setup types in the argument database
       - This method should be overridden by any subclass with special arguments, making sure to call the superclass method'''
    return argDB

  def insertArguments(self, useEnvironment = 0):
    '''Put arguments in from the command line and environment
       - This will only insert command line arguments into a given RDict once'''
    if useEnvironment:
      import os

      self.argDB.insertArgs(os.environ)
    if not hasattr(self.argDB, '_setCommandLine'):
      self.argDB.insertArgs(self.clArgs)
      self.argDB._setCommandLine = 1
    return

  def setup(self):
    '''This method should be overridden for any setup after initialization
       - Here we determine argument types and insert arguments into the dictionary'''
    self.setupArguments(self.argDB)
    self.insertArguments()
    return

  def cleanup(self):
    '''This method should be overridden for any cleanup before finalization'''
    return
