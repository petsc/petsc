class ArgumentProcessor(object):
  '''This class provides interaction with an RDict object, which by default is shared'''
  defaultDB = None

  def __init__(self, clArgs = None, argDB = None):
    '''Setup the argument database'''
    self.argDB    = self.createArgDB(argDB)
    if clArgs is None:
      import sys

      self.clArgs = sys.argv[1:]
    else:
      self.clArgs = clArgs
    return

  def __getstate__(self):
    '''We do not want to pickle the default RDict'''
    d = self.__dict__.copy()
    if 'argDB' in d:
      if d['argDB'] is ArgumentProcessor.defaultDB:
        del d['argDB']
      else:
        d['argDB'] = None
    return d

  def __setstate__(self, d):
    '''We must create the default RDict'''
    if not 'argDB' in d:
      self.argDB = self.createArgDB(None)
    self.__dict__.update(d)
    return

  def createArgDB(self, initDB):
    '''Create an argument database unless initDB is provided, and insert the command line arguments'''
    if not initDB is None:
      argDB = initDB
    else:
      if ArgumentProcessor.defaultDB is None:
        import RDict
        import os
        import sys

        ArgumentProcessor.defaultDB = RDict.RDict(parentDirectory = os.path.dirname(os.path.abspath(sys.modules['RDict'].__file__)))
      argDB = ArgumentProcessor.defaultDB
    return argDB

  def setupArguments(self, argDB):
    '''Setup types in the argument database
       - This method shouldbe overidden by any subclass with special arguments, making sure to call the superclass method'''
    return argDB

  def insertArguments(self, useEnvironment = 0):
    '''Put arguments in from the command line and environment'''
    if useEnvironment:
      import os

      self.argDB.insertArgs(os.environ)
    self.argDB.insertArgs(self.clArgs)
    return

  def setup(self):
    '''This method should be overidden for any setup after initialization
       - Here we determine argument types and insert arguments into the dictionary'''
    self.setupArguments(self.argDB)
    self.insertArguments()
    return

  def cleanup(self):
    '''This method should be overidden for any cleanup before finalization'''
    return
