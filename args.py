import logging

import atexit
import cPickle
import os
import re
import string
import sys
import types
import UserDict
import readline   #allows editing of raw_input line as typed (so delete works :-)

class ArgDict (UserDict.UserDict, logging.Logger):
  def __init__(self, filename = None, defaultParent = None):
    UserDict.UserDict.__init__(self)
    self.filename      = filename
    self.load(filename)
    atexit.register(self.save)
    self.interactive   = 1
    self.metadata      = {'help' : {}, 'default' : {}, 'parent' : {}, 'tester' : {}}
    self.argRE         = re.compile(r'\$(\w+|\{[^}]*\})')
    self.defaultParent = defaultParent

  def __getitem__(self, key):
    ok = 1
    if not self.data.has_key(key):
      (ok, item) = self.getMissingItem(key)
      if ok:
        self.data[key] = item
    if not ok:
      print
      sys.exit('Unable to get argument \''+key+'\'')
    return self.data[key]

  def has_key(self, key):
    if self.data.has_key(key):
      return 1
    elif self.getParent(key):
      return self.getParent(key).has_key(key)
    else:
      return 0

  def getMissingItem(self, key):
    if self.getParent(key):
      (ok, item) = self.retrieveItem(key, self.getParent(key))
      if ok: return (ok, item)
    return self.requestItem(key)

  def retrieveItem(self, key, parent):
    if parent.has_key(key):
      return (1, parent[key])
    else:
      return (0, None)

  def requestItem(self, key):
    if not self.interactive: return (0, None)
    if self.metadata['help'].has_key(key): print self.metadata['help'][key]
    while 1:
	try:
	    value = self.parseArg(raw_input('Please enter value for '+key+':'))
	except KeyboardInterrupt:
	    return (0, None)
	if self.metadata['tester'].has_key(key): 
            (result,value) = self.metadata['tester'][key].test(value)
	    if result:
		return (1,value)
	    else:
		print 'Try again'
	else:
	    return (1,value)


  def load(self, filename):
    if filename and os.path.exists(filename):
      dbFile    = open(filename, 'r')
      self.data = cPickle.load(dbFile)
      dbFile.close()

  def save(self):
    self.debugPrint('Saving argument database in '+self.filename, 2, 'argDB')
    dbFile = open(self.filename, 'w')
    cPickle.dump(self.data, dbFile)
    dbFile.close()

  def inputDefaultArgs(self):
    for key in self.metadata['default'].keys():
      if not self.has_key(key): self[key] = self.metadata['default'][key]

  def inputEnvVars(self):
    for key in os.environ.keys():
      self[key] = self.parseArg(os.environ[key])

  def inputCommandLineArgs(self, argList):
    if not type(argList) == types.ListType: return
    for arg in argList:
      if not arg[0] == '-':
        if self.has_key('target') and not self['target'] == ['default']:
          self['target'].append(arg)
        else:
          self['target'] = [arg]
      else:
        # Could try just using eval() on val, but we would need to quote lots of stuff
        (key, val) = string.split(arg[1:], '=')
        self[key]  = self.parseArg(val)

  def input(self, clArgs = None):
    self.inputDefaultArgs()
    self.inputEnvVars()
    self.inputCommandLineArgs(clArgs)
    self.setFromArgs(self)
    if self.filename: self.debugPrint('Read source database from '+self.filename, 2, 'argDB')

  def setHelp(self, key, docString):
    self.metadata['help'][key] = docString

  def setTester(self, key, docString):
    self.metadata['tester'][key] = docString

  def setDefault(self, key, default):
    self.metadata['default'][key] = default

  def setParent(self, key, parent):
    """This can be a filename or an ArgDict object. Arguments of form $var and ${var} in the filename will be expanded"""
    self.metadata['parent'][key] = parent

  def getParent(self, key):
    if self.metadata['parent'].has_key(key):
      parent = self.metadata['parent'][key]
    elif self.defaultParent:
      parent = self.defaultParent
    else:
      parent = None

    if parent:
      if type(parent) == types.StringType:
        parent = self.expandVars(parent)
        if not os.path.exists(parent):
          raise RuntimeError('Invalid parent database ('+parent+') for '+key)
        self.metadata['parent'][key] = ArgDict(parent)
      elif not isinstance(parent, ArgDict):
        raise RuntimeError('Invalid parent database ('+parent+') for '+key)
      return self.metadata['parent'][key]
    else:
      return None

  def parseArg(self, arg):
    if arg and arg[0] == '[' and arg[-1] == ']':
      if len(arg) > 2:
        arg = string.split(arg[1:-1], ',')
      else:
        arg = []
    return arg

  def expandVars(self, path):
    """Expand arguments of form $var and ${var}"""
    if '$' not in path: return path
    i = 0
    while 1:
      m = self.argRE.search(path, i)
      if not m:
        break
      i, j = m.span(0)
      name = m.group(1)
      if name[:1] == '{' and name[-1:] == '}':
        name = name[1:-1]
      tail = path[j:]
      path = path[:i] + self[name]
      i    = len(path)
      path = path + tail
    return path
