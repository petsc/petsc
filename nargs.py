#
#   Provides a demon dictionary that requests entries interactively from the
#  user if they are not yet in the dictionay
#
import atexit
import os
import re
import string
import sys
import types
import UserDict
import readline
import rargs

def parseArg(arg):
  if arg and arg[0] == '[' and arg[-1] == ']':
    if len(arg) > 2: arg = string.split(arg[1:-1], ',')
    else:            arg = []
  return arg

def insertArgList(list,argList):
  if not type(argList) == types.ListType: return
  for arg in argList:
    if arg[0] == '-':
      (key, val) = string.split(arg[1:], '=')
      list[key]  = parseArg(val)
    else:
      if list.has_key('target') and not list['target'] == ['default']:
        list['target'].append(arg)
      else:
        list['target'] = [arg]
        
    
#  The base class of objects that are stored in the ArgDict
class Arg:
  def __init__(self,value):
    self.value = value

  def getValue(self,key):
    #   First argument = 0 indicates already contains value
    #                    1 indicates got the value, needs to be put back in dictionary server
    return (0,self.value)

#  Objects that are stored in the ArgDict that represent directories
class ArgDir:
  def __init__(self,dirmustexist = 1, help = None):
    self.exist = dirmustexist
    self.help  = help
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      try:
        import GUI.FileBrowser
        import SIDL.Loader
        db = GUI.FileBrowser.FileBrowser(SIDL.Loader.createClass('GUI.Default.DefaultFileBrowser'))
      except:
        # default to getting directory as string
        return (1,ArgString(self.help).getValue(key))
      if self.help: db.setTitle(self.help)
      else:         db.setTitle('Select the directory for'+key)
      db.setMustExist(self.exist)
      self.value = db.getDirectory()
      return (1,self.value)
    else: return (0,self.value)
      
#  Objects that are stored in the ArgDict that are strings
class ArgString:
  def __init__(self,help = None):
    self.help  = help
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      if self.help: print self.help
      try:                      self.value = parseArg(raw_input('Please enter value for '+key+':'))
      except KeyboardInterrupt:	self.value = ''
      return (1,self.value)
    else: return (0,self.value)
    
#  Objects that are stored in the ArgDict that are integers
class ArgInt:
  def __init__(self,help = None,min = -1000000, max = 1000000):
    self.help  = help
    self.min   = min
    self.max   = max
    
  def getValue(self,key):
    if not hasattr(self,'value'):
      if self.help: print self.help
      while 1:
        try: self.value = parseArg(raw_input('Please enter integer value for '+key+':'))
        except KeyboardInterrupt: self.value = self.min
        try: self.value = int(self.value)
        except: self.value = self.min
        if self.value > self.min and self.value < self.max: return (1,self.value)
    else: return (0,self.value)

#=======================================================================================================
#   Dictionary of actual command line arguements, etc.
#
class ArgDict (rargs.RArgs):
  def __init__(self, name = "ArgDict",argList = None):
    rargs.RArgs.__init__(self,name)
    insertArgList(self,argList)

  def __setitem__(self,key,value):
    rargs.RArgs.__setitem__(self,key,Arg(value))

  def __getitem__(self,key):
    # get the remote object
    try: rval = rargs.RArgs.__getitem__(self,key)
    except: rval = ArgString()
    # get the value from it
    tup  = rval.getValue(key)
    # if needed, save the value back to the server
    if tup[0]: rargs.RArgs.__setitem__(self,key,rval)
    return tup[1]

  def __delitem__(self,key):
    rargs.RArgs.__delitem__(self,key)
    
  def has_key(self,key):
    # get the remote object
    try: rval = rargs.RArgs.__getitem__(self,key)
    except: return 0
    return hasattr(rval,'value')
  
  def setType(self,key,arg):
    if not rargs.RArgs.has_key(self,key): rargs.RArgs.__setitem__(self,key,arg)


if __name__ ==  '__main__':
  a = ArgDict("ArgDict",sys.argv[1:])
  a['hi'] = 22
  print a.dicts()
  print a['hi']
  print a['joe']
#  a.setType('mpi_dir',ArgDir())
#  print a['mpi_dir']
#  a.setType('string',ArgString('enter a string'))
  print a['string']
  print a['newstring']
  a.setType('mpi',ArgInt('greetings',0))
  print a['mpi']
  print a.has_key('mpi')
  print a.has_key('does not')
  
