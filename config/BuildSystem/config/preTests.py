'''
This module tests features so broken, that he normal test apparatus is likely
o fail. For example, there are several buggy implementations of Python, we can
recognize and work around.
'''
import os, sys

class Configure:
  '''These test must run before almost any operations
  - We maintain an internal self.log array of strings so that normal logging need not be initialized
  '''
  def __init__(self, options = {}, convertOptions = False):
    self.options        = options
    self.convertOptions = convertOptions
    self.log            = []
    return

  def checkCygwin(self):
    '''Check for versions of Cygwin below 1.5.11-1
    These version have a known bug in the Python threads module
    '''
    if os.path.exists('/usr/bin/cygcheck.exe'):
      buf = os.popen('/usr/bin/cygcheck.exe -c cygwin').read()
      if buf.find('1.5.11-1') > -1:
        return 1
      else:
        return 0
    return 0

  def checkCygwinPython(self):
    '''Check for versions of Cygwin Python 2.4 and above
    These version have a known bug in the Python threads module
    '''
    if os.path.exists('/usr/bin/cygcheck.exe'):
      buf = os.popen('/usr/bin/cygcheck.exe -c python').read()
      if buf.find('2.4') > -1:
        return 1
    return 0

  def checkRedHat9(self):
    '''Check for Redhat 9
    This version have a known bug in the Python threads module
    '''
    try:
      file = open('/etc/redhat-release','r')
    except:
      return 0
    try:
      buf = file.read()
      file.close()
    except:
      # can't read file - assume dangerous RHL9
      return 1
    if buf.find('Shrike') > -1:
      return 1
    return 0

  def checkThreads(self):
    '''Check Python threading'''
    if self.checkCygwin():
      errorMsg = '''\
      =================================================================================
       *** cygwin-1.5.11-1 detected. configure.py fails with this version ***
       *** Please upgrade to cygwin-1.5.12-1 or newer version. This can   ***
       *** be done by running cygwin-setup, selecting "next" all the way. ***
      ================================================================================='''
      sys.exit(errorMsg)
    if self.checkRedHat9():
      sys.argv.append('--useThreads=0')
      self.log.append('''\
================================================================================
   *** RHL9 detected. Threads do not work correctly with this distribution ***
    ********* Disabling thread usage for this run of configure.py ***********
================================================================================''')
      if self.checkCygwinPython():
        sys.argv.append('--useThreads=0')
        self.log.append('''\
================================================================================
** Cygwin-python-2.4 detected. Threads do not work correctly with this version *
 ********* Disabling thread usage for this run of configure.py ****************
================================================================================''')
    return

  def checkOptions(self, options, convertOptions = False):
    '''Check for some initial options, and optionally give them default values
    - If convertOptions is true, process GNU options prefixes into our canonical form
    '''
    import nargs

    if convertOptions:
      nargs.Arg.processAlternatePrefixes(sys.argv)
    for name, defaultArg in options.items():
      if nargs.Arg.findArgument(name,sys.argv) is None:
        if defaultArg:
          sys.argv.append('%s=%s' % name, str(defaultArg))
    return

  def configure(self):
    self.checkThreads()
    self.checkOptions(self.options, self.convertOptions)
    return
