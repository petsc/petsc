import args
import sys

try:
  enumerate([0, 1])
except NameError:
  def enumerate(l):
    return zip(range(len(l)), l)

class Logger(args.ArgumentProcessor):
  '''This class creates a shared log and provides methods for writing to it'''
  defaultLog = None
  defaultOut = sys.stdout

  def __init__(self, clArgs = None, argDB = None, log = None, out = defaultOut, debugLevel = None, debugSections = None, debugIndent = None):
    args.ArgumentProcessor.__init__(self, clArgs, argDB)
    self.logName       = None
    self.log           = log
    self.out           = out
    self.debugLevel    = debugLevel
    self.debugSections = debugSections
    self.debugIndent   = debugIndent
    return

  def __getstate__(self):
    '''We do not want to pickle the default log stream'''
    d = args.ArgumentProcessor.__getstate__(self)
    if 'log' in d and d['log'] is Logger.defaultLog:
      del d['log']
    if 'out' in d:
      if d['out'] is Logger.defaultOut:
        del d['out']
      else:
        d['out'] = None
    return d

  def __setstate__(self, d):
    '''We must create the default log stream'''
    args.ArgumentProcessor.__setstate__(self, d)
    if not 'log' in d:
      self.log = self.createLog(None)
    if not 'out' in d:
      self.out = Logger.defaultOut
    self.__dict__.update(d)
    return

  def setupArguments(self, argDB):
    '''Setup types in the argument database'''
    import nargs

    argDB = args.ArgumentProcessor.setupArguments(self, argDB)
    argDB.setType('log',           nargs.Arg(None, 'build.log', 'The filename for the log'))
    argDB.setType('debugLevel',    nargs.ArgInt(None, 3, 'Integer 0 to 4, where a higher level means more detail', 0, 5))
    argDB.setType('debugSections', nargs.Arg(None, [], 'Message types to print, e.g. [compile,link,bk,install]'))
    argDB.setType('debugIndent',   nargs.Arg(None, '  ', 'The string used for log indentation'))
    argDB.setType('scrollOutput',  nargs.ArgBool(None, 1, 'Flag to allow output to scroll rather than overwriting a single line'))
    argDB.setType('noOutput',      nargs.ArgBool(None, 0, 'Flag to suppress output to the terminal'))
    return argDB

  def setup(self):
    '''Setup the terminal output and filtering flags'''
    self.log = self.createLog(self.logName, self.log)
    args.ArgumentProcessor.setup(self)

    if self.argDB['noOutput']:
      self.out           = None
    if self.debugLevel is None:
      self.debugLevel    = self.argDB['debugLevel']
    if self.debugSections is None:
      self.debugSections = self.argDB['debugSections']
    if self.debugIndent is None:
      self.debugIndent   = self.argDB['debugIndent']
    return

  def checkLog(self, logName):
    import nargs
    import os

    if logName is None:
      logName = nargs.Arg.findArgument('log', self.clArgs)
    if logName is None:
      if 'log' in self.argDB:
        logName    = self.argDB['log']
      else:
        logName    = 'build.log'
    self.logName   = logName
    self.logExists = os.path.exists(self.logName)
    return self.logExists

  def createLog(self, logName, initLog = None):
    '''Create a default log stream, unless initLog is given'''
    if not initLog is None:
      log = initLog
    else:
      if Logger.defaultLog is None:
        if self.checkLog(logName):
          try:
            import os

            os.rename(self.logName, self.logName+'.bkp')
            Logger.defaultLog = file(self.logName, 'w')
          except OSError:
            print 'WARNING: Cannot backup log file, appending instead.'
            Logger.defaultLog = file(self.logName, 'a')
        else:
          Logger.defaultLog = file(self.logName, 'w')
      log = Logger.defaultLog
    return log

  def getLinewidth(self):
    if not hasattr(self, '_linewidth'):
      if self.out is None or not self.out.isatty() or self.argDB['scrollOutput']:
        self._linewidth = -1
      else:
        try:
          import curses

          curses.setupterm()
          (y, self._linewidth) = curses.initscr().getmaxyx()
          curses.endwin()
        except curses.error:
          self._linewidth = -1
        except ImportError:
          self._linewidth = -1
    return self._linewidth
  def setLinewidth(self, linewidth):
    self._linewidth = linewidth
    return
  linewidth = property(getLinewidth, getLinewidth, doc = 'The maximum number of characters per log line')

  def checkWrite(self, f, debugLevel, debugSection, writeAll = 0):
    '''Check whether the log line should be written
       - If writeAll is true, return true
       - If debugLevel >= current level, and debugSection in current section or sections is empty, return true'''
    if not isinstance(debugLevel, int):
      raise RuntimeError('Debug level must be an integer')
    if f is None:
      return False
    if writeAll:
      return True
    if self.debugLevel >= debugLevel and (not len(self.debugSections) or debugSection in self.debugSections):
      return True
    return False

  def logIndent(self, debugLevel = -1, debugSection = None):
    '''Write the proper indentation to the log streams'''
    import traceback

    indentLevel = len(traceback.extract_stack())-5
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        for i in range(indentLevel):
          f.write(self.debugIndent)
    return

  def logClear(self):
    '''Clear the current line if we are not scrolling output'''
    if not self.out is None and self.linewidth > 0:
      self.out.write(''.join(['\b'] * self.linewidth))
    return

  def logWrite(self, msg, debugLevel = -1, debugSection = None, forceScroll = 0):
    '''Write the message to the log streams'''
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        if not forceScroll and not writeAll and self.linewidth > 0:
          self.logClear()
          f.write(msg[0:self.linewidth])
          f.write(''.join([' '] * (self.linewidth - len(msg))))
        else:
          f.write(msg)
        if hasattr(f, 'flush'):
          f.flush()
    return

  def logPrint(self, msg, debugLevel = -1, debugSection = None, indent = 1):
    '''Write the message to the log streams with proper indentation and a newline'''
    if indent:
      self.logIndent(debugLevel, debugSection)
    self.logWrite(msg, debugLevel, debugSection)
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        if writeAll or self.linewidth < 0:
          f.write('\n')
    return

class defaultWriter:
  def __init__(self):
    pass

  def write(self,mess):
    sys.stdout.write(mess)

  def tab(self,tab):
    for i in range(0,tab-1):
      sys.stdout.write(' ')
    
dW = defaultWriter()

class LoggerOld(object):
  debugLevel    = None
  debugSections = None
  debugIndent   = ' '

  def __init__(self, argDB = None, log = None):
    if LoggerOld.debugLevel is None:
      LoggerOld.debugLevel    = argDB['debugLevel']
    self.debugLevel        = LoggerOld.debugLevel
    if LoggerOld.debugSections is None:
      LoggerOld.debugSections = argDB['debugSections']
    self.debugSections     = LoggerOld.debugSections
    self.debugIndent       = LoggerOld.debugIndent
    self.log               = log
    return

  def setWriter(self,writer):
    '''Allows sending the debug message to an alternative than stdout'''
    self.writer = writer
    
  def debugListStr(self, l):
    if (self.debugLevel > 4) or (len(l) < 4):
      return str(l)
    else:
      return '['+str(l[0])+'-<'+str(len(l)-2)+'>-'+str(l[-1])+']'

  def debugFileSetStr(self, set):
    import build.fileset
    if isinstance(set, build.fileset.FileSet):
      s = ''
      if set.tag:
        s += '('+set.tag+')'
      if isinstance(set, build.fileset.RootedFileSet):
        s += '('+set.projectUrl+')'
        s += '('+set.projectRoot+')'
      s += self.debugListStr(set)
      for child in set.children:
        s += ', '+self.debugFileSetStr(child)
      return s
    elif isinstance(set, list):
      return self.debugListStr(set)
    raise RuntimeError('Invalid fileset '+str(set))

  def debugPrint(self, msg, level = 1, section = None):
    global dW
    import traceback
    import sys

    if not isinstance(level, int): raise RuntimeError('Debug level must be an integer')
    indentLevel = len(traceback.extract_stack())-5
    if not self.log is None:
      for i in range(indentLevel):
        self.log.write(self.debugIndent)
      self.log.write(msg)
      self.log.write('\n')
    if self.debugLevel >= level:
      if (not section) or (not self.debugSections) or (section in self.debugSections):
        dW.tab(indentLevel)
        dW.write(msg+'\n')

