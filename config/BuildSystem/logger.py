from __future__ import absolute_import
import args
import sys
import os
import textwrap

# Ugly stuff to have curses called ONLY once, instead of for each
# new Configure object created (and flashing the screen)
global LineWidth
global RemoveDirectory
global backupRemoveDirectory
LineWidth = -1
RemoveDirectory = os.path.join(os.getcwd(),'')
backupRemoveDirectory = ''

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
    self.dividerLength = 93
    self.getRoot()
    return

  def __getstate__(self):
    '''We do not want to pickle the default log stream'''
    d = args.ArgumentProcessor.__getstate__(self)
    if 'logBkp' in d:
        del d['logBkp']
    if 'log' in d:
      if d['log'] is Logger.defaultLog:
        del d['log']
      else:
        d['log'] = None
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
    argDB.setType('log',           nargs.Arg(None, 'buildsystem.log', 'The filename for the log'))
    argDB.setType('logAppend',     nargs.ArgBool(None, 0, 'The flag determining whether we backup or append to the current log', isTemporary = 1))
    argDB.setType('debugLevel',    nargs.ArgInt(None, 3, 'Integer 0 to 4, where a higher level means more detail', 0, 5))
    argDB.setType('debugSections', nargs.Arg(None, [], 'Message types to print, e.g. [compile,link,hg,install]'))
    argDB.setType('debugIndent',   nargs.Arg(None, '  ', 'The string used for log indentation'))
    argDB.setType('scrollOutput',  nargs.ArgBool(None, 0, 'Flag to allow output to scroll rather than overwriting a single line'))
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
      if not self.argDB is None and 'log' in self.argDB:
        logName    = self.argDB['log']
      else:
        logName    = 'default.log'
    self.logName   = logName
    self.logExists = os.path.exists(self.logName)
    return self.logExists

  def createLog(self, logName, initLog = None):
    '''Create a default log stream, unless initLog is given'''
    import nargs

    if not initLog is None:
      log = initLog
    else:
      if Logger.defaultLog is None:
        appendArg = nargs.Arg.findArgument('logAppend', self.clArgs)
        if self.checkLog(logName):
          if not self.argDB is None and ('logAppend' in self.argDB and self.argDB['logAppend']) or (not appendArg is None and bool(appendArg)):
            Logger.defaultLog = open(self.logName, 'a')
          else:
            try:
              import os

              os.rename(self.logName, self.logName+'.bkp')
              Logger.defaultLog = open(self.logName, 'w')
            except OSError:
              sys.stdout.write('WARNING: Cannot backup log file, appending instead.\n')
              Logger.defaultLog = open(self.logName, 'a')
        else:
          Logger.defaultLog = open(self.logName, 'w')
      log = Logger.defaultLog
    return log

  def closeLog(self):
    '''Closes the log file'''
    self.log.close()

  def saveLog(self):
    if self.debugLevel <= 3: return
    import io
    self.logBkp = self.log
    self.log = io.StringIO()

  def restoreLog(self):
    if self.debugLevel <= 3: return
    s = self.log.getvalue()
    self.log.close()
    self.log = self.logBkp
    del(self.logBkp)
    return s

  def getLinewidth(self):
    global LineWidth
    if not hasattr(self, '_linewidth'):
      if self.out is None or not self.out.isatty() or self.argDB['scrollOutput']:
        self._linewidth = -1
      else:
        if LineWidth == -1:
          try:
            import curses

            try:
              curses.setupterm()
              (y, self._linewidth) = curses.initscr().getmaxyx()
              curses.endwin()
            except curses.error:
              self._linewidth = -1
          except:
            self._linewidth = -1
          LineWidth = self._linewidth
        else:
          self._linewidth = LineWidth
    return self._linewidth
  def setLinewidth(self, linewidth):
    self._linewidth = linewidth
    return
  linewidth = property(getLinewidth, setLinewidth, doc = 'The maximum number of characters per log line')

  def checkWrite(self, f, debugLevel, debugSection, writeAll = 0):
    '''Check whether the log line should be written
       - If writeAll is true, return true
       - If debugLevel >= current level, and debugSection in current section or sections is empty, return true'''
    if not isinstance(debugLevel, int):
      raise RuntimeError('Debug level must be an integer: '+str(debugLevel))
    if f is None:
      return False
    if writeAll:
      return True
    if self.debugLevel >= debugLevel and (not len(self.debugSections) or debugSection in self.debugSections):
      return True
    return False

  def checkANSIEscapeSequences(self, ostream):
    """
    Return True if the stream supports ANSI escape sequences, False otherwise
    """
    try:
      # _io.TextIoWrapper use 'name' attribute to store the file name
      key = ostream.name
    except AttributeError:
      return False

    try:
      return self._ansi_esc_seq_cache[key]
    except KeyError:
      pass # have not processed this stream before
    except AttributeError:
      # have never done this before
      self._ansi_esc_seq_cache = {}

    is_a_tty = hasattr(ostream,'isatty') and ostream.isatty()
    return self._ansi_esc_seq_cache.setdefault(key,is_a_tty and (
      sys.platform != 'win32' or os.environ.get('TERM','').startswith(('xterm','ANSI')) or
      # Windows Terminal supports VT codes.
      'WT_SESSION' in os.environ or
      # Microsoft Visual Studio Code's built-in terminal supports colors.
      os.environ.get('TERM_PROGRAM') == 'vscode'
    ))

  def logIndent(self, debugLevel = -1, debugSection = None, comm = None):
    '''Write the proper indentation to the log streams'''
    import traceback

    indentLevel = len(traceback.extract_stack())-5
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        if not comm is None:
          f.write('[')
          f.write(str(comm.rank()))
          f.write(']')
        for i in range(indentLevel):
          f.write(self.debugIndent)
    return

  def logBack(self):
    '''Backup the current line if we are not scrolling output'''
    if self.out is not None and self.linewidth > 0:
      self.out.write('\r')
    return

  def logClear(self):
    '''Clear the current line if we are not scrolling output'''
    out,lw = self.out,self.linewidth
    if out is not None and lw > 0:
      out.write('\r\033[K' if self.checkANSIEscapeSequences(out) else ' '*lw)
      try:
        out.flush()
      except AttributeError:
        pass
    return

  def logPrintDivider(self, single = False, length = None, **kwargs):
    if length is None:
      length = self.dividerLength
    kwargs.setdefault('rmDir',False)
    kwargs.setdefault('indent',False)
    kwargs.setdefault('forceScroll',False)
    kwargs.setdefault('forceNewLine',True)
    divider = ('-' if single else '=')*length
    return self.logPrint(divider, **kwargs)

  def logPrintWarning(self, msg, title = None, **kwargs):
    if title is None:
      title = 'WARNING'
    return self.logPrintBox(msg,title='***** {} *****'.format(title),**kwargs)

  def logPrintBox(self, msg, debugLevel = -1, debugSection = 'screen', indent = 1, comm = None, rmDir = 1, prefix = None, title = None):
    def center_wrap(banner,text,length = None,**kwargs):
      def center_line(line):
        return line.center(length).rstrip()

      if length is None:
        length = self.dividerLength
      kwargs.setdefault('break_on_hyphens',False)
      kwargs.setdefault('break_long_words',False)
      kwargs.setdefault('width',length-2)
      kwargs.setdefault('initial_indent',prefix)
      kwargs.setdefault('subsequent_indent',prefix)
      wrapped = [
        line for para in text.splitlines() for line in textwrap.wrap(textwrap.dedent(para),**kwargs)
      ]
      if len(wrapped) == 1:
        # center-justify single lines, and remove the bogus prefix
        wrapped[0] = center_line(wrapped[0].lstrip())
      if banner:
        # add the banner
        wrapped.insert(0,center_line(banner))
      return '\n'.join(wrapped)


    if prefix is None:
      prefix = ' '*2

    if rmDir:
      rmDir = center_wrap(title,self.logStripDirectory(msg))
    msg = center_wrap(title,msg)
    self.logClear()
    self.logPrintDivider(debugLevel = debugLevel, debugSection = debugSection)
    self.logPrint(msg, debugLevel = debugLevel, debugSection = debugSection, rmDir = rmDir, forceNewLine = True, forceScroll = True, indent = 0)
    self.logPrintDivider(debugLevel = debugLevel, debugSection = debugSection)
    return

  def logStripDirectory(self,msg):
    return msg.replace(RemoveDirectory,'')

  def logClearRemoveDirectory(self):
    global RemoveDirectory
    global backupRemoveDirectory
    backupRemoveDirectory = RemoveDirectory
    RemoveDirectory = ''

  def logResetRemoveDirectory(self):
    global RemoveDirectory
    global backupRemoveDirectory
    RemoveDirectory = backupRemoveDirectory


  def logWrite(self, msg, debugLevel = -1, debugSection = None, forceScroll = 0, rmDir = 1):
    '''Write the message to the log streams'''
    '''Generally goes to the file but not the screen'''
    if not msg: return
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        if rmDir:
          if isinstance(rmDir,str):
            clean_msg = rmDir
          else:
            clean_msg = self.logStripDirectory(msg)
        else:
          clean_msg = msg
        if not forceScroll and not writeAll and self.linewidth > 0:
          self.logClear()
          for ms in clean_msg.splitlines():
            f.write(ms[:self.linewidth])
        else:
          if not debugSection is None and not debugSection == 'screen' and len(msg):
            f.write(str(debugSection))
            f.write(': ')
          f.write(msg if writeAll else clean_msg)
        if hasattr(f, 'flush'):
          f.flush()
    return

  def logPrint(self, msg, debugLevel = -1, debugSection = None, indent = 1, comm = None, forceScroll = 0, rmDir = 1, forceNewLine = False):
    '''Write the message to the log streams with proper indentation and a newline'''
    '''Generally goes to the file and the screen'''
    if indent:
      self.logIndent(debugLevel, debugSection, comm)
    self.logWrite(msg, debugLevel, debugSection, forceScroll = forceScroll, rmDir = rmDir)
    for writeAll, f in enumerate([self.out, self.log]):
      if self.checkWrite(f, debugLevel, debugSection, writeAll):
        if forceNewLine or writeAll or self.linewidth < 0:
          f.write('\n')
    return


  def getRoot(self):
    '''Return the directory containing this module
       - This has the problem that when we reload a module of the same name, this gets screwed up
         Therefore, we call it in the initializer, and stash it'''
    #print '      In getRoot'
    #print hasattr(self, '__root')
    #print '      done checking'
    if not hasattr(self, '__root'):
      import os
      import sys

      # Work around a bug with pdb in 2.3
      if hasattr(sys.modules[self.__module__], '__file__') and not os.path.basename(sys.modules[self.__module__].__file__) == 'pdb.py':
        self.__root = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self.__root = os.getcwd()
    #print '      Exiting getRoot'
    return self.__root
  def setRoot(self, root):
    self.__root = root
    return
  root = property(getRoot, setRoot, doc = 'The directory containing this module')
