
class defaultWriter:
  def __init__(self):
    pass

  def write(self,mess):
    import sys
    sys.stdout.write(mess)
    
    
class Logger(object):
  debugLevel    = None
  debugSections = None
  debugIndent   = '  '

  def __init__(self, argDB = None, log = None):
    if Logger.debugLevel is None:
      Logger.debugLevel    = argDB['debugLevel']
    self.debugLevel        = Logger.debugLevel
    if Logger.debugSections is None:
      Logger.debugSections = argDB['debugSections']
    self.debugSections     = Logger.debugSections
    self.debugIndent       = Logger.debugIndent
    self.log               = log
    self.writer            = defaultWriter()
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
    import traceback
    import sys

    if not isinstance(level, int): raise RuntimeError('Debug level must be an integer')
    indentLevel = len(traceback.extract_stack())-4
    if not self.log is None:
      for i in range(indentLevel):
        self.log.write(self.debugIndent)
      self.log.write(msg)
      self.log.write('\n')
    if self.debugLevel >= level:
      if (not section) or (not self.debugSections) or (section in self.debugSections):
        for i in range(indentLevel):
          self.writer.write(self.debugIndent)
        self.writer.write(msg)
        self.writer.write('\n')
