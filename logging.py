class Logger(object):
  debugLevel    = 1
  debugSections = []
  debugIndent   = '  '

  def __init__(self, argDB = None):
    self.setFromArgs(argDB)
    self.debugLevel    = argDB['debugLevel']
    self.debugSections = argDB['debugSections']
    return

  def setFromArgs(self, argDB):
    '''Setup types in the argument database'''
    import nargs

    argDB.setType('debugLevel',    nargs.ArgInt(None, None, 'Integer 0 to 4, where a higher level means more detail', 0, 5))
    argDB.setType('debugSections', nargs.Arg(None, None, 'Message types to print, e.g. [compile,link,bk,install]'))
    return argDB

  def debugListStr(self, l):
    if (self.debugLevel > 4) or (len(l) < 4):
      return str(l)
    else:
      return '['+str(l[0])+'-<'+str(len(l)-2)+'>-'+str(l[-1])+']'

  def debugFileSetStr(self, set):
    import build.fileset
    import fileset
    if isinstance(set, build.fileset.FileSet) or isinstance(set, fileset.FileSet):
      s = ''
      if set.tag:
        s += '('+set.tag+')'
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
    if self.debugLevel >= level:
      if (not section) or (not self.debugSections) or (section in self.debugSections):
        for i in range(indentLevel):
          sys.stdout.write(self.debugIndent)
        print msg
