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

  def debugListStr(self, list):
    if (self.debugLevel > 4) or (len(list) < 4):
      return str(list)
    else:
      return '['+str(list[0])+'-<'+str(len(list)-2)+'>-'+str(list[-1])+']'

  def debugFileSetStr(self, set):
    import fileset
    if isinstance(set, fileset.FileSet):
      if set.tag:
        return '('+set.tag+')'+self.debugListStr(set.getFiles())
      else:
        return self.debugListStr(set.getFiles())
    elif isinstance(set, list):
      output = '['
      for fs in set:
        output += self.debugFileSetStr(fs)
      return output+']'
    else:
      raise RuntimeError('Invalid fileset '+str(set))

  def debugPrint(self, msg, level = 1, section = None):
    import traceback
    import sys

    indentLevel = len(traceback.extract_stack())-4
    if self.debugLevel >= level:
      if (not section) or (not self.debugSections) or (section in self.debugSections):
        for i in range(indentLevel):
          sys.stdout.write(self.debugIndent)
        print msg
