import fileset

import sys
import traceback
import types

debugLevel    = 1
debugSections = []
debugIndent   = '  '

def debugListStr(list):
  if (debugLevel > 4) or (len(list) < 4):
    return str(list)
  else:
    return '['+str(list[0])+'-<'+str(len(list)-2)+'>-'+str(list[-1])+']'

def debugFileSetStr(set):
  if isinstance(set, fileset.FileSet):
    if set.tag:
      return '('+set.tag+')'+debugListStr(set.getFiles())
    else:
      return debugListStr(set.getFiles())
  elif type(set) == types.ListType:
    output = '['
    for fs in set:
      output += debugFileSetStr(fs)
    return output+']'
  else:
    raise RuntimeError('Invalid fileset '+set)

def debugPrint(msg, level = 1, section = None):
  indentLevel = len(traceback.extract_stack())-4
  if debugLevel >= level:
    if (not section) or (not debugSections) or (section in debugSections):
      for i in range(indentLevel):
        sys.stdout.write(debugIndent)
      print msg
