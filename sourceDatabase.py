import logging

import cPickle
import errno
import md5
import os
import re
import string
import sys
import time

class SourceDB (dict, logging.Logger):
  includeRE = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')

  def __init__(self):
    dict.__init__(self)

  def __str__(self):
    output = ''
    for source in self:
      (checksum, mtime, timestamp, dependencies) = self[source]
      output += source+'\n'
      output += '  Checksum:  '+str(checksum)+'\n'
      output += '  Mod Time:  '+str(mtime)+'\n'
      output += '  Timestamp: '+str(timestamp)+'\n'
      output += '  Deps: '+str(dependencies)+'\n'
    return output

  def getChecksum(self, source):
    '''This should be a class method'''
    f = open(source, 'r')
    m = md5.new()
    size = 1024*1024
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

  def updateSource(self, source):
    dependencies = ()
    try:
      (checksum, mtime, timestamp, dependencies) = self[source]
    except KeyError:
      pass
    self.debugPrint('Updating '+source+' in source database', 3, 'sourceDB')
    self[source] = (self.getChecksum(source), os.path.getmtime(source), time.time(), dependencies)

  def calculateDependencies(self):
    self.debugPrint('Recalculating dependencies', 1, 'sourceDB')
    for source in self:
      self.debugPrint('Calculating '+source, 3, 'sourceDB')
      (checksum, mtime, timestamp, dependencies) = self[source]
      newDep = []
      try:
        file = open(source, 'r')
      except IOError, e:
        if e.errno == errno.ENOENT:
          del self[source]
        else:
          raise e
      comps  = string.split(source, '/')
      for line in file.readlines():
        m = self.includeRE.match(line)
        if m:
          filename  = m.group('includeFile')
          matchNum  = 0
          matchName = filename
          self.debugPrint('  Includes '+filename, 3, 'sourceDB')
          for s in self:
            if string.find(s, filename) >= 0:
              self.debugPrint('    Checking '+s, 3, 'sourceDB')
              c = string.split(s, '/')
              for i in range(len(c)):
                if not comps[i] == c[i]: break
              if i > matchNum:
                self.debugPrint('    Choosing '+s+'('+str(i)+')', 3, 'sourceDB')
                matchName = s
                matchNum  = i
          newDep.append(matchName)
      # Grep for #include, then put these files in a tuple, we can be recursive later in a fixpoint algorithm
      self[source] = (checksum, mtime, timestamp, tuple(newDep))

if __name__ == '__main__':
  if os.path.exists(sys.argv[1]):
    dbFile   = open(sys.argv[1], 'r')
    sourceDB = cPickle.load(dbFile)
    dbFile.close()
  else:
    sys.exit(0)
  if not isinstance(sourceDB, SourceDB):
    newDB = SourceDB()
    newDB.update(sourceDB)
    sourceDB = newDB
  print sourceDB
  if len(sys.argv) > 2:
    dbFile = open(sys.argv[2], 'w')
    cPickle.dump(sourceDB, dbFile)
    dbFile.close()
