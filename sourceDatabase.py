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
      (checksum, mtime, timestamp, dependencies, updated) = self[source]
      output += source+'\n'
      output += '  Checksum:  '+str(checksum)+'\n'
      output += '  Mod Time:  '+str(mtime)+'\n'
      output += '  Timestamp: '+str(timestamp)+'\n'
      output += '  Deps: '+str(dependencies)+'\n'
      output += '  Updated: '+str(updated)+'\n'
    return output

  def getChecksum(self, source):
    '''This should be a class method'''
    if isinstance(source, file):
      f = source
    else:
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
      (checksum, mtime, timestamp, dependencies, updated) = self[source]
    except KeyError:
      pass
    self.debugPrint('Updating '+source+' in source database', 3, 'sourceDB')
    self[source] = (self.getChecksum(source), os.path.getmtime(source), time.time(), dependencies, 1)

  def setUpdateFlag(self, source):
    self.debugPrint('Setting update flag for '+source+' in source database', 4, 'sourceDB')
    (checksum, mtime, timestamp, dependencies, updated) = self[source]
    self[source] = (checksum, mtime, timestamp, dependencies, 1)

  def clearUpdateFlag(self, source):
    self.debugPrint('Clearing update flag for '+source+' in source database', 4, 'sourceDB')
    (checksum, mtime, timestamp, dependencies, updated) = self[source]
    self[source] = (checksum, mtime, timestamp, dependencies, 0)

  def calculateDependencies(self):
    self.debugPrint('Recalculating dependencies', 1, 'sourceDB')
    for source in self:
      self.debugPrint('Calculating '+source, 3, 'sourceDB')
      (checksum, mtime, timestamp, dependencies, updated) = self[source]
      newDep = []
      try:
        file = open(source, 'r')
      except IOError, e:
        if e.errno == errno.ENOENT:
          del self[source]
        else:
          raise e
      comps  = string.split(source, '/')
      for line in file.xreadlines():
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
      self[source] = (checksum, mtime, timestamp, tuple(newDep), updated)

class DependencyAnalyzer (logging.Logger):
  def __init__(self, sourceDB):
    self.sourceDB  = sourceDB
    self.includeRE = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')

  def resolveDependency(self, source, dep):
    if dep in sourceDB: return dep
    # Choose the entry in sourceDB whose base matches dep,
    #   and who has the most path components in common with source
    # This should be replaced by an appeal to cpp
    matchNum   = 0
    matchName  = dep
    components = string.split(source, os.sep)
    self.debugPrint('  Includes '+filename, 3, 'sourceDB')
    for s in self.sourceDB:
      if string.find(s, dep) >= 0:
        self.debugPrint('    Checking '+s, 3, 'sourceDB')
        comp = string.split(s, os.sep)
        for i in range(len(comp)):
          if not components[i] == comp[i]: break
        if i > matchNum:
          self.debugPrint('    Choosing '+s+'('+str(i)+')', 3, 'sourceDB')
          matchName = s
          matchNum  = i
    if not matchName in sourceDB: raise RuntimeError('Invalid #include '+matchName+' in '+source)
    return matchName

  def getNeighbors(self, source):
    file = open(source, 'r')
    adj  = []
    for line in file.xreadlines():
      match = self.includeRE.match(line)
      if match:
        adj.append(self.resolveDependency(source, m.group('includeFile')))
    return adj

  def calculateDependencies(self):
    '''Should this be a generator?
    First assemble the DAG using #include relations
    Then calculate the depdencies with all pairs shortest-path
      - I think Floyd-Warshell and N-source Dijkstra are just as good
    '''
    # Assembling DAG
    dag = {}
    for source in self.sourceDB:
      try:
        dag[source] = self.getNeighbors(self, source)
      except IOError, e:
        if e.errno == errno.ENOENT:
          del self[source]
        else:
          raise e
    # Finding all-pairs shortest path

if __name__ == '__main__':
  if os.path.exists(sys.argv[1]):
    dbFile   = open(sys.argv[1], 'r')
    sourceDB = cPickle.load(dbFile)
    dbFile.close()
  else:
    sys.exit(0)
  newDB = SourceDB()
  for key in sourceDB:
    (checksum, mtime, timestamp, dependencies,w) = sourceDB[key]
    newDB[key] = (checksum, mtime, timestamp, dependencies, 0)
  sourceDB = newDB
  if len(sys.argv) > 2:
    dbFile = open(sys.argv[2], 'w')
    cPickle.dump(sourceDB, dbFile)
    dbFile.close()
