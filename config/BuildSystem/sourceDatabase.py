'''A source code database

    SourceDB is a database of file information used to determine whether files
    should be rebuilt by the build system. All files names are stored relative
    to a given root, which is intended as the root of a Project.

    Relative or absolute pathnames may be used as keys, but absolute pathnames
    must fall under the database root. The value format is a tuple of the following:

      Checksum:     The md5 checksum of the file
      Mod Time:     The time the file was last modified
      Timestamp:    The time theentry was last modified
      Dependencies: A tuple of files upon which this entry depends

    This script also provides some default actions:

      - insert <database file> <filename>
        Inserts this file from the database, or updates its entry if it
        already exists.

      - remove <database file> <filename>
        Removes this file from the database. The filename may also be a
        regular expression.

'''
from __future__ import print_function
from __future__ import absolute_import
import logger

import errno
import os
import re
import time

import pickle

try:
  from hashlib import md5 as new_md5
except ImportError:
  from md5 import new as new_md5 # novermin


class SourceDB (dict, logger.Logger):
  '''A SourceDB is a dictionary of file data used during the build process.'''
  includeRE = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')
  isLoading = 0

  def __init__(self, root, filename = None):
    dict.__init__(self)
    logger.Logger.__init__(self)
    self.root       = root
    self.filename   = filename
    if self.filename is None:
      self.filename = os.path.join(str(root), 'bsSource.db')
    self.isDirty    = 0
    return

  def __str__(self):
    output = ''
    for source in self:
      (checksum, mtime, timestamp, dependencies) = self[source]
      output += source+'\n'
      output += '  Checksum:  '+str(checksum)+'\n'
      output += '  Mod Time:  '+str(mtime)+'\n'
      output += '  Timestamp: '+str(timestamp)+'\n'
      output += '  Deps:      '+str(dependencies)+'\n'
    return output

  def __setstate__(self, d):
    logger.Logger.__setstate__(self, d)
    # We have to prevent recursive calls to this when the pickled database is loaded in load()
    #   This is to ensure that fresh copies of the database are obtained after unpickling
    if not SourceDB.isLoading:
      SourceDB.isLoading = 1
      self.load()
      SourceDB.isLoading = 0
    return

  def getRelativePath(self, path):
    '''Returns a relative source file path using the root'''
    if os.path.isabs(path):
      root = str(self.root)
      if not path.startswith(root+os.sep):
        raise ValueError('Absolute path '+path+' conflicts with root '+root)
      else:
        path = path[len(root)+1:]
    return path

  def checkValue(self, value):
    '''Validate the value, raising ValueError for problems'''
    if not isinstance(value, tuple):
      raise ValueError('Source database values must be tuples, '+str(type(value))+' given')
    if not len(value) == 4:
      raise ValueError('Source database values must have 4 items, '+str(len(value))+' given')
    (checksum, mtime, timestamp, dependencies) = value
    if not isinstance(checksum, str):
      raise ValueError('Invalid checksum for source database, '+str(type(checksum))+' given')
    if not isinstance(mtime, int):
      raise ValueError('Invalid modification time for source database, '+str(type(mtime))+' given')
    elif mtime < 0:
      raise ValueError('Negative modification time for source database, '+str(mtime))
    if not isinstance(timestamp, float):
      raise ValueError('Invalid timestamp for source database, '+str(type(timestamp))+' given')
    elif timestamp < 0:
      raise ValueError('Negative timestamp for source database, '+str(timestamp))
    if not isinstance(dependencies, tuple):
      raise ValueError('Invalid dependencies for source database, '+str(type(dependencies))+' given')
    return value

  def __getitem__(self, key):
    '''Converts the key to a relative source file path using the root'''
    return dict.__getitem__(self, self.getRelativePath(key))

  def __setitem__(self, key, value):
    '''Converts the key to a relative source file path using the root, and checks the validity of the value'''
    self.isDirty = 1
    return dict.__setitem__(self, self.getRelativePath(key), self.checkValue(value))

  def __delitem__(self, key):
    '''Converts the key to a relative source file path using the root'''
    self.isDirty = 1
    return dict.__delitem__(self, self.getRelativePath(key))

  def __contains__(self, key):
    '''Converts the key to a relative source file path using the root'''
    return dict.__contains__(self, self.getRelativePath(key))

  def has_key(self, key):
    '''This method just calls self.__contains__(key)'''
    return self.__contains__(key)

  def items(self):
    '''Converts each key to a relative source file path using the root'''
    return [(self.getRelativePath(item[0]), item[1]) for item in dict.items(self)]

  def keys(self):
    '''Converts each key to a relative source file path using the root'''
    return map(self.getRelativePath, dict.keys(self))

  def update(self, d):
    '''Update the dictionary with the contents of d'''
    self.isDirty = 1
    for k in d:
      self[k] = d[k]
    return

  def getChecksum(source, chunkSize = 1024*1024):
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if hasattr(source, 'close'):
      f = source
    else:
      f = open(source)
    m = new_md5()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()
  getChecksum = staticmethod(getChecksum)

  def getModificationTime(source):
    t = os.path.getmtime(source)
    if isinstance(t, float):
      t = int(t)
    return t
  getModificationTime = staticmethod(getModificationTime)

  def updateSource(self, source, noChecksum = 0):
    self.isDirty = 1
    dependencies = ()
    try:
      (checksum, mtime, timestamp, dependencies) = self[source]
    except KeyError:
      pass
    self.logPrint('Updating '+source+' in source database', 3, 'sourceDB')
    if noChecksum:
      checksum   = ''
    else:
      checksum   = SourceDB.getChecksum(source)
    self[source] = (checksum, SourceDB.getModificationTime(source), time.time(), dependencies)
    return

  def clearSource(self, source):
    '''This removes source information, but preserved dependencies'''
    if source in self:
      self.isDirty = 1
      self.logPrint('Clearing '+source+' from source database', 3, 'sourceDB')
      (checksum, mtime, timestamp, dependencies) = self[source]
      self[source] = ('', 0, time.time(), dependencies)
    return

  def getDependencies(self, source):
    try:
      (checksum, mtime, timestamp, dependencies) = self[source]
    except KeyError:
      dependencies = ()
    return dependencies

  def addDependency(self, source, dependency):
    self.isDirty = 1
    dependencies = ()
    try:
      (checksum, mtime, timestamp, dependencies) = self[source]
    except KeyError:
      checksum = ''
      mtime    = 0
    if not dependency in dependencies:
      self.logPrint('Adding dependency '+dependency+' to source '+source+' in source database', 3, 'sourceDB')
      dependencies = dependencies+(dependency,)
    self[source] = (checksum, mtime, time.time(), dependencies)
    return

  def calculateDependencies(self):
    self.logPrint('Recalculating dependencies', 1, 'sourceDB')
    for source in self:
      self.logPrint('Calculating '+source, 3, 'sourceDB')
      (checksum, mtime, timestamp, dependencies) = self[source]
      newDep = []
      try:
        file = open(source)
      except IOError as e:
        if e.errno == errno.ENOENT:
          del self[source]
        else:
          raise e
      comps  = source.split('/')
      for line in file:
        m = self.includeRE.match(line)
        if m:
          filename  = m.group('includeFile')
          matchNum  = 0
          matchName = filename
          self.logPrint('  Includes '+filename, 3, 'sourceDB')
          for s in self:
            if s.find(filename) >= 0:
              self.logPrint('    Checking '+s, 3, 'sourceDB')
              c = s.split('/')
              for i in range(len(c)):
                if not comps[i] == c[i]: break
              if i > matchNum:
                self.logPrint('    Choosing '+s+'('+str(i)+')', 3, 'sourceDB')
                matchName = s
                matchNum  = i
          newDep.append(matchName)
      # Grep for #include, then put these files in a tuple, we can be recursive later in a fixpoint algorithm
      self[source] = (checksum, mtime, timestamp, tuple(newDep))
      file.close()

  def load(self):
    '''Load the source database from the saved filename'''
    filename = str(self.filename)
    if os.path.exists(filename):
      self.clear()
      self.logPrint('Loading source database from '+filename, 2, 'sourceDB')
      dbFile = open(filename)
      newDB  = pickle.load(dbFile)
      dbFile.close()
      self.update(newDB)
    else:
      self.logPrint('Could not load source database from '+filename, 1, 'sourceDB')
    return

  def save(self, force = 0):
    '''Save the source database to a file. The saved database with have path names relative to the root.'''
    if not self.isDirty and not force:
      self.logPrint('No need to save source database in '+str(self.filename), 2, 'sourceDB')
      return
    filename = str(self.filename)
    if os.path.exists(os.path.dirname(filename)):
      self.logPrint('Saving source database in '+filename, 2, 'sourceDB')
      dbFile = open(filename, 'w')
      pickle.dump(self, dbFile)
      dbFile.close()
      self.isDirty = 0
    else:
      self.logPrint('Could not save source database in '+filename, 1, 'sourceDB')
    return

class DependencyAnalyzer (logger.Logger):
  def __init__(self, sourceDB):
    logger.Logger.__init__(self)
    self.sourceDB  = sourceDB
    self.includeRE = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')
    return

  def resolveDependency(self, source, dep):
    if dep in self.sourceDB: return dep
    # Choose the entry in sourceDB whose base matches dep,
    #   and who has the most path components in common with source
    # This should be replaced by an appeal to cpp
    matchNum   = 0
    matchName  = dep
    components = source.split(os.sep)
    self.logPrint('  Includes '+filename, 3, 'sourceDB')
    for s in self.sourceDB:
      if s.find(dep) >= 0:
        self.logPrint('    Checking '+s, 3, 'sourceDB')
        comp = s.split(os.sep)
        for i in range(len(comp)):
          if not components[i] == comp[i]: break
        if i > matchNum:
          self.logPrint('    Choosing '+s+'('+str(i)+')', 3, 'sourceDB')
          matchName = s
          matchNum  = i
    if not matchName in self.sourceDB: raise RuntimeError('Invalid #include '+matchName+' in '+source)
    return matchName

  def getNeighbors(self, source):
    file = open(source)
    adj  = []
    for line in file:
      match = self.includeRE.match(line)
      if match:
        adj.append(self.resolveDependency(source, m.group('includeFile')))
    file.close()
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
      except IOError as e:
        if e.errno == errno.ENOENT:
          del self[source]
        else:
          raise e
    # Finding all-pairs shortest path

if __name__ == '__main__':
  import sys
  try:
    if len(sys.argv) < 3:
      print('sourceDatabase.py <database filename> [insert | remove] <filename>')
    else:
      if os.path.exists(sys.argv[1]):
        dbFile   = open(sys.argv[1])
        sourceDB = pickle.load(dbFile)
        dbFile.close()
      else:
        sys.exit('Could not load source database from '+sys.argv[1])
      if sys.argv[2] == 'insert':
        if sys.argv[3] in sourceDB:
          self.logPrint('Updating '+sys.argv[3], 3, 'sourceDB')
        else:
          self.logPrint('Inserting '+sys.argv[3], 3, 'sourceDB')
        self.sourceDB.updateSource(sys.argv[3])
      elif sys.argv[2] == 'remove':
        if sys.argv[3] in sourceDB:
          sourceDB.logPrint('Removing '+sys.argv[3], 3, 'sourceDB')
          del self.sourceDB[sys.argv[3]]
        else:
          sourceDB.logPrint('Matching regular expression '+sys.argv[3]+' over source database', 1, 'sourceDB')
          removeRE = re.compile(sys.argv[3])
          removes  = list(filter(removeRE.match, sourceDB.keys()))
          for source in removes:
            self.logPrint('Removing '+source, 3, 'sourceDB')
            del self.sourceDB[source]
      else:
        sys.exit('Unknown source database action: '+sys.argv[2])
      sourceDB.save()
  except Exception as e:
    import traceback
    print(traceback.print_tb(sys.exc_info()[2]))
    sys.exit(str(e))
  sys.exit(0)
