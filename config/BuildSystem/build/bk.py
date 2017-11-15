import build.fileset
import build.transform

def convertPath(file):
  '''Converts the cygwin path to a full Windows path'''
  try:
    import cygwinpath
    return cygwinpath.convertToFullWin32Path(file)
  except ImportError:
    pass
  return file

class Tag (build.transform.Transform):
  '''Tags all relevant Bitkeeper filesets
     - Unlocked files are tagged "bkedit"
     - Locked files which are unchanged are tagged "bkrevert"
     - New implementation files are tagged "bkadd"'''
  def __init__(self, rootFunc, inputTag = None):
    import re

    build.transform.Transform.__init__(self)
    self.implRE   = re.compile(r'^(.*)_impl$')
    self.rootFunc = rootFunc
    self.inputTag = inputTag
    if not self.inputTag is None and not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    return

  def __str__(self):
    return 'BitKeeper tag transform'

  def getUnlockedFiles(self, root):
    '''Return a list of all files not locked by BitKeeper in the root directories'''
    files       = []
    lockedFiles = []
    files.extend(self.executeShellCommand('bk sfiles -g '+convertPath(root)).split())
    lockedFiles.extend(self.executeShellCommand('bk sfiles -lg '+convertPath(root)).split())
    map(files.remove, lockedFiles)
    return files

  def isImplementationFile(self, filename):
    '''Returns True if filename is an implementation file'''
    import os

    if filename[-1] == '~': return 0
    if filename[-1] == '#': return 0
    if os.path.splitext(filename)[1] == '.pyc': return 0
    if self.implRE.match(os.path.dirname(filename)):
      return 1
    return 0

  def getNewFiles(self, root):
    '''Return a list of all implementation files not under BitKeeper control in the root directories'''
    files = []
    files.extend(filter(self.isImplementationFile, self.executeShellCommand('bk sfiles -ax '+convertPath(root)).split()))
    return files

  def getUnchangedFiles(self, root):
    '''Return a list of the files locked by Bitkeeper, but unchanged'''
    lockedFiles  = []
    changedFiles = []
    lockedFiles.extend(self.executeShellCommand('bk sfiles -lg '+convertPath(root)).split())
    changedFiles.extend(self.executeShellCommand('bk sfiles -cg '+convertPath(root)).split())
    map(lockedFiles.remove, changedFiles)
    return lockedFiles

  def handleFile(self, f, set):
    '''Add new filesets to the output
       - All files under BitKeeper control are tagged "bkedit"
       - All new implementation files are tagged "bkadd"
       - All locked but unchanged files under BitKeeper control are tagged "bkrevert"'''
    root = self.rootFunc(f)
    if (self.inputTag is None or set.tag in self.inputTag) and root:
      import os
      if not os.path.isdir(root):
        os.makedirs(root)
      self.output.children.append(build.fileset.FileSet(filenames = self.getUnlockedFiles(root),  tag = 'bkedit'))
      self.output.children.append(build.fileset.FileSet(filenames = self.getNewFiles(root),       tag = 'bkadd'))
      self.output.children.append(build.fileset.FileSet(filenames = self.getUnchangedFiles(root), tag = 'bkrevert'))
    return build.transform.Transform.handleFile(self, f, set)

class Open (build.transform.Transform):
  '''This nodes handles sets with tag "bkedit", editing each file'''
  def __init__(self):
    build.transform.Transform.__init__(self)
    return

  def __str__(self):
    return 'BitKeeper open transform'

  def edit(self, set):
    '''Edit the files in set with BitKeeper'''
    if not len(set): return
    self.debugPrint('Opening files', 2, 'bk')
    command = 'bk edit '+' '.join(map(convertPath, set))
    output  = self.executeShellCommand(command)
    return self.output

  def handleFileSet(self, set):
    '''Handle sets with tag "bkedit"'''
    if set.tag == 'bkedit':
      self.edit(set)
      map(self.handleFileSet, set.children)
      return self.output
    return build.transform.Transform.handleFileSet(self, set)

class Close (build.transform.Transform):
  '''This nodes handles sets with tag "bkadd" and "bkrevert", adding new files and reverting unchanged files'''
  def __init__(self):
    build.transform.Transform.__init__(self)
    return

  def __str__(self):
    return 'BitKeeper close transform'

  def add(self, set):
    '''Add the files in set to BitKeeper'''
    if not len(set): return
    self.debugPrint('Putting new files under version control', 2, 'bk')
    map(lambda f: self.debugPrint('Adding '+f+' to version control', 3, 'bk'), set)
    command = 'bk add '+' '.join(map(convertPath, set))
    output  = self.executeShellCommand(command)
    command = 'bk co -q '+' '.join(map(convertPath, set))
    output  = self.executeShellCommand(command)
    return self.output

  def revert(self, set):
    '''Revert the files in set using BitKeeper'''
    if not len(set): return
    self.debugPrint('Reverting unchanged files', 2, 'bk')
    command = 'bk unedit '+' '.join(map(convertPath, set))
    output  = self.executeShellCommand(command)
    command = 'bk co -q '+' '.join(map(convertPath, set))
    output  = self.executeShellCommand(command)
    return self.output

  def handleFileSet(self, set):
    '''Handle sets with tag "bkadd" and "bkrevert"'''
    if set.tag == 'bkadd':
      self.add(set)
      map(self.handleFileSet, set.children)
      return self.output
    elif set.tag == 'bkrevert':
      self.revert(set)
      map(self.handleFileSet, set.children)
      return self.output
    return build.transform.Transform.handleFileSet(self, set)
