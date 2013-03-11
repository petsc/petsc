import logger
import script

class VersionControl(logger.Logger):
  def getNewFiles(self, files):
    '''Return all the files not currently under version control'''
    return files

  def getEditedFiles(self, files):
    '''Return all the files open for editing'''
    return []

  def getClosedFiles(self, files):
    '''Return all the files not open for editing'''
    return files

  def getChangedFiles(self, files):
    '''Return all the files open for editing with uncommited changes'''
    return []

  def getUnchangedFiles(self, files):
    '''Return all the files open for editing with no uncommited changes'''
    return []

  def edit(self, files):
    '''Open the set of files for editing'''
    self.logPrint('Editing '+str(files), debugSection = 'vc')
    return 0

  def add(self, files):
    '''Add the set of files to version control'''
    self.logPrint('Adding '+str(files), debugSection = 'vc')
    return 0

  def revert(self, files):
    '''Discard any changes to the set of files, and close them'''
    self.logPrint('Reverting '+str(files), debugSection = 'vc')
    return 0

  def commit(self, files):
    '''Commit any changes to the set of files, and close them'''
    self.logPrint('Commiting '+str(files), debugSection = 'vc')
    return 0

  def changeSet(self):
    '''Commit a change set'''
    self.logPrint('Commiting a change set', debugSection = 'vc')
    return 0

  def clone(self, parent, child):
    '''Clone a parent repository into a child. Both arguments are URLs.'''
    self.logPrint('Cloning '+str(parent)+' into '+str(child), debugSection = 'vc')
    return 0

  def push(self, parent = None):
    '''Update the parent repository from this one'''
    if parent is None:
      self.logPrint('Pushing changes to parent', debugSection = 'vc')
    else:
      self.logPrint('Pushing changes to '+str(parent), debugSection = 'vc')
    return 0

  def pull(self, parent = None):
    '''Update this repository from the parent'''
    if parent is None:
      self.logPrint('Pulling changes from parent', debugSection = 'vc')
    else:
      self.logPrint('Pulling changes from '+str(parent), debugSection = 'vc')
    return 0

class BitKeeper(script.Script, VersionControl):
  def __init__(self, clArgs = None, argDB = None):
    script.Script.__init__(self, clArgs, argDB)
    self.bk = 'bk'
    try:
      import cygwinpath

      self.hasCygwinPath = 1
    except ImportError:
      self.hasCygwinPath = 0
    return

  def convertPath(self, f):
    if self.hasCygwinPath:
      return cygwinpath.convertToFullWin32Path(f)
    return f

  def getNewFiles(self, files):
    '''Return all the files not currently under version control'''
    if not len(files):
      return files
    output = self.executeShellCommand(self.bk+' sfiles -ax '+' '.join(map(self.convertPath, files)))[0]
    if not output:
      return []
    return output.split('\n')

  def getEditedFiles(self, files):
    '''Return all the files under version controland open for editing'''
    if not len(files):
      return files
    output = self.executeShellCommand(self.bk+' sfiles -lg '+' '.join(map(self.convertPath, files)))[0]
    if not output:
      return []
    return output.split('\n')

  def getClosedFiles(self, files):
    '''Return all the files under version control and not open for editing'''
    if not len(files):
      return files
    output = self.executeShellCommand(self.bk+' sfiles -g '+' '.join(map(self.convertPath, files)))[0]
    if not output:
      return []
    lockedFiles = self.getEditedFiles(files)
    return filter(lambda f: not f in lockedFiles, output.split('\n'))

  def getChangedFiles(self, files):
    '''Return all the files under version control and open for editing with uncommited changes'''
    if not len(files):
      return files
    output = self.executeShellCommand(self.bk+' sfiles -cg '+' '.join(map(self.convertPath, files)))[0]
    if not output:
      return []
    return output.split('\n')

  def getUnchangedFiles(self, files):
    '''Return all the files under version control and open for editing with no uncommited changes'''
    if not len(files):
      return files
    output = self.executeShellCommand(self.bk+' sfiles -lg '+' '.join(map(self.convertPath, files)))[0]
    if not output:
      return []
    changedFiles = self.getChangedFiles(files)
    return filter(lambda f: not f in changedFiles, output.split('\n'))

  def edit(self, files):
    '''Open the set of files for editing'''
    if not len(files):
      return 0
    VersionControl.edit(self, files)
    self.executeShellCommand(self.bk+' edit '+' '.join(map(self.convertPath, files)))
    return 1

  def add(self, files):
    '''Add the set of files to version control'''
    if not len(files):
      return 0
    VersionControl.add(self, files)
    self.executeShellCommand(self.bk+' add '+' '.join(map(self.convertPath, files)))
    #self.executeShellCommand(self.bk+' co -q '+' '.join(map(self.convertPath, files)))
    return 1

  def revert(self, files):
    '''Discard any changes to the set of files, and close them'''
    if not len(files):
      return 0
    VersionControl.revert(self, files)
    self.executeShellCommand(self.bk+' unedit '+' '.join(map(self.convertPath, files)))
    #self.executeShellCommand(self.bk+' co -q '+' '.join(map(self.convertPath, files)))
    return 1

  def commit(self, files):
    '''Commit any changes to the set of files, and close them'''
    if not len(files):
      return 0
    VersionControl.commit(self, files)
    self.executeShellCommand(self.bk+' citool')
    return 1

  def changeSet(self):
    '''Commit a change set'''
    self.executeShellCommand(self.bk+' citool')
    return 1

  def clone(self, parent, child):
    '''Clone a parent repository into a child. Both arguments are URLs.'''
    VersionControl.clone(self, parent, child)
    self.executeShellCommand(self.bk+' clone '+str(parent)+' '+str(child))
    return 0

  def push(self, parent = None):
    '''Update the parent repository from this one'''
    VersionControl.push(self, parent)
    cmd = self.bk+' push'
    if not parent is None:
      cmd += ' '+str(parent)
    self.executeShellCommand(cmd)
    return 0

  def pull(self, parent = None):
    '''Update this repository from the parent'''
    VersionControl.pull(self, parent)
    cmd = self.bk+' pull'
    if not parent is None:
      cmd += ' '+str(parent)
    self.executeShellCommand(cmd)
    return 0
