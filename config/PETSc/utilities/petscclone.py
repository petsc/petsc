import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    self.sourceControl = framework.require('config.sourceControl',self)
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
    return

  def configureInstallationMethod(self):
    if os.path.exists(os.path.join(self.petscdir.dir,'bin','maint')):
      self.logPrint('bin/maint exists. This appears to be a repository clone')
      self.isClone = 1
      if os.path.exists(os.path.join(self.petscdir.dir, '.git')):
        if hasattr(self.sourceControl,'git'):
          self.addDefine('VERSION_GIT','"'+os.popen("cd "+self.petscdir.dir+" && "+self.sourceControl.git+" describe").read().strip()+'"')
          self.addDefine('VERSION_DATE_GIT','"'+os.popen("cd "+self.petscdir.dir+" && "+self.sourceControl.git+" log -1 --pretty=format:%ci").read()+'"')
          self.addDefine('VERSION_BRANCH_GIT','"'+re.compile('\* (.*)\n').search(os.popen('cd '+self.petscdir.dir+' && '+self.sourceControl.git+' branch').read()).group(1)+'"')
        else:
          self.logPrintBox('\n*****WARNING: PETSC_DIR appears to be a Git clone - but git is not found in PATH********\n')
      elif os.path.exists(os.path.join(self.petscdir.dir, '.hg')):
        if hasattr(self.sourceControl,'hg'):
          self.addDefine('VERSION_HG','"'+os.popen(self.sourceControl.hg +" -R"+self.petscdir.dir+" tip --template '{node}'").read()+'"')
          self.addDefine('VERSION_DATE_HG','"'+os.popen(self.sourceControl.hg +" -R"+self.petscdir.dir+" tip --template '{date|date}'").read()+'"')
        else:
          self.logPrintBox('\n*****WARNING: PETSC_DIR appears to be a mercurial clone - but hg is not found in PATH********\n')
      else:
        self.logPrint('This repository clone is obtained as a tarball as neither .hg nor .git dirs exist!')
    else:
      if os.path.exists(os.path.join(self.petscdir.dir, '.git')) or os.path.exists(os.path.join(self.petscdir.dir, '.hg')):
        raise RuntimeError('Your petsc source tree is broken. Use "git status" to check, or remove the entire directory and start all over again')
      else:
        self.logPrint('This is a tarball installation')
        self.isClone = 0
    return

  def configure(self):
    self.executeTest(self.configureInstallationMethod)
    return
