#!/usr/bin/env python
import os, sys

configDir = os.path.abspath('config')
sys.path.insert(0, configDir)
bsDir     = os.path.abspath(os.path.join(configDir, 'BuildSystem'))
sys.path.insert(0, bsDir)

import script

class Installer(script.Script):
  def __init__(self, clArgs = None):
    import RDict
    script.Script.__init__(self, clArgs, RDict.RDict())
    return

  def setupHelp(self, help):
    import nargs

    script.Script.setupHelp(self, help)
    help.addArgument('Installer', '-rootDir=<path>', nargs.Arg(None, None, 'Install Root Directory'))
    help.addArgument('Installer', '-installDir=<path>', nargs.Arg(None, None, 'Install Target Directory'))
    help.addArgument('Installer', '-arch=<type>', nargs.Arg(None, None, 'Architecture type'))
    help.addArgument('Installer', '-ranlib=<prog>', nargs.Arg(None, 'ranlib', 'Ranlib program'))
    help.addArgument('Installer', '-make=<prog>', nargs.Arg(None, 'make', 'Make program'))
    help.addArgument('Installer', '-libSuffix=<ext>', nargs.Arg(None, 'make', 'The static library suffix'))
    return

  def setupDirectories(self):
    self.rootDir    = os.path.abspath(self.argDB['rootDir'])
    self.installDir = os.path.abspath(self.argDB['installDir'])
    self.arch       = os.path.abspath(self.argDB['arch'])
    self.ranlib     = os.path.abspath(self.argDB['ranlib'])
    self.make       = os.path.abspath(self.argDB['make'])
    self.libSuffix  = os.path.abspath(self.argDB['libSuffix'])
    return

  def run(self):
    import re, shutil

    self.setup()
    self.setupDirectories()
    if self.installDir == self.rootDir:
      print 'Install directory is current directory; nothing needs to be done'
    else:
      print 'Installing PETSc at',self.installDir
      if not os.path.isdir(self.installDir):
        os.makedirs(self.installDir)
      rootIncludeDir    = os.path.join(self.rootDir, 'include')
      archIncludeDir    = os.path.join(self.rootDir, self.arch, 'include')
      rootConfDir       = os.path.join(self.rootDir, 'conf')
      archConfDir       = os.path.join(self.rootDir, self.arch, 'conf')
      rootBinDir        = os.path.join(self.rootDir, 'bin')
      archBinDir        = os.path.join(self.rootDir, self.arch, 'bin')
      archLibDir        = os.path.join(self.rootDir, self.arch, 'lib')
      installIncludeDir = os.path.join(self.installDir, 'include')
      installConfDir    = os.path.join(self.installDir, 'conf')
      installLibDir     = os.path.join(self.installDir, 'lib')
      installBinDir     = os.path.join(self.installDir, 'bin')
      if os.path.exists(installIncludeDir):
        shutil.rmtree(installIncludeDir)
      shutil.copytree(rootIncludeDir, installIncludeDir)
      for f in os.listdir(archIncludeDir):
        shutil.copy(os.path.join(archIncludeDir, f), os.path.join(installIncludeDir, f))
      if os.path.exists(installConfDir):
        shutil.rmtree(installConfDir)
      shutil.copytree(rootConfDir, installConfDir)
      for f in os.listdir(archConfDir):
        shutil.copy(os.path.join(archConfDir, f), os.path.join(installConfDir, f))
      for f in os.listdir(installConfDir):
        oldFile = open(os.path.join(installConfDir, f))
        lines   = []
        for line in oldFile.readlines():
          lines.append(re.sub(self.rootDir, self.installDir, re.sub(os.path.join(self.rootDir, self.arch), self.installDir, line)))
        oldFile.close()
        oldFile = open(os.path.join(installConfDir, f), 'w')
        oldFile.write(''.join(lines))
        oldFile.close()
      if os.path.exists(installBinDir):
        shutil.rmtree(installBinDir)
      shutil.copytree(rootBinDir, installBinDir)
      for f in os.listdir(archBinDir):
        shutil.copy(os.path.join(archBinDir, f), os.path.join(installBinDir, f))
      if os.path.exists(installLibDir):
        shutil.rmtree(installLibDir)
      shutil.copytree(archLibDir, installLibDir)
      for f in os.listdir(installLibDir):
        if os.path.splitext(f)[1] == '.'+self.libSuffix:
          self.executeShellCommand(self.ranlib+' '+os.path.join(installLibDir, f))
      self.executeShellCommand(self.make+' PETSC_ARCH=""'+' PETSC_DIR='+self.installDir+' shared')
      print '''
If using sh/bash, do the following:
  PETSC_DIR=%s; export PETSC_DIR
  unset PETSC_ARCH
If using csh/tcsh, do the following:
  setenv PETSC_DIR %s
  unsetenv PETSC_ARCH
Now run the testsuite to verify the install with the following:
  make test
''' % (self.installDir, self.installDir)
    return

if __name__ == '__main__':
  Installer(sys.argv[1:]).run()
