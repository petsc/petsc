#!/usr/bin/env python
import re, os, sys, shutil

if os.environ.has_key('PETSC_DIR'):
  PETSC_DIR = os.environ['PETSC_DIR']
else:
  fd = file(os.path.join('conf','petscvariables'))
  a = fd.readline()
  a = fd.readline()
  PETSC_DIR = a.split('=')[1][0:-1]
  fd.close()

if os.environ.has_key('PETSC_ARCH'):
  PETSC_ARCH = os.environ['PETSC_ARCH']
else:
  fd = file(os.path.join('conf','petscvariables'))
  a = fd.readline()
  PETSC_ARCH = a.split('=')[1][0:-1]
  fd.close()

print '*** using PETSC_DIR='+PETSC_DIR+' PETSC_ARCH='+PETSC_ARCH+' ***'
sys.path.insert(0, os.path.join(PETSC_DIR, 'config'))
sys.path.insert(0, os.path.join(PETSC_DIR, 'config', 'BuildSystem'))

import script

try:
  WindowsError
except NameError:
  WindowsError = None

class Installer(script.Script):
  def __init__(self, clArgs = None):
    import RDict
    argDB = RDict.RDict(None, None, 0, 0, readonly = True)
    if os.environ.has_key('PETSC_DIR'):
      PETSC_DIR = os.environ['PETSC_DIR']
    else:
      fd = file(os.path.join('conf','petscvariables'))
      a = fd.readline()
      a = fd.readline()
      PETSC_DIR = a.split('=')[1][0:-1]
      fd.close()
    argDB.saveFilename = os.path.join(PETSC_DIR, PETSC_ARCH, 'conf', 'RDict.db')
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    self.copies = []
    return

  def setupModules(self):
    self.setCompilers  = self.framework.require('config.setCompilers',         None)
    self.arch          = self.framework.require('PETSc.utilities.arch',        None)
    self.petscdir      = self.framework.require('PETSc.utilities.petscdir',    None)
    self.makesys       = self.framework.require('PETSc.utilities.Make',        None)
    self.compilers     = self.framework.require('config.compilers',            None)
    return
  
  def setup(self):
    script.Script.setup(self)
    self.framework = self.loadConfigure()
    self.setupModules()
    return

  def setupDirectories(self):
    self.rootDir    = self.petscdir.dir
    self.installDir = self.framework.argDB['prefix']
    self.arch       = self.arch.arch
    self.rootIncludeDir    = os.path.join(self.rootDir, 'include')
    self.archIncludeDir    = os.path.join(self.rootDir, self.arch, 'include')
    self.rootConfDir       = os.path.join(self.rootDir, 'conf')
    self.archConfDir       = os.path.join(self.rootDir, self.arch, 'conf')
    self.rootBinDir        = os.path.join(self.rootDir, 'bin')
    self.archBinDir        = os.path.join(self.rootDir, self.arch, 'bin')
    self.archLibDir        = os.path.join(self.rootDir, self.arch, 'lib')
    self.installIncludeDir = os.path.join(self.installDir, 'include')
    self.installConfDir    = os.path.join(self.installDir, 'conf')
    self.installLibDir     = os.path.join(self.installDir, 'lib')
    self.installBinDir     = os.path.join(self.installDir, 'bin')

    self.make      = self.makesys.make+' '+self.makesys.flags
    self.ranlib    = self.compilers.RANLIB
    self.libSuffix = self.compilers.AR_LIB_SUFFIX
    return

  def copytree(self, src, dst, symlinks = False, copyFunc = shutil.copy2):
    """Recursively copy a directory tree using copyFunc, which defaults to shutil.copy2().

    The destination directory must not already exist.
    If exception(s) occur, an shutil.Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.
    """
    copies = []
    names  = os.listdir(src)
    if not os.path.exists(dst):
      os.makedirs(dst)
    elif not os.path.isdir(dst):
      raise shutil.Error, 'Destination is not a directory'
    errors = []
    for name in names:
      srcname = os.path.join(src, name)
      dstname = os.path.join(dst, name)
      try:
        if symlinks and os.path.islink(srcname):
          linkto = os.readlink(srcname)
          os.symlink(linkto, dstname)
        elif os.path.isdir(srcname):
          copies.extend(self.copytree(srcname, dstname, symlinks))
        else:
          copyFunc(srcname, dstname)
          copies.append((srcname, dstname))
        # XXX What about devices, sockets etc.?
      except (IOError, os.error), why:
        errors.append((srcname, dstname, str(why)))
      # catch the Error from the recursive copytree so that we can
      # continue with other files
      except shutil.Error, err:
        errors.extend(err.args[0])
    try:
      shutil.copystat(src, dst)
    except OSError, e:
      if WindowsError is not None and isinstance(e, WindowsError):
        # Copying file access times may fail on Windows
        pass
      else:
        errors.extend((src, dst, str(e)))
    if errors:
      raise shutil.Error, errors
    return copies

  def installIncludes(self):
    self.copies.extend(self.copytree(self.rootIncludeDir, self.installIncludeDir))
    self.copies.extend(self.copytree(self.archIncludeDir, self.installIncludeDir))
    return

  def copyConf(self, src, dst):
    if os.path.isdir(dst):
      dst = os.path.join(dst, os.path.basename(src))
    lines   = []
    oldFile = open(src, 'r')
    for line in oldFile.readlines():
      # paths generated by configure could be different link-path than whats used by user, so fix both
      line = re.sub(re.escape(os.path.join(self.rootDir, self.arch)), self.installDir, line)
      line = re.sub(re.escape(os.path.realpath(os.path.join(self.rootDir, self.arch))), self.installDir, line)
      line = re.sub(re.escape(os.path.join(self.rootDir, 'bin')), self.installBinDir, line)
      line = re.sub(re.escape(os.path.realpath(os.path.join(self.rootDir, 'bin'))), self.installBinDir, line)
      line = re.sub(re.escape(os.path.join(self.rootDir, 'include')), self.installIncludeDir, line)
      line = re.sub(re.escape(os.path.realpath(os.path.join(self.rootDir, 'include'))), self.installIncludeDir, line)
      # remove PETSC_DIR/PETSC_ARCH variables from conf-makefiles. They are no longer necessary
      line = re.sub('\$\{PETSC_DIR\}/\$\{PETSC_ARCH\}', self.installDir, line)
      line = re.sub('PETSC_ARCH=\$\{PETSC_ARCH\}', '', line)
      line = re.sub('\$\{PETSC_DIR\}', self.installDir, line)
      lines.append(line)
    oldFile.close()
    newFile = open(dst, 'w')
    newFile.write(''.join(lines))
    newFile.close()
    shutil.copystat(src, dst)
    return

  def installConf(self):
    # rootConfDir can have a duplicate petscvariables - so processing it first removes the appropriate duplicate file.
    self.copies.extend(self.copytree(self.rootConfDir, self.installConfDir, copyFunc = self.copyConf))
    self.copies.extend(self.copytree(self.archConfDir, self.installConfDir))
    # Just copyConf() a couple of files manually [as the rest of the files should not be modified]
    for file in ['petscrules', 'petscvariables']:
      self.copyConf(os.path.join(self.archConfDir,file),os.path.join(self.installConfDir,file))
    return

  def installBin(self):
    self.copies.extend(self.copytree(self.rootBinDir, self.installBinDir))
    self.copies.extend(self.copytree(self.archBinDir, self.installBinDir))
    return

  def copyLib(self, src, dst):
    '''Run ranlib on the destination library if it is an archive'''
    shutil.copy2(src, dst)
    if os.path.splitext(dst)[1] == '.'+self.libSuffix:
      self.executeShellCommand(self.ranlib+' '+dst)
    return

  def installLib(self):
    self.copies.extend(self.copytree(self.archLibDir, self.installLibDir, copyFunc = self.copyLib))
    return

  def createUninstaller(self):
    uninstallscript = os.path.join(self.installConfDir, 'uninstall.py')
    f = open(uninstallscript, 'w')
    # Could use the Python AST to do this
    f.write('#!'+sys.executable+'\n')
    f.write('import os\n')

    f.write('copies = '+repr(self.copies))
    f.write('''
for src, dst in copies:
  if os.path.exists(dst):
    os.remove(dst)
''')
    f.close()
    os.chmod(uninstallscript,0744)
    return

  def outputHelp(self):
    print '''\
====================================
Install complete. It is useable with PETSC_DIR=%s [and no more PETSC_ARCH].
Now to check if the libraries are working do (in current directory):
make PETSC_DIR=%s test
====================================\
''' % (self.installDir,self.installDir)
    return

  def run(self):
    self.setup()
    self.setupDirectories()
    if os.path.exists(self.installDir) and os.path.samefile(self.installDir, os.path.join(self.rootDir,self.arch)):
      print '********************************************************************'
      print 'Install directory is current directory; nothing needs to be done'
      print '********************************************************************'
      return
    print '*** Installing PETSc at',self.installDir, ' ***'
    if not os.path.exists(self.installDir):
      try:
        os.makedirs(self.installDir)
      except:
        print '********************************************************************'
        print 'Unable to create', self.installDir, 'Perhaps you need to do "sudo make install"'
        print '********************************************************************'
        return
    if not os.path.isdir(os.path.realpath(self.installDir)):
      print '********************************************************************'
      print 'Specified prefix', self.installDir, 'is not a directory. Cannot proceed!'
      print '********************************************************************'
      return
    if not os.access(self.installDir, os.W_OK):
      print '********************************************************************'
      print 'Unable to write to ', self.installDir, 'Perhaps you need to do "sudo make install"'
      print '********************************************************************'
      return
    self.installIncludes()
    self.installConf()
    self.installBin()
    self.installLib()
    output,err,ret = self.executeShellCommand(self.make+' PETSC_ARCH=""'+' PETSC_DIR='+self.installDir+' ARCHFLAGS= shared mpi4py petsc4py')
    print output+err
    # this file will mess up the make test run since it resets PETSC_ARCH when PETSC_ARCH needs to be null now
    os.unlink(os.path.join(self.rootDir,'conf','petscvariables'))
    fd = file(os.path.join('conf','petscvariables'),'w')
    fd.close()
    # if running as root then change file ownership back to user
    if os.environ.has_key('SUDO_USER'):
      os.chown(os.path.join(self.rootDir,'conf','petscvariables'),int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']))
    self.createUninstaller()
    self.outputHelp()
    return

if __name__ == '__main__':
  Installer(sys.argv[1:]).run()
  # temporary hack - delete log files created by BuildSystem - when 'sudo make install' is invoked
  delfiles=['RDict.db','RDict.log','build.log','default.log','build.log.bkp','default.log.bkp']
  for delfile in delfiles:
    if os.path.exists(delfile) and (os.stat(delfile).st_uid==0):
      os.remove(delfile)
