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
    argDB.saveFilename = os.path.join(PETSC_DIR, PETSC_ARCH, 'conf', 'RDict.db')
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    if not clArgs is None: self.clArgs = clArgs
    self.copies = []
    return

  def setupHelp(self, help):
    import nargs
    script.Script.setupHelp(self, help)
    help.addArgument('Installer', '-destDir=<path>', nargs.Arg(None, None, 'Destination Directory for install'))
    return


  def setupModules(self):
    self.setCompilers  = self.framework.require('config.setCompilers',         None)
    self.arch          = self.framework.require('PETSc.utilities.arch',        None)
    self.petscdir      = self.framework.require('PETSc.utilities.petscdir',    None)
    self.makesys       = self.framework.require('config.programs',             None)
    self.compilers     = self.framework.require('config.compilers',            None)
    return
  
  def setup(self):
    script.Script.setup(self)
    self.framework = self.loadConfigure()
    self.setupModules()
    return

  def setupDirectories(self):
    self.rootDir    = self.petscdir.dir
    self.destDir    = os.path.abspath(self.argDB['destDir'])
    self.installDir = self.framework.argDB['prefix']
    self.arch       = self.arch.arch
    self.rootIncludeDir    = os.path.join(self.rootDir, 'include')
    self.archIncludeDir    = os.path.join(self.rootDir, self.arch, 'include')
    self.rootConfDir       = os.path.join(self.rootDir, 'conf')
    self.archConfDir       = os.path.join(self.rootDir, self.arch, 'conf')
    self.rootBinDir        = os.path.join(self.rootDir, 'bin')
    self.archBinDir        = os.path.join(self.rootDir, self.arch, 'bin')
    self.archLibDir        = os.path.join(self.rootDir, self.arch, 'lib')
    self.destIncludeDir    = os.path.join(self.destDir, 'include')
    self.destConfDir       = os.path.join(self.destDir, 'conf')
    self.destLibDir        = os.path.join(self.destDir, 'lib')
    self.destBinDir        = os.path.join(self.destDir, 'bin')
    self.installIncludeDir = os.path.join(self.installDir, 'include')
    self.installBinDir     = os.path.join(self.installDir, 'bin')
    self.rootShareDir      = os.path.join(self.rootDir, 'share')
    self.destShareDir      = os.path.join(self.destDir, 'share')

    self.make      = self.makesys.make+' '+self.makesys.flags
    self.ranlib    = self.compilers.RANLIB
    self.libSuffix = self.compilers.AR_LIB_SUFFIX
    return

  def checkPrefix(self):
    if not self.installDir:
      print '********************************************************************'
      print 'PETSc is built without prefix option. So "make install" is not appropriate.'
      print 'If you need a prefix install of PETSc - rerun configure with --prefix option.'
      print '********************************************************************'
      sys.exit(1)
    return

  def checkDestdir(self):
    if os.path.exists(self.destDir):
      if os.path.samefile(self.destDir, self.rootDir):
        print '********************************************************************'
        print 'Incorrect prefix usage. Specified destDir same as current PETSC_DIR'
        print '********************************************************************'
        sys.exit(1)
      if os.path.samefile(self.destDir, os.path.join(self.rootDir,self.arch)):
        print '********************************************************************'
        print 'Incorrect prefix usage. Specified destDir same as current PETSC_DIR/PETSC_ARCH'
        print '********************************************************************'
        sys.exit(1)
      if not os.path.isdir(os.path.realpath(self.destDir)):
        print '********************************************************************'
        print 'Specified destDir', self.destDir, 'is not a directory. Cannot proceed!'
        print '********************************************************************'
        sys.exit(1)
      if not os.access(self.destDir, os.W_OK):
        print '********************************************************************'
        print 'Unable to write to ', self.destDir, 'Perhaps you need to do "sudo make install"'
        print '********************************************************************'
        sys.exit(1)
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
        errors.extend((srcname,dstname,str(err.args[0])))
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


  def fixConfFile(self, src):
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
    newFile = open(src, 'w')
    newFile.write(''.join(lines))
    newFile.close()
    return

  def fixConf(self):
    import shutil
    # copy standard rules and variables file so that we can change them in place
    for file in ['rules', 'variables']:
      shutil.copy2(os.path.join(self.rootConfDir,file),os.path.join(self.archConfDir,file))
    for file in ['rules', 'variables','petscrules', 'petscvariables']:
      self.fixConfFile(os.path.join(self.archConfDir,file))

  def createUninstaller(self):
    uninstallscript = os.path.join(self.archConfDir, 'uninstall.py')
    f = open(uninstallscript, 'w')
    # Could use the Python AST to do this
    f.write('#!'+sys.executable+'\n')
    f.write('import os\n')

    f.write('copies = '+re.sub(self.destDir,self.installDir,repr(self.copies)))
    f.write('''
for src, dst in copies:
  if os.path.exists(dst):
    os.remove(dst)
''')
    f.close()
    os.chmod(uninstallscript,0744)
    return

  def installIncludes(self):
    self.copies.extend(self.copytree(self.rootIncludeDir, self.destIncludeDir))
    self.copies.extend(self.copytree(self.archIncludeDir, self.destIncludeDir))
    return

  def installConf(self):
    self.copies.extend(self.copytree(self.rootConfDir, self.destConfDir))
    self.copies.extend(self.copytree(self.archConfDir, self.destConfDir))

  def installBin(self):
    self.copies.extend(self.copytree(self.rootBinDir, self.destBinDir))
    self.copies.extend(self.copytree(self.archBinDir, self.destBinDir))
    return

  def installShare(self):
    self.copies.extend(self.copytree(self.rootShareDir, self.destShareDir))
    return

  def copyLib(self, src, dst):
    '''Run ranlib on the destination library if it is an archive. Also run install_name_tool on dylib on Mac'''
    # Do not install object files
    if not os.path.splitext(src)[1] == '.o':
      shutil.copy2(src, dst)
    if os.path.splitext(dst)[1] == '.'+self.libSuffix:
      self.executeShellCommand(self.ranlib+' '+dst)
    if os.path.splitext(dst)[1] == '.dylib' and os.path.isfile('/usr/bin/install_name_tool'):
      installName = re.sub(self.destDir, self.installDir, dst)
      self.executeShellCommand('/usr/bin/install_name_tool -id ' + installName + ' ' + dst)
    # preserve the original timestamps - so that the .a vs .so time order is preserved
    shutil.copystat(src,dst)
    return

  def installLib(self):
    self.copies.extend(self.copytree(self.archLibDir, self.destLibDir, copyFunc = self.copyLib))
    return


  def outputDone(self):
    print '''\
====================================
Install complete. It is useable with PETSC_DIR=%s [and no more PETSC_ARCH].
Now to check if the libraries are working do (in current directory):
make PETSC_DIR=%s test
====================================\
''' % (self.installDir,self.installDir)
    return

  def runfix(self):
    self.setup()
    self.setupDirectories()
    self.checkPrefix()
    self.checkDestdir()
    self.createUninstaller()
    self.fixConf()


  def runcopy(self):
    print '*** Installing PETSc at',self.destDir, ' ***'
    if not os.path.exists(self.destDir):
      try:
        os.makedirs(self.destDir)
      except:
        print '********************************************************************'
        print 'Unable to create', self.destDir, 'Perhaps you need to do "sudo make install"'
        print '********************************************************************'
        sys.exit(1)
    self.installIncludes()
    self.installConf()
    self.installBin()
    self.installLib()
    self.installShare()
    self.outputDone()

    return

  def run(self):
    self.runfix()
    self.runcopy()

if __name__ == '__main__':
  Installer(sys.argv[1:]).run()
  # temporary hack - delete log files created by BuildSystem - when 'sudo make install' is invoked
  delfiles=['RDict.db','RDict.log','build.log','default.log','build.log.bkp','default.log.bkp']
  for delfile in delfiles:
    if os.path.exists(delfile) and (os.stat(delfile).st_uid==0):
      os.remove(delfile)
