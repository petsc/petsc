#!/usr/bin/env python3
from __future__ import print_function
import os, re, shutil, sys
import sysconfig

if 'PETSC_DIR' in os.environ:
  PETSC_DIR = os.environ['PETSC_DIR']
else:
  fd = open(os.path.join('lib','petsc','conf','petscvariables'))
  a = fd.readline()
  a = fd.readline()
  PETSC_DIR = a.split('=')[1][0:-1]
  fd.close()

if 'PETSC_ARCH' in os.environ:
  PETSC_ARCH = os.environ['PETSC_ARCH']
else:
  fd = open(os.path.join('lib','petsc','conf','petscvariables'))
  a = fd.readline()
  PETSC_ARCH = a.split('=')[1][0:-1]
  fd.close()

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
    argDB.saveFilename = os.path.join(PETSC_DIR, PETSC_ARCH, 'lib','petsc','conf', 'RDict.db')
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    if not clArgs is None: self.clArgs = clArgs
    self.copies = []
    return

  def setupHelp(self, help):
    import nargs
    script.Script.setupHelp(self, help)
    help.addArgument('Installer', '-destDir=<path>', nargs.Arg(None, '', 'Destination Directory for install'))
    help.addArgument('Installer', '-no-examples', nargs.Arg(None, '', 'Skip installing examples'))
    return


  def setupModules(self):
    self.setCompilers  = self.framework.require('config.setCompilers',         None)
    self.arch          = self.framework.require('PETSc.options.arch',          None)
    self.petscdir      = self.framework.require('PETSc.options.petscdir',      None)
    self.compilers     = self.framework.require('config.compilers',            None)
    self.mpi           = self.framework.require('config.packages.MPI',         None)
    return

  def setup(self):
    script.Script.setup(self)
    self.framework = self.loadConfigure()
    self.setupModules()
    return

  def setupDirectories(self):
    self.rootDir    = self.petscdir.dir
    self.installDir = os.path.abspath(os.path.expanduser(self.framework.argDB['prefix']))
    self.destDir    = os.path.abspath(self.argDB['destDir']+self.installDir)
    self.arch       = self.arch.arch
    self.archDir           = os.path.join(self.rootDir, self.arch)
    self.rootIncludeDir    = os.path.join(self.rootDir, 'include')
    self.archIncludeDir    = os.path.join(self.rootDir, self.arch, 'include')
    self.rootConfDir       = os.path.join(self.rootDir, 'lib','petsc','conf')
    self.archConfDir       = os.path.join(self.rootDir, self.arch, 'lib','petsc','conf')
    self.rootBinDir        = os.path.join(self.rootDir, 'lib','petsc','bin')
    self.archBinDir        = os.path.join(self.rootDir, self.arch, 'bin')
    self.archLibDir        = os.path.join(self.rootDir, self.arch, 'lib')
    self.destIncludeDir    = os.path.join(self.destDir, 'include')
    self.destConfDir       = os.path.join(self.destDir, 'lib','petsc','conf')
    self.destLibDir        = os.path.join(self.destDir, 'lib')
    self.destBinDir        = os.path.join(self.destDir, 'lib','petsc','bin')
    self.installIncludeDir = os.path.join(self.installDir, 'include')
    self.installLibDir     = os.path.join(self.installDir, 'lib')
    self.installBinDir     = os.path.join(self.installDir, 'lib','petsc','bin')
    self.rootShareDir      = os.path.join(self.rootDir, 'share')
    self.destShareDir      = os.path.join(self.destDir, 'share')
    self.rootSrcDir        = os.path.join(self.rootDir, 'src')

    self.ranlib      = self.compilers.RANLIB
    self.arLibSuffix = self.compilers.AR_LIB_SUFFIX
    return

  def checkPrefix(self):
    if not self.installDir:
      print('********************************************************************')
      print('PETSc is built without prefix option. So "make install" is not appropriate.')
      print('If you need a prefix install of PETSc - rerun configure with --prefix option.')
      print('********************************************************************')
      sys.exit(1)
    return

  def checkDestdir(self):
    if os.path.exists(self.destDir):
      if os.path.samefile(self.destDir, self.rootDir):
        print('********************************************************************')
        print('Incorrect prefix usage. Specified destDir same as current PETSC_DIR')
        print('********************************************************************')
        sys.exit(1)
      if os.path.samefile(self.destDir, os.path.join(self.rootDir,self.arch)):
        print('********************************************************************')
        print('Incorrect prefix usage. Specified destDir same as current PETSC_DIR/PETSC_ARCH')
        print('********************************************************************')
        sys.exit(1)
      if not os.path.isdir(os.path.realpath(self.destDir)):
        print('********************************************************************')
        print('Specified destDir', self.destDir, 'is not a directory. Cannot proceed!')
        print('********************************************************************')
        sys.exit(1)
      if not os.access(self.destDir, os.W_OK):
        print('********************************************************************')
        print('Unable to write to ', self.destDir, 'Perhaps you need to do "sudo make install"')
        print('********************************************************************')
        sys.exit(1)
    return

  def setupBuild(self):
    self.using_build_backend = any(
      os.environ.get(prefix + '_BUILD_BACKEND')
      for prefix in ('_PYPROJECT_HOOKS', 'PEP517')
    )
    self.relocate_py_env = os.environ.get('CIBUILDWHEEL', '0') == '1'

  def copyfile(self, src, dst, symlinks = False, copyFunc = shutil.copy2):
    """Copies a single file    """
    copies = []
    errors = []
    if not os.path.exists(dst):
      os.makedirs(dst)
    elif not os.path.isdir(dst):
      raise shutil.Error('Destination is not a directory')
    srcname = src
    dstname = os.path.join(dst, os.path.basename(src))
    try:
      if symlinks and os.path.islink(srcname):
        linkto = os.readlink(srcname)
        os.symlink(linkto, dstname)
      else:
        copyFunc(srcname, dstname)
        copies.append((srcname, dstname))
    except (IOError, os.error) as why:
      errors.append((srcname, dstname, str(why)))
    except shutil.Error as err:
      errors.append((srcname,dstname,str(err.args[0])))
    if errors:
      raise shutil.Error(errors)
    return copies

  def fixExamplesMakefile(self, src):
    '''Change ././${PETSC_ARCH} in makefile in root PETSc directory with ${PETSC_DIR}'''
    lines   = []
    oldFile = open(src, 'r')
    alllines=oldFile.read()
    oldFile.close()
    newlines=alllines.split('\n')[0]+'\n'  # Firstline
    # Hardcode PETSC_DIR and PETSC_ARCH to avoid users doing the wrong thing
    newlines+='PETSC_DIR='+self.installDir+'\n'
    newlines+='PETSC_ARCH=\n'
    for line in alllines.split('\n')[1:]:
      if line.startswith('TESTLOGFILE'):
        newlines+='TESTLOGFILE = $(TESTDIR)/examples-install.log\n'
      elif line.startswith('CONFIGDIR'):
        newlines+='CONFIGDIR:=$(PETSC_DIR)/$(PETSC_ARCH)/share/petsc/examples/config\n'
      elif line.startswith('$(generatedtest)') and 'petscvariables' in line:
        newlines+='all: test\n\n'+line+'\n'
      else:
        newlines+=line+'\n'
    newFile = open(src, 'w')
    newFile.write(newlines)
    newFile.close()
    return

  def copyConfig(self, src, dst):
    """Copy configuration/testing files
    """
    if not os.path.isdir(dst):
      raise shutil.Error('Destination is not a directory')

    self.copies.extend(self.copyfile('gmakefile.test',dst))
    newConfigDir=os.path.join(dst,'config')  # Am not renaming at present
    if not os.path.isdir(newConfigDir): os.mkdir(newConfigDir)
    testConfFiles="gmakegentest.py gmakegen.py testparse.py example_template.py".split()
    testConfFiles+="petsc_harness.sh report_tests.py query_tests.py".split()
    for tf in testConfFiles:
      self.copies.extend(self.copyfile(os.path.join('config',tf),newConfigDir))
    return

  def copyExamples(self, src, dst):
    """copy the examples directories
    """
    for root, dirs, files in os.walk(src, topdown=False):
      if os.path.basename(root) not in ("tests", "tutorials"): continue
      self.copies.extend(self.copytree(root, root.replace(src,dst)))
    return

  def copytree(self, src, dst, symlinks = False, copyFunc = shutil.copy2, exclude = [], exclude_ext= ['.DSYM','.o','.pyc'], recurse = 1):
    """Recursively copy a directory tree using copyFunc, which defaults to shutil.copy2().

       The copyFunc() you provide is only used on the top level, lower levels always use shutil.copy2

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
      raise shutil.Error('Destination is not a directory')
    errors = []
    srclinks = []
    dstlinks = []
    for name in names:
      srcname = os.path.join(src, name)
      dstname = os.path.join(dst, name)
      try:
        if symlinks and os.path.islink(srcname):
          linkto = os.readlink(srcname)
          os.symlink(linkto, dstname)
        elif os.path.isdir(srcname) and recurse and not os.path.basename(srcname) in exclude:
          copies.extend(self.copytree(srcname, dstname, symlinks,exclude = exclude, exclude_ext = exclude_ext))
        elif os.path.isfile(srcname) and not os.path.basename(srcname) in exclude and os.path.splitext(name)[1] not in exclude_ext :
          if os.path.islink(srcname):
            srclinks.append(srcname)
            dstlinks.append(dstname)
          else:
            copyFunc(srcname, dstname)
            copies.append((srcname, dstname))
        # XXX What about devices, sockets etc.?
      except (IOError, os.error) as why:
        errors.append((srcname, dstname, str(why)))
      # catch the Error from the recursive copytree so that we can
      # continue with other files
      except shutil.Error as err:
        errors.append((srcname,dstname,str(err.args[0])))
    for srcname, dstname in zip(srclinks, dstlinks):
      try:
        copyFunc(srcname, dstname)
        copies.append((srcname, dstname))
      except (IOError, os.error) as why:
        errors.append((srcname, dstname, str(why)))
      # catch the Error from the recursive copytree so that we can
      # continue with other files
      except shutil.Error as err:
        errors.append((srcname,dstname,str(err.args[0])))
    try:
      shutil.copystat(src, dst)
    except OSError as e:
      if WindowsError is not None and isinstance(e, WindowsError):
        # Copying file access times may fail on Windows
        pass
      else:
        errors.append((src, dst, str(e)))
    if errors:
      raise shutil.Error(errors)
    return copies


  def fixConfFile(self, src):
    lines   = []
    oldFile = open(src, 'r')
    for line in oldFile.readlines():
      if line.startswith('PETSC_CC_INCLUDES =') or line.startswith('PETSC_FC_INCLUDES ='):
        continue
      line = line.replace('PETSC_CC_INCLUDES_INSTALL', 'PETSC_CC_INCLUDES')
      line = line.replace('PETSC_FC_INCLUDES_INSTALL', 'PETSC_FC_INCLUDES')
      # remove PETSC_DIR/PETSC_ARCH variables from conf-makefiles. They are no longer necessary
      line = line.replace('${PETSC_DIR}/${PETSC_ARCH}', self.installDir)
      line = line.replace('PETSC_ARCH=${PETSC_ARCH}', '')
      line = line.replace('${PETSC_DIR}', self.installDir)
      # replace PETSC_DIR/PETSC_ARCH/lib (i.e., build location) with prefix/lib
      line = line.replace(self.archLibDir,self.installLibDir)
      # replace PETSC_DIR/lib/petsc/bin with prefix/lib/petsc/bin
      line = line.replace(self.rootBinDir,self.destBinDir)
      lines.append(line)
    oldFile.close()
    newFile = open(src, 'w')
    newFile.write(''.join(lines))
    newFile.close()
    return

  def fixConf(self):
    import shutil
    for file in ['rules', 'rules_doc.mk', 'rules_util.mk', 'variables', 'petscrules', 'petscvariables']:
      self.fixConfFile(os.path.join(self.destConfDir,file))
    return

  def fixPythonWheel(self):
    import glob
    import shutil
    #
    for pattern in (
        self.destLibDir  + '/*.a',
        self.destLibDir  + '/*.la',
        self.destLibDir  + '/pkgconfig',  # TODO: keep?
        self.destConfDir + '/configure-hash',
        self.destConfDir + '/uninstall.py',
        self.destConfDir + '/reconfigure-*.py',
        self.destConfDir + '/pkg.conf.*',
        self.destConfDir + '/pkg.git*.*',
        self.destConfDir + '/modules',  # TODO: keep?
        self.destShareDir + '/*/examples/src/*',
        self.destShareDir + '/*/datafiles',
    ):
      for pathname in glob.glob(pattern):
        if os.path.isdir(pathname):
          shutil.rmtree(pathname)
        elif os.path.exists(pathname):
          os.remove(pathname)
    #
    if self.relocate_py_env:
      pydir = sys.prefix
      pylibdir = os.path.join(pydir, 'lib')
      pysitedir = sysconfig.get_paths()["platlib"]
      # petsc is installed in site-packages
      petscdir = os.path.join(pysitedir, 'petsc')
      petsclibdir = os.path.join(petscdir, 'lib')
    for filename in (
      self.destIncludeDir + '/petscconf.h',
      self.destIncludeDir + '/petscconfiginfo.h',
      self.destIncludeDir + '/petscmachineinfo.h',
      self.destShareDir + '/petsc/examples/gmakefile.test',
      self.destConfDir + '/rules',
      self.destConfDir + '/rules_doc.mk',
      self.destConfDir + '/rules_util.mk',
      self.destConfDir + '/petscrules',
      self.destConfDir + '/variables',
      self.destConfDir + '/petscvariables',
    ):
      with open(filename, 'r') as oldFile:
        contents = oldFile.read()
      contents = contents.replace(self.installDir, '${PETSC_DIR}')
      contents = contents.replace(self.rootDir, '${PETSC_DIR}')
      if self.relocate_py_env:
        pydir_from_petsc = os.path.relpath(pydir, petscdir)
        contents = contents.replace(pydir, os.path.join('${PETSC_DIR}', pydir_from_petsc))
      contents = re.sub(
        r'^(PYTHON(_EXE)?) = (.*)$',
        r'\1 = python%d' % sys.version_info[0],
        contents, flags=re.MULTILINE,
      )
      with open(filename, 'w') as newFile:
        newFile.write(contents)
    #
    def lsdir(dirname, *patterns):
      return glob.glob(os.path.join(dirname, *patterns))
    def shell(*args):
      return self.executeShellCommand(' '.join(args))[0]
    libdir = os.path.join(self.installDir, 'lib')
    if sys.platform == 'linux':
      libraries = [
        lib for lib in lsdir(self.destLibDir, 'lib*.so*')
        if not os.path.islink(lib)
      ]
      for shlib in libraries:
        # fix shared library rpath
        rpath = shell('patchelf', '--print-rpath', shlib)
        rpath = rpath.split(os.path.pathsep)
        if not self.relocate_py_env:
          if libdir in rpath:
            rpath.insert(0, '$ORIGIN')
            while libdir in rpath:
              rpath.remove(libdir)
        else:
          rpathold = rpath
          rpath = []
          # strip all rpath info, except for libraries in Python
          # sys prefix, site-packages, or petsc/lib
          rpath.insert(0, '$ORIGIN')
          for libdir in rpathold:
            if libdir.startswith(pysitedir):
              libdir_from_petsc = os.path.relpath(libdir, petsclibdir)
              rpath.insert(0, os.path.join('$ORIGIN',libdir_from_petsc))
          pylibdir_from_petsc = os.path.relpath(pylibdir, petsclibdir)
          rpath.insert(0, os.path.join('$ORIGIN',pylibdir_from_petsc))
        if rpath:
          rpath = os.path.pathsep.join(rpath)
          shell('patchelf', '--set-rpath', "'%s'" % rpath, shlib)
        # fix shared library file and symlink
        basename = os.path.basename(shlib)
        libname, ext, _ = basename.partition('.so')
        liblink = libname + ext
        soname = shell('patchelf', '--print-soname', shlib)
        for symlink in lsdir(self.destLibDir, liblink + '*'):
          if os.path.islink(symlink):
            os.unlink(symlink)
        curdir = os.getcwd()
        try:
          os.chdir(os.path.dirname(shlib))
          if soname != basename:
            os.rename(basename, soname)
          if soname != liblink:
            with open(liblink, 'w') as f:
              f.write('INPUT(' + soname + ')\n')
        finally:
          os.chdir(curdir)
    if sys.platform == 'darwin':
      def otool(cmd, dylib):
        pattern = r'''
          ^\s+ cmd \s %s$\n
          ^\s+ cmdsize \s \d+$\n
          ^\s+ (?:name|path) \s (.*) \s \(offset \s \d+\)$
        ''' % cmd
        return re.findall(
          pattern, shell('otool', '-l', dylib),
          flags=re.VERBOSE | re.MULTILINE,
        )
      libraries = [
        lib for lib in lsdir(self.destLibDir, 'lib*.dylib')
        if not os.path.islink(lib)
      ]
      for dylib in libraries:
        install_name = otool('LC_ID_DYLIB', dylib)[0]
        dependencies = otool('LC_LOAD_DYLIB', dylib)
        runtime_path = otool('LC_RPATH', dylib)
        # fix shared library install name and rpath
        install_name = '@rpath/' + os.path.basename(install_name)
        shell('install_name_tool', '-id', install_name, dylib)
        if libdir in runtime_path:
          shell('install_name_tool', '-delete_rpath', libdir, dylib)
        for rpath in ('@loader_path',):
          if rpath not in runtime_path:
            shell('install_name_tool', '-add_rpath', rpath, dylib)
        for dep in dependencies:
          if os.path.dirname(dep) in (libdir,):
            newid = '@rpath/' + os.path.basename(dep)
            shell('install_name_tool', '-change', dep, newid, dylib)
        # fix shared library file and symlink
        basename = os.path.basename(dylib)
        libname, ext = os.path.splitext(basename)
        libname = libname.partition('.')[0]
        liblink = libname + ext
        dyname = os.path.basename(install_name)
        for symlink in lsdir(self.destLibDir, libname + '*' + ext):
          if os.path.islink(symlink):
            os.unlink(symlink)
        curdir = os.getcwd()
        try:
          os.chdir(os.path.dirname(dylib))
          if dyname != basename:
            os.rename(basename, dyname)
          if dyname != liblink:
            os.symlink(dyname, liblink)
        finally:
          os.chdir(curdir)
    #
    return

  def createUninstaller(self):
    uninstallscript = os.path.join(self.destConfDir, 'uninstall.py')
    f = open(uninstallscript, 'w')
    # Could use the Python AST to do this
    f.write('#!'+sys.executable+'\n')
    f.write('import os\n')
    f.write('prefixdir = "'+self.installDir+'"\n')
    files = [dst.replace(self.destDir,self.installDir) for src, dst in self.copies]
    files.append(uninstallscript.replace(self.destDir,self.installDir))
    f.write('files = '+repr(files))
    f.write('''
for file in files:
  if os.path.exists(file) or os.path.islink(file):
    os.remove(file)
    dir = os.path.dirname(file)
    while dir not in [os.path.dirname(prefixdir),'/']:
      try: os.rmdir(dir)
      except: break
      dir = os.path.dirname(dir)
''')
    f.close()
    os.chmod(uninstallscript,0o744)
    return

  def installIncludes(self):
    exclude = ['makefile']
    if not hasattr(self.compilers.setCompilers, 'FC'):
      exclude.append('finclude')
    if not self.mpi.usingMPIUni:
      exclude.append('mpiuni')
    self.copies.extend(self.copytree(self.rootIncludeDir, self.destIncludeDir,exclude = exclude))
    self.copies.extend(self.copytree(self.archIncludeDir, self.destIncludeDir))
    return

  def installConf(self):
    self.copies.extend(self.copytree(self.rootConfDir, self.destConfDir, exclude = ['test.log']))
    self.copies.extend(self.copytree(self.archConfDir, self.destConfDir, exclude = ['sowing', 'configure.log.bkp','configure.log','make.log','gmake.log','test.log','error.log','memoryerror.log','files','testfiles','RDict.db']))
    return

  def installBin(self):
    exclude = ['bib2html','doc2lt','doctext','mapnames', 'pstogif','pstoxbm','tohtml']
    self.copies.extend(self.copytree(self.archBinDir, self.destBinDir, exclude = exclude ))
    exclude = ['maint']
    if not self.mpi.usingMPIUni:
      exclude.append('petsc-mpiexec.uni')
    self.setCompilers.pushLanguage('C')
    if self.setCompilers.getCompiler().find('win32fe') < 0:
      exclude.append('win32fe')
    self.setCompilers.popLanguage()
    self.copies.extend(self.copytree(self.rootBinDir, self.destBinDir, exclude = exclude ))
    return

  def installShare(self):
    if self.argDB['no-examples']: exclude = ['datafiles']
    else: exclude = []
    self.copies.extend(self.copytree(self.rootShareDir, self.destShareDir, exclude=exclude))
    examplesdir=os.path.join(self.destShareDir,'petsc','examples')
    if os.path.exists(examplesdir):
      shutil.rmtree(examplesdir)
    os.mkdir(examplesdir)
    os.mkdir(os.path.join(examplesdir,'src'))
    self.copyConfig(self.rootDir,examplesdir)
    if not self.argDB['no-examples']:
        self.copyExamples(self.rootSrcDir,os.path.join(examplesdir,'src'))
        self.fixExamplesMakefile(os.path.join(examplesdir,'gmakefile.test'))
    return

  def copyLib(self, src, dst):
    '''Run ranlib on the destination library if it is an archive. Also run install_name_tool on dylib on Mac'''
    # Symlinks (assumed local) are recreated at dst
    if os.path.islink(src):
      linkto = os.readlink(src)
      try:
        os.remove(dst)            # In case it already exists
      except OSError:
        pass
      os.symlink(linkto, dst)
      return
    shutil.copy2(src, dst)
    if self.setCompilers.getCompiler().find('win32fe') < 0 and os.path.splitext(dst)[1] == '.'+self.arLibSuffix:
      import shlex
      self.executeShellCommand(shlex.split(self.ranlib) + [dst])
    if os.path.splitext(dst)[1] == '.dylib' and shutil.which('otool') and shutil.which('install_name_tool'):
      [output,err,flg] = self.executeShellCommand(['otool', '-D', src])
      oldname = output[output.find("\n")+1:]
      installName = oldname.replace(os.path.realpath(self.archDir), self.installDir)
      self.executeShellCommand(['install_name_tool', '-id', installName, dst])
    # preserve the original timestamps - so that the .a vs .so time order is preserved
    shutil.copystat(src,dst)
    return

  def installLib(self):
    self.copies.extend(self.copytree(self.archLibDir, self.destLibDir, copyFunc = self.copyLib, exclude = ['.DIR'],recurse = 0))
    self.copies.extend(self.copytree(os.path.join(self.archLibDir,'pkgconfig'), os.path.join(self.destLibDir,'pkgconfig'), copyFunc = self.copyLib, exclude = ['.DIR'],recurse = 0))
    return


  def outputInstallDone(self):
    from config.packages.make import getMakeUserPath
    print('''\
====================================
To check if the libraries are working do (in current directory):
%s PETSC_DIR=%s PETSC_ARCH="" check
====================================\
''' % (getMakeUserPath(self.arch), self.installDir))
    return

  def outputDestDirDone(self):
    print('''\
====================================
Copy to DESTDIR %s is now complete.
Before use - please copy/install over to specified prefix: %s
====================================\
''' % (self.destDir,self.installDir))
    return

  def runsetup(self):
    self.setup()
    self.setupDirectories()
    self.setupBuild()
    self.checkPrefix()
    self.checkDestdir()
    return

  def runcopy(self):
    if self.destDir == self.installDir:
      print('*** Installing PETSc at prefix location:',self.destDir, ' ***')
    else:
      print('*** Copying PETSc to DESTDIR location:',self.destDir, ' ***')
    if not os.path.exists(self.destDir):
      try:
        os.makedirs(self.destDir)
      except:
        print('********************************************************************')
        print('Unable to create', self.destDir, 'Perhaps you need to do "sudo make install"')
        print('********************************************************************')
        sys.exit(1)
    self.installIncludes()
    self.installConf()
    self.installBin()
    self.installLib()
    self.installShare()
    self.createUninstaller()
    return

  def runfix(self):
    self.fixConf()
    if self.using_build_backend:
      self.fixPythonWheel()
    return

  def rundone(self):
    if self.destDir == self.installDir:
      self.outputInstallDone()
    else:
      self.outputDestDirDone()
    return

  def run(self):
    self.runsetup()
    self.runcopy()
    self.runfix()
    self.rundone()
    return

if __name__ == '__main__':
  Installer(sys.argv[1:]).run()
  # temporary hack - delete log files created by BuildSystem - when 'sudo make install' is invoked
  delfiles=['RDict.db','RDict.log','buildsystem.log','default.log','buildsystem.log.bkp','default.log.bkp']
  for delfile in delfiles:
    if os.path.exists(delfile) and (os.stat(delfile).st_uid==0):
      os.remove(delfile)
