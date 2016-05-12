from __future__ import generators
import config.base

import os

try:
  from hashlib import md5 as new_md5
except ImportError:
  from md5 import new as new_md5

class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix     = 'PETSC'
    self.substPrefix      = 'PETSC'
    self.arch             = None # The architecture identifier
    self.externalPackagesDir = os.path.abspath('externalpackages')
    # These are gderived by the configure tests
    self.found            = 0
    self.setNames()
    self.include          = []
    self.dinclude         = []   # all includes in this package and all those it depends on
    self.lib              = []
    self.dlib             = []   # all libraries in this package and all those it depends on
    self.directory        = None # path of the package installation point; for example /usr/local or /home/bsmith/mpich-2.0.1
    self.version          = ''
    # These are specified for the package
    self.required         = 0    # 1 means the package is required
    self.lookforbydefault = 0    # 1 means the package is not required, but always look for and use if found
                                 # cannot tell the difference between user requiring it with --with-PACKAGE=1 and
                                 # this flag being one so hope user never requires it. Needs to be fixed in an overhaul of
                                 # args database so it keeps track of what the user set vs what the program set
    self.useddirectly     = 1    # 1 indicates used by PETSc directly, 0 indicates used by a package used by PETSc
    self.linkedbypetsc    = 1    # 1 indicates PETSc shared libraries (and PETSc executables) need to link against this library
    self.gitcommit        = None # Git commit to use for downloads (used in preference to tarball downloads)
    self.download         = []   # list of URLs where repository or tarballs may be found
    self.deps             = []   # other packages whose dlib or include we depend on, usually we also use self.framework.require()
    self.odeps            = []   # dependent packages that are optional
    self.defaultLanguage  = 'C'  # The language in which to run tests
    self.liblist          = [[]] # list of libraries we wish to check for (override with your own generateLibList())
    self.extraLib         = []   # additional libraries needed to link
    self.includes         = []   # headers to check for
    self.functions        = []   # functions we wish to check for in the libraries
    self.functionsFortran = 0    # 1 means the above symbol is a Fortran symbol, so name-mangling is done
    self.functionsCxx     = [0, '', ''] # 1 means the above symbol is a C++ symbol, so name-mangling with prototype/call is done
    self.cxx              = 0    # 1 means requires C++
    self.fc               = 0    # 1 means requires fortran
    self.needsMath        = 0    # 1 means requires the system math library
    self.needsCompression = 0    # 1 means requires the system compression library
    self.noMPIUni         = 0    # 1 means requires a real MPI
    self.libdir           = 'lib'     # location of libraries in the package directory tree
    self.altlibdir        = 'lib64'   # alternate location of libraries in the package directory tree
    self.includedir       = 'include' # location of includes in the package directory tree
    self.license          = None # optional license text
    self.excludedDirs     = []   # list of directory names that could be false positives, SuperLU_DIST when looking for SuperLU
    self.downloadonWindows= 0  # 1 means the --download-package works on Microsoft Windows
    self.requirescxx11    = 0
    self.publicInstall    = 1  # Installs the package in the --prefix directory if it was given. Packages that are only used
                               # during the configuration/installation process such as sowing, make etc should be marked as 0
    self.parallelMake     = 1  # 1 indicates the package supports make -j np option

    self.double           = 0   # 1 means requires double precision
    self.complex          = 1   # 0 means cannot use complex
    self.requires32bitint = 0;  # 1 means that the package will not work with 64 bit integers
    self.skippackagewithoptions = 0  # packages like fblaslapack and MPICH do not support --with-package* options so do not print them in help
    self.alternativedownload = [] # Used by, for example mpi.py which does not support --download-mpi but one can use --download-mpich
    self.requirec99flag      = 0 # package must be compiled with C99 flags

    # Outside coupling
    self.defaultInstallDir= os.path.abspath('externalpackages')
    self.installSudo      = '' # if user does not have write access to prefix directory then this is set to sudo

    self.isMPI            = 0 # Is an MPI implementation, needed to check for compiler wrappers
    self.hastests         = 0 # indicates that PETSc make alltests has tests for this package
    self.hastestsdatafiles= 0 # indicates that PETSc make all tests has tests for this package that require DATAFILESPATH to be set
    self.makerulename     = '' # some packages do too many things with the make stage; this allows a package to limit to, for example, just building the libraries
    self.installedpetsc   = 0
    return

  def __str__(self):
    '''Prints the location of the packages includes and libraries'''
    output = ''
    if self.found:
      output = self.name+':\n'
      if self.version: output += '  Version:  '+self.version+'\n'
      if self.include: output += '  Includes: '+self.headers.toStringNoDupes(self.include)+'\n'
      if self.lib:     output += '  Library:  '+self.libraries.toStringNoDupes(self.lib)+'\n'
    return output

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers', self)
    self.compilers       = framework.require('config.compilers', self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.types           = framework.require('config.types', self)
    self.headers         = framework.require('config.headers', self)
    self.libraries       = framework.require('config.libraries', self)
    self.programs        = framework.require('config.programs', self)
    self.sourceControl   = framework.require('config.sourceControl',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.petscdir        = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.petscclone      = framework.require('PETSc.options.petscclone',self.setCompilers)
    # All packages depend on make
    self.make          = framework.require('config.packages.make',self)
    if not self.isMPI and not self.package == 'make':
      # force MPI to be the first package configured since all other packages
      # may depend on its compilers defined here
      self.mpi         = framework.require('config.packages.MPI',self)
    return

  def setupHelp(self,help):
    '''Prints help messages for the package'''
    import nargs
    if not self.skippackagewithoptions:
      help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required+self.lookforbydefault,'Indicate if you wish to test for '+self.name))
      help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
      help.addArgument(self.PACKAGE, '-with-'+self.package+'-pkg-config=<dir>', nargs.Arg(None, None, 'Look for '+self.name+' using pkg-config utility optional directory to look in'))
      help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dirs>',nargs.ArgDirList(None,None,'Indicate the directory of the '+self.name+' include files'))
      help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<libraries: e.g. [/Users/..../lib'+self.package+'.a,...]>',nargs.ArgLibrary(None,None,'Indicate the '+self.name+' libraries'))
    if self.download:
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,filename,url>', nargs.ArgDownload(None, 0, 'Download and install '+self.name))
      help.addArgument(self.PACKAGE, '-download-'+self.package+'-commit=commitid', nargs.ArgString(None, 0, 'The commit id from a git repository to use for the build'+self.name))
      help.addDownload(self.package,self.download)
    return

  def setNames(self):
    '''Setup various package names
    name:         The module name (usually the filename)
    package:      The lowercase name
    PACKAGE:      The uppercase name
    downloadname:     Name for download option (usually name)
    downloaddirname: name for downloaded directory (first part of string) (usually downloadname)
    '''
    import sys
    if hasattr(sys.modules.get(self.__module__), '__file__'):
      self.name       = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    else:
      self.name           = 'DEBUGGING'
    self.PACKAGE          = self.name.upper()
    self.package          = self.name.lower()
    self.downloadname     = self.name
    self.pkgname          = self.name
    self.downloaddirname  = self.downloadname;
    return

  def getDefaultPrecision(self):
    '''The precision of the library'''
    if hasattr(self, 'precisionProvider'):
      if hasattr(self.precisionProvider, 'precision'):
        return self.precisionProvider.precision
    return self._defaultPrecision
  def setDefaultPrecision(self, defaultPrecision):
    '''The precision of the library'''
    self._defaultPrecision = defaultPrecision
    return
  defaultPrecision = property(getDefaultPrecision, setDefaultPrecision, doc = 'The precision of the library')

  def getDefaultScalarType(self):
    '''The scalar type for the library'''
    if hasattr(self, 'precisionProvider'):
      if hasattr(self.precisionProvider, 'scalartype'):
        return self.precisionProvider.scalartype
    return self._defaultScalarType
  def setDefaultScalarType(self, defaultScalarType):
    '''The scalar type for the library'''
    self._defaultScalarType = defaultScalarType
    return
  defaultScalarType = property(getDefaultScalarType, setDefaultScalarType, doc = 'The scalar type for of the library')

  def getDefaultIndexSize(self):
    '''The index size for the library'''
    if hasattr(self, 'indexProvider'):
      if hasattr(self.indexProvider, 'integerSize'):
        return self.indexProvider.integerSize
    return self._defaultIndexSize
  def setDefaultIndexSize(self, defaultIndexSize):
    '''The index size for the library'''
    self._defaultIndexSize = defaultIndexSize
    return
  defaultIndexSize = property(getDefaultIndexSize, setDefaultIndexSize, doc = 'The index size for of the library')

  def checkNoOptFlag(self):
    flag = '-O0'
    if self.setCompilers.checkCompilerFlag(flag): return flag
    return ''

  def getSharedFlag(self,cflags):
    for flag in ['-PIC', '-fPIC', '-KPIC', '-qpic']:
      if cflags.find(flag) >=0: return flag
    return ''

  def getPointerSizeFlag(self,cflags):
    for flag in ['-m32', '-m64', '-xarch=v9','-q64','-mmic']:
      if cflags.find(flag) >=0: return flag
    return ''

  def getWindowsNonOptFlags(self,cflags):
    outflags = []
    for flag in cflags.split():
      if flag in ['-MT','-MTd','-MD','-MDd','-threads']:
        outflags.append(flag)
    return ' '.join(outflags)

  def removeWarningFlags(self,flags):
    outflags = []
    for flag in flags.split():
      if not flag in ['-Wall','-Wwrite-strings','-Wno-strict-aliasing','-Wno-unknown-pragmas','-Wno-unused-variable','-Wno-unused-dummy-argument','-fvisibility=hidden']:
        outflags.append(flag)
    return ' '.join(outflags)

  def getDefaultLanguage(self):
    '''The language in which to run tests'''
    if hasattr(self, 'forceLanguage'):
      return self.forceLanguage
    if hasattr(self, 'languageProvider'):
      if hasattr(self.languageProvider, 'defaultLanguage'):
        return self.languageProvider.defaultLanguage
      elif hasattr(self.languageProvider, 'clanguage'):
        return self.languageProvider.clanguage
    return self._defaultLanguage
  def setDefaultLanguage(self, defaultLanguage):
    '''The language in which to run tests'''
    if hasattr(self, 'languageProvider'):
      del self.languageProvider
    self._defaultLanguage = defaultLanguage
    return
  defaultLanguage = property(getDefaultLanguage, setDefaultLanguage, doc = 'The language in which to run tests')

  def getArch(self):
    '''The architecture identifier'''
    if hasattr(self, 'archProvider'):
      if hasattr(self.archProvider, 'arch'):
        return self.archProvider.arch
    return self._arch
  def setArch(self, arch):
    '''The architecture identifier'''
    self._arch = arch
    return
  arch = property(getArch, setArch, doc = 'The architecture identifier')

  def getDefaultInstallDir(self):
    '''The installation directroy of the library'''
    if hasattr(self, 'installDirProvider'):
      if hasattr(self.installDirProvider, 'dir'):
        return self.installDirProvider.dir
    elif not self.framework.externalPackagesDir is None:
      return self.framework.externalPackagesDir
    return self._defaultInstallDir
  def setDefaultInstallDir(self, defaultInstallDir):
    '''The installation directroy of the library'''
    self._defaultInstallDir = defaultInstallDir
    return
  defaultInstallDir = property(getDefaultInstallDir, setDefaultInstallDir, doc = 'The installation directory of the library')

  def getExternalPackagesDir(self):
    '''The directory for downloaded packages'''
    if hasattr(self, 'externalPackagesDirProvider'):
      if hasattr(self.externalPackagesDirProvider, 'dir'):
        return self.externalPackagesDirProvider.dir
    elif not self.framework.externalPackagesDir is None:
      return os.path.abspath('externalpackages')
    return self._externalPackagesDir
  def setExternalPackagesDir(self, externalPackagesDir):
    '''The directory for downloaded packages'''
    self._externalPackagesDir = externalPackagesDir
    return
  externalPackagesDir = property(getExternalPackagesDir, setExternalPackagesDir, doc = 'The directory for downloaded packages')

  def getSearchDirectories(self):
    '''By default, do not search any particular directories, but try compiler default paths'''
    return ['']

  def getInstallDir(self):
    self.confDir    = self.installDirProvider.confDir  # private install location; $PETSC_DIR/$PETSC_ARCH for PETSc
    self.packageDir = self.getDir()
    if not self.packageDir: self.packageDir = self.downLoad()
    self.updateGitDir()
    self.updatehgDir()
    if self.publicInstall:
      self.installDir = self.defaultInstallDir
      self.installSudo= self.installDirProvider.installSudo
    else:
      self.installDir = self.confDir
      self.installSudo= ''
    self.includeDir = os.path.join(self.installDir, 'include')
    self.libDir     = os.path.join(self.installDir, 'lib')
    installDir = self.Install()
    if not installDir:
      raise RuntimeError(self.package+' forgot to return the install directory from the method Install()\n')
    return os.path.abspath(installDir)

  def getChecksum(self,source, chunkSize = 1024*1024):
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if isinstance(source, file):
      f = source
    else:
      f = file(source)
    m = new_md5()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

  def generateLibList(self, directory):
    '''Generates full path list of libraries from self.liblist'''
    if [] in self.liblist: self.liblist.remove([]) # process null list later
    if self.liblist == []: # most packages don't have a liblist - so return an empty list
      return [[]]
    alllibs = []
    if not directory:  # compiler default path - so also check compiler default libs.
      alllibs.insert(0,[])
    for libSet in self.liblist:
      libs = []
      # add full path only to the first library in the list
      if len(libSet) > 0:
        libs.append(os.path.join(directory, libSet[0]))
      for library in libSet[1:]:
        # if the library name doesn't start with lib - then add the fullpath
        if library.startswith('-l') or library.startswith('lib'):
          libs.append(library)
        else:
          libs.append(os.path.join(directory, library))
      libs.extend(self.extraLib)
      alllibs.append(libs)
    return alllibs

  def getIncludeDirs(self, prefix, includeDir):
    if isinstance(includeDir, list):
      return [inc for inc in includeDir if os.path.isabs(inc)] + [os.path.join(prefix, inc) for inc in includeDir if not os.path.isabs(inc)]
    return os.path.join(prefix, includeDir)

  def generateGuesses(self):
    d = self.checkDownload()
    if d:
      for libdir in [self.libdir, self.altlibdir]:
        libdirpath = os.path.join(d, libdir)
        if not os.path.isdir(libdirpath):
          self.logPrint(self.PACKAGE+': Downloaded DirPath not found.. skipping: '+libdirpath)
          continue
        for l in self.generateLibList(libdirpath):
          yield('Download '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+d+'\n')

    if 'with-'+self.package+'-pkg-config' in self.argDB:
      if self.argDB['with-'+self.package+'-pkg-config']:
        #  user provided path to look for pkg info
        if 'PKG_CONFIG_PATH' in os.environ: path = os.environ['PKG_CONFIG_PATH']
        else: path = None
        os.environ['PKG_CONFIG_PATH'] = self.argDB['with-'+self.package+'-pkg-config']

      l,err,ret  = config.base.Configure.executeShellCommand('pkg-config '+self.pkgname+' --libs', timeout=5, log = self.log)
      l = l.strip()
      i,err,ret  = config.base.Configure.executeShellCommand('pkg-config '+self.pkgname+' --variable=includedir', timeout=5, log = self.log)
      i = i.strip()
      if self.argDB['with-'+self.package+'-pkg-config']:
        if path: os.environ['PKG_CONFIG_PATH'] = path
        else: os.environ['PKG_CONFIG_PATH'] = ''
      yield('pkg-config located libraries and includes '+self.PACKAGE, None, l, i)
      raise RuntimeError('pkg-config could not locate correct includes and libraries for '+self.package)


    if 'with-'+self.package+'-dir' in self.argDB:
      d = self.argDB['with-'+self.package+'-dir']
      # error if package-dir is in externalpackages
      if os.path.realpath(d).find(os.path.realpath(self.externalPackagesDir)) >=0:
        fakeExternalPackagesDir = d.replace(os.path.realpath(d).replace(os.path.realpath(self.externalPackagesDir),''),'')
        raise RuntimeError('Bad option: '+'--with-'+self.package+'-dir='+self.argDB['with-'+self.package+'-dir']+'\n'+
                           fakeExternalPackagesDir+' is reserved for --download-package scratch space. \n'+
                           'Do not install software in this location nor use software in this directory.')

      for libdir in [self.libdir, self.altlibdir]:
        libdirpath = os.path.join(d, libdir)
        if not os.path.isdir(libdirpath):
          self.logPrint(self.PACKAGE+': UserSpecified DirPath not found.. skipping: '+libdirpath)
          continue
        for l in self.generateLibList(libdirpath):
          yield('User specified root directory '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))

      if 'with-'+self.package+'-include' in self.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-include if you set --with-'+self.package+'-dir')
      if 'with-'+self.package+'-lib' in self.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-lib if you set --with-'+self.package+'-dir')
      raise RuntimeError('--with-'+self.package+'-dir='+self.argDB['with-'+self.package+'-dir']+' did not work')

    if 'with-'+self.package+'-include' in self.argDB and not 'with-'+self.package+'-lib' in self.argDB:
      if self.liblist[0]:
        raise RuntimeError('If you provide --with-'+self.package+'-include you must also supply with-'+self.package+'-lib\n')
    if 'with-'+self.package+'-lib' in self.argDB and not 'with-'+self.package+'-include' in self.argDB:
      if self.includes:
        raise RuntimeError('If you provide --with-'+self.package+'-lib you must also supply with-'+self.package+'-include\n')
    if 'with-'+self.package+'-include-dir' in self.argDB:
        raise RuntimeError('Use --with-'+self.package+'-include; not --with-'+self.package+'-include-dir')

    if 'with-'+self.package+'-include' in self.argDB or 'with-'+self.package+'-lib' in self.argDB:
      if self.liblist[0]:
        libs  = self.argDB['with-'+self.package+'-lib']
        slibs = str(self.argDB['with-'+self.package+'-lib'])
      else:
        libs  = []
        slibs = 'NoneNeeded'
      inc  = []
      d  = None
      if self.includes:
        inc = self.argDB['with-'+self.package+'-include']
        # hope that package root is one level above first include directory specified
        if inc:
          d   = os.path.dirname(inc[0])

      if not isinstance(inc, list): inc = inc.split(' ')
      if not isinstance(libs, list): libs = libs.split(' ')
      inc = [os.path.abspath(i) for i in inc]
      yield('User specified '+self.PACKAGE+' libraries', d, libs, inc)
      msg = '--with-'+self.package+'-lib='+slibs
      if self.includes:
        msg += ' and \n'+'--with-'+self.package+'-include='+str(self.argDB['with-'+self.package+'-include'])
      msg += ' did not work'
      raise RuntimeError(msg)

    for d in self.getSearchDirectories():
      if d:
        if not os.path.isdir(d):
          self.logPrint(self.PACKAGE+': SearchDir DirPath not found.. skipping: '+d)
          continue
        includedir = self.getIncludeDirs(d, self.includedir)
        for libdir in [self.libdir, self.altlibdir]:
          libdirpath = os.path.join(d, libdir)
          if not os.path.isdir(libdirpath):
            self.logPrint(self.PACKAGE+': DirPath not found.. skipping: '+libdirpath)
            continue
          for l in self.generateLibList(libdirpath):
            yield('Package specific search directory '+self.PACKAGE, d, l, includedir)
      else:
        includedir = ''
        for l in self.generateLibList(d): # d = '' i.e search compiler libraries
            yield('Compiler specific search '+self.PACKAGE, d, l, includedir)

    if not self.lookforbydefault or (self.framework.clArgDB.has_key('with-'+self.package) and self.argDB['with-'+self.package]):
      mesg = 'Unable to find '+self.package+' in default locations!\nPerhaps you can specify with --with-'+self.package+'-dir=<directory>\nIf you do not want '+self.name+', then give --with-'+self.package+'=0'
      if self.download: mesg +='\nYou might also consider using --download-'+self.package+' instead'
      if self.alternativedownload: mesg +='\nYou might also consider using --download-'+self.alternativedownload+' instead'
      raise RuntimeError(mesg)

  def checkDownload(self):
    '''Check if we should download the package, returning the install directory or the empty string indicating installation'''
    if not self.download:
      return ''
    if self.framework.batchBodies:
      return
    if self.argDB['download-'+self.downloadname.lower()]:
      if self.license and not os.path.isfile('.'+self.package+'_license'):
        self.logClear()
        self.logPrint("**************************************************************************************************", debugSection='screen')
        self.logPrint('Please register to use '+self.downloadname+' at '+self.license, debugSection='screen')
        self.logPrint("**************************************************************************************************\n", debugSection='screen')
        fd = open('.'+self.package+'_license','w')
        fd.close()
      return self.getInstallDir()
    return ''

  def installNeeded(self, mkfile):
    makefile       = os.path.join(self.packageDir, mkfile)
    makefileSaved  = os.path.join(self.confDir, 'lib','petsc','conf','pkg.conf.'+self.package)
    gcommfile      = os.path.join(self.packageDir, 'pkg.gitcommit')
    gcommfileSaved = os.path.join(self.confDir,'lib','petsc','conf', 'pkg.gitcommit.'+self.package)
    if not os.path.isfile(makefileSaved) or not (self.getChecksum(makefileSaved) == self.getChecksum(makefile)):
      self.log.write('Have to rebuild '+self.PACKAGE+', '+makefile+' != '+makefileSaved+'\n')
      return 1
    if os.path.isfile(gcommfile) and (not os.path.isfile(gcommfileSaved) or not (self.getChecksum(gcommfileSaved) == self.getChecksum(gcommfile))):
      self.log.write('Have to rebuild '+self.PACKAGE+', '+gcommfile+' != '+gcommfileSaved+'\n')
      return 1
    self.log.write('Do not need to rebuild '+self.PACKAGE+'\n')
    return 0

  def postInstall(self, output, mkfile):
    '''Dump package build log into configure.log - also copy package config to prevent unnecessary rebuild'''
    self.log.write('********Output of running make on '+self.PACKAGE+' follows *******\n')
    self.log.write(output)
    self.log.write('********End of Output of running make on '+self.PACKAGE+' *******\n')
    subconfDir = os.path.join(self.confDir, 'lib', 'petsc', 'conf')
    if not os.path.isdir(subconfDir):
      os.makedirs(subconfDir)
    makefile       = os.path.join(self.packageDir, mkfile)
    makefileSaved  = os.path.join(subconfDir, 'pkg.conf.'+self.package)
    gcommfile      = os.path.join(self.packageDir, 'pkg.gitcommit')
    gcommfileSaved = os.path.join(subconfDir, 'pkg.gitcommit.'+self.package)
    import shutil
    shutil.copyfile(makefile,makefileSaved)
    if os.path.exists(gcommfile):
      shutil.copyfile(gcommfile,gcommfileSaved)
    self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed '+self.PACKAGE+' into '+self.installDir)

  def matchExcludeDir(self,dir):
    '''Check is the dir matches something in the excluded directory list'''
    for exdir in self.excludedDirs:
      if dir.startswith(exdir):
        return 1
    return 0

  def gitPreReqCheck(self):
    '''Some packages may need addition prerequisites if the package comes from a git repository'''
    return 1

  def updatehgDir(self):
    '''Checkout the correct hash'''
    if hasattr(self.sourceControl, 'hg') and (self.packageDir == os.path.join(self.externalPackagesDir,'hg.'+self.package)):
      if hasattr(self,'hghash'):
        config.base.Configure.executeShellCommand([self.sourceControl.hg, 'update', '-c', self.hghash], cwd=self.packageDir, log = self.log)

  def updateGitDir(self):
    '''Checkout the correct gitcommit for the gitdir - and update pkg.gitcommit'''
    if hasattr(self.sourceControl, 'git') and (self.packageDir == os.path.join(self.externalPackagesDir,'git.'+self.package)):
      # verify that packageDir is actually a git clone
      if not os.path.isdir(os.path.join(self.packageDir,'.git')):
        raise RuntimeError(self.packageDir +': is not a git repository! '+os.path.join(self.packageDir,'.git')+' not found!')
      gitdir,err,ret = config.base.Configure.executeShellCommand([self.sourceControl.git, 'rev-parse','--git-dir'], cwd=self.packageDir, log = self.log)
      if gitdir != '.git':
        raise RuntimeError(self.packageDir +': is not a git repository! "git rev-parse --gitdir" gives: '+gitdir)

      prefetch = 0
      if self.gitcommit.startswith('origin/'):
        prefetch = 1
      else:
        try:
          config.base.Configure.executeShellCommand([self.sourceControl.git, 'cat-file', '-e', self.gitcommit+'^{commit}'], cwd=self.packageDir, log = self.log)
        except:
          prefetch = 1
      if prefetch:
        try:
          config.base.Configure.executeShellCommand([self.sourceControl.git, 'fetch'], cwd=self.packageDir, log = self.log)
        except:
          raise RuntimeError('Unable to fetch '+self.gitcommit+' in repository '+self.packageDir+
                             '.\nTo use previous git snapshot - use: --download-'+self.package+'gitcommit=HEAD')
      try:
        gitcommit_hash,err,ret = config.base.Configure.executeShellCommand([self.sourceControl.git, 'rev-parse', self.gitcommit], cwd=self.packageDir, log = self.log)
        if self.gitcommit != 'HEAD':
          config.base.Configure.executeShellCommand([self.sourceControl.git, 'stash'], cwd=self.packageDir, log = self.log)
          config.base.Configure.executeShellCommand([self.sourceControl.git, 'checkout', '-f', gitcommit_hash], cwd=self.packageDir, log = self.log)
          config.base.Configure.executeShellCommand([self.sourceControl.git, 'clean', '-f', '-d', '-x'], cwd=self.packageDir, log = self.log)
      except:
        raise RuntimeError('Unable to checkout commit: '+self.gitcommit+' in repository: '+self.packageDir+
                           '.\nPerhaps its a remote branch, if so - use: origin/'+self.gitcommit)
      # write a commit-tag file
      fd = open(os.path.join(self.packageDir,'pkg.gitcommit'),'w')
      fd.write(gitcommit_hash)
      fd.close()
    return

  def getDir(self):
    '''Find the directory containing the package'''
    packages = self.externalPackagesDir
    if not os.path.isdir(packages):
      os.makedirs(packages)
      self.framework.actions.addArgument('Framework', 'Directory creation', 'Created the external packages directory: '+packages)
    Dir = None
    pkgdirs = os.listdir(packages)
    gitpkg  = 'git.'+self.package
    hgpkg  = 'hg.'+self.package
    self.logPrint('Looking for '+self.PACKAGE+' at '+gitpkg+ ', '+hgpkg+' or a directory starting with '+str(self.downloaddirname))
    if hasattr(self.sourceControl, 'git') and gitpkg in pkgdirs:
      Dir = gitpkg
    elif hasattr(self.sourceControl, 'hg') and hgpkg in pkgdirs:
      Dir = hgpkg
    else:
      for d in pkgdirs:
        if d.startswith(self.downloaddirname) and os.path.isdir(os.path.join(packages, d)) and not self.matchExcludeDir(d):
          Dir = d
          break
    if Dir:
      self.logPrint('Found a copy of '+self.PACKAGE+' in '+str(Dir))
      return os.path.join(packages, Dir)
    else:
      self.logPrint('Could not locate an existing copy of '+self.PACKAGE+':')
      self.logPrint('  '+str(pkgdirs))
      return

  def downLoad(self):
    '''Downloads a package; using hg or ftp; opens it in the with-external-packages-dir directory'''
    import retrieval

    retriever = retrieval.Retriever(self.sourceControl, argDB = self.argDB)
    retriever.setup()
    retriever.saveLog()
    self.logPrint('Downloading '+self.name)
    # check if its http://ftp.mcs - and add ftp://ftp.mcs as fallback
    download_urls = []
    git_urls      = []
    for url in self.download:
      if url.startswith("git://"):
        git_urls.append(url)
      else:
        download_urls.append(url)
      if url.find('http://ftp.mcs.anl.gov') >=0:
        download_urls.append(url.replace('http://','ftp://'))
      # prefer giturl from a petsc gitclone, and tarball urls from a petsc tarball.
      if git_urls:
        if not hasattr(self.sourceControl, 'git'):
          self.logPrint('Git not found - skipping giturls: '+str(git_urls)+'\n')
        elif (self.petscclone.isClone) or self.framework.clArgDB.has_key('with-git'):
          download_urls = git_urls+download_urls
        else:
          download_urls = download_urls+git_urls
    # now attempt to download each url until any one succeeds.
    err =''
    for url in download_urls:
      if url.startswith('git://'):
        if not self.gitcommit: raise RuntimeError(self.PACKAGE+': giturl specified but gitcommit not set')
        if not self.gitPreReqCheck():
          err += 'Git prerequisite check failed for url: '+url+'\n'
          self.logPrint('Git prerequisite check failed - required for url: '+url+'\n')
          continue
      self.logPrintBox('Trying to download '+url+' for '+self.PACKAGE)
      try:
        retriever.genericRetrieve(url, self.externalPackagesDir, self.package)
        self.logWrite(retriever.restoreLog())
        retriever.saveLog()
        pkgdir = self.getDir()
        if not pkgdir:
          raise RuntimeError('Failed to download '+self.PACKAGE)
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.PACKAGE+' into '+pkgdir)
        retriever.restoreLog()
        return pkgdir
      except RuntimeError, e:
        self.logPrint('ERROR: '+str(e))
        err += str(e)
    self.logWrite(retriever.restoreLog())
    raise RuntimeError('Unable to download '+self.PACKAGE+'\n'+err)

  def Install(self):
    raise RuntimeError('No custom installation implemented for package '+self.package+'\n')

  def checkInclude(self, incl, hfiles, otherIncludes = [], timeout = 600.0):
    if self.cxx:
      self.headers.pushLanguage('C++')
    else:
      self.headers.pushLanguage(self.defaultLanguage)
    self.headers.saveLog()
    ret = self.executeTest(self.headers.checkInclude, [incl, hfiles],{'otherIncludes' : otherIncludes, 'timeout': timeout})
    self.logWrite(self.headers.restoreLog())
    self.headers.popLanguage()
    return ret

  def checkPackageLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = 0):
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    result = self.checkLink(includes, body, cleanup, codeBegin, codeEnd, shared)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return result

  def checkDependencies(self, libs = None, incls = None):
    for package in self.deps:
      if not hasattr(package, 'found'):
        raise RuntimeError('Package '+package.name+' does not have found attribute!')
      if not package.found:
        if self.argDB['with-'+package.package] == 1:
          raise RuntimeError('Package '+package.PACKAGE+' needed by '+self.name+' failed to configure.\nMail configure.log to petsc-maint@mcs.anl.gov.')
        else:
          str = ''
          if package.download: str = ' or --download-'+package.package
          raise RuntimeError('Did not find package '+package.PACKAGE+' needed by '+self.name+'.\nEnable the package using --with-'+package.package+str)
    for package in self.deps + self.odeps:
      if hasattr(package, 'dlib')    and not libs  is None: libs  += package.dlib
      if hasattr(package, 'include') and not incls is None: incls += package.include
    return

  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.log.write('==================================================================================\n')
    self.logPrint('Checking for a functional '+self.name)
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs  = []
    incls = []
    self.checkDependencies(libs, incls)
    if self.needsMath:
      if self.libraries.math is None:
        raise RuntimeError('Math library [libm.a or equivalent] is not found')
      libs += self.libraries.math
    if self.needsCompression:
      if self.libraries.compression is None:
        raise RuntimeError('Compression library [libz.a or equivalent] not found')
      libs += self.libraries.compression

    for location, directory, lib, incl in self.generateGuesses():
      if directory and not os.path.isdir(directory):
        self.logPrint('Directory does not exist: %s (while checking "%s" for "%r")' % (directory,location,lib))
        continue
      if lib == '': lib = []
      elif not isinstance(lib, list): lib = [lib]
      if incl == '': incl = []
      elif not isinstance(incl, list): incl = [incl]
      testedincl = list(incl)
      # weed out duplicates when adding fincs
      for loc in self.compilers.fincs:
        if not loc in incl:
          incl.append(loc)
      if self.functions:
        self.logPrint('Checking for library in '+location+': '+str(lib))
        if directory: self.logPrint('Contents: '+str(os.listdir(directory)))
      else:
        self.logPrint('Not checking for library in '+location+': '+str(lib)+' because no functions given to check for')
      self.libraries.saveLog()
      if self.executeTest(self.libraries.check,[lib, self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran, 'cxxMangle' : self.functionsCxx[0], 'prototype' : self.functionsCxx[1], 'call' : self.functionsCxx[2], 'cxxLink': self.cxx}):
        self.lib = lib
        self.logWrite(self.libraries.restoreLog())
        self.logPrint('Checking for headers '+location+': '+str(incl))
        if (not self.includes) or self.checkInclude(incl, self.includes, incls, timeout = 1800.0):
          if self.includes:
            self.include = testedincl
          self.found     = 1
          self.dlib      = self.lib+libs
          self.dinclude  = list(set(incl+incls))
          if not hasattr(self.framework, 'packages'):
            self.framework.packages = []
          self.directory = directory
          self.framework.packages.append(self)
          return
      else:
        self.logWrite(self.libraries.restoreLog())
    if not self.lookforbydefault or (self.framework.clArgDB.has_key('with-'+self.package) and self.argDB['with-'+self.package]):
      raise RuntimeError('Could not find a functional '+self.name+'\n')

  def checkSharedLibrary(self):
    '''By default we don\'t care about checking if the library is shared'''
    return 1

  def alternateConfigureLibrary(self):
    '''Called if --with-packagename=0; does nothing by default'''
    pass

  def consistencyChecks(self):
    if self.skippackagewithoptions: return
    if 'with-'+self.package+'-dir' in self.argDB and ('with-'+self.package+'-include' in self.argDB or 'with-'+self.package+'-lib' in self.argDB):
      raise RuntimeError('Specify either "--with-'+self.package+'-dir" or "--with-'+self.package+'-lib --with-'+self.package+'-include". But not both!')
    if self.argDB['with-'+self.package]:
      if self.cxx and not hasattr(self.compilers, 'CXX'):
        raise RuntimeError('Cannot use '+self.name+' without C++, make sure you do NOT have --with-cxx=0')
      if self.fc and not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot use '+self.name+' without Fortran, make sure you do NOT have --with-fc=0')
      if self.noMPIUni and self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.requirescxx11 and self.compilers.cxxdialect != 'C++11':
        raise RuntimeError('Cannot use '+self.name+' without enabling C++11, see --with-cxx-dialect=C++11')
      if self.download and self.argDB.get('download-'+self.downloadname.lower()) and not self.downloadonWindows and (self.setCompilers.CC.find('win32fe') >= 0):
        raise RuntimeError('External package '+self.name+' does not support --download-'+self.downloadname.lower()+' with Microsoft compilers')
      if self.double and not self.defaultPrecision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')
      if not self.complex and self.defaultScalarType.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')
      if self.defaultIndexSize == 64 and self.requires32bitint:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')
    if not self.download and self.argDB.has_key('download-'+self.downloadname.lower()) and self.argDB['download-'+self.downloadname.lower()]:
      raise RuntimeError('External package '+self.name+' does not support --download-'+self.downloadname.lower())
    return

  def configure(self):
    if self.download and self.argDB['download-'+self.downloadname.lower()] and not self.framework.batchBodies:
      self.argDB['with-'+self.package] = 1
      downloadPackageVal = self.argDB['download-'+self.downloadname.lower()]
      if isinstance(downloadPackageVal, str):
        self.download = [downloadPackageVal]
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-commit']:
      self.gitcommit = self.argDB['download-'+self.downloadname.lower()+'-commit']
    if not 'with-'+self.package in self.argDB:
      self.argDB['with-'+self.package] = 0
    if 'with-'+self.package+'-dir' in self.argDB or 'with-'+self.package+'-include' in self.argDB or 'with-'+self.package+'-lib' in self.argDB:
      self.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-pkg-config' in self.argDB:
      self.argDB['with-'+self.package] = 1

    self.consistencyChecks()
    if self.argDB['with-'+self.package]:
      # If clanguage is c++, test external packages with the c++ compiler
      self.libraries.pushLanguage(self.defaultLanguage)
      self.executeTest(self.configureLibrary)
      self.executeTest(self.checkSharedLibrary)
      self.libraries.popLanguage()
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return

  def updateCompilers(self, installDir, mpiccName, mpicxxName, mpif77Name, mpif90Name):
    '''Check if mpicc, mpicxx etc binaries exist - and update setCompilers() database.
    The input arguments are the names of the binaries specified by the respective pacakges
    This should really be part of compilers.py but it also uses compilerFlags.configure() so
    I am putting it here and Matt can fix it'''

    # Both MPICH and MVAPICH now depend on LD_LIBRARY_PATH for sharedlibraries.
    # So using LD_LIBRARY_PATH in configure - and -Wl,-rpath in makefiles
    if self.argDB['with-shared-libraries']:
      config.setCompilers.Configure.addLdPath(os.path.join(installDir,'lib'))
    # Initialize to empty
    mpicc=''
    mpicxx=''
    mpifc=''

    mpicc = os.path.join(installDir,"bin",mpiccName)
    if not os.path.isfile(mpicc): raise RuntimeError('Could not locate installed MPI compiler: '+mpicc)
    if hasattr(self.compilers, 'CXX'):
      mpicxx = os.path.join(installDir,"bin",mpicxxName)
      if not os.path.isfile(mpicxx): raise RuntimeError('Could not locate installed MPI compiler: '+mpicxx)
    if hasattr(self.compilers, 'FC'):
      if self.compilers.fortranIsF90:
        mpifc = os.path.join(installDir,"bin",mpif90Name)
      else:
        mpifc = os.path.join(installDir,"bin",mpif77Name)
      if not os.path.isfile(mpifc): raise RuntimeError('Could not locate installed MPI compiler: '+mpifc)
    # redo compiler detection
    self.setCompilers.updateMPICompilers(mpicc,mpicxx,mpifc)
    self.compilers.__init__(self.framework)
    self.compilers.headerPrefix = self.headerPrefix
    self.compilers.saveLog()
    self.compilers.configure()
    self.logWrite(self.compilers.restoreLog())
    self.compilerFlags.saveLog()
    self.compilerFlags.configure()
    self.logWrite(self.compilerFlags.restoreLog())
    return

  def rmArgs(self,args,rejects):
    self.logPrint('Removing configure arguments '+str(rejects))
    return [arg for arg in args if not arg in rejects]

  def rmArgsStartsWith(self,args,rejectstarts):
    rejects = []
    for i in rejectstarts:
      rejects.extend([arg for arg in args if arg.startswith(i)])
    return self.rmArgs(args,rejects)

  def checkSharedLibrariesEnabled(self):
    if self.framework.clArgDB.has_key('download-'+self.package+'-shared') and self.argDB['download-'+self.package+'-shared']:
      raise RuntimeError(self.package+' cannot use download-'+self.package+'-shared=1. This flag can only be used to disable '+self.package+' shared libraries')
    if not self.sharedLibraries.useShared or (self.framework.clArgDB.has_key('download-'+self.package+'-shared') and not self.argDB['download-'+self.package+'-shared']):
      return False
    else:
      return True

  def compilePETSc(self):
    try:
      self.logPrintBox('Compiling PETSc; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand('cd '+self.petscdir.dir+' && '+self.make.make+' all PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch,timeout=1000, log = self.log)
      self.log.write(output+err)
    except RuntimeError, e:
      raise RuntimeError('Error running make all on PETSc: '+str(e))
    if self.framework.argDB['prefix']:
      try:
        self.logPrintBox('Installing PETSc; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.petscdir.dir+' && '+self.installDirProvider.installSudo+self.make.make+' install PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch,timeout=50, log = self.log)
        self.log.write(output+err)
      except RuntimeError, e:
        raise RuntimeError('Error running make install on PETSc: '+str(e))
    elif not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing PETSc; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.petscdir.dir+' && '+self.make.make+' test PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch,timeout=50, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('error') > -1 or output.find('Error') > -1:
          raise RuntimeError('Error running make test on PETSc: '+output)
      except RuntimeError, e:
        raise RuntimeError('Error running make test on PETSc: '+str(e))
    self.installedpetsc = 1


'''
config.package.GNUPackage is a helper class whose intent is to simplify writing configure modules
for GNU-style packages that are installed using the "configure; make; make install" idiom.

Brief overview of how BuildSystem\'s configuration of packages works.
---------------------------------------------------------------------
    Configuration is carried out by "configure objects": instances of classes desendant from config.base.Configure.
  These configure objects implement the "configure()" method, and are inserted into a "framework" object,
  which makes the "configure()" calls according to the dependencies between the configure objects.
    config.package.Package extends config.base.Configure and adds instance variables and methods that facilitate
  writing classes that configure packages.  Customized package configuration classes are written by subclassing
  config.package.Package -- the "parent class".

    Packages essentially encapsulate libraries, that either
    (A) are already (prefix-)installed already somewhere on the system or
    (B) need to be dowloaded, built and installed first
  If (A), the parent class provides a generic mechanism for locating the installation, by looking in user-specified and standard locations.
  If (B), the parent class provides a generic mechanism for determining whether a download is necessary, downloading and unpacking
  the source (if the download is, indeed, required), determining whether the package needs to be built, providing the build and
  installation directories, and a few other helper tasks.  The package subclass is responsible for implementing the "Install" hook,
  which is called by the parent class when the actual installation (building the source code, etc.) is done.  As an aside, BuildSystem-
  controled build and install of a package at configuration time has a much better chance of guaranteeing language, compiler and library
  (shared or not) consistency among packages.
    No matter whether (A) or (B) is realized, the parent class control flow demands that the located or installed package
  be checked to ensure it is functional.  Since a package is conceptualized as a library, the check consists in testing whether
  a specified set of libraries can be linked against, and ahat the specified headers can be located.  The libraries and headers are specified
  by name, and the corresponding paths are supplied as a result of the process of locating or building the library.  The verified paths and
  library names are then are stored by the configure object as instance variables.  These can be used by other packages dependent on the package
  being configured; likewise, the package being configured will use the information from the packages it depends on by examining their instance
  variables.

    Thus, the parent class provides for the overall control and data flow, which goes through several configuration stages:
  "init", "setup", "location/installation", "testing".  At each stage, various "hooks" -- methods -- are called.
  Some hooks (e.g., Install) are left unimplemented by the parent class and must be implemented by the package subclass;
  other hooks are implemented by the parent class and provide generic functionality that is likely to suit most packages,
  but can be overridden for custom purposes.  Each hook typically prepares the state -- instance variables -- of the configure object
  for the next phase of configuration.  Below we describe the stages, some of the more typically-used hooks and instance variables in some
  detail.


  init:
  ----
  The init stage constructs the configure object; it is implemented by its __init__ method.
  Parent package class sets up the following useful state variables:
    self.name             - derived from module name                      [string]
    self.package          - lowercase name                                [string]
    self.PACKAGE          - uppercase name                                [string]
    self.downloadname     - same as self.name (usage a bit inconsistent)  [string]
    self.downloaddirname - same as self.name (usage a bit inconsistent)  [string]
  Package subclass typically sets up the following state variables:
    self.download         - url to download source from                   [string]
    self.includes         - names of header files to locate               [list of strings]
    self.liblist          - names of library files to locate              [list of lists of strings]
    self.functions        - names of functions to locate in libraries     [list of strings]
    self.cxx              - whether C++ compiler, (this does not require that PETSc be built with C++, should it?) is required for this package      [bool]
    self.functionsFortran - whether to mangle self.functions symbols      [bool]
  Most of these instance variables determine the behavior of the location/installation and the testing stages.
  Ideally, a package subclass would extend only the __init__ method and parameterize the remainder of
  the configure process by the appropriate variables.  This is not always possible, since some
  of the package-specific choices depend on


  setup:
  -----
  The setup stage follows init and is accomplished by the configure framework calling each configure objects
  setup hooks:

    setupHelp:
    ---------
    This is used to define the command-line arguments expected by this configure object.
    The parent package class sets up generic arguments:
      --with-<package>         [bool]
      --with-<package>-dir     [string: directory]
      --download-<package>     [string:"yes","no","filename"]
      --with-<package>-include [string: directory]
      --with-<package>-lib     [string: directory]
    Here <package> is self.package defined in the init stage.
    The package subclass can add to these arguments.  These arguments\' values are set
    from the defaults specified in setupHelp or from the user-supplied command-line arguments.
    Their values can be queried at any time during the configure process.

    setupDependencies:
    -----------------
    This is used to specify other conifigure objects that the package being configured depends on.
    This is done via the configure framework\'s "require" mechanism:
      self.framework.require(<dependentObject>, self)
    dependentObject is a string -- the name of the configure module this package depends on.

    The parent package class by default sets up some of the common dependencies:
      config.compilers, config.types, config.headers, config.libraries, config.packages.MPI,
    among others.
    The package subclass should add package-specific dependencies via the "require" mechanism,
    as well as list them in self.deps [list].  This list is used during the location/installation
    stage to ensure that the package\'s dependencies have been configured correctly.

  Location/installation:
  ---------------------
  These stages (somewhat mutually-exclusive), as well as the testing stage are carried out by the code in
  configureLibrary.  These stages calls back to certain hooks that allow the user to control the
  location/installation process by overriding these hooks in the package subclass.

  Location:
  --------
  [Not much to say here, yet.]

  Installation:
  ------------
  This stage is carried out by configure and functions called from it, most notably, configureLibrary
  The essential difficulty here is that the function calls are deeply nested (A-->B-->C--> ...),
  as opposed to a single driver repeatedly calling small single-purpose callback hooks.  This means that any
  customization would not be able to be self-contained by would need to know to call further down the chain.
  Moreover, the individual functions on the call stack combine generic code with the code that is naturally meant
  for customization by a package subclass.  Thus, a customization would have to reproduce this generic code.
  Some of the potentially customizable functionality is split between different parts of the code below
  configure (see, e.g., the comment at the end of this paragraph).
    Because of this, there are few opportunities for customization in the installation stage, without a substantial
  restructuring of configure, configureLibrary and/or its callees. Here we mention the main customizable callback
  Install along with two generic services, installNeeded and postInstall, which are provided by the parent class and
  can be used in implementing a custom Install.
    Comment: Note that configure decides whether to configure the package, in part, based on whether
             self.download is a non-empty list at the beginning of configure.
             This means that resetting self.download cannot take place later than this.
             On the other hand, constructing the correct self.download here might be premature, as it might result
             in unnecessary prompts for user input, only to discover later that a download is not required.
             Because of this a package configure class must always have at least dummy string for self.download, if
             a download is possible.

  Here is a schematic description of the main point on the call chain:

  configure:
    check whether to configure the package:
    package is configured only if
      self.download is not an empty string list and the command-line download flag is on
      OR if
      the command-line flag "-with-"self.package is present, prompting a search for the package on the system
      OR if
      the command-line flag(s) pointing to a package installation "-with-"self.package+"-dir or ...-lib, ...-include are present
    ...
    configureLibrary:
      consistencyChecks:
        ...
        check that appropriate language support is on:
          self.cxx            == 1 implies C++ compiler must be present
          self.fc             == 1 implies Fortran compiler must be present
          self.noMPIUni       == 1 implies real MPI must be present
      ...
      generateGuesses:
        ...
        checkDownload:
          ...
          check val = argDB[\'download-\'self.downloadname.tolower()\']
          /*
           note the inconsistency with setupHelp: it declares \'download-\'self.package
           Thus, in order for the correct variable to be queried here, we have to have
           self.downloadname.tolower() == self.package
          */
          if val is a string, set self.download = [val]
          check the package license
          getInstallDir:
            ...
            set the following instance variables, creating directories, if necessary:
            self.installDir   /* This is where the package will be installed, after it is built. */
            self.includeDir   /* subdir of self.installDir */
            self.libDir       /* subdir of self.installDir */
            self.confDir      /* where packages private to the configure/build process are built, such as --download-make */
                              /* The subdirectory of this 'conf' is where where the configuration information will be stored for the package */
            self.packageDir = /* this dir is where the source is unpacked and built */
            self.getDir():
              ...
              if a package dir starting with self.downloadname does not exist already
                create the package dir
                downLoad():
                  ...
                  download and unpack the source to self.packageDir,
          Install():
            /* This must be implemented by a package subclass */

    Install:
    ------
    Note that it follows from the above pseudocode, that the package source is already in self.packageDir
    and the dir instance variables (e.g., installDir, confDir) already point to existing directories.
    The user can implement whatever actions are necessary to configure, build and install
    the package.  Typically, the package is built using GNU\'s "configure; make; make install"
    idiom, so the customized Install forms GNU configure arguments using the compilers,
    system libraries and dependent packages (their locations, libs and includes) discovered
    by configure up to this point.

    It is important to check whether the package source in self.packageDir needs rebuilding, since it might
    have been downloaded in a previous configure run, as is checked by getDir() above.
    However, the package might now need to be built with different options.  For that reason,
    the parent class provides a helper method
      installNeeded(self, mkfile):
        This method compares two files: the file with name mkfile in self.packageDir and
        the file with name self.name in self.confDir (a subdir of the installation dir).
        If the former is absent or differs from the latter, this means the source has never
        been built or was built with different arguments, and needs to be rebuilt.
        This helper method should be run at the beginning of an Install implementation,
        to determine whether an install is actually needed.
    The other useful helper method provided by the parent class is
       postInstall(self, output,mkfile):
         This method will simply save string output in the file with name mkfile in self.confDir.
         Storing package configuration parameters there will enable installNeeded to do its job
         next time this package is being configured.

  testing:
  -------
  The testing is carried out by part of the code in config.package.configureLibrary,
  after the package library has been located or installed.
  The library is considered functional if two conditions are satisfied:
   (1) all of the symbols in self.functions have been resolved when linking against the libraries in self.liblist,
       either located on the system or newly installed;
   (2) the headers in self.includes have been located.
  If no symbols are supplied in self.functions, no link OR header testing is done.



  Extending package class:
  -----------------------
  Generally, extending the parent package configure class is done by overriding some
  or all of its methods (see config/BuildSystem/config/packages/hdf5.py, for example).
  Because convenient (i.e., localized) hooks are available onto to some parts of the
  configure process, frequently writing a custom configure class amounts to overriding
  configureLibrary so that pre- and post-code can be inserted before calling to
  config.package.Package.configureLibrary.

  In any event, Install must be implemented anew for any package configure class extending
  config.package.Package.  Naturally, instance variables have to be set appropriately
  in __init__ (or elsewhere), package-specific help options and dependencies must be defined.
  Therefore, the general pattern for package configure subclassing is this:
    - override __init__ and set package-specific instance variables
    - override setupHelp and setupDependencies hooks to set package-specific command-line
      arguments and dependencies on demand
    - override Install, making use of the parent class\'s installNeeded and postInstall
    - override configureLibrary, if necessary, to insert pre- and post-configure fixup code.

  GNUPackage class:
  ----------------
  This class is an attempt at making writing package configure classes easier for the packages
  that use the "configure; make; make install" idiom for the installation -- "GNU packages".
  The main contribution is in the implementation of a generic Install method, which attempts
  to automate the building of a package based on the mostly standard instance variables.


  Besides running GNU configure, GNUPackage.Install runs installNeeded, make and postInstall
  at the appropriate times, automatically determining whether a rebuild is necessary, saving
  a GNU configure arguments stamp to perform the check in the future, etc.

  setupHelp:
  ---------
  This method extends config.Package.setupHelp by adding two command-line arguments:
    "-download-"+self.package+"-version" with self.downloadversion as default or None, if it does not exist
    "-download-"+self.package+"-shared" with False as the default.

  Summary:
  -------
  In order to customize GNUPackage:
    - set up the usual instance variables in __init__, plus the following instance variables, if necessary/appropriate:
        self.downloadpath
        self.downloadext
        self.downloadversion
    - override setupHelp to declare command-line arguments that can be used anywhere below
      (GNUPackage takes care of some of the basic args, including the download version)
    - override setupDependencies to process self.odeps and enable this optional package feature in the current externalpackage.
    - override setupDownload to control the precise download URL and/or
    - override setupDownloadVersion to control the self.downloadversion string inserted into self.download between self.downloadpath and self.downloadext
'''

class GNUPackage(Package):
  def __init__(self, framework):
    Package.__init__(self,framework)
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-shared=<bool>',     nargs.ArgBool(None, 0, 'Install '+self.PACKAGE+' with shared libraries'))
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-configure-arguments=string', nargs.ArgString(None, 0, 'Additional GNU autoconf configure arguments for the build'+self.name))

  def formGNUConfigureArgs(self):
    '''This sets up the prefix, compiler flags, shared flags, and other generic arguments
       that are fed into the configure script supplied with the package.
       Override this to set options needed by a particular package'''
    args=[]
    ## prefix
    args.append('--prefix='+self.installDir)
    args.append('MAKE='+self.make.make)
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    ## compiler args
    self.pushLanguage('C')
    compiler = self.getCompiler()
    args.append('CC="'+self.getCompiler()+'"')
    args.append('CFLAGS="'+self.removeWarningFlags(self.getCompilerFlags())+'"')
    args.append('AR="'+self.setCompilers.AR+'"')
    args.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')
    if self.setCompilers.LDFLAGS:
      args.append('LDFLAGS="'+self.setCompilers.LDFLAGS+'"')
    self.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('CXX="'+self.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.removeWarningFlags(self.getCompilerFlags())+'"')
      self.popLanguage()
    else:
      args.append('--disable-cxx')
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      fc = self.getCompiler()
      if self.compilers.fortranIsF90:
        try:
          output, error, status = self.executeShellCommand(fc+' -v', log = self.log)
          output += error
        except:
          output = ''
        if output.find('IBM') >= 0:
          fc = os.path.join(os.path.dirname(fc), 'xlf')
          self.log.write('Using IBM f90 compiler, switching to xlf for compiling ' + self.PACKAGE + '\n')
        # now set F90
        args.append('F90="'+fc+'"')
        args.append('F90FLAGS="'+self.removeWarningFlags(self.getCompilerFlags()).replace('-Mfree','')+'"')
      else:
        args.append('--disable-f90')
      args.append('F77="'+fc+'"')
      args.append('FFLAGS="'+self.removeWarningFlags(self.getCompilerFlags()).replace('-Mfree','')+'"')
      args.append('FC="'+fc+'"')
      args.append('FCFLAGS="'+self.removeWarningFlags(self.getCompilerFlags()).replace('-Mfree','')+'"')
      self.popLanguage()
    else:
      args.append('--disable-fortran')
      args.append('--disable-fc')
      args.append('--disable-f77')
      args.append('--disable-f90')
    if self.checkSharedLibrariesEnabled():
      args.append('--enable-shared')
    else:
      args.append('--disable-shared')

    if self.download and self.argDB['download-'+self.downloadname.lower()+'-configure-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-configure-arguments'])
    return args

  def Install(self):
    # hypre had configure inside src directory ugh
    if not os.path.isfile(os.path.join(self.packageDir,'configure')) and not os.path.isfile(os.path.join(self.packageDir,'src','configure')):
      if not self.programs.autoreconf:
        raise RuntimeError('autoreconf required for ' + self.PACKAGE+' not found (or broken)!')
      if not self.programs.libtoolize:
        raise RuntimeError('libtoolize required for ' + self.PACKAGE+' not found!')
      try:
        self.logPrintBox('Running autoreconf on ' +self.PACKAGE+'; this may take several minutes')
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.programs.libtoolize+' && '+self.programs.autoreconf + ' --force --install', timeout=200, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running autoreconf on ' + self.PACKAGE+': '+str(e))

    ##### getInstallDir calls this, and it sets up self.packageDir (source download), self.confDir and self.installDir
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir
    ### Configure and Build package
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2000, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      if self.parallelMake: pmake = self.make.make_jnp+' '+self.makerulename+' '
      else: pmake = self.make.make+' '+self.makerulename+' '

      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' clean', timeout=200, log = self.log)
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+pmake, timeout=6000, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(self.installSudo)
      output4,err4,ret4  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+self.make.make+' install', timeout=1000, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3+output4+err4, conffile)
    return self.installDir

  def checkDependencies(self, libs = None, incls = None):
    Package.checkDependencies(self, libs, incls)
    for package in self.odeps:
      if not package.found:
        if self.argDB['with-'+package.package] == 1:
          raise RuntimeError('Package '+package.PACKAGE+' needed by '+self.name+' failed to configure.\nMail configure.log to petsc-maint@mcs.anl.gov.')
    return

class CMakePackage(Package):
  def __init__(self, framework):
    Package.__init__(self, framework)
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-shared=<bool>',     nargs.ArgBool(None, 0, 'Install '+self.PACKAGE+' with shared libraries'))
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-cmake-arguments=string', nargs.ArgString(None, 0, 'Additional CMake arguments for the build'+self.name))
  def setupDependencies(self, framework):
    Package.setupDependencies(self, framework)
    self.cmake = framework.require('config.packages.cmake',self)
    return

  def formCMakeConfigureArgs(self):
    import os
    import shlex

    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')
    self.framework.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_AR='+self.setCompilers.AR)
    ranlib = shlex.split(self.setCompilers.RANLIB)[0]
    args.append('-DCMAKE_RANLIB='+ranlib)
    cflags = self.removeWarningFlags(self.setCompilers.getCompilerFlags())
    if self.requirec99flag:
      if (self.compilers.c99flag == None):
        raise RuntimeError('Requires c99 compiler. Configure cold not determine compatilbe compiler flag. Perhaps you can specify via CFLAG')
      cflags += ' '+self.compilers.c99flag
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('-DCMAKE_CXX_COMPILER="'+self.framework.getCompiler()+'"')
      args.append('-DCMAKE_CXX_FLAGS:STRING="'+self.removeWarningFlags(self.framework.getCompilerFlags())+'"')
      self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      args.append('-DCMAKE_Fortran_FLAGS:STRING="'+self.removeWarningFlags(self.framework.getCompilerFlags())+'"')
      self.framework.popLanguage()

    if self.setCompilers.LDFLAGS:
      args.append('-DCMAKE_EXE_LINKER_FLAGS:STRING="'+self.setCompilers.LDFLAGS+'"')

    if self.checkSharedLibrariesEnabled():
      args.append('-DBUILD_SHARED_LIBS=on')
    else:
      args.append('-DBUILD_SHARED_LIBS=off')
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments'])
    return args

  def Install(self):
    import os
    args = self.formCMakeConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded(conffile):

      if not self.cmake.found:
        raise RuntimeError('CMake not found, needed to build '+self.PACKAGE+'. Rerun configure with --download-cmake.')

      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, 'build')
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
      os.mkdir(folder)

      try:
        self.logPrintBox('Configuring '+self.PACKAGE+' with cmake, this may take several minutes')
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+folder+' && '+self.cmake.cmake+' .. '+args, timeout=900, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error configuring '+self.PACKAGE+' with cmake '+str(e))
      try:
        self.logPrintBox('Compiling and installing '+self.PACKAGE+'; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+folder+' && '+self.make.make_jnp+' '+self.makerulename+' && '+self.installSudo+' '+self.make.make+' install', timeout=3000, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on  '+self.PACKAGE+': '+str(e))
      self.postInstall(output1+err1+output2+err2,conffile)
    return self.installDir
