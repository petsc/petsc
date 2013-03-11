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
    self.download         = []   # urls where repository or tarballs may be found
    self.deps             = []   # other packages whose dlib or include we depend on, usually we also use self.framework.require()
    self.defaultLanguage  = 'C'  # The language in which to run tests
    self.liblist          = [[]] # list of libraries we wish to check for (override with your own generateLibraryList())
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
    self.archIndependent  = 0    # 1 means the install directory does not incorporate the ARCH name
    self.downloadonWindows   = 0  # 1 means the --download-package works on Microsoft Windows
    self.worksonWindows      = 0  # 1 means that package can be used on Microsof Windows
    # Outside coupling
    self.defaultInstallDir= os.path.abspath('externalpackages')
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
    self.setCompilers  = framework.require('config.setCompilers', self)
    self.compilers     = framework.require('config.compilers', self)
    self.types         = framework.require('config.types', self)
    self.headers       = framework.require('config.headers', self)
    self.libraries     = framework.require('config.libraries', self)
    self.programs      = framework.require('config.programs', self)
    self.sourceControl = framework.require('config.sourceControl',self)
    self.mpi           = framework.require('config.packages.MPI',self)

    return

  def setupHelp(self,help):
    '''Prints help messages for the package'''
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required+self.lookforbydefault,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if hasattr(self, 'usePkgConfig'):
      help.addArgument(self.PACKAGE, '-with-'+self.package+'-pkg-config=<dir>', nargs.ArgDir(None, None, 'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dirs>',nargs.ArgDirList(None,None,'Indicate the directory of the '+self.name+' include files'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<libraries: e.g. [/Users/..../lib'+self.package+'.a,...]>',nargs.ArgLibrary(None,None,'Indicate the '+self.name+' libraries'))
    if self.download and not self.download[0] == 'redefine':
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,filename>', nargs.ArgDownload(None, 0, 'Download and install '+self.name))
    return

  def setNames(self):
    '''Setup various package names
    name:         The module name (usually the filename)
    package:      The lowercase name
    PACKAGE:      The uppercase name
    downloadname:     Name for download option (usually name)
    downloadfilename: name for downloaded file (first part of string) (usually downloadname)
    '''
    import sys
    if hasattr(sys.modules.get(self.__module__), '__file__'):
      self.name       = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    else:
      self.name           = 'DEBUGGING'
    self.PACKAGE          = self.name.upper()
    self.package          = self.name.lower()
    self.downloadname     = self.name
    self.downloadfilename = self.downloadname;
    return

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
    if not self.framework.externalPackagesDir is None:
      packages = os.path.abspath('externalpackages')
      return self.framework.externalPackagesDir
    return self._externalPackagesDir
  def setExternalPackagesDir(self, externalPackagesDir):
    '''The directory for downloaded packages'''
    self._externalPackagesDir = externalPackagesDir
    return
  externalPackagesDir = property(getExternalPackagesDir, setExternalPackagesDir, doc = 'The directory for downloaded packages')

  def getSearchDirectories(self):
    '''By default, do not search any particular directories'''
    return []

  def getInstallDir(self):
    self.installDir = os.path.join(self.defaultInstallDir, self.arch)
    self.confDir    = os.path.join(self.installDir, 'conf')
    self.includeDir = os.path.join(self.installDir, 'include')
    self.libDir     = os.path.join(self.installDir, 'lib')
    self.packageDir = self.getDir()
    if not os.path.isdir(self.installDir): os.mkdir(self.installDir)
    if not os.path.isdir(self.libDir):     os.mkdir(self.libDir)
    if not os.path.isdir(self.includeDir): os.mkdir(self.includeDir)
    if not os.path.isdir(self.confDir):    os.mkdir(self.confDir)
    return os.path.abspath(self.Install())

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
    alllibs = []
    for libSet in self.liblist:
      libs = []
      # add full path only to the first library in the list
      if len(libSet) > 0:
        if not self.libdir == directory:
          libs.append(os.path.join(directory, libSet[0]))
        else:
          libs.append(libSet[0])
      for library in libSet[1:]:
        # if the library name doesn't start with lib - then add the fullpath
        if library.startswith('lib') or self.libdir == directory:
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
    d = self.checkDownload(1)
    if d:
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('Download '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      for l in self.generateLibList(os.path.join(d, self.altlibdir)):
        yield('Download '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+d+'\n')

    if 'with-'+self.package+'-dir' in self.framework.argDB:
      d = self.framework.argDB['with-'+self.package+'-dir']
      # error if package-dir is in externalpackages
      if os.path.realpath(d).find(os.path.realpath(self.externalPackagesDir)) >=0:
        fakeExternalPackagesDir = d.replace(os.path.realpath(d).replace(os.path.realpath(self.externalPackagesDir),''),'')
        raise RuntimeError('Bad option: '+'--with-'+self.package+'-dir='+self.framework.argDB['with-'+self.package+'-dir']+'\n'+
                           fakeExternalPackagesDir+' is reserved for --download-package scratch space. \n'+
                           'Do not install software in this location nor use software in this directory.')
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('User specified root directory '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      for l in self.generateLibList(os.path.join(d, self.altlibdir)):
        yield('User specified root directory '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      if 'with-'+self.package+'-include' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-include if you set --with-'+self.package+'-dir')
      if 'with-'+self.package+'-lib' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-lib if you set --with-'+self.package+'-dir')
      raise RuntimeError('--with-'+self.package+'-dir='+self.framework.argDB['with-'+self.package+'-dir']+' did not work')

    if 'with-'+self.package+'-include' in self.framework.argDB and not 'with-'+self.package+'-lib' in self.framework.argDB:
      raise RuntimeError('If you provide --with-'+self.package+'-include you must also supply with-'+self.package+'-lib\n')
    if 'with-'+self.package+'-lib' in self.framework.argDB and not 'with-'+self.package+'-include' in self.framework.argDB:
      if self.includes:
        raise RuntimeError('If you provide --with-'+self.package+'-lib you must also supply with-'+self.package+'-include\n')
    if 'with-'+self.package+'-include-dir' in self.framework.argDB:
        raise RuntimeError('Use --with-'+self.package+'-include; not --with-'+self.package+'-include-dir')

    if 'with-'+self.package+'-include' in self.framework.argDB and 'with-'+self.package+'-lib' in self.framework.argDB:
      inc = self.framework.argDB['with-'+self.package+'-include']
      libs = self.framework.argDB['with-'+self.package+'-lib']
      if not isinstance(inc, list): inc = inc.split(' ')
      if not isinstance(libs, list): libs = libs.split(' ')
      inc = [os.path.abspath(i) for i in inc]
      print inc
      # hope that package root is one level above first include directory specified
      d = os.path.dirname(inc[0])
      yield('User specified '+self.PACKAGE+' libraries', d, libs, inc)
      raise RuntimeError('--with-'+self.package+'-lib='+str(self.framework.argDB['with-'+self.package+'-lib'])+' and \n'+\
                         '--with-'+self.package+'-include='+str(self.framework.argDB['with-'+self.package+'-include'])+' did not work')

    for d in self.getSearchDirectories():
      for libdir in [self.libdir, self.altlibdir]:
        for l in self.generateLibList(os.path.join(d, libdir)):
          if not d:
            includedir = ''
          else:
            includedir = self.getIncludeDirs(d, self.includedir)
          yield('Package specific search directory '+self.PACKAGE, d, l, includedir)

    d = self.checkDownload(requireDownload = 0)
    if d:
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('Download '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      for l in self.generateLibList(os.path.join(d, self.altlibdir)):
        yield('Download '+self.PACKAGE, d, l, self.getIncludeDirs(d, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+self.getInstallDir()+'\n')

    if not self.lookforbydefault:
      raise RuntimeError('You must specify a path for '+self.name+' with --with-'+self.package+'-dir=<directory>\nIf you do not want '+self.name+', then give --with-'+self.package+'=0\nYou might also consider using --download-'+self.package+' instead')

  def checkDownload(self, requireDownload = 1):
    '''Check if we should download the package, returning the install directory or the empty string indicating installation'''
    if not self.download:
      return ''
    downloadPackage = 0
    downloadPackageVal = self.framework.argDB['download-'+self.downloadname.lower()]
    if requireDownload and isinstance(downloadPackageVal, str):
      self.download = [downloadPackageVal]
      downloadPackage = 1
    elif downloadPackageVal == 1 and requireDownload:
      downloadPackage = 1
    elif downloadPackageVal == 2 and not requireDownload:
      downloadPackage = 1

    if downloadPackage:
      if not self.download:
        raise RuntimeError('Package'+self.package+' does not support automatic download.\n')
      if self.license and not os.path.isfile('.'+self.package+'_license'):
        self.framework.logClear()
        self.logPrint("**************************************************************************************************", debugSection='screen')
        self.logPrint('Please register to use '+self.downloadname+' at '+self.license, debugSection='screen')
        self.logPrint("**************************************************************************************************\n", debugSection='screen')
        fd = open('.'+self.package+'_license','w')
        fd.close()
      return self.getInstallDir()
    return ''

  def installNeeded(self, mkfile):
    makefile      = os.path.join(self.packageDir, mkfile)
    makefileSaved = os.path.join(self.confDir, self.name)
    if not os.path.isfile(makefileSaved) or not (self.getChecksum(makefileSaved) == self.getChecksum(makefile)):
      self.framework.log.write('Have to rebuild '+self.name+', '+makefile+' != '+makefileSaved+'\n')
      return 1
    else:
      self.framework.log.write('Do not need to rebuild '+self.name+'\n')
      return 0

  def postInstall(self, output, mkfile):
    '''Dump package build log into configure.log - also copy package config to prevent unnecessary rebuild'''
    self.framework.log.write('********Output of running make on '+self.name+' follows *******\n')
    self.framework.log.write(output)
    self.framework.log.write('********End of Output of running make on '+self.name+' *******\n')
    output,err,ret  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(self.packageDir, mkfile)+' '+os.path.join(self.confDir, self.name), timeout=5, log = self.framework.log)
    self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed '+self.name+' into '+self.installDir)

  def matchExcludeDir(self,dir):
    '''Check is the dir matches something in the excluded directory list'''
    for exdir in self.excludedDirs:
      if dir.startswith(exdir):
        return 1
    return 0

  def getDir(self, retry = 1):
    '''Find the directory containing the package'''
    packages = self.externalPackagesDir
    if not os.path.isdir(packages):
      os.makedirs(packages)
      self.framework.actions.addArgument('Framework', 'Directory creation', 'Created the external packages directory: '+packages)
    Dir = None
    self.framework.logPrint('Looking for '+self.PACKAGE+' in directory starting with '+str(self.downloadfilename))
    for d in os.listdir(packages):
      if d.startswith(self.downloadfilename) and os.path.isdir(os.path.join(packages, d)) and not self.matchExcludeDir(d):
        self.framework.logPrint('Found a copy of '+self.PACKAGE+' in '+str(d))
        Dir = d
        break
    if Dir is None:
      self.framework.logPrint('Could not locate an existing copy of '+self.downloadfilename+':')
      self.framework.logPrint('  '+str(os.listdir(packages)))
      if retry <= 0:
        raise RuntimeError('Unable to download '+self.downloadname)
      self.downLoad()
      return self.getDir(retry = 0)
    if not self.archIndependent:
      if not os.path.isdir(os.path.join(packages, Dir, self.arch)):
        os.mkdir(os.path.join(packages, Dir, self.arch))
    return os.path.join(packages, Dir)

  def downLoad(self):
    '''Downloads a package; using hg or ftp; opens it in the with-external-packages-dir directory'''
    import retrieval

    retriever = retrieval.Retriever(self.sourceControl, argDB = self.framework.argDB)
    retriever.setup()
    self.framework.logPrint('Downloading '+self.name)
    # check if its http://ftp.mcs - and add ftp://ftp.mcs as fallback
    download_urls = []
    for url in self.download:
      download_urls.append(url)
      if url.find('http://ftp.mcs.anl.gov') >=0:
        download_urls.append(url.replace('http://','ftp://'))
    # now attempt to download each url until any one succeeds.
    err =''
    for url in download_urls:
      try:
        retriever.genericRetrieve(url, self.externalPackagesDir, self.downloadname)
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.name+' into '+self.getDir(0))
        return
      except RuntimeError, e:
        self.logPrint('ERROR: '+str(e))
        err += str(e)
    raise RuntimeError(err)

  def Install(self):
    raise RuntimeError('No custom installation implemented for package '+self.package+'\n')

  def checkInclude(self, incl, hfiles, otherIncludes = [], timeout = 600.0):
    if self.cxx:
      self.headers.pushLanguage('C++')
    else:
      self.headers.pushLanguage(self.defaultLanguage)
    ret = self.executeTest(self.headers.checkInclude, [incl, hfiles],{'otherIncludes' : otherIncludes, 'timeout': timeout})
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
        if self.framework.argDB['with-'+package.package] == 1:
          raise RuntimeError('Package '+package.PACKAGE+' needed by '+self.name+' failed to configure.\nMail configure.log to petsc-maint@mcs.anl.gov.')
        else:
          raise RuntimeError('Did not find package '+package.PACKAGE+' needed by '+self.name+'.\nEnable the package using --with-'+package.package+' or --download-'+package.package)
      if hasattr(package, 'dlib')    and not libs  is None: libs  += package.dlib
      if hasattr(package, 'include') and not incls is None: incls += package.include
    return

  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.logPrint('Checking for a functional '+self.name)
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
        self.framework.logPrint('Directory does not exist: %s (while checking "%s" for "%r")' % (directory,location,lib))
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
        self.framework.logPrint('Checking for library in '+location+': '+str(lib))
        if directory: self.framework.logPrint('Contents: '+str(os.listdir(directory)))
      else:
        self.framework.logPrint('Not checking for library in '+location+': '+str(lib)+' because no functions given to check for')
      if self.executeTest(self.libraries.check,[lib, self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran, 'cxxMangle' : self.functionsCxx[0], 'prototype' : self.functionsCxx[1], 'call' : self.functionsCxx[2]}):
        self.lib = lib
        self.framework.logPrint('Checking for headers '+location+': '+str(incl))
        if (not self.includes) or self.checkInclude(incl, self.includes, incls, timeout = 1800.0):
          if self.includes:
            self.include = testedincl
          self.found     = 1
          self.dlib      = self.lib+libs
          if not hasattr(self.framework, 'packages'):
            self.framework.packages = []
          self.directory = directory
          self.framework.packages.append(self)
          return
    if not self.lookforbydefault:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

  def checkSharedLibrary(self):
    '''By default we don\'t care about checking if the library is shared'''
    return 1

  def alternateConfigureLibrary(self):
    '''Called if --with-packagename=0; does nothing by default'''
    pass

  def consistencyChecks(self):
    if 'with-'+self.package+'-dir' in self.framework.argDB and ('with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB):
      raise RuntimeError('Specify either "--with-'+self.package+'-dir" or "--with-'+self.package+'-lib --with-'+self.package+'-include". But not both!')
    if self.framework.argDB['with-'+self.package]:
      if self.cxx and not hasattr(self.compilers, 'CXX'):
        raise RuntimeError('Cannot use '+self.name+' without C++, make sure you do NOT have --with-cxx=0')
      if self.fc and not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot use '+self.name+' without Fortran, make sure you do NOT have --with-fc=0')
      if self.noMPIUni and self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if not self.worksonWindows and self.setCompilers.isCygwin():
        raise RuntimeError('External package '+self.name+' does not work on Microsoft Windows')
      if self.download and self.framework.argDB.has_key('download-'+self.downloadname.lower()) and self.framework.argDB['download-'+self.downloadname.lower()] and not self.downloadonWindows and self.setCompilers.isCygwin():
        raise RuntimeError('External package '+self.name+' does not support --download-'+self.downloadname.lower()+' on Microsoft Windows')
    if not self.download and self.framework.argDB.has_key('download-'+self.downloadname.lower()) and self.framework.argDB['download-'+self.downloadname.lower()]:
      raise RuntimeError('External package '+self.name+' does not support --download-'+self.downloadname.lower())
    return

  def configure(self):
    if self.download and not self.download[0] == 'redefine' and self.framework.argDB['download-'+self.downloadname.lower()]:
      self.framework.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-dir' in self.framework.argDB or 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
    if hasattr(self, 'usePkgConfig') and 'with-'+self.package+'-pkg-config' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
      self.usePkgConfig()

    self.consistencyChecks()
    if self.framework.argDB['with-'+self.package]:
      # If clanguage is c++, test external packages with the c++ compiler
      self.libraries.pushLanguage(self.defaultLanguage)
      self.executeTest(self.configureLibrary)
      self.executeTest(self.checkSharedLibrary)
      self.libraries.popLanguage()
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return
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
    self.downloadfilename     - same as self.name (usage a bit inconsistent)  [string]
  Package subclass typically sets up the following state variables:
    self.download         - url to download source from                   [string]
    self.includes         - names of header files to locate               [list of strings]
    self.liblist          - names of library files to locate              [list of lists of strings]
    self.functions        - names of functions to locate in libraries     [list of strings]
    self.cxx              - whether C++ is required for this package      [bool]
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

  Comment:
    There appears to be no good reason for separating setupHelp and setupDependencies
  from the init stage: these hooks are called immediately following configure object
  construction and do no depend on any other intervening computation.
    It appears that hooks/callbacks are necessary only when a customizable action must be carried out
  at a specific point in the configure process, which is not known a priori and/or is controlled by the framework.
  For example, setupDownload (see GNUPackage below) must be called only after it has been determined
  (by the code outside of the package class) that a download is necessary.  Otherwise (e.g., if setupDownload
  is called from __init__), setupDownload will prompt the user for the version of the package to download even
  when no download is necessary (and this is annoying).

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
          self.worksonWindows == 0 implies we cannot use Cygwin compilers
          check that download of this package works on Windows (if Windows is being used)
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
            self.confDir      /* subdir of self.installDir */
            self.includeDir   /* subdir of self.installDir */
            self.libDir       /* subdir of self.installDir */
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
  or all of its methods (see config/PETSc/packages/hdf5.py, for example).
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

  Install:
  -------
  GNUPackage.Install defines a new list of optional dependendies in __init__
    self.odeps (Cf. self.deps),
  which can be, like self.deps set in the setupDependencies callback, as well as a new callback
    formGNUConfigureDepArgs,
  which constructs the GNU configure options based on self.deps and self.odeps the following way:
    for each d in self.deps and in self.odeps, configure option \'--with-\'+d.package+\'=\'+d.directory
    is added to the argument list.
  The formGNUConfigureDepArgs method is called from another callback
    formGNUConfigureArgs,
  which adds the prefix and compiler arguments to the list of GNU configure arguments.
  GNUPackage.Install then runs GNU configure on the package with the arguments obtained from formGNUConfigureArgs.
  Each of the formGNUConfigure*Args callbacks can be overriden to provide more specific options.
  Note that dependencies on self.odeps are optional in the sense that if they are not found,
  the package is still configured, but the corresponding "--with-" argument is omitted from the GNU
  configure options.

  Besides running GNU configure, GNUPackage.Install runs installNeeded, make and postInstall
  at the appropriate times, automatically determining whether a rebuild is necessary, saving
  a GNU configure arguments stamp to perform the check in the future, etc.

  setupDownload:
  -------------
  GNUPackage provides a new callback
    setupDownload
  which is called only when the package is downloaded (as opposed to being used from a tar file).
  By default this method constructs self.download from the other instance variables as follows:
    self.download = [self.downloadpath+self.downloadname+self.downloadversion+self.downloadext]
  Variables self.downloadpath, self.downloadext and self.downloadversion can be set in __init__ or
  using the following hook, which is called at the beginning of setupDownload:
    setupVersion
  is provided that will set self.downloadversion from the command-line argument "--download-"+self.package+"-version",
  prompting for user input, if necessary.
  Clearly, both setupDownload and setupDownloadVersion can be overridden by specific package configure subclasses.
  They are intended to be a convenient hooks for isolating the download url management based on the command-line arguments
  and user input.

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
    - override setupDependencies to "require" dependent objects and to set up the following instance veriables
        self.deps
        self.odeps
      as appropriate
    - override setupDownload to control the precise download URL and/or
    - override setupDownloadVersion to control the self.downloadversion string inserted into self.download between self.downloadpath and self.downloadext
    - override formGNUConfigureDepArgs and/or formGNUConfigureArgs to control the GNU configure options
'''

class GNUPackage(Package):
  def __init__(self, framework):
    Package.__init__(self,framework)
    self.downloadpath=''
    self.downloadversion=''
    self.downloadext=''
    self.setupDefaultDownload()
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    downloadversion = None
    if hasattr(self, 'downloadversion'):
      downloadversion = self.downloadversion
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-version=<string>',  nargs.Arg(None, downloadversion, 'Version number of '+self.PACKAGE+' to download'))
    help.addArgument(self.PACKAGE, '-download-'+self.package+'-shared=<bool>',     nargs.ArgBool(None, 0, 'Install '+self.PACKAGE+' with shared libraries'))

  def setupDependencies(self,framework):
    config.package.Package.setupDependencies(self, framework)
    # optional dependencies, that will be turned off in GNU configure, if they are absent
    self.odeps = []

  def setupDownloadVersion(self):
    '''Use this to construct a valid download URL.'''
    if self.framework.argDB['download-'+self.package+'-version']:
      self.downloadversion = self.framework.argDB['download-'+self.package+'-version']

  def setupDefaultDownload(self):
    '''This is used to set up the default download url, without potentially prompting for user input,
    to make sure that the package configuration is not skipped by configureLibrary.'''
    if hasattr(self,'downloadpath') and hasattr(self,'downloadname') and hasattr(self,'downloadversion') and hasattr(self,'downloadext'):
      self.download = [self.downloadpath+self.downloadname+'-'+self.downloadversion+'.'+self.downloadext]

  def setupDownload(self):
    '''Override this, if necessary, to set up a custom download URL.'''
    self.setupDownloadVersion()
    self.setupDefaultDownload()

  def checkDownload(self, requireDownload = 1):
    self.setupDownload()
    return Package.checkDownload(self,requireDownload)

  def formGNUConfigureDepArgs(self):
    '''Add args corresponding to --with-<deppackage>=<deppackage-dir>.'''
    args = []
    for d in self.deps:
      if d.directory is not None and not d.directory == "":
        args.append('--with-'+d.package+'='+d.directory)
    for d in self.odeps:
      if hasattr(d,'found') and d.found:
        args.append('--with-'+d.package+'='+d.directory)
    return args

  def formGNUConfigureArgs(self):
    '''This sets up the prefix, compiler flags, shared flags, and other generic arguments
       that are fed into the configure script supplied with the package.'''
    args=[]
    ## prefix
    args.append('--prefix='+self.installDir)
    ## compiler args
    self.pushLanguage('C')
    compiler = self.getCompiler()
    args.append('CC="'+self.getCompiler()+'"')
    args.append('CFLAGS="'+self.getCompilerFlags()+'"')
    self.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('CXX="'+self.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.getCompilerFlags()+'"')
      self.popLanguage()
    else:
      args.append('--disable-cxx')
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      fc = self.getCompiler()
      if self.compilers.fortranIsF90:
        try:
          output, error, status = self.executeShellCommand(fc+' -v')
          output += error
        except:
          output = ''
        if output.find('IBM') >= 0:
          fc = os.path.join(os.path.dirname(fc), 'xlf')
          self.framework.log.write('Using IBM f90 compiler, switching to xlf for compiling ' + self.PACKAGE + '\n')
        # now set F90
        args.append('F90="'+fc+'"')
        args.append('F90FLAGS="'+self.getCompilerFlags().replace('-Mfree','')+'"')
      else:
        args.append('--disable-f90')
      args.append('F77="'+fc+'"')
      args.append('FFLAGS="'+self.getCompilerFlags().replace('-Mfree','')+'"')
      self.popLanguage()
    else:
      args.append('--disable-f77')
      args.append('--disable-f90')
    if self.framework.argDB['with-shared-libraries'] or self.framework.argDB['download-'+self.package+'-shared']:
      if self.compilers.isGCC or config.setCompilers.Configure.isIntel(compiler):
        if config.setCompilers.Configure.isDarwin():
          args.append('--enable-sharedlibs=gcc-osx')
        else:
          args.append('--enable-sharedlibs=gcc')
      elif config.setCompilers.Configure.isSun(compiler):
        args.append('--enable-sharedlibs=solaris-cc')
      else:
        args.append('--enable-sharedlibs=libtool')
    else:
        args.append('--disable-shared')
    args.extend(self.formGNUConfigureDepArgs())
    return args

  def Install(self):
    ##### getInstallDir calls this, and it sets up self.packageDir (source download), self.confDir and self.installDir
    if not os.path.isdir(self.installDir):
      os.mkdir(self.installDir)
    ### Build the configure arg list, dump it into a conffile
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package)
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir
    ### Configure and Build package
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2000, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=6000, log = self.framework.log)
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && make clean', timeout=200, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3, self.package)
    return self.installDir

  def configure(self):
    self.setupDefaultDownload()
    Package.configure(self)

  def checkDependencies(self, libs = None, incls = None):
    Package.checkDependencies(self, libs, incls)
    for package in self.odeps:
      if not package.found:
        if self.framework.argDB['with-'+package.package] == 1:
          raise RuntimeError('Package '+package.PACKAGE+' needed by '+self.name+' failed to configure.\nMail configure.log to petsc-maint@mcs.anl.gov.')
      if hasattr(package, 'dlib')    and not libs  is None: libs  += package.dlib
      if hasattr(package, 'include') and not incls is None: incls += package.include
    return

  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.logPrint('Checking for a functional '+self.name)
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    self.checkDependencies(libs, incls)
    if self.needsMath:
      if self.libraries.math is None:
        raise RuntimeError('Math library [libm.a or equivalent] is not found')
      libs += self.libraries.math
    if self.needsCompression:
      if self.libraries.compression is None:
        raise RuntimeError('Compression [libz.a or equivalent] library not found')
      libs += self.libraries.compression

    for location, directory, lib, incl in self.generateGuesses():
      if directory and not os.path.isdir(directory):
        self.framework.logPrint('Directory does not exist: %s (while checking "%s" for "%r")' % (directory,location,lib))
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
        self.framework.logPrint('Checking for library in '+location+': '+str(lib))
        if directory: self.framework.logPrint('Contents: '+str(os.listdir(directory)))
      else:
        self.framework.logPrint('Not checking for library in '+location+': '+str(lib)+' because no functions given to check for')
      if self.executeTest(self.libraries.check,[lib, self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran, 'cxxMangle' : self.functionsCxx[0], 'prototype' : self.functionsCxx[1], 'call' : self.functionsCxx[2]}):
        self.lib = lib
        self.framework.logPrint('Checking for headers '+location+': '+str(incl))
        if (not self.includes) or self.checkInclude(incl, self.includes, incls, timeout = 1800.0):
          if self.includes:
            self.include = testedincl
          self.found     = 1
          self.dlib      = self.lib+libs
          if not hasattr(self.framework, 'packages'):
            self.framework.packages = []
          self.directory = directory
          self.framework.packages.append(self)
          return
    if not self.lookforbydefault:
      raise RuntimeError('Could not find a functional '+self.name+'\n')
