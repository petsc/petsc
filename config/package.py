from __future__ import generators
import config.base

import os

class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix     = 'PETSc'
    self.substPrefix      = 'PETSc'
    self.arch             = None # The architecture identifier
    self.externalPackagesDir = os.path.abspath('externalpackages')
    # These are derived by the configure tests
    self.found            = 0
    self.setNames()
    self.include          = []
    self.lib              = []
    self.dlib             = []   # all libraries in this package and all those it depends on
    self.directory        = None # path of the package installation point; for example /usr/local or /home/bsmith/mpich-2.0.1
    self.version          = ''
    # These are specified for the package
    self.required         = 0    # 1 means the package is required
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
    self.libdir           = 'lib'     # location of libraries in the package directory tree
    self.includedir       = 'include' # location of includes in the package directory tree
    self.license          = None # optional license text
    self.excludedDirs     = []   # list of directory names that could be false positives, SuperLU_DIST when looking for SuperLU
    self.archIndependent  = 0    # 1 means the install directory does not incorporate the ARCH name
    return
    
  def __str__(self):
    '''Prints the location of the packages includes and libraries'''
    output = ''
    if self.found:
      output = self.name+':\n'
      if self.version: output += '  Version:  '+self.version+'\n'
      if self.include: output += '  Includes: '+str(self.include)+'\n'
      if self.lib:     output += '  Library:  '+str(self.lib)+'\n'
    return output

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers  = framework.require('config.setCompilers', self)
    self.compilers     = framework.require('config.compilers', self)
    self.headers       = framework.require('config.headers', self)
    self.libraries     = framework.require('config.libraries', self)
    self.sourceControl = framework.require('config.sourceControl',self)
    return

  def setupHelp(self,help):
    '''Prints help messages for the package'''
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if self.download and not self.download[0] == 'redefine':
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded,filename>', nargs.ArgDownload(None, 0, 'Download and install '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of the '+self.name+' include files'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<libraries: e.g. [/Users/..../libparmetis.a,...]>',nargs.ArgLibrary(None,None,'Indicate the '+self.name+' libraries'))    
    return

  def setNames(self):
    '''Setup various package names
    name:         The module name (usually the filename)
    package:      The lowercase name
    PACKAGE:      The uppercase name
    downloadname: Name for download option and file (usually name)
    '''
    import sys
    if hasattr(sys.modules.get(self.__module__), '__file__'):
      self.name       = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    else:
      self.name       = 'DEBUGGING'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    self.downloadname = self.name
    return

  def getDefaultLanguage(self):
    '''The language in which to run tests'''
    if hasattr(self, 'languageProvider'):
      if hasattr(self.languageProvider, 'defaultLanguage'):
        return self.languageProvider.defaultLanguage
      elif hasattr(self.languageProvider, 'clanguage'):
        return self.languageProvider.clanguage
    return self._defaultLanguage
  def setDefaultLanguage(self, defaultLanguage):
    '''The language in which to run tests'''
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
    if self.archIndependent:
      return os.path.abspath(self.Install())
    return os.path.abspath(os.path.join(self.Install(), self.arch))

  def generateLibList(self, directory):
    '''Generates full path list of libraries from self.liblist'''
    alllibs = []
    for libSet in self.liblist:
      libs = []
      # add full path only to the first library in the list
      if not self.libdir == directory and len(libSet) > 0:
        libs.append(os.path.join(directory, libSet[0]))
      for library in libSet[1:]:
        # if the library name doesn't start with lib - then add the fullpath
        if library.startswith('lib') or self.libdir == directory:
          libs.append(library)
        else:
          libs.append(os.path.join(directory, library))
      libs.extend(self.extraLib)
      alllibs.append(libs)
    return alllibs

  def generateGuesses(self):
    d = self.checkDownload(1)
    if d:
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('Download '+self.PACKAGE, d, l, os.path.join(d, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+d+'\n')

    if 'with-'+self.package+'-dir' in self.framework.argDB:     
      d = self.framework.argDB['with-'+self.package+'-dir']
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('User specified root directory '+self.PACKAGE, d, l, os.path.join(d, self.includedir))
      if 'with-'+self.package+'-include' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-include if you set --with-'+self.package+'-dir')
      if 'with-'+self.package+'-lib' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-lib if you set --with-'+self.package+'-dir')
      raise RuntimeError('--with-'+self.package+'-dir='+self.framework.argDB['with-'+self.package+'-dir']+' did not work')

    if 'with-'+self.package+'-include' in self.framework.argDB and not 'with-'+self.package+'-lib' in self.framework.argDB:
      raise RuntimeError('If you provide --with-'+self.package+'-include you must also supply with-'+self.package+'-lib\n')
    if 'with-'+self.package+'-lib' in self.framework.argDB and not 'with-'+self.package+'-include' in self.framework.argDB:
      raise RuntimeError('If you provide --with-'+self.package+'-lib you must also supply with-'+self.package+'-include\n')
    if 'with-'+self.package+'-include-dir' in self.framework.argDB:
        raise RuntimeError('Use --with-'+self.package+'-include; not --with-'+self.package+'-include-dir') 

    if 'with-'+self.package+'-include' in self.framework.argDB and 'with-'+self.package+'-lib' in self.framework.argDB:
      # hope that package root is one level above include directory
      d = os.path.dirname(self.framework.argDB['with-'+self.package+'-include'])
      inc = self.framework.argDB['with-'+self.package+'-include']
      libs = self.framework.argDB['with-'+self.package+'-lib']
      if not isinstance(libs, list): libs = [libs]
      libs = [os.path.abspath(l) for l in libs]
      yield('User specified '+self.PACKAGE+' libraries', d, libs, os.path.abspath(inc))
      raise RuntimeError('--with-'+self.package+'-lib='+str(self.framework.argDB['with-'+self.package+'-lib'])+' and \n'+\
                         '--with-'+self.package+'-include='+str(self.framework.argDB['with-'+self.package+'-include'])+' did not work') 

    for d in self.getSearchDirectories():
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        if isinstance(self.includedir, list):
          includedir = ([inc for inc in self.includedir if os.path.isabs(inc)] +
                        [os.path.join(d, inc) for inc in self.includedir if not os.path.isabs(inc)])
        elif d:
          includedir = os.path.join(d, self.includedir)
        else:
          includedir = ''
        yield('Package specific search directory '+self.PACKAGE, d, l, includedir)

    d = self.checkDownload(requireDownload = 0)
    if d:
      for l in self.generateLibList(os.path.join(d, self.libdir)):
        yield('Download '+self.PACKAGE, d, l, os.path.join(d, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+self.getInstallDir()+'\n')

    raise RuntimeError('You must specify a path for '+self.name+' with --with-'+self.package+'-dir=<directory>')

  def checkDownload(self, requireDownload = 1):
    '''Check if we should download the package, returning the install directory or the empty string indicating installation'''
    if not self.download:
      return ''
    downloadPackage = 0
    if requireDownload and isinstance(self.framework.argDB['download-'+self.downloadname.lower()], str):
      self.download = ['file://'+os.path.abspath(self.framework.argDB['download-'+self.downloadname.lower()])]
      downloadPackage = 1
    elif self.framework.argDB['download-'+self.downloadname.lower()] == 1 and requireDownload:
      downloadPackage = 1
    elif self.framework.argDB['download-'+self.downloadname.lower()] == 2 and not requireDownload:
      downloadPackage = 1

    if downloadPackage:
      if not self.download:
        raise RuntimeError('URL missing for package'+self.package+'.\n')
      if self.license and not os.path.isfile(os.path.expanduser(os.path.join('~','.'+self.package+'_license'))):
        self.framework.logClear()
        self.logPrint("**************************************************************************************************", debugSection='screen')
        self.logPrint('You must register to use '+self.downloadname+' at '+self.license, debugSection='screen')
        self.logPrint('    Once you have registered, config/configure.py will continue and download and install '+self.downloadname+' for you', debugSection='screen')
        self.logPrint("**************************************************************************************************\n", debugSection='screen')
        fd = open(os.path.expanduser(os.path.join('~','.'+self.package+'_license')),'w')
        fd.close()
      return self.getInstallDir()
    return ''

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
      os.mkdir(packages)
      self.framework.actions.addArgument('Framework', 'Directory creation', 'Created the external packages directory: '+packages)
    Dir = None
    for d in os.listdir(packages):
      if d.startswith(self.downloadname) and os.path.isdir(os.path.join(packages, d)) and not self.matchExcludeDir(d):
        Dir = d
        break
    if Dir is None:
      self.framework.logPrint('Could not locate an existing copy of '+self.downloadname+':')
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
    '''Downloads a package; using bk or ftp; opens it in the with-external-packages-dir directory'''
    import install.retrieval

    retriever = install.retrieval.Retriever(self.sourceControl, argDB = self.framework.argDB)
    retriever.setup()
    failureMessage = []
    self.framework.logPrint('Downloading '+self.name)
    for url in self.download:
      try:
        retriever.genericRetrieve(url, self.externalPackagesDir, self.downloadname)
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.name+' into '+self.getDir(0))
        return
      except RuntimeError, e:
        failureMessage.append('  Failed to download '+url+'\n'+str(e))
    failureMessage = 'Unable to download '+self.package+' from locations '+str(self.download)+'\n'+'\n'.join(failureMessage)
    raise RuntimeError(failureMessage)

  def Install(self):
    raise RuntimeError('No custom installation implemented for package '+self.package+'\n')

  def checkInclude(self, incl, hfiles, otherIncludes = [], timeout = 600.0):
    if self.cxx:
      self.headers.pushLanguage('C++')
    ret = self.executeTest(self.headers.checkInclude, [incl, hfiles],{'otherIncludes' : otherIncludes, 'timeout': timeout})
    if self.cxx:
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

  def configureLibrary(self):
    '''Find an installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.logPrint('Checking for a functional '+self.name)
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    for package in self.deps:
      if not hasattr(package, 'found'):
        raise RuntimeError('Package '+package.name+' does not have found attribute!')
      if not package.found:
        if self.framework.argDB['with-'+package.package] == 1:
          raise RuntimeError('Package '+package.PACKAGE+' needed by '+self.name+' failed to configure.\nMail configure.log to petsc-maint@mcs.anl.gov.')
        else:
          raise RuntimeError('Did not find package '+package.PACKAGE+' needed by '+self.name+'.\nEnable the package using --with-'+package.package)
      if hasattr(package, 'dlib'):    libs  += package.dlib
      if hasattr(package, 'include'): incls += package.include
    if self.needsMath:
      if self.libraries.math is None:
        raise RuntimeError('Math library not found')
      libs += self.libraries.math
      
    for location, directory, lib, incl in self.generateGuesses():
      if lib == '': lib = []
      elif not isinstance(lib, list): lib = [lib]
      if incl == '': incl = []
      elif not isinstance(incl, list): incl = [incl]
      incl += self.compilers.fincs
      self.framework.logPrint('Checking for library in '+location+': '+str(lib))
      if self.executeTest(self.libraries.check,[lib, self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran, 'cxxMangle' : self.functionsCxx[0], 'prototype' : self.functionsCxx[1], 'call' : self.functionsCxx[2]}):
        self.lib = lib	
        self.framework.logPrint('Checking for headers '+location+': '+str(incl))
        if (not self.includes) or self.checkInclude(incl, self.includes, incls, timeout = 1800.0):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
          if not hasattr(self.framework, 'packages'):
            self.framework.packages = []
          self.directory = directory
          self.framework.packages.append(self)
          return
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
        raise RuntimeError('Cannot use '+self.name+' without C++, run config/configure.py --with-cxx')
      if self.fc and not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot use '+self.name+' without Fortran, run config/configure.py --with-fc')
    return

  def configure(self):
    if self.download and not self.download[0] == 'redefine' and self.framework.argDB['download-'+self.downloadname.lower()]:
      self.framework.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-dir' in self.framework.argDB or 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1

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
