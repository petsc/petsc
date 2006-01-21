from __future__ import generators
import config.base
import os
import re
import sys
import md5

import nargs

class ArgDownload(nargs.Arg):
  '''Arguments that represent software downloads'''
  def __init__(self, key, value = None, help = '', isTemporary = 0):
    nargs.Arg.__init__(self, key, value, help, isTemporary)
    return

  def valueName(self, value):
    if value == 0:
      return 'no'
    elif value == 1:
      return 'yes'
    elif value == 2:
      return 'ifneeded'
    return str(value)

  def __str__(self):
    if not self.isValueSet():
      return 'Empty '+str(self.__class__)
    elif isinstance(self.value, list):
      return str(map(self.valueName, self.value))
    return self.valueName(self.value)

  def getEntryPrompt(self):
    return 'Please enter download value for '+str(self.key)+': '

  def setValue(self, value):
    '''Set the value. SHOULD MAKE THIS A PROPERTY'''
    try:
      if   value == '0':        value = 0
      elif value == '1':        value = 1
      elif value == 'no':       value = 0
      elif value == 'yes':      value = 1
      elif value == 'false':    value = 0
      elif value == 'true':     value = 1
      elif value == 'ifneeded': value = 2
      elif not isinstance(value, int):
        value = str(value)
    except:
      raise TypeError('Invalid download value: '+str(value)+' for key '+str(self.key))
    if isinstance(value, str) and not os.path.isfile(value):
      raise ValueError('Invalid download location: '+str(value)+' for key '+str(self.key))
    self.value = value
    return

class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix     = 'PETSc'
    self.substPrefix      = 'PETSc'
    self.found            = 0
    self.lib              = []
    # this packages libraries and all those it depends on
    self.dlib             = []
    self.include          = []
    if hasattr(sys.modules.get(self.__module__), '__file__'):
      self.name           = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    else:
      self.name           = 'DEBUGGING'
    self.downloadname     = self.name
    self.excludename      = []  # list of names that could be false positives for ex: SuperLU_DIST when looking for SuperLU
    self.PACKAGE          = self.name.upper()
    self.package          = self.name.lower()
    self.version          = ''
    self.license          = None
    # ***********  these are optional items set in the particular packages file
    self.complex          = 0   # 0 means cannot use complex
    self.cxx              = 0   # 1 means requires C++
    self.fc               = 0   # 1 means requires fortran
    self.double           = 1   # 1 means requires double precision 
    self.requires32bitint = 1;
    # urls where bk or tarballs may be found
    self.download         = []
    # other packages whose dlib or include we depend on (maybe can be figured automatically)
    # usually this is the list of all the assignments in the package that use self.framework.require()
    self.deps             = []
    # functions we wish to check in the libraries
    self.functions        = []
    # indicates if the above symbol is a Fortran symbol [if so - name-mangling check is done]
    self.functionsFortran = 0
    # indicates if the above symbol is a C++ symbol [if so - name-mangling check with prototype is done]
    self.functionsCxx     = [0, '', '']
    # include files we wish to check for
    self.includes         = []
    # list of libraries we wish to check for (can be overwritten by providing your own generateLibraryList()
    self.liblist          = [[]]
    self.extraLib         = []
    # location of libraries and includes in packages directory tree
    self.libdir           = 'lib'
    self.includedir       = 'include'
    # package defaults to being required (MPI and BlasLapack)
    self.required         = 0
    # package needs the system math library
    self.needsMath        = 0
    # path of the package installation point; for example /usr/local or /home/bsmith/mpich-2.0.1
    self.directory        = None
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers  = self.framework.require('config.setCompilers',self)
    self.compilers     = self.framework.require('config.compilers',self)
    self.headers       = self.framework.require('config.headers',self)
    self.libraries     = self.framework.require('config.libraries',self)
    self.languages     = self.framework.require('PETSc.utilities.languages',self)
    self.arch          = self.framework.require('PETSc.utilities.arch',self)
    self.petscdir      = self.framework.require('PETSc.utilities.petscdir',self)
    self.programs      = self.framework.require('PETSc.utilities.programs', self)
    self.sourceControl = self.framework.require('config.sourceControl',self)
    # Need this for the with-64-bit-indices option
    self.libraryOptions = framework.require('PETSc.utilities.libraryOptions', self)
    return

  def getChecksum(self,source, chunkSize = 1024*1024):  
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if isinstance(source, file):
      f = source
    else:
      f = file(source)
    m = md5.new()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()
    
  def __str__(self):
    '''Prints the location of the packages includes and libraries'''
    output=''
    if self.found:
      output  = self.name+':\n'
      if self.version: output += '  Version: '+self.version+'\n'
      if self.include: output += '  Includes: '+str(self.include)+'\n'
      if self.lib:     output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def setupHelp(self,help):
    '''Prints help messages for the package'''
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if self.download:
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded,filename>', ArgDownload(None, 0, 'Download and install '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of the '+self.name+' include files'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<libraries: e.g. [/Users/..../libparmetis.a,...]>',nargs.ArgLibrary(None,None,'Indicate the '+self.name+' libraries'))    
    return

  # by default, just check for all the libraries in self.liblist 
  def generateLibList(self, dir):
    '''Generates full path list of libraries from self.liblist'''
    alllibs = []
    for libSet in self.liblist:
      libs = []
      # add full path only to the first library in the list
      if not self.libdir == dir and libSet != []:
        libs.append(os.path.join(dir, libSet[0]))
      for library in libSet[1:]:
        # if the library name doesn't start with lib - then add the fullpath
        if library.startswith('lib') or self.libdir == dir:
          libs.append(library)
        else:
          libs.append(os.path.join(dir, library))
      libs.extend(self.extraLib)
      alllibs.append(libs)
    return alllibs
    
  # By default, don't search any particular directories
  def getSearchDirectories(self):
    return []

  def checkDownload(self,preOrPost):
    '''Check if we should download the package'''
    dowork=0
    if preOrPost==1 and isinstance(self.framework.argDB['download-'+self.downloadname.lower()], str):
      self.download = ['file://'+os.path.abspath(self.framework.argDB['download-'+self.downloadname.lower()])]
      dowork=1
    elif self.framework.argDB['download-'+self.downloadname.lower()] == preOrPost:
      dowork=1

    if not self.download:
      raise RuntimeError('URL missing for package'+self.package+'. perhaps a PETSc bug\n')
    
    if dowork:
      if self.license and not os.path.isfile(os.path.expanduser(os.path.join('~','.'+self.package+'_license'))):
        self.framework.logClear()
        self.logPrint("**************************************************************************************************", debugSection='screen')
        self.logPrint('You must register to use '+self.downloadname+' at '+self.license, debugSection='screen')
        self.logPrint('    Once you have registered, config/configure.py will continue and download and install '+self.downloadname+' for you', debugSection='screen')
        self.logPrint("**************************************************************************************************\n", debugSection='screen')
        fd = open(os.path.expanduser(os.path.join('~','.'+self.package+'_license')),'w')
        fd.close()
      return os.path.abspath(os.path.join(self.Install(),self.arch.arch))
    else:
      return ''

  def generateGuesses(self):
    dir = self.checkDownload(1)
    if dir:
      for l in self.generateLibList(os.path.join(dir, self.libdir)):
        yield('Download '+self.PACKAGE, dir,l, os.path.join(dir, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+dir+'\n')

    if 'with-'+self.package+'-dir' in self.framework.argDB:     
      dir = self.framework.argDB['with-'+self.package+'-dir']
      for l in self.generateLibList(os.path.join(dir, self.libdir)):
        yield('User specified root directory '+self.PACKAGE, dir,l, os.path.join(dir,self.includedir))
      raise RuntimeError('--with-'+self.package+'-dir='+self.framework.argDB['with-'+self.package+'-dir']+' did not work')

    if 'with-'+self.package+'-include-dir' in self.framework.argDB:
        raise RuntimeError('Use --with-'+self.package+'-include; not --with-'+self.package+'-include-dir') 

    if 'with-'+self.package+'-include' in self.framework.argDB and 'with-'+self.package+'-lib' in self.framework.argDB:
      # hope that package root is one level above include directory
      dir = os.path.dirname(self.framework.argDB['with-'+self.package+'-include'])
      libs = self.framework.argDB['with-'+self.package+'-lib']
      if not isinstance(libs, list): libs = [libs]
      libs = [os.path.abspath(l) for l in libs]
      yield('User specified '+self.PACKAGE+' libraries', dir,libs, os.path.abspath(self.framework.argDB['with-'+self.package+'-include']))
      raise RuntimeError('--with-'+self.package+'-lib='+str(self.framework.argDB['with-'+self.package+'-lib'])+' and \n'+\
        '--with-'+self.package+'-include='+str(self.framework.argDB['with-'+self.package+'-include'])+' did not work') 

    for d in self.getSearchDirectories():
      for l in self.generateLibList(os.path.join(d,self.libdir)):
        if d: includedir = os.path.join(d,self.includedir)
        else: includedir = ''
        yield('User specified root directory '+self.PACKAGE, d,l, includedir)

    dir = self.checkDownload(2)
    if dir:
      for l in self.generateLibList(os.path.join(dir,self.libdir)):
        yield('Download '+self.PACKAGE, dir,l, os.path.join(dir,self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+os.path.join(self.Install(),self.arch.arch)+'\n')

    raise RuntimeError('You must specify a path for '+self.name+' with --with-'+self.package+'-dir=<directory>')

  def downLoad(self):
    '''Downloads a package; using bk or ftp; opens it in the with-external-packages-dir directory'''
    import install.retrieval

    retriever = install.retrieval.Retriever(self.sourceControl, argDB = self.framework.argDB)
    retriever.setup()
    failureMessage = []
    self.framework.log.write('Downloading '+self.name+'\n')
    for url in self.download:
      try:
        retriever.genericRetrieve(url, self.petscdir.externalPackagesDir, self.downloadname)
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.name+' into '+self.getDir(0))
        return
      except RuntimeError, e:
        failureMessage.append('  Failed to download '+url+'\n'+str(e))
    failureMessage = 'Unable to download '+self.package+' from locations '+str(self.download)+'\n'+'\n'.join(failureMessage)
    raise RuntimeError(failureMessage)

  # Check is the dir matches something in the excludename list
  def matchExcludeDir(self,dir):
    for exdir in self.excludename:
      if dir.startswith(exdir):
        return 1
    return 0
      
  def getDir(self, retry = 1):
    '''Find the directory containing the package'''
    packages = self.petscdir.externalPackagesDir
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    Dir = None
    for dir in os.listdir(packages):
      if dir.startswith(self.downloadname) and os.path.isdir(os.path.join(packages, dir)) and not self.matchExcludeDir(dir):
        Dir = dir
        break
    if Dir is None:
      self.framework.log.write('Could not locate an existing copy of '+self.downloadname+':\n'+str(os.listdir(packages)))
      if retry <= 0:
        raise RuntimeError('Unable to download '+self.downloadname)
      self.downLoad()
      return self.getDir(retry = 0)
    if not os.path.isdir(os.path.join(packages,Dir,self.arch.arch)):
      os.mkdir(os.path.join(packages,Dir,self.arch.arch))
    return os.path.join(packages, Dir)

  def checkPackageLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = 0):
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    result = self.checkLink(includes, body, cleanup, codeBegin, codeEnd, shared)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return result

  def checkSharedLibrary(self):
    '''By default we don\'t care about checking if shared'''
    return 1

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    for l in self.deps:
      if not hasattr(l,'found'):
        raise RuntimeError(l.PACKAGE+' does not have found attribute!')
      if not l.found:
        raise RuntimeError('Did not find '+l.PACKAGE+' needed by '+self.name)
      if hasattr(l,'dlib'):    libs  += l.dlib
      if hasattr(l,'include'): incls += l.include
    if self.needsMath:
      if self.libraries.math is None:
        raise RuntimeError('Math library not found')
      libs += self.libraries.math
      
    for location, dir, lib, incl in self.generateGuesses():
      if lib == '': lib = []
      elif not isinstance(lib, list): lib = [lib]
      if incl == '': incl = []
      elif not isinstance(incl, list): incl = [incl]
      incl += self.compilers.fincs
      self.framework.log.write('Checking for library in '+location+': '+str(lib)+'\n')
      if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran, 'cxxMangle' : self.functionsCxx[0], 'prototype' : self.functionsCxx[1], 'call' : self.functionsCxx[2]}):
        self.lib = lib	
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.headers.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
          if not hasattr(self.framework, 'packages'):
            self.framework.packages = []
          self.directory = dir
          self.framework.packages.append(self)
          return
    raise RuntimeError('Could not find a functional '+self.name+'\n')

  def alternateConfigureLibrary(self):
    '''Called if --with-packagename=0; does nothing by default'''
    pass

  def configure(self):
    '''Determines if the package should be configured for, then calls the configure'''
    if self.download and self.framework.argDB['download-'+self.package]:
      self.framework.argDB['with-'+self.package] = 1

    if 'with-'+self.package+'-dir' in self.framework.argDB and ('with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB):
      raise RuntimeError('Use either --with-'+self.package+'-dir or --with-'+self.package+'-lib and --with-'+self.package+'-include Not both!')

    if 'with-'+self.package+'-dir' in self.framework.argDB or 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
      
    if self.framework.argDB['with-'+self.package]:
      if hasattr(self,'mpi') and self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.libraryOptions.integerSize == 64 and self.requires32bitint:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')    
      if self.double and not self.languages.precision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')    
      if not self.complex and self.languages.scalartype.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')    
      if self.cxx and not self.languages.clanguage == 'Cxx':
        raise RuntimeError('Cannot use '+self.name+' without C++, run config/configure.py --with-clanguage=C++')    
      if self.fc and not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot use '+self.name+' without Fortran, run config/configure.py --with-fc')
      # If clanguage is c++, test external packages with the c++ compiler
      self.libraries.pushLanguage(self.languages.clanguage)
      self.executeTest(self.configureLibrary)
      self.executeTest(self.checkSharedLibrary)
      self.libraries.popLanguage()
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return

