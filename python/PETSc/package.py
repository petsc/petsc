from __future__ import generators
import config.base
import os
import re
import sys
import md5

class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSc'
    self.substPrefix  = 'PETSc'
    self.compilers    = self.framework.require('config.compilers',self)
    self.setCompilers = self.framework.require('config.setCompilers',self)  
    self.libraries    = self.framework.require('config.libraries',self)
    self.clanguage    = self.framework.require('PETSc.utilities.clanguage',self)
    self.arch         = self.framework.require('PETSc.utilities.arch',self)        
    self.functions    = self.framework.require('config.functions',self)
    self.source       = self.framework.require('config.sourceControl',self)    
    self.found        = 0
    self.lib          = []
    # this packages libraries and all those it depends on
    self.dlib         = []
    self.include      = []
    if hasattr(sys.modules.get(self.__module__), '__file__'):
      self.name       = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    else:
      self.name       = 'DEBUGGING'
    self.downloadname = self.name
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    self.version      = ''
    self.license      = None
    # ***********  these are optional items set in the particular packages file
    self.complex      = 0   # 1 means cannot use complex
    self.cxx          = 0   # 1 means requires C++
    self.fc           = 0   # 1 means requires fortran
    # urls where bk or tarballs may be found
    self.download     = []
    # other packages whose dlib or include we depend on (maybe can be figured automatically)
    # usually this is the list of all the assignments in the package that use self.framework.require()
    self.deps         = []
    # functions we wish to check in the libraries
    self.functions    = []
    # include files we wish to check for
    self.includes     = []
    # list of libraries we wish to check for (can be overwritten by providing your own generateLibraryList()
    self.liblist      = [[]]
    self.extraLib     = []
    # location of libraries and includes in packages directory tree
    self.libdir       = 'lib'
    self.includedir   = 'include'
    # package defaults to being required (MPI and BlasLapack)
    self.required     = 0
    # Need this for the with-64-bit-ints option
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
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if self.download:
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Download and install '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of the '+self.name+' include files'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<dir,or list of libraries>',nargs.ArgLibrary(None,None,'Indicate the directory of the '+self.name+' libraries or a list of libraries'))    
    return

  def checkLibrary(self, library):
    if os.path.isfile(library):
      return 1
    libBase, ext = os.path.splitext(library)
    if os.path.isfile(libBase+'.'+self.setCompilers.sharedLibraryExt):
      return 1
    if os.path.isfile(libBase+'.'+self.setCompilers.AR_LIB_SUFFIX):
      return 1
    self.framework.log.write('Nonexistent library '+str(library)+'\n')
    return 0

  # by default, just check for all the libraries in self.liblist 
  def generateLibList(self, dir):
    '''Generates full path list of libraries from self.liblist'''
    alllibs = []
    for libSet in self.liblist:
      libs = []
      for library in libSet:
        if not self.libdir == dir:
          fullLib = os.path.join(dir, library)
        else:
          fullLib = library
        if self.checkLibrary(fullLib):
          libs.append(fullLib)
        libs.extend(self.extraLib)
      alllibs.append(libs)
    return alllibs

  # By default, don't search any particular directories
  def getSearchDirectories(self):
    return []

  def checkDownload(self,preOrPost):
    '''Check if we should download the package'''
    if self.download and self.framework.argDB['download-'+self.downloadname.lower()] == preOrPost:
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
        self.framework.log.write('Testing library '+str(l)+'\n')
        yield('Download '+self.PACKAGE, l, os.path.join(dir, self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+dir+'\n')

    if 'with-'+self.package+'-dir' in self.framework.argDB:     
      dir = self.framework.argDB['with-'+self.package+'-dir']
      for l in self.generateLibList(os.path.join(dir, self.libdir)):
        yield('User specified root directory '+self.PACKAGE, l, os.path.join(dir,self.includedir))

    if 'with-'+self.package+'-include' in self.framework.argDB and 'with-'+self.package+'-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-'+self.package+'-lib']
      if not isinstance(libs, list): libs = [libs]
      libs = [os.path.abspath(l) for l in libs]
      yield('User specified '+self.PACKAGE+' libraries', libs, os.path.abspath(self.framework.argDB['with-'+self.package+'-include']))

    for d in self.getSearchDirectories():
      for l in self.generateLibList(os.path.join(d,self.libdir)):
        yield('User specified root directory '+self.PACKAGE, l, os.path.join(d,self.includedir))

    dir = self.checkDownload(2)
    if dir:
      for l in self.generateLibList(os.path.join(dir,self.libdir)):
        yield('Download '+self.PACKAGE, l, os.path.join(dir,self.includedir))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+os.path.join(self.Install(),self.arch.arch)+'\n')

    raise RuntimeError('You must specify a path for '+self.name+' with --with-'+self.package+'-dir=<directory>')

  def downLoad(self):
    '''Downloads a package; using bk or ftp; opens it in the with-external-packages-dir directory'''
    self.framework.log.write('Downloading '+self.name+'\n')
    packages  = self.framework.argDB['with-external-packages-dir']
    
    if hasattr(self.source,'bk'):
      for url in self.download:
        if url.startswith('bk://'):
          import commands

          try:
            self.framework.log.write('Downloading it using "bk clone '+url+' '+os.path.join(packages,self.package)+'"\n')
            (status,output) = commands.getstatusoutput('bk clone '+url+' '+os.path.join(packages,self.package))
          except RuntimeError, e:
            raise RuntimeError('Error bk cloning '+self.package+' '+str(e))        
          if status:
            if output.find('ommand not found') >= 0:
              raise RuntimeError('Unable to locate bk (Bitkeeper) to download BuildSystem; make sure bk is in your path')
            elif output.find('Cannot resolve host') >= 0:
              raise RuntimeError('Unable to download '+self.package+'. You must be off the network. Connect to the internet and run config/configure.py again')
          self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir(0))
          return

    for url in self.download:
      if url.startswith('http://') or url.startswith('ftp://'):      
        import urllib
        tarname   = self.name+'.tar'
        tarnamegz = tarname+'.gz'
        try:
          urllib.urlretrieve(url, os.path.join(packages,tarnamegz ))
        except Exception, e:
          raise RuntimeError('Error downloading '+self.name+': '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; gunzip '+tarnamegz, log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error unzipping '+tarname+': '+str(e))
        try:
          config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf '+tarname, log = self.framework.log)
        except RuntimeError, e:
          raise RuntimeError('Error doing tar -xf '+tarname+': '+str(e))
        os.unlink(os.path.join(packages, tarname))
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir(0))
        return
    raise RuntimeError('Unable to download '+self.package+' from locations '+str(self.download)) 

  def getDir(self, retry = 1):
    '''Find the directory containing the package'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    Dir = None
    for dir in os.listdir(packages):
      if dir.startswith(self.downloadname) and os.path.isdir(os.path.join(packages, dir)):
        Dir = dir
        break
    if Dir is None:
      self.framework.log.write('Could not locate an existing copy of '+self.downloadname+'\n')
      if retry <= 0:
        raise RuntimeError('Unable to download '+self.downloadname)
      self.downLoad()
      return self.getDir(retry = 0)
    if not os.path.isdir(os.path.join(packages,Dir,self.arch.arch)):
      os.mkdir(os.path.join(packages,Dir,self.arch.arch))
    return os.path.join(packages, Dir)

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
      
    for location, lib, incl in self.generateGuesses():
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for library in '+location+': '+str(lib)+'\n')
      if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs}):      
        self.lib = lib	
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.libraries.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          if self.checkSharedLibrary():
            self.found   = 1
            self.dlib    = self.lib+libs
            if not hasattr(self.framework, 'packages'):
              self.framework.packages = []
            self.framework.packages.append(self)
            break
    if not self.found:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

  def configure(self):
    '''Determines if the package should be configured for, then calls the configure'''
    if self.download and self.framework.argDB['download-'+self.package]:
      self.framework.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-dir' in self.framework.argDB or 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
      
    if self.framework.argDB['with-'+self.package]:
      if hasattr(self,'mpi') and self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.libraryOptions.integerSize == 64:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')    
      if not self.clanguage.precision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')    
      if not self.complex and self.clanguage.scalartype.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')    
      if self.cxx and not self.clanguage.language.lower() == 'cxx':
        raise RuntimeError('Cannot use '+self.name+' without C++, run config/configure.py --with-language=c++')    
      if self.fc and not 'FC' in self.framework.argDB:
        raise RuntimeError('Cannot use '+self.name+' without Fortran, run config/configure.py --with-fc')    
      self.executeTest(self.configureLibrary)
    return

