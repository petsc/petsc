from __future__ import generators
import config.base
import os
import re
import sys
import md5

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
    self.PACKAGE          = self.name.upper()
    self.package          = self.name.lower()
    self.version          = ''
    self.license          = None
    # ***********  these are optional items set in the particular packages file
    self.complex          = 0   # 1 means cannot use complex
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
    # indicates if the above symblo is a fortran symbol [if so - name-mangling check is done]
    self.functionsFortran = 0
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
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<libraries: e.g. [/Users/..../libparmetis.a,...]>',nargs.ArgLibrary(None,None,'Indicate the '+self.name+' libraries'))    
    return

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
    self.framework.log.write('Downloading '+self.name+'\n')
    packages = self.petscdir.externalPackagesDir
    
    if hasattr(self.sourceControl, 'bk'):
      for url in self.download:
        if url.startswith('bk://'):
          failedmessage = 'Unable to bk clone '+self.package+'\n'+\
                   'You may be off the network. Connect to the internet and run config/configure.py again \n'+\
                   'or from the directory externalpackages try: \n bk clone '+url+' '+self.package+\
                   ' if that succeeds then rerun config/configure.py'
          try:
            self.framework.log.write('Downloading it using "bk clone '+url+' '+os.path.join(packages,self.package)+'"\n')
            (output, error, status) = config.base.Configure.executeShellCommand('bk clone '+url+' '+os.path.join(packages,self.downloadname))
          except RuntimeError, e:
            status = 1
            output = str(e)
            error  = ''
          if status:
            if output.find('ommand not found') >= 0:
              raise RuntimeError('Unable to locate bk (Bitkeeper) to download packages; make sure bk is in your path')
            elif output.find('Cannot resolve host') >= 0:
              self.framework.log.write('Cannot bk clone:'+'\n'+output+'\n'+error+'\n')
              raise RuntimeError(failedmessage)
            else:
              # Bitkeeper ports could be blocked
              try:
                (output, error, status) = config.base.Configure.executeShellCommand('bk clone '+url.replace('bk://', 'http://')+' '+os.path.join(packages,self.package))
              except RuntimeError, e:
                status = 1
              if status:
                self.framework.log.write('Cannot bk clone:'+'\n'+output+'\n'+error+'\n')
                raise RuntimeError(failedmessage)
          self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir(0))
          return

    for url in self.download:
      if url.startswith('http://') or url.startswith('ftp://'):      
        import urllib
        tarname   = self.name+'.tar'
        tarnamegz = tarname+'.gz'
        self.framework.log.write('Downloading '+url+' to '+os.path.join(packages,self.package)+'\n')
        try:
          urllib.urlretrieve(url, os.path.join(packages,tarnamegz ))
        except Exception, e:
          failedmessage = 'Unable to download '+self.package+'\n'+\
                   'You may be off the network. Connect to the internet and run config/configure.py again \n'+\
                   'or put in the directory externalpackages the uncompressed untared file obtained \nfrom '+url+'\n'
          raise RuntimeError(failedmessage)
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
    packages = self.petscdir.externalPackagesDir
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    Dir = None
    for dir in os.listdir(packages):
      if dir.startswith(self.downloadname) and os.path.isdir(os.path.join(packages, dir)):
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
    self.compilers.CPPFLAGS += ' '.join([self.headers.getIncludeArgument(inc) for inc in self.include])
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
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      incl += self.compilers.fincs
      self.framework.log.write('Checking for library in '+location+': '+str(lib)+'\n')
      if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs, 'fortranMangle' : self.functionsFortran}):
        self.lib = lib	
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.headers.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          if self.executeTest(self.checkSharedLibrary):
            self.found   = 1
            self.dlib    = self.lib+libs
            if not hasattr(self.framework, 'packages'):
              self.framework.packages = []
            self.directory = dir
            self.framework.packages.append(self)
            break
    if not self.found:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

  def alternateConfigureLibrary(self):
    '''Called if --with-packagename=0; does nothing by default'''
    pass

  def configure(self):
    '''Determines if the package should be configured for, then calls the configure'''
    if self.download and self.framework.argDB['download-'+self.package]:
      self.framework.argDB['with-'+self.package] = 1
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
      self.executeTest(self.configureLibrary)
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return

