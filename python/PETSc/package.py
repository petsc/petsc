from __future__ import generators
import config.base
import os
import re
import sys


class Package(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSc'
    self.substPrefix  = 'PETSc'
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.clanguage    = self.framework.require('PETSc.utilities.clanguage',self)    
    self.functions    = self.framework.require('config.functions',self)
    self.source       = self.framework.require('config.sourceControl',self)    
    self.found        = 0
    self.lib          = []
    # this packages libraries and all those it depends on
    self.dlib         = []
    self.include      = []
    self.name         = os.path.splitext(os.path.basename(sys.modules.get(self.__module__).__file__))[0]
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    # these are optional items set in the particular packages file
    self.complex      = 0
    self.download     = []
    self.deps         = []
    self.functions    = []
    self.includes     = []
    
  def __str__(self):
    '''Prints the location of the packages includes and libraries'''
    output=''
    if self.found:
      output  = self.name+':\n'
      if self.include: output += '  Includes: '+str(self.include)+'\n'
      if self.lib:     output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def setupHelp(self,help):
    '''Prints help messages for the package'''
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,0,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    if self.download:
      help.addArgument(self.PACKAGE, '-download-'+self.package+'=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Download and install '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of the '+self.name+' include files'))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<dir,or list of libraries>',nargs.ArgDir(None,None,'Indicate the directory of the '+self.name+' libraries or a list of libraries'))    
    return

  def generateGuesses(self):
    if self.download and self.framework.argDB['download-'+self.package] == 1:
      dir = os.path.join(self.Install(),self.framework.argDB['PETSC_ARCH'])
      yield('Download '+self.PACKAGE,self.generateLibList(os.path.join(dir,'lib')) ,os.path.join(dir,'include'))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+dir+'\n')
    if 'with-'+self.package+'-dir' in self.framework.argDB:     
      dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
      yield('User specified '+self.PACKAGE+' root directory',self.generateLibList(os.path.join(dir,'lib')),os.path.join(dir,'include'))
    if 'with-'+self.package+'-include' in self.framework.argDB and 'with-'+self.package+'-lib' in self.framework.argDB:
      dir1 = os.path.abspath(self.framework.argDB['with-'+self.package+'-lib'])
      if os.path.isdir(dir1): libs = self.generateLibList(dir1)
      else: libs = dir1
      if not isinstance(libs, list): libs = [libs]
      dir2 = os.path.abspath(self.framework.argDB['with-'+self.package+'-include'])      
      yield('User specified '+self.PACKAGE+' root directory',libs,dir2)
    if self.download and self.framework.argDB['download-'+self.package] == 2:
      dir = os.path.join(self.Install(),self.framework.argDB['PETSC_ARCH'])
      yield('Download '+self.PACKAGE,self.generateLibList(os.path.join(dir,'lib')) ,os.path.join(dir,'include'))
      raise RuntimeError('Downloaded '+self.package+' could not be used. Please check install in '+dir+'\n')
    raise RuntimeError('You must specifiy a path for '+self.name+' with --with-'+self.package+'-dir=<directory>')

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
          self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir())
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
        self.framework.actions.addArgument(self.PACKAGE, 'Download', 'Downloaded '+self.package+' into '+self.getDir())
        return
    raise RuntimeError('Unable to download '+self.package+' from locations '+str(self.download)) 

  def getDir(self):
    '''Find the directory containing the package'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    Dir = None
    for dir in os.listdir(packages):
      if dir.startswith(self.name) and os.path.isdir(os.path.join(packages, dir)):
        Dir = dir
        break
    if Dir is None:
      self.framework.log.write('Did not located already downloaded '+self.name+'\n')
      self.downLoad()
      return self.getDir()
    return os.path.join(packages, Dir)

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
      if hasattr(l,'dlib'):    libs  += l.dlib
      if hasattr(l,'include'): incls += l.include
      
    for location, lib,incl in self.generateGuesses():
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for library '+location+': '+str(lib)+'\n')
      if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs}):      
        self.lib = lib
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.libraries.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
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
      if self.framework.argDB['with-64-bit-ints']:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')    
      if not self.clanguage.precision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')    
      if not self.complex and self.clanguage.scalartype.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')    
      self.executeTest(self.configureLibrary)
    return

