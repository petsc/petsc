#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the Spooles-2.2

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'Spooles'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    output=''
    if self.found:
      output  = self.name+':\n'
      output += '  Includes: '+ str(self.include)+'\n'
      output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def setupHelp(self,help):
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,0,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE, '-download-spooles=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Spooles'))
    return

  def generateIncludeGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-include' in self.framework.argDB:
        incl = self.framework.argDB['with-'+self.package+'-include']
        yield('User specified '+self.PACKAGE+' header location',incl)
      elif 'with-'+self.package+'-lib' in self.framework.argDB:
        incl         = self.lib[1]  #=spooles-2.2/spooles.a
        (incl,dummy) = os.path.split(incl)
        yield('based on found library location',incl)
      elif 'with-'+self.package+'-dir' in self.framework.argDB:
        incl = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('based on found root directory',incl)
    return

  def checkInclude(self,incl,hfile):
    if not isinstance(incl,list): incl = [incl]
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl+self.mpi.include])              
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.compilers.CPPFLAGS = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-spooles-lib' in self.framework.argDB and 'with-spooles-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Spooles library with --with-spooles-lib=<lib> and search directory with --with-spooles-dir=<dir>')
    if self.framework.argDB['download-spooles'] == 1:
      (name, lib_mpi, lib) = self.downloadSpooles()
      yield (name, lib_mpi, lib)
      raise RuntimeError('Downloaded Spooles could not be used. Please check install in '+os.path.dirname(lib[0][0])+'\n')
    
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-lib' in self.framework.argDB: #~spooles-2.2/MPI/src/spoolesMPI.a ~spooles-2.2/spooles.a
        lib = self.framework.argDB['with-'+self.package+'-lib']
        (lib_mpi,dummy) = os.path.split(lib)
        lib_mpi = os.path.join(lib_mpi,'MPI/src/spoolesMPI.a')
        yield ('User specified '+self.PACKAGE+' library',lib_mpi,lib)
      elif 'with-'+self.package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+self.package+'-include'] #~spooles-2.2
        lib = os.path.join(dir,'spooles.a')
        lib_mpi = os.path.join(dir,'MPI/src/spoolesMPI.a')
        yield('User specified '+self.PACKAGE+' directory of header files',lib_mpi,lib)
      elif 'with-'+self.package+'-dir' in self.framework.argDB: 
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir']) #~spooles-2.2
        lib = os.path.join(dir,'spooles.a')
        lib_mpi = os.path.join(dir,'MPI/src/spoolesMPI.a')
        yield('User specified '+self.PACKAGE+' root directory',lib_mpi,lib)
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+self.PACKAGE+'\n')
    # If necessary, download Spooles
    if not self.found and self.framework.argDB['download-spooles'] == 2:
      (name, lib_mpi, lib) = self.downloadSpooles()
      yield (name, lib_mpi, lib)
      raise RuntimeError('Downloaded Spooles could not be used. Please check in install in '+os.path.dirname(lib[0][0])+'\n')
    return
        
  def checkLib(self,lib,libfile):
    if not isinstance(lib,list): lib = [lib]
    oldLibs = self.framework.argDB['LIBS']  
    found = self.libraries.check(lib,libfile)
    self.framework.argDB['LIBS']=oldLibs  
    if found:
      self.framework.log.write('Found functional '+libfile+' in '+lib[0]+'\n')
    return found

  def getDir(self):
    '''Find the directory containing Spooles'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    spoolesDir = None
    for dir in os.listdir(packages):
      if dir.startswith('spooles-2.2') and os.path.isdir(os.path.join(packages, dir)):
        spoolesDir = dir
    if spoolesDir is None:
      self.framework.logPrint('Could not locate already downloaded Spooles')
      raise RuntimeError('Error locating Spooles directory')
    return os.path.join(packages, spoolesDir)

  def downloadSpooles(self):
    self.framework.logPrint('Downloading Spooles')
    try:
      spoolesDir = self.getDir()
      self.framework.logPrint('Spooles already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.logPrint("Retrieving Spooles; this may take several minutes\n", debugSection='screen')
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/spooles-2.2.tar.gz', os.path.join(packages, 'spooles-2.2.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Spooles: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip spooles-2.2.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping spooles-2.2.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf spooles-2.2.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf spooles-2.2.tar: '+str(e))
      os.unlink(os.path.join(packages, 'spooles-2.2.tar'))
      self.framework.actions.addArgument('Spooles', 'Download', 'Downloaded Spooles into '+self.getDir())
      
    # Get the Spooles directories
    spoolesDir = self.getDir()
    if not os.path.isdir(spoolesDir):
      os.mkdir(spoolesDir)
    # Configure and Build Spooles
    self.framework.pushLanguage('C')
    args = ['--prefix='+spoolesDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    if not 'FC' in self.framework.argDB:
      args.append('--with-fc=0')
    if not self.framework.argDB['with-shared']:
      args.append('--with-shared=0')      
    argsStr = ' '.join(args)
    try:
      fd         = file(os.path.join(spoolesDir,'config.args'))
      oldArgsStr = fd.readline()
      fd.close()
    except:
      oldArgsStr = ''
    if not oldArgsStr == argsStr:
      self.framework.log.write('Have to rebuild Spooles oldargs = '+oldArgsStr+' new args '+argsStr+'\n')
      self.logPrint("Configuring and compiling Spooles; this may take several minutes\n", debugSection='screen')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(spoolesDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(spoolesDir, 'build.log'), 'w')
        oldLevel = self.argDB['debugLevel']
        #self.argDB['debugLevel'] = 0
        oldIgnore = self.argDB['ignoreCompileOutput']
        #self.argDB['ignoreCompileOutput'] = 1
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')

        self.executeShellCommand('make lib')
        self.executeShellCommand('cd MPI/src; make spoolesMPI.a')
        self.argDB['ignoreCompileOutput'] = oldIgnore
        self.argDB['debugLevel'] = oldLevel
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Spooles: '+str(e))
      fd = file(os.path.join(spoolesDir,'config.args'), 'w')
      fd.write(argsStr)
      fd.close()
      self.framework.actions.addArgument('Spooles', 'Install', 'Installed Spooles into '+spoolesDir)
    lib_mpi = os.path.join(spoolesDir,'MPI/src/spoolesMPI.a')
    lib     = os.path.join(spoolesDir, 'spooles.a')
    return ('Downloaded Spooles', lib_mpi, lib)  
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found     = 0
    foundh    = 0
    for (configstr,lib_mpi,lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[lib,'InpMtx_init'])  
      if found:
        self.lib = [lib_mpi,lib]
        break
    if found:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'MPI/spoolesMPI.h'])
        if foundh:
          self.include = [incl]
          self.found   = 1
          self.setFoundOutput()
          break
    else:
      raise RuntimeError('Could not find a functional '+self.name+'\n')
    return

  def setFoundOutput(self):
    self.framework.packages.append(self)
    
  def configure(self):
    #if self.framework.argDB['with-'+self.package]:
    if (self.framework.argDB['with-spooles'] or self.framework.argDB['download-spooles'] == 1):
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.framework.argDB['with-64-bit-ints']:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')   
      self.executeTest(self.configureLibrary)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
