#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for the Superlu_3.0

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'Superlu'
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
    help.addArgument(self.PACKAGE, '-download-superlu=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Superlu'))
    return

  def checkInclude(self,incl,hfile):
    if not isinstance(incl,list): incl = [incl]
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl])
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.compilers.CPPFLAGS = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-superlu-lib' in self.framework.argDB and 'with-superlu-dir' in self.framework.argDB:
      raise RuntimeError('You cannot give BOTH Superlu library with --with-superlu-lib=<lib> and search directory with --with-superlu-dir=<dir>')
    if self.framework.argDB['download-superlu'] == 1:
      (name, lib) = self.downloadSuperlu()
      yield (name, lib)
      raise RuntimeError('Downloaded Superlu could not be used. Please check install in '+os.path.dirname(lib[0][0])+'\n')
    
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-lib' in self.framework.argDB: #~SuperLU_3.0/superlu_linux.a
        yield ('User specified '+self.PACKAGE+' library',self.framework.argDB['with-'+self.package+'-lib'])
      elif 'with-'+self.package+'-include' in self.framework.argDB:
        dir = self.framework.argDB['with-'+self.package+'-include'] #~SuperLU_3.0/SRC
        (dir,dummy) = os.path.split(dir)
        yield('User specified '+self.PACKAGE+'/Include',os.path.join(dir,'superlu_linux.a'))
      elif 'with-'+self.package+'-dir' in self.framework.argDB: 
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('User specified '+self.PACKAGE+' root directory',os.path.join(dir,'superlu_linux.a'))
      else:
        self.framework.log.write('Must specify either a library or installation root directory for '+self.PACKAGE+'\n')
    # If necessary, download Superlu
    if not self.found and self.framework.argDB['download-superlu'] == 2:
      (name, lib) = self.downloadSuperlu()
      yield (name, lib)
      raise RuntimeError('Downloaded Superlu could not be used. Please check in install in '+os.path.dirname(lib[0][0])+'\n')
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
    '''Find the directory containing Superlu'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    superluDir = None
    for dir in os.listdir(packages):
      if dir.startswith('SuperLU_3.0') and os.path.isdir(os.path.join(packages, dir)):
        superluDir = dir
    if superluDir is None:
      self.framework.logPrint('Could not locate already downloaded Superlu')
      raise RuntimeError('Error locating Superlu directory')
    return os.path.join(packages, superluDir)

  def downloadSuperlu(self):
    self.framework.logPrint('Downloading Superlu')
    try:
      superluDir = self.getDir()
      self.framework.logPrint('Superlu already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.logPrint("Retrieving Superlu; this may take several minutes\n", debugSection='screen')
        urllib.urlretrieve('http://crd.lbl.gov/~xiaoye/SuperLU/superlu_3.0.tar.gz', os.path.join(packages, 'superlu_3.0.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Superlu: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip superlu_3.0.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping superlu_3.0.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf superlu_3.0.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf superlu_3.0.tar: '+str(e))
      os.unlink(os.path.join(packages, 'superlu_3.0.tar'))
      self.framework.actions.addArgument('Superlu', 'Download', 'Downloaded Superlu into '+self.getDir())
      
    # Get the Superlu directories
    superluDir = self.getDir()
    if not os.path.isdir(superluDir):
      os.mkdir(superluDir)
    # Configure and Build Superlu
    self.framework.pushLanguage('C')
    args = ['--prefix='+superluDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    if not 'FC' in self.framework.argDB:
      args.append('--with-fc=0')
    if not self.framework.argDB['with-shared']:
      args.append('--with-shared=0')      
    argsStr = ' '.join(args)
    try:
      fd         = file(os.path.join(superluDir,'config.args'))
      oldArgsStr = fd.readline()
      fd.close()
    except:
      oldArgsStr = ''
    if not oldArgsStr == argsStr:
      self.framework.log.write('Have to rebuild Superlu oldargs = '+oldArgsStr+' new args '+argsStr+'\n')
      self.logPrint("Configuring and compiling Superlu; this may take several minutes\n", debugSection='screen')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(superluDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(superluDir, 'build.log'), 'w')
        oldLevel = self.argDB['debugLevel']
        #self.argDB['debugLevel'] = 0
        oldIgnore = self.argDB['ignoreCompileOutput']
        #self.argDB['ignoreCompileOutput'] = 1
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')

        self.executeShellCommand('cp MAKE_INC/make.inc .')
        self.executeShellCommand('make install lib')
        self.argDB['ignoreCompileOutput'] = oldIgnore
        self.argDB['debugLevel'] = oldLevel
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Superlu: '+str(e))
      fd = file(os.path.join(superluDir,'config.args'), 'w')
      fd.write(argsStr)
      fd.close()
      self.framework.actions.addArgument('Superlu', 'Install', 'Installed Superlu into '+superluDir)
    lib     = os.path.join(superluDir, 'superlu_linux.a')  
    return ('Downloaded Superlu', lib)  
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found  = 0
    foundh = 0
    for (configstr,lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[lib,'set_default_options'])  
      if found:
        self.lib = [lib]
        break
    if found:
      incl = self.lib[0]
      (incl,dummy) = os.path.split(incl)
      incl = os.path.join(incl,'SRC')
      foundh = self.executeTest(self.checkInclude,[incl,'dsp_defs.h'])
      if foundh:
        self.include = [incl]
        self.found   = 1
        self.setFoundOutput()
      else:
        self.framework.log.write('Could not find a functional '+self.name+'\n')
    else:
      self.framework.log.write('Could not find a functional '+self.name+'\n')
    return

  def setFoundOutput(self):
    self.framework.packages.append(self)
    

  def configure(self):
    if (self.framework.argDB['with-superlu'] or self.framework.argDB['download-superlu'] == 1):
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
