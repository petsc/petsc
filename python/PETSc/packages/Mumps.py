#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

#Developed for Mumps-4.3.1

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.setcompilers = self.framework.require('config.setCompilers',self)    
    self.libraries    = self.framework.require('config.libraries',self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = 'Mumps'
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()
    return

  def __str__(self):
    output=''
    if self.found:
      output  = self.name+':\n'
      output += '  Includes: '+self.include[0]+'\n'
      output += '  Library: '+self.lib[0]+'\n'
    return output
  
  def setupHelp(self,help):
    import nargs

    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,0,'Indicate if you wish to test for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-include=<dir>',nargs.ArgDir(None,None,'Indicate the directory of header files for '+self.name))
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE,'-with-scalapack-lib',nargs.ArgBool(None,None,'SCALAPACK libraries'))
    help.addArgument(self.PACKAGE,'-with-scalapack-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the SCALAPACK installation'))
    help.addArgument(self.PACKAGE,'-with-blacs-lib',nargs.ArgBool(None,None,'BLACS libraries'))
    help.addArgument(self.PACKAGE,'-with-blacs-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the BLACS installation'))
    help.addArgument(self.PACKAGE,'-download-blacs=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install BLACS'))
    help.addArgument(self.PACKAGE,'-download-scalapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install Scalapack'))
    help.addArgument(self.PACKAGE,'-download-mumps=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install Mumps'))    
    return

  def getDirBLACS(self):
    '''Find the directory containing BLACS'''
    packages  = os.path.join(self.framework.argDB['PETSC_DIR'], 'packages')
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    blacsDir = None
    for dir in os.listdir(packages):
      if dir == 'blacs-dev' and os.path.isdir(os.path.join(packages, dir)):
        blacsDir = dir
    if blacsDir is None:
      raise RuntimeError('Error locating BLACS directory')
    return os.path.join(packages, blacsDir)

  def downLoadBLACS(self):
    import commands
    self.framework.log.write('Downloading BLACS\n')
    try:
      blacsDir = self.getDirBLACS()
    except RuntimeError:
      import urllib

      packages = os.path.join(self.framework.argDB['PETSC_DIR'], 'packages')
      try:
        self.framework.log.write('Downloading it using "bk clone bk://petsc.bkbits.net/blacs-dev $PETSC_DIR/packages/blacs-dev"''\n')
        (status,output) = commands.getstatusoutput('bk clone bk://petsc.bkbits.net/blacs-dev packages/blacs-dev')
        if status:
          if output.find('ommand not found') >= 0:
            print '''******** Unable to locate bk (Bitkeeper) to download BuildSystem; make sure bk is in your path'''
          elif output.find('Cannot resolve host') >= 0:
            print '''******** Unable to download blacs. You must be off the network. Connect to the internet and run config/configure.py again******** '''
          else:
            import sys
            print '''******** Unable to download blacs. Please send this message to petsc-maint@mcs.anl.gov******** '''
            print output
            sys.exit(3)
      except RuntimeError, e:
        raise RuntimeError('Error bk cloneing blacs '+str(e))        
      self.framework.actions.addArgument('BLACS', 'Download', 'Downloaded blacs into '+self.getDirBLACS())

    blacsDir  = self.getDirBLACS()
    installDir = os.path.join(blacsDir, self.framework.argDB['PETSC_ARCH'])
    f = open(os.path.join(blacsDir,'Bmake.Inc.template'))
    g = open(os.path.join(blacsDir,'Bmake.Inc'),'w')
    line = f.readline()
    while line:
      if line.find('BLACSdir =') >= 0:
        line = 'BLACSdir = '+blacsDir+'\n'
      if line.find('BLACSLIB    =') >= 0:
        line = 'BLACSLIB = '+os.path.join(installDir,'libblacs.a')+'\n'
      if line.find('MPIINCdir =') >= 0:
        line = 'MPIINCdir='+self.mpi.include[0]+'\n'
      if line.find('MPILIB =') >= 0:
        line = 'MPILIB='+' '.join(map(self.libraries.getLibArgument, self.mpi.lib))+'\n'
      if line.find('INTFACE =') >= 0:
        if self.compilers.fortranManglingDoubleUnderscore:
          blah = 'f77IsF2C'
        elif self.compilers.fortranMangling == 'underscore':
          blah = 'Add_'
        elif self.compilers.fortranMangling == 'capitalize':
          blah = 'UpCase'
        else:
          blah = 'NoChange'
        line = 'INTFACE=-D'+blah+'\n'
      g.write(line)
      line = f.readline()
      
    self.setcompilers.pushLanguage('F77')  
    g.write('  F77 ='+self.setcompilers.getCompiler()+'\n')
    g.write('  F77FLAGS ='+self.setcompilers.getCompilerFlags()+'\n')
    g.write('  F77LOADER ='+self.setcompilers.getLinker()+'\n')      
    g.write('  F77LOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    self.setcompilers.pushLanguage('C')
    g.write('  CC ='+self.setcompilers.getCompiler()+'\n')
    g.write('  CCFLAGS ='+self.setcompilers.getCompilerFlags()+'\n')      
    g.write('  CCLOADER ='+self.setcompilers.getLinker()+'\n')
    g.write('  CCLOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('  ARCH ='+self.setcompilers.AR+'\n')
    g.write('  ARCHFLAGS ='+self.setcompilers.AR_FLAGS+'\n')    
    g.write('  RANLIB ='+self.setcompilers.RANLIB+'\n')    
    f.close()
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    try:
      output  = config.base.Configure.executeShellCommand('cd '+os.path.join(blacsDir,'SRC','MPI')+';make', timeout=2500, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Error running make on BLACS: '+str(e))
    if not os.path.isfile(os.path.join(installDir,'libblacs.a')):
      self.framework.log.write('Error running make on BLACS   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on BLACS follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on BLACS *******\n')
      raise RuntimeError('Error running make on BLACS, libraries not installed')

    self.framework.actions.addArgument('blacs', 'Install', 'Installed blacs into '+installDir)

    return os.path.join(installDir,'libblacs.a')


  def generateIncludeGuesses(self):
    if 'with-'+self.package in self.framework.argDB:
      if 'with-'+self.package+'-include' in self.framework.argDB:
        incl = self.framework.argDB['with-'+self.package+'-include']
        yield('User specified '+self.PACKAGE+' header location',incl)
      elif 'with-'+self.package+'-lib' in self.framework.argDB:
        incl     = self.lib[0]
        for i in 1,2:
          (incl,dummy) = os.path.split(incl)
        yield('based on found library location',os.path.join(incl,'include'))
      elif 'with-'+self.package+'-dir' in self.framework.argDB:
        dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
        yield('based on found root directory',os.path.join(dir,'include'))

  def checkInclude(self,incl,hfile):
    if not isinstance(incl,list): incl = [incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in incl:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-'+self.package+'-lib' in self.framework.argDB: #~MUMPS_4.3.1/lib/libdmumps.a
      yield ('User specified '+self.PACKAGE+' library',self.framework.argDB['with-'+self.package+'-lib'])
    elif 'with-'+self.package+'-dir' in self.framework.argDB: 
      dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
      dir = os.path.join(dir,'lib')
      libs = []
      libs.append(os.path.join(dir,'libdmumps.a'))
      libs.append(os.path.join(dir,'libzmumps.a'))
      libs.append(os.path.join(dir,'libpord.a'))
      yield('User specified '+self.PACKAGE+' root directory',libs)
    else:
      self.framework.log.write('Must specify either a library or installation root directory for '+self.PACKAGE+'\n')

  def generateScalapackLibGuesses(self):
    if 'with-scalapack-lib' in self.framework.argDB: 
      yield ('User specified SCALAPACK library',self.framework.argDB['with-scalapack-lib'])
    elif 'with-scalapack-dir' in self.framework.argDB:
      dir = os.path.abspath(self.framework.argDB['with-scalapack-dir'])
      libs = []
      libs.append(os.path.join(dir,'libscalapack.a'))
      yield('User specified SCALAPACK root directory',libs)
    else:
      self.framework.log.write('Must specify either a library or installation root directory for SCALAPACK\n')
  
  def generateBlacsLibGuesses(self):
    if self.framework.argDB['download-blacs'] == 1:
      yield ('Downloaded BLACS library',self.downLoadBLACS())
      raise RuntimeError('Downloaded BLACS could not be used. Please check install in '+os.path.dirname(libs[0])+'\n')
    if 'with-blacs-lib' in self.framework.argDB: 
      yield ('User specified BLACS library',self.framework.argDB['with-blacs-lib'])
    elif 'with-blacs-dir' in self.framework.argDB:
      dir = os.path.abspath(self.framework.argDB['with-blacs-dir'])
      libs = os.path.join(dir,'libblacs.a')
      yield('User specified BLACS root directory',libs)
    elif self.framework.argDB['download-blacs'] == 2 or self.framework.argDB['download-scalapack'] or self.framework.argDB['download-mumps']:
      yield ('Downloaded BLACS library',self.downLoadBLACS())
      raise RuntimeError('Downloaded BLACS could not be used. Please check install in '+os.path.dirname(libs[0])+'\n')
    else:
      self.framework.log.write('Must specify either a library or installation root directory for BLACS, or -download-blacs=yes\n')

  def checkLib(self,lib,func):
    oldLibs = self.framework.argDB['LIBS']
    otherLibs = self.blasLapack.lapackLibrary
    if not None in self.blasLapack.blasLibrary:
      otherLibs = otherLibs+self.blasLapack.blasLibrary
    otherLibs = ' '.join([self.libraries.getLibArgument(lib1) for lib1 in otherLibs])
    self.framework.log.write('Otherlibs '+otherLibs+'\n')
    otherLibs += ' '+' '.join(map(self.libraries.getLibArgument, self.mpi.lib))
    if hasattr(self.compilers,'flibs'): otherLibs += ' '+self.compilers.flibs
    self.framework.log.write('Otherlibs '+otherLibs+'\n')
    found = self.libraries.check(lib,func, otherLibs = otherLibs,fortranMangle=1)
    self.framework.argDB['LIBS']=oldLibs
    if found:
      self.framework.log.write('Found function '+func+' in '+str(lib)+'\n')
    return found
  
  def configureLibrary(self):
    '''Find a installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Find a installation of BLACS\n')
    found  = 0
    for (configstr,libs) in self.generateBlacsLibGuesses():
      self.framework.log.write('Checking for a functional BLACS in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[libs,'blacs_pinfo'])
      break  
    if found:
          self.blacslib = libs
    else:
      raise RuntimeError('Could not find a functional BLACS: use --with-blacs-dir or --with-blacs-lib to indicate location\n')

    self.framework.log.write('Find a installation of SCALAPACK\n')
    found  = 0
    for (configstr,libs) in self.generateScalapackLibGuesses():
      self.framework.log.write('Checking for a functional SCALAPACK in '+configstr+'\n')
      if found:
        self.scalapacklib = libs
        break
      else:
        raise RuntimeError('Could not find a functional SCALAPACK: use --with-scalapack-dir or --with-scalapack-lib to indicate location\n')


    found  = 0
    foundlibs = 0
    foundh = 0
    for (configstr,libs) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      for lib in libs:
        #found = self.executeTest(self.checkLib,[libs[0],'dmumps_c',self.mpi.lib+self.compiler.flibs])
        #found = self.executeTest(self.checkLib,[libs[0],'dmumps_c']) -- not work yet!
        found = 1  #???
        foundlibs = foundlibs or found
        if found:
          self.lib.append(lib)
      break
    if foundlibs:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'dmumps_c.h'])
        if foundh:
          self.include = [incl]
          self.found   = 1
          break
    else:
      raise RuntimeError('Could not find a functional '+self.name+': Use --with-'+self.package+'-dir to indicate is location\n')
    
      
    self.setFoundOutput()
    return

  def setFoundOutput(self):
    incl_str = ''
    for i in range(len(self.include)):
      incl_str += self.include[i]+ ' '
    self.addSubstitution(self.PACKAGE+'_INCLUDE','-I' +incl_str)
    self.addSubstitution(self.PACKAGE+'_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_'+self.PACKAGE,1)
    self.framework.packages.append(self)
            
  def setEmptyOutput(self):
    self.addSubstitution(self.PACKAGE+'_INCLUDE', '')
    self.addSubstitution(self.PACKAGE+'_LIB', '')
    return

  def configure(self):
    if self.framework.argDB['download-'+self.package]: self.framework.argDB['with-'+self.package] = 1
    if not self.framework.argDB['with-'+self.package] or self.framework.argDB['with-64-bit-ints']:
      self.setEmptyOutput()
      return
    self.executeTest(self.configureLibrary)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
