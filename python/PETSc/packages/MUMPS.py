#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import md5

#Developed for Mumps-4.3.1

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.setcompilers = self.framework.require('config.setCompilers',self)    
    self.libraries    = self.framework.require('config.libraries',self)
    self.arch         = self.framework.require('PETSc.utilities.arch',self)
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
    help.addArgument(self.PACKAGE,'-with-'+self.package+'-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the '+self.name+' installation'))
    help.addArgument(self.PACKAGE,'-with-scalapack-lib',nargs.ArgBool(None,None,'SCALAPACK libraries'))
    help.addArgument(self.PACKAGE,'-with-scalapack-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the SCALAPACK installation'))
    help.addArgument(self.PACKAGE,'-with-blacs-lib',nargs.ArgBool(None,None,'BLACS libraries'))
    help.addArgument(self.PACKAGE,'-with-blacs-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root directory of the BLACS installation'))
    help.addArgument(self.PACKAGE,'-download-blacs=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install BLACS'))
    help.addArgument(self.PACKAGE,'-download-scalapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install Scalapack'))
    help.addArgument(self.PACKAGE,'-download-mumps=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Install Mumps'))    
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

  def getDirBLACS(self):
    '''Find the directory containing BLACS'''
    packages  = self.framework.argDB['with-external-packages-dir']
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

      packages = self.framework.argDB['with-external-packages-dir']
      try:
        self.framework.log.write('Downloading it using "bk clone bk://petsc.bkbits.net/blacs-dev '+os.path.join(packages,'blacs-dev')+'"\n')
        (status,output) = commands.getstatusoutput('bk clone bk://petsc.bkbits.net/blacs-dev '+os.path.join(packages,'blacs-dev'))
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
    installDir = os.path.join(blacsDir, self.arch.arch)
    g = open(os.path.join(blacsDir,'Bmake.Inc'),'w')
    g.write('SHELL = /bin/sh\n')
    g.write('COMMLIB = MPI\n')
    g.write('SENDIS = -DSndIsLocBlk\n')
    g.write('WHATMPI = -DUseF77Mpi\n')
    g.write('DEBUGLVL = -DBlacsDebugLvl=1\n')
    g.write('BLACSdir = '+blacsDir+'\n')
    g.write('BLACSLIB = '+os.path.join(installDir,'libblacs.a')+'\n')
    g.write('MPIINCdir='+self.mpi.include[0]+'\n')
    g.write('MPILIB='+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SYSINC = -I$(MPIINCdir)\n')
    g.write('BTLIBS = $(BLACSLIB)  $(MPILIB) \n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('INTFACE=-D'+blah+'\n')
    g.write('DEFS1 = -DSYSINC $(SYSINC) $(INTFACE) $(DEFBSTOP) $(DEFCOMBTOP) $(DEBUGLVL)\n')
    g.write('BLACSDEFS = $(DEFS1) $(SENDIS) $(BUFF) $(TRANSCOMM) $(WHATMPI) $(SYSERRORS)\n')
    self.setcompilers.pushLanguage('FC')  
    g.write('F77 ='+self.setcompilers.getCompiler()+'\n')
    g.write('F77FLAGS ='+self.setcompilers.getCompilerFlags()+'\n')
    g.write('F77LOADER ='+self.setcompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    self.setcompilers.pushLanguage('C')
    g.write('CC ='+self.setcompilers.getCompiler()+'\n')
    g.write('CCFLAGS ='+self.setcompilers.getCompilerFlags()+'\n')      
    g.write('CCLOADER ='+self.setcompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('ARCH ='+self.setcompilers.AR+'\n')
    g.write('ARCHFLAGS ='+self.setcompilers.AR_FLAGS+'\n')    
    g.write('RANLIB ='+self.setcompilers.RANLIB+'\n')    
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Bmake.Inc')) or not (self.getChecksum(os.path.join(installDir,'Bmake.Inc')) == self.getChecksum(os.path.join(blacsDir,'Bmake.Inc'))):
      try:
        self.logPrint("Compiling Blacs; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(blacsDir,'SRC','MPI')+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLACS: '+str(e))
    else:
      self.framework.log.write('Do NOT need to compile BLACS downloaded libraries\n')
    if not os.path.isfile(os.path.join(installDir,'libblacs.a')):
      self.framework.log.write('Error running make on BLACS   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on BLACS follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on BLACS *******\n')
      raise RuntimeError('Error running make on BLACS, libraries not installed')
    try:
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(blacsDir,'Bmake.Inc')+' '+installDir, timeout=5, log = self.
framework.log)[0]
    except RuntimeError, e:
      pass
    self.framework.actions.addArgument('blacs', 'Install', 'Installed blacs into '+installDir)
    return os.path.join(installDir,'libblacs.a')

  def getDirSCALAPACK(self):
    '''Find the directory containing SCALAPACK'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    scalapackDir = None
    for dir in os.listdir(packages):
      if dir.startswith('SCALAPACK') and os.path.isdir(os.path.join(packages, dir)):
        scalapackDir = dir
    if scalapackDir is None:
      raise RuntimeError('Error locating SCALAPACK directory')
    return os.path.join(packages, scalapackDir)

  def downLoadSCALAPACK(self):
    import commands
    self.framework.log.write('Downloading SCALAPACK\n')
    try:
      scalapackDir = self.getDirSCALAPACK()
    except RuntimeError:
      import urllib
      packages = self.framework.argDB['with-external-packages-dir']
      try:
        urllib.urlretrieve('http://www.netlib.org/scalapack/scalapack.tgz', os.path.join(packages, 'scalapack.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading SCALAPACK: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip scalapack.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping scalapack.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf scalapack.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf scalapack.tar: '+str(e))
      os.unlink(os.path.join(packages, 'scalapack.tar'))
      self.framework.actions.addArgument('SCALAPACK', 'Download', 'Downloaded scalapack into '+self.getDirSCALAPACK())

    scalapackDir  = self.getDirSCALAPACK()
    installDir = os.path.join(scalapackDir, self.arch.arch)
    g = open(os.path.join(scalapackDir,'SLmake.inc'),'w')
    g.write('SHELL = /bin/sh\n')
    g.write('home = '+self.getDirSCALAPACK()+'\n')    
    g.write('USEMPI        = -DUsingMpiBlacs\n')
    g.write('SENDIS = -DSndIsLocBlk\n')
    g.write('WHATMPI = -DUseF77Mpi\n')
    g.write('BLACSDBGLVL = -DBlacsDebugLvl=1\n')
    g.write('BLACSLIB = '+self.libraries.toString(self.blacslib)+'\n')
    g.write('SMPLIB='+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SCALAPACKLIB  = '+os.path.join('$(home)',self.arch.arch,'libscalapack.a')+' \n')
    g.write('CBLACSLIB     = $(BLACSCINIT) $(BLACSLIB) $(BLACSCINIT)\n')
    g.write('FBLACSLIB     = $(BLACSFINIT) $(BLACSLIB) $(BLACSFINIT)\n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('CDEFS=-D'+blah+' -DUsingMpiBlacs\n')
    g.write('PBLASdir      = $(home)/PBLAS\n')
    g.write('SRCdir        = $(home)/SRC\n')
    g.write('TOOLSdir      = $(home)/TOOLS\n')
    g.write('REDISTdir     = $(home)/REDIST\n')
    self.setcompilers.pushLanguage('FC')  
    g.write('F77 ='+self.setcompilers.getCompiler()+'\n')
    g.write('F77FLAGS ='+self.setcompilers.getCompilerFlags()+'\n')
    g.write('F77LOADER ='+self.setcompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    self.setcompilers.pushLanguage('C')
    g.write('CC ='+self.setcompilers.getCompiler()+'\n')
    g.write('CCFLAGS ='+self.setcompilers.getCompilerFlags()+'\n')      
    g.write('CCLOADER ='+self.setcompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('ARCH ='+self.setcompilers.AR+'\n')
    g.write('ARCHFLAGS ='+self.setcompilers.AR_FLAGS+'\n')    
    g.write('RANLIB ='+self.setcompilers.RANLIB+'\n')    
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'SLmake.inc')) or not (self.getChecksum(os.path.join(installDir,'SLmake.inc')) == self.getChecksum(os.path.join(scalapackDir,'SLmake.inc'))):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrint("Compiling Scalapack; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SCALAPACK: '+str(e))
    else:
      self.framework.log.write('Did not need to compile downloaded SCALAPACK\n')
    if not os.path.isfile(os.path.join(installDir,'libscalapack.a')):
      self.framework.log.write('Error running make on SCALAPACK   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on SCALAPACK follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on SCALAPACK *******\n')
      raise RuntimeError('Error running make on SCALAPACK, libraries not installed')
    try:
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(scalapackDir,'SLmake.inc')+' '+installDir, timeout=5, log = self.
framework.log)[0]
    except RuntimeError, e:
      pass
    self.framework.actions.addArgument('scalapack', 'Install', 'Installed SCALAPACK into '+installDir)
    return os.path.join(installDir,'libscalapack.a')

  def generateIncludeGuesses(self):
    if 'with-'+self.package+'-dir' in self.framework.argDB:
      dir = os.path.abspath(self.framework.argDB['with-'+self.package+'-dir'])
      yield('based on found root directory',os.path.join(dir,'include'))

  def checkInclude(self,incl,hfile):
    if not isinstance(incl,list): incl = [incl]
    incl.extend(self.mpi.include)
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in incl])
    found = self.checkPreprocess('#include <' +hfile+ '>\n')
    self.compilers.CPPFLAGS = oldFlags
    if found:
      self.framework.log.write('Found header file ' +hfile+ ' in '+incl[0]+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-'+self.package+'-dir' in self.framework.argDB: 
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
    if self.framework.argDB['download-scalapack'] == 1:
      yield ('Downloaded SCALAPACK library',self.downLoadSCALAPACK())
      raise RuntimeError('Downloaded SCALAPACK could not be used. Please check install in '+os.path.dirname(libs[0])+'\n')
    if 'with-scalapack-lib' in self.framework.argDB: 
      yield ('User specified SCALAPACK library',self.framework.argDB['with-scalapack-lib'])
    elif 'with-scalapack-dir' in self.framework.argDB:
      dir = os.path.abspath(self.framework.argDB['with-scalapack-dir'])
      libs = []
      libs.append(os.path.join(dir,'libscalapack.a'))
      yield('User specified SCALAPACK root directory',libs)
    elif self.framework.argDB['download-scalapack'] == 2 or self.framework.argDB['download-mumps']:
      yield ('Downloaded SCALAPACK library',self.downLoadSCALAPACK())
      raise RuntimeError('Downloaded BLACS could not be used. Please check install in '+os.path.dirname(libs[0])+'\n')
    else:
      self.framework.log.write('Must specify either a library or installation root directory for SCALAPACK, or -download-scalapack=yes\n')
  
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

  def checkLib(self,lib,func,mangle,otherLibs = []):
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
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
      found = self.executeTest(self.checkLib,[libs,'blacs_pinfo',1])
      break  
    if found:
      if not isinstance(libs,list): self.blacslib = [libs]
      else: self.blacslib = libs
    else:
      raise RuntimeError('Could not find a functional BLACS: use --with-blacs-dir or --with-blacs-lib to indicate location\n')

    self.framework.log.write('Find a installation of SCALAPACK\n')
    found  = 0
    for (configstr,libs) in self.generateScalapackLibGuesses():
      self.framework.log.write('Checking for a functional SCALAPACK in '+configstr+'\n')
      found = self.executeTest(self.checkLib,[libs,'ssytrd',1,self.blacslib])
      break
    if found:
      self.scalapacklib = libs
    else:
      raise RuntimeError('Could not find a functional SCALAPACK: use --with-scalapack-dir or --with-scalapack-lib to indicate location\n')

    foundlibs = 0
    foundh = 0
    for (configstr,libs) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional '+self.name+' in '+configstr+'\n')
      foundlibs = self.executeTest(self.checkLib,[libs,'dmumps_c',0,self.scalapacklib+self.blacslib])
      self.lib = libs+self.scalapacklib+self.blacslib
      break
    if foundlibs:
      for (inclstr,incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for headers '+inclstr+': '+incl+'\n')
        foundh = self.executeTest(self.checkInclude,[incl,'dmumps_c.h'])
        if foundh:
          self.include = [incl]
          break
      if not foundh:
        raise RuntimeError('Could not find include files '+self.name+': Use --with-'+self.package+'-dir to indicate is location\n')        
    else:
      raise RuntimeError('Could not find libraries for '+self.name+': Use --with-'+self.package+'-dir to indicate is location\n')
    self.setFoundOutput()
    self.found = 1
    return

  def setFoundOutput(self):
    self.addDefine('HAVE_'+self.PACKAGE,1)
    self.framework.packages.append(self)
            
  def configure(self):
    if self.framework.argDB['download-'+self.package]: self.framework.argDB['with-'+self.package] = 1
    if self.framework.argDB['with-'+self.package]:
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
