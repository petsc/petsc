import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version     = '3.83'
    self.versionname = 'ZOLTAN_VERSION_NUMBER'
    self.download    = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/zoltan_distrib_v'+self.version+'.tar.gz']
    self.functions   = ['Zoltan_LB_Partition']
    self.includes    = ['zoltan.h']
    self.liblist     = [['libzoltan.a']]
    self.license     = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.parmetis = framework.require('config.packages.parmetis',self)
    self.ptscotch = framework.require('config.packages.PTScotch',self)
    self.mpi      = framework.require('config.packages.MPI',self)
    self.mathlib  = framework.require('config.packages.mathlib',self)
    self.deps     = [self.mpi,self.mathlib]
    self.odeps    = [self.parmetis, self.ptscotch]

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.getDefaultIndexSize() == 64:
      args.append('--with-id-type=ullong')
    args.append('--enable-mpi')
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: Zoltan requires C++ compiler. None specified')

    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')
    if hasattr(self.compilers, 'FC'):
      args.append('--enable-f90interface')
      args.append('FCFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    if self.parmetis.found:
      args.append('--with-parmetis')
    if self.ptscotch.found:
      args.append('--with-scotch')
    return args

  def Install(self):
    '''Zoltan does not have a make clean'''
    packageDir = os.path.join(self.packageDir,'petsc-build')
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile):
      return self.installDir

    output1,err1,ret1  = config.base.Configure.executeShellCommand('rm -rf '+packageDir+' &&  mkdir '+packageDir, timeout=2000, log = self.log)
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+packageDir+' && ../configure '+args, timeout=2000, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')

      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+packageDir+' && '+self.make.make+' everything', timeout=6000, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+packageDir+' && '+self.make.make+' install', timeout=300, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3, conffile)
    return self.installDir

