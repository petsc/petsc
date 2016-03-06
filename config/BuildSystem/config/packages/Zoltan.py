import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/zoltan_distrib_v3.81.tar.gz']
    self.functions = ['Zoltan_LB_Partition']
    self.includes  = ['zoltan.h']
    self.liblist   = [['libzoltan.a']]
    self.license   = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.x              = framework.require('config.packages.X',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.ptscotch       = framework.require('config.packages.PTScotch',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.deps           = [self.mpi, self.parmetis, self.ptscotch]

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.indexTypes.integerSize == 64:
      args.append('--with-id-type=ullong')
    args.append('--enable-mpi')
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: Zoltan requires C++ compiler. None specified')

    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      args.append('--with-mpi-incdir="'+self.mpi.include[0]+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-mpi-libdir="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-mpi-libs="'+libs+'"')
    if self.parmetis.found:
      args.append('--with-parmetis')
      args.append('--with-parmetis-incdir="'+self.parmetis.include[0]+'"')
      args.append('--with-parmetis-libdir="'+self.libraries.toString(self.parmetis.lib)+'"')
    if self.ptscotch.found:
      args.append('--with-scotch')
      args.append('--with-scotch-incdir="'+self.ptscotch.include[0]+'"')
      args.append('--with-scotch-libdir="'+self.libraries.toString(self.ptscotch.lib)+'"')
    return args

  def Install(self):
    '''Zoltan does not have a make clean'''
    packageDir = os.path.join(self.packageDir,'build')
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile):
      return self.installDir

    output1,err1,ret1  = config.base.Configure.executeShellCommand('rm -rf '+packageDir+' &&  mkdir '+packageDir, timeout=2000, log = self.log)
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+packageDir+' && ../configure '+args, timeout=2000, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')

      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+packageDir+' && '+self.make.make+' everything', timeout=6000, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(self.installSudo)
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+packageDir+' && '+self.installSudo+self.make.make+' install', timeout=300, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3, conffile)
    return self.installDir

