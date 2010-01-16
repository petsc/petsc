import PETSc.package
import os

#
#   See the comment for qd.py
#
class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpack-0.6.4.tar.gz']
    self.functions        = []
#  cannot list self.includes because these are C++ include files, but test is done with C compiler
#    self.includes         = ['mblas_dd.h']
    self.liblist          = [['libmblas_dd.a']]
    self.needsMath        = 1
    self.complex          = 1
    self.cxx              = 0
    self.double           = 0      
    self.requires32bitint = 1;    
    self.include         = [os.path.abspath(os.path.join('include', 'mpack'))]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.qd            = framework.require('PETSc.packages.qd',self)
    self.gmp           = framework.require('PETSc.packages.gmp',self)    
    self.deps          = [self.qd,self.gmp]
    self.headers       = framework.require('config.headers',           self)
    self.libraries     = framework.require('config.libraries',         self)
    return

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('Cxx')
    args.append('--prefix='+self.installDir)
    args.append('CXX="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    args.append('FC=0')
    args.append('--with-qd-includedir='+self.qd.includeDir)
    args.append('--with-qd-libdir='+self.qd.libDir)
    args.append('--with-gmp-includedir="'+self.gmp.include[0]+'"')
    # following is trashy, need a way to properly pull out the library directory from the list
    args.append('--with-gmp-libdir="'+ self.gmp.lib[0][:-8]+'"')
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'mpack'), 'w')
    fd.write(args)
    fd.close()

    # mpack ./configure doesn't properly propogate the --with-xxx-yy flags to the makefiles so put them in manually
    includes = '-I'+self.qd.includeDir+' -I'+self.gmp.include[0]
    libs = '-L'+self.qd.libDir+' -L'+self.gmp.lib[0][:-8]
    FLAGS = 'CXXFLAGS="'+includes+'" ; export CXXFLAGS;  CPPFLAGS="'+includes+'" ; export CPPFLAGS  ; CFLAGS="'+includes+'" ; export CFLAGS ; LDFLAGS="'+libs+'"; export LDFLAGS'

    if self.installNeeded('mpack'):
      try:
        self.logPrintBox('Configuring MPACK; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';'+FLAGS+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on MPACK: '+str(e))
      try:
        self.logPrintBox('Compiling MPACK; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';'+FLAGS+'; make ; make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on MPACK: '+str(e))
      self.postInstall(output1+err1+output2+err2,'mpack')
    return self.installDir

