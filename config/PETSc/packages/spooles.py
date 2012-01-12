import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/spooles-2.2-dec-2008.tar.gz']
    self.functions  = ['InpMtx_init']
    self.includes   = ['MPI/spoolesMPI.h']
    self.liblist    = [['libspooles.a']]
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'Make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +' '+self.headers.toString(self.mpi.include)+'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS     = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.close()
    if self.installNeeded('Make.inc'):
      try:
        self.logPrintBox('Compiling spooles; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make lib', timeout=2500, log = self.framework.log)
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && cp -f *.h '+self.installDir+'/include && HLISTS=`ls *.h|egrep -v \(SPOOLES\|cfiles\|timings\)` && for hlist in $HLISTS MPI.h; do dir=`echo ${hlist} | sed s/"\.h"//`; mkdir -p '+self.installDir+'/include/$dir; cp -f $dir/*.h '+self.installDir+'/include/$dir/.; done && cp -f libspooles.a '+self.installDir+'/lib', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPOOLES: '+str(e))
      self.postInstall(output1+err1+output2+err2,'Make.inc')
    return self.installDir
