import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    #self.download          = ['http://www.columbia.edu/~ma2325/Prometheus-1.8.10.tar.gz']
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/Prometheus-1.8.10.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = [['libpromfei.a','libprometheus.a']]
    self.compilePrometheus = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.parmetis   = framework.require('PETSc.packages.parmetis',self)
    self.deps       = [self.parmetis, self.mpi, self.blasLapack]
    return

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but Prometheus requires the extra C++ library'''
    alllibs = PETSc.package.NewPackage.generateLibList(self,dir)
    import config.setCompilers
    if self.languages.clanguage == 'C':
      alllibs[0].extend(self.compilers.cxxlibs)
    return alllibs

  def Install(self):
    import os

    args = ''
    args += 'SHELL          = '+self.programs.SHELL+'\n'
    args += 'CP             = '+self.programs.cp+'\n'
    args += 'RM             = '+self.programs.RM+'\n'
    args += 'MKDIR          = '+self.programs.mkdir+'\n'


    args += 'PREFIX         = '+self.installDir+'\n'
    args += 'BUILD_DIR      = '+self.packageDir+'\n'
    args += 'LIB_DIR        = $(BUILD_DIR)/lib/\n'
    args += 'PETSC_INCLUDE  = -I'+os.path.join(self.petscdir.dir,self.arch,'include')+' -I'+os.path.join(self.petscdir.dir)+' -I'+os.path.join(self.petscdir.dir,'include')+' '+self.headers.toString(self.mpi.include+self.parmetis.include)+'\n'
    args += 'RANLIB         = '+self.setCompilers.RANLIB+'\n'
    args += 'AR             = '+self.setCompilers.AR+'\n'
    args += 'ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n'
    args += 'PROM_LIB       = libprometheus.'+ self.setCompilers.AR_LIB_SUFFIX+'\n'
    args += 'FEI_LIB        = libpromfei.'+ self.setCompilers.AR_LIB_SUFFIX+'\n'


    self.framework.pushLanguage('C++')
    args += 'CXX            = '+self.framework.getCompiler()
    args += ' -DPROM_HAVE_METIS'

    if self.blasLapack.mangling == 'underscore':
      args += ' -DHAVE_FORTRAN_UNDERSCORE=1'
      if self.compilers.fortranManglingDoubleUnderscore:
        args += ' -DHAVE_FORTRAN_UNDERSCORE_UNDERSCORE=1'
    elif self.blasLapack.mangling == 'unchanged':
      args += ' -DHAVE_FORTRAN_NOUNDERSCORE=1'
    elif self.blasLapack.mangling == 'caps':
      args += ' -DHAVE_FORTRAN_CAPS=1'
    elif self.blasLapack.mangling == 'stdcall':
      args += ' -DHAVE_FORTRAN_STDCALL=1'
      args += ' -DSTDCALL=__stdcall'
      args += ' -DHAVE_FORTRAN_CAPS=1'
      args += ' -DHAVE_FORTRAN_MIXED_STR_ARG=1'

    args += '\n'
    args += 'PETSCFLAGS     = '+self.framework.getCompilerFlags()+'\n'
    self.framework.popLanguage()
    fd = file(os.path.join(self.packageDir,'makefile.petsc'),'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('makefile.petsc'):
      self.framework.logClear()
      self.logPrint("**************************************************************************************************", debugSection='screen')
      self.logPrint('Prometheus is scheduled to be decommissioned.  Interested parties are encouraged to switch to the ', debugSection='screen')
      self.logPrint('native AMG implementation GAMG using -pc_type gamg -pc_gamg_type agg to get smoothed aggregaton as', debugSection='screen')
      self.logPrint('is implemented in Prometheus.  GAMG provides almost all of the functionality of Prometheus and    ', debugSection='screen')
      self.logPrint('should have better performance.  We are actively developing GAMG and welcome requests for new     ', debugSection='screen')
      self.logPrint('functionality and reports of any performance problems.                                            ', debugSection='screen')
      self.logPrint("**************************************************************************************************\n", debugSection='screen')
      fd = file(os.path.join(self.packageDir,'makefile.in'),'a')
      fd.write('include makefile.petsc\n')
      fd.close()
      self.compilePrometheus = 1
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      # Prometheus requires LAPACK routine dorgqr()
      if not self.blasLapack.checkForRoutine('dorgqr'):
        raise RuntimeError('Prometheus requires the LAPACK routine dorgqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nIf you are using the IBM ESSL library, it does not contain this function. After installing a complete copy of lapack\n You can run ./configure with --with-blas-lib=libessl.a --with-lapack-lib=/usr/local/lib/liblapack.a')
      self.framework.log.write('Found dorgqr() in Lapack library as needed by Prometheus\n')
    return

  def postProcess(self):
    if self.compilePrometheus:
      try:
        self.logPrintBox('Compiling Prometheus; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean cleanlib && make prom minstall',timeout=1000, log = self.framework.log)
        self.framework.log.write(output)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Prometheus: '+str(e))
      self.postInstall(output+err,'makefile.petsc')
