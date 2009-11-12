import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://gforge.inria.fr/frs/download.php/10140/pastix_release_1789.tar.bz2']
    self.downloadname = self.name.lower()
    self.liblist      = [['libpastix.a'],
                         ['libpastix.a','libpthread.a','librt.a']]
    self.functions    = ['pastix']
    self.includes     = ['pastix.h']
    self.complex      = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.scotch     = framework.require('PETSc.packages.Scotch',self)
    self.make       = framework.require('PETSc.utilities.Make', self)
    self.deps       = [self.mpi, self.blasLapack, self.scotch]   
    return
  
  def Install(self):
    import os
    self.logPrintBox('Creating Pastix '+os.path.join(os.path.join(self.packageDir,'src'),'config.in')+'\n')
    # Patching csc_utils for memcpy error
    g = open(os.path.join(os.path.join(self.packageDir,'src'),'patch.memcpy'),'w')
    g.write("284c284,285\n")
    g.write("<       memcpy((*newa) , a , l * sizeof(FLOAT)); \n")
    g.write("---\n")
    g.write(">       if (a != NULL)\n")
    g.write(">       memcpy((*newa) , a , l * sizeof(FLOAT)); \n")
    g.close();

    output  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+';patch sopalin/src/csc_utils.c < patch.memcpy', timeout=2500, log = self.framework.log)[0]
    
    g = open(os.path.join(os.path.join(self.packageDir,'src'),'config.in'),'w')

    g.write('HOSTHARCH   = i686_pc_linux\n')
    g.write('VERSIONBIT  = _32bit\n')
    g.write('EXEEXT      = \n')
    g.write('OBJEXT      = .o\n')
    g.write('LIBEXT      = .'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CCPROG      = '+self.setCompilers.getCompiler()+'\n')
    # common.c tries to use some silly clock_gettime() routine that Mac doesn't have unless this is set
    if self.setCompilers.isDarwin():    
      cflags = ' -DX_ARCHi686_mac    '
    else: cflags = ''
    g.write('CCFOPT      = '+self.setCompilers.getCompilerFlags()+' '+self.headers.toString(self.mpi.include)+' '+cflags+'\n')
    self.setCompilers.popLanguage()
    if not self.compilers.fortranIsF90:
      raise RuntimeError('Installing PaStiX requires a F90 compiler') 
    self.setCompilers.pushLanguage('FC') 
    g.write('CFPROG      = '+self.setCompilers.getCompiler()+'\n')
    g.write('CF90PROG    = '+self.setCompilers.getCompiler()+'\n')
    g.write('MCFPROG     = '+self.setCompilers.getCompiler()+'\n')
    g.write('CF90CCPOPT  = '+ self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('\n')
    g.write('LKFOPT      =\n')
    g.write('MKPROG      = '+self.make.make+'\n')
    g.write('MPCCPROG    = '+self.setCompilers.getCompiler()+'\n')
    g.write('ARFLAGS     = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('ARPROG      = '+self.setCompilers.AR+'\n')
    extralib = ''
    if self.libraries.add('-lm','sin'): extralib += ' -lm'
    if self.libraries.add('-lrt','timer_create'): extralib += ' -lrt'
    
    g.write('EXTRALIB    = '+extralib+' \n')
    g.write('\n')
    g.write('VERSIONMPI  = _mpi\n')
    g.write('VERSIONSMP  = _smp\n')
    g.write('VERSIONBUB  = _nobubble\n')
    g.write('VERSIONINT  = _int\n')
    g.write('VERSIONPRC  = _simple\n')
    g.write('VERSIONFLT  = _real\n')
    g.write('VERSIONORD  = _scotch\n')
    g.write('\n')
    ###################################################################
    #                          INTEGER TYPE                           #
    ###################################################################
    g.write('\n')
    if self.libraryOptions.integerSize == 64:
      g.write('#---------------------------\n')
      g.write('VERSIONINT  = _int64\n')
      g.write('CCTYPES     = -DFORCE_INT64\n')
      g.write('\n')
    else:
      g.write('#---------------------------\n')
      g.write('VERSIONINT  = _int32\n')
      g.write('CCTYPES     = -DFORCE_INT32\n')
      g.write('\n')
    ###################################################################
    #                           FLOAT TYPE                            #
    ###################################################################
    if self.scalartypes.precision == 'double':
      g.write('VERSIONPRC  = _double\n')
      g.write('CCTYPES    := $(CCTYPES) -DFORCE_DOUBLE\n')
      g.write('\n')
    g.write('# uncomment the following lines for float=complex support\n')
    g.write('#VERSIONFLT  = _complex\n')
    g.write('#CCTYPES    := $(CCTYPES) -DFORCE_COMPLEX\n')
    g.write('\n')
    g.write('\n')
    g.write('###################################################################\n')
    g.write('#                          MPI/THREADS                            #\n')
    g.write('###################################################################\n')
    g.write('\n')
    g.write('# uncomment the following lines for sequential (NOMPI) version\n')
    g.write('#VERSIONMPI  = _nompi\n')
    g.write('#CCTYPES    := $(CCTYPES) -DFORCE_NOMPI\n')
    g.write('#MPCCPROG    = $(CCPROG)\n')
    g.write('#MCFPROG     = $(CFPROG)\n')
    g.write('\n')
    g.write('# uncomment the following lines for non-threaded (NOSMP) version\n')
    g.write('#VERSIONSMP  = _nosmp\n')
    g.write('#CCTYPES    := $(CCTYPES) -DFORCE_NOSMP\n')
    g.write('\n')
    g.write('# Uncomment the following line to enable a progression thread\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DTHREAD_COMM\n')
    g.write('\n')
    g.write('# Uncomment the following line if your MPI doesn\'t support MPI_THREAD_MULTIPLE leve\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DPASTIX_FUNNELED\n')
    g.write('\n')
    g.write('# Uncomment the following line if your MPI doesn\'t support MPI_Datatype correctly\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DNO_MPI_TYPE\n')
    g.write('\n')
    g.write('###################################################################\n')
    g.write('#                          Options                                #\n')
    g.write('###################################################################\n')
    g.write('\n')
    g.write('# Uncomment the following lines for NUMA-aware allocation (recommended)\n')
    g.write('CCPASTIX   := $(CCPASTIX) -DNUMA_ALLOC\n')
    g.write('\n')
    g.write('# Show memory usage statistics\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DMEMORY_USAGE\n')
    g.write('\n')
    g.write('# Show memory usage statistics in solver\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DSTATS_SOPALIN\n')
    g.write('\n')
    g.write('# Uncomment following line for dynamic thread scheduling support\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DPASTIX_BUBBLE\n')
    g.write('\n')
    g.write('# Uncomment the following lines for Out-of-core\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DOOC\n')
    g.write('\n')
    g.write('###################################################################\n')
    g.write('#                      GRAPH PARTITIONING                         #\n')
    g.write('###################################################################\n')
    g.write('\n')
    g.write('# uncomment the following lines for using metis ordering\n')
    g.write('#VERSIONORD  = _metis\n')
    g.write('#METIS_HOME  = ${HOME}/metis-4.0\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DMETIS -I$(METIS_HOME)/Lib\n')
    g.write('#EXTRALIB   := $(EXTRALIB) -L$(METIS_HOME) -lmetis\n')
    g.write('\n')
    g.write('# Scotch always needed to compile\n')
    g.write('#scotch								\n')
    g.write('CCPASTIX   := $(CCPASTIX) '+self.headers.toString(self.scotch.include)+'\n')
    g.write('EXTRALIB   := $(EXTRALIB) '+self.libraries.toString(self.scotch.dlib)+'\n')
    g.write('\n')
    g.write('###################################################################\n')
    g.write('#                             MARCEL                              #\n')
    g.write('###################################################################\n')
    g.write('\n')
    g.write('# Uncomment following lines for marcel thread support\n')
    g.write('#VERSIONSMP := $(VERSIONSMP)_marcel\n')
    g.write('#CCPASTIX   := $(CCPASTIX) `pm2-config --cflags` -I${PM2_ROOT}/marcel/include/pthre\n')
    g.write('#EXTRALIB   := $(EXTRALIB) `pm2-config --libs`\n')
    g.write('# ---- Thread Posix ------\n')
    g.write('EXTRALIB   := $(EXTRALIB) -lpthread\n')
    g.write('\n')
    g.write('# Uncomment following line for bubblesched framework support (need marcel support)\n')
    g.write('#VERSIONBUB  = _bubble\n')
    g.write('#CCPASTIX   := $(CCPASTIX) -DPASTIX_USE_BUBBLE\n')
    g.write('\n')
    ###################################################################
    #                              BLAS                               #
    ###################################################################
    g.write('BLASLIB  = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('###################################################################\n')
    g.write('#                          DO NOT TOUCH                           #\n')
    g.write('###################################################################\n')
    g.write('\n')
    g.write('FOPT      := $(CCFOPT)\n')
    g.write('FDEB      := $(CCFDEB)\n')
    g.write('CCHEAD    := $(CCPROG) $(CCTYPES) $(CCFOPT)\n')
    g.write('CCFOPT    := $(CCFOPT) $(CCTYPES) $(CCPASTIX)\n')
    g.write('CCFDEB    := $(CCFDEB) $(CCTYPES) $(CCPASTIX)\n')
    g.close();

    if self.installNeeded(os.path.join('src','config.in')):
      try:
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+';make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling PaStiX; this may take several minutes')
        output = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+'; make expor install',timeout=2500, log = self.framework.log)[0]
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        output = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'; cp -f install/*.a '+libDir+'/.; cp -f install/*.h '+includeDir+'/.;', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PaStiX: '+str(e))
      self.postInstall(output,os.path.join('src','config.in'))
    return self.installDir
