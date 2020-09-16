import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '5.2.3'
    self.versionname      = 'PASTIX_MAJOR_VERSION.PASTIX_MEDIUM_VERSION.PASTIX_MINOR_VERSION'
    # 'https://gforge.inria.fr/frs/download.php/file/36212/pastix_'+self.version+'.tar.bz2',
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/pastix_'+self.version+'.tar.bz2']
    self.liblist          = [['libpastix.a'],
                            ['libpastix.a','libpthread.a','librt.a']]
    self.functions        = ['pastix']
    self.includes         = ['pastix.h']
    self.precisions       = ['double']
    self.downloaddirnames = ['pastix']
    self.fc               = 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return


  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.scotch         = framework.require('config.packages.PTScotch',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.pthread        = framework.require('config.packages.pthread',self)
    # PaStiX.py does not absolutely require hwloc, but it performs better with it and can fail (in ways not easily tested) without it
    # https://gforge.inria.fr/forum/forum.php?thread_id=32824&forum_id=599&group_id=186
    # https://solverstack.gitlabpages.inria.fr/pastix/Bindings.html
    self.hwloc          = framework.require('config.packages.hwloc',self)
    self.deps           = [self.mpi, self.blasLapack, self.scotch, self.pthread, self.hwloc]
    return

  def Install(self):
    import os
    g = open(os.path.join(os.path.join(self.packageDir,'src'),'config.in'),'w')

    # pastix Makefile can pickup these variables from env - so set them manually so that env variables don't get used.
    g.write('PREFIX      = '+os.path.join(self.packageDir,'install\n'))
    g.write('INCLUDEDIR  = ${PREFIX}/include\n')
    g.write('LIBDIR      = ${PREFIX}/lib\n')
    g.write('BINDIR      = ${PREFIX}/bin\n')

    # This one should be the only one needed
    # all other tests for mac should not be useful.
    if self.setCompilers.isDarwin(self.log):
      g.write('HOSTARCH   = i686_mac\n')
    else:
      g.write('HOSTARCH   = i686_pc_linux\n')
    g.write('VERSIONBIT  = _XXbit\n')
    g.write('EXEEXT      = \n')
    g.write('OBJEXT      = .o\n')
    g.write('LIBEXT      = .'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CCPROG      = '+self.setCompilers.getCompiler()+'\n')
    # common.c tries to use some silly clock_gettime() routine that Mac doesn't have unless this is set
    if self.setCompilers.isDarwin(self.log):
      cflags = ' -DX_ARCHi686_mac    '
    else:
      cflags = ''
    if self.mpi.found:
      g.write('CCFOPT      = '+self.updatePackageCFlags(self.setCompilers.getCompilerFlags())+' '+self.headers.toString(self.mpi.include)+' '+cflags+'\n')
    else:
      g.write('CCFOPT      = '+self.updatePackageCFlags(self.setCompilers.getCompilerFlags())+' '+cflags+'\n')
    self.setCompilers.popLanguage()
    g.write('CFPROG      = \n')
    g.write('CF90PROG    = \n')
    g.write('MCFPROG     = \n')
    g.write('CF90CCPOPT  = \n')
    g.write('\n')
    g.write('LKFOPT      =\n')
    g.write('MKPROG      = '+self.make.make+'\n')
    # PaStiX make system has error where in one location it doesn't pass in CCFOTP
    if self.setCompilers.isDarwin(self.log):
      g.write('MPCCPROG    = '+self.setCompilers.getCompiler()+' -DX_ARCHi686_mac \n')
    else:
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
    if self.indexTypes.integerSize == 64:
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
    # Now PaStiX supports multi arithmetic with [sdcz]pastix calls
    g.write('#VERSIONPRC  = _double\n')
    g.write('#CCTYPES    := $(CCTYPES) -DFORCE_DOUBLE -DPREC_DOUBLE\n')
    g.write('#\n')
    g.write('# uncomment the following lines for float=complex support\n')
    g.write('#VERSIONFLT  = _complex\n')
    g.write('#CCTYPES  := $(CCTYPES) -DFORCE_COMPLEX -DTYPE_COMPLEX\n')
    g.write('\n')
    g.write('\n')
    g.write('###################################################################\n')
    g.write('#                          MPI/THREADS                            #\n')
    g.write('###################################################################\n')
    g.write('\n')
    if not self.mpi.found:
      g.write('# uncomment the following lines for sequential (NOMPI) version\n')
      g.write('VERSIONMPI  = _nompi\n')
      g.write('CCTYPES    := $(CCTYPES) -DFORCE_NOMPI\n')
      g.write('MPCCPROG    = $(CCPROG)\n')
      g.write('MCFPROG     = $(CFPROG)\n')
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
    g.write('CCPASTIX   := $(CCPASTIX) -DWITH_HWLOC '+self.headers.toString(self.hwloc.include)+'\n')
    g.write('EXTRALIB   := $(EXTRALIB) '+self.libraries.toString(self.hwloc.dlib)+'\n')
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
    if (self.mpi.found):
      g.write('CCPASTIX   := $(CCPASTIX) -DDISTRIBUTED -DWITH_SCOTCH '+self.headers.toString(self.scotch.include)+'\n')
    else:
      g.write('CCPASTIX   := $(CCPASTIX) -DWITH_SCOTCH '+self.headers.toString(self.scotch.include)+'\n')
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
    ###################################################################
    #                        MURGE COMPATIBILITY                      #
    ###################################################################

    g.write('MAKE     = $(MKPROG)\n')
    g.write('CC       = $(MPCCPROG)\n')
    if self.setCompilers.isDarwin(self.log):
      cflags = ' -DX_ARCHi686_mac    '
    else: cflags = ''
    g.write('CFLAGS   = $(CCFOPT) $(CCTYPES)'+cflags+'\n')
    g.write('FC       = $(MCFPROG)\n')
    g.write('FFLAGS   = $(CCFOPT)\n')
    g.write('LDFLAGS  = $(EXTRALIB) $(BLASLIB)\n')

    g.close()

    if self.installNeeded(os.path.join('src','config.in')):
      try:
        output,err,ret  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean', timeout=2500, log = self.log)
      except RuntimeError as e:
        pass
      try:
        self.logPrintBox('Compiling PaStiX; this may take several minutes')
        output,err,ret = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make all',timeout=2500, log = self.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        self.logPrintBox('Installing PaStiX; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+'mkdir -p '+libDir+' && '+self.installSudo+'cp -f install/*.a '+libDir+'/. && '+self.installSudo+'mkdir -p '+includeDir+' && '+self.installSudo+'cp -f install/*.h '+includeDir+'/.', timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on PaStiX: '+str(e))
      self.postInstall(output+err,os.path.join('src','config.in'))
    return self.installDir
