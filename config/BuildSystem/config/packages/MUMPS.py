import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '5.3.1'
    self.minversion       = '5.2.1'
    self.versionname      = 'MUMPS_VERSION'
    self.gitcommit        = 'v'+self.version+'-p1'
    self.download         = ['git://https://bitbucket.org/petsc/pkg-mumps.git',
                             'https://bitbucket.org/petsc/pkg-mumps/get/'+self.gitcommit+'.tar.gz']
    self.download_darwin  = ['https://bitbucket.org/petsc/pkg-mumps/get/v5.2.1-p2.tar.gz']
    self.downloaddirnames = ['petsc-pkg-mumps','MUMPS']
    self.liblist          = [['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a'],
                            ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a','libpthread.a'],
                            ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a','libmpiseq.a'],
                            ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a','libpthread.a','libmpiseq.a']]
    self.functions        = ['dmumps_c']
    self.includes         = ['dmumps_c.h']
    #
    self.fc               = 1
    self.precisions       = ['single','double']
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MUMPS', '-with-mumps-serial', nargs.ArgBool(None, 0, 'Use serial build of MUMPS'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.flibs            = framework.require('config.packages.flibs',self)
    self.blasLapack       = framework.require('config.packages.BlasLapack',self)
    self.mpi              = framework.require('config.packages.MPI',self)
    self.metis            = framework.require('config.packages.metis',self)
    self.parmetis         = framework.require('config.packages.parmetis',self)
    self.ptscotch         = framework.require('config.packages.PTScotch',self)
    self.scalapack        = framework.require('config.packages.scalapack',self)
    if self.argDB['with-mumps-serial']:
      self.deps           = [self.blasLapack,self.flibs]
      self.odeps          = [self.metis]
    else:
      self.deps           = [self.scalapack,self.mpi,self.blasLapack,self.flibs]
      self.odeps          = [self.metis,self.parmetis,self.ptscotch]
    self.openmp           = framework.require('config.packages.openmp',self)
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package] or self.argDB['download-'+self.package]:
      if self.mpi.usingMPIUni and not self.argDB['with-mumps-serial']:
        raise RuntimeError('Since you are building without MPI you must use --with-mumps-serial to install the correct MUMPS.')
    if self.argDB['with-mumps-serial']:
      if not self.mpi.usingMPIUni:
        raise RuntimeError('Serial MUMPS version is only compatible with MPIUni\nReconfigure using --with-mpi=0')
    return

  def Install(self):
    import os

    if self.openmp.found:
      #  MUMPS has no make flags for turning on/off OpenMP it just uses it if it can
      self.usesopenmp = 'yes'
      # use OMP_NUM_THREADS to control the number of threads used

    if not self.fortran.FortranDefineCompilerOption:
      raise RuntimeError('Fortran compiler cannot handle preprocessing directives from command line.')
    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    g.write('LPORDDIR   = $(topdir)/PORD/lib/\n')
    g.write('IPORD      = -I$(topdir)/PORD/include/\n')
    g.write('LPORD      = -L$(LPORDDIR) -lpord\n')
    g.write('PLAT       = \n')
    orderingsc = '-Dpord'
    orderingsf = self.fortran.FortranDefineCompilerOption+'pord'
    # Disable threads on BGL
    if self.libraries.isBGL():
      orderingsc += ' -DWITHOUT_PTHREAD'
    if self.metis.found:
      g.write('IMETIS = '+self.headers.toString(self.metis.include)+'\n')
      g.write('LMETIS = '+self.libraries.toString(self.metis.lib)+'\n')
      orderingsc += ' -Dmetis'
      orderingsf += ' '+self.fortran.FortranDefineCompilerOption+'metis'
    if self.parmetis.found:
      g.write('IPARMETIS = '+self.headers.toString(self.parmetis.include)+'\n')
      g.write('LPARMETIS = '+self.libraries.toString(self.parmetis.lib)+'\n')
      orderingsc += ' -Dparmetis'
      orderingsf += ' '+self.fortran.FortranDefineCompilerOption+'parmetis'
    if self.ptscotch.found:
      g.write('ISCOTCH = '+self.headers.toString(self.ptscotch.include)+'\n')
      g.write('LSCOTCH = '+self.libraries.toString(self.ptscotch.lib)+'\n')
      orderingsc += ' -Dscotch  -Dptscotch'
      orderingsf += ' '+self.fortran.FortranDefineCompilerOption+'scotch '+self.fortran.FortranDefineCompilerOption+'ptscotch'

    g.write('ORDERINGSC = '+orderingsc+'\n')
    g.write('ORDERINGSF = '+orderingsf+'\n')
    g.write('LORDERINGS  = $(LPARMETIS) $(LMETIS) $(LPORD) $(LSCOTCH)\n')
    g.write('IORDERINGSC = $(IPARMETIS) $(IMETIS) $(IPORD) $(ISCOTCH)\n')
    g.write('IORDERINGSF = $(ISCOTCH)\n')

    g.write('RM = /bin/rm -f\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTC    = ' + self.removeWarningFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.write('OUTC = -o \n')
    self.setCompilers.popLanguage()
    if not self.fortran.fortranIsF90:
      raise RuntimeError('Installing MUMPS requires a F90 compiler')
    self.setCompilers.pushLanguage('FC')
    g.write('FC = '+self.setCompilers.getCompiler()+'\n')
    g.write('FL = '+self.setCompilers.getCompiler()+'\n')
    extra_fcflags = ''
    if config.setCompilers.Configure.isNAG(self.setCompilers.getLinker(), self.log):
      extra_fcflags = '-dusty -dcfuns '
    elif config.setCompilers.Configure.isGfortran100plus(self.setCompilers.getCompiler(), self.log):
      extra_fcflags = '-fallow-argument-mismatch '
    g.write('OPTF    = '+extra_fcflags+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','').replace('-Mfree','')+'\n')
    if self.blasLapack.mkl and self.blasLapack.foundversion.isdigit() and int(self.blasLapack.foundversion) >= 110300:
      g.write('OPTF   += -DGEMMT_AVAILABLE \n')
    g.write('OUTF = -o \n')
    self.setCompilers.popLanguage()

    # set fortran name mangling
    # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag
    if self.compilers.fortranManglingDoubleUnderscore:
      g.write('CDEFS   = -DAdd__\n')
    elif self.compilers.fortranMangling == 'underscore':
      g.write('CDEFS   = -DAdd_\n')
    elif self.compilers.fortranMangling == 'caps':
      g.write('CDEFS   = -DUPPPER\n')

    g.write('AR      = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+' \n')
    g.write('LIBEXT  = .'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB  = '+self.setCompilers.RANLIB+'\n')
    g.write('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+'\n')
    if not self.argDB['with-mumps-serial']:
      g.write('INCPAR  = '+self.headers.toString(self.mpi.include)+'\n')
      g.write('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib)+'\n')
    else:
      g.write('INCPAR  = -I../libseq\n')
    g.write('INCSEQ  = -I$(topdir)/libseq\n')
    g.write('LIBSEQ  =  $(LAPACK) -L$(topdir)/libseq -lmpiseq\n')
    g.write('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('OPTL    = -O -I.\n')
    g.write('INCS = $(INCPAR)\n')
    g.write('LIBS = $(LIBPAR)\n')
    if self.argDB['with-mumps-serial']:
      g.write('LIBSEQNEEDED = libseqneeded\n')
      g.write('LIBS = $(LIBSEQ)\n')
    else:
      g.write('LIBSEQNEEDED =\n')
    g.close()
    if self.installNeeded('Makefile.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        pass
      try:
        self.logPrintBox('Compiling Mumps; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand(self.make.make_jnp+' alllib', cwd=self.packageDir, timeout=2500, log = self.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        self.logPrintBox('Installing Mumps; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommandSeq(
          [self.installSudo+'mkdir -p '+libDir+' '+includeDir,
           self.installSudo+'cp -f lib/*.* '+libDir+'/.',
           self.installSudo+'cp -f include/*.* '+includeDir+'/.'
          ], cwd=self.packageDir, timeout=60, log = self.log)
        if self.argDB['with-mumps-serial']:
          output,err,ret = config.package.Package.executeShellCommand([self.installSudo+'cp', '-f', 'libseq/libmpiseq.a', libDir+'/.'], cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on MUMPS: '+str(e))
        raise RuntimeError('Error running make on MUMPS')
      self.postInstall(output1+err1+output2+err2,'Makefile.inc')
    return self.installDir

