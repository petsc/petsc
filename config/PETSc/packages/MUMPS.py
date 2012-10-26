import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/MUMPS_4.10.0-p3.tar.gz']
    self.liblist   = [['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a'],
                     ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a','libpthread.a'],
                     ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libmumps_common.a','libpord.a','libmpiseq.a']]
    self.functions = ['dmumps_c']
    self.includes  = ['dmumps_c.h']
    #
    # Mumps does NOT work with 64 bit integers without a huge number of hacks we ain't making
    self.double    = 0
    self.complex   = 1
    self.worksonWindows   = 1
    self.downloadonWindows= 1
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('MUMPS', '-with-mumps-serial', nargs.ArgBool(None, 0, 'Use serial build of MUMPS'))
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack   = framework.require('config.packages.BlasLapack',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    if self.framework.argDB['with-mumps-serial']:
      self.deps       = [self.blasLapack]
    else:
      self.blacs      = framework.require('PETSc.packages.blacs',self)
      self.scalapack  = framework.require('PETSc.packages.SCALAPACK',self)
      self.deps       = [self.scalapack,self.blacs,self.mpi,self.blasLapack]
      self.parmetis   = framework.require('PETSc.packages.parmetis',self)
      self.ptscotch   = framework.require('PETSc.packages.PTScotch',self)
    return

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-mumps-serial']:
      if not self.mpi.usingMPIUni:
        raise RuntimeError('Serial MUMPS version is only compatible with MPIUni\nReconfigure using --with-mpi=0')
      elif self.mpi.usingMPIUniFortranBinding:
        raise RuntimeError('Serial MUMPS version is incompatible with the MPIUni Fortran bindings\nReconfigure using --with-mpiuni-fortran-binding=0')
    return

  def Install(self):
    import os

    if not hasattr(self.compilers, 'FC'):
      raise RuntimeError('Cannot install '+self.name+' without Fortran, make sure you do NOT have --with-fc=0')
    if not self.compilers.FortranDefineCompilerOption:
      raise RuntimeError('Fortran compiler cannot handle preprocessing directives from command line.')
    if self.framework.argDB['with-mumps-serial']:
      raise RuntimeError('Cannot automatically install the serial version of MUMPS.')
    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    g.write('LPORDDIR   = $(topdir)/PORD/lib/\n')
    g.write('IPORD      = -I$(topdir)/PORD/include/\n')
    g.write('LPORD      = -L$(LPORDDIR) -lpord\n')
    orderingsc = '-Dpord'
    orderingsf = self.compilers.FortranDefineCompilerOption+'pord'
    # Disable threads on BGL
    if self.libraryOptions.isBGL():
      orderingsc += ' -DWITHOUT_PTHREAD'
    if self.parmetis.found:
      g.write('IMETIS = '+self.headers.toString(self.parmetis.include)+'\n')
      g.write('LMETIS = '+self.libraries.toString(self.parmetis.lib)+'\n')
      orderingsc += ' -Dmetis -Dparmetis'
      orderingsf += ' '+self.compilers.FortranDefineCompilerOption+'metis '+self.compilers.FortranDefineCompilerOption+'parmetis'
    if self.ptscotch.found:
      g.write('ISCOTCH = '+self.headers.toString(self.ptscotch.include)+'\n')
      g.write('LSCOTCH = '+self.libraries.toString(self.ptscotch.lib)+'\n')
      orderingsc += ' -Dscotch  -Dptscotch'
      orderingsf += ' '+self.compilers.FortranDefineCompilerOption+'scotch '+self.compilers.FortranDefineCompilerOption+'ptscotch'

    g.write('ORDERINGSC = '+orderingsc+'\n')
    g.write('ORDERINGSF = '+orderingsf+'\n')
    g.write('LORDERINGS  = $(LMETIS) $(LPORD) $(LSCOTCH)\n')
    g.write('IORDERINGSC = $(IMETIS) $(IPORD) $(ISCOTCH)\n')
    g.write('IORDERINGSF = $(ISCOTCH)\n')

    g.write('RM = /bin/rm -f\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTC    = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    g.write('OUTC = -o \n')
    self.setCompilers.popLanguage()
    if not self.compilers.fortranIsF90:
      raise RuntimeError('Installing MUMPS requires a F90 compiler') 
    self.setCompilers.pushLanguage('FC') 
    g.write('FC = '+self.setCompilers.getCompiler()+'\n')
    g.write('FL = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTF    = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','').replace('-Mfree','') +'\n')
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
    g.write('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+' '+self.libraries.toString(self.blacs.lib)+'\n')
    g.write('INCPAR  = '+self.headers.toString(self.mpi.include)+'\n')
    g.write('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib)+'\n') #PARALLE LIBRARIES USED by MUMPS
    g.write('INCSEQ  = -I$(topdir)/libseq\n')
    g.write('LIBSEQ  =  $(LAPACK) -L$(topdir)/libseq -lmpiseq\n')
    g.write('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('OPTL    = -O -I.\n')
    g.write('INCS = $(INCPAR)\n')
    g.write('LIB = $(LIBPAR)\n')
    g.write('LIBSEQNEEDED =\n')
    g.close()
    if self.installNeeded('Makefile.inc'):
      try:
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Mumps; this may take several minutes')
        output2,err2,ret2 = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' &&  make alllib',timeout=2500, log = self.framework.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && mv -f lib/*.* '+libDir+'/. && cp -f include/*.* '+includeDir+'/.', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on MUMPS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'Makefile.inc')
    return self.installDir

  def configureLibrary(self):
    if not self.framework.argDB['with-mumps-serial']:
      if self.parmetis.found:
        self.deps.append(self.parmetis)
      if self.ptscotch.found:
        self.deps.append(self.ptscotch)
    PETSc.package.NewPackage.configureLibrary(self)
     # [parallem mumps] make sure either ptscotch or parmetis is enabled
    if not self.framework.argDB['with-mumps-serial'] and not self.ptscotch.found and not self.parmetis.found:
       raise RuntimeError('MUMPS requires either Parmetis or PTScotch. Use either --download-parmetis or --download-ptscotch')
