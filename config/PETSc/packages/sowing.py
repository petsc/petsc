import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/sowing/sowing-1.1.15.tar.gz']
    self.complex          = 1
    self.double           = 0;
    self.requires32bitint = 0;
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('SOWING', '--download-sowing-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for sowing configure'))
    help.addArgument('SOWING', '--download-sowing-cxx=<prog>',                    nargs.Arg(None, None, 'CXX compiler for sowing configure'))
    help.addArgument('SOWING', '--download-sowing-cpp=<prog>',                    nargs.Arg(None, None, 'CPP for sowing configure'))
    help.addArgument('SOWING', '--download-sowing-cxxcpp=<prog>',                 nargs.Arg(None, None, 'CXX CPP for sowing configure'))
    help.addArgument('SOWING', '--download-sowing-configure-options=<options>',   nargs.Arg(None, None, 'additional options for sowing configure'))
    return

  def Install(self):
    import os
    args = ['--prefix='+self.installDir]
    if 'download-sowing-cc' in self.framework.argDB and self.framework.argDB['download-sowing-cc']:
      args.append('CC="'+self.framework.argDB['download-sowing-cc']+'"')
    if 'download-sowing-cxx' in self.framework.argDB and self.framework.argDB['download-sowing-cxx']:
      args.append('CXX="'+self.framework.argDB['download-sowing-cxx']+'"')
    if 'download-sowing-cpp' in self.framework.argDB and self.framework.argDB['download-sowing-cpp']:
      args.append('CPP="'+self.framework.argDB['download-sowing-cpp']+'"')
    if 'download-sowing-cxxcpp' in self.framework.argDB and self.framework.argDB['download-sowing-cxxcpp']:
      args.append('CXXCPP="'+self.framework.argDB['download-sowing-cxxcpp']+'"')
    if 'download-sowing-configure-options' in self.framework.argDB and self.framework.argDB['download-sowing-configure-options']:
      args.append(self.framework.argDB['download-sowing-configure-options'])
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Sowing (install manually): '+str(e))
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Sowing (install manually): '+str(e))
      self.framework.actions.addArgument('Sowing', 'Install', 'Installed Sowing into '+self.installDir)
    self.binDir   = os.path.join(self.installDir, 'bin')
    self.bfort    = os.path.join(self.binDir, 'bfort')
    self.doctext  = os.path.join(self.binDir, 'doctext')
    self.mapnames = os.path.join(self.binDir, 'mapnames')
    # bug does not install bib2html so use original location if needed
    if os.path.isfile(os.path.join(self.binDir, 'bib2html')):
      self.bib2html = os.path.join(self.binDir, 'bib2html')
    else:
      self.bib2html = os.path.join(self.packageDir,'bin', 'bib2html')
    for prog in [self.bfort, self.doctext, self.mapnames]:
      if not (os.path.isfile(prog) and os.access(prog, os.X_OK)):
        raise RuntimeError('Error in Sowing installation: Could not find '+prog)
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir,'sowing')+' '+self.confDir+'/sowing', timeout=5, log = self.framework.log)
    self.addMakeMacro('BFORT ', self.bfort)
    self.addMakeMacro('DOCTEXT ', self.doctext)
    self.addMakeMacro('MAPNAMES ', self.mapnames)
    self.addMakeMacro('BIB2HTML ', self.bib2html)    
    self.getExecutable('pdflatex', getFullPath = 1)
    return self.installDir

  def buildFortranStubs(self):
    if hasattr(self.compilers, 'FC'):
      if self.framework.argDB['with-batch'] and not hasattr(self,'bfort'):
        self.logPrintBox('Batch build that could not generate bfort, skipping generating Fortran stubs\n \
                          you will need to copy them from some other system (src/fortran/auto)')
      else:     
        self.framework.log.write('           Running '+self.bfort+' to generate fortran stubs\n')
        try:
          import os,sys
          sys.path.insert(0, os.path.abspath(os.path.join('bin','maint')))
          import generatefortranstubs
          del sys.path[0]
          generatefortranstubs.main(self.bfort)
          self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs')
        except RuntimeError, e:
          raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload(1)

  def configure(self):
    '''Determine whether the Sowing exist or not'''

    # If download option is specified always build sowing
    if self.framework.argDB['download-sowing'] == 'ifneeded':
      self.framework.argDB['download-sowing'] = 0

    if self.framework.argDB['download-sowing']:
      PETSc.package.NewPackage.configure(self)
      if self.petscdir.isClone:
        self.framework.logPrint('PETSc clone, Building FortranStubs [with download-sowing=1]\n')
        self.buildFortranStubs()
      else:
        self.framework.logPrint('Not a clone, skipping FortranStubs [with download-sowing=1]\n')
      return

    # autodetect if sowing/bfort is required
    if self.petscdir.isClone:
      self.framework.logPrint('PETSc clone, checking for Sowing or if it is needed\n')

      self.getExecutable('bfort', getFullPath = 1)
      self.getExecutable('doctext', getFullPath = 1)
      self.getExecutable('mapnames', getFullPath = 1)
      self.getExecutable('bib2html', getFullPath = 1)
      self.getExecutable('pdflatex', getFullPath = 1)

      if hasattr(self, 'bfort'):
        self.framework.logPrint('Found bfort, not installing sowing')
      else:
        self.framework.logPrint('Bfort not found. Installing sowing for FortranStubs')
        self.framework.argDB['download-sowing'] = 1
        PETSc.package.NewPackage.configure(self)
      self.buildFortranStubs()
    else:
      self.framework.logPrint("Not a clone of PETSc, don't need Sowing\n")
    return

