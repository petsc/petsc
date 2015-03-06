import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.giturls           = ['https://bitbucket.org/petsc/pkg-sowing.git']
    self.gitcommit         = '9c5b20c'
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/sowing-1.1.17-p1.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.requires32bitint  = 0
    self.downloadonWindows = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0  # sowing does not support make -j np
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('SOWING', '-download-sowing-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxx=<prog>',                    nargs.Arg(None, None, 'CXX compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cpp=<prog>',                    nargs.Arg(None, None, 'CPP for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxxcpp=<prog>',                 nargs.Arg(None, None, 'CXX CPP for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-configure-options=<options>',   nargs.Arg(None, None, 'additional options for sowing configure'))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.petscdir       = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.petscclone     = framework.require('PETSc.options.petscclone',self.setCompilers)
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.confDir]
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
    return args

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def configure(self):
    if (self.framework.clArgDB.has_key('with-sowing') and not self.framework.argDB['with-sowing']) or \
          (self.framework.clArgDB.has_key('download-sowing') and not self.framework.argDB['download-sowing']):
      self.framework.logPrint("Not checking sowing on user request\n")
      return

    if self.petscclone.isClone and hasattr(self.compilers, 'FC'):
      self.framework.logPrint('PETSc clone, checking for Sowing \n')
      self.getExecutable('pdflatex', getFullPath = 1)

      if self.framework.clArgDB.has_key('with-sowing-dir') and self.framework.argDB['with-sowing-dir']:
        installDir = os.path.join(self.framework.argDB['with-sowing-dir'],'bin')

        self.getExecutable('bfort',    path=installDir, getFullPath = 1)
        self.getExecutable('doctext',  path=installDir, getFullPath = 1)
        self.getExecutable('mapnames', path=installDir, getFullPath = 1)
        self.getExecutable('bib2html', path=installDir, getFullPath = 1)
        if hasattr(self, 'bfort'):
          self.framework.logPrint('Found bfort in user provided directory, not installing sowing')
          self.found = 1
        else:
          raise RuntimeError('You passed --with-sowing-dir='+installDir+' but it does not contain sowings bfort')

      else:
        self.getExecutable('bfort', getFullPath = 1)
        self.getExecutable('doctext', getFullPath = 1)
        self.getExecutable('mapnames', getFullPath = 1)
        self.getExecutable('bib2html', getFullPath = 1)

        if hasattr(self, 'bfort') and not self.framework.argDB['download-sowing']:
          self.framework.logPrint('Found bfort, not installing sowing')
        else:
          self.framework.logPrint('Bfort not found. Installing sowing for FortranStubs')
          if (not self.framework.argDB['download-sowing']):  self.framework.argDB['download-sowing'] = 1
          config.package.GNUPackage.configure(self)

          installDir = os.path.join(self.installDir,'bin')
          self.getExecutable('bfort',    path=installDir, getFullPath = 1)
          self.getExecutable('doctext',  path=installDir, getFullPath = 1)
          self.getExecutable('mapnames', path=installDir, getFullPath = 1)
          self.getExecutable('bib2html', path=installDir, getFullPath = 1)

      self.buildFortranStubs()
    else:
      self.framework.logPrint("Not a clone of PETSc or no Fortran, don't need Sowing\n")
    return

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
          generatefortranstubs.main(self.petscdir.dir, self.bfort, self.petscdir.dir,0)
          if self.compilers.fortranIsF90:
            generatefortranstubs.processf90interfaces(self.petscdir.dir,0)
          self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs')
        except RuntimeError, e:
          raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return
