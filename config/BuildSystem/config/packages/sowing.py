import config.package
import os

def noCheck(command, status, output, error):
  ''' Do no check result'''
  return

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.minversion        = '1.1.26'
    self.gitcommit         = 'v'+self.minversion+'-p1'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-sowing.git','https://bitbucket.org/petsc/pkg-sowing/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['petsc-pkg-sowing']
    self.complex           = 1
    self.downloadonWindows = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0  # sowing does not support make -j np
    self.executablename    = 'bfort'
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('SOWING', '-download-sowing-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxx=<prog>',                    nargs.Arg(None, None, 'CXX compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cpp=<prog>',                    nargs.Arg(None, None, 'CPP for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxxpp=<prog>',                  nargs.Arg(None, None, 'CXX CPP for sowing configure'))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.petscclone = framework.require('PETSc.options.petscclone', None)
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.installDir]
    if 'download-sowing-cc' in self.argDB and self.argDB['download-sowing-cc']:
      args.append('CC="'+self.argDB['download-sowing-cc']+'"')
    if 'download-sowing-cxx' in self.argDB and self.argDB['download-sowing-cxx']:
      args.append('CXX="'+self.argDB['download-sowing-cxx']+'"')
    if 'download-sowing-cpp' in self.argDB and self.argDB['download-sowing-cpp']:
      args.append('CPP="'+self.argDB['download-sowing-cpp']+'"')
    if 'download-sowing-cxxpp' in self.argDB and self.argDB['download-sowing-cxxpp']:
      args.append('CXXPP="'+self.argDB['download-sowing-cxxpp']+'"')
    return args

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def checkBfortVersion(self):
    try:
      import re
      (output, error, status) = config.base.Configure.executeShellCommand(self.bfort+' -version', checkCommand=noCheck, log = self.log)
      ver = re.compile('bfort \(sowing\) release ([0-9]+).([0-9]+).([0-9]+)').match(output)
      foundversion = tuple(map(int,ver.groups()))
      self.foundversion = ".".join(map(str,foundversion))
    except RuntimeError as e:
      self.log.write(self.bfort+' version check failed: '+str(e)+'\n')
      return
    version = tuple(map(int, self.minversion.split('.')))
    if foundversion < version:
      raise RuntimeError(self.bfort+' version '+".".join(map(str,foundversion))+' is older than required '+self.minversion+'. Perhaps a stale install of sowing?')
    return

  def configure(self):
    if ('with-sowing' in self.framework.clArgDB and not self.argDB['with-sowing']):
      if hasattr(self.compilers, 'FC') and self.framework.argDB['with-fortran-bindings'] and self.petscclone.isClone:
        raise RuntimeError('Cannot use --with-sowing=0 if using Fortran (bindings) and git repository for PETSc')
      self.logPrint("Not checking sowing on user request of --with-sowing=0\n")
      return

    if self.framework.batchBodies:
      self.logPrint('In --with-batch mode with outstanding batch tests to be made; hence skipping sowing for this configure')
      return

    if (self.petscclone.isClone and hasattr(self.compilers, 'FC') and self.framework.argDB['with-fortran-bindings']) or ('download-sowing' in self.framework.clArgDB and self.argDB['download-sowing']):
      self.logPrint('PETSc clone, checking for Sowing \n')
      self.getExecutable('pdflatex', getFullPath = 1)

      if 'with-sowing-dir' in self.framework.clArgDB and self.argDB['with-sowing-dir']:
        installDir = os.path.join(self.argDB['with-sowing-dir'],'bin')

        self.getExecutable('bfort',    path=installDir, getFullPath = 1)
        self.getExecutable('doctext',  path=installDir, getFullPath = 1)
        self.getExecutable('mapnames', path=installDir, getFullPath = 1)
        self.getExecutable('bib2html', path=installDir, getFullPath = 1)
        if hasattr(self, 'bfort'):
          self.logPrint('Found bfort in user provided directory, not installing sowing')
          self.found = 1
        else:
          raise RuntimeError("You passed --with-sowing-dir='+installDir+' but it does not contain sowing's bfort")

      else:
        if not self.argDB['download-sowing']:
          self.getExecutable('bfort', getFullPath = 1)
          self.getExecutable('doctext', getFullPath = 1)
          self.getExecutable('mapnames', getFullPath = 1)
          self.getExecutable('bib2html', getFullPath = 1)

        if hasattr(self, 'bfort'):
          self.logPrint('Found bfort, not installing sowing')
          self.found = 1
        else:
          self.logPrint('Bfort not found. Installing sowing for FortranStubs')
          if (not self.argDB['download-sowing']):  self.argDB['download-sowing'] = 1
          #check cygwin has g++
          if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/g++.exe'):
            raise RuntimeError('Error! sowing on windows requires cygwin/g++. Please install it with cygwin setup.exe')
          config.package.GNUPackage.configure(self)
          installDir = os.path.join(self.installDir,'bin')
          self.getExecutable('bfort',    path=installDir, getFullPath = 1)
          self.getExecutable('doctext',  path=installDir, getFullPath = 1)
          self.getExecutable('mapnames', path=installDir, getFullPath = 1)
          self.getExecutable('bib2html', path=installDir, getFullPath = 1)
          self.found = 1
          if not hasattr(self,'bfort'): raise RuntimeError('Unable to locate bfort (part of sowing) in its expected location in '+installDir+'\n\
Perhaps the installation has been corrupted or changed, remove the directory '+os.path.join(self.petscdir.dir,self.arch)+'\n\
and run configure again\n')

      self.checkBfortVersion()
      self.buildFortranStubs()
    else:
      self.logPrint("Not a clone of PETSc or no Fortran compiler or fortran-bindings disabled, don't need Sowing\n")
    return

  def buildFortranStubs(self):
    if hasattr(self.compilers, 'FC'):
      if self.argDB['with-batch'] and not hasattr(self,'bfort'):
        self.logPrintBox('Batch build that could not generate bfort, skipping generating Fortran stubs\n \
                          you will need to copy them from some other system (src/fortran/auto)')
      else:
        self.logWrite('           Running '+self.bfort+' to generate fortran stubs\n')
        try:
          import os,sys
          sys.path.insert(0, os.path.abspath(os.path.join('lib','petsc','bin','maint')))
          import generatefortranstubs
          del sys.path[0]
          generatefortranstubs.main(self.petscdir.dir, self.bfort, self.petscdir.dir,0)
          if self.fortran.fortranIsF90:
            generatefortranstubs.processf90interfaces(self.petscdir.dir,0)
          self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs')
        except RuntimeError as e:
          raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return
