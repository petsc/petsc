import os
import re
import shutil

import config.package
from config.utilities.parseVersion import parseVersion

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.13.1'
    self.minversion        = '1.8.2'
    self.download          = ['https://github.com/ninja-build/ninja/archive/refs/tags/v'+self.version+'.tar.gz']
    self.lookforbydefault  = 1
    self.publicInstall     = 0
    self.linkedbypetsc     = 0
    self.useddirectly      = 0
    self.skipMPIDependency = 1
    self.installwithbatch  = 1
    self.buildLanguages    = []
    self.executablename    = 'ninja'
    self.skippackagelibincludedirs = 1

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('NINJA', '-with-ninja-exec=<executable>', nargs.Arg(None, None, 'Ninja executable to look for'))
    help.addArgument('NINJA', '-download-ninja-cxx=<prog>', nargs.Arg(None, None, 'C++ compiler for building Ninja on the build machine'))

  def Install(self):
    env = os.environ.copy()
    if self.argDB.get('download-ninja-cxx'):
      env['CXX'] = self.argDB['download-ninja-cxx']
    conffile = os.path.join(self.packageDir, 'ninja.petscconf')
    with open(conffile, 'w') as out:
      out.write(repr([self.python.pyexe]+[env.get(key, '') for key in ['CXX', 'CXXFLAGS', 'CFLAGS', 'LDFLAGS']]))
    if not self.installNeeded(conffile):
      return self.installDir
    self.logPrintBox('Bootstrapping Ninja; this may take several minutes')
    self.executeShellCommand([self.python.pyexe, 'configure.py', '--bootstrap'], cwd=self.packageDir, env=env, timeout=900, log=self.log)
    bindir = os.path.join(self.installDir, 'bin')
    os.makedirs(bindir, exist_ok=True)
    shutil.copy2(os.path.join(self.packageDir, 'ninja'), bindir)
    self.postInstall(conffile)
    return self.installDir

  def configureLibrary(self):
    directory = self.checkDownload()
    if directory:
      executable = os.path.join(directory, 'bin', 'ninja')
    elif 'with-ninja-exec' in self.argDB:
      executable = self.argDB['with-ninja-exec']
    elif 'with-ninja-dir' in self.argDB:
      executable = os.path.join(self.argDB['with-ninja-dir'], 'bin', 'ninja')
    else:
      executable = 'ninja'
    self.getExecutable(executable, getFullPath=1, resultName='ninja')
    self.found = int(hasattr(self, 'ninja'))
    if not self.found and (directory or 'with-ninja-exec' in self.argDB or 'with-ninja-dir' in self.argDB or self.framework.clArgDB.get('with-ninja')):
      raise RuntimeError('Ninja was not found. Use --with-ninja-exec or --download-ninja.')

  def configure(self):
    if 'with-ninja-exec' in self.argDB:
      self.argDB['with-ninja'] = 1
    config.package.Package.configure(self)

  def checkVersion(self):
    if not self.found:
      return
    try:
      output, error, status = self.executeShellCommand([self.ninja, '--version'], log=self.log)
      version = re.match(r'(\d+\.\d+(?:\.\d+)?)', output.strip())
      if not version or parseVersion(version.group(1)) < parseVersion(self.minversion):
        raise RuntimeError('Ninja '+self.minversion+' or newer is required (reported '+output.strip()+')')
      self.foundversion = version.group(1)
    except RuntimeError as e:
      self.found = 0
      if self.argDB['download-ninja'] or 'with-ninja-exec' in self.argDB or 'with-ninja-dir' in self.argDB or self.framework.clArgDB.get('with-ninja'):
        raise RuntimeError(str(e)+'. Use --with-ninja-exec or --download-ninja.')
      self.logPrint('Ninja is unavailable: '+str(e))
