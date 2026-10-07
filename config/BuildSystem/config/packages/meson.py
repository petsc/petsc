import config.package
import os
from config.utilities.parseVersion import parseVersion

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.9.2'
    self.download          = ['https://github.com/mesonbuild/meson/releases/download/'+self.version+'/meson-'+self.version+'.tar.gz']
    self.lookforbydefault  = 1
    self.useddirectly      = 0
    self.linkedbypetsc     = 0
    self.publicInstall     = 0
    self.executablename    = 'meson'
    self.skippackagelibincludedirs = 1
    self.maxminMesonVersion = (0,55,0) # minimum Meson version needed by all active packages
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MESON', '-with-meson-exec=<executable>', nargs.Arg(None, None, 'Meson executable to look for'))
    return

  def Install(self):
    conffile = os.path.join(self.packageDir, 'meson.petscconf')
    with open(conffile, 'w') as out:
      out.write(self.python.pyexe+'\n')
    if not self.installNeeded(conffile):
      return self.installDir
    bindir = os.path.join(self.installDir, 'bin')
    os.makedirs(bindir, exist_ok=True)
    self.logPrintBox('Installing the Meson executable')
    self.executeShellCommand([self.python.pyexe, 'packaging/create_zipapp.py', '--outfile', os.path.join(bindir, 'meson'), '--interpreter', self.python.pyexe], cwd=self.packageDir, timeout=60, log=self.log)
    self.postInstall(conffile)
    return self.installDir

  def locateMeson(self):
    if 'with-meson-exec' in self.argDB:
      self.log.write('Looking for specified Meson executable '+self.argDB['with-meson-exec']+'\n')
      self.getExecutable(self.argDB['with-meson-exec'], getFullPath=1, resultName='meson')
    elif 'with-meson-dir' in self.argDB:
      self.log.write('Looking for Meson in '+os.path.join(self.argDB['with-meson-dir'], 'bin')+'\n')
      self.getExecutable('meson', path=os.path.join(self.argDB['with-meson-dir'], 'bin'), getFullPath=1)
    else:
      self.log.write('Looking for default Meson executable\n')
      self.getExecutable('meson', getFullPath=1, resultName='meson')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def configure(self):
    '''Locate Meson and download it if requested'''
    if self.argDB['download-meson']:
      self.log.write('Installing Meson\n')
      config.package.Package.configure(self)
      self.log.write('Looking for Meson in '+os.path.join(self.installDir,'bin')+'\n')
      self.getExecutable('meson', path=os.path.join(self.installDir,'bin'), getFullPath=1)
    elif (not self.argDB['with-meson'] == 0 and not self.argDB['with-meson'] == 'no') or 'with-meson-exec' in self.argDB or 'with-meson-dir' in self.argDB:
      self.executeTest(self.locateMeson)
    else:
      self.log.write('Not checking for Meson\n')
    self.found = 0
    if hasattr(self, 'meson'):
      import re
      try:
        (output, error, status) = config.base.Configure.executeShellCommand([self.meson, '--version'], log=self.log)
        if status:
          self.log.write('meson --version failed: '+str(error)+'\n')
          return
      except RuntimeError as e:
        self.log.write('meson --version failed: '+str(e)+'\n')
        return
      output = output.replace('stdout: ', '').strip()
      minMesonVersion = '.'.join(map(str, self.maxminMesonVersion))
      gver = re.match(r'(\d+\.\d+\.\d+)', output)
      if gver:
        self.foundversion = gver.group(1)
        self.log.write('Meson version found '+self.foundversion+'\n')
        if parseVersion(self.foundversion) < parseVersion(minMesonVersion):
          if self.argDB['download-meson'] or 'with-meson-exec' in self.argDB or 'with-meson-dir' in self.argDB or self.framework.clArgDB.get('with-meson'):
            raise RuntimeError('Meson version '+minMesonVersion+' or newer is required (detected version is '+self.foundversion+'): use --download-meson')
          self.log.write('Ignoring Meson '+self.foundversion+'; it is older than '+minMesonVersion+'\n')
          return
        self.found = 1
      else:
        self.log.write('Meson version check failed\n')
    else:
      self.log.write('Meson not found\n')
    return
