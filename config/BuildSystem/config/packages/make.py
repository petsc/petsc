from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['http://ftp.gnu.org/gnu/make/make-3.82.tar.gz','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/make-3.82.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.requires32bitint  = 0
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    self.useddirectly      = 0
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('Make', '-download-make=<no,yes,filename>',             nargs.ArgDownload(None, 0, 'Download and install GNU make'))
    help.addArgument('Make', '-download-make-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for GNU make configure'))
    help.addArgument('Make', '-download-make-configure-options=<options>',   nargs.Arg(None, None, 'additional options for GNU make configure'))
    return

  def Install(self):
    import os
    args = ['--prefix='+self.installDir]
    args.append('--program-prefix=g')
    if self.framework.argDB.has_key('download-make-cc'):
      args.append('CC="'+self.framework.argDB['download-make-cc']+'"')
    if self.framework.argDB.has_key('download-make-configure-options'):
      args.append(self.framework.argDB['download-make-configure-options'])
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'make.args'), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded('make.args'):
      try:
        self.logPrintBox('Configuring GNU Make; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on GNU make (install manually): '+str(e))
      try:
        self.logPrintBox('Compiling GNU Make; this may take several minutes')
        if self.getExecutable('make', getFullPath = 1,resultName='make'):
          output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make+' &&  '+self.make+' install && '+self.make+' clean', timeout=2500, log = self.framework.log)
        else:
          output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./build.sh && ./make install && ./make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on GNU Make (install manually): '+str(e))
      self.postInstall(output+err,'make.args')
    self.binDir = os.path.join(self.installDir, 'bin')
    self.gmake = os.path.join(self.binDir, 'gmake')
    self.addMakeMacro('OMAKE',self.gmake)
    return self.installDir

  def configure(self):
    '''Determine whether the GNU make exist or not'''

    if (self.framework.argDB['download-make']):
      config.package.Package.configure(self)
    else:
      self.getExecutable('gmake', getFullPath = 1,resultName='gmake')
      if hasattr(self, 'gmake'):
        self.addMakeMacro('OMAKE ', self.gmake)
        self.found = 1
    return
