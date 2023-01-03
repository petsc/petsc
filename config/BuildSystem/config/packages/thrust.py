import config.package

# Note thrust is a C++ header library. Users can still use --download-thrust, --with-thrust-dir etc.
# But for --with-thrust-dir=mydir, the mydir should have a structure mydir/include/{thrust, cub}/...
# For --with-thrust-include=myinc, the myinc should have a structure myinc/{thrust, cub}/...
# For --download-thrust=mytarball.tgz, after unzip, its structure should be mytarball/{thrust, cub}/...
class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.minversion       = '1.9.8'
    self.versionname      = 'THRUST_VERSION'
    self.versioninclude   = 'thrust/version.h'
    self.gitcommit        = '1.17.2'
    self.download         = ['git://https://github.com/NVIDIA/thrust.git','https://github.com/NVIDIA/thrust/archive/'+self.gitcommit+'.tar.gz']
    self.includes         = ['thrust/version.h']
    self.precisions       = ['single','double']
    self.buildLanguages   = ['Cxx']
    return

  def versionToStandardForm(self,ver):
    '''Converts from thrust 100908 notation to standard notation 1.9.8'''
    return ".".join(map(str,[int(ver)//100000, int(ver)//100%1000, int(ver)%100]))

  def updateGitDir(self):
    import os
    config.package.GNUPackage.updateGitDir(self)
    if not hasattr(self.sourceControl, 'git') or (self.packageDir != os.path.join(self.externalPackagesDir,'git.'+self.package)):
      return
    try:
      self.executeShellCommand([self.sourceControl.git, 'submodule', 'update', '--init'], cwd=self.packageDir, log=self.log)
    except RuntimeError:
      raise RuntimeError('Could not initialize cub submodule needed by thrust')
    return

  def Install(self):
    ''''Install thrust and return its installation dir '''
    import os
    with open(os.path.join(self.packageDir,'petsc.mk'),'w') as g:
      g.write('#empty\n')

    if not self.installNeeded('petsc.mk'):
      return self.installDir

    incDir       = self.includeDir
    srcThrustDir = os.path.join(self.packageDir,'thrust')
    srcCubDir    = os.path.join(self.packageDir,'cub')

    cub_cuh = os.path.join(srcCubDir,'cub.cuh')
    if not os.path.isfile(cub_cuh):
      raise RuntimeError(cub_cuh+' does not exist. You might have forgot to download the cub submodule in thrust.')

    # srcCubDir might be a symbol link
    cpstr = ' mkdir -p '+incDir + ' && cp -RL '+srcThrustDir+' '+srcCubDir+' '+incDir
    try:
      self.logPrintBox('Copying THRUST; this may take several seconds')
      output,err,ret = config.package.Package.executeShellCommand(cpstr,timeout=100,log=self.log)
    except RuntimeError as e:
      self.logPrint('Error executing "'+cpstr+'": '+str(e))
      raise RuntimeError('Error copying THRUST')
    self.postInstall(output+err,'petsc.mk')
    return self.installDir
