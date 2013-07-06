import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath      = 'http://ftp.mcs.anl.gov/pub/fathom/'
    # To track MOAB.git, update gitcommit to 'git describe --always' or 'git rev-parse HEAD'
    self.gitcommit         = '23ce6f7956f8a9e233edd17ac82e20726d3be500' # master 2013-06-25
    self.giturls           = ['https://bitbucket.org/fathomteam/moab.git']
    self.downloadname      = 'moab'
    self.downloadfilename  = 'moab'
    self.downloadversion   = 'nightly'
    self.downloadext       = 'tar.gz'
    self.functions         = ['Core']
    self.functionsCxx     = [1, 'namespace moab {class Core {public: Core();};}','moab::Core *mb = new moab::Core()']
    self.includes          = ['moab/Core.hpp']
    self.liblist           = [['libiMesh.a', 'libMOAB.a'],['libMOAB.a']]
    self.cxx               = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi       = framework.require('config.packages.MPI', self)
    self.hdf5      = framework.require('config.packages.hdf5', self)
    self.netcdf    = framework.require('config.packages.netcdf', self)
#    self.netcdfcxx = framework.require('config.packages.netcdf-cxx', self)
    self.odeps     = [self.mpi, self.hdf5, self.netcdf]
    return

  def gitPreReqCheck(self):
    return self.programs.autoreconf_flg

  def gitPreInstallCheck(self):
    '''check for git repo - and then regenerate configure'''
    import os
    if os.path.isdir(os.path.join(self.packageDir,'.git')):
      if not self.programs.autoreconf_flg:
        raise RuntimeError('autoreconf required for git ' + self.PACKAGE+' not found (or broken)! Try removing :',self.packageDir)
      try:
        self.logPrintBox('Running autoreconf on ' +self.PACKAGE+'; this may take several minutes')
        output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.programs.autoreconf + ' -fi', timeout=200, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running autoreconf on ' + self.PACKAGE+': '+str(e))
    return


