import config.package

class Configure(config.package.Package):
  def __init__(self,framework):
    config.package.Package.__init__(self,framework)
    self.gitcommit              = '9c004f24326c454eb34df5d155f145c7f902d573' # main aug-03-2021
    self.download               = ['git://https://github.com/htool-ddm/htool','https://github.com/htool-ddm/htool/archive/'+self.gitcommit+'.tar.gz']
    self.minversion             = '0.5.0'
    self.versionname            = 'HTOOL_VERSION'
    self.versioninclude         = 'htool/misc/define.hpp'
    self.minCxxVersion          = 'c++11'
    self.cxx                    = 1
    self.functions              = []
    self.includes               = ['htool/htool.hpp']
    self.skippackagewithoptions = 1
    self.precisions             = ['double'] # coordinates are stored in double precision, other scalars are templated, just enforce PetscReal == double during ./configure, for now
    self.usesopenmp             = 'yes'
    return

  def setupDependencies(self,framework):
    config.package.Package.setupDependencies(self,framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.mathlib      = framework.require('config.packages.mathlib',self)
    self.cxxlibs      = framework.require('config.packages.cxxlibs',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    self.blasLapack   = framework.require('config.packages.BlasLapack',self)
    self.openmp       = framework.require('config.packages.openmp',self)
    self.deps         = [self.blasLapack,self.cxxlibs,self.mathlib,self.mpi]
    self.odeps        = [self.openmp]
    return

  def Install(self):
    import os
    incDir = os.path.join(self.installDir,self.includedir)
    if self.installSudo:
      newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
      newuser = ''
    self.include = [incDir]
    if not hasattr(self.framework,'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    cpstr = newuser+' mkdir -p '+incDir+' && '+newuser+' cp -r '+os.path.join(self.packageDir,'include','*')+' '+incDir
    self.logPrintBox('Copying Htool; this may take several seconds')
    output,err,ret = config.package.Package.executeShellCommand(cpstr,timeout=100,log=self.log)
    self.log.write(output+err)
    return self.installDir
