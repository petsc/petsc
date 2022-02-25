import config.package

class Configure(config.package.Package):
  def __init__(self,framework):
    config.package.Package.__init__(self,framework)
    self.gitcommit              = 'a23ddbb86c23d17beb147db43b42502fbe0db0ca' # main jan-27-2022
    self.download               = ['git://https://github.com/htool-ddm/htool','https://github.com/htool-ddm/htool/archive/'+self.gitcommit+'.tar.gz']
    self.minversion             = '0.8.0'
    self.versionname            = 'HTOOL_VERSION'
    self.versioninclude         = 'htool/misc/define.hpp'
    self.buildLanguages         = ['Cxx']
    self.functions              = []
    self.includes               = ['htool/misc/define.hpp'] # no C++11 in this header
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
    import shutil
    import os
    incDir = os.path.join(self.installDir,self.includedir)
    self.include = [incDir]
    if not hasattr(self.framework,'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    srcdir = os.path.join(self.packageDir,'include','htool')
    destdir = os.path.join(incDir,'htool')
    try:
      self.logPrintBox('Copying Htool; this may take several seconds')
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError as e:
      raise RuntimeError('Error copying Htool: '+str(e))
    return self.installDir
