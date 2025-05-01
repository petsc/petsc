from __future__ import generators
import config.package
import sysconfig
from pathlib import Path

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.88.0'
    self.download          = ['https://archives.boost.io/release/'+self.version+'/source/boost_'+self.version.replace('.','_')+'.tar.bz2',
                              'https://web.cels.anl.gov/projects/petsc/download/externalpackages/boost_'+self.version.replace('.','_')+'.tar.bz2']
    self.includes          = ['boost/multi_index_container.hpp']
    self.liblist           = []
    self.buildLanguages    = ['Cxx']
    self.downloadonWindows = 1
    self.useddirectly      = 0

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('BOOST', '-download-boost-headers-only=<bool>', nargs.ArgBool(None, 0, 'When true, do not build Boost libraries, only install headers'))
    help.addArgument('BOOST', '-download-boost-bootstrap-arguments=<string>', nargs.ArgString(None, 0, 'Additional arguments for bootstrap of Boost build'))

  def Install(self):
    conffile = Path(self.packageDir) / (self.package + '.petscconf')
    conffile.write_text(self.installDir)

    if not self.installNeeded(str(conffile)):
      return self.installDir

    if self.argDB['download-boost-headers-only']:
      boostIncludeDir = Path(self.installDir) / self.includedir / 'boost'
      self.logPrintBox('Configure option --boost-headers-only is ENABLED ... boost libraries will not be built')
      self.logPrintBox('Installing boost headers, this should not take long')
      try:
        if boostIncludeDir.exists() or boostIncludeDir.is_symlink():
          boostIncludeDir.unlink()
        cmd = 'cd {} && ln -s $PWD/boost {}'.format(self.packageDir, boostIncludeDir)
        config.base.Configure.executeShellCommand(cmd, timeout=6000, log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Error linking Boost headers:\n'+str(e))
    else:
      if not self.checkCompile('#include <bzlib.h>', ''):
        raise RuntimeError('Boost requires bzlib.h. Please install it in default compiler search location.')

      if not (Path(sysconfig.get_paths()['include']) / 'pyconfig.h').is_file():
        raise RuntimeError('pyconfig.h missing: Boost requires python development version to be installed. (pythonX.x-dev)')

      with self.Language('Cxx'):
          cxx = self.getCompiler().lower()
          cxxflags = self.getCompilerFlags()

      if config.setCompilers.Configure.isGNU(cxx, self.log):
        toolset = 'gcc'
        pch = 'on'
      elif config.setCompilers.Configure.isOneAPI(cxx, self.log) or config.setCompilers.Configure.isIntel(cxx, self.log):
        toolset = 'intel-linux'
        pch = 'off' # https://github.com/bfgroup/b2/issues/413
      elif config.setCompilers.Configure.isClang(cxx, self.log):
        toolset='clang'
        pch='on'
      else:
        raise RuntimeError('Invalid CXX compiler specifield for boost: {}'.format(cxx))

      self.logPrintBox('Building Boost with toolset "{}", compiler "{}"'.format(toolset, cxx))

      jamfile = Path(self.packageDir) / 'user-config.jam'
      jamfile.write_text('using {} : : {} : <cxxflags>"{}" ;\n'.format(toolset, cxx, cxxflags))

      if 'download-boost-bootstrap-arguments' in self.argDB and self.argDB['download-boost-bootstrap-arguments']:
        bootstrap_arguments = self.argDB['download-boost-bootstrap-arguments']
      else:
        bootstrap_arguments = ''

      bootstrap_cmd = 'cd {} && ./bootstrap.sh --with-toolset={} --prefix={}'.format(self.packageDir, toolset, self.installDir)
      out, err, ret = config.base.Configure.executeShellCommand(bootstrap_cmd, timeout=6000, log=self.log)

      build_cmd = 'cd {} && ./b2 toolset={} {} pch={} -j{}'.format(self.packageDir, toolset, bootstrap_arguments, pch, self.make.make_np)
      out, err, ret = config.base.Configure.executeShellCommand(build_cmd, timeout=6000, log=self.log)

      install_cmd = 'cd {} && ./b2 toolset={} {} pch={} -j{} install'.format(self.packageDir, toolset, bootstrap_arguments, pch, self.make.make_np)
      out, err, ret = config.base.Configure.executeShellCommand(install_cmd, timeout=6000, log=self.log)

      self.postInstall(out + err, str(conffile))
    return self.installDir
