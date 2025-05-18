import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '1.15.2'
    self.gitcommit        = 'v' + self.version
    self.download         = ['git://https://github.com/gabime/spdlog.git']
    self.downloaddirnames = ['spdlog']
    self.functions        = ['']
    self.functionsCxx     = [1,'namespace spdlog {void disable_backtrace();}','spdlog::disable_backtrace();']
    self.includes         = ['spdlog/spdlog.h']
    self.liblist          = [['libspdlog.a'], ['libspdlogd.a']]
    self.precisions       = ['double']
    self.buildLanguages   = ['Cxx']
    return
