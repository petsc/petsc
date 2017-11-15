import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.libraries.libraries.extend([('cygwin', 'log')])
    return

  def setupDependencies(self, framework):
    self.setCompilers = framework.require('config.setCompilers', self)
    self.libraries    = framework.require('config.libraries',    self)
    self.python       = framework.require('config.python',       self)
    return

  def configureCygwin(self):
    '''If libcygwin.a is found, define HAVE_CYGWIN'''
    self.hasCygwin = self.libraries.haveLib('cygwin')
    if self.hasCygwin:
      self.framework.addDefine('HAVE_CYGWIN', 1)
      self.framework.addSubstitution('HAVE_CYGWIN', 1)
    else:
      self.framework.addSubstitution('HAVE_CYGWIN', 0)
    return

  def checkCompiler(self):
    '''Make sure the compiler is recent enough'''
    if self.setCompilers.isGCXX:
      def checkCommand(command, status, output, error):
        if status:
          raise RuntimeError('g++ is not in your path; please make sure that you have a g++ of at least version 3 installed in your path. Get gcc/g++ at http://gcc.gnu.com')
        return

      (output, error, status) = self.executeShellCommand(self.framework.argDB['CXX']+' -dumpversion', checkCommand)
      version = output.split('.')[0]
      if not version == '3':
        raise RuntimeError('The g++ in your path is version '+version+'; please install a g++ of at least version 3 or fix your path. Get gcc/g++ at http://gcc.gnu.com')
    return

  def configure(self):
    self.executeTest(self.checkCompiler)
    self.executeTest(self.configureCygwin)
    return
