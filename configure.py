import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    libraries = [('cygwin', 'log')]
    # Add dependencies
    self.compilers = self.framework.require('config.compilers', self)
    self.libraries = self.framework.require('config.libraries', self)
    self.libraries.libraries.extend(libraries)
    return

  def configureCygwin(self):
    '''If libcygwin.a is found, define HAVE_CYGWIN'''
    if self.libraries.haveLib('cygwin'):
      self.framework.addDefine('HAVE_CYGWIN', 1)
      self.framework.addSubstitution('HAVE_CYGWIN', 1)
    else:
      self.framework.addSubstitution('HAVE_CYGWIN', 0)
    return

  def checkCompiler(self):
    '''Make sure the compiler is recent enough'''
    if self.compilers.isGCXX:
      import commands

      (status,output) = commands.getstatusoutput(self.framework.argDB['CXX']+' -dumpversion')
      if not status == 0:
        raise RuntimeError('g++ is not in your path; please make sure that you have a g++ of at least version 3 installed in your path. Get gcc/g++ at http://gcc.gnu.com')
      version = output.split('.')[0]
      if not version == '3':
        raise RuntimeError('The g++ in your path is version '+version+'; please install a g++ of at least version 3 or fix your path. Get gcc/g++ at http://gcc.gnu.com')
    return

  def configure(self):
    import os

    self.executeTest(self.checkCompiler)
    self.executeTest(self.configureCygwin)
    return
