import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    matlab = None
    if 'with-matlab-dir' in self.framework.argDB:
      if os.path.exists(os.path.join(self.framework.argDB['with-matlab-dir'], 'bin', 'matlab')):
        matlab = self.framework.argDB['with-matlab-dir']
      else:
        raise RuntimeError('You set a value for --with-mpi-dir, but '+os.path.join(self.framework.argDB['with-matlab-dir'],'bin','matlab')+' does not exist')

    elif self.getExecutable('matlab', getFullPath = 1):
      matlab = os.path.dirname(os.path.dirname(self.matlab))
    
    if matlab:
      self.addDefine('HAVE_MATLAB', 1)
      self.addSubstitution('CMEX', os.path.join(matlab,'bin','mex'))
      self.addSubstitution('MCC', '${C_CC}')
      self.addSubstitution('MATLABCOMMAND', os.path.join(matlab,'bin','matlab'))
    else:
      self.addSubstitution('CMEX', '')
      self.addSubstitution('MCC', '')
      self.addSubstitution('MATLABCOMMAND', '')

    return

  def configure(self):
    self.setOutput()
    return
