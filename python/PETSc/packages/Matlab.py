import config.base
import os
import commands

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
      (status,output) = commands.getstatusoutput(os.path.join(matlab,'bin','matlab')+' -nojvm -nodisplay -r "ver; exit"')
      if status:
        raise RuntimeError('Unable to run '+os.path.join(self.framework.argDB['with-matlab-dir'],'bin','matlab')+'\n'+output)

      # hope there is always only one arch installation in the location
      matlab_arch = os.listdir(os.path.join(matlab,'extern','lib'))[0]

      self.addDefine('HAVE_MATLAB', 1)
      self.addSubstitution('MATLAB_MEX', os.path.join(matlab,'bin','mex'))
      self.addSubstitution('MATLAB_CC', '${C_CC}')
      self.addSubstitution('MATLAB_COMMAND', os.path.join(matlab,'bin','matlab'))
      self.addSubstitution('MATLAB_DIR', os.path.join(matlab))
      if matlab_arch == 'mac':
        self.addSubstitution('MATLAB_DL', '-L'+os.path.join(matlab,'sys','os','mac')+' -ldl')
      else:
        self.addSubstitution('MATLAB_DL', '')
      self.addSubstitution('MATLAB_ARCH', matlab_arch)
    else:
      self.addSubstitution('MATLAB_MEX', '')
      self.addSubstitution('MATLAB_CC', '')
      self.addSubstitution('MATLAB_COMMAND', '')
      self.addSubstitution('MATLAB_DIR', '')
      self.addSubstitution('MATLAB_DL', '')
      self.addSubstitution('MATLAB_ARCH', '')
    return

  def configure(self):
    self.setOutput()
    return
