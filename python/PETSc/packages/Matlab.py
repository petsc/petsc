import config.base
import os
import commands

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def configureHelp(self, help):
    import nargs
    help.addArgument('Matlab', '-with-matlab',                nargs.ArgBool(None, 1, 'Activate Matlab'))
    help.addArgument('Matlab', '-with-matlab-dir=<root dir>', nargs.ArgDir(None, None, 'Specify the root directory of the Matlab installation'))
    return

  def configureLibrary(self):
    '''Find a Matlab installation and check if it can work with PETSc'''
    matlab = None
    if 'with-matlab-dir' in self.framework.argDB:
      if os.path.exists(os.path.join(self.framework.argDB['with-matlab-dir'], 'bin', 'matlab')):
        matlab = self.framework.argDB['with-matlab-dir']
      else:
        raise RuntimeError('You set a value for --with-matlab-dir, but '+os.path.join(self.framework.argDB['with-matlab-dir'],'bin','matlab')+' does not exist')
    elif self.getExecutable('matlab', getFullPath = 1):
      matlab = os.path.dirname(os.path.dirname(self.matlab))

    if matlab:
      interpreter = os.path.join(matlab,'bin','matlab')
      (status,output) = commands.getstatusoutput(interpreter+' -nojvm -nodisplay -r "ver; exit"')
      if status:
        self.framework.log.write('WARNING: Found Matlab at '+matlab+' but unable to run\n')
        self.framework.log.write(output)
        matlab = None
      else:
        import re
        r = re.compile('Version ([0-9]*.[0-9]*)').search(output).group(1)
        r = float(r)
        if r < 6.0:
          self.framework.log.write('WARNING:Matlab version must be at least 6; yours is '+str(r)+'\n')
          matlab = None
        else:
          # hope there is always only one arch installation in the location
          matlab_arch = os.listdir(os.path.join(matlab,'extern','lib'))[0]

          self.framework.log.write('Configuring PETSc to use the Matlab at '+matlab+'\n')
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

    if not matlab:
      self.framework.log.write('Configuring PETSc to not use Matlab\n')
      self.addSubstitution('MATLAB_MEX', '')
      self.addSubstitution('MATLAB_CC', '')
      self.addSubstitution('MATLAB_COMMAND', '')
      self.addSubstitution('MATLAB_DIR', '')
      self.addSubstitution('MATLAB_DL', '')
      self.addSubstitution('MATLAB_ARCH', '')
    return

  def configure(self):
    if not self.framework.argDB['with-matlab']:
      return
    self.executeTest(self.configureLibrary)
    return
