from __future__ import generators
import config.base
import os
import commands

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.foundMatlab  = 0
    return

  def __str__(self):
    if self.foundMatlab: return 'Matlab: Using '+self.matlab+'\n'
    return ''
    
  def configureHelp(self, help):
    import nargs
    help.addArgument('Matlab', '-with-matlab',                nargs.ArgBool(None, 1, 'Activate Matlab'))
    help.addArgument('Matlab', '-with-matlab-dir=<root dir>', nargs.ArgDir(None, None, 'Specify the root directory of the Matlab installation'))
    return

  def generateGuesses(self):
    '''Generate list of possible locations of Matlab'''
    if 'with-matlab-dir' in self.framework.argDB:
      yield self.framework.argDB['with-matlab-dir']
      raise RuntimeError('You set a value for --with-matlab-dir, but '+self.framework.argDB['with-matlab-dir']+' cannot be used\n')
    if self.getExecutable('matlab', getFullPath = 1):
      yield os.path.dirname(os.path.dirname(self.matlab))
    return

  def configureLibrary(self):
    '''Find a Matlab installation and check if it can work with PETSc'''
    for matlab in self.generateGuesses():
      interpreter = os.path.join(matlab,'bin','matlab')
      (status,output) = commands.getstatusoutput(interpreter+' -nojvm -nodisplay -r "ver; exit"')
      if status:
        self.framework.log.write('WARNING: Found Matlab at '+matlab+' but unable to run\n')
        self.framework.log.write(output)
        self.framework.log.write('        Run with --with-matlab-dir=Matlabrootdir if you know where it is\n')
        matlab = None
      else:
        import re
        r = re.compile('Version ([0-9]*.[0-9]*)').search(output).group(1)
        r = float(r)
        if r < 6.0:
          self.framework.log.write('WARNING:Matlab version must be at least 6; yours is '+str(r)+'\n')
          self.framework.log.write('        Run with --with-matlab-dir=Matlabrootdir if you know where it is\n')
          matlab = None
        else:

          # make sure this is true root of Matlab
          if not os.path.isdir(os.path.join(matlab,'extern','lib')):
            self.framework.log.write('WARNING:'+matlab+' is not the root directory for Matlab\n')
            self.framework.log.write('        Run with --with-matlab-dir=Matlabrootdir if you know where it is\n')
            matlab = None
          else:
            matlab_arch = os.listdir(os.path.join(matlab,'extern','lib'))[0]

            # hope there is always only one arch installation in the location
            self.foundMatlab = 1
            self.matlab      = matlab
            matlab_arch = os.listdir(os.path.join(matlab,'extern','lib'))[0]

            self.framework.log.write('Configuring PETSc to use the Matlab at '+matlab+'\n')
            self.addDefine('HAVE_MATLAB', 1)
            self.addSubstitution('MATLAB_MEX', os.path.join(matlab,'bin','mex'))
            self.addSubstitution('MATLAB_CC', '${C_CC}')
            self.addSubstitution('MATLAB_COMMAND', os.path.join(matlab,'bin','matlab'))
            self.addSubstitution('MATLAB_INCLUDE', '-I'+os.path.join(matlab,'extern','include'))
            if matlab_arch == 'mac':
              matlab_dl = ' -L'+os.path.join(matlab,'sys','os','mac')+' -ldl'
            else:
              matlab_dl = ''
            self.addSubstitution('MATLAB_LIB','${CLINKER_SLFLAG}'+os.path.join(matlab,'extern','lib',matlab_arch)+' -L'+os.path.join(matlab,'extern','lib',matlab_arch)+' -leng -lmx -lmat -lut'+matlab_dl)

    if not self.foundMatlab:
      self.emptySubstitutions()
    return

  def emptySubstitutions(self):
    self.framework.log.write('Configuring PETSc to not use Matlab\n')
    self.addSubstitution('MATLAB_MEX', '')
    self.addSubstitution('MATLAB_CC', '')
    self.addSubstitution('MATLAB_COMMAND', '')
    self.addSubstitution('MATLAB_DIR', '')
    self.addSubstitution('MATLAB_INCLUDE', '')
    self.addSubstitution('MATLAB_LIB', '')

  def configure(self):
    if not self.framework.argDB['with-matlab']:
      self.emptySubstitutions()
      return
    self.executeTest(self.configureLibrary)
    return
