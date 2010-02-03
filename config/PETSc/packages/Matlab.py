from __future__ import generators
import PETSc.package

import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('Matlab', '-with-matlab=<bool>',         nargs.ArgBool(None, 0, 'Activate Matlab'))
    help.addArgument('Matlab', '-with-matlab-engine=<bool>',  nargs.ArgBool(None, 0, 'Activate Matlab Engine (need --with-matlab also) '))    
    help.addArgument('Matlab', '-with-matlab-dir=<root dir>', nargs.ArgDir(None, None, 'Specify the root directory of the Matlab installation'))
    help.addArgument('Matlab', '-with-matlab-arch=<string>',  nargs.ArgString(None, None, 'Use Matlab Architecture (default use first-found)'))
    return

  def generateGuesses(self):
    '''Generate list of possible locations of Matlab'''
    if 'with-matlab-dir' in self.framework.argDB:
      yield self.framework.argDB['with-matlab-dir']
      raise RuntimeError('You set a value for --with-matlab-dir, but '+self.framework.argDB['with-matlab-dir']+' cannot be used\n')
    if self.getExecutable('matlab', getFullPath = 1):
      # follow any symbolic link of this path
      self.matlab = os.path.realpath(self.matlab)
      yield os.path.dirname(os.path.dirname(self.matlab))
    for dir in os.listdir('/Applications'):
      if dir.startswith('MATLAB'):
        if os.path.isfile(os.path.join('/Applications',dir,'bin','matlab')):
          yield os.path.join('/Applications',dir)
    return

  def alternateConfigureLibrary(self):
    if self.framework.argDB['with-matlab-engine']:
      raise RuntimeError('--with-matlab-engine flag also requires --with-matlab flag\n')

  def configureLibrary(self):
    '''Find a Matlab installation and check if it can work with PETSc'''
    import re
          
    versionPattern = re.compile('Version ([0-9]*.[0-9]*)')
    for matlab in self.generateGuesses():
      self.framework.log.write('Testing Matlab at '+matlab+'\n')
      interpreter = os.path.join(matlab,'bin','matlab')
      output      = ''
      try:
        output,err,ret = PETSc.package.NewPackage.executeShellCommand(interpreter+' -nojvm -nodisplay -r "[\'Version \' version]; exit"', log = self.framework.log)
      except:
        self.framework.log.write('WARNING: Found Matlab at '+matlab+' but unable to run\n')
        continue

      match  = versionPattern.search(output)
      r = float(match.group(1))
      if r < 6.0:
        self.framework.log.write('WARNING: Matlab version must be at least 6; yours is '+str(r))
        continue
      # make sure this is true root of Matlab
      if not os.path.isdir(os.path.join(matlab,'extern','lib')):
        self.framework.log.write('WARNING:'+matlab+' is not the root directory for Matlab\n')
        self.framework.log.write('        Run with --with-matlab-dir=Matlabrootdir if you know where it is\n')
      else:
        self.matlab      = matlab
        ls = os.listdir(os.path.join(matlab,'extern','lib'))
        if ls:
          if 'with-matlab-arch' in self.framework.argDB:
            matlab_arch = self.framework.argDB['with-matlab-arch']
            if not matlab_arch in ls:
              self.framework.log.write('WARNING: You indicated --with-matlab-arch='+matlab_arch+' but that arch does not exist;\n possibilities are '+str(ls))
              continue
          else:
            matlab_arch = ls[0]
          self.framework.log.write('Configuring PETSc to use the Matlab at '+matlab+' Matlab arch '+matlab_arch+'\n')
          self.mex = os.path.join(matlab,'bin','mex')
          self.cc = '${CC}'
          self.command = os.path.join(matlab,'bin','matlab')
          self.include = [os.path.join(matlab,'extern','include')]
          if self.framework.argDB['with-matlab-engine']:          
            if matlab_arch == 'mac':
              matlab_dl = [' -L'+os.path.join(matlab,'sys','os','mac'),' -ldl']
            else:
              matlab_dl = ['']
            # Matlab libraries require libstdc++-libc6.1-2.so.3 which they provide in the sys/os directory
            if matlab_arch == 'glnx86':
              matlab_sys = ':'+os.path.join(matlab,'sys','os',matlab_arch)
            else:
              matlab_sys = ''
            matlab_sys += ':'+os.path.join(matlab,'bin',matlab_arch)+':'+os.path.join(matlab,'extern','lib',matlab_arch)
            self.lib = [self.setCompilers.CSharedLinkerFlag+matlab_sys,'-L'+os.path.join(matlab,'bin',matlab_arch),'-L'+os.path.join(matlab,'extern','lib',matlab_arch),'-leng','-lmx','-lmat','-lut','-licudata','-licui18n','-licuuc','-lustdio'] + matlab_dl

            self.addDefine('HAVE_MATLAB_ENGINE', 1)
          self.framework.packages.append(self)
          self.addMakeMacro('MATLAB_MEX',self.mex)
          self.addMakeMacro('MATLAB_COMMAND',self.command)        
          self.found = 1
          return
        else:
          self.framework.log.write('WARNING:Unable to use Matlab because cannot locate Matlab external libraries at '+os.path.join(matlab,'extern','lib')+'\n')
    raise RuntimeError('Could not find a functional Matlab\nRun with --with-matlab-dir=Matlabrootdir if you know where it is\n')
    return
