from __future__ import generators
import config.base
import os
import re
    
class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return
     
  def configureHelp(self, help):
    import nargs
    return

  def configureFortranStubs(self):
    '''Determine whether the Fortran stubs exist or not'''
    stubDir = os.path.join(self.framework.argDB['PETSC_DIR'], 'src', 'fortran', 'auto')
    if not os.path.exists(os.path.join(stubDir, 'makefile.src')):
      self.framework.log.write('WARNING: Fortran stubs have not been generated in '+stubDir+'\n')
      self.framework.getExecutable('bfort', getFullPath = 1)
      if hasattr(self.framework, 'bfort'):
        self.framework.log.write('           Running '+self.framework.bfort+' to generate Fortran stubs\n')
        (output,err,status) = config.base.Configure.executeShellCommand('export PETSC_ARCH=linux;make allfortranstubs', timeout = 15*60.0, log = self.framework.log)
        # filter out the normal messages, user has to cope with error messages
        count = 0
        for line in map(lambda l: l.strip(), output.split('\n')):
          if line and not (line.startswith('fortranstubs in:') or line.startswith('Fixing pointers') or line.find('ACTION=') >= 0):
            if not count:
              self.framework.log.write('*******Error generating Fortran stubs****\n')
            count += 1
            self.framework.log.write(line+'\n')
        if not count:
          self.framework.log.write('           Completed generating Fortran stubs\n')
        else:
          self.framework.log.write('*******End of error messages from generating Fortran stubs****\n')
      else:
        message = 'See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\nto obtain bfort to generate the Fortran stubs or make sure\nbfort is in your path\n'
        self.framework.log.write(message)
        if 'FC' in self.framework.argDB and self.framework.argDB['FC']:
          raise RuntimeError('You have a Fortran compiler but the PETSc Fortran stubs are not built and cannot be built.\n'+message+'or run with with --with-fc=0 to turn off the Fortran compiler')
    else:
      self.framework.log.write('Fortran stubs do exist in '+stubDir+'\n')
    return

  def configure(self):
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranStubs)
    return
