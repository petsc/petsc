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
        (output,err,status) = self.executeShellCommand('export PETSC_ARCH=linux;make allfortranstubs')
        # filter out the normal messages, user has to cope with error messages
        cnt = 0
        for i in output.split('\n'):
          if not (i.startswith('fortranstubs in:') or i.startswith('Fixing pointers') or i.find('ACTION=') >= 0):
            if not cnt:
              self.framework.log.write('*******Error generating Fortran stubs****\n')
            cnt = cnt + 1
            self.framework.log.write(i+'\n')
        if not cnt:
          self.framework.log.write('           Completed generating Fortran stubs\n')
        else:
          self.framework.log.write('*******End of error messages from generating Fortran stubs****\n')
      else:
        self.framework.log.write('           See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\n')
        self.framework.log.write('           to obtain bfort to generate the Fortran stubs or make sure\n')
        self.framework.log.write('           bfort is in your path\n')
        self.framework.log.write('WARNING: Turning off Fortran interfaces for PETSc')
        del self.framework.argDB['FC']
        self.compilers.addSubstitution('FC', '')
    else:
      self.framework.log.write('Fortran stubs do exist in '+stubDir+'\n')
    return

  def configure(self):
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranStubs)
    return
