import config.base
import os
import re

class Options(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def getCFlags(self, compiler, bopt):
    flags = []
    # GNU gcc
    if config.compilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.append('-Wall')
      elif bopt == 'g':
        flags.append('-g3')
      elif bopt == 'O':
        if os.environ['USER'] in ['barrysmith','bsmith','knepley','buschelm','balay','petsc']:
          self.pushLanguage('C')
          for i in ['-Wshadow','-Wwrite-strings']:
            try:
              self.framework.checkCompilerFlag(i)
              flags.append(i)
            except: pass
          self.popLanguage()
        flags.extend(['-O', '-fomit-frame-pointer'])
    # Alpha
    elif re.match(r'alphaev[5-9]', self.framework.host_cpu):
      # Compaq C
      if compiler == 'cc':
        if bopt == 'O':
          flags.extend(['-O2', '-Olimit 2000'])
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro C
      if compiler == 'cc':
        if bopt == '':
          flags.extend(['-woff 1164', '-woff 1552', '-woff 1174'])
        elif bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.extend(['-O2', '-OPT:Olimit=6500'])
    # Generic
    else:
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return ' '.join(flags)

  def getCxxFlags(self, compiler, bopt):
    flags = []
    # GNU g++
    if config.compilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.append('-Wall')
      elif bopt == 'g':
        flags.append('-g3')
      elif bopt == 'O':
        if os.environ['USER'] in ['barrysmith','bsmith','knepley','buschelm','petsc','balay']:
          for i in ['-Wshadow','-Wwrite-strings']:
            try:
              self.framework.checkCompilerFlag(i)
              flags.append(i)
            except: pass
        flags.extend(['-O', '-fomit-frame-pointer'])
    # Alpha
    elif re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq C++
      if compiler == 'cxx':
        if bopt == 'O':
          flags.append('-O2')
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro C++
      if compiler == 'cc':
        if bopt == '':
          flags.extend(['-woff 1164', '-woff 1552', '-woff 1174'])
        elif bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.extend(['-O2', '-OPT:Olimit=6500'])
    # Generic
    else:
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return ' '.join(flags)

  def getFortranFlags(self, compiler, bopt):
    flags = []
    # Alpha
    if re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq Fortran
      if compiler == 'fort':
        if bopt == 'O':
          flags.append('-O2')
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Portland Group Fortran 90
      if compiler == 'pgf90':
        if bopt == 'O':
          flags.extend(['-fast', '-tp p6', '-Mnoframe'])
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro Fortran
      if compiler == 'f90':
        if bopt == '':
          flags.append('-cpp')
        elif bopt == 'g':
          flags.extend(['-g', '-trapuv'])
        elif bopt == 'O':
          flags.extend(['-O2', '-IPA:cprop=OFF', '-OPT:IEEE_arithmetic=1'])
    # Generic
    else:
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return ' '.join(flags)

  def getCompilerFlags(self, language, compiler, bopt):
    flags = ''
    if language == 'C':
      flags = self.getCFlags(compiler, bopt)
    elif language == 'Cxx':
      flags = self.getCxxFlags(compiler, bopt)
    elif language in ['Fortran', 'F77']:
      flags = self.getFortranFlags(compiler, bopt)
    return flags

  def getCompilerVersion(self, language, compiler):
    if compiler is None:
      raise RuntimeError('Invalid compiler for version determination')
    version = 'Unknown'
    try:
      if language == 'C':
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'cc':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language == 'Cxx':
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'cxx':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language in ['Fortran', 'F77']:
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'fort':
          flags = '-version'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler == 'f90':
          flags = '-V'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler == 'pgf90':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'f90':
          flags = '-version'
        else:
          flags = '--version'
      (output, error, status) = self.executeShellCommand(compiler+' '+flags)
      if not status:
        version = output.split('\n')[0]
    except RuntimeError, e:
      self.framework.log.write('Could not determine compiler version: '+str(e))
    return version
