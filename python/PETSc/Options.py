import config.base

import re

class Options(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def getCFlags(self, compiler, bopt):
    flags = ''
    # Generic
    if bopt == 'g':
      flags = '-g'
    elif bopt == 'O':
      flags = '-O'
    # GNU gcc
    if compiler == 'gcc':
      if bopt == 'g':
        flags = '-g3 -Wall'
      elif bopt == 'O':
        flags = '-O -Wall -Wshadow -fomit-frame-pointer'
    # Alpha
    if re.match(r'alphaev[5-9]', self.configure.framework.host_cpu):
      # Compaq C
      if compiler == 'cc':
        if bopt == 'O':
          flags = '-O2 -Olimit 2000'
    # MIPS
    elif re.match(r'mips', self.configure.framework.host_cpu):
      # MIPS Pro C
      if compiler == 'cc':
        if bopt == 'g':
          flags = '-woff 1164 -woff 1552 -woff 1174 -g'
        elif bopt == 'O':
          flags = '-woff 1164 -woff 1552 -woff 1174 -O2 -OPT:Olimit=6500'
    return flags

  def getCxxFlags(self, compiler, bopt):
    flags = ''
    # Generic
    if bopt == 'g':
      flags = '-g'
    elif bopt == 'O':
      flags = '-O'
    # GNU g++
    if compiler == 'g++':
      if bopt == 'g':
        flags = '-g3 -Wall'
      elif bopt == 'O':
        flags = '-O -Wall -Wshadow -fomit-frame-pointer'
    # Alpha
    elif re.match(r'alphaev[0-9]', self.configure.framework.host_cpu):
      # Compaq C++
      if compiler == 'cxx':
        if bopt == 'O':
          flags = '-O2'
    # MIPS
    elif re.match(r'mips', self.configure.framework.host_cpu):
      # MIPS Pro C++
      if compiler == 'cc':
        if bopt == 'g':
          flags = '-woff 1164 -woff 1552 -woff 1174 -g'
        elif bopt == 'O':
          flags = '-woff 1164 -woff 1552 -woff 1174 -O2 -OPT:Olimit=6500'
    return flags

  def getFortranFlags(self, compiler, bopt):
    flags = ''
    # Generic
    if bopt == 'g':
      flags = '-g'
    elif bopt == 'O':
      flags = '-O'
    # Alpha
    if re.match(r'alphaev[0-9]', self.configure.framework.host_cpu):
      # Compaq Fortran
      if compiler == 'fort':
        if bopt == 'O':
          flags = '-O2'
    # Intel
    elif re.match(r'i[3-9]86', self.configure.framework.host_cpu):
      # Portland Group Fortran 90
      if compiler == 'pgf90':
        if bopt == 'O':
          flags = '-fast -tp p6 -Mnoframe'
    # MIPS
    elif re.match(r'mips', self.configure.framework.host_cpu):
      # MIPS Pro Fortran
      if compiler == 'f90':
        if bopt == 'g':
          flags = '-cpp -g -trapuv'
        elif bopt == 'O':
          flags = '-cpp -O2 -IPA:cprop=OFF -OPT:IEEE_arithmetic=1'
    return flags

  def getCompilerFlags(self, language, compiler, bopt, configure):
    flags = ''
    self.configure = configure
    if language == 'C':
      flags = self.getCFlags(compiler, bopt)
    elif language == 'Cxx':
      flags = self.getCxxFlags(compiler, bopt)
    elif language == 'Fortran':
      flags = self.getFortranFlags(compiler, bopt)
    return flags

  def getCompilerVersion(self, language, compiler, configure):
    version = 'Unknown'
    try:
      if language == 'C':
        if re.match(r'alphaev[0-9]', configure.framework.host_cpu) and compiler == 'cc':
          flags = '-V'
        elif re.match(r'mips', configure.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language == 'Cxx':
        if re.match(r'alphaev[0-9]', configure.framework.host_cpu) and compiler == 'cxx':
          flags = '-V'
        elif re.match(r'mips', configure.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language == 'Fortran':
        if re.match(r'alphaev[0-9]', configure.framework.host_cpu) and compiler == 'fort':
          flags = '-version'
        elif re.match(r'i[3-9]86', configure.framework.host_cpu) and compiler == 'f90':
          flags = '-V'
        elif re.match(r'i[3-9]86', configure.framework.host_cpu) and compiler == 'pgf90':
          flags = '-V'
        elif re.match(r'mips', configure.framework.host_cpu) and compiler == 'f90':
          flags = '-version'
        else:
          flags = '--version'
      (output, error, status) = self.executeShellCommand(compiler+' '+flags)
      if not status:
        version = output.split('\n')[0]
    except RuntimeError, e:
      self.framework.log.write('Could not determine compiler version: '+str(e))
    return version
