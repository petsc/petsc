import commands
import re

class Options:
  def __init__(self):
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
    if re.match(r'alphaev[5-9]', self.configure.host_cpu):
      # Compaq C
      if compiler == 'cc':
        if bopt == 'O':
          flags = '-O2 -Olimit 2000'
    # MIPS
    elif re.match(r'mips', self.configure.host_cpu):
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
    if re.match(r'alphaev[0-9]', self.configure.host_cpu):
      # Compaq C++
      if compiler == 'cxx':
        if bopt == 'O':
          flags = '-O2'
    # MIPS
    elif re.match(r'mips', self.configure.host_cpu):
      # MIPS Pro C++
      if compiler == 'cc':
        if bopt == 'g':
          flags = '-woff 1164 -woff 1552 -woff 1174 -g'
        elif bopt == 'O':
          flags = '-woff 1164 -woff 1552 -woff 1174 -O2 -OPT:Olimit=6500'
    return 

  def getFortranFlags(self, compiler, bopt):
    flags = ''
    # Generic
    if bopt == 'g':
      flags = '-g'
    elif bopt == 'O':
      flags = '-O'
    # Alpha
    if re.match(r'alphaev[0-9]', self.configure.host_cpu):
      # Compaq Fortran
      if compiler == 'fort':
        if bopt == 'O':
          flags = '-O2'
    # Intel
    elif re.match(r'i[3-9]86', self.configure.host_cpu):
      # Portland Group Fortran 90
      if compiler == 'pgf90':
        if bopt == 'O':
          flags = '-fast -tp p6 -Mnoframe'
    # MIPS
    elif re.match(r'mips', self.configure.host_cpu):
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
    if language == 'C':
      if re.match(r'alphaev[0-9]', configure.host_cpu) and compiler == 'cc':
        (status, output) = commands.getstatusoutput(compiler+' -V')
        if not status: version = output.split('\n')[0]
      elif re.match(r'mips', configure.host_cpu) and compiler == 'cc':
        (status, output) = commands.getstatusoutput(compiler+' -version')
        if not status: version = output.split('\n')[0]
      else:
        (status, output) = commands.getstatusoutput(compiler+' --version')
        if not status: version = output.split('\n')[0]
    elif language == 'Cxx':
      if re.match(r'alphaev[0-9]', configure.host_cpu) and compiler == 'cxx':
        (status, output) = commands.getstatusoutput(compiler+' -V')
        if not status: version = output.split('\n')[0]
      elif re.match(r'mips', configure.host_cpu) and compiler == 'cc':
        (status, output) = commands.getstatusoutput(compiler+' -version')
        if not status: version = output.split('\n')[0]
      else:
        (status, output) = commands.getstatusoutput(compiler+' --version')
        if not status: version = output.split('\n')[0]
    elif language == 'Fortran':
      if re.match(r'alphaev[0-9]', configure.host_cpu) and compiler == 'fort':
        (status, output) = commands.getstatusoutput(compiler+' -version')
        if not status: version = output.split('\n')[0]
      elif re.match(r'i[3-9]86', configure.host_cpu) and compiler == 'f90':
        (status, output) = commands.getstatusoutput(compiler+' -V')
        if not status: version = output.split('\n')[0]
      elif re.match(r'i[3-9]86', configure.host_cpu) and compiler == 'pgf90':
        (status, output) = commands.getstatusoutput(compiler+' -V 2> /dev/null')
        if not status: version = output.split('\n')[0]
      elif re.match(r'mips', configure.host_cpu) and compiler == 'f90':
        (status, output) = commands.getstatusoutput(compiler+' -version')
        if not status: version = output.split('\n')[0]
      else:
        (status, output) = commands.getstatusoutput(compiler+' --version')
        if not status: version = output.split('\n')[0]
    return version
