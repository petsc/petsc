import config.base
import os
import re

class CompilerOptions(config.base.Configure):
  def getCFlags(self, compiler, bopt):
    import config.setCompilers

    if compiler.find('mpicc') >=0:
      try:
        output   = self.executeShellCommand(compiler + ' -show')[0]
        compiler = output.split(' ')[0]
      except:
        pass
      
    flags = []
    # GNU gcc
    if config.setCompilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.extend(['-Wall', '-Wwrite-strings', '-Wno-strict-aliasing'])
      elif bopt == 'g':
        if self.framework.argDB['with-gcov']:
          flags.extend(['-fprofile-arcs', '-ftest-coverage'])
        flags.append('-g3')
      elif bopt == 'O':
        flags.append('-O')
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
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler) and not compiler.find('win32fe') >=0:
        if bopt == '':
          flags.append('-wd1572')
          flags.append('-Qoption,cpp,--extended_float_type')
        elif bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          flags.extend(['-MT'])
        elif bopt == 'g':
          flags.extend(['-Z7'])
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          flags.extend(['-MT','-wd4996'])
        elif bopt == 'g':
          flags.extend(['-Z7'])
        elif bopt == 'O':
          flags.extend(['-O2', '-QxW'])
      # Windows Borland
      elif compiler.find('win32fe bcc32') >= 0:
        if bopt == '':
          flags.append('-RT -w-8019 -w-8060 -w-8057 -w-8004 -w-8066')
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return flags

  def getCxxFlags(self, compiler, bopt):
    import config.setCompilers

    if compiler.find('mpiCC') >=0  or compiler.find('mpicxx') >=0 :
      try:
        output   = self.executeShellCommand(compiler+' -show')[0]
        compiler = output.split(' ')[0]
      except:
        pass
    
    flags = []
    # GNU g++
    if config.setCompilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.extend(['-Wall', '-Wwrite-strings', '-Wno-strict-aliasing'])
      elif bopt in ['g']:
        if self.framework.argDB['with-gcov']:
          flags.extend(['-fprofile-arcs', '-ftest-coverage'])
        # -g3 causes an as SEGV on OSX
        flags.append('-g')
      elif bopt in ['O']:
        if os.environ.has_key('USER'):
          flags.append('-O')
    # Alpha
    elif re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq C++
      if compiler == 'cxx':
        if bopt in ['O']:
          flags.append('-O2')
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro C++
      if compiler == 'cc':
        if bopt == '':
          flags.extend(['-woff 1164', '-woff 1552', '-woff 1174'])
        elif bopt in ['g']:
          flags.append('-g')
        elif bopt in ['O']:
          flags.extend(['-O2', '-OPT:Olimit=6500'])
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler) and not compiler.find('win32fe') >=0:
        if bopt == '':
          flags.append('-wd1572')
        elif bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          flags.extend(['-MT','-GX','-GR'])
        elif bopt in ['g']:
          flags.extend(['-Z7'])
        elif bopt in ['O']:
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          flags.extend(['-MT','-GR','-EHsc']) # removing GX in favor of EHsc
        elif bopt == 'g':
          flags.extend(['-Z7','-Zm200'])
        elif bopt == 'O':
          flags.extend(['-O2','-QxW','-Zm200'])
      # Windows Borland
      elif compiler.find('win32fe bcc32') >= 0:
        if bopt == '':
          flags.append('-RT -w-8019 -w-8060 -w-8057 -w-8004 -w-8066')
    # IBM
    elif compiler.find('mpCC') >= 0 or compiler.find('xlC') >= 0:
      if bopt == '':
        flags.append('-qrtti=dyna')  # support dynamic casts in C++
      elif bopt in ['g']:
        flags.append('-g')
      elif bopt in ['O']:
        flags.append('-O')
      
    # Generic
    if not len(flags):
      if bopt in ['g']:
        flags.append('-g')
      elif bopt in ['O']:
        flags.append('-O')
    return flags

  def getFortranFlags(self, compiler, bopt):

    if compiler.endswith('mpif77') or compiler.endswith('mpif90'):
      try:
        output   = self.executeShellCommand(compiler+' -show')[0]
        compiler = output.split(' ')[0]
      except:
        pass

    flags = []
    if config.setCompilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.extend(['-Wall', '-Wno-unused-variable'])
        if config.setCompilers.Configure.isGfortran450(compiler):
          flags.extend(['-Wno-line-truncation'])
      elif bopt == 'g':
        if self.framework.argDB['with-gcov']:
          flags.extend(['-fprofile-arcs', '-ftest-coverage'])
        # g77 3.2.3 preprocesses the file into nothing if we give -g3
        flags.append('-g')
      elif bopt == 'O':
        flags.extend(['-O'])
    # Alpha
    elif re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq Fortran
      if compiler == 'fort':
        if bopt == 'O':
          flags.append('-O2')
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Portland Group Fortran 90
      if compiler == 'pgf90':
        if bopt == '':
          flags.append('-Mfree')
        elif bopt == 'O':
          flags.extend(['-fast', '-Mnoframe'])
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler) and not compiler.find('win32fe') >=0:
        if bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe ifl') >= 0 or compiler.find('win32fe ifort') >= 0:
        if bopt == '':
          flags.extend(['-MT'])
        elif bopt == 'g':
          flags.extend(['-Z7'])
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Compaq Visual FORTRAN
      elif compiler.find('win32fe f90') >= 0 or compiler.find('win32fe df') >= 0:
        if bopt == '':
          flags.append('-threads')
        elif bopt == 'g':
          flags.extend(['-debug:full'])
        elif bopt == 'O':
          flags.extend(['-optimize:5', '-fast'])
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
    # MacOSX on Apple Power PC
    elif config.setCompilers.Configure.isDarwin():
      # IBM
      if bopt == '' and (compiler.find('f90') or re.match(r'\w*xl[fF]\w*', compiler)):
        import commands
        output  = commands.getoutput(compiler+' -v')
        if output.find('IBM') >= 0:
          flags.append('-qextname')
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return flags

  def getCompilerFlags(self, language, compiler, bopt):
    flags = ''
    if language == 'C':
      flags = self.getCFlags(compiler, bopt)
    elif language == 'Cxx':
      flags = self.getCxxFlags(compiler, bopt)
    elif language in ['Fortran', 'FC']:
      flags = self.getFortranFlags(compiler, bopt)
    return flags

  def getCompilerVersion(self, language, compiler):
    if compiler is None:
      raise RuntimeError('Invalid compiler for version determination')
    version = 'Unknown'
    try:
      if language == 'C':
        if compiler.endswith('xlc') or compiler.endswith('mpcc'):
          flags = "lslpp -L vac.C | grep vac.C | awk '{print $2}'"
        elif re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler.endswith('cc'):
          flags = compiler+' -V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler.endswith('cc'):
          flags = compiler+' -version'
        else:
          flags = compiler+' --version'
      elif language == 'Cxx':
        if compiler.endswith('xlC') or compiler.endswith('mpCC'):
          flags = "lslpp -L vacpp.cmp.core  | grep vacpp.cmp.core  | awk '{print $2}'"
        elif re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler.endswith('cxx'):
          flags = compiler+' -V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler.endswith('cc'):
          flags = compiler+' -version'
        else:
          flags = compiler+' --version'
      elif language in ['Fortran', 'FC']:
        if compiler.endswith('xlf') or compiler.endswith('xlf90'):
          flags = "lslpp -L xlfcmp | grep xlfcmp | awk '{print $2}'"
        elif re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler.endswith('fort'):
          flags = compiler+' -version'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler.endswith('pgf90'):
          flags = compiler+' -V'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler.endswith('f90'):
          flags = compiler+' -V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler.endswith('f90'):
          flags = compiler+' -version'
        else:
          flags = compiler+' --version'
      (output, error, status) = config.base.Configure.executeShellCommand(flags, log = self.framework.log)
      if not status:
        if compiler.find('win32fe') > -1:
          version = '\\n'.join(output.split('\n')[0:2])
        else:
          #PGI/Windows writes an empty '\r\n' on the first line of output
          if output.count('\n') > 1 and output.split('\n')[0] == '\r':
            version = output.split('\r\n')[1]
          else:
            version = output.split('\n')[0]
          
    except RuntimeError, e:
      self.framework.log.write('Could not determine compiler version: '+str(e))
    self.framework.log.write('getCompilerVersion: '+str(compiler)+' '+str(version)+'\n')
    return version
