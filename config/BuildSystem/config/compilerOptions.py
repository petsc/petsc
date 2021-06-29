import config.base
import os
import re
import nargs

class CompilerOptions(config.base.Configure):
  def getCFlags(self, compiler, bopt, language):
    import config.setCompilers

    if language == 'C':
      if [s for s in ['mpicc','mpiicc'] if os.path.basename(compiler).find(s)>=0]:
        try:
          output   = self.executeShellCommand(compiler + ' -show', log = self.log)[0]
          self.framework.addMakeMacro('MPICC_SHOW',output.strip().replace('\n','\\\\n'))
        except:
          self.framework.addMakeMacro('MPICC_SHOW',"Unavailable")
      else:
        self.framework.addMakeMacro('MPICC_SHOW',"Unavailable")

    flags = []
    # GNU gcc
    if config.setCompilers.Configure.isGNU(compiler, self.log) or config.setCompilers.Configure.isClang(compiler, self.log):
      if bopt == '':
        flags.extend(['-Wall', '-Wwrite-strings', '-Wno-strict-aliasing','-Wno-unknown-pragmas'])
        if config.setCompilers.Configure.isGcc110plus(compiler, self.log):
          flags.extend(['-Wno-misleading-indentation','-Wno-stringop-overflow'])
        # skip -fstack-protector for brew gcc - as this gives SEGV
        if not (config.setCompilers.Configure.isDarwin(self.log) and config.setCompilers.Configure.isGNU(compiler, self.log)):
          flags.extend(['-fstack-protector'])
        if config.setCompilers.Configure.isDarwinCatalina(self.log) and config.setCompilers.Configure.isClang(compiler, self.log):
          flags.extend(['-fno-stack-check'])
        flags.extend(['-mfp16-format=ieee']) #  arm for utilizing 16 bit storage of floating point
        if config.setCompilers.Configure.isClang(compiler, self.log):
          flags.extend(['-Qunused-arguments'])
        if self.argDB['with-visibility']:
          flags.extend(['-fvisibility=hidden'])
        arg = nargs.Arg.findArgument('with-errorchecking', self.clArgs)
        if not nargs.ArgBool('with-errorchecking', arg if arg is not None else '1', isTemporary=True).getValue():
          flags.extend(['-Wno-unused-but-set-variable'])
      elif bopt == 'g':
        flags.extend(['-g3','-O0'])
      elif bopt == 'gcov':
        flags.extend(['--coverage','-Og']) # --coverage is equal to -fprofile-arcs -ftest-coverage. Use -Og to have accurate coverage results and good performance
      elif bopt == 'O':
        flags.append('-g')
        if config.setCompilers.Configure.isClang(compiler, self.log):
          flags.append('-O3')
        else:
          flags.append('-O')
    else:
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler, self.log) and not compiler.find('win32fe') >=0:
        if bopt == '':
          flags.append('-wd1572')
          # next one fails in OpenMP build and we don't use it anyway so remove
          # flags.append('-Qoption,cpp,--extended_float_type')
        elif bopt == 'g':
          flags.extend(['-g','-O0'])
        elif bopt == 'O':
          flags.append('-g')
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          flags.extend(['-Qstd=c99'])
          if self.argDB['with-shared-libraries']:
            flags.extend(['-MD'])
          else:
            flags.extend(['-MT'])
        elif bopt == 'g':
          flags.extend(['-Z7','-Od'])
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          dir(self)
          if self.argDB['with-shared-libraries']:
            flags.extend(['-MD','-wd4996'])
          else:
            flags.extend(['-MT','-wd4996'])
        elif bopt == 'g':
          flags.extend(['-Z7','-Od'])
        elif bopt == 'O':
          flags.extend(['-O2', '-QxW'])
      # Windows Borland
      elif compiler.find('win32fe bcc32') >= 0:
        if bopt == '':
          flags.append('-RT -w-8019 -w-8060 -w-8057 -w-8004 -w-8066')
      if config.setCompilers.Configure.isNVCC(compiler, self.log):
        if bopt == 'g':
          flags.extend(['-lineinfo'])
        elif bopt == 'O':
          flags.append('-O3')
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.extend(['-g','-O0'])
      elif bopt == 'O':
        flags.append('-O')
    if bopt == 'O':
      self.logPrintBox('***** WARNING: Using default optimization '+language+' flags '+' '.join(flags)+'\nYou might consider manually setting optimal optimization flags for your system with\n '+language.upper()+'OPTFLAGS="optimization flags" see config/examples/arch-*-opt.py for examples')
    return flags

  def getCxxFlags(self, compiler, bopt):
    import config.setCompilers

    if [s for s in ['mpiCC','mpic++','mpicxx','mpiicxx','mpiicpc'] if os.path.basename(compiler).find(s)>=0]:
      try:
        output   = self.executeShellCommand(compiler+' -show', log = self.log)[0]
        self.framework.addMakeMacro('MPICXX_SHOW',output.strip().replace('\n','\\\\n'))
      except:
        self.framework.addMakeMacro('MPICXX_SHOW',"Unavailable")
    else:
      self.framework.addMakeMacro('MPICXX_SHOW',"Unavailable")

    flags = []
    # GNU g++
    if config.setCompilers.Configure.isGNU(compiler, self.log) or config.setCompilers.Configure.isClang(compiler, self.log):
      if bopt == '':
        flags.extend(['-Wall', '-Wwrite-strings', '-Wno-strict-aliasing','-Wno-unknown-pragmas'])
        if not any([
            # skip -fstack-protector for brew gcc - as this gives SEGV
            config.setCompilers.Configure.isDarwin(self.log) and config.setCompilers.Configure.isGNU(compiler, self.log),
            # hipcc for ROCm-4.0 crashes on some source files with -fstack-protector
            config.setCompilers.Configure.isHIP(compiler, self.log),
        ]):
          flags.extend(['-fstack-protector'])
        if config.setCompilers.Configure.isDarwinCatalina(self.log) and config.setCompilers.Configure.isClang(compiler, self.log):
          flags.extend(['-fno-stack-check'])
        # The option below would prevent warnings about compiling C as C++ being deprecated, but it causes Clang to SEGV, http://llvm.org/bugs/show_bug.cgi?id=12924
        # flags.extend([('-x','c++')])
        if self.argDB['with-visibility']:
          flags.extend(['-fvisibility=hidden'])
        arg = nargs.Arg.findArgument('with-errorchecking', self.clArgs)
        if not nargs.ArgBool('with-errorchecking', arg if arg is not None else '1', isTemporary=True).getValue():
          flags.extend(['-Wno-unused-but-set-variable'])
      elif bopt in ['g']:
        # -g3 causes an as SEGV on OSX
        flags.extend(['-g','-O0'])
      elif bopt == 'gcov':
        flags.extend(['--coverage','-Og'])
      elif bopt in ['O']:
        flags.append('-g')
        if 'USER' in os.environ:
          if config.setCompilers.Configure.isClang(compiler, self.log):
            flags.append('-O3')
          else:
            flags.append('-O')
    # IBM
    elif compiler.find('mpCC') >= 0 or compiler.find('xlC') >= 0:
      if bopt == '':
        flags.append('-qrtti=dyna')  # support dynamic casts in C++
      elif bopt in ['g']:
        flags.extend(['-g','-O0'])
      elif bopt in ['O']:
        flags.append('-O')
    else:
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler, self.log) and not compiler.find('win32fe') >=0:
        if bopt == '':
          flags.append('-wd1572')
        elif bopt == 'g':
          flags.extend(['-g','-O0'])
        elif bopt == 'O':
          flags.append('-g')
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          if self.argDB['with-shared-libraries']:
            flags.extend(['-MD','-GR','-EHsc'])
          else:
            flags.extend(['-MT','-GR','-EHsc']) # removing GX in favor of EHsc
        elif bopt in ['g']:
          flags.extend(['-Z7','-Od'])
        elif bopt in ['O']:
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          if self.argDB['with-shared-libraries']:
            flags.extend(['-MD','-GR','-EHsc'])
          else:
            flags.extend(['-MT','-GR','-EHsc']) # removing GX in favor of EHsc
        elif bopt == 'g':
          flags.extend(['-Z7','-Zm200','-Od'])
        elif bopt == 'O':
          flags.extend(['-O2','-QxW','-Zm200'])
      # Windows Borland
      elif compiler.find('win32fe bcc32') >= 0:
        if bopt == '':
          flags.append('-RT -w-8019 -w-8060 -w-8057 -w-8004 -w-8066')
    # Generic
    if not len(flags):
      if bopt in ['g']:
        flags.extend(['-g','-O0'])
      elif bopt in ['O']:
        flags.append('-O')
    if bopt == 'O':
      self.logPrintBox('***** WARNING: Using default C++ optimization flags '+' '.join(flags)+'\nYou might consider manually setting optimal optimization flags for your system with\n CXXOPTFLAGS="optimization flags" see config/examples/arch-*-opt.py for examples')
    return flags

  def getFortranFlags(self, compiler, bopt):

    if [s for s in ['mpif77','mpif90','mpifort','mpiifort'] if os.path.basename(compiler).find(s)>=0]:
      try:
        output   = self.executeShellCommand(compiler+' -show', log = self.log)[0]
        self.framework.addMakeMacro('MPIFC_SHOW',output.strip().replace('\n','\\\\n'))
      except:
        self.framework.addMakeMacro('MPIFC_SHOW',"Unavailable")
    else:
      self.framework.addMakeMacro('MPIFC_SHOW',"Unavailable")

    flags = []
    if config.setCompilers.Configure.isGNU(compiler, self.log):
      if bopt == '':
        flags.extend(['-Wall', '-ffree-line-length-0'])
        if config.setCompilers.Configure.isGfortran46plus(compiler, self.log):
          flags.extend(['-Wno-unused-dummy-argument']) # Silence warning because dummy parameters are sometimes necessary
        if not config.setCompilers.Configure.isGfortran47plus(compiler, self.log):
          flags.extend(['-Wno-unused-variable']) # older gfortran warns about unused common block constants
        if config.setCompilers.Configure.isGfortran45x(compiler, self.log):
          flags.extend(['-Wno-line-truncation']) # Work around bug in this series, fixed in 4.6: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=42852
      elif bopt == 'g':
        # g77 3.2.3 preprocesses the file into nothing if we give -g3
        flags.extend(['-g','-O0'])
      elif bopt == 'gcov':
        flags.extend(['--coverage','-Og'])
      elif bopt == 'O':
        flags.append('-g')
        flags.extend(['-O'])
    else:
      # Portland Group Fortran 90
      if config.setCompilers.Configure.isPGI(compiler, self.log):
        self.framework.addDefine('PETSC_HAVE_PGF90_COMPILER','1')
        if bopt == '':
          flags.append('-Mfree')
        elif bopt == 'O':
          flags.extend(['-fast', '-Mnoframe'])
      # Linux Intel
      if config.setCompilers.Configure.isIntel(compiler, self.log) and not compiler.find('win32fe') >=0:
        if bopt == 'g':
          flags.extend(['-g','-O0'])
        elif bopt == 'O':
          flags.append('-g')
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe ifl') >= 0 or compiler.find('win32fe ifort') >= 0:
        if bopt == '':
          if self.argDB['with-shared-libraries']:
            flags.extend(['-MD'])
          else:
            flags.extend(['-MT'])
        elif bopt == 'g':
         flags.extend(['-Z7','-Od'])
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Compaq Visual FORTRAN
      elif compiler.find('win32fe f90') >= 0 or compiler.find('win32fe df') >= 0:
        if bopt == '':
          flags.append('-threads')
        elif bopt == 'g':
          flags.extend(['-debug:full','-Od'])
        elif bopt == 'O':
          flags.extend(['-optimize:5', '-fast'])
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.extend(['-g','-O0'])
      elif bopt == 'O':
        flags.append('-O')
    if bopt == 'O':
      self.logPrintBox('***** WARNING: Using default FORTRAN optimization flags '+' '.join(flags)+'\nYou might consider manually setting optimal optimization flags for your system with\n FOPTFLAGS="optimization flags" see config/examples/arch-*-opt.py for examples')
    return flags

  def getCompilerFlags(self, language, compiler, bopt):
    if bopt == 'gcov' and (language == 'CUDA' or language == 'HIP' or language == 'SYCL'):
      return ''
    if bopt == 'gcov' and not config.setCompilers.Configure.isGNU(compiler, self.log) and not config.setCompilers.Configure.isClang(compiler, self.log):
      raise RuntimeError('Have --with-gcov but the compiler is neither GCC nor Clang, we do not know how to do gcov with other compilers')
    flags = ''
    if language == 'C' or language == 'CUDA':
      flags = self.getCFlags(compiler, bopt, language)
    elif language == 'Cxx' or language == 'HIP' or language == 'SYCL':
      flags = self.getCxxFlags(compiler, bopt)
    elif language in ['Fortran', 'FC']:
      flags = self.getFortranFlags(compiler, bopt)
    return flags

  def getCompilerVersion(self, language, compiler):
    if compiler is None:
      raise RuntimeError('Invalid compiler for version determination')
    version = 'Unknown'
    try:
      if language == 'C' or language == 'CUDA':
        if compiler.endswith('xlc') or compiler.endswith('mpcc'):
          flags = "lslpp -L vac.C | grep vac.C | awk '{print $2}'"
        else:
          flags = compiler+' --version'
      elif language == 'Cxx' or language == 'HIP' or language == 'SYCL':
        if compiler.endswith('xlC') or compiler.endswith('mpCC'):
          flags = "lslpp -L vacpp.cmp.core  | grep vacpp.cmp.core  | awk '{print $2}'"
        else:
          flags = compiler+' --version'
      elif language in ['Fortran', 'FC']:
        if compiler.endswith('xlf') or compiler.endswith('xlf90'):
          flags = "lslpp -L xlfcmp | grep xlfcmp | awk '{print $2}'"
        else:
          flags = compiler+' --version'
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(flags, log = self.log)
      except:
        flags = compiler+' -v'
        (output, error, status) = config.base.Configure.executeShellCommand(flags, log = self.log)
        output = error + output
      if not status:
        if compiler.find('win32fe') > -1:
          version = '\\n'.join(output.split('\n')[0:2])
          version = version.replace('\r','')
        else:
          #PGI/Windows writes an empty '\r\n' on the first line of output
          if output.count('\n') > 1 and output.split('\n')[0] == '\r':
            version = output.split('\r\n')[1]
          else:
            version = output.split('\n')[0]

    except RuntimeError as e:
      self.logWrite('Could not determine compiler version: '+str(e))
    self.logWrite('getCompilerVersion: '+str(compiler)+' '+str(version)+'\n')
    self.framework.addMakeMacro(language+'_VERSION',version)
    return version
