#!/usr/bin/env python
import user
import os
import re

try:
  import script
except ImportError:
  import sys

  if os.path.isdir('python'):
    sys.path.insert(0, os.path.join('python', 'BuildSystem'))
  elif os.path.isdir('..', 'python'):
    sys.path.insert(0, os.path.join(os.path.abspath('..'), 'python', 'BuildSystem'))
  import script

class BuildChecker(script.Script):
  def __init__(self):
    import RDict

    script.Script.__init__(self, argDB = RDict.RDict())
    return

  def setupHelp(self, help):
    import nargs

    help = script.Script.setupHelp(self, help)
    help.addArgument('BuildCheck', '-remoteMachine', nargs.Arg(None, 'terra.mcs.anl.gov', 'The machine on which PETSc logs are stored'))
    help.addArgument('BuildCheck', '-logDirectory',  nargs.Arg(None, os.path.join('/home', 'petsc', 'logs', 'nightly'), 'The directory in which PETSc logs are stored'))
    help.addArgument('BuildCheck', '-archCompilers', nargs.Arg(None, {}, 'A mapping from architecture names to lists of compiler names'))
    return help

  compilers = {'aix5.1.0.0':           ['ibm'],
               'cygwin':               ['gcc'],
               'cygwin-borland':       ['win32fe', 'borland'],
               'cygwin-ms':            ['win32fe', 'ms'],
               'freebsd5.1':           ['gcc'],
               'linux':                ['gcc'],
               'linux-gnu':            ['gcc'],
               'linux-gnu-amd64':      ['gcc'],
               'linux-gnu-gcc-absoft': ['gcc', 'absoftF90'],
               'linux-gnu-gcc-ifc':    ['gcc', 'intelF90'],
               'linux-gnu-gcc-pgf90':  ['gcc', 'pgF90'],
               'linux-gnu-ia64':       ['gcc'],
               'linux-gnu-ia64-intel': ['intel', 'intelF90'],
               'linux-gnu-uni':        ['gcc'],
               'linux-gnu-valgrind':   ['gcc'],
               'linux-gnu-alpha':      ['gcc'],
               'linux-mcs':            ['gcc'],
               'macosx-gnu':           ['gcc'],
               'macosx-ibm':           ['ibm'],
               'macosx-nagf90':        ['gcc'],
               'osf5.0':               ['mipsUltrix'],
               'solaris2.9':           ['solaris'],
               'solaris-gnu':          ['gcc'],
               'solaris2.9-lam':       ['solaris'],
               'solaris-uni':          ['solaris'],
               # Untested architectures
               'irix6.5':         ['sgiMipsPro'],
               't3e':             ['cray'],
               'win32_borland':   ['win32fe', 'borland'],
               'win32_ms_mpich':  ['win32fe', 'ms']}

  # Stolen (like at good code) from the XEmacs compile.el
  #   Each compiler is mapped to a list of tuples
  #   Each tuple has the following structure:
  #     - The regular expression matching compiler output
  #     - The match number containing the filename
  #     - The match number containing the line number
  compileErrorRE = {
    ## Absoft Fortran 90
    ##   ???
    ## Absoft FORTRAN 77 Compiler 3.1.3
    ##   error on line 19 of fplot.f: spelling error?
    ##   warning on line 17 of fplot.f: data type is undefined for variable d
    'absoftF90': [r'[^\n]*(?P<type>error|warning) on line[ \t]+(?P<line>[0-9]+)[ \t]+of[ \t]+([^:\n]+):'],
    ## Borland C++ 5.5.1 for Win32 Copyright (c) 1993, 2000 Borland
    ##   Error E2303 h:\home\balay\PETSC-~1\include\petscsys.h 89: Type name expected
    ##   Error ping.c 15: Unable to open include file 'sys/types.h'
    ##   Warning ping.c 68: Call to function 'func' with no prototype
    'borland': [r'(?P<type>Error|Warning) (?P<subtype>[EW][0-9]+) (?P<filename>[a-zA-Z]?:?[^:(\s]+) (?P<line>[0-9]+)([) \t]|:[^0-9\n])'],
    ## Cray C compiler error messages
    ##   CC-29 CC: ERROR File = pcregis.c, Line = 82
    'cray': [r'[^\n]*: (?P<type>ERROR|WARNING) File = (?P<filename>[^,\n]+), Line = (?P<line>[0-9]+)'],
    ## GCC 3.0.4
    ##   /usr/local/qhull/include/qhull.h:38:5: warning: "__MWERKS__" is not defined
    ##   pcregis.c:82: parse error
    ##   pcregis.c:82:34: operator '&&' has no right operand
    'gcc': [r'(?P<filename>[^:\s]+):(?P<line>[0-9]+):((?P<column>[0-9]+):)? (?P<type>error|warning|parse error)?'],
    ## IBM RS6000
    ##   "vvouch.c", line 19.5: 1506-046 (S) Syntax error.
    ##   "pcregis.c", line 82.34: 1540-186: (S) The expression is not a valid preprocessor constant expression.
    ## IBM AIX xlc compiler
    ##   "src/swapping.c", line 30.34: 1506-342 (W) "/*" detected in comment.
    'ibm': [r'[^\n]*"(?P<filename>[^,"\s]+)", line (?P<line>[0-9]+)\.(?P<column>[0-9]+): (?P<subtype>[0-9]+-[0-9]+):? \((?P<type>\w)\)'],
    ## Intel C/C++ 8.0
    ##   matptapf.c(81): error: incomplete type is not allowed
    ##   matptapf.c(99): warning #12: parsing restarts here after previous syntax error
    'intel': [r'(?P<filename>[^\(]+)\((?P<line>[0-9]+)\): (?P<type>error|warning)( #(?P<num>[0-9]+))?:'],
    ## Intel Fortran 90
    ##   ??? (Using gcc)
    'intelF90': [r'(?P<filename>[^:\s]+):(?P<line>[0-9]+):((?P<column>[0-9]+):)? (?P<type>error|warning):'],
    ## MIPS RISC CC - the one distributed with Ultrix
    ##   ccom: Error: foo.c, line 2: syntax error
    ## DEC AXP OSF/1 cc
    ##   /usr/lib/cmplrs/cc/cfe: Error: foo.c: 1: blah blah
    ## Tru64 UNIX Compiler Driver 5.0, Compaq C V6.1-019 on Compaq Tru64 UNIX V5.0A (Rev. 1094)
    ##   cxx: Warning: gs.c, line 668: statement either is unreachable or causes unreachable code
    ##   cc: Error: matptapf.c, line 81: Missing ";". (nosemi)
    ##   cc: Severe: /usr/sandbox/petsc/petsc-dev/include/petscmath.h, line 33: Cannot find file <complex> specified in #include directive. (noinclfile)
    'mipsUltrix': [r'[^\n]*(?P<type>Error|Warning|Severe): (?P<filename>[^,"\s]+)[,:] (line )?(?P<line>[0-9]+):'],
    ## Microsoft C/C++:
    ##   keyboard.c(537) : warning C4005: 'min' : macro redefinition
    ##   d:\tmp\test.c(23) : error C2143: syntax error : missing ';' before 'if'
    ##   c:\home\petsc\PETSC-~1\src\sles\pc\INTERF~1\pcregis.c(82) : fatal error C1017: invalid integer constant expression
    'ms': [r'(?P<filename>([a-zA-Z]:)?[^:(\s]+)\((?P<line>[0-9]+)\)[ \t]*:[ \t]*(?P<type>warning|error|fatal error) (?P<subtype>[^:\n]+):'],
    ## Portland Group Fortran 90
    ##   ??? (Using gcc)
    'pgF90': [r'(?P<filename>[^:\s]+):(?P<line>[0-9]+):((?P<column>[0-9]+):)? (?P<type>error|warning):'],
    ## IRIX 5.2
    ##   cfe: Warning 712: foo.c, line 2: illegal combination of pointer and ...
    ##   cfe: Warning 600: xfe.c: 170: Not in a conditional directive while ...
    'sgi': [r'[^\n]*(?P<type>Error|Warning): (?P<filename>[^,"\s]+)[,:] (line )?(?P<line>[0-9]+):'],
    ## SGI Mipspro 7.3 compilers
    ##   cc-1020 CC: ERROR File = CUI_App.h, Line = 735
    ##   cc-1174 CC: WARNING File = da1.c, Line = 136
    'sgiMipsPro': [r'^cc-[0-9]* (cc|CC|f77): (?P<type>REMARK|WARNING|ERROR) File = (?P<filename>.*), Line = (?P<line>[0-9]*)'],
    ## WorkShop Compilers 5.0 98/12/15 C++ 5.0
    ##   "dl.c", line 259: Warning (Anachronism): Cannot cast from void* to int(*)(const char*).
    'solaris': [r'[^\n]*"(?P<filename>[^,"\s]+)", line (?P<line>[0-9]+): (?P<type>Warning|Error)( \((?P<subtype>\w+)\):)?'],
    ## Win32fe, Petsc front end for Windows compilers
    ##   Warning: win32fe Include Path Not Found: /home/balay/petsc-test
    'win32fe': [r'(?P<type>Warning|Error): (?P<filename>win32fe)']
    }

  def flatten(self, l):
    flat = []
    if not isinstance(l, list) and not isinstance(l, tuple):
      return [l]
    for item in l:
      flat.extend(self.flatten(item))
    return flat

  def checkFile(self, filename):
    ##logRE = r'build_(?P<arch>[\w-]*\d+\.\d+)\.(?P<bopt>[\w+]*)\.(?P<machine>[\w@.]*)\.log'
    logRE = r'build_(?P<arch>[\w.\d-]+)_(?P<machine>[\w.\d-]+)\.log'
    configureRE = re.compile(r'\*{3,5} (?P<errorMsg>[^*]+) \*{3,5}')

    print 'Checking',filename
    if self.isLocal and not os.path.exists(filename):
      raise RuntimeError('Invalid filename: '+filename)
    m = re.match(logRE, os.path.basename(filename))
    if not m:
      m = re.match(logRE, os.path.basename(filename))
      if not m:
        raise RuntimeError('Invalid filename '+filename)
    arch    = m.group('arch')
    machine = m.group('machine')
    print arch,machine
    if arch in self.compilers:
      compilers = self.compilers[arch]
    elif arch in self.argDB['archCompilers']:
      compilers = self.argDB['archCompilers'][arch]
    else:
      raise RuntimeError('No compilers for architecture '+arch)
    try:
      # Why doesn't Python have a fucking flatten
      regExps = map(re.compile, self.flatten([self.compileErrorRE[compiler] for compiler in compilers]))
    except KeyError:
      raise RuntimeError('No regular expressions for compiler '+compiler)

    if self.isLocal:
      f     = file(filename)
      lines = f.xreadlines()
    else:
      import tempfile

      (output, error, status) = self.executeShellCommand('ssh '+self.argDB['remoteMachine']+' cat '+filename)
      lines = output.split('\n')
    for line in lines:
      m = configureRE.search(line)
      if m:
        print 'From '+filename+': configure error: '+m.group('errorMsg')
        continue
      for regExp in regExps:
        m = regExp.match(line)
        if m:
          # For Solaris
          try:
            if m.group('subtype') == 'Anachronism': continue
          except IndexError:
            pass
          # Skip configure log
          try:
            if m.group('filename') == 'conftest.c': continue
          except IndexError:
            pass
          try:
            type = m.group('type')
            if not type: type = 'Error'
            print 'From '+filename+': '+type+' in file '+m.group('filename')+' on line '+m.group('line')
          except IndexError:
            # For win32fe
            print 'From '+filename+': '+m.group('type')+' for '+m.group('filename')
    if self.isLocal:
      f.close()
    return

  def getBuildFileNames(self):
    buildRE = re.compile(r'^.*build_.*$')

    if self.isLocal:
      files = os.listdir(self.argDB['logDirectory'])
    else:
      (output, error, status) = self.executeShellCommand('ssh '+self.argDB['remoteMachine']+' ls -1 '+self.argDB['logDirectory'])
      files = output.split('\n')
    print files
    return filter(lambda fname: buildRE.match(fname), files)

  def run(self):
    self.setup()
    self.isLocal = os.path.isdir(self.argDB['logDirectory'])
    map(lambda f: self.checkFile(os.path.join(self.argDB['logDirectory'], f)), self.getBuildFileNames())
    return

if __name__ == '__main__':
  BuildChecker().run()
