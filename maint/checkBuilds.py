#!/usr/bin/env python
import os
import re
import sys

class BuildChecker:
  compilers = {'IRIX':            ['sgiMipsPro'],
               'IRIX64':          ['sgiMipsPro'],
               'alpha':           ['mipsUltrix'],
               'freebsd':         ['gcc'],
               'linux':           ['gcc'],
               'linux64':         ['gcc'],
               'linux_absoft':    ['gcc', 'absoftF90'],
               'linux_alpha':     ['gcc'],
               'linux_alpha_dec': ['mipsUltrix'],
               'linux_gcc_pgf90': ['gcc', 'pgF90'],
               'macx':            ['gcc'],
               'rs6000_64':       ['ibm'],
               'rs6000_gnu':      ['gcc'],
               'rs6000_sp':       ['ibm'],
               'solaris':         ['solaris'],
               'solaris64':       ['solaris'],
               't3e':             ['cray'],
               'win32_borland':   ['win32fe', 'borland'],
               'win32_gnu':       ['gcc'],
               'win32_ms_mpich':  ['win32fe', 'ms']}

  # Stolen (like at good code) from the XEmacs compile.el
  #   Each compiler is mapped to a list of tuples
  #   Each tuple has the following structure:
  #     - The regular expression matching compiler output
  #     - The match number containing the filename
  #     - The match number containing the line number
  compileErrorRE = {
    ## SGI Mipspro 7.3 compilers
    ##   cc-1020 CC: ERROR File = CUI_App.h, Line = 735
    'sgiMipsPro': [r'^cc-[0-9]* (cc|CC|f77): (?P<type>REMARK|WARNING|ERROR) File = (?P<filename>.*), Line = (?P<line>[0-9]*)'],
    ## IRIX 5.2
    ##   cfe: Warning 712: foo.c, line 2: illegal combination of pointer and ...
    ##   cfe: Warning 600: xfe.c: 170: Not in a conditional directive while ...
    'sgi': [r'[^\n]*(?P<type>Error|Warning): (?P<filename>[^,"\s]+)[,:] (line )?(?P<line>[0-9]+):'],
    ## MIPS RISC CC - the one distributed with Ultrix
    ##   ccom: Error: foo.c, line 2: syntax error
    ## DEC AXP OSF/1 cc
    ##   /usr/lib/cmplrs/cc/cfe: Error: foo.c: 1: blah blah
    ## Tru64 UNIX Compiler Driver 5.0, Compaq C V6.1-019 on Compaq Tru64 UNIX V5.0A (Rev. 1094)
    ##   cxx: Warning: gs.c, line 668: statement either is unreachable or causes unreachable code
    'mipsUltrix': [r'[^\n]*(?P<type>Error|Warning): (?P<filename>[^,"\s]+)[,:] (line )?(?P<line>[0-9]+):'],
    ## GCC 3.0.4
    ##   /usr/local/qhull/include/qhull.h:38:5: warning: "__MWERKS__" is not defined
    ##   pcregis.c:82: parse error
    ##   pcregis.c:82:34: operator '&&' has no right operand
    'gcc': [r'(?P<filename>[^:\s]+):(?P<line>[0-9]+):((?P<column>[0-9]+):)? (?P<type>error|warning|parse error)?'],
    ## Absoft Fortran 90
    ##   ???
    ## Absoft FORTRAN 77 Compiler 3.1.3
    ##   error on line 19 of fplot.f: spelling error?
    ##   warning on line 17 of fplot.f: data type is undefined for variable d
    'absoftF90': [r'[^\n]*(?P<type>error|warning) on line[ \t]+(?P<line>[0-9]+)[ \t]+of[ \t]+([^:\n]+):'],
    ## Portland Group Fortran 90
    ##   ??? (Using gcc)
    'pgF90': [r'(?P<filename>[^:\s]+):(?P<line>[0-9]+):((?P<column>[0-9]+):)? (?P<type>error|warning):'],
    ## IBM RS6000
    ##   "vvouch.c", line 19.5: 1506-046 (S) Syntax error.
    ##   "pcregis.c", line 82.34: 1540-186: (S) The expression is not a valid preprocessor constant expression.
    ## IBM AIX xlc compiler
    ##   "src/swapping.c", line 30.34: 1506-342 (W) "/*" detected in comment.
    'ibm': [r'[^\n]*"(?P<filename>[^,"\s]+)", line (?P<line>[0-9]+)\.(?P<column>[0-9]+): (?P<subtype>[0-9]+-[0-9]+):? \((?P<type>\w)\)'],
    ## WorkShop Compilers 5.0 98/12/15 C++ 5.0
    ##   "dl.c", line 259: Warning (Anachronism): Cannot cast from void* to int(*)(const char*).
    'solaris': [r'[^\n]*"(?P<filename>[^,"\s]+)", line (?P<line>[0-9]+): (?P<type>Warning|Error)( \((?P<subtype>\w+)\):)?'],
    ## Cray C compiler error messages
    ##   CC-29 CC: ERROR File = pcregis.c, Line = 82
    'cray':
    [r'[^\n]*: (?P<type>ERROR|WARNING) File = (?P<filename>[^,\n]+), Line = (?P<line>[0-9]+)'],
    ## Borland C++ 5.5.1 for Win32 Copyright (c) 1993, 2000 Borland
    ##   Error ping.c 15: Unable to open include file 'sys/types.h'
    ##   Warning ping.c 68: Call to function 'func' with no prototype
    'borland': [r'(?P<type>Error|Warning) (?P<subtype>[EW][0-9]+) (?P<filename>[a-zA-Z]?:?[^:(\s]+) (?P<line>[0-9]+)([) \t]|:[^0-9\n])'],
    ## Microsoft C/C++:
    ##   keyboard.c(537) : warning C4005: 'min' : macro redefinition
    ##   d:\tmp\test.c(23) : error C2143: syntax error : missing ';' before 'if'
    ##   c:\home\petsc\PETSC-~1\src\sles\pc\INTERF~1\pcregis.c(82) : fatal error C1017: invalid integer constant expression
    'ms': [r'(?P<filename>([a-zA-Z]:)?[^:(\s]+)\((?P<line>[0-9]+)\)[ \t]*:[ \t]*(?P<type>warning|error|fatal error) (?P<subtype>[^:\n]+):'],
    ## Win32fe, Petsc front end for Windows compilers
    ##   Warning: win32fe Include Path Not Found: /home/balay/petsc-test
    'win32fe': [r'(?P<type>Warning|Error): (?P<filename>win32fe)']
    }

  def __init__(self, filename):
    self.filename = filename

  def flatten(self, l):
    flat = []
    if not isinstance(l, list) and not isinstance(l, tuple):
      return [l]
    for item in l:
      flat.extend(self.flatten(item))
    return flat

  def run(self):
    if not os.path.exists(self.filename):
      raise RuntimeError('Invalid filename: '+filename)
    m = re.match(r'build_(?P<arch>\w*)\.(?P<bopt>[\w+]*)\.(?P<machine>[\w@.]*)\.log', os.path.basename(self.filename))
    if not m:
      raise RuntimeError('Invalid filename '+self.filename)
    try:
      compilers = self.compilers[m.group('arch')]
    except KeyError:
      raise RuntimeError('No compilers for architecture '+m.group('arch'))
    try:
      # Why doesn't Python have a fucking flatten
      regExps = map(re.compile, self.flatten([self.compileErrorRE[compiler] for compiler in compilers]))
    except KeyError:
      raise RuntimeError('No regular expressions for compiler '+compiler)

    f = file(self.filename)
    for line in f.xreadlines():
      for regExp in regExps:
        m = regExp.match(line)
        if m:
          # For Solaris
          try:
            if m.group('subtype') == 'Anachronism': continue
          except IndexError:
            pass
          try:
            type = m.group('type')
            if not type: type = 'Error'
            print 'From '+self.filename+': '+type+' in file '+m.group('filename')+' on line '+m.group('line')
          except IndexError:
            # For win32fe
            print 'From '+self.filename+': '+m.group('type')+' for '+m.group('filename')

if __name__ == '__main__': map(lambda filename: BuildChecker(filename).run(), sys.argv[1:])
