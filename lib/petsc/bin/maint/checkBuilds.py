#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import re

try:
  import script
except ImportError:
  import sys

  if os.path.isdir('config'):
    sys.path.insert(0, os.path.join('config', 'BuildSystem'))
  elif os.path.isdir(os.path.join('..', 'config')):
    sys.path.insert(0, os.path.join(os.path.abspath('..'), 'config', 'BuildSystem'))
  import script

class BuildChecker(script.Script):
  def __init__(self):
    import RDict

    script.Script.__init__(self, argDB = RDict.RDict())

    # (commit of log file, file name) -> (line numbers)
    self.commitfileDict = {}
    # (commit of log file, file name, line number) -> (warnings)
    self.filelineDict = {}
    # (author) -> (offending commit, file name:line number, warnings)
    self.blameDict = {}
    return

  def setupHelp(self, help):
    import nargs,datetime

    help = script.Script.setupHelp(self, help)
    help.addArgument('BuildCheck', '-remoteMachine',    nargs.Arg(None, 'login.mcs.anl.gov', 'The machine on which PETSc logs are stored'))
    help.addArgument('BuildCheck', '-logDirectory',     nargs.Arg(None, os.path.join('/mcs', 'ftp', 'pub', 'petsc','nightlylogs'), 'The directory in which PETSc logs are stored'))
    help.addArgument('BuildCheck', '-archCompilers',    nargs.Arg(None, {}, 'A mapping from architecture names to lists of compiler names'))
    help.addArgument('BuildCheck', '-blameMail',        nargs.ArgBool(None, 1, 'Generate blame emails'))
    help.addArgument('BuildCheck', '-blameMailPost',    nargs.ArgBool(None, 1, 'Post (send) blame emails'))
    help.addArgument('BuildCheck', '-ignoreDeprecated', nargs.ArgBool(None, 1, 'Ignore deprecated warnings'))
    help.addArgument('BuildCheck', '-ignorePragma',     nargs.ArgBool(None, 1, 'Ignore unknown pragma'))
    help.addArgument('BuildCheck', '-ignoreNote',       nargs.ArgBool(None, 1, 'Ignore note warnings'))
    help.addArgument('BuildCheck', '-blameMailDate',    nargs.Arg(None, str(datetime.date.today()), 'Date given in blame mail subject'))
    help.addArgument('BuildCheck', '-buildBranch',      nargs.Arg(None, '', 'Check build logs corresponding to given branch name'))
    return help


  compilers = {'aix5.1.0.0':                      ['ibm'],
               'cygwin-borland':                  ['win32fe', 'borland'],
               'cygwin-ms':                       ['win32fe', 'ms'],
               'arch-mswin':                      ['win32fe', 'ms'],
               'arch-mswin-cxx-cmplx':            ['win32fe', 'ms'],
               'linux-gnu-gcc-absoft':            ['gcc', 'absoftF90'],
               'linux-gnu-gcc-ifc':               ['gcc', 'intelF90'],
               'linux-gnu-gcc-pgf90':             ['gcc', 'pgF90'],
               'linux-gnu-ia64-intel':            ['intel', 'intelF90'],
               'linux-rhAS3-intel81' :            ['intel', 'intelF90'],
               'macosx-ibm':                      ['ibm'],
               'osf5.0':                          ['mipsUltrix'],
               'solaris2.9':                      ['solaris'],
               'solaris2.9-lam':                  ['solaris'],
               'solaris-uni':                     ['solaris'],
               'arch-opensolaris':                ['solaris'],
               'arch-opensolaris-cmplx-pkgs-dbg': ['solaris'],
               'arch-opensolaris-misc':           ['solaris'],
               'arch-opensolaris-pkgs-opt':       ['solaris'],
               'arch-pardiso':                    ['intel'],
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
    'solaris': [r'[^\n]*"(?P<filename>[^,"\s]+)", line (?P<line>[0-9]+): (?P<type>[Ww]arning|[Ee]rror)( \((?P<subtype>\w+)\):)?'],
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

  def fileNameToRelPath(self, filename, petscdir, arch):
    ''' we're not on the systems that made the output, plus there could be some mswin/cygwin issues '''
    topabsdir = petscdir
    absfile = filename
    if re.search(r'freebsd',arch):
      ''' /home aliases /usr/home: strip /usr '''
      topabsdir = re.sub(r'^/usr','',topabsdir)
      absfile   = re.sub(r'^/usr','',absfile)
    if re.search(r'mswin',arch):
      ''' normalize to unix directory notation '''
      topabsdir = '/petscdir'
      ''' uuuuuuugggggghhh '''
      absfile = re.sub(r'^.*PETSC\~\d\.CLO','/petscdir',absfile)
      absfile = re.sub(r'TUTORI\~\d+','tutorials',absfile)
      absfile = re.sub(r'POWER_\~\d+','power_grid',absfile)
      absfile = re.sub(r'STABIL\~\d+','stability_9bus',absfile)
      absfile = re.sub(r'\\','/',absfile)
    relpath = os.path.relpath(absfile,topabsdir)
    return relpath

  def addLineBlameDict(self,line,filename,ln,petscdir,commit,arch,logfile):
    # avoid solaris compiler errors
    if re.search(r'warning: loop not entered at top',line): return
    if re.search(r'warning: statement not reached',line): return
    # avoid C++ instantiation sequences
    if re.search(r'instantiated from here',line):      return
    # avoid MPI argument checks that cannot handle const
    if re.search(r"\(aka 'const double \*'\) doesn't match specified 'MPI' type tag that requires 'double \*'",line): return
    if re.search(r"\(aka 'const int \*'\) doesn't match specified 'MPI' type tag that requires 'int \*'",line): return
    # avoid MPI argument checks that cannot handle long long * vs long *
    if re.search(r"\(aka 'long \*'\) doesn't match specified 'MPI' type tag that requires 'long long \*",line): return
    if re.search(r"\(aka 'const long \*'\) doesn't match specified 'MPI' type tag that requires 'long long \*",line): return
    # avoid Microsoft VC's dubious macro-expansion warnings: /questions/10684169/preprocessor-macros-as-parameters-to-other-macros */
    if re.search(r"warning C4003: not enough actual parameters for macro",line): return
    if self.argDB['ignoreDeprecated'] and re.search(r'deprecated',line):  return
    if self.argDB['ignorePragma'] and re.search(r'unrecognized #pragma',line):  return
    message = line.rstrip()
    if self.argDB['ignoreNote'] and re.search(r'note:',line):
      return
    relpath = self.fileNameToRelPath(filename,petscdir,arch)
    message = '['+os.path.join(self.logurl,logfile)+']\n      '+message
    if (commit,relpath) not in self.commitfileDict:
      self.commitfileDict[(commit,relpath)] = {ln}
    else:
      self.commitfileDict[(commit,relpath)].add(ln)
    if (commit,relpath,ln) not in self.filelineDict:
      self.filelineDict[(commit,relpath,ln)] = {message}
    else:
      self.filelineDict[(commit,relpath,ln)].add(message)

  def checkFile(self, filename):
    ##logRE = r'build_(?P<arch>[\w-]*\d+\.\d+)\.(?P<bopt>[\w+]*)\.(?P<machine>[\w@.]*)\.log'
    logRE = r'(build|examples)_(?P<branch>[\w.\d-]+)_(?P<arch>[\w.\d-]+)_(?P<machine>[\w.\d-]+)\.log'
    commitRE = re.compile(r'^commit: (?P<commit>[0-9a-z]{40})')
    petscdirRE = re.compile(r'PETSC_DIR[:= ]+(?P<petscdir>\S+)')
    configureRE = re.compile(r'\*{3,5} (?P<errorMsg>[^*]+) \*{3,5}')
    addBlameDict = self.argDB['blameMail']
    commit = ''
    petscdir = ''
    filelinedict = {}

    print('Checking',filename)
    if self.isLocal and not os.path.exists(filename):
      raise RuntimeError('Invalid filename: '+filename)
    m = re.match(logRE, os.path.basename(filename))
    if not m:
      m = re.match(logRE, os.path.basename(filename))
      if not m:
        raise RuntimeError('Invalid filename '+filename)
    arch    = m.group('arch')
    branch  = m.group('branch')
    machine = m.group('machine')
    if arch in self.compilers:
      compilers = self.compilers[arch]
    elif arch in self.argDB['archCompilers']:
      compilers = self.argDB['archCompilers'][arch]
    else:
      # default to gnu
      compilers = ['gcc']
    try:
      # Why doesn't Python have a fucking flatten
      regExps = map(re.compile, self.flatten([self.compileErrorRE[compiler] for compiler in compilers]))
    except KeyError:
      raise RuntimeError('No regular expressions for compiler '+compiler)

    if self.isLocal:
      f     = file(filename)
      lines = f
    else:
      import tempfile

      (output, error, status) = self.executeShellCommand('ssh '+self.argDB['remoteMachine']+' cat '+filename)
      lines = output.split('\n')
    for line in lines:
      ''' figure out the version of the code that generated the output '''
      if not commit:
        matchCommit = commitRE.match(line)
        if matchCommit:
          commit = matchCommit.group('commit')
      ''' figure out the topdir '''
      if not petscdir:
        matchPetscdir = petscdirRE.search(line)
        if matchPetscdir:
          petscdir = matchPetscdir.group('petscdir')
      m = configureRE.search(line)
      if m:
        print('From '+filename+': configure error: '+m.group('errorMsg'))
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
            print('From '+filename+': '+type+' in file '+m.group('filename')+' on line '+m.group('line'))
            if addBlameDict and commit and petscdir:
              self.addLineBlameDict(line,m.group('filename'),m.group('line'),petscdir,commit,arch,os.path.basename(filename))
          except IndexError:
            # For win32fe
            print('From '+filename+': '+m.group('type')+' for '+m.group('filename'))
    if self.isLocal:
      f.close()
    return

  def getBuildFileNames(self):
    buildRE = re.compile(r'^.*(build|examples)_'+self.argDB['buildBranch']+'.*$')
    buildExcludeRE = re.compile(r'^.*(build|examples)_'+self.argDB['buildBranch']+'_arch-linux-analyzer'+'.*$')

    if self.isLocal:
      files = os.listdir(self.argDB['logDirectory'])
    else:
      (output, error, status) = self.executeShellCommand('ssh '+self.argDB['remoteMachine']+' ls -1 '+self.argDB['logDirectory'])
      files = output.split('\n')
    print(files)
    print(filter(lambda fname: buildRE.match(fname) and not buildExcludeRE.match(fname), files))
    return filter(lambda fname: buildRE.match(fname) and not buildExcludeRE.match(fname), files)

  def blameMail(self):
    for key in sorted(self.commitfileDict.keys()):
      lns = sorted(self.commitfileDict[key])
      pairs = [ln+','+ln for ln in sorted(self.commitfileDict[key])]
      output=''
      try:
        # Requires git version 1.9 or newer!
        git_blame_cmd = 'git blame -w -M --line-porcelain --show-email -L '+' -L '.join(pairs)+' '+key[0]+' -- '+key[1]
        (output, error, status) = self.executeShellCommand(git_blame_cmd)
      except:
        print('Error running:',git_blame_cmd)
      if output:
        blamelines = output.split('\n')
        current = -1
        author = 'Unknown author'
        email = '<unknown@author>'
        commit = '(unknown commit)'
        for bl in blamelines:
          if re.match(r'^[0-9a-z]{40}',bl):
            if current >= 0:
              warnings = self.filelineDict[(key[0],key[1],lns[current])]
              fullauthor = author+' '+email
              if not fullauthor in self.blameDict:
                self.blameDict[fullauthor] = [(commit,key[1]+":"+lns[current],warnings)]
              else:
                self.blameDict[fullauthor].append((commit,key[1]+":"+lns[current],warnings))
            commit = bl[0:7]
            current = current+1
            author = ''
            email = ''
          m = re.match(r'^author (?P<author>.*)',bl)
          if m:
            author =  m.group('author')
          m = re.match(r'^author-mail (?P<mail>.*)',bl)
          if m:
            email =  m.group('mail')
          m = re.match(r'^summary (?P<summary>.*)',bl)
          if m:
            commit = "https://gitlab.com/petsc/petsc/commit/" + commit + '\n' + m.group('summary')
        warnings = self.filelineDict[(key[0],key[1],lns[current])]
        fullauthor = author+' '+email
        if not fullauthor in self.blameDict:
          self.blameDict[fullauthor] = [(commit,key[1]+":"+lns[current],warnings)]
        else:
          self.blameDict[fullauthor].append((commit,key[1]+":"+lns[current],warnings))
    for author in self.blameDict.keys():

      buf ='''Dear PETSc developer,

This email contains listings of contributions attributed to you by
`git blame` that caused compiler errors or warnings in PETSc automated
testing.  Follow the links to see the full log files. Please attempt to fix
the issues promptly or let us know at petsc-dev@mcs.anl.gov if you are unable
to resolve the issues.

Thanks,
  The PETSc development team
'''

      allwarnings = self.blameDict[author]
      allwarnings = sorted(allwarnings)
      for i in range(0,len(allwarnings)):
        newcommit = False
        newline = False
        warning = allwarnings[i]
        if i == 0 or not warning[0] == allwarnings[i-1][0]:
          buf +="\n----\n\nwarnings attributed to commit %s\n" % warning[0]
          newcommit = True
        if newcommit or not warning[1] == allwarnings[i-1][1]:
          buf +="\n  %s\n" % warning[1]
        for message in warning[2]:
          buf +="    %s\n" % message
      buf += '\n----\nTo opt-out from receiving these messages - send a request to petsc-dev@mcs.anl.gov.\n'

      #The followng chars appear to cause grief to sendmail - so replace them
      buf = buf.replace("‘","'").replace("’","'")

      # now send e-mail
      import smtplib
      from email.mime.text import MIMEText

      if author in ['Mark Adams <mark.adams@columbia.edu>','Mark Adams <cal2princeton@yahoo.com>'] :
        author =  'Mark Adams <mfadams@lbl.gov>'
      if author == 'Karl Rupp <rupp@mcs.anl.gov>':
        author =  'Karl Rupp <me@karlrupp.net>'

      checkbuilds = 'PETSc checkBuilds <petsc-checkbuilds@mcs.anl.gov>'
      dev = 'petsc-dev <petsc-dev@mcs.anl.gov>'
      today = self.argDB['blameMailDate']
      FROM = checkbuilds
      TO = [author,checkbuilds]
      REPLY_TO = [dev,checkbuilds]

      msg = MIMEText(buf)
      msg['From'] = FROM
      msg['To'] = ','.join(TO)
      msg['Reply-To'] = ','.join(REPLY_TO)
      msg['Subject'] = "PETSc blame digest (%s) %s\n\n" % (self.argDB['buildBranch'], today)

      if author in ['Peter Brune <brune@mcs.anl.gov>']:
        print('Skipping sending e-mail to:',author)
      elif self.argDB['blameMailPost']:
        server = smtplib.SMTP('localhost')
        server.sendmail(FROM, TO, msg.as_string())
        server.quit()

      # create log of e-mails sent in PETSC_DIR
      justaddress = re.search(r'<(?P<address>.*)>',author).group('address')
      mailname = '-'.join(['blame',self.argDB['buildBranch'],today,justaddress])
      mail = open(mailname,"w")
      mail.write(msg.as_string())
      mail.close()

  def run(self):
    self.setup()
    self.isLocal = os.path.isdir(self.argDB['logDirectory'])
    if self.argDB['logDirectory'].startswith('/mcs/ftp/pub/petsc/nightlylogs/'):
      self.logurl = self.argDB['logDirectory'].replace('/mcs/ftp/pub/petsc/nightlylogs/','http://ftp.mcs.anl.gov/pub/petsc/nightlylogs//')
    else:
      self.logurl=''
    print(self.isLocal)
    map(lambda f: self.checkFile(os.path.join(self.argDB['logDirectory'], f)), self.getBuildFileNames())
    if self.argDB['blameMail']:
      self.blameMail()

    return

if __name__ == '__main__':
  print('')
  print('Logs located at http://ftp.mcs.anl.gov/pub/petsc/nightlylogs/')
  print('')
  BuildChecker().run()
