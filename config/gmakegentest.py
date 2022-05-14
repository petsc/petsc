#!/usr/bin/env python3

from __future__ import print_function
import pickle
import os,shutil, string, re
import sys
import logging, time
import types
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from collections import defaultdict
from gmakegen import *

import inspect
thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,thisscriptdir)
import testparse
import example_template


"""

There are 2 modes of running tests: Normal builds and run from prefix of
install.  They affect where to find things:


Case 1.  Normal builds:

     +---------------------+----------------------------------+
     | PETSC_DIR           | <git dir>                        |
     +---------------------+----------------------------------+
     | PETSC_ARCH          | arch-foo                         |
     +---------------------+----------------------------------+
     | PETSC_LIBDIR        | PETSC_DIR/PETSC_ARCH/lib         |
     +---------------------+----------------------------------+
     | PETSC_EXAMPLESDIR   | PETSC_DIR/src                    |
     +---------------------+----------------------------------+
     | PETSC_TESTDIR       | PETSC_DIR/PETSC_ARCH/tests       |
     +---------------------+----------------------------------+
     | PETSC_GMAKEFILETEST | PETSC_DIR/gmakefile.test         |
     +---------------------+----------------------------------+
     | PETSC_GMAKEGENTEST  | PETSC_DIR/config/gmakegentest.py |
     +---------------------+----------------------------------+


Case 2.  From install dir:

     +---------------------+-------------------------------------------------------+
     | PETSC_DIR           | <prefix dir>                                          |
     +---------------------+-------------------------------------------------------+
     | PETSC_ARCH          | ''                                                    |
     +---------------------+-------------------------------------------------------+
     | PETSC_LIBDIR        | PETSC_DIR/PETSC_ARCH/lib                              |
     +---------------------+-------------------------------------------------------+
     | PETSC_EXAMPLESDIR   | PETSC_DIR/share/petsc/examples/src                    |
     +---------------------+-------------------------------------------------------+
     | PETSC_TESTDIR       | PETSC_DIR/PETSC_ARCH/tests                            |
     +---------------------+-------------------------------------------------------+
     | PETSC_GMAKEFILETEST | PETSC_DIR/share/petsc/examples/gmakefile.test         |
     +---------------------+-------------------------------------------------------+
     | PETSC_GMAKEGENTEST  | PETSC_DIR/share/petsc/examples/config/gmakegentest.py |
     +---------------------+-------------------------------------------------------+

"""

def install_files(source, destdir):
  """Install file or directory 'source' to 'destdir'.  Does not preserve
  mode (permissions).
  """
  if not os.path.isdir(destdir):
    os.makedirs(destdir)
  if os.path.isdir(source):
    for name in os.listdir(source):
      install_files(os.path.join(source, name), os.path.join(destdir, os.path.basename(source)))
  else:
    shutil.copyfile(source, os.path.join(destdir, os.path.basename(source)))

def nameSpace(srcfile,srcdir):
  """
  Because the scripts have a non-unique naming, the pretty-printing
  needs to convey the srcdir and srcfile.  There are two ways of doing this.
  """
  if srcfile.startswith('run'): srcfile=re.sub('^run','',srcfile)
  prefix=srcdir.replace("/","_")+"-"
  nameString=prefix+srcfile
  return nameString

class generateExamples(Petsc):
  """
    gmakegen.py has basic structure for finding the files, writing out
      the dependencies, etc.
  """
  def __init__(self,petsc_dir=None, petsc_arch=None, pkg_dir=None, pkg_arch=None, pkg_name=None, pkg_pkgs=None, testdir='tests', verbose=False, single_ex=False, srcdir=None, check=False):
    super(generateExamples, self).__init__(petsc_dir=petsc_dir, petsc_arch=petsc_arch, pkg_dir=pkg_dir, pkg_arch=pkg_arch, pkg_name=pkg_name, pkg_pkgs=pkg_pkgs, verbose=verbose)

    self.single_ex=single_ex
    self.srcdir=srcdir
    self.check_output=check

    # Set locations to handle movement
    self.inInstallDir=self.getInInstallDir(thisscriptdir)

    # Special configuration for CI testing
    if self.petsc_arch.find('valgrind') >= 0:
      self.conf['PETSCTEST_VALGRIND']=1

    if self.inInstallDir:
      # Case 2 discussed above
      # set PETSC_ARCH to install directory to allow script to work in both
      dirlist=thisscriptdir.split(os.path.sep)
      installdir=os.path.sep.join(dirlist[0:len(dirlist)-4])
      self.arch_dir=installdir
      if self.srcdir is None:
        self.srcdir=os.path.join(os.path.dirname(thisscriptdir),'src')
    else:
      if petsc_arch == '':
        raise RuntimeError('PETSC_ARCH must be set when running from build directory')
      # Case 1 discussed above
      self.arch_dir=os.path.join(self.petsc_dir,self.petsc_arch)
      if self.srcdir is None:
        self.srcdir=os.path.join(self.petsc_dir,'src')

    self.testroot_dir=os.path.abspath(testdir)

    self.verbose=verbose
    # Whether to write out a useful debugging
    self.summarize=True if verbose else False

    # For help in setting the requirements
    self.precision_types="single double __float128 int32".split()
    self.integer_types="int32 int64 long32 long64".split()
    self.languages="fortran cuda hip sycl cxx cpp".split()    # Always requires C so do not list

    # Things that are not test
    self.buildkeys=testparse.buildkeys

    # Adding a dictionary for storing sources, objects, and tests
    # to make building the dependency tree easier
    self.sources={}
    self.objects={}
    self.tests={}
    for pkg in self.pkg_pkgs:
      self.sources[pkg]={}
      self.objects[pkg]=[]
      self.tests[pkg]={}
      for lang in LANGS:
        self.sources[pkg][lang]={}
        self.sources[pkg][lang]['srcs']=[]
        self.tests[pkg][lang]={}

    if not os.path.isdir(self.testroot_dir): os.makedirs(self.testroot_dir)

    self.indent="   "
    if self.verbose: print('Finishing the constructor')
    return

  def srcrelpath(self,rdir):
    """
    Get relative path to source directory
    """
    return os.path.relpath(rdir,self.srcdir)

  def getInInstallDir(self,thisscriptdir):
    """
    When petsc is installed then this file in installed in:
         <PREFIX>/share/petsc/examples/config/gmakegentest.py
    otherwise the path is:
         <PETSC_DIR>/config/gmakegentest.py
    We use this difference to determine if we are in installdir
    """
    dirlist=thisscriptdir.split(os.path.sep)
    if len(dirlist)>4:
      lastfour=os.path.sep.join(dirlist[len(dirlist)-4:])
      if lastfour==os.path.join('share','petsc','examples','config'):
        return True
      else:
        return False
    else:
      return False

  def getLanguage(self,srcfile):
    """
    Based on the source, determine associated language as found in gmakegen.LANGS
    Can we just return srcext[1:\] now?
    """
    langReq=None
    srcext = getlangext(srcfile)
    if srcext in ".F90".split(): langReq="F90"
    if srcext in ".F".split(): langReq="F"
    if srcext in ".cxx".split(): langReq="cxx"
    if srcext in ".kokkos.cxx".split(): langReq="kokkos_cxx"
    if srcext in ".hip.cpp".split(): langReq="hip_cpp"
    if srcext in ".raja.cxx".split(): langReq="raja_cxx"
    if srcext in ".cpp".split(): langReq="cpp"
    if srcext == ".cu": langReq="cu"
    if srcext == ".c": langReq="c"
    #if not langReq: print("ERROR: ", srcext, srcfile)
    return langReq

  def _getAltList(self,output_file,srcdir):
    ''' Calculate AltList based on output file-- see
       src/snes/tutorials/output/ex22*.out
    '''
    altlist=[output_file]
    basefile = getlangsplit(output_file)
    for i in range(1,9):
      altroot=basefile+"_alt"
      if i > 1: altroot=altroot+"_"+str(i)
      af=altroot+".out"
      srcaf=os.path.join(srcdir,af)
      fullaf=os.path.join(self.petsc_dir,srcaf)
      if os.path.isfile(fullaf): altlist.append(srcaf)

    return altlist


  def _getLoopVars(self,inDict,testname, isSubtest=False):
    """
    Given: 'args: -bs {{1 2 3 4 5}} -pc_type {{cholesky sor}} -ksp_monitor'
    Return:
      inDict['args']: -ksp_monitor
      inDict['subargs']: -bs ${bs} -pc_type ${pc_type}
      loopVars['subargs']['varlist']=['bs' 'pc_type']   # Don't worry about OrderedDict
      loopVars['subargs']['bs']=[["bs"],["1 2 3 4 5"]]
      loopVars['subargs']['pc_type']=[["pc_type"],["cholesky sor"]]
    subst should be passed in instead of inDict
    """
    loopVars={}; newargs=[]
    lsuffix='+'
    argregex = re.compile(' (?=-[a-zA-Z])')
    from testparse import parseLoopArgs
    for key in inDict:
      if key in ('SKIP', 'regexes'):
        continue
      akey=('subargs' if key=='args' else key)  # what to assign
      if akey not in inDict: inDict[akey]=''
      if akey == 'nsize' and not inDict['nsize'].startswith('{{'):
        # Always generate a loop over nsize, even if there is only one value
        inDict['nsize'] = '{{' + inDict['nsize'] + '}}'
      keystr = str(inDict[key])
      varlist = []
      for varset in argregex.split(keystr):
        if not varset.strip(): continue
        if '{{' in varset:
          keyvar,lvars,ftype=parseLoopArgs(varset)
          if akey not in loopVars: loopVars[akey]={}
          varlist.append(keyvar)
          loopVars[akey][keyvar]=[keyvar,lvars]
          if akey=='nsize':
            if len(lvars.split()) > 1:
              lsuffix += akey +'-${i' + keyvar + '}'
          else:
            inDict[akey] += ' -'+keyvar+' ${i' + keyvar + '}'
            lsuffix+=keyvar+'-${i' + keyvar + '}_'
        else:
          if key=='args':
            newargs.append(varset.strip())
        if varlist:
          loopVars[akey]['varlist']=varlist

    # For subtests, args are always substituted in (not top level)
    if isSubtest:
      inDict['subargs'] += " "+" ".join(newargs)
      inDict['args']=''
      if 'label_suffix' in inDict:
        inDict['label_suffix']+=lsuffix.rstrip('+').rstrip('_')
      else:
        inDict['label_suffix']=lsuffix.rstrip('+').rstrip('_')
    else:
      if loopVars:
        inDict['args'] = ' '.join(newargs)
        inDict['label_suffix']=lsuffix.rstrip('+').rstrip('_')
    return loopVars

  def getArgLabel(self,testDict):
    """
    In all of the arguments in the test dictionary, create a simple
    string for searching within the makefile system.  For simplicity in
    search, remove "-", for strings, etc.
    Also, concatenate the arg commands
    For now, ignore nsize -- seems hard to search for anyway
    """
    # Collect all of the args associated with a test
    argStr=("" if 'args' not in testDict else testDict['args'])
    if 'subtests' in testDict:
      for stest in testDict["subtests"]:
         sd=testDict[stest]
         argStr=argStr+("" if 'args' not in sd else sd['args'])

    # Now go through and cleanup
    argStr=re.sub('{{(.*?)}}',"",argStr)
    argStr=re.sub('-'," ",argStr)
    for digit in string.digits: argStr=re.sub(digit," ",argStr)
    argStr=re.sub("\.","",argStr)
    argStr=re.sub(",","",argStr)
    argStr=re.sub('\+',' ',argStr)
    argStr=re.sub(' +',' ',argStr)  # Remove repeated white space
    return argStr.strip()

  def addToSources(self,exfile,rpath,srcDict):
    """
      Put into data structure that allows easy generation of makefile
    """
    pkg=rpath.split(os.path.sep)[0]
    relpfile=os.path.join(rpath,exfile)
    lang=self.getLanguage(exfile)
    if not lang: return
    if pkg not in self.sources: return
    self.sources[pkg][lang]['srcs'].append(relpfile)
    self.sources[pkg][lang][relpfile] = []
    if 'depends' in srcDict:
      depSrcList=srcDict['depends'].split()
      for depSrc in depSrcList:
        depObj = getlangsplit(depSrc)+'.o'
        self.sources[pkg][lang][relpfile].append(os.path.join(rpath,depObj))

    # In gmakefile, ${TESTDIR} var specifies the object compilation
    testsdir=rpath+"/"
    objfile="${TESTDIR}/"+testsdir+getlangsplit(exfile)+'.o'
    self.objects[pkg].append(objfile)
    return

  def addToTests(self,test,rpath,exfile,execname,testDict):
    """
      Put into data structure that allows easy generation of makefile
      Organized by languages to allow testing of languages
    """
    pkg=rpath.split("/")[0]
    nmtest=os.path.join(rpath,test)
    lang=self.getLanguage(exfile)
    if not lang: return
    if pkg not in self.tests: return
    self.tests[pkg][lang][nmtest]={}
    self.tests[pkg][lang][nmtest]['exfile']=os.path.join(rpath,exfile)
    self.tests[pkg][lang][nmtest]['exec']=execname
    self.tests[pkg][lang][nmtest]['argLabel']=self.getArgLabel(testDict)
    return

  def getExecname(self,exfile,rpath):
    """
      Generate bash script using template found next to this file.
      This file is read in at constructor time to avoid file I/O
    """
    if self.single_ex:
      execname=rpath.split("/")[1]+"-ex"
    else:
      execname=getlangsplit(exfile)
    return execname

  def getSubstVars(self,testDict,rpath,testname):
    """
      Create a dictionary with all of the variables that get substituted
      into the template commands found in example_template.py
    """
    subst={}

    # Handle defaults of testparse.acceptedkeys (e.g., ignores subtests)
    if 'nsize' not in testDict: testDict['nsize'] = '1'
    if 'timeoutfactor' not in testDict: testDict['timeoutfactor']="1"
    for ak in testparse.acceptedkeys:
      if ak=='test': continue
      subst[ak]=(testDict[ak] if ak in testDict else '')

    # Now do other variables
    subst['execname']=testDict['execname']
    subst['error']=''
    if 'filter' in testDict:
      if testDict['filter'].startswith("Error:"):
        subst['error']="Error"
        subst['filter']=testDict['filter'].lstrip("Error:")
      else:
        subst['filter']=testDict['filter']

    # Others
    subst['subargs']=''  # Default.  For variables override
    subst['srcdir']=os.path.join(self.srcdir, rpath)
    subst['label_suffix']=''
    subst['comments']="\n#".join(subst['comments'].split("\n"))
    if subst['comments']: subst['comments']="#"+subst['comments']
    subst['exec']="../"+subst['execname']
    subst['testroot']=self.testroot_dir
    subst['testname']=testname
    dp = self.conf.get('DATAFILESPATH','')
    subst['datafilespath_line'] = 'DATAFILESPATH=${DATAFILESPATH:-"'+dp+'"}'

    # This is used to label some matrices
    subst['petsc_index_size']=str(self.conf['PETSC_INDEX_SIZE'])
    subst['petsc_scalar_size']=str(self.conf['PETSC_SCALAR_SIZE'])

    subst['petsc_test_options']=self.conf['PETSC_TEST_OPTIONS']

    #Conf vars
    if self.petsc_arch.find('valgrind')>=0:
      subst['mpiexec']='petsc_mpiexec_valgrind ' + self.conf['MPIEXEC']
    else:
      subst['mpiexec']=self.conf['MPIEXEC']
    subst['pkg_name']=self.pkg_name
    subst['pkg_dir']=self.pkg_dir
    subst['pkg_arch']=self.petsc_arch
    subst['CONFIG_DIR']=thisscriptdir
    subst['PETSC_BINDIR']=os.path.join(self.petsc_dir,'lib','petsc','bin')
    subst['diff']=self.conf['DIFF']
    subst['rm']=self.conf['RM']
    subst['grep']=self.conf['GREP']
    subst['petsc_lib_dir']=self.conf['PETSC_LIB_DIR']
    subst['wpetsc_dir']=self.conf['wPETSC_DIR']

    # Output file is special because of subtests override
    defroot = testparse.getDefaultOutputFileRoot(testname)
    if 'output_file' not in testDict:
      subst['output_file']="output/"+defroot+".out"
    subst['redirect_file']=defroot+".tmp"
    subst['label']=nameSpace(defroot,self.srcrelpath(subst['srcdir']))

    # Add in the full path here.
    subst['output_file']=os.path.join(subst['srcdir'],subst['output_file'])

    subst['regexes']={}
    for subkey in subst:
      if subkey=='regexes': continue
      if not isinstance(subst[subkey],str): continue
      patt="@"+subkey.upper()+"@"
      subst['regexes'][subkey]=re.compile(patt)

    return subst

  def _substVars(self,subst,origStr):
    """
      Substitute variables
    """
    Str=origStr
    for subkey in subst:
      if subkey=='regexes': continue
      if not isinstance(subst[subkey],str): continue
      if subkey.upper() not in Str: continue
      Str=subst['regexes'][subkey].sub(lambda x: subst[subkey],Str)
    return Str

  def getCmds(self,subst,i, debug=False):
    """
      Generate bash script using template found next to this file.
      This file is read in at constructor time to avoid file I/O
    """
    nindnt=i # the start and has to be consistent with below
    cmdindnt=self.indent*nindnt
    cmdLines=""

    # MPI is the default -- but we have a few odd commands
    if not subst['command']:
      cmd=cmdindnt+self._substVars(subst,example_template.mpitest)
    else:
      cmd=cmdindnt+self._substVars(subst,example_template.commandtest)
    cmdLines+=cmd+"\n"+cmdindnt+"res=$?\n\n"

    cmdLines+=cmdindnt+'if test $res = 0; then\n'
    diffindnt=self.indent*(nindnt+1)

    # Do some checks on existence of output_file and alt files
    if not os.path.isfile(os.path.join(self.petsc_dir,subst['output_file'])):
      if not subst['TODO']:
        print("Warning: "+subst['output_file']+" not found.")
    altlist=self._getAltList(subst['output_file'], subst['srcdir'])

    # altlist always has output_file
    if len(altlist)==1:
      cmd=diffindnt+self._substVars(subst,example_template.difftest)
    else:
      if debug: print("Found alt files: ",altlist)
      # Have to do it by hand a bit because of variable number of alt files
      rf=subst['redirect_file']
      cmd=diffindnt+example_template.difftest.split('@')[0]
      for i in range(len(altlist)):
        af=altlist[i]
        cmd+=af+' '+rf
        if i!=len(altlist)-1:
          cmd+=' > diff-${testname}-'+str(i)+'.out 2> diff-${testname}-'+str(i)+'.out'
          cmd+=' || ${diff_exe} '
        else:
          cmd+='" diff-${testname}.out diff-${testname}.out diff-${label}'
          cmd+=subst['label_suffix']+' ""'  # Quotes are painful
    cmdLines+=cmd+"\n"
    cmdLines+=cmdindnt+'else\n'
    cmdLines+=diffindnt+'petsc_report_tapoutput "" ${label} "SKIP Command failed so no diff"\n'
    cmdLines+=cmdindnt+'fi\n'
    return cmdLines

  def _writeTodoSkip(self,fh,tors,reasons,footer):
    """
    Write out the TODO and SKIP lines in the file
    The TODO or SKIP variable, tors, should be lower case
    """
    TORS=tors.upper()
    template=eval("example_template."+tors+"line")
    tsStr=re.sub("@"+TORS+"COMMENT@",', '.join(reasons),template)
    tab = ''
    if reasons:
      fh.write('if ! $force; then\n')
      tab = tab + '    '
    if reasons == ["Requires DATAFILESPATH"]:
      # The only reason not to run is DATAFILESPATH, which we check at run-time
      fh.write(tab + 'if test -z "${DATAFILESPATH}"; then\n')
      tab = tab + '    '
    if reasons:
      fh.write(tab+tsStr+"\n" + tab + "total=1; "+tors+"=1\n")
      fh.write(tab+footer+"\n")
      fh.write(tab+"exit\n")
    if reasons == ["Requires DATAFILESPATH"]:
      fh.write('    fi\n')
    if reasons:
      fh.write('fi\n')
    fh.write('\n\n')
    return

  def getLoopVarsHead(self,loopVars,i,usedVars={}):
    """
    Generate a nicely indented string with the format loops
    Here is what the data structure looks like
      loopVars['subargs']['varlist']=['bs' 'pc_type']   # Don't worry about OrderedDict
      loopVars['subargs']['bs']=["i","1 2 3 4 5"]
      loopVars['subargs']['pc_type']=["j","cholesky sor"]
    """
    outstr=''; indnt=self.indent

    for key in loopVars:
      for var in loopVars[key]['varlist']:
        varval=loopVars[key][var]
        outstr += "{0}_in=${{{0}:-{1}}}\n".format(*varval)
    outstr += "\n\n"

    for key in loopVars:
      for var in loopVars[key]['varlist']:
        varval=loopVars[key][var]
        outstr += indnt * i + "for i{0} in ${{{0}_in}}; do\n".format(*varval)
        i = i + 1
    return (outstr,i)

  def getLoopVarsFoot(self,loopVars,i):
    outstr=''; indnt=self.indent
    for key in loopVars:
      for var in loopVars[key]['varlist']:
        i = i - 1
        outstr += indnt * i + "done\n"
    return (outstr,i)

  def genRunScript(self,testname,root,isRun,srcDict):
    """
      Generate bash script using template found next to this file.
      This file is read in at constructor time to avoid file I/O
    """
    # runscript_dir directory has to be consistent with gmakefile
    testDict=srcDict[testname]
    rpath=self.srcrelpath(root)
    runscript_dir=os.path.join(self.testroot_dir,rpath)
    if not os.path.isdir(runscript_dir): os.makedirs(runscript_dir)
    with open(os.path.join(runscript_dir,testname+".sh"),"w") as fh:

      # Get variables to go into shell scripts.  last time testDict used
      subst=self.getSubstVars(testDict,rpath,testname)
      loopVars = self._getLoopVars(subst,testname)  # Alters subst as well
      if 'subtests' in testDict:
        # The subtests inherit inDict, so we don't need top-level loops.
        loopVars = {}

      #Handle runfiles
      for lfile in subst.get('localrunfiles','').split():
        install_files(os.path.join(root, lfile),
                      os.path.join(runscript_dir, os.path.dirname(lfile)))
      # Check subtests for local runfiles
      for stest in subst.get("subtests",[]):
        for lfile in testDict[stest].get('localrunfiles','').split():
          install_files(os.path.join(root, lfile),
                        os.path.join(runscript_dir, os.path.dirname(lfile)))

      # Now substitute the key variables into the header and footer
      header=self._substVars(subst,example_template.header)
      # The header is done twice to enable @...@ in header
      header=self._substVars(subst,header)
      footer=re.sub('@TESTROOT@',subst['testroot'],example_template.footer)

      # Start writing the file
      fh.write(header+"\n")

      # If there is a TODO or a SKIP then we do it before writing out the
      # rest of the command (which is useful for working on the test)
      # SKIP and TODO can be for the source file or for the runs
      self._writeTodoSkip(fh,'todo',[s for s in [srcDict.get('TODO',''), testDict.get('TODO','')] if s],footer)
      self._writeTodoSkip(fh,'skip',srcDict.get('SKIP',[]) + testDict.get('SKIP',[]),footer)

      j=0  # for indentation

      if loopVars:
        (loopHead,j) = self.getLoopVarsHead(loopVars,j)
        if (loopHead): fh.write(loopHead+"\n")

      # Subtests are special
      allLoopVars=list(loopVars.keys())
      if 'subtests' in testDict:
        substP=subst   # Subtests can inherit args but be careful
        k=0  # for label suffixes
        for stest in testDict["subtests"]:
          subst=substP.copy()
          subst.update(testDict[stest])
          subst['label_suffix']='+'+string.ascii_letters[k]; k+=1
          sLoopVars = self._getLoopVars(subst,testname,isSubtest=True)
          if sLoopVars:
            (sLoopHead,j) = self.getLoopVarsHead(sLoopVars,j,allLoopVars)
            allLoopVars+=list(sLoopVars.keys())
            fh.write(sLoopHead+"\n")
          fh.write(self.getCmds(subst,j)+"\n")
          if sLoopVars:
            (sLoopFoot,j) = self.getLoopVarsFoot(sLoopVars,j)
            fh.write(sLoopFoot+"\n")
      else:
        fh.write(self.getCmds(subst,j)+"\n")

      if loopVars:
        (loopFoot,j) = self.getLoopVarsFoot(loopVars,j)
        fh.write(loopFoot+"\n")

      fh.write(footer+"\n")

    os.chmod(os.path.join(runscript_dir,testname+".sh"),0o755)
    #if '10_9' in testname: sys.exit()
    return

  def  genScriptsAndInfo(self,exfile,root,srcDict):
    """
    Generate scripts from the source file, determine if built, etc.
     For every test in the exfile with info in the srcDict:
      1. Determine if it needs to be run for this arch
      2. Generate the script
      3. Generate the data needed to write out the makefile in a
         convenient way
     All tests are *always* run, but some may be SKIP'd per the TAP standard
    """
    debug=False
    rpath=self.srcrelpath(root)
    execname=self.getExecname(exfile,rpath)
    isBuilt=self._isBuilt(exfile,srcDict)
    for test in srcDict:
      if test in self.buildkeys: continue
      if debug: print(nameSpace(exfile,root), test)
      srcDict[test]['execname']=execname   # Convenience in generating scripts
      isRun=self._isRun(srcDict[test])
      self.genRunScript(test,root,isRun,srcDict)
      srcDict[test]['isrun']=isRun
      self.addToTests(test,rpath,exfile,execname,srcDict[test])

    # This adds to datastructure for building deps
    if isBuilt: self.addToSources(exfile,rpath,srcDict)
    return

  def _isBuilt(self,exfile,srcDict):
    """
    Determine if this file should be built.
    """
    # Get the language based on file extension
    srcDict['SKIP'] = []
    lang=self.getLanguage(exfile)
    if (lang=="F" or lang=="F90"):
      if not self.have_fortran:
        srcDict["SKIP"].append("Fortran required for this test")
      elif lang=="F90" and 'PETSC_USING_F90FREEFORM' not in self.conf:
        srcDict["SKIP"].append("Fortran f90freeform required for this test")
    if lang=="cu" and 'PETSC_HAVE_CUDA' not in self.conf:
      srcDict["SKIP"].append("CUDA required for this test")
    if lang=="hip" and 'PETSC_HAVE_HIP' not in self.conf:
      srcDict["SKIP"].append("HIP required for this test")
    if lang=="sycl" and 'PETSC_HAVE_SYCL' not in self.conf:
      srcDict["SKIP"].append("SYCL required for this test")
    if lang=="kokkos_cxx" and 'PETSC_HAVE_KOKKOS' not in self.conf:
      srcDict["SKIP"].append("KOKKOS required for this test")
    if lang=="raja_cxx" and 'PETSC_HAVE_RAJA' not in self.conf:
      srcDict["SKIP"].append("RAJA required for this test")
    if lang=="cxx" and 'PETSC_HAVE_CXX' not in self.conf:
      srcDict["SKIP"].append("C++ required for this test")
    if lang=="cpp" and 'PETSC_HAVE_CXX' not in self.conf:
      srcDict["SKIP"].append("C++ required for this test")

    # Deprecated source files
    if srcDict.get("TODO"):
      return False

    # isRun can work with srcDict to handle the requires
    if "requires" in srcDict:
      if srcDict["requires"]:
        return self._isRun(srcDict)

    return srcDict['SKIP'] == []


  def _isRun(self,testDict, debug=False):
    """
    Based on the requirements listed in the src file and the petscconf.h
    info, determine whether this test should be run or not.
    """
    indent="  "

    if 'SKIP' not in testDict:
      testDict['SKIP'] = []
    # MPI requirements
    if 'MPI_IS_MPIUNI' in self.conf:
      if testDict.get('nsize', '1') != '1':
        testDict['SKIP'].append("Parallel test with serial build")

      # The requirements for the test are the sum of all the run subtests
      if 'subtests' in testDict:
        if 'requires' not in testDict: testDict['requires']=""
        for stest in testDict['subtests']:
          if 'requires' in testDict[stest]:
            testDict['requires']+=" "+testDict[stest]['requires']
          if testDict[stest].get('nsize', '1') != '1':
            testDict['SKIP'].append("Parallel test with serial build")
            break

    # Now go through all requirements
    if 'requires' in testDict:
      for requirement in testDict['requires'].split():
        requirement=requirement.strip()
        if not requirement: continue
        if debug: print(indent+"Requirement: ", requirement)
        isNull=False
        if requirement.startswith("!"):
          requirement=requirement[1:]; isNull=True
        # Precision requirement for reals
        if requirement in self.precision_types:
          if self.conf['PETSC_PRECISION']==requirement:
            if isNull:
              testDict['SKIP'].append("not "+requirement+" required")
              continue
            continue  # Success
          elif not isNull:
            testDict['SKIP'].append(requirement+" required")
            continue
        # Precision requirement for ints
        if requirement in self.integer_types:
          if requirement=="int32":
            if self.conf['PETSC_SIZEOF_INT']==4:
              if isNull:
                testDict['SKIP'].append("not int32 required")
                continue
              continue  # Success
            elif not isNull:
              testDict['SKIP'].append("int32 required")
              continue
          if requirement=="int64":
            if self.conf['PETSC_SIZEOF_INT']==8:
              if isNull:
                testDict['SKIP'].append("NOT int64 required")
                continue
              continue  # Success
            elif not isNull:
              testDict['SKIP'].append("int64 required")
              continue
          if requirement.startswith("long"):
            reqsize = int(requirement[4:])//8
            longsize = int(self.conf['PETSC_SIZEOF_LONG'].strip())
            if longsize==reqsize:
              if isNull:
                testDict['SKIP'].append("not %s required" % requirement)
                continue
              continue  # Success
            elif not isNull:
              testDict['SKIP'].append("%s required" % requirement)
              continue
        # Datafilespath
        if requirement=="datafilespath" and not isNull:
          testDict['SKIP'].append("Requires DATAFILESPATH")
          continue
        # Defines -- not sure I have comments matching
        if "defined(" in requirement.lower():
          reqdef=requirement.split("(")[1].split(")")[0]
          if reqdef in self.conf:
            if isNull:
              testDict['SKIP'].append("Null requirement not met: "+requirement)
              continue
            continue  # Success
          elif not isNull:
            testDict['SKIP'].append("Required: "+requirement)
            continue

        # Rest should be packages that we can just get from conf
        if requirement in ["complex","debug"]:
          petscconfvar="PETSC_USE_"+requirement.upper()
          pkgconfvar=self.pkg_name.upper()+"_USE_"+requirement.upper()
        else:
          petscconfvar="PETSC_HAVE_"+requirement.upper()
          pkgconfvar=self.pkg_name.upper()+'_HAVE_'+requirement.upper()
        petsccv = self.conf.get(petscconfvar)
        pkgcv = self.conf.get(pkgconfvar)

        if petsccv or pkgcv:
          if isNull:
            if petsccv:
              testDict['SKIP'].append("Not "+petscconfvar+" requirement not met")
              continue
            else:
              testDict['SKIP'].append("Not "+pkgconfvar+" requirement not met")
              continue
          continue  # Success
        elif not isNull:
          if not petsccv and not pkgcv:
            if debug: print("requirement not found: ", requirement)
            if self.pkg_name == 'petsc':
              testDict['SKIP'].append(petscconfvar+" requirement not met")
            else:
              testDict['SKIP'].append(petscconfvar+" or "+pkgconfvar+" requirement not met")
            continue
    return testDict['SKIP'] == []

  def  checkOutput(self,exfile,root,srcDict):
    """
     Check and make sure the output files are in the output directory
    """
    debug=False
    rpath=self.srcrelpath(root)
    for test in srcDict:
      if test in self.buildkeys: continue
      if debug: print(rpath, exfile, test)
      if 'output_file' in srcDict[test]:
        output_file=srcDict[test]['output_file']
      else:
        defroot = testparse.getDefaultOutputFileRoot(test)
        if 'TODO' in srcDict[test]: continue
        output_file="output/"+defroot+".out"

      fullout=os.path.join(root,output_file)
      if debug: print("---> ",fullout)
      if not os.path.exists(fullout):
        self.missing_files.append(fullout)

    return

  def genPetscTests_summarize(self,dataDict):
    """
    Required method to state what happened
    """
    if not self.summarize: return
    indent="   "
    fhname=os.path.join(self.testroot_dir,'GenPetscTests_summarize.txt')
    with open(fhname, "w") as fh:
      for root in dataDict:
        relroot=self.srcrelpath(root)
        pkg=relroot.split("/")[1]
        fh.write(relroot+"\n")
        allSrcs=[]
        for lang in LANGS: allSrcs+=self.sources[pkg][lang]['srcs']
        for exfile in dataDict[root]:
          # Basic  information
          rfile=os.path.join(relroot,exfile)
          builtStatus=(" Is built" if rfile in allSrcs else " Is NOT built")
          fh.write(indent+exfile+indent*4+builtStatus+"\n")
          for test in dataDict[root][exfile]:
            if test in self.buildkeys: continue
            line=indent*2+test
            fh.write(line+"\n")
            # Looks nice to have the keys in order
            #for key in dataDict[root][exfile][test]:
            for key in "isrun abstracted nsize args requires script".split():
              if key not in dataDict[root][exfile][test]: continue
              line=indent*3+key+": "+str(dataDict[root][exfile][test][key])
              fh.write(line+"\n")
            fh.write("\n")
          fh.write("\n")
        fh.write("\n")
    return

  def genPetscTests(self,root,dirs,files,dataDict):
    """
     Go through and parse the source files in the directory to generate
     the examples based on the metadata contained in the source files
    """
    debug=False
    # Use examplesAnalyze to get what the makefles think are sources
    #self.examplesAnalyze(root,dirs,files,anlzDict)

    dataDict[root]={}

    for exfile in files:
      #TST: Until we replace files, still leaving the orginals as is
      #if not exfile.startswith("new_"+"ex"): continue
      #if not exfile.startswith("ex"): continue

      # Ignore emacs and other temporary files
      if exfile.startswith("."): continue
      if exfile.startswith("#"): continue
      if exfile.endswith("~"): continue
      # Only parse source files
      ext=getlangext(exfile).lstrip('.').replace('.','_')
      if ext not in LANGS: continue

      # Convenience
      fullex=os.path.join(root,exfile)
      if self.verbose: print('   --> '+fullex)
      dataDict[root].update(testparse.parseTestFile(fullex,0))
      if exfile in dataDict[root]:
        if not self.check_output:
          self.genScriptsAndInfo(exfile,root,dataDict[root][exfile])
        else:
          self.checkOutput(exfile,root,dataDict[root][exfile])

    return

  def walktree(self,top):
    """
    Walk a directory tree, starting from 'top'
    """
    if self.check_output:
      print("Checking for missing output files")
      self.missing_files=[]

    # Goal of action is to fill this dictionary
    dataDict={}
    for root, dirs, files in os.walk(top, topdown=True):
      dirs.sort()
      files.sort()
      if "/tests" not in root and "/tutorials" not in root: continue
      if "dSYM" in root: continue
      if "tutorials"+os.sep+"build" in root: continue
      if os.path.basename(root.rstrip("/")) == 'output': continue
      if self.verbose: print(root)
      self.genPetscTests(root,dirs,files,dataDict)

    # If checking output, report results
    if self.check_output:
      if self.missing_files:
        for file in set(self.missing_files):  # set uniqifies
          print(file)
        sys.exit(1)

    # Now summarize this dictionary
    if self.verbose: self.genPetscTests_summarize(dataDict)
    return dataDict

  def gen_gnumake(self, fd):
    """
     Overwrite of the method in the base PETSc class
    """
    def write(stem, srcs):
      for lang in LANGS:
        if srcs[lang]['srcs']:
          fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang.replace('_','.'), srcs=' '.join(srcs[lang]['srcs'])))
    for pkg in self.pkg_pkgs:
        srcs = self.gen_pkg(pkg)
        write('testsrcs-' + pkg, srcs)
        # Handle dependencies
        for lang in LANGS:
            for exfile in srcs[lang]['srcs']:
                if exfile in srcs[lang]:
                    ex='$(TESTDIR)/'+getlangsplit(exfile)
                    exfo=ex+'.o'
                    deps = [os.path.join('$(TESTDIR)', dep) for dep in srcs[lang][exfile]]
                    if deps:
                        # The executable literally depends on the object file because it is linked
                        fd.write(ex   +": " + " ".join(deps) +'\n')
                        # The object file containing 'main' does not normally depend on other object
                        # files, but it does when it includes their modules.  This dependency is
                        # overly blunt and could be reduced to only depend on object files for
                        # modules that are used, like "*f90aux.o".
                        fd.write(exfo +": " + " ".join(deps) +'\n')

    return self.gendeps

  def gen_pkg(self, pkg):
    """
     Overwrite of the method in the base PETSc class
    """
    return self.sources[pkg]

  def write_gnumake(self, dataDict, output=None):
    """
     Write out something similar to files from gmakegen.py

     Test depends on script which also depends on source
     file, but since I don't have a good way generating
     acting on a single file (oops) just depend on
     executable which in turn will depend on src file
    """
    # Different options for how to set up the targets
    compileExecsFirst=False

    # Open file
    with open(output, 'w') as fd:
      # Write out the sources
      gendeps = self.gen_gnumake(fd)

      # Write out the tests and execname targets
      fd.write("\n#Tests and executables\n")    # Delimiter

      for pkg in self.pkg_pkgs:
        # These grab the ones that are built
        for lang in LANGS:
          testdeps=[]
          for ftest in self.tests[pkg][lang]:
            test=os.path.basename(ftest)
            basedir=os.path.dirname(ftest)
            testdeps.append(nameSpace(test,basedir))
          fd.write("test-"+pkg+"."+lang.replace('_','.')+" := "+' '.join(testdeps)+"\n")
          fd.write('test-%s.%s : $(test-%s.%s)\n' % (pkg, lang.replace('_','.'), pkg, lang.replace('_','.')))

          # test targets
          for ftest in self.tests[pkg][lang]:
            test=os.path.basename(ftest)
            basedir=os.path.dirname(ftest)
            testdir="${TESTDIR}/"+basedir+"/"
            nmtest=nameSpace(test,basedir)
            rundir=os.path.join(testdir,test)
            script=test+".sh"

            # Deps
            exfile=self.tests[pkg][lang][ftest]['exfile']
            fullex=os.path.join(self.srcdir,exfile)
            localexec=self.tests[pkg][lang][ftest]['exec']
            execname=os.path.join(testdir,localexec)
            fullscript=os.path.join(testdir,script)
            tmpfile=os.path.join(testdir,test,test+".tmp")

            # *.counts depends on the script and either executable (will
            # be run) or the example source file (SKIP or TODO)
            fd.write('%s.counts : %s %s'
                % (os.path.join('$(TESTDIR)/counts', nmtest),
                   fullscript,
                   execname if exfile in self.sources[pkg][lang]['srcs'] else fullex)
                )
            if exfile in self.sources[pkg][lang]:
              for dep in self.sources[pkg][lang][exfile]:
                fd.write(' %s' % os.path.join('$(TESTDIR)',dep))
            fd.write('\n')

            # Now write the args:
            fd.write(nmtest+"_ARGS := '"+self.tests[pkg][lang][ftest]['argLabel']+"'\n")

    return

  def write_db(self, dataDict, testdir):
    """
     Write out the dataDict into a pickle file
    """
    with open(os.path.join(testdir,'datatest.pkl'), 'wb') as fd:
      pickle.dump(dataDict,fd)
    return

def main(petsc_dir=None, petsc_arch=None, pkg_dir=None, pkg_arch=None,
         pkg_name=None, pkg_pkgs=None, verbose=False, single_ex=False,
         srcdir=None, testdir=None, check=False):
    # Allow petsc_arch to have both petsc_dir and petsc_arch for convenience
    testdir=os.path.normpath(testdir)
    if petsc_arch:
        petsc_arch=petsc_arch.rstrip(os.path.sep)
        if len(petsc_arch.split(os.path.sep))>1:
            petsc_dir,petsc_arch=os.path.split(petsc_arch)
    output = os.path.join(testdir, 'testfiles')

    pEx=generateExamples(petsc_dir=petsc_dir, petsc_arch=petsc_arch,
                         pkg_dir=pkg_dir, pkg_arch=pkg_arch, pkg_name=pkg_name, pkg_pkgs=pkg_pkgs,
                         verbose=verbose, single_ex=single_ex, srcdir=srcdir,
                         testdir=testdir,check=check)
    dataDict=pEx.walktree(os.path.join(pEx.srcdir))
    if not pEx.check_output:
        pEx.write_gnumake(dataDict, output)
        pEx.write_db(dataDict, testdir)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-dir', help='Set PETSC_DIR different from environment', default=os.environ.get('PETSC_DIR'))
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--srcdir', help='Set location of sources different from PETSC_DIR/src', default=None)
    parser.add_option('-s', '--single_executable', dest='single_executable', action="store_false", help='Whether there should be single executable per src subdir.  Default is false')
    parser.add_option('-t', '--testdir', dest='testdir',  help='Test directory [$PETSC_ARCH/tests]')
    parser.add_option('-c', '--check-output', dest='check_output', action="store_true",
                      help='Check whether output files are in output director')
    parser.add_option('--pkg-dir', help='Set the directory of the package (different from PETSc) you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-name', help='Set the name of the package you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-arch', help='Set the package arch name you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-pkgs', help='Set the package folders (comma separated list, different from the usual sys,vec,mat etc) you want to generate the makefile rules for', default=None)

    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    if opts.testdir is None:
      opts.testdir = os.path.join(opts.petsc_arch, 'tests')

    main(petsc_dir=opts.petsc_dir, petsc_arch=opts.petsc_arch,
         pkg_dir=opts.pkg_dir,pkg_arch=opts.pkg_arch,pkg_name=opts.pkg_name,pkg_pkgs=opts.pkg_pkgs,
         verbose=opts.verbose,
         single_ex=opts.single_executable, srcdir=opts.srcdir,
         testdir=opts.testdir, check=opts.check_output)
