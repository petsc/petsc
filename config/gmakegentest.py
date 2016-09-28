#!/usr/bin/env python

import os,shutil, string, re
from distutils.sysconfig import parse_makefile
import sys
import logging
import types
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cmakegen import Mistakes, stripsplit, AUTODIRS, SKIPDIRS
from cmakegen import defaultdict # collections.defaultdict, with fallback for python-2.4
from gmakegen import *

import inspect
thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,thisscriptdir) 
import testparse
import example_template

class generateExamples(Petsc):
  """
    gmakegen.py has basic structure for finding the files, writing out
      the dependencies, etc.
  """
  def __init__(self,petsc_dir=None, petsc_arch=None, verbose=False, single_ex=False):
    super(generateExamples, self).__init__(petsc_dir=None, petsc_arch=None, verbose=False)

    self.single_ex=single_ex
    self.arch_dir=os.path.join(self.petsc_dir,self.petsc_arch)
    self.ptNaming=True
    # Whether to write out a useful debugging
    #if verbose: self.summarize=True
    self.summarize=True

    # For help in setting the requirements
    self.precision_types="single double quad int32".split()
    self.integer_types="int32 int64".split()
    self.languages="fortran cuda cxx".split()    # Always requires C so do not list

    # Things that are not test
    self.buildkeys=testparse.buildkeys

    # Adding a dictionary for storing sources, objects, and tests
    # to make building the dependency tree easier
    self.sources={}
    self.objects={}
    self.tests={}
    for pkg in PKGS:
      self.sources[pkg]={}
      self.objects[pkg]=[]
      self.tests[pkg]={}
      for lang in LANGS:
        self.sources[pkg][lang]={}
        self.sources[pkg][lang]['srcs']=[]
        self.tests[pkg][lang]={}

    # Copy petsc tests harness script
    harness_file=os.path.join(self.petsc_dir,"config","petsc_harness.sh")
    reports_file=os.path.join(self.petsc_dir,"config","report_tests.sh")
    self.testroot_dir=os.path.join(self.arch_dir,"tests")
    if not os.path.isdir(self.testroot_dir): os.makedirs(self.testroot_dir)
    shutil.copy(harness_file,self.testroot_dir)
    shutil.copy(reports_file,self.testroot_dir)

    return

  def nameSpace(self,srcfile,srcdir):
    """
    Because the scripts have a non-unique naming, the pretty-printing
    needs to convey the srcdir and srcfile.  There are two ways of doing this.
    """
    if self.ptNaming:
      cdir=srcdir.split('src')[1].lstrip("/").rstrip("/")
      prefix=cdir.replace('/examples/','_').replace("/","_")+"-"
      nameString=prefix+srcfile
    else:
      #nameString=srcdir+": "+srcfile
      nameString=srcfile
    return nameString

  def getLanguage(self,srcfile):
    """
    Based on the source, determine associated language as found in gmakegen.LANGS
    Can we just return srcext[1:\] now?
    """
    langReq=None
    srcext=os.path.splitext(srcfile)[-1]
    if srcext in ".F90".split(): langReq="F90"
    if srcext in ".F".split(): langReq="F"
    if srcext in ".cxx".split(): langReq="cxx"
    if srcext == ".cu": langReq="cu"
    if srcext == ".c": langReq="c"
    if not langReq: print "ERROR: ", srcext, srcfile
    return langReq

  def getArgLabel(self,testDict):
    """
    In all of the arguments in the test dictionary, create a simple
    string for searching within the makefile system.  For simplicity in
    search, remove "-", for strings, etc.
    Also, concatenate the arg commands
    For now, ignore nsize -- seems hard to search for anyway
    """
    # Collect all of the args associated with a test
    argStr=("" if not testDict.has_key('args') else testDict['args'])
    if testDict.has_key('subtests'):
      for stest in testDict["subtests"]:
         sd=testDict[stest]
         argStr=argStr+("" if not sd.has_key('args') else sd['args'])

    # Now go through and cleanup
    argStr=re.sub('{{(.*?)}}',"",argStr)
    argStr=re.sub('-'," ",argStr)
    for digit in string.digits: argStr=re.sub(digit," ",argStr)
    argStr=re.sub("\.","",argStr)
    argStr=re.sub(",","",argStr)
    argStr=re.sub('\+',' ',argStr)
    argStr=re.sub(' +',' ',argStr)  # Remove repeated white space
    return argStr.strip()

  def addToSources(self,exfile,root,srcDict):
    """
      Put into data structure that allows easy generation of makefile
    """
    pkg=self.relpath(self.petsc_dir,root).split("/")[1]
    fullfile=os.path.join(root,exfile)
    relpfile=self.relpath(self.petsc_dir,fullfile)
    lang=self.getLanguage(exfile)
    self.sources[pkg][lang]['srcs'].append(relpfile)
    if srcDict.has_key('depends'):
      depSrc=srcDict['depends']
      depObj=os.path.splitext(depSrc)[0]+".o"
      self.sources[pkg][lang][exfile]=depObj

    # In gmakefile, ${TESTDIR} var specifies the object compilation
    testsdir=self.relpath(self.petsc_dir,root)+"/"
    objfile="${TESTDIR}/"+testsdir+os.path.splitext(exfile)[0]+".o"
    self.objects[pkg].append(objfile)
    return

  def addToTests(self,test,root,exfile,execname,testDict):
    """
      Put into data structure that allows easy generation of makefile
      Organized by languages to allow testing of languages
    """
    pkg=self.relpath(self.petsc_dir,root).split("/")[1]
    #nmtest=self.nameSpace(test,root)
    rpath=self.relpath(self.petsc_dir,root)
    nmtest=os.path.join(rpath,test)
    lang=self.getLanguage(exfile)
    self.tests[pkg][lang][nmtest]={}
    self.tests[pkg][lang][nmtest]['exfile']=os.path.join(rpath,exfile)
    self.tests[pkg][lang][nmtest]['exec']=execname
    self.tests[pkg][lang][nmtest]['argLabel']=self.getArgLabel(testDict)
    return

  def getFor(self,subst,i,j):
    """
      Get the for and done lines
    """
    forlines=""
    donlines=""
    indent="   "
    nsizeStr=subst['nsize']
    for loop in re.findall('{{(.*?)}}',subst['nsize']):
      lindex=string.ascii_lowercase[i]
      forline=indent*j+"for "+lindex+" in '"+loop+"'; do"
      nsizeStr=re.sub("{{"+loop+"}}","${"+lindex+"}",nsizeStr)
      donline=indent*j+"done"
      forlines=forlines+forline+"\n"
      donlines=donlines+donline+"\n"
      i=i+1
      j=j+1
    subst['nsize']=nsizeStr
    argStr=subst['args']
    for loop in re.findall('{{(.*?)}}',subst['args']):
      lindex=string.ascii_lowercase[i]
      forline=indent*j+"for "+lindex+" in '"+loop+"'; do"
      argStr=re.sub("{{"+loop+"}}","${"+lindex+"}",argStr)
      donline=indent*j+"done"
      forlines=forlines+forline+"\n"
      donlines=donlines+donline+"\n"
      i=i+1
      j=j+1
    subst['args']=argStr

    # The do lines have reverse order with respect to indentation
    dl=donlines.rstrip("\n").split("\n")
    dl.reverse()
    donlines="\n".join(dl)+"\n"

    return forlines,donlines,i,j


  def getExecname(self,exfile,root):
    """
      Generate bash script using template found next to this file.  
      This file is read in at constructor time to avoid file I/O
    """
    rpath=self.relpath(self.petsc_dir,root)
    if self.single_ex:
      execname=rpath.split("/")[1]+"-ex"
    else:
      execname=os.path.splitext(exfile)[0]
    return execname

  def getSubstVars(self,testDict,rpath,testname):
    """
      Create a dictionary with all of the variables that get substituted
      into the template commands found in example_template.py
      TODO: Cleanup
    """
    subst={}
    # Handle defaults
    if not testDict.has_key('nsize'): testDict['nsize']=1
    if not testDict.has_key('filter'): testDict['filter']=""
    if not testDict.has_key('args'): testDict['args']=""
    defroot=(re.sub("run","",testname) if testname.startswith("run") else testname)
    if not testDict.has_key('redirect_file'): testDict['redirect_file']=defroot+".tmp"
    if not testDict.has_key('output_file'): testDict['output_file']="output/"+defroot+".out"

    # Setup the variables in template_string that need to be substituted
    subst['srcdir']=os.path.join(self.petsc_dir,rpath)
    subst['label']=self.nameSpace(defroot,subst['srcdir'])
    subst['output_file']=os.path.join(subst['srcdir'],testDict['output_file'])
    subst['exec']="../"+testDict['execname']
    subst['redirect_file']=testDict['redirect_file']
    subst['filter']=testDict['filter']
    subst['testroot']=self.testroot_dir
    subst['testname']=testname

    # Be careful with this
    if testDict.has_key('command'): subst['command']=testDict['command']

    # These can have for loops and are treated separately later
    if testDict.has_key('nsize'): subst['nsize']=str(testDict['nsize'])
    if testDict.has_key('args'):  subst['args']=testDict['args']

    #Conf vars
    subst['mpiexec']=self.conf['MPIEXEC']  # make sure PETSC_DIR is defined!
    subst['diff']=self.conf['DIFF']
    subst['rm']=self.conf['RM']
    subst['grep']=self.conf['GREP']

    return subst

  def getCmds(self,subst,i):
    """
      Generate bash script using template found next to this file.  
      This file is read in at constructor time to avoid file I/O
    """
    indent="   "
    nindent=i # the start and has to be consistent with below
    cmdLines=""
    # MPI is the default -- but we have a few odd commands
    if not subst.has_key('command'):
      cmd=indent*nindent+self._substVars(subst,example_template.mpitest)
    else:
      cmd=indent*nindent+self._substVars(subst,example_template.commandtest)
    cmdLines=cmdLines+cmd+"\n\n"

    cmd=indent*nindent+self._substVars(subst,example_template.difftest)
    cmdLines=cmdLines+cmd+"\n"
    return cmdLines

  def _substVars(self,subst,origStr):
    """
      Substitute varial
    """
    Str=origStr
    for subkey in subst:
      if type(subst[subkey])!=types.StringType: continue
      patt="@"+subkey.upper()+"@"
      Str=re.sub(patt,subst[subkey],Str)
    return Str

  def genRunScript(self,testname,root,isRun,srcDict):
    """
      Generate bash script using template found next to this file.  
      This file is read in at constructor time to avoid file I/O
    """
    # runscript_dir directory has to be consistent with gmakefile
    testDict=srcDict[testname]
    rpath=self.relpath(self.petsc_dir,root)
    runscript_dir=os.path.join(self.testroot_dir,rpath)
    if not os.path.isdir(runscript_dir): os.makedirs(runscript_dir)
    fh=open(os.path.join(runscript_dir,testname+".sh"),"w")
    petscvarfile=os.path.join(self.arch_dir,'lib','petsc','conf','petscvariables')
    
    subst=self.getSubstVars(testDict,rpath,testname)

    # Now substitute the key variables into the header and footer
    header=self._substVars(subst,example_template.header)
    footer=re.sub('@TESTSROOT@',subst['testroot'],example_template.footer)

    # Start writing the file
    fh.write(header+"\n")

    # If there is a TODO or a SKIP then we do it before writing out the
    # rest of the command (which is useful for working on the test)
    # SKIP and TODO can be for the source file or for the runs
    if srcDict.has_key("SKIP") or srcDict.has_key("TODO"):
      if srcDict.has_key("TODO"):
        todo=re.sub("@TODOCOMMENT@",srcDict['TODO'],example_template.todoline)
        fh.write(todo+"\ntotal=1; todo=1\n")
        fh.write(footer+"\n")
        fh.write("exit\n\n\n")
      elif srcDict.has_key("SKIP") or srcDict.has_key("TODO"):
        skip=re.sub("@SKIPCOMMENT@",srcDict['SKIP'],example_template.skipline)
        fh.write(skip+"\ntotal=1; skip=1\n")
        fh.write(footer+"\n")
        fh.write("exit\n\n\n")
    elif not isRun:
      skip=re.sub("@SKIPCOMMENT@",testDict['SKIP'],example_template.skipline)
      fh.write(skip+"\ntotal=1; skip=1\n")
      fh.write(footer+"\n")
      fh.write("exit\n\n\n")
    elif testDict.has_key('TODO'):
      todo=re.sub("@TODOCOMMENT@",testDict['TODO'],example_template.todoline)
      fh.write(todo+"\ntotal=1; todo=1\n")
      fh.write(footer+"\n")
      fh.write("exit\n\n\n")

    # Need to handle loops
    i=8  # for loop counters
    j=0  # for indentation 

    doForP=False
    if "{{" in subst['args']+subst['nsize']:
      doForP=True
      flinesP,dlinesP,i,j=self.getFor(subst,i,j)
      fh.write(flinesP+"\n")

    # Subtests are special
    if testDict.has_key("subtests"):
      substP=subst   # Subtests can inherit args but be careful
      if not substP.has_key("arg"): substP["arg"]=""
      jorig=j
      for stest in testDict["subtests"]:
        j=jorig
        subst=substP
        subst.update(testDict[stest])
        subst['nsize']=str(subst['nsize'])
        if not testDict[stest].has_key('args'): testDict[stest]['args']=""
        subst['args']=substP['args']+testDict[stest]['args']
        doFor=False
        if "{{" in subst['args']+subst['nsize']:
          doFor=True
          flines,dlines,i,j=self.getFor(subst,i,j)
          fh.write(flines+"\n")
        fh.write(self.getCmds(subst,j)+"\n")
        if doFor: fh.write(dlines+"\n")
    else:
      fh.write(self.getCmds(subst,j)+"\n")
      if doForP: fh.write(dlinesP+"\n")

    fh.write(footer+"\n")
    os.chmod(os.path.join(runscript_dir,testname+".sh"),0777)
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
    fileIsTested=False
    execname=self.getExecname(exfile,root)
    isBuilt=self._isBuilt(exfile,srcDict)
    for test in srcDict:
      if test in self.buildkeys: continue
      if debug: print self.nameSpace(exfile,root), test
      srcDict[test]['execname']=execname   # Convenience in generating scripts
      isRun=self._isRun(srcDict[test])
      self.genRunScript(test,root,isRun,srcDict)
      srcDict[test]['isrun']=isRun
      if isRun: fileIsTested=True
      self.addToTests(test,root,exfile,execname,srcDict[test])

    # This adds to datastructure for building deps
    if fileIsTested and isBuilt: self.addToSources(exfile,root,srcDict)
    #print self.nameSpace(exfile,root), fileIsTested
    return

  def _isBuilt(self,exfile,srcDict):
    """
    Determine if this file should be built. 
    """
    # Get the language based on file extension
    lang=self.getLanguage(exfile)
    if lang=="f" and not self.have_fortran: 
      srcDict["SKIP"]="Fortran required for this test"
      return False
    if lang=="cu" and not self.conf.has_key('PETSC_HAVE_CUDA'): 
      srcDict["SKIP"]="CUDA required for this test"
      return False
    if lang=="cxx" and not self.conf.has_key('PETSC_HAVE_CXX'): 
      srcDict["SKIP"]="C++ required for this test"
      return False

    # Deprecated source files
    if srcDict.has_key("TODO"): return False

    # isRun can work with srcDict to handle the requires
    if srcDict.has_key("requires"): 
      if len(srcDict["requires"])>0: 
        return self._isRun(srcDict)

    return True


  def _isRun(self,testDict):
    """
    Based on the requirements listed in the src file and the petscconf.h
    info, determine whether this test should be run or not.
    """
    indent="  "
    debug=False

    # MPI requirements
    if testDict.has_key('nsize'):
      if testDict['nsize']>1 and self.conf.has_key('MPI_IS_MPIUNI'): 
        if debug: print indent+"Cannot run parallel tests"
        testDict['SKIP']="Parallel test with serial build"
        return False
 
    # The requirements for the test are the sum of all the run subtests
    if testDict.has_key('subtests'):
      if not testDict.has_key('requires'): testDict['requires']=""
      for stest in testDict['subtests']:
        if testDict[stest].has_key('requires'):
          testDict['requires']=testDict['requires']+" "+testDict[stest]['requires']


    # Now go through all requirements
    if testDict.has_key('requires'):
      for requirement in testDict['requires'].split():
        requirement=requirement.strip()
        if not requirement: continue
        if debug: print indent+"Requirement: ", requirement
        isNull=False
        if requirement.startswith("!"):
          requirement=requirement[1:]; isNull=True
        # Scalar requirement
        if requirement=="complex":
          if self.conf['PETSC_SCALAR']=='complex':
            testDict['SKIP']="Non-complex build required"
            if isNull: return False
          else:
            testDict['SKIP']="Complex build required"
            return False
        # Precision requirement for reals
        if requirement in self.precision_types:
          if self.conf['PETSC_PRECISION']==requirement:
            testDict['SKIP']="not "+requirement+" required"
            if isNull: return False
          else:
            testDict['SKIP']=requirement+" required"
            return False
        # Precision requirement for ints
        if requirement in self.integer_types:
          if requirement=="int32":
            if self.conf['PETSC_SIZEOF_INT']==4:
              testDict['SKIP']="not int32 required"
              if isNull: return False
            else:
              testDict['SKIP']="int32 required"
              return False
          if requirement=="int64":
            if self.conf['PETSC_SIZEOF_INT']==8:
              testDict['SKIP']="NOT int64 required"
              if isNull: return False
            else:
              testDict['SKIP']="int64 required"
              return False
        # Datafilespath
        if requirement=="datafilespath": 
          testDict['SKIP']="Requires DATAFILESPATH"
          return False
        # Defines -- not sure I have comments matching
        if "define(" in requirement:
          reqdef=requirement.split("(")[1].split(")")[0]
          val=(reqdef.split()[1] if " " in reqdef else "")
          if self.conf.has_key(reqdef):
            if val:
              if self.conf[reqdef]==val:
                if isNull: 
                  testDict['SKIP']="Null requirement not met: "+requirement
                  return False
              else:
                testDict['SKIP']="Required: "+requirement
                return False
            else:
              if isNull: 
                testDict['SKIP']="Null requirement not met: "+requirement
                return False
          else:
            testDict['SKIP']="Requirement not met: "+requirement
            return False

        # Rest should be packages that we can just get from conf
        petscconfvar="PETSC_HAVE_"+requirement.upper()
        if self.conf.get(petscconfvar):
          if isNull: 
            testDict['SKIP']="Not "+petscconfvar+" requirement not met"
            return False
        else:
          if debug: print "requirement not found: ", requirement
          testDict['SKIP']=petscconfvar+" requirement not met"
          return False

    return True

  def genPetscTests_summarize(self,dataDict):
    """
    Required method to state what happened
    """
    if not self.summarize: return
    indent="   "
    fhname="GenPetscTests_summarize.txt"
    fh=open(fhname,"w")
    print "See ", fhname
    for root in dataDict:
      relroot=self.relpath(self.petsc_dir,root)
      pkg=relroot.split("/")[1]
      fh.write(relroot+"\n")
      allSrcs=[]
      for lang in LANGS: allSrcs=allSrcs+self.sources[pkg][lang]['srcs']
      for exfile in dataDict[root]:
        # Basic  information
        fullfile=os.path.join(root,exfile)
        rfile=self.relpath(self.petsc_dir,fullfile)
        builtStatus=(" Is built" if rfile in allSrcs else " Is NOT built")
        fh.write(indent+exfile+indent*4+builtStatus+"\n")

        for test in dataDict[root][exfile]:
          if test in self.buildkeys: continue
          line=indent*2+test
          fh.write(line+"\n")
          # Looks nice to have the keys in order
          #for key in dataDict[root][exfile][test]:
          for key in "isrun abstracted nsize args requires script".split():
            if not dataDict[root][exfile][test].has_key(key): continue
            line=indent*3+key+": "+str(dataDict[root][exfile][test][key])
            fh.write(line+"\n")
          fh.write("\n")
        fh.write("\n")
      fh.write("\n")
    #fh.write("\nClass Sources\n"+str(self.sources)+"\n")
    #fh.write("\nClass Tests\n"+str(self.tests)+"\n")
    fh.close()
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
      if not exfile.startswith("ex"): continue

      # Convenience
      fullex=os.path.join(root,exfile)
      relpfile=self.relpath(self.petsc_dir,fullex)
      if debug: print relpfile
      dataDict[root].update(testparse.parseTestFile(fullex))
      # Need to check and make sure tests are in the file
      # if verbosity>=1: print relpfile
      if dataDict[root].has_key(exfile):
        self.genScriptsAndInfo(exfile,root,dataDict[root][exfile])

    return

  def walktree(self,top,action="printFiles"):
    """
    Walk a directory tree, starting from 'top'
    """
    #print "action", action
    # Goal of action is to fill this dictionary
    dataDict={}
    for root, dirs, files in os.walk(top, topdown=False):
      if not "examples" in root: continue
      if not os.path.isfile(os.path.join(root,"makefile")): continue
      bname=os.path.basename(root.rstrip("/"))
      if bname=="tests" or bname=="tutorials":
        eval("self."+action+"(root,dirs,files,dataDict)")
      if type(top) != types.StringType:
          raise TypeError("top must be a string")
    # Now summarize this dictionary
    eval("self."+action+"_summarize(dataDict)")
    return dataDict

  def gen_gnumake(self, fd,prefix='srcs-'):
    """
     Overwrite of the method in the base PETSc class 
    """
    def write(stem, srcs):
        fd.write('%s :=\n' % stem)
        for lang in LANGS:
            fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang, srcs=' '.join(srcs[lang]['srcs'])))
            fd.write('%(stem)s += $(%(stem)s.%(lang)s)\n' % dict(stem=stem, lang=lang))
    for pkg in PKGS:
        srcs = self.gen_pkg(pkg)
        write(prefix + pkg, srcs)
    return self.gendeps

  def gen_pkg(self, pkg):
    """
     Overwrite of the method in the base PETSc class 
    """
    return self.sources[pkg]

  def write_gnumake(self,dataDict):
    """
     Write out something similar to files from gmakegen.py

     There is not a lot of has_key type checking because
     should just work and need to know if there are bugs

     Test depends on script which also depends on source
     file, but since I don't have a good way generating
     acting on a single file (oops) just depend on
     executable which in turn will depend on src file
    """
    # Open file
    arch_files = self.arch_path('lib','petsc','conf', 'testfiles')
    arg_files = self.arch_path('lib','petsc','conf', 'testargfiles')
    fd = open(arch_files, 'w')
    fa = open(arg_files, 'w')

    # Write out the sources
    gendeps = self.gen_gnumake(fd,prefix="testsrcs-")

    # Write out the tests and execname targets
    fd.write("\n#Tests and executables\n")    # Delimiter
    testdeps=" ".join(["test-"+pkg for pkg in PKGS])
    testexdeps=" ".join(["test-ex-"+pkg for pkg in PKGS])
    fd.write("test: testex "+testdeps+" report_tests\n")    # Main test target
    # Add executables to build right way to make the `make test` look
    # nicer
    fd.write("testex: "+testexdeps+"\n")    # Main test target

    for pkg in PKGS:
      # These grab the ones that are built
      # Package tests
      testdeps=" ".join(["test-"+pkg+"-"+lang for lang in LANGS])
      fd.write("test-"+pkg+": "+testdeps+"\n")
      testexdeps=" ".join(["test-ex-"+pkg+"-"+lang for lang in LANGS])
      fd.write("test-ex-"+pkg+": "+testexdeps+"\n")
      # This needs work
      if self.single_ex:
        execname=pkg+"-ex"
        fd.write(execname+": "+" ".join(self.objects[pkg])+"\n\n")
      for lang in LANGS:
        testdeps=""
        for ftest in self.tests[pkg][lang]:
          test=os.path.basename(ftest)
          basedir=os.path.dirname(ftest)
          testdeps=testdeps+" "+self.nameSpace(test,basedir)
        fd.write("test-"+pkg+"-"+lang+":"+testdeps+"\n")

        # test targets
        for ftest in self.tests[pkg][lang]:
          test=os.path.basename(ftest)
          basedir=os.path.dirname(ftest)
          testdir="${TESTDIR}/"+basedir+"/"
          nmtest=self.nameSpace(test,basedir)
          rundir=os.path.join(testdir,test)
          #print test, nmtest
          script=test+".sh"

          # Deps
          exfile=self.tests[pkg][lang][ftest]['exfile']
          fullex=os.path.join(self.petsc_dir,exfile)
          localexec=self.tests[pkg][lang][ftest]['exec']
          execname=os.path.join(testdir,localexec)

          # SKIP and TODO tests do not depend on exec
          if exfile in self.sources[pkg][lang]['srcs']:
            #print "Found dep: "+exfile, execname
            fd.write(nmtest+": "+execname+"\n")
          else:
            # Still add dependency to file
            fd.write(nmtest+": "+fullex+"\n")
          cmd=testdir+"/"+script
          fd.write("\t-@"+cmd+"\n")
          # Now write the args:
          fa.write(nmtest+"_ARGS='"+self.tests[pkg][lang][ftest]['argLabel']+"'\n")

        # executable targets -- add these to build earlier
        testexdeps=""
        if not self.single_ex:
          for exfile in self.sources[pkg][lang]['srcs']:
            localexec=os.path.basename(os.path.splitext(exfile)[0])
            basedir=os.path.dirname(exfile)
            testdir="${TESTDIR}/"+basedir+"/"
            execname=os.path.join(testdir,localexec)
            testexdeps=testexdeps+" "+execname
          fd.write("test-ex-"+pkg+"-"+lang+":"+testexdeps+"\n")

        for exfile in self.sources[pkg][lang]['srcs']:
          root=os.path.join(self.petsc_dir,os.path.dirname(exfile))
          basedir=os.path.dirname(exfile)
          testdir="${TESTDIR}/"+basedir+"/"
          base=os.path.basename(exfile)
          objfile=testdir+os.path.splitext(base)[0]+".o"
          linker=self.getLanguage(exfile)[0].upper()+"LINKER"
          if self.sources[pkg][lang].has_key(exfile):
            # Dependency for file
            objfile=objfile+" "+self.sources[pkg][lang][exfile]
            print objfile
          if not self.single_ex:
            localexec=os.path.basename(os.path.splitext(exfile)[0])
            execname=os.path.join(testdir,localexec)
            localobj=os.path.basename(objfile)
            petsc_lib="${PETSC_"+pkg.upper()+"_LIB}"
            fd.write("\n"+execname+": "+objfile+" ${libpetscall}\n")
            # There should be a better way here
            line="\t-cd "+testdir+"; ${"+linker+"} -o "+localexec+" "+localobj+" "+petsc_lib
            fd.write(line+"\n")
          linker=self.getLanguage(exfile)[0].upper()+"LINKER"

    fd.write("helptests:\n\t -@grep '^[a-z]' ${generatedtest} | cut -f1 -d:\n")
    # Write out tests
    return

  def writeHarness(self,output,dataDict):
    """
     This is set up to write out multiple harness even if only gnumake
     is supported now
    """
    eval("self.write_"+output+"(dataDict)")
    return

def main(petsc_dir=None, petsc_arch=None, output=None, verbose=False, single_ex=False):
    if output is None:
        output = 'gnumake'


    pEx=generateExamples(petsc_dir=petsc_dir, petsc_arch=petsc_arch, verbose=verbose, single_ex=single_ex)
    dataDict=pEx.walktree(os.path.join(pEx.petsc_dir,'src'),action="genPetscTests")
    pEx.writeHarness(output,dataDict)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--output', help='Location to write output file', default=None)
    parser.add_option('-s', '--single_executable', dest='single_executable', action="store_false", help='Whether there should be single executable per src subdir.  Default is false')
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, output=opts.output, verbose=opts.verbose, single_ex=opts.single_executable)
