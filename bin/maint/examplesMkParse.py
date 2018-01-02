#!/usr/bin/env python

import glob
import sys
import re
import os
import stat
import types
import optparse
import string
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 

"""
Class for extracting the shell scripts from the makefile

Usage: examplesMkParse.py <-p petsc_dir> <-m makefile> <-v verbosity>

In general:
    cd src/ksp/ksp/examples/tutorials
    ../../../../../bin/maint/examplesMkParse.py -v 1
 
"""

class makeParse(object):
  def __init__(self,petsc_dir,replaceSource,verbosity):

    self.petsc_dir=petsc_dir
    self.writeScripts=True
    #self.scriptsSubdir=""
    self.scriptsSubdir="from_makefile"
    self.ptNaming=False
    self.insertIntoSrc=True
    self.verbosity=verbosity
    self.replaceSource=replaceSource
    return

  def makeRunDict(self,mkDict,curdir):
    """
    parseTESTline produces a dictionary of the form:
      dataDict[TESTEXAMPLES_*]['tsts']
      dataDict[TESTEXAMPLES_*]['srcs']
    Put it into a form that is easier for process:
      General form:
        runDict[srcfile][runtest]['test']={}
        runDict[srcfile][runtest]['test']['args']={}
      or if subtests
        runDict[srcfile][runtest]['test']['test1']['args']={}
      Info:
        runDict['not_tested']  -- the run* targets in makefile that are not invoked
        runDict[srcfile]['requires'] -- build requirements from makefile (TEST* info)
        runDict[srcfile]['types']    -- What TEST* types it was found in the makefile
        runDict[srcfile][runtest]['requires'] -- build requirements from makefile (TEST* info)
        runDict[srcfile][runtest]['types']    -- What TEST* types it was found in the makefile
        runDict[srcfile][runtest]['abstracted']  -- Abstraction successful?
    """
    import convertExamplesUtils
    makefileMap=convertExamplesUtils.makefileMap
    def getReqs(typeList,reqtype):
      """ Determine requirements associated with srcs and tests """
      allRqs=[]
      for atype in typeList:
        i=0
        for subtype in atype.split("_"):
          # Messy
          if subtype=="TESTEXAMPLES": continue
          if subtype=="C": continue
          if subtype=="MKL": subtype="MKL_PARDISO"
          if subtype=="SUPERLU" and "SUPERLU_DIST" in subtype: subtype="SUPERLU_DIST"
          if makefileMap.has_key(subtype):
            if makefileMap[subtype].startswith(reqtype):
              rqList=makefileMap[subtype].split("requires:")[1].strip().split()
              allRqs=allRqs+rqList
      if len(allRqs)>0: return list(set(allRqs))  # Uniquify the requirements
      return allRqs

    # As I reorder the dictionary, keep track of the types
    rDict={}
    for extype in mkDict:
      if not extype.startswith("TEST"): continue
      for exfile in mkDict[extype]['srcs']:
        srcfile=self.undoNameSpace(exfile)
        if srcfile=="printdot": continue
        if not rDict.has_key(srcfile): rDict[srcfile]={}
        if not rDict[srcfile].has_key('types'): rDict[srcfile]['types']=[]
        rDict[srcfile]['types'].append(extype)
        testList=self.findTests(exfile,mkDict[extype]['tsts'])
        for test in testList:
          if not rDict[srcfile].has_key(test):
            rDict[srcfile][test]={}
            rDict[srcfile][test]['types']=[]
          rDict[srcfile][test]['types'].append(extype)
          # Transfer information from makefile, especially script
          if mkDict.has_key(test): rDict[srcfile][test].update(mkDict[test])

    # Now that all types are known, determine requirements
    for sfile in rDict:
      rDict[sfile]['requires']=getReqs(rDict[sfile]['types'],"buildrequires")
      for test in rDict[sfile]:
        if not test.startswith("run"): continue
        rDict[sfile][test]['requires']=getReqs(rDict[sfile][test]['types'],"requires")

    # Determine tests that are not invoked (in any circumstance)
    # In this case, we have to guess at the sourcefile
    # We have both stray tests and stray files.  This find stray files
    # but we label those tests as stray as well
    rDict['not_tested']=[]
    srcfilesCheck=[]
    for mkrun in mkDict:
      if not mkrun.startswith("run"): continue
      found=False
      for sfile in rDict:
        for runex in rDict[sfile]:
          if runex==mkrun: found=True
      if not found: 
        rDict['not_tested'].append(mkrun)
        sfile=self.findSrcfile(mkrun,curdir)
        if sfile==None:
          relpath=os.path.relpath(curdir, self.petsc_dir)
          print "Cannot associate source file with "+mkrun+" in "+relpath
        else:
          if not rDict.has_key(sfile): rDict[sfile]={}
          if not rDict[sfile].has_key(mkrun): rDict[sfile][mkrun]={}
          rDict[sfile][mkrun].update(mkDict[mkrun])
          rDict[sfile]['TODO']="Need to determine if deprecated"
          rDict[sfile][mkrun]['TODO']="Need to determine if deprecated"
        
    # Now can start abstracting the script itself to figure out 
    # the test structures
    sfiles=rDict.keys()
    for sfile in sfiles:
      if sfile=="not_tested": continue
      tests=rDict[sfile].keys()
      for runex in tests:
        if not runex.startswith("run"): continue
        if not self.abstractScript(runex,rDict[sfile][runex],curdir):
          del rDict[sfile][runex]
        else:
          # Cleanup the TODO and REQUIRES level now that 'test' exists
          if rDict[sfile][runex].has_key('TODO'):
            rDict[sfile][runex]['test']['TODO']=rDict[sfile][runex]['TODO']
            del rDict[sfile][runex]['TODO']
          if rDict[sfile][runex].has_key('requires'):
            if len(rDict[sfile][runex]['requires'])>0:
              rDict[sfile][runex]['test']['requires']=rDict[sfile][runex]['requires']
            del rDict[sfile][runex]['requires']

    return rDict

  def fixScript(self,scriptStr,varVal):
    """
    makefile is commands are not proper bash so need to fix that
    Our naming scheme is slightly different as well, so fix that
    Simple replaces done here -- this is not sophisticated
    Also, this may contain variables defined in makefile and need to do a
    substitution
    """
    scriptStr=scriptStr.replace("then","; then")
    #scriptStr=scriptStr.strip().replace("\\","")
    scriptStr=scriptStr.replace("; \\","\n")
    scriptStr=scriptStr.replace("do\\","do\n")
    scriptStr=scriptStr.replace("do \\","do\n")
    scriptStr=scriptStr.replace("done;\\","done\n")
    # Note the comment out -- I like to see the output
    scriptStr=scriptStr.replace("-@","")
    # Thsi is for ts_eimex*.sh
    scriptStr=scriptStr.replace("$$","$")
    if 'for' in scriptStr:
      scriptStr=scriptStr.replace("$(seq","")
      scriptStr=scriptStr.replace(");",";")
    #if '(${DIFF}' in scriptStr:
    #  scriptStr=scriptStr.split("(")[1]
    #  scriptStr=scriptStr.split(")")[0]
    #  tmpscriptStr=sh.readline()
    if '${DIFF}' in scriptStr.lower() and '||' in scriptStr:
      scriptStr=scriptStr.split("||")[0].strip()
    for var in varVal.keys():
      if var.startswith("run"): continue
      if var in scriptStr:
        replStr="${"+var+"}"
        scriptStr=scriptStr.replace(replStr,varVal[var])
    scriptStr=scriptStr.replace("\n\n","\n")
    scriptStr=scriptStr.replace("\\\n","")
    return scriptStr

  def getVarVal(self,line,fh):
    """
    Process lines of form var = val.  Mostly have to handle
    continuation lines
    """
    while 1:
      last_pos=fh.tell()
      if line.strip().endswith("\\"):
        line=line.strip().rstrip("\\")+" "+fh.readline().strip()
      else:
        fh.seek(last_pos)  # might be grabbing next var
        break
    if self.verbosity>=3: print "       getVarVal> line ", line
    valList=line.split("=")[1:]
    val=" ".join(valList).strip()
    return val

  def extractRunFile(self,fh,line,mkfile,varVal):
    """
    Given the file handle which points to the location in the file where
    a runex: has just been read in, write it out to the file in the
    directory where mkfile is located
    """
    runexName=line.split(":")[0].strip()
    alphabet=tuple(string.ascii_letters)
    shStr=""
    basedir=os.path.dirname(mkfile)
    shName=os.path.join(basedir,runexName+".sh")
    while 1:
      last_pos=fh.tell()
      line=fh.readline()
      if not line: break
      if line.startswith(alphabet) or line.startswith("#"): 
        fh.seek(last_pos)  # might be grabbing next script so rewind
        break
      shStr=shStr+" "+line
    if not shStr.strip(): return "",""
    newShStr=self.fixScript(shStr,varVal)
    # 
    if self.writeScripts:
      if self.scriptsSubdir: 
        subdir=os.path.join(basedir,self.scriptsSubdir)
        if not os.path.isdir(subdir): os.mkdir(subdir)
        shName=os.path.join(subdir,runexName+".sh")
      shh=open(shName,"w")
      shh.write(newShStr)
      shh.close()
      os.chmod(shName,0777)
    return runexName, newShStr

  def parseRunsFromMkFile(self,fullmake):
    """
     Parse out the runs and related variables.  
     Return two dictionaries, one with makefile info 
      and one containing all of the tests
    """
    fh=open(fullmake,"r")
    root=os.path.dirname(os.path.realpath(fullmake))

    # Go through the makefile, and for each run* target: 
    #     extract, abstract, insert
    mkDict={}
    i=0
    if self.verbosity>=3: print "parseRunsFromMkFile> ", fullmake
    alphabet=tuple(string.ascii_letters)
    while 1:
      line=fh.readline()
      if not line: break
      # Scripts might have substitutions so need to store all of the
      # defined variables in the makefile
      # For TESTS* vars, we want info in special dictionary
      if line.startswith(alphabet) and "=" in line: 
        var=line.split("=")[0].strip()
        # This does substitution to get filenames
        if line.startswith("TEST"):
          mkDict[var]=self.parseTESTline(fh,line,root)
        else:
          mkDict[var]=self.getVarVal(line,fh)
      
      # Only keep the run targets in addition to vars
      # Do some transformation of the script string at this stage
      if line.startswith("run"): 
        # ts/examples/tutorials/makefile has runex40: runex40_a ...
        # Need to handle this phony target
        if line.split(":")[1].split("#")[0].strip(): continue

        runex,shStr=self.extractRunFile(fh,line,fullmake,mkDict)
        if self.verbosity>=3: 
          print "parseRunsFromMkFile> ", runex
          print "parseRunsFromMkFile>    ", shStr
        mkDict[runex]={}
        mkDict[runex]['script']=shStr
    fh.close()

    # mkDict in form related to parsing files.  Need it in 
    # form easier for generating the tests
    runDict=self.makeRunDict(mkDict,root)

    return runDict

  def getOrderedRuns(self,subDict):
    """
    Order the runs by suffix for niceness
    Probably could use orderedDict's
    Might be able to just sort via the keys, but doing suffixes anyway
    """
    order=[]
    oDict={}
    for run in subDict:
      if not run.startswith("run"): continue
      suffix=subDict[run]['test']['suffix']
      order.append(suffix)
      oDict[suffix]=run
    order.sort()
    orderedRuns=[]
    for s in order: orderedRuns.append(oDict[s])
    return orderedRuns

  def getOrderedKeys(self,subDict):
    """
    It looks nicer to have the keys in an ordered way, and 
    do not write out defaults
    """
    order=[]
    if subDict.has_key("suffix"):
      if subDict["suffix"].strip(): order.append("suffix")
    if subDict.has_key("nsize"):
      try:
        nint=int(subDict["nsize"])
        if nint>1: order.append("nsize")
      except:
        order.append("nsize")
    if subDict.has_key("requires"):
      if len(subDict["requires"])>0: order.append("requires")
    if subDict.has_key("args"):
      if subDict["args"].strip(): order.append("args")
    if subDict.has_key("filter"):
      if subDict["filter"].strip(): order.append("filter")
    if subDict.has_key("output_file"):
      if subDict["output_file"].strip(): order.append("output_file")
    if subDict.has_key("localrunfiles"):
      if subDict["localrunfiles"].strip(): order.append("localrunfiles")
     
    tests=[]
    for k in subDict: 
      if k.startswith("test"): tests.append(k)
    tests.sort()
    return order+tests

  def insertTestInfo(self,fileStr,srcDict,isFortran):
    """
    This gets all of the stuff between /*TEST ... TEST*/
    """
    indent="   "
    testStr=""
    for runex in self.getOrderedRuns(srcDict):
      testStr=testStr+indent+"\n"+indent+"test:\n"
      # For various reasons, this is at an awkward level
      if srcDict[runex].has_key("TODO"):
        if srcDict[runex]["TODO"]:
          todostr=re.sub("TODO:","",srcDict[runex]["TODO"])
          testStr=testStr+indent*2+"TODO: "+todostr+"\n"
      # Need to check and see if we have to push this down
      if srcDict[runex].has_key("requires"):
        print srcDict[runex]["requires"]
      rDict=srcDict[runex]['test']
      # Do all of the general info
      for rkey in self.getOrderedKeys(rDict):
        if not rkey.startswith('test'):
          rval=(rDict[rkey] if rkey!="requires" else " ".join(rDict[rkey]))
          testStr=testStr+indent*2+rkey+": "+str(rval)+"\n"
        # Do the subtests
        else:
          testStr=testStr+"\n"+indent*2+"test:\n"
          for tkey in self.getOrderedKeys(rDict[rkey]):
            rval=(rDict[rkey][tkey] if tkey!="requires" else " ".join(rDict[rkey][tkey]))
            testStr=testStr+indent*3+tkey+": "+str(rval)+"\n"
      # For various reasons, this is at an awkward level
      if rDict.has_key("TODO"):
        if rDict["TODO"]:
          todostr=re.sub("TODO:","",rDict["TODO"])
          testStr=testStr+indent*2+"TODO: "+todostr+"\n"
    # If no test, then put it on the TODO list
    if not testStr.strip(): 
      testStr=testStr+"\n"+indent+"test:"+"\n"+indent*2+"TODO: Need to implement test\n"
    testStr="\n\n/*TEST\n"+testStr+'\nTEST*/\n'
    if isFortran: 
      testStr=testStr.replace("\n","\n!").rstrip("!")
      testStr=re.sub("!\s+\n","!\n",testStr)
    return testStr

  def _isFortran(self,filename):
    """  Return boolean on whether it is a fortran file or not """
    if os.path.splitext(filename)[1].startswith(".F"):
      return True
    else:
      return False

  def insertSrcInfo(self,fileStr,srcDict,isFortran):
    """
    Put the information in the dictionary into the source file
    """
    # First see if we have stuff to add
    if srcDict.has_key('requires'):
      if len(srcDict['requires'])==0: del(srcDict['requires'])
    if not srcDict.has_key('requires') and not srcDict.has_key('TODO'): return fileStr
      
    if "/*T\n" in fileStr or "/*T " in fileStr:
      # The file info is already here and need to append
      Part1=fileStr.split("T*/")[0]
      suffix=" ".join(fileStr.split("T*/")[1:])
      prefix=Part1.split("/*T")[0]
      fileInfo=Part1.split("/*T")[1]
    else:
      prefix=fileStr.split("#include")[0]
      nlines=len(prefix.split("\n"))
      suffix="\n".join(fileStr.split("\n")[nlines-1:])
      fileInfo=""

    indent="   "
    if isFortran: indent="!"+indent
    insertStr=""
    if srcDict.has_key('requires'):
      insertStr=insertStr+indent+"requires: "+" ".join(srcDict['requires'])
    if srcDict.has_key('TODO'): 
      insertStr=insertStr+indent+"TODO: "+srcDict['TODO']
    fileInfo=fileInfo.lstrip("\n")+insertStr+"\n"
    if not isFortran: 
      newFileStr=prefix+"/*T\n"+fileInfo+"T*/\n\n"+suffix
    else:
      newFileStr=prefix+"!/*T\n"+fileInfo+"!T*/\n\n"+suffix
    return newFileStr

  def insertInfoIntoSrc(self,fullmake,testDict):
    """
    Put the information in the dictionary into the source file
    """
    basedir=os.path.dirname(os.path.realpath(fullmake))
    startdir=os.path.realpath(os.path.curdir)
    os.chdir(basedir)

    for sfile in testDict:
      if sfile=='not_tested': continue
      if self.verbosity>=2: print sfile
      sh=open(sfile,"r"); fileStr=sh.read(); sh.close()
      isFortran=self._isFortran(sfile)
      newFileStr=self.insertSrcInfo(fileStr,testDict[sfile],isFortran)
      testFileStr=self.insertTestInfo(fileStr,testDict[sfile],isFortran)
      # Write out new file
      fname=(sfile if self.replaceSource else "new_"+sfile)
      fh=open(fname,"w")
      fh.write(newFileStr+testFileStr)
      fh.close()

    os.chdir(startdir)
    return 

  def abstractForLoops(self,scriptStr):
    """
    If it has a for loop, then need to modify the script string
    to use the {{ ... }} syntax
    """
    new=""
    forvars=[]
    forvals={}
    for line in scriptStr.split("\n"):
      if "for " in line:
        fv=line.split("for")[1].split("in")[0].strip()
        forvars.append(fv)
        forvals[fv]=line.split("in")[1].split(";")[0].strip()
      elif " done" in line and not "MPIEXEC" in line:
        pass
      else:
        for fv in forvars:
          fl=forvals[fv]
          line=re.sub("\$"+fv+" ","{{"+fl+"}} ",line)
        if "done;" in line: line=re.sub("done;","",line).strip().rstrip(";")
        new=new+line+"\n"
    return new

  def abstractMultiMpiTest(self,runexName,scriptStr,subDict,subdir):
    """
    Multiple mpi tests leads to a new tests then need to create tests
    """
    subDict['subtests']=[]
    i=0
    allAbstracted=True
    for line in scriptStr.split("\n"):
      if "MPIEXEC" in line:
        subtestName="test"+str(i); i=i+1
        subDict['subtests'].append(subtestName)
        subDict[subtestName]={}
        abstracted=self.abstractMpiTest(runexName,line,subDict[subtestName],subdir)
        if not abstracted: allAbstracted=False
    return allAbstracted

  def _hasFilter(self,scriptStr):
    """ Determine if the filter keyword needs to be filled """
    for fltr in ['sed','grep','sort']:
      if fltr+" " in scriptStr: return True
    return False

  def abstractFilter(self,runexName,scriptStr,subDict):
    """
    Figure out how the abstract work
    """
    allFilters=[]
    for line in scriptStr.split("\n"):
      if self._hasFilter(line):
        splitl=line.split(">")[0]
        if "|" in splitl:
          testFilter="|".join(splitl.split("|")[1:]).strip()
          # This picks up filtering of output file in sys-tutorials
          if "; then" in testFilter: testFilter="TODO: Need to check for filter and/or filter_output"
        else:
          if splitl[-1].endswith(".tmp"):
            testFilter=" ".join(splitl[:-1])
          else:
            if "MPIEXEC" in splitl:
              testFilter="TODO: Could not abstract filter"
              return ""
            else:
              testFilter=" ".join(splitl[:])
        allFilters.append(testFilter)

    return " | ".join(allFilters)

  def abstractTest(self,runexName,scriptStr,subDict,subdir):
    """
    Handle tests that are non-MPI
    """
    # Help in parsing
    runexBase=runexName.split("_")[0]
    exName=runexBase[3:]

    # Assume it is of the form 'cmd > output_file'
    cmd=scriptStr.split(" >")[0]
    subDict['command']=cmd

    # Don't even worry about trying to get the arguments

    # Determine requirements based on args
    import convertExamplesUtils
    argMap=convertExamplesUtils.argMap
    if subDict.has_key('args'):
      if subDict.has_key('requires'):
        allRqs=subDict['requires']
      else:
        allRqs=[]
      for matchStr in argMap:
        if matchStr in subDict['args']:
          rqList=argMap[matchStr].split("requires:")[1].strip().split()
          allRqs=allRqs+rqList
      subDict['requires']=list(set(allRqs))  # Uniquify the requirements
      if 'options_file_yaml' in subDict['args']:
        subDict['localrunfiles']=subDict['args'].split('options_file_yaml')[1].split()[0]

    # Pull out the redirect file
    if "> " in scriptStr:
      redfile=scriptStr.split('> ')[1].split('2>')[0].strip()
    else:
      # Without redirect, cannot do diffs so it needs work
      subDict['TODO']="Need to develop comparison test"

    # Do filters
    if self._hasFilter(scriptStr):
      if self.verbosity>=1: print "FILTER", runexName
      filterTxt=self.abstractFilter(runexName,scriptStr,subDict)
      if not "TODO:" in filterTxt:
        subDict['filter']=filterTxt
      else:
        subDict['TODO']=filterTxt

    # If subtests, then we do not have the diff here or it could
    if not "{DIFF}" in scriptStr: return

    # Check the diff and make sure everything is OK with files
    if not subDict.has_key('TODO'):
      diffPart=scriptStr.split("{DIFF}")[1].split("\n")[0]
      outfile=diffPart.split()[0]
      diffredfile=diffPart.split()[1].rstrip(")")
      defaultOut="output/"+runexName[3:]+".out"
      if outfile!=defaultOut: subDict['output_file']=outfile
      if redfile!=diffredfile: 
        # Usually this problem can't be fit into our current system
        # so just flag this as a test that needs work
        subDict['TODO']="Needs further development from conversion"

    return True

  def abstractMpiTest(self,runexName,scriptStr,subDict,subdir):
    """
    If it has a for loop, then need to modify the script string
    to use the {{ ... }} syntax
    subDict is at the [src][runex]['test'] level of directory dictionary
    """
    # We always want nsize even if not abstracted 
    mpiLine=scriptStr.split("{MPIEXEC}")[1]
    if "\n" in mpiLine: mpiLine=mpiLine.split("\n")[0]
    nsizeStr=mpiLine.split(" -n ")[1].split()[0].strip()
    if "{{" in nsizeStr:
      # Loops are special
      ns=mpiLine.split(" -n ")[1].split("}}")[0].split("{{")[1]
      subDict['nsize']="{{"+ns+"}}"
    else:
      try:
        subDict['nsize']=int(nsizeStr)
      except:
        print "Problem in finding nsize: ", runexName
        return False

    # Help in parsing
    runexBase=runexName.split("_")[0]
    exName=runexBase[3:]
    firstPart=mpiLine.split(" >")[0]+" "

    # Args
    args=firstPart.split(exName+" ")[1].strip().strip("\\")
    # Peel off filters with "|" and remove extraneous white space with split/join
    if args.strip(): subDict['args']=" ".join(args.split("|")[0].split())

    # Determine requirements based on args
    import convertExamplesUtils
    argMap=convertExamplesUtils.argMap
    if subDict.has_key('args'):
      if subDict.has_key('requires'):
        allRqs=subDict['requires']
      else:
        allRqs=[]
      for matchStr in argMap:
        argMatch=matchStr.lower()+" "
        if argMatch in subDict['args']:
          rqList=argMap[matchStr].split("requires:")[1].strip().split()
          allRqs=allRqs+rqList
      subDict['requires']=list(set(allRqs))  # Uniquify the requirements
      if 'options_file_yaml' in subDict['args']:
        subDict['localrunfiles']=subDict['args'].split('options_file_yaml')[1].strip().split()[0]

    # Pull out the redirect file
    if "> " in mpiLine:
      redfile=mpiLine.split('> ')[1].split('2>')[0].strip()
    else:
      # Without redirect, cannot do diffs so it needs work
      subDict['TODO']="Need to develop comparison test"

    # Do filters
    if self._hasFilter(scriptStr):
      if self.verbosity>=1: print "FILTER", runexName
      filterTxt=self.abstractFilter(runexName,scriptStr,subDict)
      if not "TODO:" in filterTxt:
        subDict['filter']=filterTxt
      else:
        subDict['TODO']=filterTxt

    # If subtests, then we do not have the diff here or it could
    if not "{DIFF}" in scriptStr: return

    # Check the diff and make sure everything is OK with files
    if not subDict.has_key('TODO'):
      diffPart=scriptStr.split("{DIFF}")[1].split("\n")[0]
      outfile=diffPart.split()[0]
      diffredfile=diffPart.split()[1].rstrip(")")
      defaultOut="output/"+runexName[3:]+".out"
      if outfile!=defaultOut: subDict['output_file']=outfile
      if redfile!=diffredfile: 
        # Usually this problem can't be fit into our current system
        # so just flag this as a test that needs work
        subDict['TODO']="Needs further development from conversion"

    return True

  def abstractScript(self,runexName,abstract,curdir):
    """
    Do a preliminary pass of abstracting the script
    Abstract script tries to take a normal script string and then parse it
    out such that  it is in the new format
    Abstraction is to fill out:
      NSIZE, ARGS, OUTPUT_SUFFIX 
    If we can't abstract, we can't abstract and we'll just put in a script
    """
    abstract['abstracted']=False
    if not abstract.has_key('script'):
      relpath=os.path.relpath(curdir, self.petsc_dir)
      print "Cannot find script "+runexName+" in "+relpath
      return False
    scriptStr=abstract['script']
    # remove whats in print message to reduce false positives
    noPrintScript=re.sub('printf.*;','',scriptStr)

    #Create subdictionary
    abstract['test']={}

    # OUTPUT SUFFIX should be equivalent to runex target so we just go by that.
    runexBase=runexName.split("_")[0]
    abstract['test']['suffix']=runexName[len(runexBase)+1:]

    # Do for loop first because that is local parsing
    if "for " in noPrintScript:
      if self.verbosity>=1: print "FOR LOOP", runexName
      scriptStr=self.abstractForLoops(scriptStr)

    # Handle subtests if needed.  
    if noPrintScript.count("MPIEXEC")>1:
      if self.verbosity>=1: print "MultiMPI", runexName
      abstract['abstracted']=self.abstractMultiMpiTest(runexName,scriptStr,abstract['test'],curdir)
    elif noPrintScript.count("MPIEXEC")==1:
      abstract['abstracted']=self.abstractMpiTest(runexName,scriptStr,abstract['test'],curdir)
    else:
      if self.verbosity>=1: print "No MPI", runexName
      abstract['abstracted']=self.abstractTest(runexName,scriptStr,abstract['test'],curdir)

    return True

  def undoNameSpace(self,srcfile):
    """
    Undo the nameSpaceing
    """
    if self.ptNaming:
      nameString=srcfile.split("-")[1]
    else:
      nameString=srcfile
    return nameString

  def nameSpace(self,srcfile,srcdir):
    """
    Because the scripts have a non-unique naming, the pretty-printing
    needs to convey the srcdir and srcfile.  There are two ways of doing
    this.
    """
    if self.ptNaming:
      cdir=srcdir.split('src')[1].lstrip("/").rstrip("/")
      prefix=cdir.replace('/examples/','_').replace("/","_")+"-"
      nameString=prefix+srcfile
    else:
      #nameString=srcdir+": "+srcfile
      nameString=srcfile
    return nameString

  def getTestsStr(self,fileStr):
    """
    Given a string that has the /*TESTS testsStr TESTS*/ 
    embedded within it, return testsStr
    """
    if not "/*TESTS" in fileStr: return fileStr,""
    first=fileStr.split("/*TESTS")[0]
    fsplit=fileStr.split("/*TESTS")[1]
    testsStr=fsplit.split("TESTS*/")[0]
    return first,testsStr

  def getSourceFileName(self,petscName,srcdir):
    """
    Given a PETSc name of the form ex1.PETSc or ex2.F.PETSc 
    find the source file name
    Source directory is needed to handle the fortran
    """
    # Bad spelling
    word=petscName
    if word.rstrip(".PETSc")[-1]=='f':
      newword = word.replace('PETSc','F')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','F90')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this fortran file: ", srcdir, word
    elif 'f90' in word:
      newword = word.replace('PETSc','F90')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','F')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this f90 file: ", srcdir, word
        newword=""
    # For files like  
    elif os.path.splitext(word)[0].endswith('cu'):
      newword = word.replace('PETSc','cu')
    else:
      # This is a c file required for the 
      newword = word.replace('PETS','')
      # This means there is a bug in the makefile.  Move along
      if not os.path.isfile(os.path.join(srcdir,newword)):
        newword = word.replace('PETSc','cxx')
      if not os.path.isfile(os.path.join(srcdir,newword)):
        print "I give up on this: ", srcdir, word
        newword=""
    return newword

  def findSrcfile(self,testname,curdir):
    """
    Given a testname of the form runex10_9, try to figure out the source file
    """
    base=testname[3:].split("_")[0]
    if base.endswith("f") or base.endswith("f90"):
      for ext in ['F','F90']:
        guess=os.path.join(curdir,base+"."+ext)
        if os.path.exists(guess): return os.path.basename(guess)
    else:
      for ext in ['c','cxx']:
        guess=os.path.join(curdir,base+"."+ext)
        if os.path.exists(guess): return os.path.basename(guess)
    # Sometimes the underscore is in the executable (ts-tutorials)
    base=testname[3:]
    if base.endswith("f") or base.endswith("f90"):
      for ext in ['F','F90']:
        guess=os.path.join(curdir,base+"."+ext)
        if os.path.exists(guess): return os.path.basename(guess)
    else:
      for ext in ['c','cxx']:
        guess=os.path.join(curdir,base+"."+ext)
        if os.path.exists(guess): return os.path.basename(guess)
    return None

  def findTests(self,srcfile,testList):
    """
    Given a source file of the form ex1.c and a list of tests of the form
    ['runex1', 'runex1_1', 'runex10', ...]
    Return the list of tests that should be associated with that srcfile
    """
    mtch=os.path.splitext(srcfile)[0]
    if self.ptNaming: mtch=mtch.split("-")[1]
    newList=[]
    for test in testList:
      if self.ptNaming: test=test.split("-")[1]
      if test.split("_")[0][3:]==mtch: newList.append(test)
    return newList

  def parseTESTline(self,fh,line,srcdir):
    """
    For a line of the form:
      VAR = ex1.PETSc runex1
    return two lists of the source files and run files
    getSourceFileName is used to change PETSc into the 
    appropriate file extension
     - fh is the file handle to the makefile
     - srcdir is where the makefile and examples are located
    Note for EXAMPLESC and related vars, it ex1.c instead of ex1.PETSc
    """
    parseDict={}; parseDict['srcs']=[]; parseDict['tsts']=[]
    while 1:
      last_pos=fh.tell()
      if line.strip().endswith("\\"):
        line=line.strip().rstrip("\\")+" "+fh.readline().strip()
      else:
        fh.seek(last_pos)  # might be grabbing next var
        break
    if self.verbosity>=3: print "       parseTESTline> line ", line
    # Clean up the lines to only have a dot-c name
    justfiles=line.split("=")[1].strip()
    justfiles=justfiles.split("#")[0].strip() # Remove comments
    if len(justfiles.strip())==0: return parseDict
    examplesList=justfiles.split(" ")
    # Now parse the line and put into lists
    srcList=[]; testList=[]; removeList=[]
    for exmpl in examplesList:
      if len(exmpl.strip())==0: continue
      if exmpl.endswith(".PETSc"): 
        srcfile=self.getSourceFileName(exmpl,srcdir)
        parseDict[srcfile]=[] # Create list of tests assocated with src file
        srcList.append(self.nameSpace(srcfile,srcdir))
      elif exmpl.startswith("run"): 
        testList.append(self.nameSpace(exmpl,srcdir))
        parseDict[srcfile].append(exmpl)
      elif exmpl.endswith(".rm"): 
        removeList.append(exmpl) # Can remove later if needed
      else:
        srcList.append(self.nameSpace(exmpl,srcdir))
    if self.verbosity>=3: print "       parseTESTline> ", srcList, testList
    #if "pde_constrained" in srcdir: raise ValueError('Testing')
    parseDict['srcs']=srcList
    parseDict['tsts']=testList
    return parseDict
   
def printMkParseDict(rDict,verbosity):
  """
  This is for debugging
  """
  indent="  "

  fh=open("examplesMkParseInfo.txt","w")
  for sfile in rDict:
    if sfile=='not_tested': continue
    fh.write(sfile+"\n")
    for runex in rDict[sfile]:
      if not runex.startswith("run"):
        fh.write(indent+runex+": "+str(rDict[sfile][runex])+"\n")
    for runex in rDict[sfile]:
      if runex.startswith("run"):
      #if type(rDict[sfile][runex])==types.DictType: 
        fh.write(indent+runex+"\n")
        # Possibly should be recursive even if it is fixed depth
        for rkey in rDict[sfile][runex]:
          if rkey=='test':
            fh.write(indent*2+rkey+":\n")
            for tkey in rDict[sfile][runex][rkey]:
              if type(rDict[sfile][runex][rkey][tkey])==types.DictType: 
                fh.write(indent*3+tkey+":\n")
                for ukey in rDict[sfile][runex][rkey][tkey]:
                  fh.write(indent*4+ukey+": "+str(rDict[sfile][runex][rkey][tkey][ukey]).strip()+"\n")
              else:
                fh.write(indent*3+tkey+": "+str(rDict[sfile][runex][rkey][tkey]).strip()+"\n")
          elif rkey=='script':
            if verbosity>=2:
              fh.write(indent*2+rkey+": "+str(rDict[sfile][runex][rkey]).strip()+"\n")
          else:
            fh.write(indent*2+rkey+": "+str(rDict[sfile][runex][rkey]).strip()+"\n")
      fh.write("\n")

  if rDict.has_key('not_tested'):
    fh.write('not_tested'+"\n")
    for runex in rDict['not_tested']:
      fh.write(indent+runex+"\n")

  fh.close()
  return 

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-r', '--replace', dest='replaceSource',
                      action="store_true", default=False, 
                      help='Replace the source files.  Default is false')
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-m', '--makefile', dest='makefile',
                      help='Function to evaluate while traversing example dirs: genAllRunFiles cleanAllRunFiles', 
                      default='makefile')
    parser.add_option('-v', '--verbosity', dest='verbosity',
                      help='Verbosity of output by level: 1, 2, or 3', 
                      default='0')
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    # Need verbosity to be an integer
    try:
      verbosity=int(options.verbosity)
    except:
      raise Exception("Error: Verbosity must be integer")

    petsc_dir=None
    if options.petsc_dir: petsc_dir=options.petsc_dir
    if petsc_dir is None: petsc_dir=os.path.dirname(os.path.dirname(currentdir))
    if petsc_dir is None:
      petsc_dir = os.environ.get('PETSC_DIR')
      if petsc_dir is None:
        petsc_dir=os.path.dirname(os.path.dirname(currentdir))

    if not options.makefile: 
      print "Use -m to specify makefile"
      return

    pEx=makeParse(petsc_dir,options.replaceSource,verbosity)
    runDict=pEx.parseRunsFromMkFile(options.makefile)
    printMkParseDict(runDict,verbosity)
    pEx.insertInfoIntoSrc(options.makefile,runDict)


if __name__ == "__main__":
        main()
