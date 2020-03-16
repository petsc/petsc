#!/usr/bin/env python
"""
Parse the test file and return a dictionary.

Quick usage::

  lib/petsc/bin/maint/testparse.py -t src/ksp/ksp/examples/tutorials/ex1.c

From the command line, it prints out the dictionary.  
This is meant to be used by other scripts, but it is 
useful to debug individual files.

Example language
----------------

/*TEST
   build:
     requires: moab
   # This is equivalent to test:
   testset:
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full

   testset:
      suffix: 2
      nsize: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full

   testset:
      suffix: 2
      nsize: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full
      test:

TEST*/

"""
from __future__ import print_function

import os, re, glob, types
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import inspect
thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
maintdir=os.path.join(os.path.join(os.path.dirname(thisscriptdir),'bin'),'maint')
sys.path.insert(0,maintdir) 

# These are special keys describing build
buildkeys="requires TODO SKIP depends".split()

acceptedkeys="test nsize requires command suffix args filter filter_output localrunfiles comments TODO SKIP output_file timeoutfactor".split()
appendlist="args requires comments".split()

import re

def getDefaultOutputFileRoot(testname):
  """
  Given testname, give DefaultRoot and DefaultOutputFilename
  e.g., runex1 gives ex1_1, output/ex1_1.out
  """
  defroot=(re.sub("run","",testname) if testname.startswith("run") else testname)
  if not "_" in defroot: defroot=defroot+"_1"
  return defroot

def _stripIndent(block,srcfile,entireBlock=False,fileNums=[]):
  """
  Go through and remove a level of indentation
  Also strip of trailing whitespace
  """
  # The first entry should be test: but it might be indented. 
  ext=os.path.splitext(srcfile)[1]
  stripstr=" "
  if len(fileNums)>0: lineNum=fileNums[0]-1
  for lline in block.split("\n"):
    if len(fileNums)>0: lineNum+=1
    line=lline[1:] if lline.startswith("!") else lline
    if not line.strip(): continue
    if line.strip().startswith('#'): continue
    if entireBlock:
      var=line.split(":")[0].strip()
      if not var in ['test','testset','build']: 
        raise Exception("Formatting error: Cannot find test in file: "+srcfile+" at line: "+str(lineNum)+"\n")
    nspace=len(line)-len(line.lstrip(stripstr))
    newline=line[nspace:]
    break

  # Strip off any indentation for the whole string and any trailing
  # whitespace for convenience
  newTestStr="\n"
  if len(fileNums)>0: lineNum=fileNums[0]-1
  firstPass=True
  for lline in block.split("\n"):
    if len(fileNums)>0: lineNum+=1
    line=lline[1:] if lline.startswith("!") else lline
    if not line.strip(): continue
    if line.strip().startswith('#'): 
      newTestStr+=line+'\n'
    else:
      newline=line[nspace:]
      newTestStr+=newline.rstrip()+"\n"
    # Do some basic indentation checks
    if entireBlock:
      # Don't need to check comment lines
      if line.strip().startswith('#'): continue
      if not newline.startswith(" "):
        var=newline.split(":")[0].strip()
        if not var in ['test','testset','build']: 
          err="Formatting error in file "+srcfile+" at line: " +line+"\n"
          if len(fileNums)>0:
            raise Exception(err+"Check indentation at line number: "+str(lineNum))
          else:
            raise Exception(err)
      else:
        var=line.split(":")[0].strip()
        if var in ['test','testset','build']: 
          subnspace=len(line)-len(line.lstrip(stripstr))
          if firstPass:
            firstsubnspace=subnspace
            firstPass=False
          else:
            if firstsubnspace!=subnspace:
              err="Formatting subtest error in file "+srcfile+" at line: " +line+"\n"
              if len(fileNums)>0:
                raise Exception(err+"Check indentation at line number: "+str(lineNum))
              else:
                raise Exception(err)

  # Allow line continuation character '\'
  return newTestStr.replace('\\\n', ' ')

def parseLoopArgs(varset):
  """
  Given:   String containing loop variables
  Return: tuple containing separate/shared and string of loop vars
  """
  keynm=varset.split("{{")[0].strip().lstrip('-')
  if not keynm.strip(): keynm='nsize'
  lvars=varset.split('{{')[1].split('}')[0]
  suffx=varset.split('{{')[1].split('}')[1]
  ftype='separate' if suffx.startswith('separate') else 'shared' 
  return keynm,lvars,ftype

def _getSeparateTestvars(testDict):
  """
  Given: dictionary that may have 
  Return:  Variables that cause a test split
  """
  vals=None
  sepvars=[]
  # Check nsize
  if 'nsize' in testDict: 
    varset=testDict['nsize']
    if '{{' in varset:
      keynm,lvars,ftype=parseLoopArgs(varset)
      if ftype=='separate': sepvars.append(keynm)

  # Now check args
  if 'args' not in testDict: return sepvars
  for varset in re.split('(^|\W)-(?=[a-zA-Z])',testDict['args']):
    if not varset.strip(): continue
    if '{{' in varset:
      # Assuming only one for loop per var specification
      keynm,lvars,ftype=parseLoopArgs(varset)
      if ftype=='separate': sepvars.append(keynm)

  return sepvars

def _getNewArgs(args):
  """
  Given: String that has args that might have loops in them
  Return:  All of the arguments/values that do not have 
             for 'separate output' in for loops
  """
  newargs=''
  if not args.strip(): return args
  for varset in re.split('(^|\W)-(?=[a-zA-Z])',args):
    if not varset.strip(): continue
    if '{{' in varset and 'separate' in varset: continue
    newargs+="-"+varset.strip()+" "

  return newargs

def _getVarVals(findvar,testDict):
  """
  Given: variable that is either nsize or in args
  Return:  Values to loop over and the other arguments
    Note that we keep the other arguments even if they have
    for loops to enable stepping through all of the for lops
  """
  save_vals=None
  if findvar=='nsize':
    varset=testDict[findvar]
    keynm,save_vals,ftype=parseLoopArgs('nsize '+varset)
  else:
    varlist=[]
    for varset in re.split('-(?=[a-zA-Z])',testDict['args']):
      if not varset.strip(): continue
      if '{{' not in varset: continue
      keyvar,vals,ftype=parseLoopArgs(varset)
      if keyvar==findvar: 
        save_vals=vals

  if not save_vals: raise Exception("Could not find separate_testvar: "+findvar)
  return save_vals

def genTestsSeparateTestvars(intests,indicts,final=False):
  """
  Given: testname, sdict with 'separate_testvars
  Return: testnames,sdicts: List of generated tests
    The tricky part here is the {{ ... }separate output}
    that can be used multiple times
  """
  testnames=[]; sdicts=[]
  for i in range(len(intests)):
    testname=intests[i]; sdict=indicts[i]; i+=1
    separate_testvars=_getSeparateTestvars(sdict)
    if len(separate_testvars)>0:
      sep_dicts=[sdict.copy()]
      if 'args' in sep_dicts[0]:
        sep_dicts[0]['args']=_getNewArgs(sdict['args'])
      sep_testnames=[testname]
      for kvar in separate_testvars:
        kvals=_getVarVals(kvar,sdict)

        # Have to do loop over previous var/val combos as well
        # and accumulate as we go
        val_testnames=[]; val_dicts=[]
        for val in kvals.split():
          gensuffix="_"+kvar+"-"+val.replace(',','__')
          for kvaltestnm in sep_testnames:
            val_testnames.append(kvaltestnm+gensuffix)
          for kv in sep_dicts:
            kvardict=kv.copy()
            # If the last var then we have the final version
            if 'suffix' in sdict:
              kvardict['suffix']+=gensuffix
            else:
              kvardict['suffix']=gensuffix
            if kvar=='nsize':
              kvardict[kvar]=val
            else:
              kvardict['args']+="-"+kvar+" "+val+" "
            val_dicts.append(kvardict)
        sep_testnames=val_testnames
        sep_dicts=val_dicts
      testnames+=sep_testnames
      sdicts+=sep_dicts
    else:
      # These are plain vanilla tests (no subtests, no loops) that 
      # do not have a suffix.  This makes the targets match up with
      # the output file (testname_1.out)
      if final:
          if '_' not in testname: testname+='_1'
      testnames.append(testname)
      sdicts.append(sdict)
  return testnames,sdicts

def genTestsSubtestSuffix(testnames,sdicts):
  """
  Given: testname, sdict with separate_testvars
  Return: testnames,sdicts: List of generated tests
  """
  tnms=[]; sdcts=[]
  for i in range(len(testnames)):
    testname=testnames[i]
    rmsubtests=[]; keepSubtests=False
    if 'subtests' in sdicts[i]:
      for stest in sdicts[i]["subtests"]:
        if 'suffix' in sdicts[i][stest]:
          rmsubtests.append(stest)
          gensuffix="_"+sdicts[i][stest]['suffix']
          newtestnm=testname+gensuffix
          tnms.append(newtestnm)
          newsdict=sdicts[i].copy()
          del newsdict['subtests']
          # Have to hand update
          # Append
          for kup in appendlist:
            if kup in sdicts[i][stest]:
              if kup in sdicts[i]:
                newsdict[kup]=sdicts[i][kup]+" "+sdicts[i][stest][kup]
              else:
                newsdict[kup]=sdicts[i][stest][kup]
          # Promote
          for kup in acceptedkeys:
            if kup in appendlist: continue
            if kup in sdicts[i][stest]: 
              newsdict[kup]=sdicts[i][stest][kup]
          # Cleanup
          for st in sdicts[i]["subtests"]: del newsdict[st]
          sdcts.append(newsdict)
        else:
          keepSubtests=True
    else:
      tnms.append(testnames[i])
      sdcts.append(sdicts[i])
    # If a subtest without a suffix exists, then save it
    if keepSubtests:
      tnms.append(testnames[i])
      newsdict=sdicts[i].copy()
      # Prune the tests to prepare for keeping
      for rmtest in rmsubtests:
        newsdict['subtests'].remove(rmtest)
        del newsdict[rmtest]
      sdcts.append(newsdict)
    i+=1
  return tnms,sdcts

def splitTests(testname,sdict):
  """
  Given: testname and YAML-generated dictionary
  Return: list of names and dictionaries corresponding to each test
          given that the YAML language allows for multiple tests
  """

  # Order: Parent sep_tv, subtests suffix, subtests sep_tv
  testnames,sdicts=genTestsSeparateTestvars([testname],[sdict])
  testnames,sdicts=genTestsSubtestSuffix(testnames,sdicts)
  testnames,sdicts=genTestsSeparateTestvars(testnames,sdicts,final=True)

  # Because I am altering the list, I do this in passes.  Inelegant 

  return testnames, sdicts

def parseTest(testStr,srcfile,verbosity):
  """
  This parses an individual test
  YAML is hierarchial so should use a state machine in the general case,
  but in practice we only support two levels of test:
  """
  basename=os.path.basename(srcfile)
  # Handle the new at the begininng
  bn=re.sub("new_","",basename)
  # This is the default
  testname="run"+os.path.splitext(bn)[0]

  # Tests that have default everything (so empty effectively)
  if len(testStr)==0: return [testname], [{}]

  striptest=_stripIndent(testStr,srcfile)

  # go through and parse
  subtestnum=0
  subdict={}
  comments=[]
  indentlevel=0
  for ln in striptest.split("\n"):
    line=ln.split('#')[0].rstrip()
    if verbosity>2: print(line)
    comment=("" if len(ln.split("#"))>0 else " ".join(ln.split("#")[1:]).strip())
    if comment: comments.append(comment)
    if not line.strip(): continue
    lsplit=line.split(':')
    if len(lsplit)==0: raise Exception("Missing : in line: "+line)
    indentcount=lsplit[0].count(" ")
    var=lsplit[0].strip()
    val=line[line.find(':')+1:].strip()
    if not var in acceptedkeys: raise Exception("Not a defined key: "+var+" from:  "+line)
    # Start by seeing if we are in a subtest
    if line.startswith(" "):
      if var in subdict[subtestname]:
        subdict[subtestname][var]+=" "+val 
      else: 
        subdict[subtestname][var]=val
      if not indentlevel: indentlevel=indentcount
      #if indentlevel!=indentcount: print("Error in indentation:", ln)
    # Determine subtest name and make dict
    elif var=="test":
      subtestname="test"+str(subtestnum)
      subdict[subtestname]={}
      if "subtests" not in subdict: subdict["subtests"]=[]
      subdict["subtests"].append(subtestname)
      subtestnum=subtestnum+1
    # The rest are easy
    else:
      # For convenience, it is sometimes convenient to list twice
      if var in subdict:
        if var in appendlist:
          subdict[var]+=" "+val
        else:
          raise Exception(var+" entered twice: "+line)
      else:
        subdict[var]=val
      if var=="suffix":
        if len(val)>0:
          testname+="_"+val

  if len(comments): subdict['comments']="\n".join(comments).lstrip("\n")

  # A test block can create multiple tests.  This does that logic
  testnames,subdicts=splitTests(testname,subdict)
  return testnames,subdicts

def parseTests(testStr,srcfile,fileNums,verbosity):
  """
  Parse the yaml string describing tests and return 
  a dictionary with the info in the form of:
    testDict[test][subtest]
  This is an inelegant parser as we do not wish to 
  introduce a new dependency by bringing in pyyaml.
  The advantage is that validation can be done as 
  it is parsed (e.g., 'test' is the only top-level node)
  """

  testDict={}

  # The first entry should be test: but it might be indented. 
  newTestStr=_stripIndent(testStr,srcfile,entireBlock=True,fileNums=fileNums)
  if verbosity>2: print(srcfile)

  ## Check and see if we have build requirements
  addToRunRequirements=None
  if "\nbuild:" in newTestStr:
    testDict['build']={}
    # The file info is already here and need to append
    Part1=newTestStr.split("build:")[1]
    fileInfo=re.split("\ntest(?:set)?:",newTestStr)[0]
    for bkey in buildkeys:
      if bkey+":" in fileInfo:
        testDict['build'][bkey]=fileInfo.split(bkey+":")[1].split("\n")[0].strip()
        #if verbosity>1: bkey+": "+testDict['build'][bkey]
      # If a runtime requires are put into build, push them down to all run tests
      # At this point, we are working with strings and not lists
      if 'requires' in testDict['build']:
         if 'datafilespath' in testDict['build']['requires']: 
             newreqs=re.sub('datafilespath','',testDict['build']['requires'])
             testDict['build']['requires']=newreqs.strip()
             addToRunRequirements='datafilespath'


  # Now go through each test.  First elem in split is blank
  for test in re.split("\ntest(?:set)?:",newTestStr)[1:]:
    testnames,subdicts=parseTest(test,srcfile,verbosity)
    for i in range(len(testnames)):
      if testnames[i] in testDict:
        raise RuntimeError("Multiple test names specified: "+testnames[i]+" in file: "+srcfile)
      # Add in build requirements that need to be moved
      if addToRunRequirements:
          if 'requires' in subdicts[i]:
              subdicts[i]['requires']+=addToRunRequirements
          else:
              subdicts[i]['requires']=addToRunRequirements
      testDict[testnames[i]]=subdicts[i]

  return testDict

def parseTestFile(srcfile,verbosity):
  """
  Parse single example files and return dictionary of the form:
    testDict[srcfile][test][subtest]
  """
  debug=False
  basename=os.path.basename(srcfile)
  if basename=='makefile': return {}

  curdir=os.path.realpath(os.path.curdir)
  basedir=os.path.dirname(os.path.realpath(srcfile))
  os.chdir(basedir)

  testDict={}
  sh=open(basename,"r"); fileStr=sh.read(); sh.close()

  ## Start with doing the tests
  #
  fsplit=fileStr.split("/*TEST\n")[1:]
  fstart=len(fileStr.split("/*TEST\n")[0].split("\n"))+1
  # Allow for multiple "/*TEST" blocks even though it really should be
  # one
  srcTests=[]
  for t in fsplit: srcTests.append(t.split("TEST*/")[0])
  testString=" ".join(srcTests)
  flen=len(testString.split("\n"))
  fend=fstart+flen-1
  fileNums=range(fstart,fend)
  testDict[basename]=parseTests(testString,srcfile,fileNums,verbosity)
  # Massage dictionary for build requirements
  if 'build' in testDict[basename]:
    testDict[basename].update(testDict[basename]['build'])
    del testDict[basename]['build']


  os.chdir(curdir)
  return testDict

def parseTestDir(directory,verbosity):
  """
  Parse single example files and return dictionary of the form:
    testDict[srcfile][test][subtest]
  """
  curdir=os.path.realpath(os.path.curdir)
  basedir=os.path.realpath(directory)
  os.chdir(basedir)

  tDict={}
  for test_file in sorted(glob.glob("new_ex*.*")):
    tDict.update(parseTestFile(test_file,verbosity))

  os.chdir(curdir)
  return tDict

def printExParseDict(rDict):
  """
  This is useful for debugging
  """
  indent="   "
  for sfile in rDict:
    print(sfile)
    sortkeys=list(rDict[sfile].keys())
    sortkeys.sort()
    for runex in sortkeys:
      if runex == 'requires':
        print(indent+runex+':'+str(rDict[sfile][runex]))
        continue
      print(indent+runex)
      if type(rDict[sfile][runex])==bytes:
        print(indent*2+rDict[sfile][runex])
      else:
        for var in rDict[sfile][runex]:
          if var.startswith("test"): continue
          print(indent*2+var+": "+str(rDict[sfile][runex][var]))
        if 'subtests' in rDict[sfile][runex]:
          for var in rDict[sfile][runex]['subtests']:
            print(indent*2+var)
            for var2 in rDict[sfile][runex][var]:
              print(indent*3+var2+": "+str(rDict[sfile][runex][var][var2]))
      print("\n")
  return

def main(directory='',test_file='',verbosity=0):

    if directory:
      tDict=parseTestDir(directory,verbosity)
    else:
      tDict=parseTestFile(test_file,verbosity)
    if verbosity>0: printExParseDict(tDict)

    return

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-d', '--directory', dest='directory',
                      default="", help='Directory containing files to parse')
    parser.add_option('-t', '--test_file', dest='test_file',
                      default="", help='Test file, e.g., ex1.c, to parse')
    parser.add_option('-v', '--verbosity', dest='verbosity',
                      help='Verbosity of output by level: 1, 2, or 3', default=0)
    opts, extra_args = parser.parse_args()

    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    if not opts.test_file and not opts.directory:
      print("test file or directory is required")
      parser.print_usage()
      sys.exit()

    # Need verbosity to be an integer
    try:
      verbosity=int(opts.verbosity)
    except:
      raise Exception("Error: Verbosity must be integer")

    main(directory=opts.directory,test_file=opts.test_file,verbosity=verbosity)
