#!/usr/bin/env python
"""
Parse the test file and return a dictionary.

Quick usage::

  bin/maint/testparse.py -t src/ksp/ksp/examples/tutorials/ex1.c

From the command line, it prints out the dictionary.  
This is meant to be used by other scripts, but it is 
useful to debug individual files.



Example language
----------------
/*T
   Concepts:
   requires: moab
T*/



/*TEST
   
   test:
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full
      output_file: output/ex25_1.out
      redirect_file: ex25_1.tmp
   
   test:
      suffix: 2
      nsize: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full

TEST*/

"""

import os, re, glob, types
from distutils.sysconfig import parse_makefile
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import inspect
thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
maintdir=os.path.join(os.path.join(os.path.dirname(thisscriptdir),'bin'),'maint')
sys.path.insert(0,maintdir) 

# These are special keys describing build
buildkeys="requires TODO SKIP depends".split()

def _stripIndent(block,srcfile):
  """
  Go through and remove a level of indentation
  Also strip of trailing whitespace
  """
  # The first entry should be test: but it might be indented. 
  ext=os.path.splitext(srcfile)[1]
  for line in block.split("\n"):
    if not line.strip(): continue
    stripstr=" " if not ext.startswith(".F") else "! "
    nspace=len(line)-len(line.lstrip(stripstr))
    newline=line[nspace:]
    break

  # Strip off any indentation for the whole string and any trailing
  # whitespace for convenience
  newTestStr="\n"
  for line in block.split("\n"):
    if not line.strip(): continue
    newline=line[nspace:]
    newTestStr=newTestStr+newline.rstrip()+"\n"

  return newTestStr

def parseTest(testStr,srcfile):
  """
  This parses an individual test
  YAML is hierarchial so should use a state machine in the general case,
  but in practice we only support to levels of test:
  """
  basename=os.path.basename(srcfile)
  # Handle the new at the begininng
  bn=re.sub("new_","",basename)
  # This is the default
  testname="run"+os.path.splitext(bn)[0]

  # Tests that have default everything (so empty effectively)
  if len(testStr)==0: return testname, {}

  striptest=_stripIndent(testStr,srcfile)

  # go through and parse
  subtestnum=0
  subdict={}
  for line in striptest.split("\n"):
    if not line.strip(): continue
    var=line.split(":")[0].strip()
    val=line.split(":")[1].strip()
    # Start by seeing if we are in a subtest
    if line.startswith(" "):
      subdict[subtestname][var]=val
    # Determine subtest name and make dict
    elif var=="test":
      subtestname="test"+str(subtestnum)
      subdict[subtestname]={}
      if not subdict.has_key("subtests"): subdict["subtests"]=[]
      subdict["subtests"].append(subtestname)
      subtestnum=subtestnum+1
    # The reset are easy
    else:
      subdict[var]=val
      if var=="suffix":
        if len(val)>0:
          testname=testname+"_"+val

  return testname,subdict

def parseTests(testStr,srcfile):
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
  newTestStr=_stripIndent(testStr,srcfile)

  # Now go through each test.  First elem in split is blank
  for test in newTestStr.split("\ntest:\n")[1:]:
    testname,subdict=parseTest(test,srcfile)
    if testDict.has_key(testname):
      print "Multiple test names specified: "+testname+" in file: "+srcfile
    testDict[testname]=subdict
      
  return testDict

def parseTestFile(srcfile):
  """
  Parse single example files and return dictionary of the form:
    testDict[srcfile][test][subtest]
  """
  debug=False
  curdir=os.path.realpath(os.path.curdir)
  basedir=os.path.dirname(os.path.realpath(srcfile))
  basename=os.path.basename(srcfile)
  os.chdir(basedir)

  testDict={}
  sh=open(srcfile,"r"); fileStr=sh.read(); sh.close()

  ## Start with doing the tests
  #
  fsplit=fileStr.split("/*TEST\n")[1:]
  if len(fsplit)==0: 
    if debug: print "No test found in: "+srcfile
    return {}
  # Allow for multiple "/*TEST" blocks even though it really should be
  # on
  srcTests=[]
  for t in fsplit: srcTests.append(t.split("TEST*/")[0])
  testString=" ".join(srcTests)
  if len(testString.strip())==0:
    print "No test found in: "+srcfile
    return {}
  testDict[basename]=parseTests(testString,srcfile)

  ## Check and see if we have build reuqirements
  #
  if "/*T\n" in fileStr or "/*T " in fileStr:
    # The file info is already here and need to append
    Part1=fileStr.split("T*/")[0]
    fileInfo=Part1.split("/*T")[1]
    for bkey in buildkeys:
      if bkey+":" in fileInfo:
        testDict[basename][bkey]=fileInfo.split(bkey+":")[1].split("\n")[0].strip()

  os.chdir(curdir)
  return testDict

def parseTestDir(directory):
  """
  Parse single example files and return dictionary of the form:
    testDict[srcfile][test][subtest]
  """
  curdir=os.path.realpath(os.path.curdir)
  basedir=os.path.realpath(directory)
  os.chdir(basedir)

  tDict={}
  for test_file in glob.glob("new_ex*.*"):
    tDict.update(parseTestFile(test_file))

  os.chdir(curdir)
  return tDict

def printExParseDict(rDict):
  """
  This is useful for debugging
  """
  indent="   "
  for sfile in rDict:
    print "\n\n"+sfile
    for runex in rDict[sfile]:
      print indent+runex
      if type(rDict[sfile][runex])==types.StringType:
        print indent*2+rDict[sfile][runex]
      else:
        for var in rDict[sfile][runex]:
          if var.startswith("test"):
            print indent*2+var
            for var2 in rDict[sfile][runex][var]:
              print indent*3+var2+": "+str(rDict[sfile][runex][var][var2])
          else:
            print indent*2+var+": "+str(rDict[sfile][runex][var])
  return

def main(directory='',test_file='',verbosity=0):

    if directory:
      tDict=parseTestDir(directory)
    else:
      tDict=parseTestFile(test_file)
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
      print "test file or directory is required"
      parser.print_usage()
      sys.exit()

    # Need verbosity to be an integer
    try:
      verbosity=int(opts.verbosity)
    except:
      raise Exception("Error: Verbosity must be integer")

    main(directory=opts.directory,test_file=opts.test_file,verbosity=verbosity)
