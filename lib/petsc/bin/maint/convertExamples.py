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
from examplesWalker import *
from examplesMkParse import *

"""
Tools for converting PETSc examples to new format
Based on examplesWalker.py

Quick start
-----------

  lib/petsc/bin/maint/convertExamples.py -f genAllRunFiles src
     - Generate scripts from the makefiles
     - Try to abstract the scripts and put the metadata into the source code

  lib/petsc/bin/maint/convertExamples.py -f cleanAllRunFiles src

"""

class convertExamples(PETScExamples):
  def __init__(self,petsc_dir,replaceSource,verbosity):
    super(convertExamples, self).__init__(petsc_dir,replaceSource,verbosity)
    self.writeScripts=True
    #self.scriptsSubdir=""
    self.scriptsSubdir="from_makefile"
    return

  def cleanAllRunFiles_summarize(self,dataDict):
    """
    Required routine
    """
    return

  def cleanAllRunFiles(self,root,dirs,files,dataDict):
    """
    Cleanup from genAllRunFiles
    """
    if self.writeScripts:
      globstr=root+"/new_*"
      if self.scriptsSubdir:  globstr=root+"/"+self.scriptsSubdir+"/run*"
      for runfile in glob.glob(globstr): os.remove(runfile)
    for newfile in glob.glob(root+"/new_*"): os.remove(newfile)
    for tstfile in glob.glob(root+"/TEST*.sh"): os.remove(tstfile)
    for newfile in glob.glob(root+"/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*/*.tmp*"): os.remove(newfile)
    for newfile in glob.glob(root+"/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*/*/trashz"): os.remove(newfile)
    for newfile in glob.glob(root+"/*.mod"): os.remove(newfile)
    for newfile in glob.glob(root+"/examplesMkParseInfo.txt"): os.remove(newfile)
    return

  def genAllRunFiles_summarize(self,dataDict):
    """
    Summarize the results.
    """
    # This needs to be upgraded as well
    return
    indent="  "
    fhname="GenAllRunFiles_summarize.txt"
    fh=open(fhname,"w")
    print "See ", fhname
    for mkfile in dataDict:
      fh.write(mkfile+"\n")
      for runex in dataDict[mkfile]:
        if runex=='nonUsedTests': continue
        fh.write(indent+runex+"\n")
        for key in self.getOrderedKeys(dataDict[mkfile][runex]):
          if not dataDict[mkfile][runex].has_key(key): continue
          s=dataDict[mkfile][runex][key]
          if isinstance(s, basestring):
            line=indent*2+key+": "+s
          elif key=='nsize' or key=='abstracted':
            line=indent*2+key+": "+str(s)
          else:
            line=indent*2+key
          fh.write(line+"\n")
        fh.write("\n")
      line=" ".join(dataDict[mkfile]['nonUsedTests'])
      if len(line)>0:
        fh.write(indent+"Could not insert into source from "+mkfile+": "+line+"\n")
      fh.write("\n")
    return

  def genAllRunFiles(self,root,dirs,files,dataDict):
    """
     For all of the TESTEXAMPLES* find the run* targets, convert to
     script, abstract if possible, and create new_ex* source files that
     have the abstracted info.  

     Because the generation of the new source files requires 
    """
    # Because of coding, clean up the directory before parsing makefile
    self.cleanAllRunFiles(root,dirs,files,{})

    debug=False
    insertIntoSrc=True

    # Information comes form makefile
    fullmake=os.path.join(root,"makefile")

    # Go through the makefile, and for each run* target: 
    #     extract, abstract, insert
    dataDict[fullmake]={}
    dataDict[fullmake]['nonUsedTests']=[]
    i=0
    varVal={}
    if self.verbosity>=1: print fullmake

    # This gets all of the run* targets in makefile.  
    # Can be tested independently in examplesMkParse.py
    runDict=self.parseRunsFromMkFile(fullmake)
    self.insertInfoIntoSrc(fullmake,runDict)

    return

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-r', '--replace', dest='replaceSource',
                      action="store_true", default=False, 
                      help='Replace the source files.  Default is false')
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='PETSC_DIR (default is from environment)',
                      default='')
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-f', '--functioneval', dest='functioneval',
                      help='Function to run while traversing example dirs: genAllRunFiles cleanAllRunFiles', 
                      default='genAllRunFiles')
    parser.add_option('-v', '--verbosity', dest='verbosity',
                      help='Verbosity of output by level: 1, 2, or 3', 
                      default='0')
    options, args = parser.parse_args()

    # Need verbosity to be an integer
    try:
      verbosity=int(options.verbosity)
    except:
      raise Exception("Error: Verbosity must be integer")

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    petsc_dir=None
    if options.petsc_dir: petsc_dir=options.petsc_dir
    if petsc_dir is None: petsc_dir=os.path.dirname(os.path.dirname(currentdir))
    if petsc_dir is None:
      petsc_dir = os.environ.get('PETSC_DIR')
      if petsc_dir is None:
        petsc_dir=os.path.dirname(os.path.dirname(currentdir))

    if not options.startdir: options.startdir=os.path.join(petsc_dir,'src')

    pEx=convertExamples(petsc_dir,options.replaceSource,verbosity)
    if not options.functioneval=='':
      pEx.walktree(options.startdir,action=options.functioneval)
    else:
      pEx.walktree(options.startdir)

if __name__ == "__main__":
        main()
