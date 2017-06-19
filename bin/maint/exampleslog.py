#!/usr/bin/python

import fnmatch
import glob
import optparse
import os
import re
import sys
import time
import types

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 
#import runhtml

class logParse(object):
  def __init__(self,petsc_dir,logdir,verbosity):

    self.petsc_dir=petsc_dir
    self.verbosity=verbosity
    self.logdir=logdir
    return

  def findSrcfile(self,testname):
    """
    Given a testname of the form runex10_9, try to figure out the source file
    """
    testnm=testname.replace("diff-","")
    dirpart=testnm.split("-")[0]
    namepart=testnm.split("-")[1].split("_")[0]
    # First figure out full directory to source file
    filedir=re.sub("tests","examples_tests",dirpart)
    filedir=re.sub("tutorials","examples_tutorials",filedir)
    filedir=os.path.join(filedir.replace("_","/"))
    filedir=os.path.join(self.petsc_dir,"src",filedir)

    # See what files exists with guessing the extension
    base=namepart
    if base.endswith("f") or base.endswith("f90"):
      for ext in ['F','F90']:
        guess=os.path.join(filedir,base+"."+ext)
        if os.path.exists(guess): return guess
    else:
      for ext in ['c','cxx']:
        guess=os.path.join(filedir,base+"."+ext)
        if os.path.exists(guess): return guess
    # Sometimes the underscore is in the executable (ts-tutorials)
    base=namepart[3:]  # I can't remember how this works
    if base.endswith("f") or base.endswith("f90"):
      for ext in ['F','F90']:
        guess=os.path.join(filedir,base+"."+ext)
        if os.path.exists(guess): return guess
    else:
      for ext in ['c','cxx']:
        guess=os.path.join(filedir,base+"."+ext)
        if os.path.exists(guess): return guess
    raise Exception("Error: Cannot find file for "+testname)
    return 

  def getGitPerson(self,fullFileName):
    """
    Given a testname, find the file in petsc_dir and find out who did the last
    commit
    """
    git_authorname_cmd='git log -1 --pretty=format:"%an <%ae>" '+fullFileName
    try:
      #git_blame_cmd = 'git blame -w -M --line-porcelain --show-email -L '+' -L '.join(pairs)+' '+key[0]+' -- '+key[1]
      fh=os.popen(git_authorname_cmd)
      output=fh.read(); fh.close
    except:
      raise Exception("Error running: "+git_authorname_cmd)
    return output

  def getTestDict(self,logDict):
    """
     Summarize all of the logfile data by test and then comments
     Want to know if same error occurs on multiple machines
    """
    testDict={}
    for logfile in logDict:
      lfile=logfile.replace("examples_","").replace(".log","")
      for test in logDict[logfile]:
        filename=self.findSrcfile(test)
        fname=os.path.relpath(filename,self.petsc_dir).replace("src/","")
        testname=test.replace("diff-","") if test.startswith("diff-") else test
        comment=logDict[logfile][test].strip()
        comment=comment if test==testname else "Diff errors:\n"+comment
        # Organize by filename
        if not fname in testDict: 
          testDict[fname]={}
          testDict[fname]['gitPerson']=str(self.getGitPerson(filename))
        # Now organize by test and comments
        if testname in testDict[fname]:
          if comment in testDict[fname][testname]['comments']:
            testDict[fname][testname]['comments'][comment].append(lfile)
          else:
            testDict[fname][testname]['comments'][comment]=[lfile]
        else:
          testDict[fname][testname]={}
          # We'll be adding other keys later
          testDict[fname][testname]['comments']={}
          testDict[fname][testname]['comments'][comment]=[lfile]

    return testDict

  def printTestDict(self,testDict):
    """
     Just do a simple pretty print
    """
    indent="  "
    for fname in testDict:
      for test in testDict[fname]:
        print "\n ----------------------------------------------------"
        print test
        print indent+testDict[fname][test]['gitPerson']
        for comment in testDict[fname][test]['comments']:
          print "\n ----- "
          print 2*indent+" ".join(testDict[fname][test]['comments'][comment])
          print 2*indent+comment

    return testDict

  def writeHTML(self,testDict,outprefix):
    """
     Put it into an HTML table
    """
    import htmltemplate

    # ----------------------------------------------------------------
    # This is by package and test name
    ofh=open(outprefix+"sortByPkg.html","w")
    ofh.write(htmltemplate.getHeader("PETSc Examples - Sort By Name"))
    ofh.write("\n\n")

    pkgs="sys vec mat dm ksp snes ts tao".split()

    ofh.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">PETSc Example Summary - Sorted by Package/Testname</span><br />Last update: " + time.strftime("%c") + "</center>\n\n")
    ofh.write("</span></center><br><br>\n\n")
    ofh.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">\n")
    ofh.write('Packages:  \n');
    for pkg in pkgs:
      ofh.write('<b><a href="#'+pkg+'">'+pkg+'</a></b>\n');
    ofh.write("</span></center><br><br>\n")

    ofh.write("<center><table>\n");

    #  sort by example
    allGitPersons={}
    for pkg in pkgs:
      ofh.write('<tr><th class="gray" colspan=4></th></tr>\n');
      ofh.write('<tr><a name="'+pkg+'"></a> <th colspan=4>'+pkg+' Package</th></tr>\n');
      ofh.write("\n\n")
      ofh.write("<tr><th>Test Name</th><th>Error</th><th>Arch</th><th>Log</th></tr>\n");
      ofh.write("\n\n")
      for fname in testDict:
        if not fname.startswith(pkg): continue  # Perhaps could be more efficient
        gp=testDict[fname]['gitPerson']
        gitPerson=gp.replace("<","&lt").replace("<","&gt")
        gpName=gp.split("<")[0].strip(); gpLastName=gpName.split()[-1]
        if not gpLastName in allGitPersons:
          allGitPersons[gpLastName]=(gp,gpName)
        ofh.write('<tr><th colspan="4">'+fname+" &nbsp&nbsp ("+gitPerson+')</th></tr>\n\n')
        for test in testDict[fname]:
          if test=='gitPerson': continue
          i=0
          for comment in testDict[fname][test]['comments']:
            teststr=test if i==0 else ""
            arches=testDict[fname][test]['comments'][comment]
            rsnum=len(arches)
            # Smaller arch string looks nice
            archstr=[]
            for arch in arches:
              archstr.append(arch.replace("next_",'').replace("master_",'').replace("arch-",''))
            ofh.write('<!-- New row  -->\n')
            if len(comment)==0: 
              comstr="No comment"
            else:
              comstr="<pre>"+comment+"</pre>"
              #comstr=comment
            logstr='<td><a href=\"examples_'+arches[0]+'.log\">[log]</a></td>'
            ofh.write('<tr><td>'+teststr+'</td><td rowspan="'+str(rsnum)+'">'+comstr+'</td><td>'+archstr[0]+'</td>'+logstr+'</tr>\n')
            if len(arches)>1: 
              for i in range(1,rsnum):
                logstr='<td><a href=\"examples_'+arches[i]+'.log\">[log]</a></td>'
                ofh.write("<tr><td></td>                     <td>"+archstr[i]+"</td> "+logstr+"</tr>\n")
            ofh.write("\n\n")

    ofh.write("</table>\n<br>\n")
    ofh.close()

    # ----------------------------------------------------------------
    # This is by person
    ofh=open(outprefix+"sortByPerson.html","w")
    ofh.write(htmltemplate.getHeader("PETSc Examples - Sort By Person"))
    ofh.write("\n\n")

    ofh.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">PETSc Example Summary - Sorted by Person</span><br />Last update: " + time.strftime("%c") + "</center>\n\n")
    ofh.write("</span></center><br><br>\n\n")
    ofh.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">\n")

    happyShinyPeople=allGitPersons.keys()
    happyShinyPeople.sort()  # List alphabetically by last name

    ofh.write('People:  \n');
    for person in happyShinyPeople:
      (gpFullName,gpName)=allGitPersons[person]
      ofh.write('<a href="#'+person+'">'+gpName+'  </a> &nbsp\n');
    ofh.write("</span></center><br><br>\n")


    #  sort by example
    #   3 rows minimum.  More than 3 if len(arches)>3
    ofh.write("<center><table>\n");
    for person in happyShinyPeople:
      (gpFullName,gpName)=allGitPersons[person]
      ofh.write('<tr><th class="gray" colspan=4></th></tr>\n');
      ofh.write('<tr><a name="'+person+'"></a> <th colspan=4>'+gpFullName+'</th></tr>\n');
      ofh.write("\n\n")
      ofh.write("<tr><th>Test Name</th><th>Error</th><th></th><th>Arch</th></tr>\n");
      ofh.write("\n\n")
      for fname in testDict:
        gitPerson=testDict[fname]['gitPerson'].replace("<","&lt").replace("<","&gt")
        if not gitPerson.startswith(gpName): continue
        ofh.write('<tr><th colspan="4">'+fname+'</th></tr>\n\n')
        for test in testDict[fname]:
          if test=='gitPerson': continue
          i=0
          for comment in testDict[fname][test]['comments']:
            teststr=test if i==0 else ""
            arches=testDict[fname][test]['comments'][comment]
            rsnum=len(arches)
            # Smaller arch string looks nice
            archstr=[]
            for arch in arches:
              archstr.append(arch.replace("next_",'').replace("master_",'').replace("arch-",''))
            ofh.write('<!-- New row  -->\n')
            if len(comment)==0: 
              comstr="No comment"
            else:
              comstr="<pre>"+comment+"</pre>"
              #comstr=comment
            logstr='<td><a href=\"examples_'+arches[0]+'.log\">[log]</a></td>'
            ofh.write('<tr><td>'+teststr+'</td><td rowspan="'+str(rsnum)+'">'+comstr+'</td><td>'+archstr[0]+'</td>'+logstr+'</tr>\n')
            if len(arches)>1: 
              for i in range(1,rsnum):
                logstr='<td><a href=\"examples_'+arches[i]+'.log\">[log]</a></td>'
                ofh.write("<tr> <td></td>           <td>"+archstr[i]+"</td> "+logstr+"</tr>\n")
            ofh.write("\n\n")

    ofh.write("</table>")
    ofh.close()
    return testDict

  def doLogFiles(self):
    """
     Go through all of the log files and call the parser for each one
     Get a simple dictionary for each logfile -- later we process
     to make nice printouts
    """
    logDict={}
    startdir=os.path.abspath(os.path.curdir)
    os.chdir(self.logdir)
    for logfile in glob.glob('examples*.log'):
      if logfile=='examples_full_next.log': continue
      logDict[logfile]=self.parseLogFile(logfile)
    os.chdir(startdir)
    return logDict


  def parseLogFile(self,logfile,printDict=False):
    """
     Do the actual parsing of the file and return
     a dictionary with all of the failed tests along with why they failed
    """
    lDict={}
    with open(logfile,"r") as infile:
      while 1:
        line=infile.readline()
        if not line: break
        if line.startswith("not ok "): 
          last_pos=infile.tell()
          test=line.replace("not ok ","").strip()
          comments=''
          while 1:
            newline=infile.readline()
            if newline.startswith('#'):
              comments=newline.lstrip('#').strip()
              last_pos=infile.tell()
            else:
              break
          infile.seek(last_pos)  # Go back b/c we grabbed another test
          lDict[test]=comments
    if printDict:
      for lkey in lDict:
        print lkey
        print "  "+lDict[lkey].replace("\n","\n  ")
    return lDict



def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-f', '--logfile', dest='logfile',
                      help='Parse a single file and print out dictionary for debugging')
    parser.add_option('-l', '--logdir', dest='logdir',
                      help='Directory where to find the log files')
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='where to find the git repo',
                      default='')
    parser.add_option('-o', '--outfile', dest='outfile',
                      help='The output file prefix where the HTML code will be written to', 
                      default='example_summary-')
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

    if not options.logdir: 
      print "Use -l to specify makefile"
      return

    logP=logParse(petsc_dir,options.logdir,verbosity)
    if options.logfile: 
       l=logP.parseLogFile(options.logfile,printDict=True)
       return
    else:
      logDict=logP.doLogFiles()
      testDict=logP.getTestDict(logDict)
      if verbosity>2:
        logP.printTestDict(testDict)
      logP.writeHTML(testDict,options.outfile)

    return

if __name__ == "__main__":
        main()
