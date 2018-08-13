#!/usr/bin/python

from __future__ import print_function
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
    splt=(re.split("_tests",dirpart) if "_tests" in dirpart 
          else re.split("_tutorials",dirpart))
    tdir="tests" if "_tests" in dirpart else "tutorials"
    filedir=os.path.join(self.petsc_dir,"src",
                      splt[0].replace("_","/"),"examples",tdir)

    # Directory names with "-" cause problems, so more work required
    if testnm.count('-') > 2:
       namepart=testnm.split("-")[-1].split("_")[0]
       splitl=(re.split("_tests",testnm) if "_tests" in testnm 
                else re.split("_tutorials",testnm))
       subdir='-'.join(splitl[1].lstrip('_').split('-')[:-1])
       filedir=os.path.join(filedir,subdir)

    # Directory names with underscores cause problems, so more work required
    subdir=""
    if len(splt)>1:
        for psub in splt[1].split("_"):
            subdir+=psub
            if os.path.isdir(os.path.join(filedir,subdir)):
                filedir=os.path.join(filedir,subdir)
                subdir=""
                continue
            subdir+="_"

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
    print(filedir, namepart)
    print("Warning: Cannot find file for "+testname)
    return None

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
     Summarize all of the logfile data by test and then errors
     Want to know if same error occurs on multiple machines
    """
    testDict={}
    testDict['info']={'branch':logDict['branch']}
    testDict['info']['errors']={}
    for logfile in logDict:
      if logfile=='branch': continue
      lfile=logfile.replace("examples_","").replace(".log","")
      for test in logDict[logfile]:
        filename=self.findSrcfile(test)
        if not filename: continue
        fname=os.path.relpath(filename,self.petsc_dir).replace("src/","")
        testname=test.replace("diff-","") if test.startswith("diff-") else test
        error=logDict[logfile][test].strip()
        error=error if test==testname else "Diff errors:\n"+error
        if error=="": error="No error"
        # Organize by filename
        if not fname in testDict: 
          testDict[fname]={}
          testDict[fname]['gitPerson']=str(self.getGitPerson(filename))
        # Now organize by test and errors
        if testname in testDict[fname]:
          if error in testDict[fname][testname]['errors']:
            testDict[fname][testname]['errors'][error].append(lfile)
          else:
            testDict[fname][testname]['errors'][error]=[lfile]
        else:
          #print testname+","+fname+","
          testDict[fname][testname]={}
          # We'll be adding other keys later
          testDict[fname][testname]['errors']={}
          testDict[fname][testname]['errors'][error]=[lfile]
        # Create additional datastructure to sort by errors
        if error in testDict['info']['errors']:
          testDict['info']['errors'][error][testname]=fname
        else:
          testDict['info']['errors'][error]={testname:fname}

    # Place holder for later -- keep log of analysis
    for fname in testDict:
      if fname=='info': continue
      for test in testDict[fname]:
        if test=='gitPerson': continue
        testDict[fname][test]['ndays']=0
        testDict[fname][test]['fdate']='Date'

    return testDict

  def writeSummaryLog(self,testDict,outprefix):
    """
     Just do a simple pretty print
    """
    branch=testDict['info']['branch']
    fh=open(outprefix+branch+".csv","w")
    c=','
    for fname in testDict:
      if fname=='info': continue
      for test in testDict[fname]:
        if test=='gitPerson': continue
        ndays=testDict[fname][test]['ndays']
        fdate=testDict[fname][test]['fdate']
        fh.write(fname+c+test+c+str(ndays)+fdate)

    fh.close()
    return


  def printTestDict(self,testDict):
    """
     Just do a simple pretty print
    """
    indent="  "
    for fname in testDict:
      if fname=='info': continue
      for test in testDict[fname]:
        print("\n ----------------------------------------------------")
        print(test)
        print(indent+testDict[fname][test]['gitPerson'])
        for error in testDict[fname][test]['errors']:
          print("\n ----- ")
          print(2*indent+" ".join(testDict[fname][test]['errors'][error]))
          print(2*indent+error)

    return testDict

  def getLogLink(self,arch):
    """
     For the html for showing the log link
    """
    return '<td class="border"><a href=\"examples_'+arch+'.log\">[log]</a></td>'

  def writeHTML(self,testDict,outprefix):
    """
     Put it into an HTML table
    """
    import htmltemplate

    branch=testDict['info']['branch']
    branchtitle="PETSc Examples ("+branch+")"
    branchhtml=branch+".html"
    htmlfiles=[]
    for hf in "sortByPkg sortByPerson sortByErrors".split(): 
      htmlfiles.append(outprefix+branch+'-'+hf+".html")

    # ----------------------------------------------------------------
    # This is by package and test name
    ofh=open(htmlfiles[0],"w")
    ofh.write(htmltemplate.getHeader(branchtitle+" - Sort By Package/Test"))

    ofh.write('See also:  \n')
    ofh.write('<a href="'+branchhtml+'">'+branchhtml+'</a> \n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[1]+'">'+htmlfiles[1]+'</a>\n\n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[2]+'">'+htmlfiles[2]+'</a><br><br>\n\n')

    pkgs="sys vec mat dm ksp snes ts tao".split()
    ofh.write('Packages:  \n')
    for pkg in pkgs:
      ofh.write('<b><a href="#'+pkg+'">'+pkg+'</a></b>\n')
    ofh.write("</span></center><br>\n")

    ofh.write("<center><table>\n")

    #  sort by example
    allGitPersons={}
    for pkg in pkgs:
      ofh.write('<tr><th class="gray" colspan=4></th></tr>\n')
      ofh.write('<tr id="'+pkg+'"><th colspan=4>'+pkg+' Package</th></tr>\n')
      ofh.write("\n\n")
      ofh.write("<tr><th>Test Name</th><th>Errors</th><th>Arch</th><th>Log</th></tr>\n")
      ofh.write("\n\n")
      for fname in testDict:
        if fname=='info': continue
        if not fname.startswith(pkg): continue  # Perhaps could be more efficient
        gp=testDict[fname]['gitPerson']
        gitPerson=gp.replace("<","&lt").replace("<","&gt")
        gpName=gp.split("<")[0].strip(); gpLastName=gpName.split()[-1]
        if not gpLastName in allGitPersons:
          allGitPersons[gpLastName]=(gp,gpName)
        permlink=htmlfiles[0]+'#'+fname
        plhtml='<a id="'+permlink+'" href="'+permlink+'"> (permlink)</a>'
        ofh.write('<tr><th colspan="4">'+fname+" &nbsp&nbsp ("+gitPerson+') '+plhtml+'</th></tr>\n\n')
        for test in testDict[fname]:
          if test=='gitPerson': continue
          i=0
          for error in testDict[fname][test]['errors']:
            ofh.write('<!-- New row  -->\n')

            teststr='<td class="border">'+test+'</td>' if i==0 else '<td></td>'
            i+=1
            arches=testDict[fname][test]['errors'][error]
            rsnum=len(arches)
            # Smaller arch string looks nice
            archstr=[arch.replace(branch+'_','').replace("arch-",'') for arch in arches]

            comstr="<pre>"+error+"</pre>" if error else "No error"
            comstr='<td class="border" rowspan="'+str(rsnum)+'">'+comstr+'</td>'

            logstr=self.getLogLink(arches[0]) 
            ofh.write('<tr>'+teststr+comstr+'<td class="border">'+archstr[0]+'</td>'+logstr+'</tr>\n')
            # Log files are then hanging
            if len(arches)>1: 
              for j in range(1,rsnum):
                logstr=self.getLogLink(arches[j]) 
                ofh.write("<tr><td></td>                     <td>"+archstr[j]+"</td> "+logstr+"</tr>\n")
            ofh.write("\n\n")

    ofh.write("</table>\n<br>\n")
    ofh.close()

    # ----------------------------------------------------------------
    # This is by person
    ofh=open(htmlfiles[1],"w")
    ofh.write(htmltemplate.getHeader(branchtitle+" - Sort By Person")) 
    ofh.write("\n\n")

    ofh.write('See also:  \n')
    ofh.write('<a href="'+branchhtml+'">'+branchhtml+'</a> \n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[0]+'">'+htmlfiles[0]+'</a> \n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[2]+'">'+htmlfiles[2]+'</a><br><br>\n')

    happyShinyPeople=allGitPersons.keys()
    happyShinyPeople.sort()  # List alphabetically by last name

    ofh.write('People:  \n')
    for person in happyShinyPeople:
      (gpFullName,gpName)=allGitPersons[person]
      ofh.write('<a href="#'+person+'">'+gpName+'  </a> &nbsp\n')
    ofh.write("</span></center><br>\n")


    #  sort by person
    ofh.write("<center><table>\n")
    for person in happyShinyPeople:
      (gpFullName,gpName)=allGitPersons[person]
      ofh.write('<tr><th class="gray" colspan=4></th></tr>\n')
      ofh.write('<tr id="'+person+'"><th colspan=4>'+gpFullName+'</th></tr>\n')
      ofh.write("\n\n")
      ofh.write("<tr><th>Test Name</th><th>Error</th><th></th><th>Arch</th></tr>\n")
      ofh.write("\n\n")
      for fname in testDict:
        if fname=='info': continue
        gitPerson=testDict[fname]['gitPerson'].replace("<","&lt").replace("<","&gt")
        if not gitPerson.startswith(gpName): continue
        permlink=htmlfiles[0]+'#'+fname
        plhtml=' <a id="'+permlink+'" href="'+permlink+'"> (permlink)</a>'
        ofh.write('<tr><th colspan="4">'+fname+plhtml+'</th></tr>\n\n')
        for test in testDict[fname]:
          if test=='gitPerson': continue
          i=0
          for error in testDict[fname][test]['errors']:
            ofh.write('<!-- New row  -->\n')

            teststr='<td class="border">'+test+'</td>' if i==0 else '<td></td>'
            i+=1
            arches=testDict[fname][test]['errors'][error]
            rsnum=len(arches)
            archstr=[arch.replace(branch+'_','').replace("arch-",'') for arch in arches]
            comstr="<pre>"+error+"</pre>" if error else "No error"
            comstr='<td class="border" rowspan="'+str(rsnum)+'">'+comstr+'</td>'
            logstr=self.getLogLink(arches[0]) 

            ofh.write('<tr>'+teststr+comstr+'<td class="border">'+archstr[0]+'</td>'+logstr+'</tr>\n')
            if len(arches)>1: 
              for j in range(1,rsnum):
                logstr=self.getLogLink(arches[j]) 
                ofh.write("<tr> <td></td>           <td>"+archstr[j]+"</td> "+logstr+"</tr>\n")
            ofh.write("\n\n")

    ofh.write("</table>")
    ofh.close()

    # ----------------------------------------------------------------
    # This is by errors
    ofh=open(htmlfiles[2],"w")
    ofh.write(htmltemplate.getHeader(branchtitle+" - Sort By Errors")) 
    ofh.write("\n\n")

    ofh.write('See also:  \n')
    ofh.write('<a href="'+branchhtml+'">'+branchhtml+'</a> \n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[0]+'">'+htmlfiles[0]+'</a> \n')
    ofh.write('&nbsp \n')
    ofh.write('<a href="'+htmlfiles[1]+'">'+htmlfiles[1]+'</a><br><br>\n')
    ofh.write("</span></center><br>\n")


    #  sort by error
    ofh.write("<center><table>\n")
    ofh.write("<tr><th>Error</th><th>Test Name</th><th></th><th>Arch</th></tr>\n")
    for error in testDict['info']['errors']:
      ofh.write('<tr><th class="gray" colspan=4></th></tr>\n')
      ofh.write("\n\n")
      i=0
      comstr="<pre>"+error+"</pre>" if error else "No error"
      comstr='<td class="border">'+comstr+'</td>'
      for test in testDict['info']['errors'][error]:
        fname=testDict['info']['errors'][error][test]

        permlink=htmlfiles[0]+'#'+fname
        plhtml=' <a id="'+permlink+'" href="'+permlink+'"> (permlink)</a>'

        arches=testDict[fname][test]['errors'][error]
        rsnum=len(arches)
        if i>0: comstr='<td></td>'
        i+=1
        teststr='<td class="border" rowspan="'+str(rsnum)+'">'+test+'</td>'
        archstr=[arch.replace(branch+'_','').replace("arch-",'') for arch in arches]
        logstr=self.getLogLink(arches[0]) 
        ofh.write('<tr>'+comstr+teststr+'<td class="border">'+archstr[0]+'</td>'+logstr+'</tr>\n')
        if len(arches)>1: 
          for j in range(1,rsnum):
            logstr=self.getLogLink(arches[j]) 
            ofh.write("<tr> <td></td>           <td>"+archstr[j]+"</td> "+logstr+"</tr>\n")
        ofh.write("\n\n")

    ofh.write("</table>")
    ofh.close()



    return

  def doLogFiles(self,branch):
    """
     Go through all of the log files and call the parser for each one
     Get a simple dictionary for each logfile -- later we process
     to make nice printouts
    """
    logDict={'branch':branch}
    startdir=os.path.abspath(os.path.curdir)
    os.chdir(self.logdir)
    for logfile in glob.glob('examples_'+branch+'_*.log'):
      if logfile.startswith('examples_full'): continue
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
          errors=''
          while 1:
            newline=infile.readline()
            if newline.startswith('#'):
              errors=newline.lstrip('#').strip()
              last_pos=infile.tell()
            else:
              break
          infile.seek(last_pos)  # Go back b/c we grabbed another test
          lDict[test]=errors
    if printDict:
      for lkey in lDict:
        print(lkey)
        print("  "+lDict[lkey].replace("\n","\n  "))
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
                      default='examples_summary-')
    parser.add_option('-v', '--verbosity', dest='verbosity',
                      help='Verbosity of output by level: 1, 2, or 3', 
                      default='0')
    parser.add_option('-b', '--branch', dest='branch',
                      help='Comma delimitted list of branches to parse files of form: examples_<branch>_<arch>.log', 
                      default='master,next')
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
      print("Use -l to specify makefile")
      return

    logP=logParse(petsc_dir,options.logdir,verbosity)
    if options.logfile: 
       l=logP.parseLogFile(options.logfile,printDict=True)
       return
    else:
      for b in options.branch.split(','):
        logDict=logP.doLogFiles(b)
        testDict=logP.getTestDict(logDict)
        if verbosity>2:
          logP.printTestDict(testDict)
        logP.writeHTML(testDict,options.outfile)
        logP.writeSummaryLog(testDict,options.outfile)

    return

if __name__ == "__main__":
        main()
