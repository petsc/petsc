#!/usr/bin/env python
from __future__ import print_function
import glob, os, re, stat
import optparse
import inspect

"""
Quick script for parsing the output of the test system and summarizing the results.
"""

def inInstallDir():
  """
  When petsc is installed then this file in installed in:
       <PREFIX>/share/petsc/examples/config/gmakegentest.py
  otherwise the path is:
       <PETSC_DIR>/config/gmakegentest.py
  We use this difference to determine if we are in installdir
  """
  thisscriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  dirlist=thisscriptdir.split(os.path.sep)
  if len(dirlist)>4:
    lastfour=os.path.sep.join(dirlist[len(dirlist)-4:])
    if lastfour==os.path.join('share','petsc','examples','config'):
      return True
    else:
      return False
  else:
    return False

def summarize_results(directory,make,ntime,etime):
  ''' Loop over all of the results files and summarize the results'''
  startdir = os.getcwd()
  try:
    os.chdir(directory)
  except OSError:
    print('# No tests run')
    return
  summary={'total':0,'success':0,'failed':0,'failures':[],'todo':0,'skip':0,
           'time':0, 'cputime':0}
  timesummary={}
  cputimesummary={}
  timelist=[]
  for cfile in glob.glob('*.counts'):
    with open(cfile, 'r') as f:
      for line in f:
        l = line.split()
        if l[0] == 'failures':
           if len(l)>1:
             summary[l[0]] += l[1:]
        elif l[0] == 'time':
           if len(l)==1: continue
           summary[l[0]] += float(l[1])
           summary['cputime'] += float(l[2])
           timesummary[cfile]=float(l[1])
           cputimesummary[cfile]=float(l[2])
           timelist.append(float(l[1]))
        elif l[0] not in summary:
           continue
        else:
           summary[l[0]] += int(l[1])

  failstr=' '.join(summary['failures'])
  print("\n# -------------")
  print("#   Summary    ")
  print("# -------------")
  if failstr.strip(): print("# FAILED " + failstr)

  for t in "success failed todo skip".split():
    percent=summary[t]/float(summary['total'])*100
    print("# %s %d/%d tests (%3.1f%%)" % (t, summary[t], summary['total'], percent))
  print("#")
  if etime:
    print("# Wall clock time for tests: %s sec"% etime)
  print("# Approximate CPU time (not incl. build time): %s sec"% summary['cputime'])

  if failstr.strip():
      fail_targets=(
          re.sub('cmd-','',
          re.sub('diff-','',failstr+' '))
          )
      # Strip off characters from subtests
      fail_list=[]
      for failure in fail_targets.split():
         fail_list.append(failure.split('+')[0])
      fail_list=list(set(fail_list))
      fail_targets=' '.join(fail_list)

      # create simple little script
      sfile=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'echofailures.sh')
      with open(sfile,'w') as f:
          f.write('echo '+fail_targets.strip())
      st = os.stat(sfile)
      os.chmod(sfile, st.st_mode | stat.S_IEXEC)

      #Make the message nice
      makefile="gmakefile.test" if inInstallDir() else "gmakefile"

      print("#\n# To rerun failed tests: ")
      print("#     "+make+" -f "+makefile+" test test-fail=1")

  if ntime>0:
      print("#\n# Timing summary (actual test time / total CPU time): ")
      timelist=list(set(timelist))
      timelist.sort(reverse=True)
      nlim=(ntime if ntime<len(timelist) else len(timelist))
      # Do a double loop to sort in order
      for timelimit in timelist[0:nlim]:
        for cf in timesummary:
          if timesummary[cf] == timelimit:
            print("#   %s: %.2f sec / %.2f sec" % (re.sub('.counts','',cf), timesummary[cf], cputimesummary[cf]))
  os.chdir(startdir)
  return
  
def get_test_data(directory):
    """
    Create dictionary structure with test data
    """
    startdir= os.getcwd()
    try:
        os.chdir(directory)
    except OSError:
        print('# No tests run')
        return
    # loop over *.counts files for all the problems tested in the test suite
    testdata = {}
    for cfile in glob.glob('*.counts'):
        # first we get rid of the .counts extension, then we split the name in two
        # to recover the problem name and the package it belongs to
        fname = cfile.split('.')[0]
        testname = fname.split('-')
        probname = ''
        for i in range(1,len(testname)):
            probname += testname[i]
        # we split the package into its subcomponents of PETSc module (e.g.: snes) 
        # and test type (e.g.: tutorial)
        testname_list = testname[0].split('_')
        pkgname = testname_list[0]
        testtype = testname_list[-1]
        # in order to correct assemble the folder path for problem outputs, we 
        # iterate over any possible subpackage names and test suffixes
        testname_short = testname_list[:-1]
        prob_subdir = os.path.join('', *testname_short)
        probfolder = 'run%s'%probname
        probdir = os.path.join('..', prob_subdir, 'examples', testtype, probfolder)
        if not os.path.exists(probdir):
            probfolder = probfolder.split('_')[0]
            probdir = os.path.join('..', prob_subdir, 'examples', testtype, probfolder)
        probfullpath=os.path.normpath(os.path.join(directory,probdir))
        # assemble the final full folder path for problem outputs and read the files
        try:
            with open('%s/diff-%s.out'%(probdir, probfolder),'r') as probdiff:
                difflines = probdiff.readlines()
        except IOError:
            difflines = []
        try:
            with open('%s/%s.err'%(probdir, probfolder),'r') as probstderr:
                stderrlines = probstderr.readlines()
        except IOError:
            stderrlines = []
        try:
            with open('%s/%s.tmp'%(probdir, probname), 'r') as probstdout:
                stdoutlines = probstdout.readlines()
        except IOError:
            stdoutlines = []
        # join the package, subpackage and problem type names into a "class"
        classname = pkgname
        for item in testname_list[1:]:
            classname += '.%s'%item
        # if this is the first time we see this package, initialize its dict
        if pkgname not in testdata.keys():
            testdata[pkgname] = {
                'total':0,
                'success':0,
                'failed':0,
                'errors':0,
                'todo':0,
                'skip':0,
                'time':0,
                'problems':{}
            }
        # add the dict for the problem into the dict for the package
        testdata[pkgname]['problems'][probname] = {
            'classname':classname,
            'time':0,
            'failed':False,
            'skipped':False,
            'diff':difflines,
            'stdout':stdoutlines,
            'stderr':stderrlines,
            'probdir':probfullpath,
            'fullname':fname
        }
        # process the *.counts file and increment problem status trackers
        if len(testdata[pkgname]['problems'][probname]['stderr'])>0:
            testdata[pkgname]['errors'] += 1
        with open(cfile, 'r') as f:
            for line in f:
                l = line.split()
                if l[0] == 'time':
                    if len(l)==1: continue
                    testdata[pkgname]['problems'][probname][l[0]] = float(l[1])
                    testdata[pkgname][l[0]] += float(l[1])
                elif l[0] in testdata[pkgname].keys():
                    num_int=int(l[1])
                    testdata[pkgname][l[0]] += num_int
                    if l[0] in ['failed','skip'] and num_int:
                        testdata[pkgname]['problems'][probname][l[0]] = True
                else:
                    continue
    os.chdir(startdir)  # Keep function in good state
    return testdata

def show_fail(testdata):
    """ Show the failures and commands to run them
    """
    for pkg in testdata.keys():
        testsuite = testdata[pkg]
        for prob in testsuite['problems'].keys():
            p = testsuite['problems'][prob]
            cdbase='cd '+p['probdir']+' && '
            if p['skipped']:
                # if we got here, the TAP output shows a skipped test
                pass
            elif len(p['stderr'])>0:
                # if we got here, the test crashed with an error
                # we show the stderr output under <error>
                shbase=os.path.join(p['probdir'], p['fullname'])
                shfile=shbase+".sh"
                if not os.path.exists(shfile):
                    shfile=glob.glob(shbase+"*")[0]
                with open(shfile, 'r') as sh:
                    cmd = sh.read()
                print(p['fullname']+': '+cdbase+cmd.split('>')[0])
            elif len(p['diff'])>0:
                # if we got here, the test output did not match the stored output file
                # we show the diff between new output and old output under <failure>
                shbase=os.path.join(p['probdir'], 'diff-'+p['fullname'])
                shfile=shbase+".sh"
                if not os.path.exists(shfile):
                    shfile=glob.glob(shbase+"*")[0]
                with open(shfile, 'r') as sh:
                    cmd = sh.read()
                print(p['fullname']+': '+cdbase+cmd.split('>')[0])
                pass
    return

def generate_xml(testdata,directory):
    """ write testdata information into a jUnit formatted XLM file
    """
    startdir= os.getcwd()
    try:
        os.chdir(directory)
    except OSError:
        print('# No tests run')
        return
    junit = open('../testresults.xml', 'w')
    junit.write('<?xml version="1.0" ?>\n')
    junit.write('<testsuites>\n')
    for pkg in testdata.keys():
        testsuite = testdata[pkg]
        junit.write('  <testsuite errors="%i" failures="%i" name="%s" tests="%i">\n'%(
            testsuite['errors'], testsuite['failed'], pkg, testsuite['total']))
        for prob in testsuite['problems'].keys():
            p = testsuite['problems'][prob]
            junit.write('    <testcase classname="%s" name="%s" time="%f">\n'%(
                p['classname'], prob, p['time']))
            if p['skipped']:
                # if we got here, the TAP output shows a skipped test
                junit.write('      <skipped/>\n')
            elif len(p['stderr'])>0:
                # if we got here, the test crashed with an error
                # we show the stderr output under <error>
                junit.write('      <error type="crash">\n')
                junit.write("<![CDATA[\n") # CDATA is necessary to preserve whitespace
                # many times error messages also go to stdout so we print both
                if len(p['stdout'])>0:
                    for line in p['stdout']:
                        junit.write("%s\n"%line.rstrip())
                for line in p['stderr']:
                    junit.write("%s\n"%line.rstrip())
                junit.write("]]>")
                junit.write('      </error>\n')
            elif len(p['diff'])>0:
                # if we got here, the test output did not match the stored output file
                # we show the diff between new output and old output under <failure>
                junit.write('      <failure type="output">\n')
                junit.write("<![CDATA[\n") # CDATA is necessary to preserve whitespace
                for line in p['diff']:
                    junit.write("%s\n"%line.rstrip())
                junit.write("]]>")
                junit.write('      </failure>\n')
            junit.write('    </testcase>\n')
        junit.write('  </testsuite>\n')
    junit.write('</testsuites>')
    junit.close()
    os.chdir(startdir)
    return

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-d', '--directory', dest='directory',
                      help='Directory containing results of petsc test system',
                      default=os.path.join(os.environ.get('PETSC_ARCH',''),
                                           'tests','counts'))
    parser.add_option('-e', '--elapsed_time', dest='elapsed_time',
                      help='Report elapsed time in output',
                      default=None)
    parser.add_option('-m', '--make', dest='make',
                      help='make executable to report in summary',
                      default='make')
    parser.add_option('-t', '--time', dest='time',
                      help='-t n: Report on the n number expensive jobs',
                      default=0)
    parser.add_option('-f', '--fail', dest='show_fail', action="store_true", 
                      help='Show the failed tests and how to run them')
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return
    

    if not options.show_fail:
      summarize_results(options.directory,options.make,int(options.time),options.elapsed_time)
    testresults=get_test_data(options.directory)

    if options.show_fail:
      show_fail(testresults)
    else:
      generate_xml(testresults, options.directory)

if __name__ == "__main__":
        main()
