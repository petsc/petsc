#!/usr/bin/env python
import glob, os, re
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

def summarize_results(directory,make,ntime):
  ''' Loop over all of the results files and summarize the results'''
  startdir=os.path.realpath(os.path.curdir)
  try:
    os.chdir(directory)
  except OSError:
    print('# No tests run')
    return
  summary={'total':0,'success':0,'failed':0,'failures':[],'todo':0,'skip':0,
           'time':0}
  timesummary={}
  timelist=[]
  for cfile in glob.glob('*.counts'):
    with open(cfile, 'r') as f:
      for line in f:
        l = line.split()
        summary[l[0]] += l[1:] if l[0] == 'failures' else int(l[1])
        if l[0] == 'time' and int(l[1])>0:
            timesummary[cfile]=int(l[1])
            timelist.append(int(l[1]))

  failstr=' '.join(summary['failures'])
  print("\n# -------------")
  print("#   Summary    ")
  print("# -------------")
  if failstr.strip(): print("# FAILED " + failstr)

  for t in "success failed todo skip".split():
    percent=summary[t]/float(summary['total'])*100
    print("# %s %d/%d tests (%3.1f%%)" % (t, summary[t], summary['total'], percent))
  print("#\n# Approximate time (not incl. build time): %s sec"% summary['time'])

  if failstr.strip():
      fail_targets=(
          re.sub('(?<=[0-9]_\w)_.*','',
          re.sub('_1 ',' ',
          re.sub('cmd-','',
          re.sub('diff-','',failstr+' '))))
          )
      # Need to make sure we have a unique list
      fail_targets=' '.join(list(set(fail_targets.split())))

      #Make the message nice
      makefile="gmakefile.test" if inInstallDir() else "gmakefile"

      print("#\n# To rerun failed tests: ")
      print("#     "+make+" -f "+makefile+" test search='" + fail_targets.strip()+"'")

  if ntime>0:
      print("# Timing summary: ")
      timelist=list(set(timelist))
      timelist.sort(reverse=True)
      nlim=(ntime if ntime<len(timelist) else len(timelist))
      # Do a double loop to sort in order
      for timelimit in timelist[0:nlim]:
        for cf in timesummary:
          if timesummary[cf] == timelimit:
              print("# %s: %d sec" % (re.sub('.counts','',cf), timesummary[cf]))

  return

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-d', '--directory', dest='directory',
                      help='Directory containing results of petsc test system',
                      default=os.path.join(os.environ.get('PETSC_ARCH',''),
                                           'tests','counts'))
    parser.add_option('-m', '--make', dest='make',
                      help='make executable to report in summary',
                      default='make')
    parser.add_option('-t', '--time', dest='time',
                      help='-t n: Report on the n number expensive jobs',
                      default=0)
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    summarize_results(options.directory,options.make,int(options.time))

if __name__ == "__main__":
        main()
