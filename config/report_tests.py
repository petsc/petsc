#!/usr/bin/env python
import glob, os
import optparse
"""
Quick script for parsing the output of the test system and summarizing the results.
"""

def summarize_results(directory):
  ''' Loop over all of the results files and summarize the results'''
  startdir=os.path.realpath(os.path.curdir)
  try:
    os.chdir(directory)
  except OSError:
    print('# No tests run')
    return
  summary={'total':0,'success':0,'failed':0,'failures':'','todo':0,'skip':0}
  for cfile in glob.glob('*.counts'):
    sh=open(cfile,"r"); fileStr=sh.read(); sh.close()
    for line in fileStr.split('\n'):
      if not line: break
      try:
        var,val=line.split()
        if not val.strip(): continue
        val=int(val)
      except:
        var=line.split()[0]
        lval=len(line.split())-1
        if lval==0: continue
        val=line.split()[1]
        if not val.strip(): continue
        append=" ("+str(lval)+"), " if lval>1 else ", "
        val=val+append

      summary[var]=summary[var]+val

  print "\n# -------------"
  print "#   Summary    "
  print "# -------------"
  print "# FAILED "+summary['failures'].rstrip(', ')
  total=str(summary['total'])

  for t in "success failed todo skip".split():
    percent=summary[t]/float(summary['total'])*100
    print ("# "+t+" "+ str(summary[t])+"/"+total+" tests (%3.1f%%)") % (percent)
  return

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-d', '--directory', dest='directory',
                      help='Directory containing results of petsc test system',
                      default='counts')
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    summarize_results(options.directory)

if __name__ == "__main__":
        main()
