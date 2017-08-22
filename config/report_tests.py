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
  summary={'total':0,'success':0,'failed':0,'failures':[],'todo':0,'skip':0}
  for cfile in glob.glob('*.counts'):
    with open(cfile, 'r') as f:
      for line in f:
        l = line.split()
        summary[l[0]] += l[1:] if l[0] == 'failures' else int(l[1])

  print("\n# -------------")
  print("#   Summary    ")
  print("# -------------")
  print("# FAILED " + ' '.join(summary['failures']))

  for t in "success failed todo skip".split():
    percent=summary[t]/float(summary['total'])*100
    print("# %s %d/%d tests (%3.1f%%)" % (t, summary[t], summary['total'], percent))
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
