#!/usr/bin/env python
#!/bin/env python
#
#    Patches PETSc examples to support latex in the formated comments
#
#  See http://www.mathjax.org/docs/1.1/options/tex2jax.html#configure-tex2jax
import os
import re
from exceptions import *
import sys
from string import *
import commands

def processexample(example):
  mat   = re.compile('<center>[a-zA-Z0-9_ -]*</center>')
  nam   = re.compile('<a name=\"[a-z]*\">')
  inp   = 0

  f     = open(example)
  lines = f.readlines()
  f.close()
  g     = open(example,'w')
  for line in lines:
    if '/*F' in line:
      inp = 1
      g.write(line)
      g.write('</pre>')
      g.write('''<script type="text/x-mathjax-config">
                   MathJax.Hub.Config({
                       tex2jax: {inlineMath: [['$','$'], ['\\\\(','\\\\)']]}
                   });
                 </script>
                 <script type="text/javascript"
                      src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                 </script>''')
    elif 'F*/' in line:
      inp = 0
      g.write('<pre width="80">')
      g.write(line)
    elif inp:
      s = line.find('"#B22222">')
      g.write(line[s+10:-8]+'<BR>\n\n')
    else:
      g.write(line)
  g.close()

def main():
  for examples in sys.argv[1:]:
    processexample(examples)

#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

