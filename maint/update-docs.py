#!/usr/bin/env python
#!/bin/env python
# $Id: update-docs.py,v 1.7 2001/08/30 17:51:36 bsmith Exp $ 
#
# update-docs.py DOCS_DIR
import os
import glob
import posixpath
import string
from sys import *

def modifyfile(filename):
    print 'processing file : ' + filename

    import re
    try:
        fd = open(filename,'r')
    except:
        print 'Error! Cannot open file:',filename
        exit()
    buf    = fd.read()
    fd.close()

    header = string.split(string.split(buf, '<!##end>')[0],'<!##begin>')[1]
    body = string.split(string.split(buf, '<!##end>')[1],'<!##begin>')[1]

    outbuf = '<html>\n<body BGCOLOR="FFFFFF">\n' + header + '\n' + body + '</body>\n</html>\n'

    #fix http://www-unix.mcs.anl.gov/petsc/petsc-current/docs
    w = re.compile(r'http://www-unix.mcs.anl.gov/petsc/petsc-current/docs/')
    outbuf = w.sub('',outbuf)

    #fix http://www-unix.mcs.anl.gov/petsc/petsc-current/include (for petscversion.h)
    w = re.compile(r'http://www-unix.mcs.anl.gov/petsc/petsc-current/include/')
    outbuf = w.sub('',outbuf)

    # now revert all the links to the splitmanuals back to the website
    w = re.compile(r'splitmanual/')
    outbuf = w.sub('http://www-unix.mcs.anl.gov/petsc/petsc-current/docs/splitmanual/',outbuf)
    
    # Now overwrite the original file 
    outfilename = filename
    try:
        fd = open( outfilename,'w')
    except:
        print 'Error writing to file',outfilename
        exit()

    fd.write(outbuf)
    fd.close()
    return
    

def main():
    arg_len = len(argv)
    
    if arg_len < 2: 
        print 'Error! Insufficient arguments.'
        print 'Usage:', argv[0], 'DOCS_DIR'
        exit()

    DOCS_DIR  = argv[1]
    # Remove unnecessary stuff from the dir
    os.system('/bin/rm ' + DOCS_DIR + '/petsc.html')
    os.system('/bin/rm ' + DOCS_DIR + '/referencing.htm')
    os.system('/bin/rm -rf ' + DOCS_DIR + '/tutorials')
    os.system('/bin/rm -rf ' + DOCS_DIR + '/_vti_cnf')
    os.system('/bin/rm -rf ' + DOCS_DIR + '/*/_vti_cnf')
    

    htmlfiles     = glob.glob(DOCS_DIR + '/*.htm') + \
                    glob.glob(DOCS_DIR + '/*.html') + \
                    glob.glob(DOCS_DIR + '/*/*.htm') + \
                    glob.glob(DOCS_DIR + '/*/*.html')

    #print htmlfiles
    for file in htmlfiles:
        modifyfile(file)


      
# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
 
