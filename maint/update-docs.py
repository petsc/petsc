#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: update-docs.py,v 1.1 2000/03/25 00:21:13 balay Exp balay $ 
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

    outbuf = header + '\n' + body

    #fix http://www-unix.mcs.anl.gov/petsc/docs
    w = re.compile(r'http://www-unix.mcs.anl.gov/petsc/docs/')
    outbuf = w.sub('',outbuf)

    #fix http://www-unix.mcs.anl.gov/petsc/include (for petscversion.h)
    w = re.compile(r'http://www-unix.mcs.anl.gov/petsc/include/')
    outbuf = w.sub('',outbuf)

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
 
