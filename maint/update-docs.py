#!/usr/bin/env python
#!/bin/env python
# $Id: update-docs.py,v 1.7 2001/08/30 17:51:36 bsmith Exp $ 
#
# update-docs.py LOC
#

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

# given docs/installation/unix.htm, check if the dir docs/installation exists.
# if not, create it.
def chkdir(dirname):
    if not os.path.isdir(dirname):
        print 'Creating dir', dirname
        os.mkdir(dirname)

def main():
    arg_len = len(argv)

    if arg_len < 2:
        print 'Error Insufficient arguments.'
        print 'Usage:', argv[0], 'LOC'

    LOC = argv[1]
    baseurl = 'http://www-fp.mcs.anl.gov/petsc'
    htmlfiles = [
        'docs/bugreporting.html',
        'docs/codemanagement.html',
        'docs/copyright.html',
        'docs/faq.html',
        'docs/index.html',
        'docs/machines.html',
        'docs/troubleshooting.html',
        'docs/changes/2015.htm',
        'docs/changes/2016.htm',
        'docs/changes/2017.htm',
        'docs/changes/2022.htm',
        'docs/changes/2024.htm',
        'docs/changes/2028.htm',
        'docs/changes/2029.htm',
        'docs/changes/21.htm',
        'docs/changes/211.htm',
        'docs/changes/2918-21.htm',
        'docs/changes/index.htm',
        'docs/installation/index.htm',
        'docs/installation/unix-ams.htm',
        'docs/installation/unix.htm',
        'docs/installation/win.htm',
        'docs/installation/packages.htm']

    for basename in htmlfiles:
        urlname  = baseurl + '/' + basename
        filename = LOC + '/' + basename
        dirpath = os.path.dirname(filename)
        chkdir(dirpath)
        wgetcmd = 'wget -nv ' + urlname + ' -O ' + filename
        os.system(wgetcmd)
        modifyfile(filename)
        
# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
 
