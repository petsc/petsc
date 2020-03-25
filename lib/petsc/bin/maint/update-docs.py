#!/usr/bin/env python
#!/bin/env python
#
# update-docs.py LOC
# update-docs.py LOC clean
#

from __future__ import print_function
import os
import glob
import posixpath
from sys import *
import shutil
import os.path

def modifyfile(filename):
    print('processing file : ' + filename)

    import re
    try:
        fd = open(filename,'r')
    except:
        print('Error! Cannot open file:',filename)
        exit()
    buf    = fd.read()
    fd.close()

    header = buf.split('<!--end-->')[0].split('<!--begin-->')[1]
    body = buf.split('<!--end-->')[1].split('<!--begin-->')[1]

    outbuf = '''
 <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
 <html>
  <head>
    <meta http-equiv="content-type" content="text/html;charset=utf-8">
    <title>''' + header + '''</title>
  </head>
  <body bgcolor="#ffffff">

    <h1>''' + header + '''</h1>

    ''' + body + '''
  </body>
</html>
'''

    #fix https://www.mcs.anl.gov/petsc/petsc-current/docs/
    w = re.compile(r'https://www.mcs.anl.gov/petsc/petsc-current/docs/')
    outbuf = w.sub('',outbuf)

    #fix  https://www.mcs.anl.gov/petsc/petsc-current/include/ (for petscversion.h)
    w = re.compile(r'https://www.mcs.anl.gov/petsc/petsc-current/include/')
    outbuf = w.sub('',outbuf)

    # Now overwrite the original file
    outfilename = filename
    try:
        fd = open( outfilename,'w')
    except:
        print('Error writing to file',outfilename)
        exit()

    fd.write(outbuf)
    fd.close()
    return

# given docs/installation/unix.htm, check if the dir docs/installation exists.
# if not, create it.
def chkdir(dirname):
    if not os.path.isdir(dirname):
        print('Creating dir', dirname)
        os.mkdir(dirname)

def rmfile(filename):
    if  os.path.isfile(filename):
        os.remove(filename)

def main():
    arg_len = len(argv)

    if arg_len < 3:
        print('Error Insufficient arguments.')
        print('Usage:', argv[0], 'PETSC_DIR LOC')
    PETSC_DIR = argv[1]
    LOC = argv[2]

    cleanfiles = 0
    if arg_len == 4:
        if argv[3] == 'clean' :
          cleanfiles = 1

    baseurl = PETSC_DIR + '/src/docs/website/documentation/'
    htmlfiles = [
        'bugreporting.html',
        'codemanagement.html',
        'copyright.html',
        'faq.html',
        'index.html',
        'license.html',
        'linearsolvertable.html',
        'nonlinearsolvertable.html',
        'changes/2015.html',
        'changes/2016.html',
        'changes/2017.html',
        'changes/2018-21.html',
        'changes/2022.html',
        'changes/2024.html',
        'changes/2028.html',
        'changes/2029.html',
        'changes/21.html',
        'changes/211.html',
        'changes/212.html',
        'changes/213.html',
        'changes/215.html',
        'changes/216.html',
        'changes/220.html',
        'changes/221.html',
        'changes/230.html',
        'changes/231.html',
        'changes/232.html',
        'changes/233.html',
        'changes/300.html',
        'changes/31.html',
        'changes/32.html',
        'changes/33.html',
        'changes/34.html',
        'changes/35.html',
        'changes/36.html',
        'changes/37.html',
        'changes/38.html',
        'changes/39.html',
        'changes/310.html',
        'changes/311.html',
        'changes/312.html',
        'changes/313.html',
        'changes/dev.html',
        'changes/index.html',
        'installation.html']

    # if clean option is provided then delete the files and exit
    if cleanfiles == 1 :
        for basename in htmlfiles:
            urlname  = baseurl + basename
            filename = LOC + '/docs/' + basename
            if os.path.isfile(filename): rmfile(filename)
        exit()

    for basename in htmlfiles:
        urlname  = baseurl + basename
        filename = LOC + '/docs/' + basename
        dirpath = os.path.dirname(filename)
        chkdir(dirpath)
        if os.path.isfile(urlname): # dev.html should not exist in a release
            shutil.copyfile(urlname,filename)
            modifyfile(filename)

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
      main()

