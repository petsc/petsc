#!/usr/bin/env python1.5
# $Id: urlget.py,v 1.5 1998/01/15 21:54:18 balay Exp balay $ 
#
#  Retrieves a single file specified as a url and copy it to the specified filename
# 
#  Calling sequence: 
#      urlget.py ftp://hostname/directoryname/file
#
#

#fp = ()
def writefile(buf,flg = 1,filename = ''):
    import sys
    print hello
    if flg == 0 : # Open the file
        fp = open(filename,'w')
    elif flg == 1:
        fp.write(buf)
    else:
        fp.close()


import urllib
import urlparse
import sys
import os
import re
import tempfile
import ftplib

arg_len = len(sys.argv)
if arg_len < 2 : 
    print 'Error! Insufficient arguments.'
    print 'Usage:', sys.argv[0], 'urlfilename localfilename'
    sys.exit()

urlfilename = sys.argv[1]
url_split   = urlparse.urlparse(urlfilename)
# If the url is ftp:// use ftp module to retrive the file
if re.search('ftp',url_split[0]) >= 0 :   
    ftp = ftplib.FTP(url_split[1])
    ftp.login()
    try:      
        outfilename = tempfile.mktemp()
        fp = open(outfilename,'w')
        #This function is a callback function which is called by ftp.retrbinary
        # with the data as argument. This function is responsible to write the
        # data to the file. The fp is opened before the function is
        # declared so as to pass the filepointer to the function
        def writefile(buf,fp = fp):
            fp.write(buf)
        ftp.retrbinary('RETR '+ url_split[2],writefile)
        fp.close()
        ftp.quit()
    except:
        print 'Error! Accessing url on the server',sys.exc_type, sys.exc_value
        ftp.close()
        sys.exit()
# if the url is http: use the url module to retrive the file
elif re.search('http',url_split[0]) >= 0 :  
    tmpfilename = ()
    try:
        tmpfilename = urllib.urlretrieve(urlfilename)
    except:
        print 'Error! Accessing url on the server',sys.exc_type, sys.exc_value
        sys.exit()
    tmpfile = open(tmpfilename[0],'r')
    filesize = os.lseek(tmpfile.fileno(),0,2)
    # Determine the file size to see if it contains error messages
    # Assumption: if filesize < 2000 bytes, it contains error messages
    os.lseek(tmpfile.fileno(),0,0)
    if filesize < 2000 :
        print 'Error! Accessing url on the server. bytes-received :',filesize
        sys.exit()
    outfilename = tempfile.mktemp()
    # tmp file created by urlretrieve() is deleted when python exits
    os.link(tmpfilename[0],outfilename)
else:
    print 'Error! Unknown url format. Use http:// or ftp:// formats only'


os.chmod(outfilename,500)
print outfilename
sys.exit()


