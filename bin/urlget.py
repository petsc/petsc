#!/usr/bin/env python1.5
# $Id: urlget.py,v 1.11 1998/02/03 21:13:12 balay Exp balay $ 
#
#  Retrieves a single file specified as a url and stores it locally.
# 
#  Calling sequence: 
#      urlget.py ftp://hostname/directoryname/file
#
import urllib
import urlparse
import sys
import os
import re
import tempfile
import ftplib
import httplib
from exceptions import *

# Given a url, create a unique filename 
def urltofile(url) :
    from string import *
    from urlparse import *
    url_split = urlparse(urlunparse(urlparse(url)))
    return replace(join(url_split[0:3],'@'),'/','!')
    
# Get the timestamp of the URL from the ftp server
# and convert it to a timestamp useable by utime()
def getftptimestamp(ftp,filename) :
    from string import *
    from time import *
    global buf
    buf = ''
    def readftplines(buf1) :
        global buf
        buf        = buf + buf1
    ftp.retrlines('LIST ' +filename,readftplines)
    month,day,year = split(buf)[5:8]
    hour,min       = '0','0'
    if len(split(year,':')) >=2 :
        hour,min   = split(year,':')
        year       = gmtime(time())[0]
    month_d = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,\
               'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    newtime = mktime((atoi(year),month_d[month],atoi(day),atoi(hour),atoi(min),0,-1,-1,0))
    c_time  = time()
    if newtime > c_time:
        newtime = mktime((atoi(year)-1,month_d[month],atoi(day),atoi(hour),atoi(min),0,-1,-1,0))
    return newtime - timezone

# Convert the Timestamp returned by the http server to a value useable by utime()     
def geturltimestamp(date) :
    from string import *
    from time import *
    month_d             = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,\
                           'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    day,month,year,time = split(date)[1:-1] 
    hour,min,sec        = split(time,':')
    newtime             = (atoi(year),month_d[month],atoi(day),atoi(hour),atoi(min),\
                           atoi(sec),-1,-1,0)
    return mktime(newtime) - timezone

def main() :
    # Set the temp dir location as /tmp
    tempfile.tempdir='/tmp'
    arg_len = len(sys.argv)

    if arg_len < 2 : 
        print 'Error! Insufficient arguments.'
        print 'Usage:', sys.argv[0], 'url-filename'
        sys.exit()

    urlfilename = sys.argv[1]
    url_split   = urlparse.urlparse(urlfilename)
    outfilename = os.path.join(tempfile.tempdir,urltofile(urlfilename))

    # If the url is ftp:// use ftp module to retrive the file
    if re.search('ftp',url_split[0]) >= 0 :   
        ftp = ftplib.FTP(url_split[1])
        ftp.login()
        try:      
            timestamp = getftptimestamp(ftp,url_split[2])
            # if local file exists, check the timestamp, and get the URL only if it is more recent
            uselocalcopy = 0
            if os.path.isfile(outfilename) == 1 :
                mtime = os.stat(outfilename)[7]
                if mtime >= timestamp:
                    uselocalcopy = 1
            if uselocalcopy == 0 :
                fp = open(outfilename,'w')
                # This function is a callback function which is called by ftp.retrbinary
                # with the data as argument. This function is responsible to write the
                # data to the file. The fp is opened before the function is
                # declared so as to pass the filepointer to the function
                def writefile(buf,fp = fp):
                    fp.write(buf)
                ftp.retrbinary('RETR '+ url_split[2],writefile)
                fp.close()
                os.utime(outfilename,(timestamp,timestamp))
            ftp.quit()
        except:
            print 'Error! Accessing url on the server',sys.exc_type, sys.exc_value
            ftp.close()
            sys.exit()
            # if the url is http: use the url module to retrive the file
    elif re.search('http',url_split[0]) >= 0 :  
        tmpfilename = ()
        try:
            h = httplib.HTTP(url_split[1])
            h.putrequest('GET', url_split[2])
            h.putheader('Accept','*/*')
            h.endheaders()
            errcode, errmesg, headers = h.getreply()
            if errcode != 200 :
                print 'Error! Accessing url on the server.',errorcode,errmesg
                h.close()
                sys.exit()
            filesize   = headers.dict['content-length']
            if filesize < 2000 :
                print 'Error! Accessing url on the server. bytes-received :',filesize
                h.close()
                sys.exit()
            # Get Tie remote timestamps
            urltimestamp = headers.dict['last-modified']
            timestamp = geturltimestamp(urltimestamp)
            # if local file exists, check the timestamp, and get the URL only if it is more recent
            uselocalcopy = 0
            if os.path.isfile(outfilename) == 1 :
                mtime = os.stat(outfilename)[7]
                if mtime >= timestamp:
                    uselocalcopy = 1
            if uselocalcopy == 0 :
                f    = h.getfile()
                data = f.read()
                # Now write this data to a file
                fp = open(outfilename,'w')
                fp.write(data)
                fp.close()
                # Change the modified time of the file
                os.utime(outfilename,(timestamp,timestamp))
                f.close()
            h.close()
        except:
            print 'Error! Accessing url on the server',sys.exc_type, sys.exc_value
            h.close()
            sys.exit()
    else:
        print 'Error! Unknown protocol. Use http or ftp protocol only'
        sys.exit()

    os.chmod(outfilename,500)
    print outfilename
    sys.exit()

main()
