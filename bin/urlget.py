#!/usr/bin/env python
#!/bin/env python
# $Id: urlget.py,v 1.28 2001/06/13 15:39:23 bsmith Exp $ 
#
# change python1.5 to whatever is needed on your system to invoke python
#
#  Retrieves a single file specified as a url and stores it locally.
# 
#  Calling sequence: 
#      urlget.py [-v] [-tmp tmpdir] [-] [http,ftp][://hostname][/]directoryname/file [local_filename]
#
#  Options:
#       -v             - Print version number and exit
#       -tmp           - Uses tmpdir if one is provided, else, uses /tmp
#       -              - use store the file in current dir, with the same filename as the url
#       local_filename - if provided, store the url in the specified  file [relative to CWD]
#  Notes:
#       If the url has a .gz or .Z suffix, the url is uncompressed
#       The uncompresion is not done if local_filename is specified [or the option '-' is used]
#       If the url is neither ftp, nor http, then a local file copy is assumed.
#
import urllib
import os
import sys
import ftplib
import httplib
import string
import urlparse
from time import *
from exceptions import *
from string import *


def parseargs(search_arg,return_nargs,arg_list):
    try:
        index = arg_list.index(search_arg)
    except:
        # Argument not found in list, hence return flag = 0,return val = None
        return 0,None
    
    if return_nargs == 0:
        arg_list.remove(search_arg)
        return 1,None
    
    if index+1 == len(arg_list):
        print 'Error! Option has no value!'
        print 'Expecting value with option: ' + search_arg
        sys.exit()
    else:
        ret_arg = arg_list[index+1]
        arg_list.remove(search_arg)
        arg_list.remove(ret_arg)
        return 1,ret_arg

def extension(filename):
    return split(filename,'.')[-1]

def basename(filename):
    return join(split(filename,'.')[0:-1],'.')

def uncompress(filename):
    ext = extension(filename)
    if ext == 'gz':
        err = os.system('gunzip ' + filename)
        if err != 0:
            print 'Error unable to invoke gunzip on ' + filename
            sys.exit()
    elif ext == 'Z':
        err = os.system('uncompress ' + filename)        
        if err != 0:
            print 'Error unable to invoke uncompress on ' + filename
            sys.exit()

def compressed(filename):
    ext = extension(filename)
    if ext == 'gz' or ext == 'Z':
        return 1
    else:
        return 0
    
# Defines a meta class, whose member functions are common/required
# by ftp/http object classes
class url_object:
    def gettime(): 
        print 'Error Derived function should be implemented'
        sys.exit()
    def getfile(filename):
        print 'Error Derived function should be implemented'
        sys.exit()

class local_object(url_object):
    def __init__(self,machine,urlpath):
        self.localfilename = urlpath
        if os.path.isfile(self.localfilename) == 0:
            print 'Error! file:',self.localfilename,' does not exist'
            sys.exit()
            
    def gettime(self):
        return os.stat(self.localfilename)[7]

    def getfile(self,outfilename):
        fp_in  = open(self.localfilename,'r')
        fp_out = open(outfilename,'w')
        fp_out.write(fp_in.read())
        fp_in.close()
        fp_out.close()

class ftp_object(url_object):
    def __init__(self,machine,urlpath):
        self.machine = machine
        self.urlpath = urlpath
        self.buf     = ''
        try :
            self.ftp     = ftplib.FTP(self.machine)
        except:
            print 'Error! accessing server',self.machine
            sys.exit()
            
        self.ftp.login()

    def __del__(self):
        self.ftp.close()

    def readftplines(self,buf1):
        self.buf   = self.buf + buf1

    def gettime(self):
        
        self.buf       = ''
        self.ftp.retrlines('LIST ' +self.urlpath,self.readftplines)
        if self.buf == '':
            print 'Error! file does not exist on the server'
            self.ftp.close()
            sys.exit()

        month,day,year = split(self.buf)[5:8]
        hour,min       = '0','0'

        if len(split(year,':')) >=2:
            hour,min   = split(year,':')
            year       = gmtime(time())[0]
        else:
            year = atoi(year)

        month_d = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                   'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

        newtime = mktime((year,month_d[month],atoi(day),atoi(hour),
                          atoi(min),0,-1,-1,0))
        c_time  = time()
        if newtime > c_time:
            newtime = mktime((year-1,month_d[month],atoi(day),
                              atoi(hour),atoi(min),0,-1,-1,0))
        self.remotetime = newtime - timezone
        return self.remotetime
    
    def writefile(self,buf):
        self.fp.write(buf)

    def getfile(self,outfilename):
        self.fp  = open(outfilename,'w')
        self.ftp.retrbinary('RETR '+ self.urlpath,self.writefile)
        self.fp.close()


class http_object(url_object):
    def __init__(self,machine,urlpath):
        self.machine = machine
        self.urlpath = urlpath
        try:
            self.http = httplib.HTTP(self.machine)
        except:
            print 'Error! accessing server',self.machine
            sys.exit()
        
        self.http.putrequest('GET',self.urlpath)
        self.http.putheader('Accept','*/*')
        self.http.endheaders()
        errcode, errmesg, self.headers = self.http.getreply()
        if errcode != 200:
            print 'Error! Accessing url on the server.',errcode,errmesg
            self.http.close()
            sys.exit()
        filesize   = self.headers.dict['content-length']
        if filesize < 2000 :
            print 'Error! Accessing url on the server. bytes-received :',filesize
            self.http.close()
            sys.exit()

    def __del__(self):
        self.http.close()


    def gettime(self):

        # Get the remote timestamps
        urltimestamp = self.headers.dict['last-modified']
        month_d             = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                               'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

        if len(split(urltimestamp)) == 6 :      #Sun, 06 Nov 1994 08:49:37 GMT
            day,month,year,time = split(urltimestamp)[1:-1] 
        elif len(split(urltimestamp)) == 4 :    #Sunday, 06-Nov-94 08:49:37 GMT
            time           = split(urltimestamp)[2] 
            day,month,year = split(split(urltimestamp)[1],'-')
        else :                                  #Sun Nov  6 08:49:37 1994
            month,day,time,year = split(urltimestamp)[1:]

        hour,min,sec        = split(time,':')
        newtime             = (atoi(year),month_d[month],atoi(day),atoi(hour),
                               atoi(min),atoi(sec),-1,-1,0)
        self.remotetime     = mktime(newtime) - timezone
        return self.remotetime

    def getfile(self,outfilename):
        #read the data
        f    = self.http.getfile()
        data = f.read()
        f.close()
        # Now write this data to a file
        fp = open(outfilename,'w')
        fp.write(data)
        fp.close()
        
class urlget:

    def __init__(self,url,filename ='',tmpdir='/tmp'):
        self.url                                = urlparse.urlunparse(urlparse.urlparse(url))
        self.protocol,self.machine,self.urlpath = urlparse.urlparse(self.url)[0:3]
        self.compressed = 0
        self.cachefilename = 0

        # Uncompress is not done if the filename is provided
        if filename != '':
            self.cache         = 0
            self.filename      = filename
            self.cachefilename = filename
        else:
            self.cache      = 1
            self.filename   = tmpdir+'/'+replace(join(urlparse.urlparse(self.url)[0:3],'@'),'/','_')
            self.compressed = compressed(self.filename)
            if self.compressed == 1:
                self.cachefilename = basename(self.filename)
            else:
                self.cachefilename = self.filename
                
        if self.protocol == 'ftp':
            self.url_obj = ftp_object(self.machine,self.urlpath)
        elif self.protocol == 'http':
            self.url_obj = http_object(self.machine,self.urlpath)
        else:
            # Assume local file copy
            self.url_obj = local_object(self.machine,self.urlpath)
            #print 'Error! Unknown protocol. use ftp or http protocols only'
            #sys.exit()
        timestamp = self.url_obj.gettime()
        uselocalcopy = 0
        if os.path.isfile(self.cachefilename) == 1:
            mtime = os.stat(self.cachefilename)[7]
            if mtime >= timestamp:
                uselocalcopy = 1

        if self.cache == 0 and os.path.isfile(self.cachefilename) == 1:
            flag = 0
            while flag == 0:
                print self.filename,'exists. Would you like to replace it? (y/n)'
                c = stdin.readline()[0]
                if c == 'y': 
                    uselocalcopy = 0
                    flag = 1
                elif c == 'n':
                    uselocalcopy = 1
                    flag = 1
                    
        if uselocalcopy == 0 :
            self.url_obj.getfile(self.filename)
            os.utime(self.filename,(timestamp,timestamp))
            if self.compressed == 1:
                uncompress(self.filename)
            os.chmod(self.cachefilename,500)

    
def main():

    # Parse for known options.
    flg_tmp,val_tmp = parseargs('-tmp',1,sys.argv)
    flg_hyp,val_hyp = parseargs('-',0,sys.argv)
    flg_v,val_v = parseargs('-v',0,sys.argv)

    if flg_v:
        print 'Version 1.0'
        sys.exit()

    arg_len = len(sys.argv)
    if arg_len < 2: 
        print 'Error! Insufficient arguments.'
        print 'Usage:', sys.argv[0], '[-v] [-tmp tmpdir] [-]  url-filename [local-filename]' 
        sys.exit()

    #Default Values
    tmpdir = '/tmp'
    outfilename = ''
    url = sys.argv[1]
    
    if arg_len == 3:
        outfilename =  sys.argv[2]
    elif flg_hyp:
        outfilename = os.path.basename(url)

    if flg_tmp:
        tmpdir = val_tmp

    x = urlget(url,outfilename,tmpdir)
    print x.cachefilename

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
    main()

