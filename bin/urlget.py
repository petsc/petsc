#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: urlget.py,v 1.21 1999/08/11 20:59:06 balay Exp balay $ 
#
#  Retrieves a single file specified as a url and stores it locally.
# 
#  Calling sequence: 
#      urlget.py ftp://hostname/directoryname/file [local filename]
#
import urllib
import os
import ftplib
import httplib
from exceptions import *
from sys import *
from string import *

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
            exit()
    elif ext == 'Z':
        err = os.system('uncompress ' + filename)        
        if err != 0:
            print 'Error unable to invoke uncompress on ' + filename
            exit()

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
        exit()
    def getfile(filename):
        print 'Error Derived function should be implemented'
        exit()

class ftp_object(url_object):
    def __init__(self,machine,urlpath):
        self.machine = machine
        self.urlpath = urlpath
        self.buf     = ''
        try :
            self.ftp     = ftplib.FTP(self.machine)
        except:
            print 'Error! accessing server',self.machine
            exit()
            
        self.ftp.login()

    def __del__(self):
        self.ftp.close()

    def readftplines(self,buf1):
        self.buf   = self.buf + buf1

    def gettime(self):
        from string import *
        from time import *
        
        self.buf       = ''
        self.ftp.retrlines('LIST ' +self.urlpath,self.readftplines)
        if self.buf == '':
            print 'Error! file does not exist on the server'
            self.ftp.close()
            exit()

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
            exit()
        
        self.http.putrequest('GET',self.urlpath)
        self.http.putheader('Accept','*/*')
        self.http.endheaders()
        errcode, errmesg, self.headers = self.http.getreply()
        if errcode != 200:
            print 'Error! Accessing url on the server.',errcode,errmesg
            self.http.close()
            exit()
        filesize   = self.headers.dict['content-length']
        if filesize < 2000 :
            print 'Error! Accessing url on the server. bytes-received :',filesize
            self.http.close()
            exit()

    def __del__(self):
        self.http.close()


    def gettime(self):
        from string import *
        from time import *

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

    def __init__(self,url,filename =''):
        from urlparse import *
        from string   import *
        self.url                                = urlunparse(urlparse(url))
        self.protocol,self.machine,self.urlpath = urlparse(self.url)[0:3]
        self.compressed = 0
        self.cachefilename = 0
        
        if filename != '':
            self.cache         = 0
            self.filename      = filename
            self.cachefilename = filename
        else:
            self.cache      = 1
            self.filename   = '/tmp/'+replace(join(urlparse(self.url)[0:3],'@'),'/','_')
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
            print 'Error! Unknown protocol. use ftp or http protocols only'
            exit()
            
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
    arg_len = len(argv)

    if arg_len < 2: 
        print 'Error! Insufficient arguments.'
        print 'Usage:', argv[0], 'url-filename [local filename]' 
        exit()

    url = argv[1]
    if arg_len == 3:
        outfilename =  argv[2]
    else:
        outfilename = ''

    x = urlget(url,outfilename)
    print x.cachefilename

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
    main()
