
import atexit
import cPickle
import os
import re
import string
import sys
import types
import UserDict

import SocketServer
import socket
import time

class Args (UserDict.UserDict):
  def __init__(self,name,readpw,addpw,writepw):
    UserDict.UserDict.__init__(self)
    self.name    = name
    self.readpw  = readpw
    self.addpw   = addpw
    self.writepw = writepw

class ProcessHandler(SocketServer.StreamRequestHandler):
  def handle(self):
    object  = cPickle.load(self.rfile)
    #  all messages are of the form ("request",dict,key,readpw,addpw,writepw <,value>)
    request = object[0]
    name    = object[1]
    key     = object[2]
    readpw  = object[3]
    dictpw  = object[4]
    addpw   = object[5]
    writepw = object[6]

    
    dargs = self.server.dargs
    dargs.logfile.write("Received "+request+" in "+name+" "+" from "+self.client_address[0]+" "+time.asctime(time.localtime())+'\n')
    dargs.logfile.flush()
    if request == "__setitem__":
      if not dargs.data.has_key(name):
        if dargs.dictpw == dictpw:
          dargs.data[name] = Args(name,readpw,addpw,writepw)
        else:
          dargs.logfile.write("Rejected, wrong dictpw\n");
          cPickle.dump((0,None),self.wfile)
          return
              
      if dargs.data[name].writepw == writepw or (dargs.data[name].addpw == addpw and not dargs.data[name].has_key(key)):
         dargs.data[name].data[key] = object[7]
      cPickle.dump((0,None),self.wfile)
        
    elif request == "__getitem__":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data[key]),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, key or wrong readpw\n");
        cPickle.dump((0,None),self.wfile)

    elif request == "has_key":
      if dargs.data.has_key(name) and dargs.data[name].data.has_key(key) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,None),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, key or wrong readpw\n");
        cPickle.dump((0,None),self.wfile)

    elif request == "__len__":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((len(dargs.data[name].data),None),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, or wrong readpw\n");
        cPickle.dump((0,None),self.wfile)

    elif request == "keys":
      if dargs.data.has_key(name) and dargs.data[name].readpw == readpw:
        cPickle.dump((1,dargs.data[name].data.keys()),self.wfile)
      else:
        dargs.logfile.write("Rejected, missing dictionary, or wrong readpw\n");
        cPickle.dump((0,None),self.wfile)

    elif request == "dicts":
      di = []
      for d in dargs.data.keys():
        if dargs.data[d].readpw == readpw:
          di.append(d)
      cPickle.dump((0,tuple(di)),self.wfile)

    elif request == "clear":
      if dargs.data.has_key(name) and dargs.data[name].writepw == writepw:
        dargs.data[name].data.clear()
      else:
        dargs.logfile.write("Rejected, missing dictionary, wrong writepw\n");
      cPickle.dump((0,None),self.wfile)

    elif request == "__delitem__":
      if dargs.data.has_key(name):
        if dargs.data[name].writepw == writepw:
          try:
            del dargs.data[name].data[key]
          except:
            dargs.logfile.write("Rejected, missing key\n");
        else:
          dargs.logfile.write("Rejected, wrong writepw\n");
      else:
        dargs.logfile.write("Rejected, missing dictionary\n");
      cPickle.dump((0,None),self.wfile)

class DArgs:
  def __init__(self, filename = "DArgs.db", dictpw = "open"):
    self.data      = UserDict.UserDict()
    self.filename  = filename
    self.load(filename)
    self.dictpw    = dictpw
    self.logfile   = open("DArgs.log",'a')
    atexit.register(self.save)

  def load(self, filename):
    if filename and os.path.exists(filename):
      dbFile    = open(filename, 'r')
      self.data = cPickle.load(dbFile)
      dbFile.close()

  def save(self):
    dbFile = open(self.filename, 'w')
    cPickle.dump(self.data, dbFile)
    dbFile.close()
    try: os.unlink('DArgs.loc')
    except: pass

  def loop(self):
    # wish there was a better way to get a usable socket
    p    = 1
    flag = "nosocket"
    while p < 1000 and flag == "nosocket":
      try:
        server = SocketServer.TCPServer((socket.gethostname(),6000+p),ProcessHandler)
        flag   = "socket"
      except:
        p = p + 1
    if p == 1000:
      raise RuntimeError,"Cannot get available socket"
        
    f = open("DArgs.loc", 'w')
    cPickle.dump(server.server_address, f)
    f.close()

    self.logfile.write("Started server"+time.asctime(time.localtime())+'\n')
    self.logfile.flush()
    server.dargs = self
    server.serve_forever()
    

if __name__ ==  '__main__':
    dargs = DArgs()
    dargs.loop()
