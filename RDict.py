#!/usr/bin/env python
#
#   RDict - A remote dictionary server
#
# This is necessary for us to create Project objects on load
import project
import nargs

import cPickle
import os

class RDict(dict):
  '''An RDict is a typed dictionary, which may be hierarchically composed. All elements derive from the
Arg class, which wraps the usual value.'''
  def __init__(self, parentAddr = None, parentDirectory = None):
    import atexit
    self.logFile         = file('RDict.log', 'a')
    self.target          = ['default']
    self.parent          = None
    self.saveTimer       = None
    self.saveFilename    = 'RDict.db'
    self.addrFilename    = 'RDict.loc'
    self.parentAddr      = parentAddr
    self.isServer        = 0
    self.parentDirectory = parentDirectory
    self.stopCmd         = cPickle.dumps(('stop',))
    self.writeLogLine('Greetings')
    self.connectParent(self.parentAddr, self.parentDirectory)
    self.load()
    atexit.register(self.shutdown)
    return

  def writeLogLine(self, message):
    '''Writes the message to the log along with the current time'''
    import time
    self.logFile.write('('+str(os.getpid())+')('+str(id(self))+')'+message+' ['+time.asctime(time.localtime())+']\n')
    self.logFile.flush()
    return

  def __len__(self):
    '''Returns the length of both the local and parent dictionaries'''
    length = dict.__len__(self)
    if not self.parent is None:
      length += self.send()
    return length

  def getType(self, key):
    '''Checks for the key locally, and if not found consults the parent. Returns the Arg object or None if not found.'''
    if dict.has_key(self, key):
      self.writeLogLine('getType: Getting local type for '+key+' '+str(dict.__getitem__(self, key)))
      return dict.__getitem__(self, key)
    elif not self.parent is None:
      return self.send(key)
    return None

  def __getitem__(self, key):
    '''Checks for the key locally, and if not found consults the parent. Returns the value of the Arg.
       - If the value has not been set, the user will be prompted for input'''
    if dict.has_key(self, key):
      self.writeLogLine('__getitem__: '+key+' has local type')
      pass
    elif not self.parent is None:
      self.writeLogLine('__getitem__: Checking parent value')
      if self.send(key, operation = 'has_key'):
        self.writeLogLine('__getitem__: Parent has value')
        return self.send(key)
      else:
        self.writeLogLine('__getitem__: Checking parent type')
        arg = self.send(key, operation = 'getType')
        if not arg:
          self.writeLogLine('__getitem__: Parent has no type')
          arg = nargs.Arg(key)
        value = arg.getValue()
        self.writeLogLine('__getitem__: Setting parent value '+str(value))
        self.send(key, value, operation = '__setitem__')
        return value
    else:
      self.writeLogLine('__getitem__: Setting local type for '+key)
      dict.__setitem__(self, key, nargs.Arg(key))
    self.writeLogLine('__getitem__: Setting local value for '+key)
    return dict.__getitem__(self, key).getValue()

  def setType(self, key, value, forceLocal = 0):
    '''Checks for the key locally, and if not found consults the parent. Sets the type for this key.'''
    if not isinstance(value, nargs.Arg):
      raise TypeError('An argument type must be a subclass of Arg')
    value.setKey(key)
    if forceLocal or self.parent is None or dict.has_key(self, key):
      if dict.has_key(self, key):
        v = dict.__getitem__(self, key)
        if isinstance(value, v.__class__) and v.isValueSet():
          value.setValue(v.getValue())
      dict.__setitem__(self, key, value)
      self.save()
    else:
      return self.send(key, value)
    return

  def __setitem__(self, key, value):
    '''Checks for the key locally, and if not found consults the parent. Sets the value of the Arg.'''
    if not dict.has_key(self, key):
      if not self.parent is None:
        return self.send(key, value)
      else:
        dict.__setitem__(self, key, nargs.Arg(key))
    dict.__getitem__(self, key).setValue(value)
    self.writeLogLine('__setitem__: Set value for '+key+' to '+str(dict.__getitem__(self, key)))
    self.save()
    return

  def __delitem__(self, key):
    '''Checks for the key locally, and if not found consults the parent. Deletes the Arg completely.'''
    if dict.has_key(self, key):
      dict.__delitem__(self, key)
      self.save()
    elif not self.parent is None:
      self.send(key)
    return

  def clear(self):
    '''Clears both the local and parent dictionaries'''
    if dict.__len__(self):
      dict.clear(self)
      self.save()
    if not self.parent is None:
      self.send()
    return

  def __contains__(self, key):
    '''This method just calls self.has_key(key)'''
    return self.has_key(key)

  def has_key(self, key):
    '''Checks for the key locally, and if not found consults the parent. Then checks whether the value has been set'''
    if dict.has_key(self, key):
      if dict.__getitem__(self, key).isValueSet():
        self.writeLogLine('has_key: Have value for '+key)
      else:
        self.writeLogLine('has_key: Do not have value for '+key)
      return dict.__getitem__(self, key).isValueSet()
    elif not self.parent is None:
      return self.send(key)
    return 0

  def hasType(self, key):
    '''Checks for the key locally, and if not found consults the parent. Then checks whether the type has been set'''
    if dict.has_key(self, key):
      return 1
    elif not self.parent is None:
      return self.send(key)
    return 0

  def items(self):
    '''Return a list of all accessible items, as (key, value) pairs.'''
    l = dict.items(self)
    if not self.parent is None:
      l.extend(self.send())
    return l

  def localitems(self):
    '''Return a list of all the items stored locally, as (key, value) pairs.'''
    return dict.items(self)

  def keys(self):
    '''Returns the list of keys in both the local and parent dictionaries'''
    keyList = filter(lambda key: dict.__getitem__(self, key).isValueSet(), dict.keys(self))
    if not self.parent is None:
      keyList.extend(self.send())
    return keyList

  def types(self):
    '''Returns the list of keys for which types are defined in both the local and parent dictionaries'''
    keyList = dict.keys(self)
    if not self.parent is None:
      keyList.extend(self.send())
    return keyList

  def insertArg(self, key, value, arg):
    '''Insert a (key, value) pair into the dictionary. If key is None, arg is put into the target list.'''
    if not key is None:
      self[key] = value
    else:
      if not self.target == ['default']:
        self.target.append(arg)
      else:
        self.target = [arg]
    return

  def insertArgs(self, args):
    '''Insert some text arguments into the dictionary (list and dictionaries are recognized)'''
    import UserDict

    if isinstance(args, list):
      for arg in args:
        (key, value) = nargs.Arg.parseArgument(arg)
        self.insertArg(key, value, arg)
    # Necessary since os.environ is a UserDict
    elif isinstance(args, dict) or isinstance(args, UserDict.UserDict):
      for key in args.keys():
        if isinstance(args[key], str):
          value = nargs.Arg.parseValue(args[key])
        else:
          value = args[key]
        self.insertArg(key, value, None)
    elif isinstance(args, str):
        (key, value) = nargs.Arg.parseArgument(args)
        self.insertArg(key, value, args)
    return

  def hasParent(self):
    return not self.parent is None

  def getServerAddr(self, dir):
    filename = os.path.join(dir, self.addrFilename)
    for i in range(10):
      try:
        if not os.path.exists(filename):
          self.startServer(filename)
        f    = open(filename, 'r')
        addr = cPickle.load(f)
        f.close()
        # Check if server is running
        #   This must provide the address, not the directory to prevent an infinite loop
        rdict     = RDict(parentAddr = addr)
        hasParent = rdict.hasParent()
        del rdict
        if not hasParent:
          os.remove(filename)
          self.startServer(filename)
        return addr
      except Exception:
        self.startServer(filename)
    raise RuntimeError('Could not get server address in '+filename)

  def writeServerAddr(self, server):
    f = file(self.addrFilename, 'w')
    cPickle.dump(server.server_address, f)
    f.close()
    self.writeLogLine('SERVER: Wrote lock file '+os.path.abspath(self.addrFilename))
    return

  def startServer(self, addrFilename):
    import RDict # Need this to locate server script
    import sys
    import time

    try: os.remove(addrFilename)
    except: pass
    oldDir = os.getcwd()
    source = os.path.join(os.path.dirname(os.path.abspath(sys.modules['RDict'].__file__)), 'RDict.py')
    os.chdir(os.path.dirname(addrFilename))
    os.spawnvp(os.P_NOWAIT, 'python', ['python', source, 'server'])
    os.chdir(oldDir)
    for i in range(10):
      time.sleep(1)
      if os.path.exists(addrFilename): return
    raise RuntimeError('No running server: Could not start it')

  def connectParent(self, addr, dir):
    if addr is None:
      if dir is None: return 0
      addr = self.getServerAddr(dir)

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.writeLogLine('CLIENT: Trying to connect to '+str(addr))
      s.connect(addr)
    except Exception, e:
      self.writeLogLine('CLIENT: Failed to connect: '+str(e))
      return 0
    self.parent = s
    self.writeLogLine('CLIENT: Connected to '+str(self.parent))
    return 1

  def send(self, key = None, value = None, operation = None):
    import inspect

    objString = ''
    for i in range(3):
      try:
        sendPacket = []
        if operation is None:
          operation = inspect.stack()[1][3]
        sendPacket.append(operation)
        if not key is None:
          sendPacket.append(key)
          if not value is None:
            sendPacket.append(value)
        self.writeLogLine('CLIENT: Sending packet '+str(sendPacket))
        self.parent.sendall(cPickle.dumps(tuple(sendPacket)))
        self.writeLogLine('CLIENT: Receiving value')
        objString = ''
        for i in range(10):
          try:
            objString += self.parent.recv(100000)
            #self.writeLogLine('CLIENT: Unpickling '+objString)
            response   = cPickle.loads(objString)
            break
          except Exception, e:
            self.writeLogLine('CLIENT: Error receiving packet '+str(e)+' '+str(e.__class__))
            continue
        break
      except IOError, e:
        self.writeLogLine('CLIENT: IOError '+str(e))
        if e.errno == 32:
          self.connectParent(self.parentAddr, self.parentDirectory)
      except Exception, e:
        self.writeLogLine('CLIENT: Exception '+str(e)+' '+str(e.__class__))
    try:
      if isinstance(response, Exception):
        self.writeLogLine('CLIENT: Got an exception '+str(response))
        raise response
      else:
        self.writeLogLine('CLIENT: Received value '+str(response)+' '+str(type(response)))
    except AttributeError:
      self.writeLogLine('CLIENT: Could not unpickle response '+objString)
      response  = None
    return response

  def serve(self):
    '''Start a server'''
    import socket
    import SocketServer

    class ProcessHandler(SocketServer.StreamRequestHandler):
      def handle(self):
        self.server.rdict.writeLogLine('SERVER: Started new handler')
        while 1:
          try:
            value = cPickle.load(self.rfile)
            self.server.rdict.writeLogLine('SERVER: Received packet '+str(value))
          except EOFError, e:
            self.server.rdict.writeLogLine('SERVER: EOFError receiving packet '+str(e)+' '+str(e.__class__))
            return
          except Exception, e:
            self.server.rdict.writeLogLine('SERVER: Error receiving packet '+str(e)+' '+str(e.__class__))
            cPickle.dump(e, self.wfile)
            continue
          if value[0] == 'stop': break
          response = getattr(self.server.rdict, value[0])(*value[1:])
          self.server.rdict.writeLogLine('SERVER: Sending response '+str(response))
          cPickle.dump(response, self.wfile)
          self.server.rdict.writeLogLine('SERVER: Sent response '+str(response))
        return

    # check if server is running
    if os.path.exists(self.addrFilename):
      rdict     = RDict(parentDirectory = '.')
      hasParent = rdict.hasParent()
      del rdict
      if hasParent:
        self.writeLogLine('SERVER: Another server is already running')
        raise RuntimeError('Server already running')

    # wish there was a better way to get a usable socket
    basePort = 8000
    flag     = 'nosocket'
    p        = 1
    while p < 1000 and flag == 'nosocket':
      try:
        server = SocketServer.ThreadingTCPServer((socket.gethostname(), basePort+p), ProcessHandler)
        flag   = 'socket'
      except Exception, e:
        p = p + 1
    if flag == 'nosocket':
      p = 1
      while p < 1000 and flag == 'nosocket':
        try:
          server = SocketServer.ThreadingTCPServer(('localhost', basePort+p), ProcessHandler)
          flag   = 'socket'
        except Exception, e:
          p = p + 1
    if flag == 'nosocket':
      raise RuntimeError,'Cannot get available socket'

    self.isServer = 1
    self.writeServerAddr(server)
 
    server.rdict = self
    self.writeLogLine('SERVER: Started server')
    server.serve_forever()
    return

  def load(self):
    '''Load the saved dictionary'''
    if not self.parentDirectory is None and os.path.samefile(os.getcwd(), self.parentDirectory):
      return
    if os.path.exists(self.saveFilename):
      try:
        dbFile = file(self.saveFilename)
        data   = cPickle.load(dbFile)
        self.update(data)
        dbFile.close()
        self.writeLogLine('Loaded dictionary from '+self.saveFilename)
      except Exception, e:
        self.writeLogLine('Problem loading dictionary from '+self.saveFilename+'\n--> '+str(e))
    else:
      self.writeLogLine('No dictionary to load in this file: '+self.saveFilename)
    return

  def save(self, force = 0):
    '''Save the dictionary after 5 seconds, ignoring all subsequent calls until the save
       - Giving force = True will cause an immediate save'''
    if force:
      self.saveTimer = None
      # This should be a critical section
      dbFile = file(self.saveFilename, 'w')
      data   = dict(filter(lambda i: not i[1].getTemporary(), self.localitems()))
      cPickle.dump(data, dbFile)
      dbFile.close()
      self.writeLogLine('Saved local dictionary to '+os.path.abspath(self.saveFilename))
    elif not self.saveTimer:
      import threading
      self.saveTimer = threading.Timer(5, self.save, [], {'force': 1})
      self.saveTimer.start()
    return

  def shutdown(self):
    if self.saveTimer:
      self.saveTimer.cancel()
    if self.isServer and os.path.isfile(self.addrFilename):
      os.remove(self.addrFilename)
    if not self.parent is None:
      self.parent.sendall(self.stopCmd)
      self.parent.close()
      self.parent = None
    self.writeLogLine('Shutting down')
    self.logFile.close()
    return

if __name__ ==  '__main__':
  import sys
  try:
    if len(sys.argv) < 2:
      print 'RDict.py [server | client | clear]'
    else:
      action = sys.argv[1]
      parent = None
      if len(sys.argv) > 2:
        parent = sys.argv[2]
      if action == 'server':
        RDict().serve()
      elif action == 'client':
        print 'Entries in server dictionary'
        rdict = RDict(parentDirectory = parent)
        for key in rdict.keys():
          print str(key)+' '+str(rdict[key])
      elif action == 'clear':
        print 'Clearing all dictionaries'
        RDict().clear()
      elif action == 'insert':
        rdict = RDict(parentDirectory = parent)
        rdict[sys.argv[3]] = sys.argv[4]
      else:
        sys.exit('Unknown action: '+action)
  except Exception, e:
    import traceback
    print traceback.print_tb(sys.exc_info()[2])
    sys.exit(str(e))
  sys.exit(0)
