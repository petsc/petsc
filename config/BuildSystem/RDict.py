'''A remote dictionary server

    RDict is a typed, hierarchical, persistent dictionary intended to manage
    all arguments or options for a program. The interface remains exactly the
    same as dict, but the storage is more complicated.

    Argument typing is handled by wrapping all values stored in the dictionary
    with nargs.Arg or a subclass. A user can call setType() to set the type of
    an argument without any value being present. Whenever __getitem__() or
    __setitem__() is called, values are extracted or replaced in the wrapper.
    These wrappers can be accessed directly using getType(), setType(), and
    types().

    Hierarchy is allowed using a single "parent" dictionary. All operations
    cascade to the parent. For instance, the length of the dictionary is the
    number of local keys plus the number of keys in the parent, and its
    parent, etc. Also, a dictionary need not have a parent. If a key does not
    appear in the local dicitonary, the call if passed to the parent. However,
    in this case we see that local keys can shadow those in a parent.
    Communication with the parent is handled using sockets, with the parent
    being a server and the interactive dictionary a client.

    The default persistence mechanism is a pickle file, RDict.db, written
    whenever an argument is changed locally. A timer thread is created after
    an initial change, so that many rapid changes do not cause many writes.
    Each dictionary only saves its local entries, so all parents also
    separately save data in different RDict.db files. Each time a dictionary
    is created, the current directory is searched for an RDict.db file, and
    if found the contents are loaded into the dictionary.

    This script also provides some default actions:

      - server [parent]
        Starts a server in the current directory with an optional parent. This
        server will accept socket connections from other dictionaries and act
        as a parent.

      - client [parent]
        Creates a dictionary in the current directory with an optional parent
        and lists the contents. Notice that the contents may come from either
        an RDict.db file in the current directory, or from the parent.

      - clear [parent]
        Creates a dictionary in the current directory with an optional parent
        and clears the contents. Notice that this will also clear the parent.

      - insert <parent> <key> <value>
        Creates a dictionary in the current directory with a parent, and inserts
        the key-value pair. If "parent" is "None", no parent is assigned.

      - remove <parent> <key>
        Creates a dictionary in the current directory with a parent, and removes
        the given key. If "parent" is "None", no parent is assigned.
'''
from __future__ import print_function
from __future__ import absolute_import
try:
  import project          # This is necessary for us to create Project objects on load
  import build.buildGraph # This is necessary for us to create BuildGraph objects on load
except ImportError:
  pass
import nargs

import pickle
import os
import sys
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads is None:
  useThreads = 0 # workaround issue with parallel configure
elif useThreads == 'no' or useThreads == '0':
  useThreads = 0
elif useThreads == 'yes' or useThreads == '1':
  useThreads = 1
else:
  raise RuntimeError('Unknown option value for --useThreads ',useThreads)

class RDict(dict):
  '''An RDict is a typed dictionary, which may be hierarchically composed. All elements derive from the
Arg class, which wraps the usual value.'''
  # The server will self-shutdown after this many seconds
  shutdownDelay = 60*60*5

  def __init__(self, parentAddr = None, parentDirectory = None, load = 1, autoShutdown = 1, readonly = False):
    import atexit
    import time
    import xdrlib

    self.logFile         = None
    self.setupLogFile()
    self.target          = ['default']
    self.parent          = None
    self.saveTimer       = None
    self.shutdownTimer   = None
    self.lastAccess      = time.time()
    self.saveFilename    = 'RDict.db'
    self.addrFilename    = 'RDict.loc'
    self.parentAddr      = parentAddr
    self.isServer        = 0
    self.readonly        = readonly
    self.parentDirectory = parentDirectory
    self.packer          = xdrlib.Packer()
    self.unpacker        = xdrlib.Unpacker('')
    self.stopCmd         = pickle.dumps(('stop',))
    self.writeLogLine('Greetings')
    self.connectParent(self.parentAddr, self.parentDirectory)
    if load: self.load()
    if autoShutdown and useThreads:
      atexit.register(self.shutdown)
    self.writeLogLine('SERVER: Last access '+str(self.lastAccess))
    return

  def __getstate__(self):
    '''Remove any parent socket object, the XDR translators, and the log file from the dictionary before pickling'''
    self.writeLogLine('Pickling RDict')
    d = self.__dict__.copy()
    if 'parent'    in d: del d['parent']
    if 'saveTimer' in d: del d['saveTimer']
    if '_setCommandLine' in d: del d['_setCommandLine']
    del d['packer']
    del d['unpacker']
    del d['logFile']
    return d

  def __setstate__(self, d):
    '''Reconnect the parent socket object, recreate the XDR translators and reopen the log file after unpickling'''
    self.logFile  = open('RDict.log', 'a')
    self.writeLogLine('Unpickling RDict')
    self.__dict__.update(d)
    import xdrlib
    self.packer   = xdrlib.Packer()
    self.unpacker = xdrlib.Unpacker('')
    self.connectParent(self.parentAddr, self.parentDirectory)
    return

  def setupLogFile(self, filename = 'RDict.log'):
    if not self.logFile is None:
      self.logFile.close()
    if os.path.isfile(filename) and os.stat(filename).st_size > 10*1024*1024:
      if os.path.isfile(filename+'.bkp'):
        os.remove(filename+'.bkp')
      os.rename(filename, filename+'.bkp')
      self.logFile = open(filename, 'w')
    else:
      self.logFile = open(filename, 'a')
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
      length = length + self.send()
    return length

  def getType(self, key):
    '''Checks for the key locally, and if not found consults the parent. Returns the Arg object or None if not found.'''
    try:
      value = dict.__getitem__(self, key)
      self.writeLogLine('getType: Getting local type for '+key+' '+str(value))
      return value
    except KeyError:
      pass
    if self.parent:
      return self.send(key)
    return None

  def dict_has_key(self, key):
    """Utility to check whether the key is present in the dictionary without RDict side-effects."""
    return key in dict(self)

  def __getitem__(self, key):
    '''Checks for the key locally, and if not found consults the parent. Returns the value of the Arg.
       - If the value has not been set, the user will be prompted for input'''
    if self.dict_has_key(key):
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
        try:
          value = arg.getValue()
        except AttributeError as e:
          self.writeLogLine('__getitem__: Parent had invalid entry: '+str(e))
          arg   = nargs.Arg(key)
          value = arg.getValue()
        self.writeLogLine('__getitem__: Setting parent value '+str(value))
        self.send(key, value, operation = '__setitem__')
        return value
    else:
      self.writeLogLine('__getitem__: Setting local type for '+key)
      dict.__setitem__(self, key, nargs.Arg(key))
      #self.save()
    self.writeLogLine('__getitem__: Setting local value for '+key)
    return dict.__getitem__(self, key).getValue()

  def setType(self, key, value, forceLocal = 0):
    '''Checks for the key locally, and if not found consults the parent. Sets the type for this key.
       - If a value for the key already exists, it is converted to the new type'''
    if not isinstance(value, nargs.Arg):
      raise TypeError('An argument type must be a subclass of Arg')
    value.setKey(key)
    if forceLocal or self.parent is None or self.dict_has_key(key):
      if self.dict_has_key(key):
        v = dict.__getitem__(self, key)
        if v.isValueSet():
          try:
            value.setValue(v.getValue())
          except TypeError:
            print(value.__class__.__name__[3:])
            print('-----------------------------------------------------------------------')
            print('Warning! Incorrect argument type specified: -'+str(key)+'='+str(v.getValue())+' - expecting type '+value.__class__.__name__[3:]+'.')
            print('-----------------------------------------------------------------------')
            pass
      dict.__setitem__(self, key, value)
      #self.save()
    else:
      return self.send(key, value)
    return

  def __setitem__(self, key, value):
    '''Checks for the key locally, and if not found consults the parent. Sets the value of the Arg.'''
    if not self.dict_has_key(key):
      if not self.parent is None:
        return self.send(key, value)
      else:
        dict.__setitem__(self, key, nargs.Arg(key))
    dict.__getitem__(self, key).setValue(value)
    self.writeLogLine('__setitem__: Set value for '+key+' to '+str(dict.__getitem__(self, key)))
    #self.save()
    return

  def __delitem__(self, key):
    '''Checks for the key locally, and if not found consults the parent. Deletes the Arg completely.'''
    if self.dict_has_key(key):
      dict.__delitem__(self, key)
      #self.save()
    elif not self.parent is None:
      self.send(key)
    return

  def clear(self):
    '''Clears both the local and parent dictionaries'''
    if dict.__len__(self):
      dict.clear(self)
      #self.save()
    if not self.parent is None:
      self.send()
    return

  def __contains__(self, key):
    '''Checks for the key locally, and if not found consults the parent. Then checks whether the value has been set'''
    if self.dict_has_key(key):
      if dict.__getitem__(self, key).isValueSet():
        self.writeLogLine('has_key: Have value for '+key)
      else:
        self.writeLogLine('has_key: Do not have value for '+key)
      return dict.__getitem__(self, key).isValueSet()
    elif not self.parent is None:
      return self.send(key)
    return 0

  def get(self, key, default=None):
    if key in self:
      return self.__getitem__(key)
    else:
      return default

  def hasType(self, key):
    '''Checks for the key locally, and if not found consults the parent. Then checks whether the type has been set'''
    if self.dict_has_key(key):
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
    keyList = [key for key in dict.keys(self) if dict.__getitem__(self, key).isValueSet()]
    if not self.parent is None:
      keyList.extend(self.send())
    return keyList

  def types(self):
    '''Returns the list of keys for which types are defined in both the local and parent dictionaries'''
    keyList = dict.keys(self)
    if not self.parent is None:
      keyList.extend(self.send())
    return keyList

  def update(self, d):
    '''Update the dictionary with the contents of d'''
    for k in d:
      self[k] = d[k]
    return

  def updateTypes(self, d):
    '''Update types locally, which is equivalent to the dict.update() method'''
    return dict.update(self, d)

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

    if isinstance(args, list):
      for arg in args:
        (key, value) = nargs.Arg.parseArgument(arg)
        self.insertArg(key, value, arg)
    elif hasattr(args, 'keys'):
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
    '''Return True if this RDict has a parent dictionary'''
    return not self.parent is None

  def getServerAddr(self, dir):
    '''Read the server socket address (in pickled form) from a file, usually RDict.loc
       - If we fail to connect to the server specified in the file, we spawn it using startServer()'''
    filename = os.path.join(dir, self.addrFilename)
    if not os.path.exists(filename):
      self.startServer(filename)
    if not os.path.exists(filename):
      raise RuntimeError('Server address file does not exist: '+filename)
    try:
      f    = open(filename, 'r')
      addr = pickle.load(f)
      f.close()
      return addr
    except Exception as e:
      self.writeLogLine('CLIENT: Exception during server address determination: '+str(e.__class__)+': '+str(e))
    raise RuntimeError('Could not get server address in '+filename)

  def writeServerAddr(self, server):
    '''Write the server socket address (in pickled form) to a file, usually RDict.loc.'''
    f = open(self.addrFilename, 'w')
    pickle.dump(server.server_address, f)
    f.close()
    self.writeLogLine('SERVER: Wrote lock file '+os.path.abspath(self.addrFilename))
    return

  def startServer(self, addrFilename):
    '''Spawn a new RDict server in the parent directory'''
    import RDict # Need this to locate server script
    import sys
    import time
    import sysconfig

    self.writeLogLine('CLIENT: Spawning a new server with lock file '+os.path.abspath(addrFilename))
    if os.path.exists(addrFilename):
      os.remove(addrFilename)
    oldDir      = os.getcwd()
    source      = os.path.join(os.path.dirname(os.path.abspath(sys.modules['RDict'].__file__)), 'RDict.py')
    interpreter = os.path.join(sysconfig.get_config_var('BINDIR'), sysconfig.get_config_var('PYTHON'))
    if not os.path.isfile(interpreter):
      interpreter = 'python'
    os.chdir(os.path.dirname(addrFilename))
    self.writeLogLine('CLIENT: Executing '+interpreter+' '+source+' server"')
    try:
      os.spawnvp(os.P_NOWAIT, interpreter, [interpreter, source, 'server'])
    except:
      self.writeLogLine('CLIENT: os.spawnvp failed.\n \
      This is a typical problem on CYGWIN systems.  If you are using CYGWIN,\n \
      you can fix this problem by running /bin/rebaseall.  If you do not have\n \
      this program, you can install it with the CYGWIN installer in the package\n \
      Rebase, under the category System.  You must run /bin/rebaseall after\n \
      turning off all cygwin services -- in particular sshd, if any such services\n \
      are running.  For more information about rebase, go to http://www.cygwin.com')
      print('\n \
      This is a typical problem on CYGWIN systems.  If you are using CYGWIN,\n \
      you can fix this problem by running /bin/rebaseall.  If you do not have\n \
      this program, you can install it with the CYGWIN installer in the package\n \
      Rebase, under the category System.  You must run /bin/rebaseall after\n \
      turning off all cygwin services -- in particular sshd, if any such services\n \
      are running.  For more information about rebase, go to http://www.cygwin.com\n')
      raise
    os.chdir(oldDir)
    timeout = 1
    for i in range(10):
      time.sleep(timeout)
      timeout *= 2
      if timeout > 100: timeout = 100
      if os.path.exists(addrFilename): return
    self.writeLogLine('CLIENT: Could not start server')
    return

  def connectParent(self, addr, dir):
    '''Try to connect to a parent RDict server
       - If addr and dir are both None, this operation fails
       - If addr is None, check for an address file in dir'''
    if addr is None:
      if dir is None: return 0
      addr = self.getServerAddr(dir)

    import socket
    import errno
    connected = 0
    s         = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    timeout   = 1
    for i in range(10):
      try:
        self.writeLogLine('CLIENT: Trying to connect to '+str(addr))
        s.connect(addr)
        connected = 1
        break
      except socket.error as e:
        self.writeLogLine('CLIENT: Failed to connect: '+str(e))
        if e[0] == errno.ECONNREFUSED:
          try:
            import time
            time.sleep(timeout)
            timeout *= 2
            if timeout > 100: timeout = 100
          except KeyboardInterrupt:
            break
          # Try to spawn parent
          if dir:
            filename = os.path.join(dir, self.addrFilename)
            if os.path.isfile(filename):
              os.remove(filename)
            self.startServer(filename)
      except Exception as e:
        self.writeLogLine('CLIENT: Failed to connect: '+str(e.__class__)+': '+str(e))
    if not connected:
      self.writeLogLine('CLIENT: Failed to connect to parent')
      return 0
    self.parent = s
    self.writeLogLine('CLIENT: Connected to '+str(self.parent))
    return 1

  def sendPacket(self, s, packet, source = 'Unknown', isPickled = 0):
    '''Pickle the input packet. Send first the size of the pickled string in 32-bit integer, and then the string itself'''
    self.writeLogLine(source+': Sending packet '+str(packet))
    if isPickled:
      p = packet
    else:
      p = pickle.dumps(packet)
    self.packer.reset()
    self.packer.pack_uint(len(p))
    if hasattr(s, 'write'):
      s.write(self.packer.get_buffer())
      s.write(p)
    else:
      s.sendall(self.packer.get_buffer())
      s.sendall(p)
    self.writeLogLine(source+': Sent packet')
    return

  def recvPacket(self, s, source = 'Unknown'):
    '''Receive first the size of the pickled string in a 32-bit integer, and then the string itself. Return the unpickled object'''
    self.writeLogLine(source+': Receiving packet')
    if hasattr(s, 'read'):
      s.read(4)
      value = pickle.load(s)
    else:
      # I probably need to check that it actually read these 4 bytes
      self.unpacker.reset(s.recv(4))
      length    = self.unpacker.unpack_uint()
      objString = ''
      while len(objString) < length:
        objString += s.recv(length - len(objString))
      value = pickle.loads(objString)
    self.writeLogLine(source+': Received packet '+str(value))
    return value

  def send(self, key = None, value = None, operation = None):
    '''Send a request to the parent'''
    import inspect

    objString = ''
    for i in range(3):
      try:
        packet = []
        if operation is None:
          operation = inspect.stack()[1][3]
        packet.append(operation)
        if not key is None:
          packet.append(key)
          if not value is None:
            packet.append(value)
        self.sendPacket(self.parent, tuple(packet), source = 'CLIENT')
        response = self.recvPacket(self.parent, source = 'CLIENT')
        break
      except IOError as e:
        self.writeLogLine('CLIENT: IOError '+str(e))
        if e.errno == 32:
          self.connectParent(self.parentAddr, self.parentDirectory)
      except Exception as e:
        self.writeLogLine('CLIENT: Exception '+str(e)+' '+str(e.__class__))
    try:
      if isinstance(response, Exception):
        self.writeLogLine('CLIENT: Got an exception '+str(response))
        raise response
      else:
        self.writeLogLine('CLIENT: Received value '+str(response)+' '+str(type(response)))
    except UnboundLocalError:
      self.writeLogLine('CLIENT: Could not unpickle response')
      response  = None
    return response

  def serve(self):
    '''Start a server'''
    import socket
    import SocketServer # novermin

    if not useThreads:
      raise RuntimeError('Cannot run a server if threads are disabled')

    class ProcessHandler(SocketServer.StreamRequestHandler):
      def handle(self):
        import time

        self.server.rdict.lastAccess = time.time()
        self.server.rdict.writeLogLine('SERVER: Started new handler')
        while 1:
          try:
            value = self.server.rdict.recvPacket(self.rfile, source = 'SERVER')
          except EOFError as e:
            self.server.rdict.writeLogLine('SERVER: EOFError receiving packet '+str(e)+' '+str(e.__class__))
            return
          except Exception as e:
            self.server.rdict.writeLogLine('SERVER: Error receiving packet '+str(e)+' '+str(e.__class__))
            self.server.rdict.sendPacket(self.wfile, e, source = 'SERVER')
            continue
          if value[0] == 'stop': break
          try:
            response = getattr(self.server.rdict, value[0])(*value[1:])
          except Exception as e:
            self.server.rdict.writeLogLine('SERVER: Error executing operation '+str(e)+' '+str(e.__class__))
            self.server.rdict.sendPacket(self.wfile, e, source = 'SERVER')
          else:
            self.server.rdict.sendPacket(self.wfile, response, source = 'SERVER')
        return

    # check if server is running
    if os.path.exists(self.addrFilename):
      rdict     = RDict(parentDirectory = '.')
      hasParent = rdict.hasParent()
      del rdict
      if hasParent:
        self.writeLogLine('SERVER: Another server is already running')
        raise RuntimeError('Server already running')

    # Daemonize server
    self.writeLogLine('SERVER: Daemonizing server')
    if os.fork(): # Launch child
      os._exit(0) # Kill off parent, so we are not a process group leader and get a new PID
    os.setsid()   # Set session ID, so that we have no controlling terminal
    # We choose to leave cwd at RDict.py: os.chdir('/') # Make sure root directory is not on a mounted drive
    os.umask(0o77) # Fix creation mask
    for i in range(3): # Crappy stopgap for closing descriptors
      try:
        os.close(i)
      except OSError as e:
        if e.errno != errno.EBADF:
          raise RuntimeError('Could not close default descriptor '+str(i))

    # wish there was a better way to get a usable socket
    self.writeLogLine('SERVER: Establishing socket server')
    basePort = 8000
    flag     = 'nosocket'
    p        = 1
    while p < 1000 and flag == 'nosocket':
      try:
        server = SocketServer.ThreadingTCPServer((socket.gethostname(), basePort+p), ProcessHandler)
        flag   = 'socket'
      except Exception as e:
        p = p + 1
    if flag == 'nosocket':
      p = 1
      while p < 1000 and flag == 'nosocket':
        try:
          server = SocketServer.ThreadingTCPServer(('localhost', basePort+p), ProcessHandler)
          flag   = 'socket'
        except Exception as e:
          p = p + 1
    if flag == 'nosocket':
      self.writeLogLine('SERVER: Could not established socket server on port '+str(basePort+p))
      raise RuntimeError('Cannot get available socket')
    self.writeLogLine('SERVER: Established socket server on port '+str(basePort+p))

    self.isServer = 1
    self.writeServerAddr(server)
    self.serverShutdown(os.getpid())

    server.rdict = self
    self.writeLogLine('SERVER: Started server')
    server.serve_forever()
    return

  def load(self):
    '''Load the saved dictionary'''
    if not self.parentDirectory is None and os.path.samefile(os.getcwd(), self.parentDirectory):
      return
    self.saveFilename = os.path.abspath(self.saveFilename)
    if os.path.exists(self.saveFilename):
      try:
        dbFile = open(self.saveFilename, 'rb')
        data   = pickle.load(dbFile)
        self.updateTypes(data)
        dbFile.close()
        self.writeLogLine('Loaded dictionary from '+self.saveFilename)
      except Exception as e:
        self.writeLogLine('Problem loading dictionary from '+self.saveFilename+'\n--> '+str(e))
    else:
      self.writeLogLine('No dictionary to load in this file: '+self.saveFilename)
    return

  def save(self, force = 1):
    '''Save the dictionary after 5 seconds, ignoring all subsequent calls until the save
       - Giving force = True will cause an immediate save'''
    if self.readonly: return
    if force:
      self.saveTimer = None
      # This should be a critical section
      dbFile = open(self.saveFilename, 'wb')
      data   = dict([i for i in self.localitems() if not i[1].getTemporary()])
      pickle.dump(data, dbFile)
      dbFile.close()
      self.writeLogLine('Saved local dictionary to '+os.path.abspath(self.saveFilename))
    elif not self.saveTimer:
      import threading
      self.saveTimer = threading.Timer(5, self.save, [], {'force': 1})
      self.saveTimer.setDaemon(1)
      self.saveTimer.start()
    return

  def shutdown(self):
    '''Shutdown the dictionary, writing out changes and notifying parent'''
    if self.saveTimer:
      self.saveTimer.cancel()
      self.save(force = 1)
    if self.isServer and os.path.isfile(self.addrFilename):
      os.remove(self.addrFilename)
    if not self.parent is None:
      self.sendPacket(self.parent, self.stopCmd, isPickled = 1)
      self.parent.close()
      self.parent = None
    self.writeLogLine('Shutting down')
    self.logFile.close()
    return

  def serverShutdown(self, pid, delay = shutdownDelay):
    if self.shutdownTimer is None:
      import threading

      self.shutdownTimer = threading.Timer(delay, self.serverShutdown, [pid], {'delay': 0})
      self.shutdownTimer.setDaemon(1)
      self.shutdownTimer.start()
      self.writeLogLine('SERVER: Set shutdown timer for process '+str(pid)+' at '+str(delay)+' seconds')
    else:
      try:
        import signal
        import time

        idleTime = time.time() - self.lastAccess
        self.writeLogLine('SERVER: Last access '+str(self.lastAccess))
        self.writeLogLine('SERVER: Idle time '+str(idleTime))
        if idleTime < RDict.shutdownDelay:
          self.writeLogLine('SERVER: Extending shutdown timer for '+str(pid)+' by '+str(RDict.shutdownDelay - idleTime)+' seconds')
          self.shutdownTimer = None
          self.serverShutdown(pid, RDict.shutdownDelay - idleTime)
        else:
          self.writeLogLine('SERVER: Killing server '+str(pid))
          os.kill(pid, signal.SIGTERM)
      except Exception as e:
        self.writeLogLine('SERVER: Exception killing server: '+str(e))
    return

if __name__ ==  '__main__':
  import sys
  try:
    if len(sys.argv) < 2:
      print('RDict.py [server | client | clear | insert | remove] [parent]')
    else:
      action = sys.argv[1]
      parent = None
      if len(sys.argv) > 2:
        if not sys.argv[2] == 'None': parent = sys.argv[2]
      if action == 'server':
        RDict(parentDirectory = parent).serve()
      elif action == 'client':
        print('Entries in server dictionary')
        rdict = RDict(parentDirectory = parent)
        for key in rdict.types():
          if not key.startswith('cacheKey') and not key.startswith('stamp-'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'cacheClient':
        print('Cache entries in server dictionary')
        rdict = RDict(parentDirectory = parent)
        for key in rdict.types():
          if key.startswith('cacheKey'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'stampClient':
        print('Stamp entries in server dictionary')
        rdict = RDict(parentDirectory = parent)
        for key in rdict.types():
          if key.startswith('stamp-'):
            print(str(key)+' '+str(rdict.getType(key)))
      elif action == 'clear':
        print('Clearing all dictionaries')
        RDict(parentDirectory = parent).clear()
      elif action == 'insert':
        rdict = RDict(parentDirectory = parent)
        rdict[sys.argv[3]] = sys.argv[4]
      elif action == 'remove':
        rdict = RDict(parentDirectory = parent)
        del rdict[sys.argv[3]]
      else:
        sys.exit('Unknown action: '+action)
  except Exception as e:
    import traceback
    print(traceback.print_tb(sys.exc_info()[2]))
    sys.exit(str(e))
  sys.exit(0)
