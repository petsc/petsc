import SocketServer
from socket import *
from urllib import *
import string
import os
import fcntl
from os import fdopen
from re import search
import threading

# Strings to return to the Zope webpage
global gs
global glast
global info
global startmsg
global error
global sockfd
global status

# Is handler for information received on the socket
class RecHandler(SocketServer.StreamRequestHandler):
	def handle(self):
		global gs
		global glast
		global info
		global error
		global status
		infocheck = errorcheck = 1
		#Checks to see if this is the first time the strings are used
		if gs == startmsg: 
			gs = ""
		peer = self.client_address
		alias = gethostbyaddr(peer[0])
		#Used to distingush between seperate communications
		start = "\n======= %s %s ======\n\n" % (alias[0], peer[1])
		gs += start
		glast = start
		while 1:
			line = self.rfile.readline()
			linestrip = line.strip()
			joinline = "".join(line)
			# Check to see if the string is info or error output
			endinfo = search(r'\[[0-9]+\]', joinline)
			infocheckRE = (endinfo != None)
			errorinfo = linestrip.rfind("PETSC ERROR:")
			if joinline.rfind("<<<start>>>") >= 0:
				status = 1
				joinline = joinline.lstrip("<<<start>>>")
			if joinline.rfind("<<<end>>>") >= 0:
				status = 0
				joinline = joinline.lstrip("<<<end>>>")
			if infocheckRE:
				if infocheck:
					if info == startmsg:
						info = ""
					infocheck = 0
					info += start
				info += joinline.lstrip("<<info>>")
			elif errorinfo >= 0:
				if errorcheck:
					if error == startmsg:
						error = ""
					errorcheck = 0
					error += start
				error += joinline
			else:
				glast += joinline
				gs += joinline
			if not string.strip(line):
				#Ending tag for a communication
				end = "\n========== END ==========\n\n"
				msgdelim = "|MSGDELIM|"
				end += msgdelim
				gs += end
				glast += end
				petscdir = os.environ["PETSC_DIR"]
				file = petscdir + "/zope/Extensions/bufupdate"
				f = open(file, "a");
				fcntl.flock(f.fileno(), fcntl.LOCK_EX)
				if not infocheck:
					info += end
					delimstring = "2~/~/~"
					writestring = delimstring+""+info
					f.write(writestring)
					info = ""
				if not errorcheck:
					error += end
					delimstring = "3~/~/~"
					writestring = delimstring+""+error
					f.write(writestring)
					error = ""										            
				delimstring = "1~/~/~"
				writestring = delimstring+""+gs
				f.write(writestring)
				gs = ""
				fcntl.flock(f.fileno(), fcntl.LOCK_UN)
				f.close()
				break

#Start the server and intilize the global variables
def runserver():
	global gs
	global glast
	global info
	global error
	global startmsg
	global sockfd
	global status
	status = -1
	startmsg = "No Output"
	gs = glast = info = error = startmsg
	serv = SocketServer.ThreadingTCPServer(("", 9999), RecHandler)
	sockfd = serv.fileno()
	thread = threading.Thread(target=serv.serve_forever)
	thread.setDaemon(1)
	thread.start()
	return "TCP server started"

def writefd():
	global sockfd
	f = fdopen(sockfd, "w")
	f.write("hello world\n")

def getgs():
	return gs

def getglast():
	return glast

def getinfo():
	return info

def geterror():
	return error

def getstatus():
	global stauts
	if status == -1:
		return "No Programs have been started"
	elif status == 1:
		return "Active Program"
	else: 
		return "Not Active"

def update():
	petscdir = os.environ["PETSC_DIR"]
	file = petscdir+"/zope/Extensions/bufupdate"
	f = open(file, "r")
	fcntl.flock(f.fileno(), fcntl.LOCK_EX)
	out = f.read()
	fcntl.flock(f.fileno(), fcntl.LOCK_UN)
	f.close()
	return out

def clear():
	petscdir = os.environ["PETSC_DIR"]
	file = petscdir+"/zope/Extensions/bufupdate"
	f = open(file, "w+")
	f.close()

if __name__ == '__main__':
	runserver()
