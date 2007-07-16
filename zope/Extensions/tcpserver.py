import SocketServer
from socket import *
from urllib import *
import string
from os import fdopen
import threading
from re import search

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
			if infocheckRE :
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
				gs += end
				glast += end
				if not infocheck:
					info += end
				if not errorcheck:
					error += end
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
	thread.run()
	return
	#serv.serve_forever()

def quickreturn(var):
	var = 2;
	return

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

def cleargs():
	global gs
	gs = startmsg
	global glast
	glast = startmsg
	global info
	info = startmsg
	global error
	error = startmsg
	return

if __name__ == '__main__':
	runserver()
