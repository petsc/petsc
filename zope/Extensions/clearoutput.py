def clear():
	petscdir = os.environ["PETSC_DIR"]
	file = petscdir+"/zope/Extensions/bufupdate"
	f = open(file, "w+")
	f.close()

clear()
