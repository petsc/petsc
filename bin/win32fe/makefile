# $Id: makefile,v 1.4 2001/04/17 18:23:48 buschelm Exp buschelm $ 
CXX       = cl
CXX_FLAGS = -nologo -GX -GR -MD
CXXFLAGS  =

# Note: under WindowsNT (2000) the DOS shell is named cmd
RM        = cmd /C del
# Note: under Windows9x the DOS shell is named command
# RM        = command /C del

SOURCES = petsccompilerfe.cpp petscbccfe.cpp petscclfe.cpp \
          petscarchiverfe.cpp petsclibfe.cpp petsctlibfe.cpp \
          petsccreate.cpp petsctoolfe.cpp petscfe.cpp

OBJECTS = petsccompilerfe.obj petscbccfe.obj petscclfe.obj \
          petscarchiverfe.obj petsclibfe.obj petsctlibfe.obj \
          petsccreate.obj petsctoolfe.obj petscfe.obj

INCLUDEPATHS = -Iinclude

win32fe: $(SOURCES)
	$(CXX) $(CXX_FLAGS) $(CXXFLAGS) -Fewin32fe.exe $(SOURCES) $(INCLUDEPATHS) -link setargv.obj
	$(RM) $(OBJECTS)
