# $Id: makefile,v 1.6 2001/04/18 00:36:41 buschelm Exp buschelm $ 
CXX       = cl
CXX_FLAGS = -nologo -GX -GR -MD
CXXFLAGS  =

# Note: under WindowsNT (2000) the DOS shell is named cmd
RM        = cmd /C del
# Note: under Windows9x the DOS shell is named command
# RM        = command /C del

SOURCES = petsccompilerfe.cpp petscbccfe.cpp petscclfe.cpp petsciclfe.cpp \
          petscarchiverfe.cpp petsclibfe.cpp petsctlibfe.cpp \
          petsccreate.cpp petsctoolfe.cpp petscfe.cpp

OBJECTS = petsccompilerfe.obj petscbccfe.obj petscclfe.obj petsciclfe.cpp \
          petscarchiverfe.obj petsclibfe.obj petsctlibfe.obj \
          petsccreate.obj petsctoolfe.obj petscfe.obj

INCLUDEPATHS = -Iinclude

win32fe: $(SOURCES)
	$(CXX) $(CXX_FLAGS) $(CXXFLAGS) -Fewin32fe.exe $(SOURCES) $(INCLUDEPATHS) 
	$(RM) $(OBJECTS)
