# $Id:$ 
CXX       = cl
CXX_FLAGS = -nologo -GX -GR -MD
CXXFLAGS  =

# Note: under WindowsNT (2000) the DOS shell is named cmd
RM        = cmd /C del
# Note: under Windows9x the DOS shell is named command
# RM        = command /C del

SOURCES = petsccompilefe.cpp petscbccfe.cpp petscclfe.cpp \
          petscarchivefe.cpp petsclibfe.cpp petsctlibfe.cpp \
          petsccreate.cpp petsctoolfe.cpp petscfe.cpp

OBJECTS = petsccompilefe.obj petscbccfe.obj petscclfe.obj \
          petscarchivefe.obj petsclibfe.obj petsctlibfe.obj \
          petsccreate.obj petsctoolfe.obj petscfe.obj

INCLUDEPATHS = -Iinclude

win32fe.exe: petscfe.cpp $(SOURCES)
	$(CXX) $(CXX_FLAGS) $(CXXFLAGS) -Fewin32fe.exe $(SOURCES) $(INCLUDEPATHS)
	$(RM) $(OBJECTS)
