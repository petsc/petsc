# $Id: makefile,v 1.8 2001/05/04 22:03:27 buschelm Exp $ 
CXX       = cl
CXXFLAGS  = 
CXX_FLAGS = -nologo -GX -GR -MD $(CXXFLAGS)

# Note: under WindowsNT (2000) the DOS shell is named cmd
RM        = cmd /C del
# Note: under Windows9x the DOS shell is named command
# RM        = command /C del

SOURCES = petsccompilerfe.cpp petscbccfe.cpp petscclfe.cpp petsciclfe.cpp \
          petscarchiverfe.cpp petsclibfe.cpp petsctlibfe.cpp \
          petsccreate.cpp petsctoolfe.cpp petscfe.cpp

OBJECTS = petsccompilerfe.obj petscbccfe.obj petscclfe.obj petsciclfe.obj \
          petscarchiverfe.obj petsclibfe.obj petsctlibfe.obj \
          petsccreate.obj petsctoolfe.obj petscfe.obj

INCLUDEPATHS = -Iinclude

LIBS = Advapi32.lib

.SUFFIXES: .cpp .obj

.cpp.obj:
	$(CXX) $(CXX_FLAGS) -c $< $(INCLUDEPATHS)

win32fe: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -Fewin32fe.exe $(OBJECTS) $(LIBS)
	$(RM) $(OBJECTS)

clean:
	$(RM) $(OBJECTS) *~