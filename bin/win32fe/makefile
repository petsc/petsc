# $Id: makefile,v 1.2 2001/04/11 07:49:14 buschelm Exp $ 
CXX       = cl
CXX_FLAGS = -nologo -GX -GR -MD
CXXFLAGS  =

# Note: under WindowsNT (2000) the DOS shell is named cmd
RM        = cmd /C del
# Note: under Windows9x the DOS shell is named command
# RM        = command /C del

SOURCES = compilerfe.cpp bccfe.cpp clfe.cpp \
          archiverfe.cpp libfe.cpp tlibfe.cpp \
          create.cpp toolfe.cpp fe.cpp

OBJECTS = compilerfe.obj bccfe.obj clfe.obj \
          archiverfe.obj libfe.obj tlibfe.obj \
          create.obj toolfe.obj fe.obj

INCLUDEPATHS = -Iinclude

win32fe: $(SOURCES)
	$(CXX) $(CXX_FLAGS) $(CXXFLAGS) -Fewin32fe.exe $(SOURCES) $(INCLUDEPATHS) -link setargv.obj
	$(RM) $(OBJECTS)
