ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I.. $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o 
WSOURCEC = itfuncf.c itcreatef.c iterativf.c itresf.c
WOBJS    = itfuncf.o itcreatef.o iterativf.o itresf.o
LIBBASE  = libpetscksp
LINCLUDE = 
DIRS     =
MANSEC   = 4

include $(ITOOLSDIR)/bmake/$(PARCH)

