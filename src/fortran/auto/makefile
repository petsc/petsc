PETSC_DIR = ../../..

ALL: lib

CFLAGS	 = $(OPT) $(PETSC_INCLUDE) -I../..  $(PCONF)\
	   $(CONF) -D__DIR__='"fortran/auto/"'

SOURCEC	 = archf.c asmf.c axisf.c axpyf.c bjacobif.c cgeigf.c chebyf.c cmeshf.c	 \
	   dacornf.c daghostf.c dagtolf.c daltogf.c daltolf.c daviewf.c dbufff.c \
	   dclearf.c dcoorf.c dflushf.c dgcoorf.c dgpausef.c dlinef.c dlinegwf.c \
	   dlinewf.c dmousef.c dpausef.c dpointf.c dpointsf.c drectf.c dsclearf.c \
	   dsetpausef.c dsflushf.c dtextgsf.c dtextsf.c dtrif.c dviewpf.c eisenf.c \
	   fpf.c gcookief.c gmresf.c iccbsf.c iluf.c indexf.c iscompf.c itclf.c \
	   itcreatef.c itfuncf.c itresf.c lgf.c luf.c makefile matrixf.c mgf.c \
	   mgfuncf.c mhasf.c mtrf.c pcnullf.c pcsetf.c plogf.c preconf.c psleepf.c \
	   randomf.c richf.c shellf.c signalf.c slesf.c snesf.c snesmfjf.c snesutf.c \
	   sorf.c sortdf.c sortif.c sortipf.c stridef.c trf.c umlsf.c umtrf.c \
	   vectorf.c vinvf.c vscatf.c zerodiagf.c

SOURCEF	 =
SOURCEH	 = 
OBJSC	 = archf.o asmf.o axisf.o axpyf.o bjacobif.o cgeigf.o chebyf.o cmeshf.o	 \
	   dacornf.o daghostf.o dagtolf.o daltogf.o daltolf.o daviewf.o dbufff.o \
	   dclearf.o dcoorf.o dflushf.o dgcoorf.o dgpausef.o dlinef.o dlinegwf.o \
	   dlinewf.o dmousef.o dpausef.o dpointf.o dpointsf.o drectf.o dsclearf.o \
	   dsetpausef.o dsflushf.o dtextgsf.o dtextsf.o dtrif.o dviewpf.o eisenf.o \
	   fpf.o gcookief.o gmresf.o iccbsf.o iluf.o indexf.o iscompf.o itclf.o \
	   itcreatef.o itfuncf.o itresf.o lgf.o luf.o makefile matrixf.o mgf.o \
	   mgfuncf.o mhasf.o mtrf.o pcnullf.o pcsetf.o plogf.o preconf.o psleepf.o \
	   randomf.o richf.o shellf.o signalf.o slesf.o snesf.o snesmfjf.o snesutf.o \
	   sorf.o sortdf.o sortif.o sortipf.o stridef.o trf.o umlsf.o umtrf.o \
	   vectorf.o vinvf.o vscatf.o zerodiagf.o

OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = 1

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

