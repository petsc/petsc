PETSC_DIR = ../../..

ALL: lib

CFLAGS	 = $(OPT) $(PETSC_INCLUDE) -I../..  $(PCONF)\
	   $(CONF) -D__DIR__='"fortran/auto/"'

SOURCEC	 = asmf.c axisf.c axpyf.c bjacobif.c cgf.c chebyf.c cmeshf.c \
	cookief.c da2f.c dacornf.c daghostf.c dagtolf.c daltogf.c \
	daltolf.c daviewf.c dbufff.c dclearf.c dcoorf.c dflushf.c \
	dgcoorf.c dgpausef.c dlinef.c dlinegwf.c dlinewf.c dmousef.c \
	dpausef.c dpointf.c dpointsf.c drawvf.c drectf.c dsclearf.c \
	dsetpausef.c dsflushf.c dtextgsf.c dtextsf.c dtrif.c dviewpf.c \
	eisenf.c flushf.c fpf.c gcookief.c gmresf.c iluf.c indexf.c \
	iscompf.c itclf.c itcreatef.c itfuncf.c itresf.c lgf.c \
	luf.c  matrixf.c mgf.c mgfuncf.c mhasf.c mtrf.c \
	pcnullf.c pcsetf.c preconf.c pseudof.c psleepf.c randomf.c \
	richf.c sda2f.c shellf.c signalf.c slesf.c snesf.c snesmfjf.c \
	snesutf.c sorf.c sortdf.c sortif.c sortipf.c stridef.c \
	trf.c tsf.c tsregf.c umlsf.c umtrf.c vectorf.c vgridf.c \
	viewf.c vinvf.c vscatf.c zerodiagf.c

SOURCEF	 =
SOURCEH	 = 
OBJSC	 = asmf.o axisf.o axpyf.o bjacobif.o cgf.o chebyf.o cmeshf.o \
	cookief.o da2f.o dacornf.o daghostf.o dagtolf.o daltogf.o \
	daltolf.o daviewf.o dbufff.o dclearf.o dcoorf.o dflushf.o \
	dgcoorf.o dgpausef.o dlinef.o dlinegwf.o dlinewf.o dmousef.o \
	dpausef.o dpointf.o dpointsf.o drawvf.o drectf.o dsclearf.o \
	dsetpausef.o dsflushf.o dtextgsf.o dtextsf.o dtrif.o dviewpf.o \
	eisenf.o flushf.o fpf.o gcookief.o gmresf.o iluf.o indexf.o \
	iscompf.o itclf.o itcreatef.o itfuncf.o itresf.o lgf.o \
	luf.o matrixf.o mgf.o mgfuncf.o mhasf.o mtrf.o \
	pcnullf.o pcsetf.o preconf.o pseudof.o psleepf.o randomf.o \
	richf.o sda2f.o shellf.o signalf.o slesf.o snesf.o snesmfjf.o \
	snesutf.o sorf.o sortdf.o sortif.o sortipf.o stridef.o \
	trf.o tsf.o tsregf.o umlsf.o umtrf.o vectorf.o vgridf.o \
	viewf.o vinvf.o vscatf.o zerodiagf.o

OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = 1

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

