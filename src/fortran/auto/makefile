#PETSC_DIR = ../../..

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 

SOURCEC	 = indexf.c stridef.c blockf.c iscoloringf.c isltogf.c \
	vinvf.c vscatf.c vectorf.c itclf.c itfuncf.c itcreatef.c \
	itresf.c eigef.c cgtypef.c gmres2f.c gmpref.c chebyf.c \
	richscalef.c signalf.c fpf.c mtrf.c mpiuf.c sortif.c \
	sortdf.c sortipf.c psleepf.c gcookief.c cookief.c memf.c \
	preconf.c pcsetf.c pcnullf.c sorf.c luf.c \
	bjacobif.c mgf.c mgfuncf.c eisenf.c iluf.c asmf.c matrixf.c \
	mhasf.c fdmatrixf.c shellf.c axpyf.c zerodiagf.c slesf.c drawf.c \
	dlinef.c dlinewf.c dlinegwf.c dtextsf.c dtextgsf.c dpointf.c \
	dpointsf.c dviewpf.c dcoorf.c dpausef.c dgpausef.c dgcoorf.c \
	dbufff.c dsetpausef.c dflushf.c dsflushf.c dclearf.c dsclearf.c \
	drectf.c dtrif.c dmousef.c axisf.c lgf.c dscatterf.c \
	tsf.c tsregf.c posindepf.c dfvecf.c viewf.c flushf.c snesf.c \
	snesmfjf.c snesutf.c trf.c umtrf.c umlsf.c da2f.c daghostf.c \
	dacornf.c dagtolf.c daltolf.c daltogf.c plogf.c plogmpef.c \
	aof.c gvecf.c gridf.c griddataf.c gveckspf.c gvecsnesf.c \
	rect1df.c rect2df.c

SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = indexf.o stridef.o blockf.o iscoloringf.o isltogf.o vinvf.o \
	vscatf.o vectorf.o itclf.o itfuncf.o itcreatef.o itresf.o eigef.o \
	cgtypef.o gmres2f.o gmpref.o chebyf.o richscalef.o signalf.o \
	fpf.o mtrf.o mpiuf.o sortif.o sortdf.o sortipf.o psleepf.o \
	gcookief.o cookief.o memf.o preconf.o pcsetf.o \
	pcnullf.o sorf.o luf.o bjacobif.o mgf.o mgfuncf.o eisenf.o \
	iluf.o asmf.o matrixf.o mhasf.o fdmatrixf.o shellf.o axpyf.o \
	zerodiagf.o slesf.o drawf.o dlinef.o dlinewf.o dlinegwf.o \
	dtextsf.o dtextgsf.o dpointf.o dpointsf.o dviewpf.o dcoorf.o \
	dpausef.o dgpausef.o dgcoorf.o dbufff.o dsetpausef.o dflushf.o \
	dsflushf.o dclearf.o dsclearf.o drectf.o dtrif.o dmousef.o  \
	axisf.o lgf.o dscatterf.o tsf.o tsregf.o posindepf.o dfvecf.o \
	viewf.o flushf.o snesf.o snesmfjf.o snesutf.o trf.o umtrf.o \
	umlsf.o da2f.o daghostf.o dacornf.o dagtolf.o daltolf.o daltogf.o \
	plogf.o plogmpef.o aof.o gvecf.o gridf.o griddataf.o gveckspf.o \
	gvecsnesf.o rect1df.o rect2df.o

OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = 1
LOCDIR   = src/fortran/auto/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

