# $Id: makefile,v 1.36 1997/10/09 15:10:19 balay Exp balay $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 

SOURCEC	 = aodataf.c aof.c asmf.c axisf.c axpyf.c bjacobif.c blockf.c cgtypef.c \
        chebyf.c cookief.c da2f.c dacornf.c daghostf.c dagtolf.c daltogf.c \
        daltolf.c dbufff.c dclearf.c dcoorf.c dflushf.c dfvecf.c dgcoorf.c \
        dgpausef.c dlinef.c dlinegwf.c dlinewf.c dmousef.c dpausef.c dpointf.c \
        dpointsf.c drawf.c drectf.c dscatterf.c dsclearf.c dsetpausef.c \
        dsflushf.c dtextgsf.c dtextsf.c dtrif.c dviewpf.c eigef.c \
        eisenf.c fdmatrixf.c flushf.c fpf.c gcookief.c gmpref.c \
        gmres2f.c griddataf.c gridf.c gvecf.c gveckspf.c gvecsnesf.c \
        histsf.c iluf.c indexf.c inheritf.c iscoloringf.c isdifff.c \
        isltogf.c itclf.c itcreatef.c itfuncf.c itresf.c lgf.c \
        luf.c matrixf.c memf.c mgf.c mgfuncf.c mhasf.c \
        mpiuf.c mtrf.c optionsf.c pbvecf.c pcnullf.c pcsetf.c \
        petscpvodef.c plogf.c plogmpef.c posindepf.c preconf.c psleepf.c \
        richscalef.c shellf.c signalf.c slesf.c snesf.c snesmfjf.c \
        snesutf.c sorf.c sortdf.c sortif.c sortipf.c stridef.c \
        trf.c tsf.c tsregf.c umlsf.c umtrf.c vectorf.c \
        viewf.c vinvf.c vrmlmeshf.c vscatf.c zerodiagf.c


SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = indexf.o stridef.o blockf.o iscoloringf.o isltogf.o \
	vinvf.o vscatf.o vectorf.o itclf.o itfuncf.o itcreatef.o \
	itresf.o eigef.o cgtypef.o gmres2f.o gmpref.o chebyf.o \
	richscalef.o signalf.o fpf.o mtrf.o mpiuf.o sortif.o \
	sortdf.o sortipf.o psleepf.o gcookief.o cookief.o memf.o \
	preconf.o pcsetf.o pcnullf.o sorf.o luf.o \
	bjacobif.o mgf.o mgfuncf.o eisenf.o iluf.o asmf.o matrixf.o \
	mhasf.o fdmatrixf.o shellf.o axpyf.o zerodiagf.o slesf.o drawf.o \
	dlinef.o dlinewf.o dlinegwf.o dtextsf.o dtextgsf.o dpointf.o \
	dpointsf.o dviewpf.o dcoorf.o dpausef.o dgpausef.o dgcoorf.o \
	dbufff.o dsetpausef.o dflushf.o dsflushf.o dclearf.o dsclearf.o \
	drectf.o dtrif.o dmousef.o axisf.o lgf.o dscatterf.o \
	tsf.o tsregf.o posindepf.o dfvecf.o viewf.o flushf.o snesf.o \
	snesmfjf.o snesutf.o trf.o umtrf.o umlsf.o da2f.o daghostf.o \
	dacornf.o dagtolf.o daltolf.o daltogf.o plogf.o plogmpef.o \
	aof.o gvecf.o gridf.o griddataf.o gveckspf.o gvecsnesf.o \
	vrmlmeshf.o petscpvodef.o 
OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = 1
LOCDIR   = src/fortran/auto/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

