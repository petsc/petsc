# $Id: makefile,v 1.41 1997/10/28 13:58:06 bsmith Exp balay $ 

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
        mpiuf.c mtrf.c partitionf.c pbvecf.c pcnullf.c pcsetf.c \
        petscpvodef.c plogf.c plogmpef.c pmetisf.c posindepf.c preconf.c psleepf.c \
        richscalef.c shellf.c signalf.c slesf.c snesf.c snesmfjf.c \
        snesutf.c sorf.c sortdf.c sortif.c sortipf.c stridef.c \
        trf.c tsf.c tsregf.c umlsf.c umtrf.c vectorf.c \
        viewf.c vinvf.c vrmlmeshf.c vscatf.c xopsf.c zerodiagf.c


SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = aodataf.o aof.o asmf.o axisf.o axpyf.o bjacobif.o blockf.o cgtypef.o \
        chebyf.o cookief.o da2f.o dacornf.o daghostf.o dagtolf.o daltogf.o \
        daltolf.o dbufff.o dclearf.o dcoorf.o dflushf.o dfvecf.o dgcoorf.o \
        dgpausef.o dlinef.o dlinegwf.o dlinewf.o dmousef.o dpausef.o dpointf.o \
        dpointsf.o drawf.o drectf.o dscatterf.o dsclearf.o dsetpausef.o \
        dsflushf.o dtextgsf.o dtextsf.o dtrif.o dviewpf.o eigef.o \
        eisenf.o fdmatrixf.o flushf.o fpf.o gcookief.o gmpref.o \
        gmres2f.o griddataf.o gridf.o gvecf.o gveckspf.o gvecsnesf.o \
        histsf.o iluf.o indexf.o inheritf.o iscoloringf.o isdifff.o \
        isltogf.o itclf.o itcreatef.o itfuncf.o itresf.o lgf.o \
        luf.o matrixf.o memf.o mgf.o mgfuncf.o mhasf.o \
        mpiuf.o mtrf.o partitionf.o pbvecf.o pcnullf.o pcsetf.o \
        petscpvodef.o plogf.o plogmpef.o pmetisf.o posindepf.o preconf.o psleepf.o \
        richscalef.o shellf.o signalf.o slesf.o snesf.o snesmfjf.o \
        snesutf.o sorf.o sortdf.o sortif.o sortipf.o stridef.o \
        trf.o tsf.o tsregf.o umlsf.o umtrf.o vectorf.o \
        viewf.o vinvf.o vrmlmeshf.o vscatf.o xopsf.o zerodiagf.o


OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = 1
LOCDIR   = src/fortran/auto/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

