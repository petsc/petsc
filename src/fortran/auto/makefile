# $Id: makefile,v 1.111 2000/08/03 14:22:07 balay Exp balay $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = aof.c asmf.c axisf.c axpyf.c bjacobif.c blockf.c cputimef.c cgtypef.c \
        chebyf.c compositef.c cookief.c da2f.c dacornf.c daghostf.c dagtolf.c daltogf.c \
        daltolf.c dbufff.c dclearf.c dcoorf.c dflushf.c dgcoorf.c \
        dgpausef.c dlinef.c dlinegwf.c dlinewf.c dmousef.c dpausef.c dpointf.c \
        dpointsf.c drawf.c drectf.c dscatterf.c dsclearf.c dsetpausef.c \
        dsflushf.c dtextgsf.c dtextsf.c dtrif.c dviewpf.c eigef.c \
        eisenf.c fdmatrixf.c flushf.c gcookief.c getcolvf.c gmpref.c \
        gmres2f.c drawvf.c histsf.c iluf.c indexf.c inheritf.c isdifff.c \
        isltogf.c itclf.c itcreatef.c itfuncf.c itresf.c lgf.c \
        luf.c matrixf.c memf.c mgf.c mgfuncf.c mhasf.c mpibaijf.c \
        mtrf.c partitionf.c pbvecf.c pcsetf.c pcslesf.c \
        petscpvodef.c plogf.c plogmpef.c pmetisf.c posindepf.c preconf.c psleepf.c \
        richscalef.c shellf.c signalf.c slesf.c snesf.c snesmfjf.c \
        snesutf.c sorf.c sortdf.c sortif.c sortipf.c stridef.c \
        tsf.c tsregf.c vectorf.c dagetarrayf.c gcreatevf.c \
        vinvf.c vscatf.c zerodiagf.c duplf.c gr1f.c dapff.c matregf.c \
	cmeshf.c optionsf.c snesmfjdeff.c combf.c redundantf.c iterativf.c \
	matnullf.c sbaijf.c mpisbaijf.c

SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = aof.o asmf.o axisf.o axpyf.o bjacobif.o blockf.o cputimef.o cgtypef.o \
        chebyf.o compositef.o cookief.o da2f.o dacornf.o daghostf.o dagtolf.o daltogf.o \
        daltolf.o dbufff.o dclearf.o dcoorf.o dflushf.o dgcoorf.o \
        dgpausef.o dlinef.o dlinegwf.o dlinewf.o dmousef.o dpausef.o dpointf.o \
        dpointsf.o drawf.o drectf.o dscatterf.o dsclearf.o dsetpausef.o \
        dsflushf.o dtextgsf.o dtextsf.o dtrif.o dviewpf.o eigef.o \
        eisenf.o fdmatrixf.o flushf.o gcookief.o getcolvf.o gmpref.o \
        gmres2f.o drawvf.o histsf.o iluf.o indexf.o inheritf.o isdifff.o \
        isltogf.o itclf.o itcreatef.o itfuncf.o itresf.o lgf.o \
        luf.o matrixf.o memf.o mgf.o mgfuncf.o mhasf.o mpibaijf.o \
        mtrf.o partitionf.o pbvecf.o pcsetf.o pcslesf.o \
        petscpvodef.o plogf.o plogmpef.o pmetisf.o posindepf.o preconf.o psleepf.o \
        richscalef.o shellf.o signalf.o slesf.o snesf.o snesmfjf.o \
        snesutf.o sorf.o sortdf.o sortif.o sortipf.o stridef.o \
        tsf.o tsregf.o vectorf.o dagetarrayf.o gcreatevf.o \
        vinvf.o vscatf.o zerodiagf.o duplf.o gr1f.o dapff.o matregf.o \
	cmeshf.o optionsf.o snesmfjdeff.o combf.o redundantf.o iterativf.o \
	matnullf.o sbaijf.o mpisbaijf.o


OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = Sys
LOCDIR   = src/fortran/auto/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

fixfortran:
	@echo "Fixing pointers in src/fortran/auto"
	-@for FILENAME in ${SOURCEC} ; do \
	  if [ ! -f $$FILENAME ]; then \
	    echo "Error: file $$FILENAME specified in SOURCEC does not exist."; \
	  else \
	    cat $$FILENAME | \
	    sed "s/^void /void PETSC_STDCALL /" | \
	    sed "s/PetscToPointer(int)/PetscToPointer(void *)/" | \
	    sed "s/PetscRmPointer(int)/PetscRmPointer(void *)/" | \
	    sed "s/PetscToPointer(a) (a)/PetscToPointer(a) (*(long *)(a))/" | \
	    sed "s/PetscFromPointer(a) (int)(a)/PetscFromPointer(a) (long)(a)/" | \
	    sed "s/PetscToPointer( \*(int\*)/PetscToPointer( /" >_$$FILENAME; \
	    /bin/mv _$$FILENAME $$FILENAME ;\
	  fi; \
	done
