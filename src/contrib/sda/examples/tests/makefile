PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PCONF) $(PETSC_INCLUDE) -D__DIR__='"contrib/sda/"' $(CONF) 
SOURCEC  = sda2.c sda2f.c
SOURCEF  =
SOURCEH  = sda.h
OBJSC    = sda2.o sda2f.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LOCDIR   = src/contrib/sda/

EXAMPLES_2    = ex1
EXAMPLES_3    = ex2


include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(FLINKER) -o ex2 ex2.o  $(PETSC_FORTRAN_LIB) $(SLES_LIB)
	$(RM) -f ex2.o

