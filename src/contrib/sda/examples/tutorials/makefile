#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(PCONF) $(PETSC_INCLUDE) -D__DIR__='"contrib/sda/examples/tutorials"' \
           $(CONF) -I../src
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LOCDIR   = src/contrib/sda/examples/tutorials/

EXAMPLES_2    =
EXAMPLES_3    =

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

PETSC_CONTRIB = -L$(LDIR) -lpetsccontrib

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o $(PETSC_CONTRIB)  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(FLINKER) -o ex2 ex2.o  $(PETSC_FORTRAN_LIB) $(PETSC_CONTRIB) \
                    $(SLES_LIB)
	$(RM) -f ex2.o

