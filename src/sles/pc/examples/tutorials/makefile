#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' $(BS_INCLUDE)
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsles
LOCDIR          = src/pc/examples/tutorials/
TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_4  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_4 =

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(SLES_LIB)
	$(RM) ex1.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
