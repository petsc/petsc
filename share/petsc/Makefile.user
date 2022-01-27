# -*- mode: makefile -*-

#  This sample GNU Makefile can be used to compile PETSc applications
#  It relies on pkg_config tool (see $PETSC_DIR/share/petsc/Makefile.basic.user if you cannot use pkg_config)
#  Copy this file to your source directory as "Makefile" and modify as needed.
#  You must set the environmental variable(s) PETSC_DIR (and PETSC_ARCH if PETSc was not configured with the --prefix option)
#  See also share/petsc/Makefile.basic.user for a makefile that does not require pkg_config
#
#  For example - a single source file (ex1.c or ex1.F90) can be compiled with:
#
#      make ex1
#
#  You do not need to edit this makefile at all.
#
#  For a multi-file case, suppose you have the source files a.c, b.c, and c.cxx
#  This can be built by uncommenting the following two lines.
#
# app : a.o b.o c.o
# 	$(LINK.C) -o $@ $^ $(LDLIBS)
#
#  When linking in a multi-files with Fortran source files a.F90, b.c, and c.cxx
#  You may need to use
#
# app : a.o b.o c.o
# 	$(LINK.F) -o $@ $^ $(LDLIBS)

#  If the file c.cxx needs to link with a C++ standard library -lstdc++ , then
#  you'll need to add it explicitly.  It can go in the rule above or be added to
#  a target-specific variable by uncommenting the line below.
#
# app : LDLIBS += -lstdc++
#
#  The following variable must either be a path to petsc.pc or just "petsc" if petsc.pc
#  has been installed to a system location or can be found in PKG_CONFIG_PATH.
petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

# Additional libraries that support pkg-config can be added to the list of PACKAGES below.
PACKAGES := $(petsc.pc)

CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CXX := $(shell pkg-config --variable=cxxcompiler $(PACKAGES))
FC := $(shell pkg-config --variable=fcompiler $(PACKAGES))
CFLAGS_OTHER := $(shell pkg-config --cflags-only-other $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
CXXFLAGS := $(shell pkg-config --variable=cxxflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
FFLAGS := $(shell pkg-config --variable=fflags_extra $(PACKAGES))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm
CUDAC := $(shell pkg-config --variable=cudacompiler $(PACKAGES))
CUDAC_FLAGS := $(shell pkg-config --variable=cudaflags_extra $(PACKAGES))
CUDA_LIB := $(shell pkg-config --variable=cudalib $(PACKAGES))
CUDA_INCLUDE := $(shell pkg-config --variable=cudainclude $(PACKAGES))

print:
	@echo CC=$(CC)
	@echo CXX=$(CXX)
	@echo FC=$(FC)
	@echo CFLAGS=$(CFLAGS)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo FFLAGS=$(FFLAGS)
	@echo CPPFLAGS=$(CPPFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
	@echo CUDAC=$(CUDAC)
	@echo CUDAC_FLAGS=$(CUDAC_FLAGS)
	@echo CUDA_LIB=$(CUDA_LIB)
	@echo CUDA_INCLUDE=$(CUDA_INCLUDE)

# Many suffixes are covered by implicit rules, but you may need to write custom rules
# such as these if you use suffixes that do not have implicit rules.
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
% : %.F90
	$(LINK.F) -o $@ $^ $(LDLIBS)
%.o: %.F90
	$(COMPILE.F) $(OUTPUT_OPTION) $<
% : %.cxx
	$(LINK.cc) -o $@ $^ $(LDLIBS)
%.o: %.cxx
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
%.o : %.cu
	$(CUDAC) -c $(CPPFLAGS) $(CUDAC_FLAGS) $(CUDA_INCLUDE) -o $@ $<
