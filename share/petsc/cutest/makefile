include ${PETSC_DIR}/lib/petsc/conf/variables

ifeq ($(PETSC_PRECISION),single)
SIFPRECISION=sp
else ifeq ($(PETSC_PRECISION),double)
SIFPRECISION=dp
else ifeq ($(PETSC_PRECISION),__float128)
SIFPRECISION=qp
endif

SIFNAME=$(basename $(notdir $(SIF)))
# SIFDecode generates Fortran with unused variables and labels, so do not treat warnings as errors.
FCFLAGS=$(filter-out -Werror,${FC_FLAGS} ${FFLAGS})

.PHONY: all
all:
	${SIFDECODER} -$(SIFPRECISION) $(SIF)
	${FC} $(FCFLAGS) -c *.f
	${SL_LINKER} ${SL_LINKER_FLAGS} $(call SL_LINKER_FUNCTION,libcutest_$(SIFNAME),0,0) -o libcutest_$(SIFNAME).${SL_LINKER_SUFFIX} *.o ${PETSC_EXTERNAL_LIB_BASIC}
