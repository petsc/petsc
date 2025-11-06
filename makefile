#
# See https://petsc.org/release/install/ for instructions on installing PETSc
#
# This is the top level makefile for compiling PETSc.
#   * make help - useful messages on functionality
#   * make all  - compile the PETSc libraries and utilities, run after ./configure
#   * make check - runs a quick test that the libraries are built correctly and PETSc applications can run
#
#   * make install - for use with ./configure is run with the --prefix=directory option
#   * make test - runs a comprehensive test suite (requires gnumake)
#   * make docs - build the entire PETSc website of documentation (locally)
#   * a variety of rules that print library properties useful for building applications (use make help)
#   * a variety of rules for PETSc developers
#
# gmakefile - manages the compiling PETSc in parallel
# gmakefile.test - manages running the comprehensive test suite
#
# This makefile does not require GNUmake
ALL: all

# next line defines PETSC_DIR and PETSC_ARCH if they are not set
include ././${PETSC_ARCH}/lib/petsc/conf/petscvariables
include ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/petscrules
include ${PETSC_DIR}/lib/petsc/conf/rules_doc.mk
include ${PETSC_DIR}/lib/petsc/conf/rules_util.mk

# This makefile contains a lot of PHONY targets with improperly specified prerequisites
# where correct execution instead depends on the targets being processed in the correct
# order.
.NOTPARALLEL:

OMAKE_SELF = $(OMAKE) -f makefile
OMAKE_SELF_PRINTDIR = $(OMAKE_PRINTDIR) -f makefile
PETSCCONF_H = ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h

#********* Rules for make all **********************************************************************************************************************************

all:
	+@${OMAKE_SELF} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} chk_upgrade | tee ${PETSC_ARCH}/lib/petsc/conf/make.log
	@ln -sf ${PETSC_ARCH}/lib/petsc/conf/make.log make.log
	+@(${OMAKE_SELF_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} all-local; echo "$$?" > ${PETSC_ARCH}/lib/petsc/conf/error.log) 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log
	+@if [ "`cat ${PETSC_ARCH}/lib/petsc/conf/error.log 2> /dev/null`" != "0" ]; then \
           grep -E '(out of memory allocating.*after a total of|gfortran: fatal error: Killed signal terminated program f951|f95: fatal error: Killed signal terminated program f951)' ${PETSC_ARCH}/lib/petsc/conf/make.log | tee ${PETSC_ARCH}/lib/petsc/conf/memoryerror.log > /dev/null; \
           if test -s ${PETSC_ARCH}/lib/petsc/conf/memoryerror.log; then \
             printf ${PETSC_TEXT_HILIGHT}"**************************ERROR*************************************\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
             echo "  Error during compile, you need to increase the memory allocated to the VM and rerun " 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
             printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
           else \
             printf ${PETSC_TEXT_HILIGHT}"**************************ERROR*************************************\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
             echo "  Error during compile, check ${PETSC_ARCH}/lib/petsc/conf/make.log" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
             echo "  Send it and ${PETSC_ARCH}/lib/petsc/conf/configure.log to petsc-maint@mcs.anl.gov" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
             if [ "X${CONDA_ACTIVE}" != "X" ]; then \
               echo "  Having Conda in your shell may have caused this problem, consider turning off Conda." 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
             fi ; \
             printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
           fi \
	 else \
	  ${OMAKE_SELF} print_mesg_after_build PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log ;\
        fi #solaris make likes to print the whole command that gave error. So split this up into the smallest chunk below
	@echo "Finishing make run at `date +'%a, %d %b %Y %H:%M:%S %z'`" >> ${PETSC_ARCH}/lib/petsc/conf/make.log
	@if [ "`cat ${PETSC_ARCH}/lib/petsc/conf/error.log 2> /dev/null`" != "0" ]; then exit 1; fi

all-local: info libs matlabbin ${PETSC_POST_BUILDS}

${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/files:
	@touch -t 197102020000 ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/files

${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles:
	@${MKDIR} -p ${PETSC_DIR}/${PETSC_ARCH}/tests && touch -t 197102020000 ${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles

chk_upgrade:
	-@PETSC_DIR=${PETSC_DIR} ${PYTHON} ${PETSC_DIR}/lib/petsc/bin/petscnagupgrade.py

matlabbin:
	-@if [ "${MATLAB_MEX}" != "" -a "${MATLAB_SOCKET}" != "" -a "${PETSC_SCALAR}" = "real" -a "${PETSC_PRECISION}" = "double" ]; then \
          echo "Compiling MATLAB interface"; \
            if [ ! -d "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc" ] ; then ${MKDIR}  ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc; fi; \
            if [ ! -d "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab" ] ; then ${MKDIR}  ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab; fi; \
            cd src/sys/classes/viewer/impls/socket/mex-scripts && ${OMAKE_SELF} mex-scripts PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR}; \
            echo "========================================="; \
        fi

fortranbindings: deletefortranbindings
	@${PYTHON} config/utils/generatefortranbindings.py --petsc-dir=${PETSC_DIR} --petsc-arch=${PETSC_ARCH}

deleteshared:
	@for LIBNAME in ${SHLIBS}; \
	do \
	   if [ -d ${INSTALL_LIB_DIR}/$${LIBNAME}$${LIB_NAME_SUFFIX}.dylib.dSYM ]; then \
             echo ${RM} -rf ${INSTALL_LIB_DIR}/$${LIBNAME}$${LIB_NAME_SUFFIX}.dylib.dSYM; \
	     ${RM} -rf ${INSTALL_LIB_DIR}/$${LIBNAME}$${LIB_NAME_SUFFIX}.dylib.dSYM; \
	   fi; \
           echo ${RM} ${INSTALL_LIB_DIR}/$${LIBNAME}$${LIB_NAME_SUFFIX}.${SL_LINKER_SUFFIX}; \
           ${RM} ${INSTALL_LIB_DIR}/$${LIBNAME}$${LIB_NAME_SUFFIX}.${SL_LINKER_SUFFIX}; \
	done
	@if [ -f ${INSTALL_LIB_DIR}/so_locations ]; then \
          echo ${RM} ${INSTALL_LIB_DIR}/so_locations; \
          ${RM} ${INSTALL_LIB_DIR}/so_locations; \
	fi

deletefortranbindings:
	-@find src -type d -name ftn-auto* | xargs rm -rf
	-@if [ -n "${PETSC_ARCH}" ] && [ -d ${PETSC_ARCH} ] && [ -d ${PETSC_ARCH}/src ]; then \
          find ${PETSC_ARCH}/src -type d -name ftn-auto* | xargs rm -rf ;\
        fi

reconfigure: allclean
	@unset MAKEFLAGS && ${PYTHON} ${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py

chkopts:
	-@echo "Warning: chkopts target is deprecated and can be removed from user makefiles"

gnumake:
	+@echo "make gnumake is deprecated, use make libs"
	+@make libs

# ********  Rules for make check ****************************************************************************************************************************

RUN_TEST = ${OMAKE_SELF} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} DIFF=${PETSC_DIR}/lib/petsc/bin/petscdiff

check: check_body ${PETSC_POST_CHECKS}

check_install: check

check_body:
	-@echo "Running PETSc check examples to verify correct installation"
	-@echo "Using PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem, cannot run make check"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. cannot run make check"; \
        else \
          ${RM} -f check_error;\
          ${RUN_TEST} OMP_NUM_THREADS=1 PETSC_OPTIONS="${EXTRA_OPTIONS} ${PETSC_TEST_OPTIONS}" PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PATH}" check_build 2>&1 | tee ./${PETSC_ARCH}/lib/petsc/conf/check.log; \
          if [ -f check_error ]; then \
            echo "Error while running make check"; \
            ${RM} -f check_error;\
            exit 1; \
          fi; \
          ${RM} -f check_error;\
        fi;

check_build:
	+@cd src/snes/tutorials >/dev/null; ${RUN_TEST} clean-legacy
	+@cd src/snes/tutorials >/dev/null; ${RUN_TEST} testex19
	+@if [ ! "${MPI_IS_MPIUNI}" ]; then cd src/snes/tutorials >/dev/null; ${RUN_TEST} testex19_mpi; fi
	+@if [ "`grep -E '^#define PETSC_HAVE_HYPRE 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_HYPRE 1" ] && [ "${PETSC_SCALAR}" = "real" ]; then \
          if [ "`grep -E '^#define PETSC_HAVE_CUDA 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_CUDA 1" ]; then HYPRE_TEST=runex19_hypre_cuda; \
          elif [ "`grep -E '^#define PETSC_HAVE_HIP 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_HIP 1" ]; then HYPRE_TEST=runex19_hypre_hip; \
          else HYPRE_TEST=runex19_hypre; fi; \
          cd src/snes/tutorials >/dev/null; ${RUN_TEST} $${HYPRE_TEST}; \
        fi;
	+@if [ "`grep -E '^#define PETSC_HAVE_CUDA 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_CUDA 1" ]; then \
          cd src/snes/tutorials >/dev/null; ${RUN_TEST} runex19_cuda; \
        fi;
	+@if [ "`grep -E '^#define PETSC_HAVE_HIP 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_HIP 1" ]; then \
          cd src/snes/tutorials >/dev/null; ${RUN_TEST} runex19_hip; \
        fi;
	+@if [ "${MPI_IS_MPIUNI}" = "" ]; then \
          cd src/snes/tutorials >/dev/null; \
          if [ "`grep -E '^#define PETSC_HAVE_KOKKOS_KERNELS 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_KOKKOS_KERNELS 1" ] && [ "${PETSC_SCALAR}" = "real" ] && [ "${PETSC_PRECISION}" = "double" ]; then \
            ${RUN_TEST} runex3k_kokkos; \
          fi;\
          if [ "`grep -E '^#define PETSC_HAVE_MUMPS 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_MUMPS 1" ]; then \
             ${RUN_TEST} runex19_fieldsplit_mumps; \
          fi;\
          if [ "`grep -E '^#define PETSC_HAVE_SUITESPARSE 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_SUITESPARSE 1" ]; then \
             ${RUN_TEST} runex19_suitesparse; \
          fi;\
          if [ "`grep -E '^#define PETSC_HAVE_SUPERLU_DIST 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_SUPERLU_DIST 1" ]; then \
            ${RUN_TEST} runex19_superlu_dist; \
          fi;\
          if [ "`grep -E '^#define PETSC_HAVE_ML 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_ML 1" ] || [ "`grep -E '^#define PETSC_HAVE_TRILINOS 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_TRILINOS 1" ]; then \
            ${RUN_TEST} runex19_ml; \
          fi; \
	  ${RUN_TEST} clean-legacy; \
          cd - > /dev/null; \
          if [ "`grep -E '^#define PETSC_HAVE_AMREX 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_AMREX 1" ] && [ "`grep -E '^#define PETSC_HAVE_CUDA 1' ${PETSCCONF_H}`" != "#define PETSC_HAVE_CUDA 1" ]; then \
            echo "Running amrex test example to verify correct installation";\
            echo "Using PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}";\
            cd src/ksp/ksp/tutorials/amrex >/dev/null;\
            ${RUN_TEST} clean-legacy; \
            ${RUN_TEST} testamrex; \
            ${RUN_TEST} clean-legacy; \
            cd - > /dev/null; \
          fi;\
        fi;
	+@if [ "`grep -E '^#define PETSC_HAVE_HDF5 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_HDF5 1" ]; then \
          cd src/vec/vec/tests >/dev/null;\
          ${RUN_TEST} clean-legacy; \
          ${RUN_TEST} runex47; \
          ${RUN_TEST} clean-legacy; \
         fi;
	+@if [ "${MPI4PY}" = "yes" ]; then \
           cd src/sys/tests >/dev/null; \
           ${RUN_TEST} clean-legacy; \
           ${RUN_TEST} testex55; \
           ${RUN_TEST} clean-legacy; \
         fi;
	+@if [ "${PETSC4PY}" = "yes" ]; then \
           cd src/ksp/ksp/tutorials >/dev/null; \
           ${RUN_TEST} clean-legacy; \
           ${RUN_TEST} testex100; \
           ${RUN_TEST} clean-legacy; \
         fi;
	+@if [ "`grep -E '^#define PETSC_USE_FORTRAN_BINDINGS 1' ${PETSCCONF_H}`" = "#define PETSC_USE_FORTRAN_BINDINGS 1" ]; then \
           cd src/snes/tutorials >/dev/null; \
           ${RUN_TEST} clean-legacy; \
           ${RUN_TEST} testex5f; \
           ${RUN_TEST} clean-legacy; \
         fi;
	+@if [ "`grep -E '^#define PETSC_HAVE_MATLAB 1' ${PETSCCONF_H}`" = "#define PETSC_HAVE_MATLAB 1" ]; then \
           cd src/vec/vec/tutorials >/dev/null;\
           ${RUN_TEST} clean-legacy; \
           ${RUN_TEST} testex31; \
           ${RUN_TEST} clean-legacy; \
          fi;
	-@echo "Completed PETSc check examples"

# ********* Rules for make install *******************************************************************************************************************

install:
	@${PYTHON} ./config/install.py -destDir=${DESTDIR}
	+${OMAKE_SELF} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETSC_INSTALL=$@ install-builtafterpetsc

# A smaller install with fewer extras
install-lib:
	@${PYTHON} ./config/install.py -destDir=${DESTDIR} -no-examples
	+${OMAKE_SELF} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETSC_INSTALL=$@ install-builtafterpetsc

install-builtafterpetsc:
	@if [ "${PETSC_POST_INSTALLS}" != "" ]; then ${OMAKE_SELF} PETSC_DIR=${PETSC_DIR} PETSC_INSTALL=${PETSC_INSTALL} ${PETSC_POST_INSTALLS}; fi
	@echo "*** Install of PETSc (and any other packages) complete ***"

# Creates ${HOME}/petsc.tar.gz [and petsc-with-docs.tar.gz]
dist:
	${PETSC_DIR}/lib/petsc/bin/maint/builddist ${PETSC_DIR} main

# ******** Rules for running the full test suite ********************************************************************************************************

TESTMODE = testexamples
ALLTESTS_CHECK_FAILURES = no
ALLTESTS_MAKEFILE = ${PETSC_DIR}/gmakefile.test
VALGRIND=0
alltests: chk_in_petscdir ${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles
	-@${RM} -rf ${PETSC_ARCH}/lib/petsc/conf/alltests.log alltests.log
	+@if [ -f ${PETSC_DIR}/share/petsc/examples/gmakefile.test ] ; then \
            ALLTESTS_MAKEFILE=${PETSC_DIR}/share/petsc/examples/gmakefile.test ; \
            ALLTESTSLOG=alltests.log ;\
          else \
            ALLTESTS_MAKEFILE=${PETSC_DIR}/gmakefile.test; \
            ALLTESTSLOG=${PETSC_ARCH}/lib/petsc/conf/alltests.log ;\
            ln -s $${ALLTESTSLOG} alltests.log ;\
          fi; \
          ${OMAKE} allgtest ALLTESTS_MAKEFILE=$${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} MPIEXEC="${MPIEXEC}" DATAFILESPATH=${DATAFILESPATH} VALGRIND=${VALGRIND} 2>&1 | tee $${ALLTESTSLOG};\
          if [ x${ALLTESTS_CHECK_FAILURES} = xyes -a ${PETSC_PRECISION} != single ]; then \
            cat $${ALLTESTSLOG} | grep -E '(^not ok|not remade because of errors|^# No tests run)' | wc -l | grep '^[ ]*0$$' > /dev/null; \
          fi;

allgtests-tap: allgtest-tap
	+@${OMAKE} -f ${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} check-test-errors

allgtest-tap: ${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles
	+@MAKEFLAGS="-j$(MAKE_TEST_NP) -l$(MAKE_LOAD) $(MAKEFLAGS)" ${OMAKE} ${MAKE_SHUFFLE_FLG} -f ${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test OUTPUT=1

allgtest: ${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles
	+@MAKEFLAGS="-j$(MAKE_TEST_NP) -l$(MAKE_LOAD) $(MAKEFLAGS)" ${OMAKE} -k -f ${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test V=0 2>&1 | grep -E -v '^(ok [^#]*(# SKIP|# TODO|$$)|[A-Za-z][A-Za-z0-9_]*\.(c|F|cxx|F90).$$)'

test:
	+${OMAKE} -f ${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test
cleantest:
	+${OMAKE} -f ${ALLTESTS_MAKEFILE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} cleantest

#********* Rules for cleaning ***************************************************************************************************************************

deletelibs:
	-${RM} -rf ${PETSC_LIB_DIR}/libpetsc*.*
deletemods:
	-${RM} -f ${PETSC_DIR}/${PETSC_ARCH}/include/petsc*.mod

allclean:
	-@${OMAKE} -f gmakefile clean

clean:: allclean

distclean:
	@if [ -f ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py ]; then \
	  echo "*** Preserving ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py in ${PETSC_DIR} ***"; \
          mv -f ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py ${PETSC_DIR}/; \
	  echo "*** Deleting all build files in ${PETSC_DIR}/${PETSC_ARCH} ***"; \
	  ${RM} -rf ${PETSC_DIR}/${PETSC_ARCH}/ ; \
        else  \
	  echo "*** Build files in PETSC_ARCH=${PETSC_ARCH} not found. Skipping delete! ***"; \
        fi

info:
	+@${OMAKE} -f gmakefile gmakeinfo

check_usermakefile:
	-@echo "Testing compile with user makefile"
	-@echo "Using PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"
	@cd src/snes/tutorials; ${RUN_TEST} clean-legacy
	@cd src/snes/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} -f ${PETSC_DIR}/share/petsc/Makefile.user ex19
	@grep -E "^#define PETSC_USE_FORTRAN_BINDINGS 1" ${PETSCCONF_H} | tee .ftn.log > /dev/null; \
         if test -s .ftn.log; then \
          cd src/snes/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} -f ${PETSC_DIR}/share/petsc/Makefile.user ex5f; \
         fi; ${RM} .ftn.log;
	@cd src/snes/tutorials; ${RUN_TEST} clean-legacy
	-@echo "Completed compile with user makefile"

#********* Rules for formatting Fortran source **********************************************************************************************

# pip install fprettify
fprettify:
	@git ls-files "*.[hF]90" | xargs fprettify --indent 2 --line-length 1000 --whitespace 2 --whitespace-type F --enable-replacements --c-relations

# git clone https://github.com/louoberto/fortify.git && cd fortify && export PATH=$PATH:$(pwd)/source
fortify:
	@files=`git ls-files "*.[hF]90"`; for i in $${files}; do fortify --tab_length 2 --lowercasing F $${i}; done

#********* Rules for running clangformat ************************************************************************************************************

checkgitclean:
	@if ! git diff --quiet; then \
           echo "The repository has uncommitted files, cannot run checkclangformat" ;\
           git status -s --untracked-files=no ;\
           false;\
        fi;

# Check that all the C/C++ source code in the repository satisfies the .clang_format
checkclangformat: checkclangformatversion checkgitclean clangformat
	@if ! git diff --quiet; then \
          printf "The current commit has C/C++ source code formatting problems\n" ;\
          if [ -z "${CI_PIPELINE_ID}"  ]; then \
            printf "Please run 'git diff' to check\n"; \
            git diff --stat; \
          else \
            git diff --patch-with-stat >  ${PETSC_ARCH}/lib/petsc/conf/checkclangformat.patch; \
            git diff --patch-with-stat --color=always | head -1000; \
            if [ `wc -l < ${PETSC_ARCH}/lib/petsc/conf/checkclangformat.patch` -gt 1000 ]; then \
              printf "The diff has been trimmed, check ${PETSC_ARCH}/lib/petsc/conf/checkclangformat.patch (in CI artifacts) for all changes\n"; \
            fi;\
          fi;\
          false;\
        fi;

# Check that all the Fortran source code in the repository satisfies the fprettify format
checkfprettifyformat: checkgitclean fprettify
	@if ! git diff --quiet; then \
          printf "The current commit has Fortra source code formatting problems\n" ;\
          if [ -z "${CI_PIPELINE_ID}"  ]; then \
            printf "Please run 'git diff' to check\n"; \
            git diff --stat; \
          else \
            git diff --patch-with-stat >  ${PETSC_ARCH}/lib/petsc/conf/checkfprettifyformat.patch; \
            git diff --patch-with-stat --color=always | head -1000; \
            if [ `wc -l < ${PETSC_ARCH}/lib/petsc/conf/checkfprettifyformat.patch` -gt 1000 ]; then \
              printf "The diff has been trimmed, check ${PETSC_ARCH}/lib/petsc/conf/checkfprettifyformat.patch (in CI artifacts) for all changes\n"; \
            fi;\
          fi;\
          false;\
        fi;

# Compare ABI/API of two versions of PETSc library with the old one defined by PETSC_{DIR,ARCH}_ABI_OLD
abitest:
	@if [ "${PETSC_DIR_ABI_OLD}" = "" ] || [ "${PETSC_ARCH_ABI_OLD}" = "" ]; \
		then printf "You must set environment variables PETSC_DIR_ABI_OLD and PETSC_ARCH_ABI_OLD to run abitest\n"; \
		exit 1; \
	fi;
	-@echo "Comparing ABI/API of the following two PETSc versions (you must have already configured and built them using GCC and with -g):"
	-@echo "========================================================================================="
	-@echo "    Old: PETSC_DIR_ABI_OLD  = ${PETSC_DIR_ABI_OLD}"
	-@echo "         PETSC_ARCH_ABI_OLD = ${PETSC_ARCH_ABI_OLD}"
	-@pushd ${PETSC_DIR_ABI_OLD} >> /dev/null ; echo "         Branch             = "`git rev-parse --abbrev-ref HEAD`
	-@echo "    New: PETSC_DIR          = ${PETSC_DIR}"
	-@echo "         PETSC_ARCH         = ${PETSC_ARCH}"
	-@echo "         Branch             = "`git rev-parse --abbrev-ref HEAD`
	-@echo "========================================================================================="
	-@$(PYTHON)	${PETSC_DIR}/lib/petsc/bin/maint/abicheck.py -old_dir ${PETSC_DIR_ABI_OLD} -old_arch ${PETSC_ARCH_ABI_OLD} -new_dir ${PETSC_DIR} -new_arch ${PETSC_ARCH} -report_format html

# Run fortitude Fortran linter; pip install fortitude-lint; fortitude does not support using the preprocessor so it is of only limited utility
fortitude:
	-@fortitude check --line-length 1000 --ignore C003,C121,S241 --verbose --fix --preview

# Compare ABI/API of current PETSC_ARCH/PETSC_DIR with a previous branch
abitestcomplete:
	-@if [[ -f "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/configure.log" ]]; then \
          OPTIONS=`grep -h -m 1 "Configure Options: " ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/configure.log  | sed "s!Configure Options: --configModules=PETSc.Configure --optionsModule=config.compilerOptions!!g"` ;\
echo $${OPTIONS} ;\
        fi ; \
        if [[ "${PETSC_DIR_ABI_OLD}" != "" ]]; then \
          PETSC_DIR_OLD=${PETSC_DIR_ABI_OLD}; \
        else \
          PETSC_DIR_OLD=${PETSC_DIR}/../petsc-abi; \
        fi ; \
        echo "=================================================================================================" ;\
        echo "Doing ABI/API comparison between" ${branch} " and " `git rev-parse --abbrev-ref HEAD` "using " $${OPTIONS} ;\
        echo "=================================================================================================" ;\
        if [[ ! -d $${PETSC_DIR_OLD} ]]; then \
          git clone ${PETSC_DIR} $${PETSC_DIR_OLD} ; \
        else \
          cd $${PETSC_DIR_OLD} ; \
          git pull ; \
        fi ; \
        cd $${PETSC_DIR_OLD} ; \
        git checkout ${branch} ; \
        PETSC_DIR=`pwd` PETSC_ARCH=arch-branch-`git rev-parse ${branch}` ./configure $${OPTIONS} ; \
        PETSC_DIR=`pwd` PETSC_ARCH=arch-branch-`git rev-parse ${branch}` make all test ; \
        cd ${PETSC_DIR} ; \
        ./configure $${OPTIONS}; \
        make all test ; \
        PETSC_DIR_ABI_OLD=$${PETSC_DIR_OLD} PETSC_ARCH_ABI_OLD=arch-branch-`git rev-parse ${branch}` make abitest

# ******** Rules for running Streams benchmark ****************************************************************************************************

mpistreams:
	+@cd src/benchmarks/streams; ${OMAKE_SELF} PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PATH}" PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} mpistreams

mpistream:
	+@cd src/benchmarks/streams; ${OMAKE_SELF} PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PATH}" PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} mpistream

openmpstreams:
	+@cd src/benchmarks/streams; ${OMAKE_SELF} PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PATH}" PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} openmpstreams

openmpstream:
	+@cd src/benchmarks/streams; ${OMAKE_SELF} PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PATH}" PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} openmpstream

stream: mpistream

streams: mpistreams

# ********  Rules for generating tag files for Emacs/VIM *******************************************************************************************

alletags: etags

etags:
	-@${PYTHON} lib/petsc/bin/maint/generateetags.py && ${CP} TAGS ${PETSC_ARCH}/
	-@find config -type f -name "*.py" |grep -v SCCS | xargs etags -o TAGS_PYTHON

# obtain gtags from https://www.gnu.org/software/global/
allgtags:
	-@find ${PETSC_DIR}/include ${PETSC_DIR}/src -regex '\(.*makefile\|.*\.\(cc\|hh\|cpp\|cxx\|C\|hpp\|c\|h\|cu\|m\)$$\)' | grep -v ftn-auto  | gtags -f -

# ********* Rules for building "classic" documentation; uses rules also in lib/petsc/conf/rules_doc.mk **************************************************

docs:
	cd doc; ${OMAKE_SELF} docs

chk_in_petscdir:
	@if [ ! -f include/petscversion.h ]; then \
	  printf ${PETSC_TEXT_HILIGHT}"*********************** ERROR **********************************************\n" ; \
	  echo " This target should be invoked in top level PETSc source dir!"; \
	  printf "****************************************************************************"${PETSC_TEXT_NORMAL}"\n" ;  false; fi

chk_loc:
	@if [ ${LOC}foo = foo ] ; then \
	  printf ${PETSC_TEXT_HILIGHT}"*********************** ERROR **********************************************\n" ; \
	  echo " Please specify LOC variable for eg: make allmanpages LOC=/sandbox/petsc "; \
	  printf "****************************************************************************"${PETSC_TEXT_NORMAL}"\n" ;  false; fi
	@${MKDIR} ${LOC}/manualpages

alldocclean: deletemanualpages allcleanhtml

# Deletes man pages (.md version)
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/manualpages ]; then \
          find ${LOC}/manualpages -type f -name "*.md" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/manualpages/manualpages.cit ;\
        fi

allcleanhtml:
	-${OMAKE_SELF} ACTION=cleanhtml PETSC_DIR=${PETSC_DIR} tree

# ********* Rules for checking code coverage *********************************************************************************************

gcov:
	output_file_base_name=${PETSC_ARCH}-gcovr-report.json; \
	petsc_arch_dir=${PETSC_DIR}/${PETSC_ARCH}; \
        tar_file=$${petsc_arch_dir}/$${output_file_base_name}.tar.bz2; \
	cd $${petsc_arch_dir} && \
	gcovr --json --output $${petsc_arch_dir}/$${output_file_base_name} --exclude '.*/ftn-auto/.*' --exclude '.*/petscsys.h' --exclude-lines-by-pattern '^\s*SETERR.*' --exclude-throw-branches --exclude-unreachable-branches --gcov-ignore-parse-errors -j 8 --gcov-executable "${PETSC_COVERAGE_EXEC}" --root ${PETSC_DIR} ./obj ./tests ${PETSC_GCOV_OPTIONS} && \
	${RM} -f $${tar_file} && \
	tar --bzip2 -cf $${tar_file} -C $${petsc_arch_dir} ./$${output_file_base_name} && \
	${RM} $${petsc_arch_dir}/$${output_file_base_name}

mergegcov:
	$(PYTHON) ${PETSC_DIR}/lib/petsc/bin/maint/gcov.py --merge-branch `lib/petsc/bin/maint/check-merge-branch.sh` --html --xml ${PETSC_GCOV_OPTIONS}

countcfunctions:
	-@grep PETSC_EXTERN ${PETSC_DIR}/include/*.h  | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort | uniq > /tmp/countcfunctions

checkpackagetests:
	-@echo "Missing package tests"
	-@cat config/examples/*.py > configexamples; pushd config/BuildSystem/config/packages/; packages=`ls *.py | sed "s/\\.py//g"`;popd; for i in $${packages}; do j=`echo $${i} | tr '[:upper:]' '[:lower:]'`; printf $${j} ; grep -E "(with-$${j}|download-$${j})" configexamples | grep -v "=0" | wc -l ; done
	-@echo "Missing download package tests"
	-@cat config/examples/*.py > configexamples; pushd config/BuildSystem/config/packages/; packages=`grep -l "download " *.py  | sed "s/\\.py//g"`;popd; for i in $${packages}; do j=`echo $${i} | tr '[:upper:]' '[:lower:]'`; printf $${j} ; grep -E "(download-$${j})" configexamples | grep -v "=0" | wc -l ; done

check_petsc4py_rst:
	@${RM} -f petsc4py_rst.log
	@echo "Checking src/binding/petsc4py/DESCRIPTION.rst"
	@rst2html src/binding/petsc4py/DESCRIPTION.rst > /dev/null 2> petsc4py_rst.log
	@if test -s petsc4py_rst.log; then cat petsc4py_rst.log; exit 1; fi

# TODO: checkTestCoverage: that makes sure every tutorial test has at least one test listed in the file, perhaps handled by gmakegentest.py
# TODO: check for PetscBeginFunctionUser in non-example source
# TODO: check for __ at start of #define or symbol
# TODO: checking for missing manual pages
# TODO: check for incorrect %d
# TODO: check for double blank lines
# TODO: check for ill-formed manual pages
# TODO: check for { on line after for
# TODO: check for } then else on following line
# TODO: check for } else { with SETERRQ on following line or if () { with SETERRQ on following line

checkbadSpelling:
	-@x=`python3 ../bin/extract.py | aspell list | sort -u` ;

updatedatafiles:
	-@if [ -d "${DATAFILESPATH}" ]; then \
            echo " ** Updating datafiles at ${DATAFILESPATH} **"; \
            cd "${DATAFILESPATH}" && git pull -q; fi

.PHONY: info info_h all deletelibs allclean update \
        alletags etags etags_complete etags_noexamples etags_makefiles etags_examples etags_fexamples alldoc allmanpages \
        allcleanhtml \
        start_configure configure_petsc configure_clean matlabbin install
