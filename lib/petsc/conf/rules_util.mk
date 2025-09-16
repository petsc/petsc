# -*- mode: makefile-gmake -*-
#
#    Contains rules that are shared between SLEPc and PETSc
#
#    This file is not included by rules, or rules_doc.mk. In PETSc it is only included by the toplevel makefile

# ********* Rules for printing PETSc library properties useful for building applications  ***********************************************************

getversion:
	-@${PETSC_DIR}/lib/petsc/bin/petscversion

getmpilinklibs:
	-@echo  ${MPI_LIB}

getmpiincludedirs:
	-@echo  ${MPI_INCLUDE}

getmpiexec:
	-@echo  ${MPIEXEC}

getccompiler:
	-@echo ${CC}

getfortrancompiler:
	-@echo ${FC}

getcxxcompiler:
	-@echo ${CXX}

getlinklibs:
	-@echo  ${C_SH_LIB_PATH} ${PETSC_TS_LIB}

getincludedirs:
	-@echo  ${PETSC_CC_INCLUDES}

getcflags:
	-@echo ${CC_FLAGS}

getcxxflags:
	-@echo ${CXX_FLAGS}

getfortranflags:
	-@echo ${FC_FLAGS}

getblaslapacklibs:
	-@echo ${BLASLAPACK_LIB}

getautoconfargs:
	-@echo CC='"${CC}"' CXX='"${CXX}"'  FC='"${FC}"' CFLAGS='"${CC_FLAGS}"' CXXFLAGS='"${CXX_FLAGS}"' FCFLAGS='"${FC_FLAGS}"' LIBS='"${C_SH_LIB_PATH} ${PETSC_TS_LIB}"'

#********* Rules for checking the PETSc source code, and comments do not violate PETSc standards. And for properly formatting source code and manual pages
PETSCCLANGFORMAT ?= clang-format
# Check the version of clang-format matches PETSc requirement
checkclangformatversion:
	@version=`${PETSCCLANGFORMAT} --version | cut -d" " -f3 | cut -d"." -f 1` ;\
         if [ "$$version" = "version" ]; then version=`${PETSCCLANGFORMAT} --version | cut -d" " -f4 | cut -d"." -f 1`; fi;\
         if [ $$version != 21 ]; then echo "Require clang-format version 21! Currently used ${PETSCCLANGFORMAT} version is $$version" ;false ; fi

# Format all the source code in the given directory and down according to the file $PETSC_DIR/.clang_format
clangformat: checkclangformatversion
	-@git --no-pager ls-files -z ${GITCFSRC} | xargs -0 -P $(MAKE_NP) -L 10 ${PETSCCLANGFORMAT} -i

GITSRC = '*.[chF]' '*.F90' '*.hpp' '*.cpp' '*.cxx' '*.cu' ${GITSRCEXCL}
GITSRCEXCL = \
':!*khash/*' \
':!*valgrind/*' \
':!*yaml/*' \
':!*perfstubs/*'
GITCFSRC = '*.[ch]' '*.hpp' '*.cpp' '*.cxx' '*.cu' ${GITSRCEXCL} ${GITCFSRCEXCL}
GITCFSRCEXCL = \
':!*petscversion.h' \
':!*mpif.h' \
':!*mpiunifdef.h' \
':!*finclude/*' \
':!systems/*' \
':!*benchmarks/*' \
':!*binding/*' \
':!*ftn-mod/*'

# Check that copies of external source code that live in the PETSc repository have not been changed by developer
checkbadFileChange:
	@git diff --stat --exit-code `lib/petsc/bin/maint/check-merge-branch.sh`..HEAD -- src/sys/yaml/include src/sys/yaml/License include/petsc/private/valgrind include/petsc/private/kash

vermin:
	@vermin --violations -t=3.4- ${VERMIN_OPTIONS} ${PETSC_DIR}/config
	@vermin --violations -t=3.6- ${VERMIN_OPTIONS} ${PETSC_DIR}/src/binding/petsc4py

# Check that source code does not violate basic PETSc coding standards
checkbadsource: checkbadSource

checkbadSource:
	@git --no-pager grep -n -P 'self\.gitcommit' -- config/BuildSystem/config/packages | grep 'origin/' ; if [[ "$$?" == "0" ]]; then echo "Error: Do not use a branch name in a configure package file"; false; fi
	-@${RM} -f checkbadSource.out
	-@touch checkbadSource.out
	-@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/check_header_guard.py --action=check --kind=pragma_once -- ./src ./include >> checkbadSource.out
	-@echo "----- Double blank lines in file -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '^$$' -- ${GITSRC} > doublelinecheck.out
	-@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/doublelinecheck.py doublelinecheck.out >> checkbadSource.out
	-@${RM} -f doublelinecheck.out
	-@echo "----- Tabs in file -------------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '\t' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Tabs in makefiles --------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '[ ]*[#A-Za-z0-9][ :=_A-Za-z0-9]*\t' -- makefile  >> checkbadSource.out;true
	-@echo "----- White space at end of line -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P ' $$' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Two ;; -------------------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -e ';;' -- ${GITSRC} | grep -v ' for (' >> checkbadSource.out;true
	-@echo "----- PetscCall for an MPI error code ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -e 'PetscCall\(MPI[U]*_\w*\(.*\)\);' -- ${GITSRC} | grep -Ev 'MPIU_File' >> checkbadSource.out;true
	-@echo "----- DOS file (with DOS newlines) ---------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '\r' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- { before SETERRQ ---------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '{SETERRQ' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- PetscCall following SETERRQ ----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'SETERRQ' -- ${GITSRC} | grep ";PetscCall" >> checkbadSource.out;true
	-@echo "----- SETERRQ() without defined error code -------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'SETERRQ\((?!\))' -- ${GITSRC} | grep -v PETSC_ERR  | grep " if " | grep -v "__VA_ARGS__" | grep -v "then;" | grep -v flow.c >> checkbadSource.out;true
	-@echo "----- SETERRQ() with trailing newline ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P "SETERRQ[1-9]?.*\\\n\"" -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Define keyword used in test definition -----------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -e 'requires:.*define\(' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using if (condition) SETERRQ(...) instead of PetscCheck() ----" >> checkbadSource.out
	-@git --no-pager grep -n -P ' if +(.*) *SETERRQ' -- ${GITSRC} | grep -v 'PetscUnlikelyDebug' | grep -v 'petscerror.h' | grep -v "then;" | grep -v "__VA_ARGS__"  >> checkbadSource.out;true
	-@echo "----- Using if (PetscUnlikelyDebug(condition)) SETERRQ(...) instead of PetscAssert()" >> checkbadSource.out
	-@git --no-pager grep -n -P -E ' if +\(PetscUnlikelyDebug.*\) *SETERRQ' -- ${GITSRC} | grep -v petscerror.h >> checkbadSource.out;true
	-@echo "----- Using PetscFunctionReturn(ierr) ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'PetscFunctionReturn(ierr)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- .seealso with leading white spaces ---------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '^[ ]+.seealso:' -- ${GITSRC} ':!src/sys/tests/linter/*' >> checkbadSource.out;true
	-@echo "----- .seealso with double backticks -------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '^.seealso:.*``' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Defining a returning macro without PetscMacroReturns() -------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'define .*\w+;\s+do' -- ${GITSRC} | grep -E -v '(PetscMacroReturns|PetscDrawCollectiveBegin|MatPreallocateBegin|PetscOptionsBegin|PetscObjectOptionsBegin|PetscOptionsHeadEnd)' >> checkbadSource.out;true
	-@echo "----- Defining an error checking macro using CHKERR style ----------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'define\s+CHKERR\w*\(.*\)\s*do\s+{' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using Petsc[Array|Mem]cpy() for ops instead of assignment ----" >> checkbadSource.out
	-@git --no-pager grep -n -P 'cpy\(.*(.|->)ops, .*\)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Extra spaces in test harness rules ---------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '^(\!){0,1}[ ]*(suffix|output_file|nsize|requires|args):.*  .*' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Extra comma in test harness rules ----------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '^(\!){0,1}[ ]*requires:.*,' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using PetscInfo() without carriage return --------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'PetscCall\(PetscInfo\(' -- ${GITSRC} | grep -v '\\n' >> checkbadSource.out;true
	-@echo "----- Using Petsc(Assert|Check)() with carriage return -------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E 'Petsc(Assert|Check)\(.*[^\]\\\n' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Extra \"\" after format specifier ending a string --------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '_FMT \"\",' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- First blank line ---------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P \^\$$ -- ${GITSRC} | grep ':1:' >> checkbadSource.out;true
	-@echo "----- Blank line after PetscFunctionBegin and derivatives ----------" >> checkbadSource.out
	-@git --no-pager grep -n -E -A 1 '  PetscFunctionBegin(User|Hot){0,1};' -- ${GITSRC} | grep -E '\-[0-9]+-$$' | grep -v '^--$$' >> checkbadSource.out;true
	-@echo "----- Blank line before PetscFunctionReturn ------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -E -B 1 '  PetscFunctionReturn' -- ${GITSRC} | grep -E '\-[0-9]+-$$' | grep -v '^--$$' >> checkbadSource.out;true
	-@echo "----- No blank line before PetscFunctionBegin and derivatives ------" >> checkbadSource.out
	-@git --no-pager grep -n -E -B 1 '  PetscFunctionBegin(User|Hot){0,1};' -- ${GITSRC} ':!src/sys/tests/*' ':!src/sys/tutorials/*' | grep -E '\-[0-9]+-.*;' | grep -v '^--$$' | grep -v '\\' >> checkbadSource.out;true
	-@echo "----- Unneeded parentheses [!&~*](foo[->|.]bar) --------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '([\!\&\~\*\(]|\)\)|\([^,\*\(]+\**\))\(([a-zA-Z0-9_]+((\.|->)[a-zA-Z0-9_]+|\[[a-zA-Z0-9_ \%\+\*\-]+\])+)\)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Use PetscSafePointerPlusOffset(ptr, n) instead of ptr ? ptr + n : NULL" >> checkbadSource.out
	-@git --no-pager grep -n -Po ' ([^()\ ]+) \? (?1) \+ (.)* : NULL' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Wrong PETSc capitalization -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '[^a-zA-Z_*>{.]petsc [^+=]' -- ${GITSRC} | grep -v 'mat_solver_type petsc' | grep -v ' PETSc ' >> checkbadSource.out;true
	-@echo "----- Semi-colon at end of Fortran line ----------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E ";$$" -- '*.[hF]90' >> checkbadSource.out;true
	-@echo "----- Empty test harness output_file not named output/empty.out ----" >> checkbadSource.out
	-@git --no-pager grep -L . -- '*.out' | grep -Ev '(/empty|/[a-zA-Z0-9_-]+_alt).out' >> checkbadSource.out;true
	-@echo "----- Unnecessary braces around one-liners -------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '[ ]*(if|for|while|do|else) \(.*\) \{[^;]*;[^;]*\}( \\)?$$' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- MPI_(Allreduce|Irecv|Isend) instead of MPIU_(Allreduce|Irecv|Isend)" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '\(MPI_(Allreduce|Irecv|Isend)\([^\)]' -- ${GITSRC} ':!*/tests/*' ':!*/tutorials/*' ':!src/sys/objects/pinit.c' >> checkbadSource.out;true
	@a=`cat checkbadSource.out | wc -l`; l=`expr $$a - 36` ;\
         if [ $$l -gt 0 ] ; then \
           echo $$l " files with errors detected in source code formatting" ;\
           cat checkbadSource.out ;\
         else \
	   ${RM} -f checkbadSource.out;\
         fi;\
         test ! $$l -gt 0
	-@git --no-pager grep -P -n "[\x00-\x08\x0E-\x1F\x80-\xFF]" -- ${GITSRC} > badSourceChar.out;true
	-@w=`cat badSourceChar.out | wc -l`;\
         if [ $$w -gt 0 ] ; then \
           echo "Source files with non-ASCII characters ----------------" ;\
           cat badSourceChar.out ;\
         else \
	   ${RM} -f badSourceChar.out;\
         fi
	@test ! -s badSourceChar.out

#  Run a linter in a Python virtual environment to check (and fix) the formatting of PETSc manual pages
#     V=1        verbose
#     REPLACE=1  replace ill-formatted docs with correctly formatted docs
env-lint:
	@if [[ `which llvm-config` == "" ]]; then echo "llvm-config for version 14 must be in your path"; exit 1; fi
	@if [ `llvm-config --version | cut -f1 -d"."` != 14 ]; then echo "llvm-config for version 14 must be in your path"; exit 1; fi
	@python3 -m venv petsc-lint-env
	@source petsc-lint-env/bin/activate && python3 -m pip install --quiet -r lib/petsc/bin/maint/petsclinter/requirements.txt && \
          python3 ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter --verbose=${V} --apply-patches=${REPLACE} --clang_lib=`llvm-config --libdir`/libclang.dylib --werror 1 ./src ; deactivate

#  Run a linter to check (and fix) the formatting of PETSc manual pages
lint:
	${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter --verbose=${V} --apply-patches=${REPLACE} $(LINTER_OPTIONS) ${DIRECTORY}

help-lint:
	@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter --help
	-@echo "Basic usage:"
	-@echo "   make lint      <options> [DIRECTORY=directory]"
	-@echo "   make test-lint <options> <test only options>"
	-@echo
	-@echo "Options:"
	-@echo "  V=[0, 1, 2, 3]                        Enable increasingly verbose output"
	-@echo "  LINTER_OPTIONS=\"--opt1 --opt2 ...\"  See above for available options"
	-@echo
	-@echo "Test Only Options:"
	-@echo "  LINTER_SKIP_MYPY=[1, 0]               Enable or disable mypy checks"
	-@echo "  REPLACE=[1, 0]                        Enable or disable replacing test output"
	-@echo

checkvermin_exist:
	@ret=`which vermin > /dev/null`; \
	if [ $$? -ne 0 ]; then \
          echo "vermin is required, please install: python3 -m pip install vermin"; \
          false; \
        fi

checkmypy_exist:
	@ret=`which mypy > /dev/null`; \
	if [ $$? -ne 0 ]; then \
          echo "MyPy is required, please install: python3 -m pip install mypy"; \
          false; \
        fi

LINTER_SKIP_MYPY ?= 0

test-lint: checkvermin_exist checkmypy_exist
	vermin --config-file ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter/pyproject.toml -- ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter
	if [[ "$(LINTER_SKIP_MYPY)" == "0" ]]; then \
          cd ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter && mypy . ; \
        fi
	${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter/petsclinter/pkg_consistency_checks.py
	${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/petsclinter ${PETSC_DIR}/src/sys/tests/linter --test -j0 --werror --replace=${REPLACE} --verbose=${V} $(LINTER_OPTIONS)


#  Lists all the URLs in the PETSc repository that are unaccessible, nonexistent, or permanently moved (301)
#  REPLACE=1 locations marked as permanently moved (301) are replaced in the repository
#  This code is fragile; always check the changes after a use of REPLACE=1 before committing the changes
#
#  Notes:
#    The first tr in the line is split lines for the cases where multiple URLs are in the same line
#    The first sed handles the case (http[s]*://xxx)
#    The list is sorted by length so that if REPLACE is used the longer apply before the shorter
#    The code recursively follows the permanently moved (301) redirections until it reaches the final URL
#    For DropBox we need to check the validity of the new URL but do not want to return to user the internal "raw" URL
checkbadURLS:
	-@x=`git grep "http[s]*://" -- '*.[chF]' '*.html' '*.cpp' '*.cxx' '*.cu' '*.F90' '*.py' '*.tex' | grep -E -v "(config/packages|HandsOnExercises)" | tr '[[:blank:]]' '\n' | grep 'http[s]*://' | sed 's!.*(\(http[s]*://[-a-zA-Z0-9_./()?=&+%~]*\))!\1!g' | sed 's!.*\(http[s]*://[-a-zA-Z0-9_./()?=&+%~]*\).*!\1!g' | sed 's/\.$$//g' | sort | uniq| awk '{ print length, $$0 }' | sort -r -n -s | cut -d" " -f2` ; \
        for i in $$x; do \
          url=$$i; \
          msg=''; \
          while [[ "$${msg}D" == "D" ]] ; do \
            y1=`curl --connect-timeout 5 --head --silent $${url} | head -n 1`; \
            y2=`echo $${y1} | grep ' 4[0-9]* '`; \
            y3=`echo $${y1} | grep ' 301 '`; \
            if [[ "$${y1}D" == "D" ]] ; then \
              msg="Unable to reach site" ; \
            elif [[ "$${y2}D" != "D" ]] ; then \
              msg=$${y1} ; \
            elif [[ "$${y3}D" != "D" ]] ; then \
              l=`curl --connect-timeout 5 --head --silent $${url} | grep ocation | sed 's/.*ocation:[[:blank:]]\(.*\)/\1/g' | tr -d '\r'` ; \
              w=`echo $$l | grep 'http'` ; \
              if [[ "$${w}D" != "D" ]] ; then \
                url=$$l ; \
              else \
                ws=`echo $${url} | sed 's!\(http[s]*://[-a-zA-Z0-9_.]*\)/.*!\1!g'` ; \
                dp=`echo $${ws}$${l} | grep "dropbox.com/s/raw"` ; \
                if [[ "$${dp}D" != "D" ]] ; then \
                  b=`curl --connect-timeout 5 --head --silent $${ws}$$l | head -n 1` ; \
                  c=`echo $${b} | grep -E "( 2[0-9]* | 302 )"` ; \
                  if [[ "$${c}D" == "D" ]] ; then \
                    msg=`echo "dropbox file doesn't exist" $${c}` ; \
                  else \
                    break ; \
                  fi; \
                else \
                  url="$${ws}$$l" ; \
                fi; \
              fi; \
            else \
              break; \
            fi; \
          done;\
          if [[ "$${msg}D" == "D" && "$${url}" != "$$i" ]] ; then \
            echo "URL" $$i "has moved to valid final location:" $${url} ; \
            if [[ "$${REPLACE}D" != "D" ]] ; then \
              git psed $$i $$l ;\
            fi; \
          elif [[ "$${msg}D" != "D" && "$${url}" != "$$i" ]] ; then \
            echo "ERROR: URL" $$i "has moved to invalid final location:" $${url} $${msg} ; \
          elif [[ "$${msg}D" != "D" ]] ; then \
            echo "ERROR: URL" $$i "invalid:" $${msg} ; \
          fi; \
        done
