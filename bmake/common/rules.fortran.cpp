#
#  The rules for Fortran targets are in a seperate file because on systems
# with a Fortran compiler that does not use cpp we need different rules
#
#  These targets are for systems where Fortran uses cpp, rules.fortran.nocpp
# are for systems where Fortran does not use cpp
#
.F.o .F90.o:
	-@if [ "${PETSC_HAVE_ADIFOR}" != "" ]; then \
	   egrep "Process adifor" $< > /dev/null ; \
              if [ "$$?" = 0 ]; then \
                r=`grep "Process adifor" $< | sed -e "s?![ ]*Process adifor: \([a-zA-Z0-9]\)[ ]*?\1?g"`;\
                echo Running adifor on function $${r};\
                ${RM} $*.tmp $*.tmp.f $*.ad.F silly.cmp ;\
                echo "#include \"$<\"" > $*.ad.F;\
                ${FC} -E ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $< | grep -v '^ *#' > $*.tmp ; \
                ${PETSC_DIR}/bin/adprocess.py $*.tmp $${r}; \
                echo $*.tmp.f > silly.cmp; \
                echo ${ADIFOR_FC} AD_TOP=$${r}; \
	        ${ADIFOR_FC} AD_TOP=$${r} ; \
                sed -e 's/^C/\!/g' -e 's/^     \*/     \&/g' g_$*.tmp.f | ${PETSC_DIR}/bin/adiforfix.py >> $*.ad.F;\
	        ${ADIFOR_FC} AD_TOP=$${r} AD_SCALAR_GRADIENTS=true AD_PMAX=1 AD_PREFIX=m ; \
                sed -e 's/^C/\!/g' -e 's/^     \*/     \&/g' m_$*.tmp.f | ${PETSC_DIR}/bin/adiforfix.py >> $*.ad.F;\
	        echo ${FC} -o $*.o -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $*.ad.F ;\
	        ${FC} -o $*.o -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $*.ad.F ;\
              else \
                echo ${FC} -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $< ; \
                ${FC} -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $< ; \
              fi; \
        else \
          echo ${FC} -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $< ; \
          ${FC} -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $< ; \
        fi
.F.a: 
	-${FC} -c ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} $<
	-${AR} ${AR_FLAGS} ${LIBNAME} $*.o
	-${RM} $*.o

.f.o .f90.o: 
	-${FC} -c ${FFLAGS} ${FOPTFLAGS} $<
.f.a: 
	-${FC} -c ${FFLAGS} ${FOPTFLAGS} $<
	-${AR} ${AR_FLAGS} ${LIBNAME} $*.o
	-${RM} $*.o




