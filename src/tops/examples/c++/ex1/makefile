
SIDL        = ex1.sidl
SIDLEXCLUDE = -e=TOPS[a-zA-Z_]* -e=gov_cca[_a-zA-Z_]*
TLIBNAME    = libex1
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl* ./server/c++/gov_cca*
DIRS        = server/c++

ex1:ex1.cc
	        ${CC_LINKER}  -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ ${BABEL_INCLUDE}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}
	#${CC_LINKER} -multiply_defined suppress -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} -L${BABEL_lib} -lsidl ${SYS_lib}
	#${CC_LINKER} -o ex1 -g ${MPI_include} -I../../../server/c++/ \
		-I../../../client/c++ -I${BABEL_include}  ex1.cc \
		-L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} \
		-lclient-c++ ${PETSC_LIB} -L${BABEL_lib} -lsidl ${SYS_lib} 

include ${PETSC_DIR}/src/tops/makefile.rules

cca: 
	# Generating *.cca files
	@${PETSC_DIR}/src/tops/utils/genSCLCCA.sh cca ${PETSC_LIB_DIR}/libtops.${SL_LINKER_SUFFIX} \
		TOPS.StructuredSolver c++ dynamic global now TOPS.StructuredSolver \
		> libtops.${SL_LINKER_SUFFIX}.cca.in
	@sed -e /"@PETSC_LIB_DIR@/ s|@PETSC_LIB_DIR@|${PETSC_LIB_DIR}|" \
		-e /"@SOSUFFIX@/ s|@SOSUFFIX@|${SL_LINKER_SUFFIX}|" \
		libtops.${SL_LINKER_SUFFIX}.cca.in >  ${PETSC_LIB_DIR}/libtops.${SL_LINKER_SUFFIX}.cca
	@${PETSC_DIR}/src/tops/utils/genSCLCCA.sh cca ${PETSC_LIB_DIR}/libex1.${SL_LINKER_SUFFIX} \
		Ex1.System c++ dynamic global now Ex1.System > libex1.${SL_LINKER_SUFFIX}.cca.in
	@sed -e /"@PETSC_LIB_DIR@/ s|@PETSC_LIB_DIR@|${PETSC_LIB_DIR}|" \
	    	-e /"@SOSUFFIX@/ s|@SOSUFFIX@|${SL_LINKER_SUFFIX}|" \
		libex1.${SL_LINKER_SUFFIX}.cca.in > ${PETSC_LIB_DIR}/libex1.${SL_LINKER_SUFFIX}.cca
	@-${RM} *.cca.in
	# Creating simple Ccaffeine script
	@sed -e /"@PETSC_LIB_DIR@/ s|@PETSC_LIB_DIR@|${PETSC_LIB_DIR}|" ex1_rc.in > ex1_rc

test-cca: cca
	# Testing component example with Ccaffeine
	@echo "   ${CCAFE_HOME}/bin/ccafe-single --ccafe-rc ex1_rc > ex1_rc.log 2>1;"
	@${CCAFE_HOME}/bin/ccafe-single --ccafe-rc ex1_rc > ex1_rc.log 2>1; 
	@if [ "`grep \"specific go command successful\" ex1_rc.log | wc -l`" == "1" \
	  -a "`grep \"2 SNES Function norm \" ex1_rc.log | wc -l `" == "1" ];then \
		echo -e "\n=== Component test passed (see ex1_rc.log).\n"; \
	else echo -e "\n*** Component test failed (see ex1_rc.log).\n"; fi
