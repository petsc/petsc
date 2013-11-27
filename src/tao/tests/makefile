
LOCDIR = src/tao/tests/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

test_link1: test_link1.o chkopts

test_link1f: test_link1f.o chkopts
	-${FLINKER} -o test_link1f test_link1f.o ${PETSC_TAO_LIB}
	${RM} test_link1f.o


test_subvec1: test_subvec1.o chkopts
	-${CLINKER} -o test_subvec1 test_subvec1.o ${PETSC_TAO_LIB}
	${RM} test_subvec1.o

test_submat1: test_submat1.o chkopts
	-${CLINKER} -o test_submat1 test_submat1.o ${PETSC_TAO_LIB}
	${RM} test_submat1.o


test_linesearch1: test_linesearch1.o chkopts
	-${CLINKER} -o test_linesearch1 test_linesearch1.o ${PETSC_TAO_LIB}
	${RM} test_linesearch1.o

test_lmvmmat1: test_lmvmmat1.o chkopts
	-${CLINKER} -o test_lmvmmat1 test_lmvmmat1.o ${PETSC_TAO_LIB}
	${RM} test_lmvmmat1.o

test_approxmat1: test_approxmat1.o chkopts
	-${CLINKER} -o test_approxmat1 test_approxmat1.o ${PETSC_TAO_LIB}
	${RM} test_approxmat1.o

test_rosenbrock1: test_rosenbrock1.o chkopts
	-${CLINKER} -o test_rosenbrock1 test_rosenbrock1.o ${PETSC_TAO_LIB}
	${RM} test_rosenbrock1.o

test_limit_feval: test_limit_feval.o chkopts
	-${CLINKER} -o test_limit_feval test_limit_feval.o ${PETSC_TAO_LIB}
	${RM} test_limit_feval.o

test_limit_iter: test_limit_iter.o chkopts
	-${CLINKER} -o test_limit_iter test_limit_iter.o ${PETSC_TAO_LIB}
	${RM} test_limit_iter.o

test_limit_minf: test_limit_minf.o chkopts
	-${CLINKER} -o test_limit_minf test_limit_minf.o ${PETSC_TAO_LIB}
	${RM} test_limit_minf.o


test_rosenbrock1f: test_rosenbrock1f.o chkopts
	-${FLINKER} -o  test_rosenbrock1f test_rosenbrock1f.o ${PETSC_TAO_LIB}
	${RM} test_rosenbrock1f.o


test_limit_fevalf: test_limit_fevalf.o chkopts
	-${FLINKER} -o  test_limit_fevalf test_limit_fevalf.o ${PETSC_TAO_LIB}
	${RM} test_limit_fevalf.o


test_limit_iterf: test_limit_iterf.o chkopts
	-${FLINKER} -o  test_limit_iterf test_limit_iterf.o ${PETSC_TAO_LIB}
	${RM} test_limit_iterf.o


test_limit_minff: test_limit_minff.o chkopts
	-${FLINKER} -o  test_limit_minff test_limit_minff.o ${PETSC_TAO_LIB}
	${RM} test_limit_minff.o

test_prefix: test_prefix.o chkopts
	-${CLINKER} -o test_prefix test_prefix.o ${PETSC_TAO_LIB}
	${RM} test_prefix.o

test_prefixf: test_prefixf.o chkopts
	-${FLINKER} -o test_prefixf test_prefixf.o ${PETSC_TAO_LIB}
	${RM} test_prefixf.o

test_jbearing2: test_jbearing2.o chkopts
	-${CLINKER} -o test_jbearing2 test_jbearing2.o ${PETSC_TAO_LIB}
	${RM} test_jbearing2.o

test_gts: test_gts.o chkopts
	-${CLINKER} -o test_gts test_gts.o ${PETSC_TAO_LIB}
	${RM} test_gts.o

test_linesearch_fg: test_linesearch_fg.o chkopts
	-${CLINKER} -o test_linesearch_fg test_linesearch_fg.o ${PETSC_TAO_LIB}
	${RM} test_linesearch_fg.o

test_gtsf: test_gtsf.o chkopts
	-${FLINKER} -o test_gtsf test_gtsf.o ${PETSC_TAO_LIB}
	${RM} test_gtsf.o


test_jbearingdm: test_jbearingdm.o chkopts
	-${CLINKER} -o test_jbearingdm test_jbearingdm.o ${PETSC_TAO_LIB}
	${RM} test_jbearingdm.o



test_jbearingdmf: test_jbearingdmf.o chkopts
	-${FLINKER} -o test_jbearingdmf test_jbearingdm.o ${PETSC_TAO_LIB}
	${RM} test_jbearingdmf.o

test_plate2: test_plate2.o chkopts
	-${CLINKER} -o test_plate2 test_plate2.o ${PETSC_TAO_LIB}
	${RM} test_plate2.o

test_plate2f: test_plate2f.o chkopts
	-${FLINKER} -o test_plate2f test_plate2f.o ${PETSC_TAO_LIB}
	${RM} test_plate2f.o

test_eptorsion1: test_eptorsion1.o chkopts
	-${CLINKER} -o test_eptorsion1 test_eptorsion1.o ${PETSC_TAO_LIB}
	${RM} test_eptorsion1.o

test_eptorsion2: test_eptorsion2.o chkopts
	-${CLINKER} -o test_eptorsion2 test_eptorsion2.o ${PETSC_TAO_LIB}
	${RM} test_eptorsion2.o

test_eptorsion2f: test_eptorsion2f.o chkopts
	-${FLINKER} -o test_eptorsion2f test_eptorsion2f.o ${PETSC_TAO_LIB}
	${RM} test_eptorsion2f.o

test_minsurf1: test_minsurf1.o chkopts
	-${CLINKER} -o test_minsurf1 test_minsurf1.o ${PETSC_TAO_LIB}
	${RM} test_minsurf1.o

test_minsurf3: test_minsurf3.o chkopts
	-${CLINKER} -o test_minsurf3 test_minsurf3.o ${PETSC_TAO_LIB}
	${RM} test_minsurf3.o

test_dgesv: test_dgesv.o chkopts
	-${CLINKER} -o test_dgesv test_dgesv.o  -llapack -lblas

test_minsurf2: test_minsurf2.o chkopts
	-${CLINKER} -o test_minsurf2 test_minsurf2.o ${PETSC_TAO_LIB}
	${RM} test_minsurf2.o

test_bratu2: test_bratu2.o chkopts
	-${CLINKER} -o test_bratu2 test_bratu2.o ${PETSC_TAO_LIB}
	${RM} test_bratu2.o

test_chwirut1: test_chwirut1.o chkopts
	-${CLINKER} -o test_chwirut1 test_chwirut1.o ${PETSC_TAO_LIB}
	${RM} test_chwirut1.o

test_chwirut2: test_chwirut2.o chkopts
	-${CLINKER} -o test_chwirut2 test_chwirut2.o ${PETSC_TAO_LIB}
	${RM} test_chwirut2.o

test_chebyq1: test_chebyq1.o chkopts
	-${CLINKER} -o test_chebyq1 test_chebyq1.o ${PETSC_TAO_LIB}
	${RM} test_chebyq1.o

test_copper1: test_copper1.o chkopts
	-${CLINKER} -o test_copper1 test_copper1.o ${PETSC_TAO_LIB}
	${RM} test_copper1.o

test_chwirut1f: test_chwirut1f.o chkopts
	-${FLINKER} -o  test_chwirut1f test_chwirut1f.o ${PETSC_TAO_LIB}
	${RM} test_chwirut1f.o

test_chwirut2f: test_chwirut2f.o chkopts
	-${FLINKER} -o  test_chwirut2f test_chwirut2f.o ${PETSC_TAO_LIB}
	${RM} test_chwirut2f.o

test_toy: test_toy.o chkopts
	-${CLINKER} -o test_toy test_toy.o ${PETSC_TAO_LIB}
	${RM} test_toy.o

test_hyperbolic: test_hyperbolic.o chkopts
	-${CLINKER} -o test_hyperbolic test_hyperbolic.o ${PETSC_TAO_LIB}
	${RM} test_hyperbolic.o

test_parabolic: test_parabolic.o chkopts
	-${CLINKER} -o test_parabolic test_parabolic.o ${PETSC_TAO_LIB}
	${RM} test_parabolic.o

test_elliptic: test_elliptic.o chkopts
	-${CLINKER} -o test_elliptic test_elliptic.o ${PETSC_TAO_LIB}
	${RM} test_elliptic.o

test_bqpip: test_bqpip.o chkopts
	-${CLINKER} -o test_bqpip test_bqpip.o ${PETSC_TAO_LIB}
	${RM} test_bqpip.o

test_chwirut1_bound: test_chwirut1_bound.o chkopts
	-${CLINKER} -o test_chwirut1_bound test_chwirut1_bound.o ${PETSC_TAO_LIB}
	${RM} test_chwirut1_bound.o

test_chwirut2_bound: test_chwirut2_bound.o chkopts
	-${CLINKER} -o test_chwirut2_bound test_chwirut2_bound.o ${PETSC_TAO_LIB}
	${RM} test_chwirut2_bound.o
