#include <problem/Bratu.hh>

class FunctionTestBratu : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestBratu);

  CPPUNIT_TEST(testBratuUnitSquareInterpolated);
  CPPUNIT_TEST(testBratuUnitSquareUninterpolated);

  CPPUNIT_TEST_SUITE_END();
public:
  // Typedefs
protected:
  ALE::Obj<ALE::Problem::Bratu> _problem;
  int                           _debug; // The debugging level
  PetscInt                      _iters; // The number of test repetitions
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_problem = new ALE::Problem::Bratu(PETSC_COMM_WORLD, this->_debug);
  };

  /// Tear down data.
  void tearDown(void) {};

  void checkSolution(const double exactError, const double tolerance, const char testName[]) {
    SectionReal    solution;
    PetscReal      errorNorm;
    PetscErrorCode ierr;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "default", &solution);CHKERRXX(ierr);
    ierr = SectionRealToVec(solution, (::Mesh) this->_problem->getDM(), SCATTER_REVERSE, DMMGGetx(this->_problem->getDMMG()));CHKERRXX(ierr);
    ierr = this->_problem->calculateError(solution, &errorNorm);CHKERRXX(ierr);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exactError, errorNorm, tolerance);
  };

  void testBratuUnitSquare(double exactError) {
    this->_problem->structured(false);
    this->_problem->createMesh();
    this->_problem->createProblem();
    this->_problem->createExactSolution();
    PetscReal errorNorm;

    this->_problem->calculateError(this->_problem->exactSolution().section, &errorNorm);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exactError, errorNorm, 1.0e-6);
    this->_problem->createSolver();
    this->_problem->solve();
    this->checkSolution(exactError, 1.0e-6, "BratuUnitSquare");
  };

  void testBratuUnitSquareInterpolated(void) {
    this->_problem->interpolated(true);
    this->testBratuUnitSquare(0.337731);
  };

  void testBratuUnitSquareUninterpolated(void) {
    this->_problem->interpolated(false);
    this->testBratuUnitSquare(0.336959);
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuFunctionSuite"
PetscErrorCode RegisterBratuFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestBratu);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuStressSuite"
PetscErrorCode RegisterBratuStressSuite() {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuConvergenceSuite"
PetscErrorCode RegisterBratuConvergenceSuite() {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
