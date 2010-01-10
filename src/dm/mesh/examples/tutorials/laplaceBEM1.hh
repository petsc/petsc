#include <problem/LaplaceBEM.hh>
#include <petscmesh_formats.hh>

class FunctionTestLaplaceBEM : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestLaplaceBEM);

  CPPUNIT_TEST(testLaplaceBEMSphere);
  CPPUNIT_TEST(testLaplaceBEMUnitSquareBdInterpolated);

#if 0
  CPPUNIT_TEST(testLaplaceBEMUnitSquareInterpolated);
  CPPUNIT_TEST(testLaplaceBEMUnitSquareUninterpolated);
#endif
  CPPUNIT_TEST_SUITE_END();
public:
  // Typedefs
protected:
  ALE::Obj<ALE::Problem::LaplaceBEM> _problem;
  int                                _debug; // The debugging level
  PetscInt                           _iters; // The number of test repetitions
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
    this->_problem = new ALE::Problem::LaplaceBEM(PETSC_COMM_WORLD, this->_debug);
  };

  /// Tear down data.
  void tearDown(void) {};

  void checkSolution(const double exactError, const double tolerance, const char testName[]) {
    SectionReal    solution;
    PetscReal      errorNorm;
    PetscErrorCode ierr;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "default", &solution);CHKERRXX(ierr);
    ierr = SectionRealToVec(solution, (::Mesh) this->_problem->getDM(), SCATTER_REVERSE, DMMGGetx(this->_problem->getDMMG()));CHKERRXX(ierr);
    SectionRealView(solution, PETSC_VIEWER_STDOUT_WORLD);
    ierr = this->_problem->calculateError(solution, &errorNorm);CHKERRXX(ierr);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exactError, errorNorm, tolerance);
  };

#if 0
  void testLaplaceBEMUnitSquare(double exactError) {
    this->_problem->structured(false);
    this->_problem->createMesh();
    this->_problem->createProblem();
    this->_problem->createExactSolution();
    PetscReal errorNorm;

    this->_problem->calculateError(this->_problem->exactSolution().section, &errorNorm);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exactError, errorNorm, 1.0e-6);
    this->_problem->createSolver();
    this->_problem->solve();
    this->checkSolution(exactError, 1.0e-6, "LaplaceBEMUnitSquare");
  };

  void testLaplaceBEMUnitSquareInterpolated(void) {
    this->_problem->interpolated(true);
    this->testLaplaceBEMUnitSquare(0.337731);
  };

  void testLaplaceBEMUnitSquareUninterpolated(void) {
    this->_problem->interpolated(false);
    this->testLaplaceBEMUnitSquare(0.336959);
  };
#endif

  void testLaplaceBEMUnitSquareBd(double exactError) {
    this->_problem->setDebug(2);
    this->_problem->bcType(ALE::Problem::DIRICHLET);
    this->_problem->structured(false);
    this->_problem->createMesh();
    this->_problem->getMesh()->view("Boundary mesh");
    this->_problem->getMesh()->setDebug(2);
    // What happens here?
    this->_problem->createProblem();
    this->_problem->createExactSolution();
    PetscReal errorNorm;

    this->_problem->calculateError(this->_problem->exactSolution().section, &errorNorm);
    //CPPUNIT_ASSERT_DOUBLES_EQUAL(exactError, errorNorm, 1.0e-6);
    this->_problem->createSolver();
    SectionReal exactSolution, residual;
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> r;

    MeshGetSectionReal((Mesh) this->_problem->getDM(), "exactSolution", &exactSolution);
    MeshGetSectionReal((Mesh) this->_problem->getDM(), "residual", &residual);
    SectionRealGetSection(residual, r);
    this->_problem->getMesh()->setupField(r);
    SectionRealView(exactSolution, PETSC_VIEWER_STDOUT_WORLD);
    ALE::Problem::Functions::RhsBd_Unstructured((Mesh) this->_problem->getDM(), exactSolution, residual, this->_problem->getOptions());
    this->_problem->getMesh()->getRealSection("residual")->view("residual");
    this->_problem->solve();
    double      coords[2] = {0.5, 0.5};
    PetscScalar sol;

    ALE::Problem::Functions::PointEvaluation((Mesh) this->_problem->getDM(), exactSolution, coords, 0.25, &sol);
    PetscPrintf(PETSC_COMM_WORLD, "Potential at (%g,%g): %g\n", coords[0], coords[1], sol);
    this->checkSolution(exactError, 1.0e-6, "LaplaceBEMUnitSquareBd");
  };

  void testLaplaceBEMUnitSquareBdInterpolated(void) {
    this->_problem->interpolated(true);
    this->testLaplaceBEMUnitSquareBd(0.356716);
  };

  void testLaplaceBEMSphere(void) {
    ALE::Obj<PETSC_MESH_TYPE> mesh = ALE::Bardhan::Builder::readMesh(PETSC_COMM_WORLD, 2, std::string("/home/knepley/Desktop/tmp2/surfaces/surf1QyXZD.mesh"), false, 0);

    mesh->view("Sphere");
    PetscViewer    viewer;
    PetscErrorCode ierr;

    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRXX(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRXX(ierr);
    ierr = PetscViewerFileSetName(viewer, "sphere.vtk");CHKERRXX(ierr);
    ierr = VTKViewer::writeHeader(viewer); CHKERRXX(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRXX(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRXX(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRXX(ierr);
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterLaplaceBEMFunctionSuite"
PetscErrorCode RegisterLaplaceBEMFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestLaplaceBEM);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
