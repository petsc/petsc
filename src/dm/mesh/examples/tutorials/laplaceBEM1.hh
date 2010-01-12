#include <problem/LaplaceBEM.hh>
#include <petscmesh_formats.hh>

class Blob;

typedef FMM::Evaluator<Blob,FMM::Kernel<PETSc::Point<3>,std::complex<double>,PETSc::Point<3>,NUM_TERMS,DIMENSION,NUM_COEFFICIENTS>, DIMENSION> evaluator_type;
typedef evaluator_type::point_type point_type;
typedef evaluator_type::tree_type  tree_type;

/*
 * Initialization of particle circulation from a vector with circulations.
 */
template<typename Blob>
class VecCirculation {
public:
  typedef Blob blob_type;
protected:
  Vec gx, gy, gz;
public:
  VecCirculation(const Vec* c) {gx = c[0]; gy = c[1]; gz = c[2];};
  #undef __FUNCT__   
  #define __FUNCT__ "VecCirculationGet"
  Blob& operator()(Blob& blob) const {
    PetscInt    ix;
    PetscScalar gxd;
    PetscScalar gyd;
    PetscScalar gzd;
    PetscErrorCode ierr;

    ix = blob.getNum();
    ierr = VecGetValues(gx, 1, &ix, &gxd);CHKERRXX(ierr);
    ierr = VecGetValues(gy, 1, &ix, &gyd);CHKERRXX(ierr);
    ierr = VecGetValues(gz, 1, &ix, &gzd);CHKERRXX(ierr);
    double circulation[3] = {gxd, gyd, gzd};
    blob.setCirculation(point_type(circulation));
    return blob;
  };
};

class Blob {
public:
  typedef point_type circ_type;
  typedef point_type vel_type;
private:
  int        num;
  point_type position;
  circ_type  circulation;
  vel_type   velocity;
public:
  Blob(): num(0), position(point_type()), circulation(0.0), velocity(0.0) {};
  Blob(const int dummy): num(0), position(point_type()), circulation(0.0), velocity(0.0) {};
  Blob(const point_type& point): num(0), position(point), circulation(0.0), velocity(0.0) {};
  Blob(const int num, const point_type& point): num(num), position(point), circulation(0.0), velocity(0.0) {};
  Blob(const Blob& blob): num(blob.num), position(blob.position), circulation(blob.circulation), velocity(blob.velocity) {};
public:
  friend std::ostream& operator<<(std::ostream& stream, const Blob& blob) {
    stream << "blob num " << blob.num << std::endl;
    stream << "blob pos " << blob.position << std::endl;
    stream << "blob circ " << blob.circulation << std::endl;
    stream << "blob vel " << blob.velocity << std::endl;
    return stream;
  };
public:
  void clear() {};
  int getNum() const {return this->num;};
  point_type getPosition() const {return this->position;};
  void setPosition(const point_type& position) {this->position = position;};
  circ_type getCirculation() const {return this->circulation;};
  void setCirculation(const circ_type circulation) {this->circulation = circulation;};
  vel_type getVelocity() const {return this->velocity;};
  void setVelocity(const vel_type velocity) {this->velocity  = velocity;};
  void addVelocity(const vel_type velocity) {this->velocity += velocity;};
};

class FunctionTestLaplaceBEM : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestLaplaceBEM);

  CPPUNIT_TEST(testLaplaceBEMSphere);
#if 0
  CPPUNIT_TEST(testLaplaceBEMUnitSquareBdInterpolated);
#endif

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

  void calculateBoundingBox(ALE::Obj<PETSC_MESH_TYPE>& mesh, double lower[], double upper[], double margin) {
    const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = mesh->getRealSection("coordinates");
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&    vertices    = mesh->depthStratum(0);
    const int                                           dim         = mesh->getDimension()+1;

    {
      const double *coords = coordinates->restrictPoint(*vertices->begin());

      for(int d = 0; d < dim; ++d) {
        lower[d] = coords[d];
        upper[d] = coords[d];
      }
    }
    for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
      const double *coords = coordinates->restrictPoint(*v_iter);

      for(int d = 0; d < dim; ++d) {
        if (lower[d] > coords[d]) {lower[d] = coords[d];}
        if (upper[d] < coords[d]) {upper[d] = coords[d];}
      }
    }
  };

  void constructBlobs(ALE::Obj<PETSC_MESH_TYPE>& mesh, Vec position[], Vec circulation[]) {
    const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = mesh->getRealSection("coordinates");
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&    cells       = mesh->heightStratum(0);
    const int                                           dim         = mesh->getDimension()+1;
    const int                                           numBlobs    = cells->size();
    const int                                           numCorners  = mesh->sizeWithBC(coordinates, *cells->begin())/dim;
    PetscInt                                            c           = 0;
    PetscErrorCode                                      ierr;

    ierr = VecCreate(PETSC_COMM_WORLD, &position[0]);CHKERRXX(ierr);
    ierr = VecSetSizes(position[0], PETSC_DECIDE, numBlobs);CHKERRXX(ierr);
    ierr = VecSetFromOptions(position[0]);CHKERRXX(ierr);
    ierr = VecDuplicate(position[0], &position[1]);CHKERRXX(ierr);
    ierr = VecDuplicate(position[0], &position[2]);CHKERRXX(ierr);
    ierr = VecDuplicate(position[0], &circulation[0]);CHKERRXX(ierr);
    ierr = VecDuplicate(position[0], &circulation[1]);CHKERRXX(ierr);
    ierr = VecDuplicate(position[0], &circulation[2]);CHKERRXX(ierr);

    for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter, ++c) {
      const double *coords = mesh->restrictClosure(coordinates, *c_iter);
      double centroid[3] = {0.0, 0.0, 0.0};

      for(int p = 0; p < numCorners; ++p) {
        for(int d = 0; d < dim; ++d) {
          centroid[d] += coords[p*dim+d];
        }
      }
      for(int d = 0; d < dim; ++d) {
        ierr = VecSetValue(position[d], c, centroid[d], INSERT_VALUES);CHKERRXX(ierr);
      }

      // Area
      //   |v \times w| = |v_x w_y \hat z - v_x w_z \hat y + v_y w_z \hat x - v_y w_x \hat z + v_z w_x \hat y - v_z w_y \hat x|
      //                = |(v_x w_y - v_y w_x) \hat z + (v_z w_x - v_x w_z) \hat y + (v_y w_z - v_z w_y) \hat x|
      //                = sqrt((v_x w_y - v_y w_x)^2 + (v_z w_x - v_x w_z)^2 + (v_y w_z - v_z w_y)^2)
      double v[3] = {coords[1*dim+0] - coords[0*dim+0], coords[1*dim+1] - coords[0*dim+1], coords[1*dim+2] - coords[0*dim+2]};
      double w[3] = {coords[2*dim+0] - coords[0*dim+0], coords[2*dim+1] - coords[0*dim+1], coords[2*dim+2] - coords[0*dim+2]};
      double area = sqrt(PetscSqr(v[0]*w[1] - v[1]*w[0]) + PetscSqr(v[2]*w[0] - v[0]*w[2]) + PetscSqr(v[1]*w[2] - v[2]*w[1]));

      // Quadrature weights
      for(int d = 0; d < dim; ++d) {
        ierr = VecSetValue(circulation[d], c, area, INSERT_VALUES);CHKERRXX(ierr);
      }
    }    
  };

  void createSolutionSection(tree_type& tree, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& potential) {
    const tree_type::BlobSection             *blobSection = tree.getBlobs();
    const tree_type::BlobSection::chart_type& chart       = blobSection->getChart();

    potential->setChart(PETSC_MESH_TYPE::real_section_type::chart_type(0, blobSection->size()));
    for(tree_type::BlobSection::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      const tree_type::BlobSection::value_type *values   = blobSection->restrictPoint(*c_iter);
      const int                                 numBlobs = blobSection->getFiberDimension(*c_iter);

      for(int b = 0; b < numBlobs; ++b) {
        potential->setFiberDimension(values[b].getNum(), values[b].getVelocity().size());
      }
    }
    potential->allocatePoint();
    for(tree_type::BlobSection::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
      const tree_type::BlobSection::value_type *values   = blobSection->restrictPoint(*c_iter);
      const int                                 numBlobs = blobSection->getFiberDimension(*c_iter);

      for(int b = 0; b < numBlobs; ++b) {
        potential->updatePoint(values[b].getNum(), (double *) values[b].getVelocity());
      }
    }
  };

  void visualizeField(ALE::Obj<PETSC_MESH_TYPE>& mesh, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& potential) {
    const int      depth    = mesh->depth();
    const int      fiberDim = 3;
    Obj<PETSC_MESH_TYPE::numbering_type> cNumbering = mesh->getFactory()->getNumbering(mesh, depth);
    PetscViewer    viewer;
    PetscErrorCode ierr;

    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRXX(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRXX(ierr);
    ierr = PetscViewerFileSetName(viewer, "sphere.vtk");CHKERRXX(ierr);
    ierr = VTKViewer::writeHeader(viewer); CHKERRXX(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRXX(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRXX(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", cNumbering->getGlobalSize());CHKERRXX(ierr);
    ierr = VTKViewer::writeField(potential, "Potential", fiberDim, cNumbering, viewer);CHKERRXX(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRXX(ierr);
  };

  void testLaplaceBEMSphere(void) {
    ALE::Obj<PETSC_MESH_TYPE> mesh = ALE::Bardhan::Builder::readMesh(PETSC_COMM_WORLD, 2, std::string("/home/knepley/Desktop/tmp2/surfaces/surf1QyXZD.mesh"), false, 0);
    FMM::Options<point_type>  options;
    double lower[3];
    double upper[3];
    Vec    position[3];
    Vec    circulation[3];

    calculateBoundingBox(mesh, lower, upper, 0.05);
    constructBlobs(mesh, position, circulation);
    FMM::Utils<>::PetscVecSequence3D<point_type,VecCirculation<Blob> > particles(VecCirculation<Blob>(circulation), position);

    evaluator_type evaluator(mesh->comm(), options.debug);
    tree_type      tree(options.maxLevel+1, options.debug);
    FMM::Output::runDescription(options, tree, particles);
    evaluator.evaluate(tree, particles.begin(), particles.end(), 2*options.sigma*options.sigma, options.k);

    std::vector<Blob::vel_type> directOutput;
    for(unsigned int i = 0; i < particles.size(); ++i) {
      directOutput.push_back(1.0);
    }
    FMM::Output::computeError(options, tree, particles, directOutput);
    createSolutionSection(tree, mesh->getRealSection("potential"));
    visualizeField(mesh, mesh->getRealSection("potential"));
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterLaplaceBEMFunctionSuite"
PetscErrorCode RegisterLaplaceBEMFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestLaplaceBEM);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
