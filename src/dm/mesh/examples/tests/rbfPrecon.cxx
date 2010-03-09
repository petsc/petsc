static char help[] = "Preconditioning FEM problems using RBF discretizations.\n\n";

// Next steps:
// 1) Make sure RBF is solving 2D Laplace
//   a) Figure out if it is a delta function representation
//   b) Put in LatticeDistribution
//   c) Change BC to homogeneous Neumann
//   d) Figure out solution for FEM problem and stick it in Bratu
//   e) Compare FEM and RBF as number of elements/samples becomes large (for RBF use RMS/Max error and FEM use L_2)
// 2) Figure out right interpolation for f (charges)
// 3) Code up projection into FEM space of RBF solution

#define ALE_MEM_LOGGING

#define NUM_TERMS 17
#define DIMENSION 2
#define NUM_COEFFICIENTS 17

#include <petscsys.h>
#include <problem/Bratu.hh>

// Needed for FMM
#include <Kernel_Laplacian_2D.hh>
#include <FMM.hh>

class Blob;

typedef FMM::Evaluator<Blob,FMM::Kernel<double, std::complex<double>, std::complex<double>, NUM_TERMS,DIMENSION,NUM_COEFFICIENTS>, DIMENSION> evaluator_type;
typedef evaluator_type::point_type point_type;
typedef evaluator_type::tree_type tree_type;
PetscLogEvent setupEvent, directEvaluationEvent, errorCheckEvent;

class Blob {
public:
  typedef double               circ_type;
  typedef std::complex<double> vel_type;
private:
  int        num;
  point_type position;
  circ_type  circulation;
  vel_type   velocity;
public:
  Blob(): num(0), position(point_type()), circulation(0.0), velocity(0.0) {};
  // TODO: Remove this. It is required by allocateStorage() in the section since I have value_type dummy(0)
  Blob(const int dummy): num(0), position(point_type()), circulation(0.0), velocity(0.0) {};
  Blob(const point_type& point): num(0), position(point), circulation(0.0), velocity(0.0) {};
  Blob(const int num, const point_type& point): num(num), position(point), circulation(0.0), velocity(0.0) {};
  Blob(const int num, const point_type& point, const circ_type& circulation) : num(num), position(point), circulation(circulation), velocity(0.0) {};
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
  void setVelocity(const vel_type velocity) {this->velocity = velocity;};
  void setVelocity(const double velocity[]) {this->velocity = vel_type(velocity[0], velocity[1]);};
  void addVelocity(const vel_type velocity) {this->velocity += velocity;};
  void addVelocity(const double velocity[]) {this->velocity += vel_type(velocity[0], velocity[1]);};
};

class RBFPreconditioner
{
public:
  // Typedefs
protected:
  ALE::Obj<ALE::Problem::Bratu> _problem;
  int                           _debug; // The debugging level
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
 
    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
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

  template<typename Tree, typename BlobSequence>
  void evaluateFMM(Tree& tree, const BlobSequence& blobs, FMM::Options<point_type>& options) {
    ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
    logger.setDebug(options.debug/2);
    logger.stagePush("FMM");

    try {
      evaluator_type evaluator(PETSC_COMM_SELF, options.debug);
      PetscErrorCode ierr;

      evaluator.evaluate(tree, blobs.begin(), blobs.end(), options.sigma*options.sigma, options.k);

      FMM::Output::runDescription(options, tree, blobs);
      if (options.calcError) {
        std::vector<typename Blob::vel_type> directOutput;
        std::vector<Blob>                    empty;

        ierr = PetscLogEventBegin(directEvaluationEvent,0,0,0,0);CHKERRXX(ierr);
        for(unsigned int i = 0; i < blobs.size(); ++i) directOutput.push_back(0.0);
        if (options.trueSoln == FMM::DIRECT_SOLUTION) {
          evaluator.getKernel().evaluate(directOutput, blobs.begin(), blobs.end(), empty.begin(), empty.end(), options.sigma*options.sigma, options.k);
        } else {
          throw ALE::Exception("Invalid true solution type");
        }
        ierr = PetscLogEventEnd(directEvaluationEvent,0,0,0,0);CHKERRXX(ierr);
        ierr = PetscLogEventBegin(errorCheckEvent,0,0,0,0);CHKERRXX(ierr);
        if (options.verify) {
          // Print experimental run data
          FMM::Output::csv("output.csv", blobs.begin(), blobs.end());
          std::ostringstream verifyName;
          verifyName << options.basename << "fmm." << blobs.size() << ".verify";
          FMM::Output::verify(verifyName.str(), options, tree, blobs, directOutput);
        }
        ierr = PetscLogEventEnd(errorCheckEvent,0,0,0,0);CHKERRXX(ierr);
      }
    } catch(PETSc::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
    logger.stagePop();
    std::cout << "FMM " << logger.getNumAllocations("FMM") << " allocations " << logger.getAllocationTotal("FMM") << " bytes" << std::endl;
    std::cout << "FMM " << logger.getNumDeallocations("FMM") << " deallocations " << logger.getDeallocationTotal("FMM") << " bytes" << std::endl;
  };

  void checkSolution(const double exactError, const double tolerance, const char testName[]) {
    SectionReal    solution;
    PetscReal      errorNorm;
    PetscErrorCode ierr;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "default", &solution);CHKERRXX(ierr);
    ierr = SectionRealToVec(solution, (::Mesh) this->_problem->getDM(), SCATTER_REVERSE, DMMGGetx(this->_problem->getDMMG()));CHKERRXX(ierr);
    ierr = this->_problem->calculateError(solution, &errorNorm);CHKERRXX(ierr);
    assert(std::abs(exactError - errorNorm) < tolerance);
  };

  void testBratuUnitSquare(double exactError) {
    this->_problem->structured(false);
    this->_problem->createMesh();
    this->_problem->createProblem();
    this->_problem->createExactSolution();
    PetscReal errorNorm;

    this->_problem->calculateError(this->_problem->exactSolution().section, &errorNorm);
    assert(std::abs(exactError - errorNorm) < 1.0e-6);
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

  void testBratuUnitSquareUninterpolatedPointCharge(void) {
    this->_problem->interpolated(false);
    this->_problem->structured(false);
    this->_problem->createMesh();
    this->_problem->createProblem();
    this->_problem->createExactSolution();
    PetscReal errorNorm;
    PetscReal exactError = 0.336959;

    this->_problem->calculateError(this->_problem->exactSolution().section, &errorNorm);
    assert(std::abs(exactError - errorNorm) < 1.0e-6);
    this->_problem->createSolver();
    this->_problem->solve();
    this->checkSolution(exactError, 1.0e-6, "BratuUnitSquare");
    Obj<PETSC_MESH_TYPE::real_section_type> s;
    SectionReal    solution;
    PetscErrorCode ierr;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "default", &solution);CHKERRXX(ierr);
    ierr = SectionRealGetSection(solution, s);CHKERRXX(ierr);
    ierr = SectionRealToVec(solution, (::Mesh) this->_problem->getDM(), SCATTER_REVERSE, DMMGGetx(this->_problem->getDMMG()));CHKERRXX(ierr);

    // Get blobs from vertices
    Obj<PETSC_MESH_TYPE> m;
    ierr = MeshGetMesh((::Mesh) this->_problem->getDM(), m);CHKERRXX(ierr);
    const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
    const Obj<PETSC_MESH_TYPE::label_sequence>&    vertices    = m->depthStratum(0);
    std::vector<Blob>                              blobs;
#if 0
    SectionReal                                    charges;
    SectionReal                                    zero;
    Obj<PETSC_MESH_TYPE::real_section_type>        q;
    Obj<PETSC_MESH_TYPE::real_section_type>        z;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "charges", &charges);CHKERRXX(ierr);
    ierr = SectionRealGetSection(charges, q);CHKERRXX(ierr);
    m->setupField(q);
    q->zeroWithBC();
    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "zero", &zero);CHKERRXX(ierr);
    ierr = SectionRealGetSection(zero, z);CHKERRXX(ierr);
    m->setupField(z);
    z->zeroWithBC();
    ierr = ALE::Problem::BratuFunctions::Rhs_Unstructured((::Mesh) this->_problem->getDM(), zero, charges, this->_problem->getOptions());CHKERRXX(ierr);
    q->view("Charges");
#endif
    for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
      const int fiberDim = s->getFiberDimension(*v_iter);

      // Check for boundary points
      if (fiberDim > 0) {
        // Add blob coefficient equal to the driving force (charge)
        const PETSC_MESH_TYPE::real_section_type::value_type *coords = coordinates->restrictPoint(*v_iter);
        const int                                             dim    = s->getConstrainedFiberDimension(*v_iter);

        if (dim) {
          blobs.push_back(Blob(*v_iter, coords, -this->_problem->getOptions()->func(coords)*0.0625));
        } else {
          blobs.push_back(Blob(*v_iter, coords, 0.0));
        }
      }
    }
    std::cout << "Input Blobs:" << std::endl;
    for(std::vector<Blob>::const_iterator b_iter = blobs.begin(); b_iter != blobs.end(); ++b_iter) {
      std::cout << *b_iter << std::endl;
    }

    // Solve FMM
    FMM::Options<point_type>  options;
    evaluator_type::tree_type tree(options.maxLevel+1, options.debug);

    // Set sigma about the element diameter
    options.sigma = 1.0;
    evaluateFMM(tree, blobs, options);
    if (!this->_problem->commRank()) {
      std::cout << "Output Blobs:" << std::endl;
      const tree_type::BlobSection       *blobSection = tree.getBlobs();
      const tree_type::BlobSection::chart_type& chart = blobSection->getChart();

      for(tree_type::BlobSection::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const tree_type::BlobSection::value_type *values   = blobSection->restrictPoint(*c_iter);
        const int                                 numBlobs = blobSection->getFiberDimension(*c_iter);

        for(int b = 0; b < numBlobs; ++b) {
          std::cout << values[b] << std::endl;
        }
      }
    }
    // Project back to FEM space
    //   This is easy here since the evaluation points are also the vertices
    Obj<PETSC_MESH_TYPE::real_section_type> sol;

    ierr = MeshGetSectionReal((::Mesh) this->_problem->getDM(), "default", &solution);CHKERRXX(ierr);
    ierr = SectionRealGetSection(solution, sol);CHKERRXX(ierr);
    if (!this->_problem->commRank()) {
      const tree_type::BlobSection       *blobSection = tree.getBlobs();
      const tree_type::BlobSection::chart_type& chart = blobSection->getChart();

      for(tree_type::BlobSection::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const tree_type::BlobSection::value_type *values   = blobSection->restrictPoint(*c_iter);
        const int                                 numBlobs = blobSection->getFiberDimension(*c_iter);

        for(int b = 0; b < numBlobs; ++b) {
          PETSC_MESH_TYPE::real_section_type::value_type potential = values[b].getVelocity().real();

          sol->updatePoint(values[b].getNum(), &potential);
        }
      }
    }
    sol->view("RBF solution");
    ///ierr = SectionRealToVec(solution, (::Mesh) this->_problem->getDM(), SCATTER_FORWARD, DMMGGetx(this->_problem->getDMMG()));CHKERRXX(ierr);
    // Check error
    //   Look at convergence as a function of h
    this->_problem->calculateError(solution, &errorNorm);
    std::cout << "Error in RBF solution: " << errorNorm << std::endl;
  };
};

#undef __FUNCT__
#define __FUNCT__ "RunTests"
PetscErrorCode RunTests()
{
  RBFPreconditioner pc;

  PetscFunctionBegin;
  pc.setUp();
  pc.testBratuUnitSquareUninterpolated();
  pc.testBratuUnitSquareInterpolated();
  pc.testBratuUnitSquareUninterpolatedPointCharge();
  pc.tearDown();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = RunTests();CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
