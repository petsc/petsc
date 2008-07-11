#ifndef included_ALE_Problem_Bratu_hh
#define included_ALE_Problem_Bratu_hh

#include <DMBuilder.hh>

// How do we do this correctly?
#include "../examples/tutorials/bratu_quadrature.h"

namespace ALE {
  namespace Problem {
    namespace Functions {
      static PetscScalar lambda;

      PetscScalar zero(const double x[]) {
        return 0.0;
      };

      PetscScalar constant(const double x[]) {
        return -4.0;
      };

      PetscScalar nonlinear_2d(const double x[]) {
        return -4.0 - lambda*PetscExpScalar(x[0]*x[0] + x[1]*x[1]);
      };

      PetscScalar singularity_2d(const double x[]) {
        return 0.;
      };

      PetscScalar singularity_exact_2d(const double x[]) {
        double r = sqrt(x[0]*x[0] + x[1]*x[1]);
        double theta;
        if (r == 0.) {
          return 0.;
        } else theta = asin(x[1]/r);
        if (x[0] < 0) {
          theta = 2*M_PI - theta;
        }
        return pow(r, 2./3.)*sin((2./3.)*theta);
      };

      PetscScalar singularity_exact_3d(const double x[]) {
        return sin(x[0] + x[1] + x[2]);  
      };

      PetscScalar singularity_3d(const double x[]) {
        return (3)*sin(x[0] + x[1] + x[2]);
      };

      PetscScalar linear_2d(const double x[]) {
        return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5);
      };

      PetscScalar quadratic_2d(const double x[]) {
        return x[0]*x[0] + x[1]*x[1];
      };

      PetscScalar cubic_2d(const double x[]) {
        return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + 0.5;
      };

      PetscScalar nonlinear_3d(const double x[]) {
        return -4.0 - lambda*PetscExpScalar((2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
      };

      PetscScalar linear_3d(const double x[]) {
        return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5) - 6.0*(x[2] - 0.5);
      };

      PetscScalar quadratic_3d(const double x[]) {
        return (2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      };

      PetscScalar cubic_3d(const double x[]) {
        return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + x[2]*x[2]*x[2] - 1.5*x[2]*x[2] + 0.75;
      };

      PetscScalar cos_x(const double x[]) {
        return cos(2.0*PETSC_PI*x[0]);
      };
    };
    class Bratu : ALE::ParallelObject {
    public:
      typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
      typedef enum {NEUMANN, DIRICHLET} BCType;
      typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
      typedef union {SectionReal section; Vec vec;} ExactSolType;
      typedef struct {
        PetscInt      debug;                       // The debugging level
        RunType       run;                         // The run type
        PetscInt      dim;                         // The topological mesh dimension
        PetscTruth    reentrantMesh;               // Generate a reentrant mesh?
        PetscTruth    circularMesh;                // Generate a circular mesh?
        PetscTruth    refineSingularity;           // Generate an a priori graded mesh for the poisson problem
        PetscTruth    structured;                  // Use a structured mesh
        PetscTruth    generateMesh;                // Generate the unstructure mesh
        PetscTruth    interpolate;                 // Generate intermediate mesh elements
        PetscReal     refinementLimit;             // The largest allowable cell volume
        char          baseFilename[2048];          // The base filename for mesh files
        char          partitioner[2048];           // The graph partitioner
        PetscScalar (*func)(const double []);      // The function to project
        BCType        bcType;                      // The type of boundary conditions
        PetscScalar (*exactFunc)(const double []); // The exact solution function
        ExactSolType  exactSol;                    // The discrete exact solution
        ExactSolType  error;                       // The discrete cell-wise error
        AssemblyType  operatorAssembly;            // The type of operator assembly 
        double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
        double        lambda;                      // The parameter controlling nonlinearity
        double        reentrant_angle;              // The angle for the reentrant corner.
      } Options;
    protected:
      Options              _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
    public:
      Bratu(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm = PETSC_NULL;
      };
      ~Bratu() {
        PetscErrorCode ierr = this->destroyMesh();CHKERRXX(ierr);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "BratuProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, Options *options) {
        const char    *runTypes[3] = {"full", "test", "mesh"};
        const char    *bcTypes[2]  = {"neumann", "dirichlet"};
        const char    *asTypes[4]  = {"full", "stored", "calculated"};
        ostringstream  filename;
        PetscInt       run, bc, as;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        options->debug            = 0;
        options->run              = RUN_FULL;
        options->dim              = 2;
        options->structured       = PETSC_TRUE;
        options->generateMesh     = PETSC_TRUE;
        options->interpolate      = PETSC_TRUE;
        options->refinementLimit  = 0.0;
        options->bcType           = DIRICHLET;
        options->operatorAssembly = ASSEMBLY_FULL;
        options->lambda           = 0.0;
        options->reentrantMesh    = PETSC_FALSE;
        options->circularMesh     = PETSC_FALSE;
        options->refineSingularity= PETSC_FALSE;

        ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMG");CHKERRQ(ierr);
          ierr = PetscOptionsInt("-debug", "The debugging level", "bratu.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
          run = options->run;
          ierr = PetscOptionsEList("-run", "The run type", "bratu.cxx", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
          options->run = (RunType) run;
          ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "bratu.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-reentrant", "Make a reentrant-corner mesh", "bratu.cxx", options->reentrantMesh, &options->reentrantMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-circular_mesh", "Make a reentrant-corner mesh", "bratu.cxx", options->circularMesh, &options->circularMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-singularity", "Refine the mesh around a singularity with a priori poisson error estimation", "bratu.cxx", options->refineSingularity, &options->refineSingularity, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-structured", "Use a structured mesh", "bratu.cxx", options->structured, &options->structured, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "bratu.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "bratu.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "bratu.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
          filename << "data/bratu_" << options->dim <<"d";
          ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
          ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "bratu.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
          ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
          bc = options->bcType;
          ierr = PetscOptionsEList("-bc_type","Type of boundary condition","bratu.cxx",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
          options->bcType = (BCType) bc;
          as = options->operatorAssembly;
          ierr = PetscOptionsEList("-assembly_type","Type of operator assembly","bratu.cxx",asTypes,3,asTypes[options->operatorAssembly],&as,PETSC_NULL);CHKERRQ(ierr);
          options->operatorAssembly = (AssemblyType) as;
          ierr = PetscOptionsReal("-lambda", "The parameter controlling nonlinearity", "bratu.cxx", options->lambda, &options->lambda, PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();

        ALE::Problem::Functions::lambda = options->lambda;
        this->setDebug(options->debug);
        PetscFunctionReturn(0);
      };
    public: // Accessors
      Options *getOptions() {return &this->_options;};
      int  dim() const {return this->_options.dim;};
      bool structured() const {return this->_options.structured;};
      void structured(const bool s) {this->_options.structured = (PetscTruth) s;};
      bool interpolated() const {return this->_options.interpolate;};
      DM getDM() {return this->_dm;};
    public: // Mesh
      #undef __FUNCT__
      #define __FUNCT__ "CreateMesh"
      PetscErrorCode createMesh() {
        PetscTruth     view;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (_options.circularMesh) {
          if (_options.reentrantMesh) {
            _options.reentrant_angle = .9;
            ierr = ALE::DMBuilder::createReentrantSphericalMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          } else {
            ierr = ALE::DMBuilder::createSphericalMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          }
        } else {
          if (_options.reentrantMesh) {
            _options.reentrant_angle = .75;
            ierr = ALE::DMBuilder::createReentrantBoxMesh(comm(), dim(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          } else {
            ierr = ALE::DMBuilder::createBoxMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
          }
        }
        if (this->commSize() > 1) {
          ::Mesh parallelMesh;

          ierr = MeshDistribute((::Mesh) this->_dm, _options.partitioner, &parallelMesh);CHKERRQ(ierr);
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
          this->_dm = (DM) parallelMesh;
        }
        ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);
        if (_options.bcType == DIRICHLET) {
          this->_mesh->markBoundaryCells("marker");
        }
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
        if (view) {
          PetscViewer viewer;

          ierr = PetscViewerCreate(this->comm(), &viewer);CHKERRQ(ierr);
          ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
          ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
          ierr = PetscViewerFileSetName(viewer, "bratu.vtk");CHKERRQ(ierr);
          ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
          ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
        }
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
        if (view) {this->_mesh->view("Mesh");}
        ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_simple", &view);CHKERRQ(ierr);
        if (view) {ierr = MeshView((::Mesh) this->_dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "RefineMesh"
      PetscErrorCode refineMesh() {
        PetscErrorCode ierr;

        if (_options.refinementLimit > 0.0) {
          ::Mesh refinedMesh;

          ierr = MeshRefine((::Mesh) this->_dm, _options.refinementLimit, (PetscTruth) interpolated(), &refinedMesh);CHKERRQ(ierr);
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
          this->_dm = (DM) refinedMesh;
          ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);

          if (_options.refineSingularity) {
            ::Mesh refinedMesh2;
            double singularity[3] = {0.0, 0.0, 0.0};

            if (dim() == 2) {
              ierr = ALE::DMBuilder::MeshRefineSingularity((::Mesh) this->_dm, singularity, _options.reentrant_angle, &refinedMesh2);CHKERRQ(ierr);
            } else if (dim() == 3) {
              ierr = ALE::DMBuilder::MeshRefineSingularity_Fichera((::Mesh) this->_dm, singularity, 0.75, &refinedMesh2);CHKERRQ(ierr);
            }
            ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
            this->_dm = (DM) refinedMesh2;
            ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);
#ifndef PETSC_OPT_SIEVE
            ierr = MeshIDBoundary((::Mesh) this->_dm);CHKERRQ(ierr);
#endif
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "DestroyMesh"
      PetscErrorCode destroyMesh() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          ierr = DADestroy((DA) this->_dm);CHKERRQ(ierr);
        } else {
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateProblem"
      PetscErrorCode createProblem() {
        PetscFunctionBegin;
        if (dim() == 2) {
          if (this->_options.bcType == DIRICHLET) {
            if (this->_options.lambda > 0.0) {
              this->_options.func      = ALE::Problem::Functions::nonlinear_2d;
              this->_options.exactFunc = ALE::Problem::Functions::quadratic_2d;
            } else if (this->_options.reentrantMesh) { 
              this->_options.func      = ALE::Problem::Functions::singularity_2d;
              this->_options.exactFunc = ALE::Problem::Functions::singularity_exact_2d;
            } else {
              this->_options.func      = ALE::Problem::Functions::constant;
              this->_options.exactFunc = ALE::Problem::Functions::quadratic_2d;
            }
          } else {
            this->_options.func      = ALE::Problem::Functions::linear_2d;
            this->_options.exactFunc = ALE::Problem::Functions::cubic_2d;
          }
        } else if (dim() == 3) {
          if (this->_options.bcType == DIRICHLET) {
            if (this->_options.reentrantMesh) {
              this->_options.func      = ALE::Problem::Functions::singularity_3d;
              this->_options.exactFunc = ALE::Problem::Functions::singularity_exact_3d;
            } else {
              if (this->_options.lambda > 0.0) {
                this->_options.func    = ALE::Problem::Functions::nonlinear_3d;
              } else {
                this->_options.func    = ALE::Problem::Functions::constant;
              }
              this->_options.exactFunc = ALE::Problem::Functions::quadratic_3d;
            }
          } else {
            this->_options.func      = ALE::Problem::Functions::linear_3d;
            this->_options.exactFunc = ALE::Problem::Functions::cubic_3d;
          }
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
        }
        if (!structured()) {
          int            numBC      = (this->_options.bcType == DIRICHLET) ? 1 : 0;
          int            markers[1] = {1};
          double       (*funcs[1])(const double *coords) = {this->_options.exactFunc};
          PetscErrorCode ierr;

          if (dim() == 1) {
            ierr = CreateProblem_gen_0(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_0;
          } else if (dim() == 2) {
            ierr = CreateProblem_gen_1(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_1;
          } else if (dim() == 3) {
            ierr = CreateProblem_gen_2(this->_dm, "u", numBC, markers, funcs, this->_options.exactFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_2;
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("default");
          s->setDebug(debug());
          this->_mesh->setupField(s);
          if (debug()) {s->view("Default field");}
        }
        PetscFunctionReturn(0);
      };
    };
  }
}

#endif
