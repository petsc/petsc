#ifndef included_ALE_Problem_Bratu_hh
#define included_ALE_Problem_Bratu_hh

// TODO: Add a guard that looks for the generated quadrature header

#include <sieve/problem/Base.hh>

namespace ALE {
  namespace Problem {
    class Bratu : public ALE::ParallelObject {
    public:
    protected:
      BratuOptions         _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
      DMMG                *_dmmg;
    public:
      Bratu(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm   = PETSC_NULL;
        this->_dmmg = PETSC_NULL;
        this->_options.exactSol.vec = PETSC_NULL;
        this->_options.error.vec    = PETSC_NULL;
      };
      ~Bratu() {
        PetscErrorCode ierr;

        if (this->_dmmg)                 {ierr = DMMGDestroy(this->_dmmg);CHKERRXX(ierr);}
        if (this->_options.exactSol.vec) {ierr = this->destroyExactSolution(this->_options.exactSol);CHKERRXX(ierr);}
        if (this->_options.error.vec)    {ierr = this->destroyExactSolution(this->_options.error);CHKERRXX(ierr);}
        if (this->_dm)                   {ierr = this->destroyMesh();CHKERRXX(ierr);}
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "BratuProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, BratuOptions *options) {
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
      BratuOptions *getOptions() {return &this->_options;};
      int  dim() const {return this->_options.dim;};
      bool structured() const {return this->_options.structured;};
      void structured(const bool s) {this->_options.structured = (PetscTruth) s;};
      bool interpolated() const {return this->_options.interpolate;};
      void interpolated(const bool i) {this->_options.interpolate = (PetscTruth) i;};
      BCType bcType() const {return this->_options.bcType;};
      void bcType(const BCType bc) {this->_options.bcType = bc;};
      AssemblyType opAssembly() const {return this->_options.operatorAssembly;};
      void opAssembly(const AssemblyType at) {this->_options.operatorAssembly = at;};
      PETSC_MESH_TYPE *getMesh() {return this->_mesh;};
      DM getDM() const {return this->_dm;};
      DMMG *getDMMG() const {return this->_dmmg;};
      ALE::Problem::ExactSolType exactSolution() const {return this->_options.exactSol;};
    public: // Mesh
      #undef __FUNCT__
      #define __FUNCT__ "CreateMesh"
      PetscErrorCode createMesh() {
        PetscTruth     view;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA       da;
          PetscInt dof = 1;
          PetscInt pd  = PETSC_DECIDE;

          if (dim() == 2) {
            ierr = DACreate2d(comm(), DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DACreate3d(comm(), DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, -3, pd, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
          this->_dm = (DM) da;
          PetscFunctionReturn(0);
        }
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
        ierr = refineMesh();CHKERRQ(ierr);

        if (this->commSize() > 1) {
          ::Mesh parallelMesh;

          ierr = MeshDistribute((::Mesh) this->_dm, _options.partitioner, &parallelMesh);CHKERRQ(ierr);
          ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
          this->_dm = (DM) parallelMesh;
        }
        ierr = MeshGetMesh((::Mesh) this->_dm, this->_mesh);CHKERRQ(ierr);
        if (bcType() == DIRICHLET) {
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
        PetscFunctionBegin;
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
        PetscFunctionReturn(0);
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
          if (bcType() == DIRICHLET) {
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
          if (bcType() == DIRICHLET) {
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
          int            numBC      = (bcType() == DIRICHLET) ? 1 : 0;
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
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateExactSolution"
      PetscErrorCode createExactSolution() {
        PetscTruth     flag;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA            da = (DA) this->_dm;
          PetscScalar (*func)(const double *) = this->_options.func;
          Vec           X, U;

          ierr = DAGetGlobalVector(da, &X);CHKERRQ(ierr);
          ierr = DACreateGlobalVector(da, &this->_options.exactSol.vec);CHKERRQ(ierr);
          this->_options.func = this->_options.exactFunc;
          U                   = exactSolution().vec;
          if (dim() == 2) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::Functions::Function_Structured_2d, X, U, (void *) &this->_options);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::Functions::Function_Structured_3d, X, U, (void *) &this->_options);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DARestoreGlobalVector(da, &X);CHKERRQ(ierr);
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {ierr = VecView(U, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
          if (flag) {ierr = VecView(U, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
          this->_options.func = func;
          ierr = DACreateGlobalVector(da, &this->_options.error.vec);CHKERRQ(ierr);
        } else {
          ::Mesh mesh = (::Mesh) this->_dm;

          ierr = MeshGetSectionReal(mesh, "exactSolution", &this->_options.exactSol.section);CHKERRQ(ierr);
          const Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("exactSolution");
          this->_mesh->setupField(s);
          const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = this->_mesh->heightStratum(0);
          const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = this->_mesh->getRealSection("coordinates");
          const int                                       localDof    = this->_mesh->sizeWithBC(s, *cells->begin());
          PETSC_MESH_TYPE::real_section_type::value_type *values      = new PETSC_MESH_TYPE::real_section_type::value_type[localDof];
          double                                         *v0          = new double[dim()];
          double                                         *J           = new double[dim()*dim()];
          double                                          detJ;
          ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(this->_mesh->getSieve()->getMaxConeSize(), this->_mesh->depth())+1, true);

          for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), *c_iter, pV);
            const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
            const int                          oSize   = pV.getSize();
            int                                v       = 0;

            this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int cl = 0; cl < oSize; ++cl) {
              const int pointDim = s->getFiberDimension(oPoints[cl]);

              if (pointDim) {
                for(int d = 0; d < pointDim; ++d, ++v) {
                  values[v] = (*this->_options.integrate)(v0, J, v, this->_options.exactFunc);
                }
              }
            }
            this->_mesh->updateAll(s, *c_iter, values);
            pV.clear();
          }
          delete [] values;
          delete [] v0;
          delete [] J;
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {s->view("Exact Solution");}
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            PetscViewer viewer;

            ierr = PetscViewerCreate(this->comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "exact_sol.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = SectionRealView(exactSolution().section, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = MeshGetSectionReal(mesh, "error", &this->_options.error.section);CHKERRQ(ierr);
          const Obj<PETSC_MESH_TYPE::real_section_type>& e = this->_mesh->getRealSection("error");
          e->setChart(PETSC_MESH_TYPE::real_section_type::chart_type(*this->_mesh->heightStratum(0)));
          e->setFiberDimension(this->_mesh->heightStratum(0), 1);
          this->_mesh->allocate(e);
        }
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "DestroyExactSolution"
      PetscErrorCode destroyExactSolution(ALE::Problem::ExactSolType sol) {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          ierr = VecDestroy(sol.vec);CHKERRQ(ierr);
        } else {
          ierr = SectionRealDestroy(sol.section);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateSolver"
      PetscErrorCode createSolver() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGCreate(this->comm(), 1, &this->_options, &this->_dmmg);CHKERRQ(ierr);
        ierr = DMMGSetDM(this->_dmmg, this->_dm);CHKERRQ(ierr);
        if (structured()) {
          // Needed if using finite elements
          // ierr = PetscOptionsSetValue("-dmmg_form_function_ghost", PETSC_NULL);CHKERRQ(ierr);
          if (dim() == 2) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::Rhs_Structured_2d_FD, ALE::Problem::Functions::Jac_Structured_2d_FD, 0, 0);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::Rhs_Structured_3d_FD, ALE::Problem::Functions::Jac_Structured_3d_FD, 0, 0);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
          for(int l = 0; l < DMMGGetLevels(this->_dmmg); l++) {
            ierr = DASetUniformCoordinates((DA) (this->_dmmg)[l]->dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
          }
        } else {
          if (opAssembly() == ALE::Problem::ASSEMBLY_FULL) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::Rhs_Unstructured, ALE::Problem::Functions::Jac_Unstructured, 0, 0);CHKERRQ(ierr);
#if 0
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_CALCULATED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::Rhs_Unstructured, ALE::Problem::Functions::Jac_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_STORED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::Rhs_Unstructured, ALE::Problem::Functions::Jac_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
#endif
          } else {
            SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", opAssembly());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
        }
        if (bcType() == ALE::Problem::NEUMANN) {
          // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
          ierr = DMMGSetNullSpace(this->_dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "BratuSolve"
      PetscErrorCode solve() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGSolve(this->_dmmg);CHKERRQ(ierr);
        // Report on solve
        SNES                snes = DMMGGetSNES(this->_dmmg);
        PetscInt            its;
        PetscTruth          flag;
        SNESConvergedReason reason;

        ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
        ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
        if (debug()) {
          ierr = PetscPrintf(comm(), "Number of nonlinear iterations = %D\n", its);CHKERRQ(ierr);
          ierr = PetscPrintf(comm(), "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
        }
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
        if (flag) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
        if (flag && dim() == 2) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
        if (structured()) {
          ALE::Problem::ExactSolType sol;

          sol.vec = DMMGGetx(this->_dmmg);
          if (DMMGGetLevels(this->_dmmg) == 1) {ierr = this->checkError(sol);CHKERRQ(ierr);}
        } else {
          const Obj<PETSC_MESH_TYPE::real_section_type>& sol = this->_mesh->getRealSection("default");
          SectionReal solution;
          double      error;

          ierr = MeshGetSectionReal((::Mesh) this->_dm, "default", &solution);CHKERRQ(ierr);
          ierr = SectionRealToVec(solution, (::Mesh) this->_dm, SCATTER_REVERSE, DMMGGetx(this->_dmmg));CHKERRQ(ierr);
          ierr = this->calculateError(solution, &error);CHKERRQ(ierr);
          if (debug()) {ierr = PetscPrintf(comm(), "Total error: %g\n", error);CHKERRQ(ierr);}
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            PetscViewer viewer;

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "sol.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = SectionRealView(solution, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "error.vtk");CHKERRQ(ierr);
            ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
            ierr = SectionRealView(this->_options.error.section, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
          if (flag) {sol->view("Solution");}
          ierr = PetscOptionsHasName(PETSC_NULL, "-hierarchy_vtk", &flag);CHKERRQ(ierr);
          if (flag) {
            double      offset[3] = {2.0, 0.0, 0.25};
            PetscViewer viewer;

            ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
            ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(viewer, "mesh_hierarchy.vtk");CHKERRQ(ierr);
            ierr = PetscOptionsReal("-hierarchy_vtk", PETSC_NULL, "bratu.cxx", *offset, offset, PETSC_NULL);CHKERRQ(ierr);
            ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
            ierr = VTKViewer::writeHierarchyVertices(this->_dmmg, viewer, offset);CHKERRQ(ierr);
            ierr = VTKViewer::writeHierarchyElements(this->_dmmg, viewer);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
          }
          ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CalculateError"
      PetscErrorCode calculateError(SectionReal X, double *error) {
        Obj<PETSC_MESH_TYPE::real_section_type> u;
        Obj<PETSC_MESH_TYPE::real_section_type> s;
        PetscScalar  (*func)(const double *) = this->_options.exactFunc;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = SectionRealGetSection(X, u);CHKERRQ(ierr);
        ierr = SectionRealGetSection(this->_options.error.section, s);CHKERRQ(ierr);
        const Obj<ALE::Discretization>&                disc          = this->_mesh->getDiscretization("u");
        const int                                      numQuadPoints = disc->getQuadratureSize();
        const double                                  *quadPoints    = disc->getQuadraturePoints();
        const double                                  *quadWeights   = disc->getQuadratureWeights();
        const int                                      numBasisFuncs = disc->getBasisSize();
        const double                                  *basis         = disc->getBasis();
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = this->_mesh->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = this->_mesh->heightStratum(0);
        const int                                      dim           = this->_mesh->getDimension();
        const int                                      closureSize   = this->_mesh->sizeWithBC(u, *cells->begin()); // Should do a max of some sort
        double      *coords, *v0, *J, *invJ, detJ;
        PetscScalar *x;
        double       localError = 0.0;

        ierr = PetscMalloc(closureSize * sizeof(PetscScalar), &x);CHKERRQ(ierr);
        ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          double elemError = 0.0;

          this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (debug()) {
            std::cout << "Element " << *c_iter << " v0: (" << v0[0]<<","<<v0[1]<<")" << "J " << J[0]<<","<<J[1]<<","<<J[2]<<","<<J[3] << " detJ " << detJ << std::endl;
          }
          this->_mesh->restrictClosure(u, *c_iter, x, closureSize);
          if (debug()) {
            for(int f = 0; f < numBasisFuncs; ++f) {
              std::cout << "x["<<f<<"] " << x[f] << std::endl;
            }
          }
          // Loop over quadrature points
          for(int q = 0; q < numQuadPoints; ++q) {
            for(int d = 0; d < dim; d++) {
              coords[d] = v0[d];
              for(int e = 0; e < dim; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
              }
              if (debug()) {std::cout << "q: "<<q<<"  refCoord["<<d<<"] " << quadPoints[q*dim+d] << "  coords["<<d<<"] " << coords[d] << std::endl;}
            }
            const PetscScalar funcVal = (*func)(coords);
            if (debug()) {std::cout << "q: "<<q<<"  funcVal " << funcVal << std::endl;}

            double interpolant = 0.0;
            for(int f = 0; f < numBasisFuncs; ++f) {
              interpolant += x[f]*basis[q*numBasisFuncs+f];
            }
            if (debug()) {std::cout << "q: "<<q<<"  interpolant " << interpolant << std::endl;}
            elemError += (interpolant - funcVal)*(interpolant - funcVal)*quadWeights[q];
            if (debug()) {std::cout << "q: "<<q<<"  elemError " << elemError << std::endl;}
          }
          if (debug()) {
            std::cout << "Element " << *c_iter << " error: " << elemError << std::endl;
          }
          this->_mesh->updateAdd(s, *c_iter, &elemError);
          localError += elemError;
        }
        ierr = MPI_Allreduce(&localError, error, 1, MPI_DOUBLE, MPI_SUM, comm());CHKERRQ(ierr);
        ierr = PetscFree(x);CHKERRQ(ierr);
        ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
        *error = sqrt(*error);
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "CheckError"
      PetscErrorCode checkError(ALE::Problem::ExactSolType sol) {
        const char    *name;
        PetscScalar    norm;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        if (structured()) {
          DA  da = (DA) this->_dm;
          Vec error;

          ierr = DAGetGlobalVector(da, &error);CHKERRQ(ierr);
          ierr = VecCopy(sol.vec, error);CHKERRQ(ierr);
          ierr = VecAXPY(error, -1.0, exactSolution().vec);CHKERRQ(ierr);
          ierr = VecNorm(error, NORM_2, &norm);CHKERRQ(ierr);
          ierr = DARestoreGlobalVector(da, &error);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
        } else {
          ierr = this->calculateError(sol.section, &norm);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        }
        PetscPrintf(comm(), "Error for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "CheckResidual"
      PetscErrorCode checkResidual(ALE::Problem::ExactSolType sol) {
        const char    *name;
        PetscScalar    norm;
        PetscTruth     flag;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
        if (structured()) {
          DA  da = (DA) this->_dm;
          Vec residual;

          ierr = DAGetGlobalVector(da, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          if (dim() == 2) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::Functions::Rhs_Structured_2d_FD, sol.vec, residual, &this->_options);CHKERRQ(ierr);
          } else if (dim() == 3) {
            ierr = DAFormFunctionLocal(da, (DALocalFunction1) ALE::Problem::Functions::Rhs_Structured_3d_FD, sol.vec, residual, &this->_options);CHKERRQ(ierr);
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          ierr = VecNorm(residual, NORM_2, &norm);CHKERRQ(ierr);
          if (flag) {ierr = VecView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = DARestoreGlobalVector(da, &residual);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
        } else {
          ::Mesh      mesh = (::Mesh) this->_dm;
          SectionReal residual;

          ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          ierr = ALE::Problem::Functions::Rhs_Unstructured(mesh, sol.section, residual, &this->_options);CHKERRQ(ierr);
          if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
          ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        }
        PetscPrintf(comm(), "Residual for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
    };
  }
}

#endif
