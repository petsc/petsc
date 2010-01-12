#ifndef included_ALE_Problem_LaplaceBEM_hh
#define included_ALE_Problem_LaplaceBEM_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

#if 0
#include <Kernel_Laplacian_2D.hh>
#else
#include <Kernel_BiotSavart_3D.hh>
#endif
#include <FMM.hh>

#include <sieve/problem/Bratu.hh>

namespace ALE {
  namespace Problem {
    class LaplaceBEM : public ALE::ParallelObject {
    public:
    protected:
      LaplaceBEMOptions         _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
      DMMG                *_dmmg;
    public:
      LaplaceBEM(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm   = PETSC_NULL;
        this->_dmmg = PETSC_NULL;
        this->_options.exactSol.vec = PETSC_NULL;
        this->_options.error.vec    = PETSC_NULL;
      };
      ~LaplaceBEM() {
        PetscErrorCode ierr;

        if (this->_dmmg)                 {ierr = DMMGDestroy(this->_dmmg);CHKERRXX(ierr);}
        if (this->_options.exactSol.vec) {ierr = this->destroyExactSolution(this->_options.exactSol);CHKERRXX(ierr);}
        if (this->_options.error.vec)    {ierr = this->destroyExactSolution(this->_options.error);CHKERRXX(ierr);}
        if (this->_dm)                   {ierr = this->destroyMesh();CHKERRXX(ierr);}
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "LaplaceBEMProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, LaplaceBEMOptions *options) {
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
        options->phiCoefficient   = 0.5;

        ierr = PetscOptionsBegin(comm, "", "LaplaceBEM Problem Options", "DMMG");CHKERRQ(ierr);
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
      LaplaceBEMOptions *getOptions() {return &this->_options;};
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
            //ierr = ALE::DMBuilder::createBasketMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
            ierr = ALE::DMBuilder::createBasketMesh(comm(), dim(), structured(), interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
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
#if 0
        if (bcType() == DIRICHLET) {
          this->_mesh->markBoundaryCells("marker");
        }
#endif
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
              this->_options.func               = ALE::Problem::Functions::nonlinear_2d;
              this->_options.exactDirichletFunc = ALE::Problem::Functions::quadratic_2d;
            } else if (this->_options.reentrantMesh) { 
              this->_options.func               = ALE::Problem::Functions::singularity_2d;
              this->_options.exactDirichletFunc = ALE::Problem::Functions::singularity_exact_2d;
            } else {
              this->_options.func               = ALE::Problem::Functions::constant;
              this->_options.exactDirichletFunc = ALE::Problem::Functions::linear_2d_bem;
              this->_options.exactNeumannFunc   = ALE::Problem::Functions::linear_nder_2d;
            }
          } else {
            this->_options.func               = ALE::Problem::Functions::linear_2d;
            this->_options.exactDirichletFunc = ALE::Problem::Functions::cubic_2d;
          }
        } else if (dim() == 3) {
          if (bcType() == DIRICHLET) {
            if (this->_options.reentrantMesh) {
              this->_options.func               = ALE::Problem::Functions::singularity_3d;
              this->_options.exactDirichletFunc = ALE::Problem::Functions::singularity_exact_3d;
            } else {
              if (this->_options.lambda > 0.0) {
                this->_options.func             = ALE::Problem::Functions::nonlinear_3d;
              } else {
                this->_options.func             = ALE::Problem::Functions::constant;
              }
              this->_options.exactDirichletFunc = ALE::Problem::Functions::quadratic_3d;
            }
          } else {
            this->_options.func               = ALE::Problem::Functions::linear_3d;
            this->_options.exactDirichletFunc = ALE::Problem::Functions::cubic_3d;
          }
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
        }
        if (!structured()) {
          // Should pass bcType()
          int            numBC      = 0;
          int            markers[1] = {1};
          double       (*funcs[1])(const double *coords) = {this->_options.exactDirichletFunc};
          PetscErrorCode ierr;

          if (dim() == 2) {
            ierr = CreateProblem_gen_0(this->_dm, "u", numBC, markers, funcs, this->_options.exactDirichletFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateBdDualBasis_gen_0;
          } else if (dim() == 3) {
            ierr = CreateProblem_gen_1(this->_dm, "u", numBC, markers, funcs, this->_options.exactDirichletFunc);CHKERRQ(ierr);
            this->_options.integrate = IntegrateDualBasis_gen_1;
          } else {
            SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim());
          }
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("default");
          s->setDebug(debug());
          this->_mesh->setupField(s, 2, true);
          if (debug()) {s->view("Default field");}
          const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& t = this->_mesh->getRealSection("boundaryData");
          t->setDebug(debug());
          this->_mesh->setupField(t, 2);
          typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
          const Obj<PETSC_MESH_TYPE::label_type>&         cellExclusion = this->_mesh->getLabel("cellExclusion");
          const Obj<PETSC_MESH_TYPE::label_sequence>&     boundaryCells = this->_mesh->heightStratum(0);
          const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates   = this->_mesh->getRealSection("coordinates");
          const Obj<PETSC_MESH_TYPE::names_type>&         discs         = this->_mesh->getDiscretizations();
          const PETSC_MESH_TYPE::point_type               firstCell     = *boundaryCells->begin();
          const int                                       numFields     = discs->size();
          PETSC_MESH_TYPE::real_section_type::value_type *values        = new PETSC_MESH_TYPE::real_section_type::value_type[this->_mesh->sizeWithBC(t, firstCell)];
          int                                             embedDim      = dim();
          int                                            *v             = new int[numFields];
          double                                         *v0            = new double[embedDim];
          double                                         *J             = new double[embedDim*embedDim];
          double                                          detJ;
          Visitor pV((int) pow((double) this->_mesh->getSieve()->getMaxConeSize(), this->_mesh->depth())+1, true);

          for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
            ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), *c_iter, pV);
            const Visitor::point_type *oPoints = pV.getPoints();
            const int                  oSize   = pV.getSize();

            this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int f = 0; f < numFields; ++f) v[f] = 0;
            for(int cl = 0; cl < oSize; ++cl) {
              int f = 0;

              for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>&        disc    = this->_mesh->getDiscretization(*f_iter);
                const Obj<PETSC_MESH_TYPE::names_type> bcs     = disc->getBoundaryConditions();
                const int                              fDim    = t->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
                const int                             *indices = disc->getIndices(this->_mesh->getValue(cellExclusion, *c_iter));

                //for(PETSC_MESH_TYPE::names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
                  //const Obj<ALE::BoundaryCondition>& bc = disc->getBoundaryCondition(*bc_iter);

                  for(int d = 0; d < fDim; ++d, ++v[f]) {
                    //values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
                    values[indices[v[f]]] = IntegrateBdDualBasis_gen_0(v0, J, v[f], this->_options.exactDirichletFunc);
                  }
                //}
              }
            }
            this->_mesh->updateAll(t, *c_iter, values);
            pV.clear();
          }
          delete [] values;
          delete [] v;
          delete [] v0;
          delete [] J;

          if (debug()) {t->view("Boundary Data");}
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
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
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

            this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
            for(int cl = 0; cl < oSize; ++cl) {
              const int pointDim = s->getFiberDimension(oPoints[cl]);

              if (pointDim) {
                for(int d = 0; d < pointDim; ++d, ++v) {
                  values[v] = (*this->_options.integrate)(v0, J, v, this->_options.exactNeumannFunc);
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
          s->view("Exact Solution");
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
      // The BEM system to bbe solved is
      //
      //   (1/2 I + F) \phi = G \frac{\partial\phi}{\partial n}
      //
      // where
      //
      //   F_{ij} = \int_{S_j} \frac{\partial G(x_i, y)}{\partial n(y)} dy = \int_{S_j} \frac{1}{2\pi r} \frac{\partial r}{\partial n} dy
      //   G_{ij} = \int_{S_j} G(x_i, y) dy = \int_{S_j} \frac{1}{2\pi} \ln\frac{1}{r} dy
      PetscErrorCode createSolver() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGCreate(this->comm(), 1, &this->_options, &this->_dmmg);CHKERRQ(ierr);
        ierr = DMMGSetDM(this->_dmmg, this->_dm);CHKERRQ(ierr);
        if (structured()) {
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
        } else {
          if (opAssembly() == ALE::Problem::ASSEMBLY_FULL) {
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::RhsBd_Unstructured, ALE::Problem::Functions::JacBd_Unstructured, 0, 0);CHKERRQ(ierr);
#if 0
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_CALCULATED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::RhsBd_Unstructured, ALE::Problem::Functions::JacBd_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
          } else if (opAssembly() == ALE::Problem::ASSEMBLY_STORED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::Functions::RhsBd_Unstructured, ALE::Problem::Functions::JacBd_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
#endif
          } else {
            SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", opAssembly());
          }
          ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
        }
#if 0
        if (bcType() == ALE::Problem::NEUMANN) {
          // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
          ierr = DMMGSetNullSpace(this->_dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
        }
#endif
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "LaplaceBEMSolve"
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
        PetscScalar  (*func)(const double *) = this->_options.exactNeumannFunc;
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
        const int                                      dim           = this->dim();
        const int                                      closureSize   = this->_mesh->sizeWithBC(u, *cells->begin()); // Should do a max of some sort
        double      *coords, *v0, *J, *invJ, detJ;
        PetscScalar *x;
        double       localError = 0.0;

        ierr = PetscMalloc(closureSize * sizeof(PetscScalar), &x);CHKERRQ(ierr);
        ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
        // Loop over cells
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          double elemError = 0.0;

          this->_mesh->computeBdElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
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
              for(int e = 0; e < dim-1; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*(dim-1)+e] + 1.0);
              }
              if (debug()) {std::cout << "q: "<<q<<"  refCoord["<<d<<"] " << quadPoints[q*(dim-1)+d] << "  coords["<<d<<"] " << coords[d] << std::endl;}
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
          SETERRQ(PETSC_ERR_SUP, "Structured meshes not supported");
        } else {
          ::Mesh      mesh = (::Mesh) this->_dm;
          SectionReal residual;

          ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
          ierr = ALE::Problem::Functions::RhsBd_Unstructured(mesh, sol.section, residual, &this->_options);CHKERRQ(ierr);
          if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
          ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
          ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        }
        PetscPrintf(comm(), "Residual for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
    };
    class FMMforBEM {
    public:
      class Blob;
      typedef FMM::Evaluator<Blob,FMM::Kernel<double, std::complex<double>, std::complex<double>, NUM_TERMS,DIMENSION,NUM_COEFFICIENTS>, DIMENSION> evaluator_type;
      typedef evaluator_type::point_type point_type;
      const static int dim = DIMENSION;
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
    protected:
      ALE::Obj<evaluator_type::tree_type> tree;
      ALE::Obj<evaluator_type>            evaluator;
      FMM::Options<point_type>            options;
    public:
      FMMforBEM(ALE::Problem::LaplaceBEM& bemProblem) {
        const Obj<PETSC_MESH_TYPE::label_sequence>& edges         = bemProblem.getMesh()->heightStratum(0);
        const int                                   numEdges      = edges->size();
        double                                      minEdgeLength = 0.5;

        // Loop over edges and find minimum length
        setupTree(numEdges, minEdgeLength, bemProblem.getOptions()->debug);
        evaluator = new evaluator_type(PETSC_COMM_SELF, bemProblem.getOptions()->debug);
      };
    public:
      // Create matching tree
      //   For a square:
      //     1) Take the box width to be the smallest edge length h
      //     2) The number of boxes per dimension is N_1 = E/4 + 1 where is the total number of edges in the square
      //     3) The tree is level is thus 2^{dL} > N_1^d --> L > log(E/4 + 1)/log(2)
      //     4) The lower left is (-h/2, -h/2) and the upper right is (\frac{(2^{L+1}-1) h}{2}, \frac{(2^{L+1}-1) h}{2})
      void setupTree(int numEdges, double minEdgeLength, int debug = 0) {
        //const double h        = minEdgeLength;
        const int    N_1      = numEdges/4 + 1;
        const int    L        = std::ceil(log(N_1)/log(2));
        //const double lower[2] = {-h/2.0, -h/2.0};
        //const double upper[2] = {(((1 << (L+1)) - 1) * h)/2.0, (((1 << (L+1)) - 1) * h)/2.0};

        tree = new evaluator_type::tree_type(L+1, debug);
      };
      // Evaluate BEM solution on FEM grid
      //   This should be as simple as running the last step using different points
      void BEMtoFEM(ALE::Problem::Bratu& femProblem) {
#ifdef BROKEN_FOR_3D
        const Obj<PETSC_MESH_TYPE::label_sequence>&    vertices    = femProblem.getMesh()->depthStratum(0);
        const Obj<PETSC_MESH_TYPE::real_section_type>& coordSec    = femProblem.getMesh()->getRealSection("coordinates");
        const Obj<PETSC_MESH_TYPE::real_section_type>& solution    = femProblem.getMesh()->getRealSection("default");
        const int                                      numPoints   = vertices->size();
        point_type                                    *coordinates = new point_type[numPoints];
        evaluator_type::kernel_type::vel_type         *potentials  = new evaluator_type::kernel_type::vel_type[numPoints];
        int v = 0;

        for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter, ++v) {
          const double *coords = coordSec->restrictPoint(*v_iter);

          for(int d = 0; d < dim; ++d) {
            coordinates[v*dim+d] = coords[d];
          }
        }
        evaluator->evaluatePoints(*tree, options.sigma*options.sigma, options.k, numPoints, coordinates, potentials);
        v = 0;
        for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter, ++v) {
          PETSC_MESH_TYPE::real_section_type::value_type value = potentials[v].real();

          solution->updatePoint(*v_iter, &value);
        }
#endif
      };
    };
  }
}

#endif
