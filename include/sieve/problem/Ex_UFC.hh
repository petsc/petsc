#ifndef included_ALE_Problem_Bratu_hh
#define included_ALE_Problem_Bratu_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

#include <UFCProblem.hh>

namespace ALE {
  namespace Problem {
    typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
    typedef enum {NEUMANN, DIRICHLET} BCType;
    typedef enum {POISSON, VECTORPOISSON, STOKES, MIXEDPOISSON} UFCFormType;
    typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
    typedef union {SectionReal section; Vec vec;} ExactSolType;
    typedef struct {
      PetscInt      debug;                       // The debugging level
      RunType       run;                         // The run type
      PetscInt      dim;                         // The topological mesh dimension
      PetscTruth    reentrantMesh;               // Generate a reentrant mesh?
      PetscTruth    circularMesh;                // Generate a circular mesh?
      PetscTruth    refineSingularity;           // Generate an a priori graded mesh for the poisson problem
      PetscTruth    generateMesh;                // Generate the unstructure mesh
      PetscTruth    interpolate;                 // Generate intermediate mesh elements
      PetscReal     refinementLimit;             // The largest allowable cell volume
      char          baseFilename[2048];          // The base filename for mesh files
      char          partitioner[2048];           // The graph partitioner
      ufc::function * func;                      // The function to project -- should be the WHOLE RANK of the object
      BCType        bcType;                      // The type of boundary conditions
      ufc::function * exactFunc;                 // The exact solution function
      ExactSolType  exactSol;                    // The discrete exact solution
      ExactSolType  error;                       // The discrete cell-wise error
      AssemblyType  operatorAssembly;            // The type of operator assembly 
      //double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
      double        reentrant_angle;              // The angle for the reentrant corner.
      UFCFormType   form_type;                    // The form to solve
      
    } Ex_UFCOptions;
    namespace UFCFunctions {

      class coordinates : public ufc::function {
	int _dim;
	int nevaluations;
      public:
	coordinates(int dim) {
	  _dim = dim;
	  nevaluations = 0;
	}
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  if (nevaluations < 6) {
	    values[0] = coordinates[_dim];
	  } else {
	    values[0] = -10.;
	  }
	}
      };

      class zero_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  values[0] = 0.;
	}
      };

      class zero_vector : public ufc::function {
      private:
	int _order;
      public:
	zero_vector(int ord) : ufc::function() {
	  _order = ord;
	};
	
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  for (int i = 0; i < _order; i++) values[i] = 0.;
	}
      };

      class constant_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	    values[0] = -4.0;
	  }
      };

      class constant_vector : public ufc::function {
      private:
	int _order;
	double _constant;
      public:
	constant_vector(int ord, double constant) : ufc::function() {
	  _order = ord;
	  _constant = constant;
	}

	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  for (int i = 0; i < _order; i++) values[i] = _constant;
	}  
      };

      class nonlinear_2d_scalar : public ufc::function {
      public:
	double lambda;
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  // PetscScalar nonlinear_2d(const double x[]) {
	  values[0] = -4.0 - lambda*PetscExpScalar(coordinates[0]*coordinates[0] + coordinates[1]*coordinates[1]);
	}
      };

      class singularity_2d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar singularity_2d(const double x[]) {
	  values[0] =  0.;
	}
      };

      class singularity_exact_2d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar singularity_exact_2d(const double x[]) {
	  double r = sqrt(coordinates[0]*coordinates[0] + coordinates[1]*coordinates[1]);
	  double theta;
	  if (r == 0.) {
	    values[0] = 0.;
	    return;
	  } else theta = asin(coordinates[1]/r);
	  if (coordinates[0] < 0) {
	    theta = 2*M_PI - theta;
	  }
	  values[0] =  pow(r, 2./3.)*sin((2./3.)*theta);
	}
      };

      class singularity_3d_exact_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar singularity_exact_3d(const double x[]) {
	  values[0] = sin(coordinates[0] + coordinates[1] + coordinates[2]);  
	}
      };

      class singularity_3d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar singularity_3d(const double x[]) {
	  values[0] = (3)*sin(coordinates[0] + coordinates[1] + coordinates[2]);
	}
      };

      class linear_2d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar linear_2d(const double x[]) {
         values[0] = -6.0*(coordinates[0] - 0.5) - 6.0*(coordinates[1] - 0.5);
	}
      };

      class quadratic_2d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar quadratic_2d(const double x[]) {
        values[0] = coordinates[0]*coordinates[0] + coordinates[1]*coordinates[1];
	}
      };

      class quadratic_vector : public ufc::function {
      private:
	int _embed_dim;
	int _vector_dim;
	double _factor;
      public:
	quadratic_vector (int embed_dim, int vector_dim, double factor = 1.) : ufc::function() {
	  _embed_dim = embed_dim;
	  _vector_dim = vector_dim;
	  _factor = factor;
	}
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar quadratic_2d(const double x[]) {
	  for (int i = 0; i < _vector_dim; i++) {
	    values[i] = 0.;
	    for (int j = 0; j < _embed_dim; j++) {
	      values[i] += _factor*coordinates[j]*coordinates[j];
	    }
	  }
	}
      };

      class cubic_2d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  //PetscScalar cubic_2d(const double x[]) {
	  values[0] = coordinates[0]*coordinates[0]*coordinates[0] - 1.5*coordinates[0]*coordinates[0] 
	    + coordinates[1]*coordinates[1]*coordinates[1] - 1.5*coordinates[1]*coordinates[1] + 0.5;
	}
      };

      class nonlinear_3d_scalar : public ufc::function {
      public:
	double lambda;
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) {
	  //PetscScalar nonlinear_3d(const double x[]) {
	  values[0] = -4.0 - lambda*PetscExpScalar((2.0/3.0)*(coordinates[0]*coordinates[0] + coordinates[1]*coordinates[1] + coordinates[2]*coordinates[2]));
	}
      };

      class linear_3d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) {
	  //PetscScalar linear_3d(const double x[]) {
	  values[0] =  -6.0*(coordinates[0] - 0.5) - 6.0*(coordinates[1] - 0.5) - 6.0*(coordinates[2] - 0.5);
	}
      };

      class quadratic_3d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) {
	  //PetscScalar quadratic_3d(const double x[]) {
	  values[0] = (2.0/3.0)*(coordinates[0]*coordinates[0] + coordinates[1]*coordinates[1] + coordinates[2]*coordinates[2]);
	}
      };

      class cubic_3d_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) {
	  //PetscScalar cubic_3d(const double x[]) {
	  values[0] = coordinates[0]*coordinates[0]*coordinates[0] - 1.5*coordinates[0]*coordinates[0] 
	    + coordinates[1]*coordinates[1]*coordinates[1] - 1.5*coordinates[1]*coordinates[1] 
	    + coordinates[2]*coordinates[2]*coordinates[2] - 1.5*coordinates[2]*coordinates[2] + 0.75;
	}
      };

      class cosx_scalar : public ufc::function {
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) {
	  //PetscScalar cos_x(const double x[]) {
	  values[0] = cos(2.0*PETSC_PI*coordinates[0]);
	}
      };

      class lid_driven : public ufc::function {
      private:
	int _dim;
	int _topdim;
	double _topcoord;
	int _flowdirectiondim;
	int _curdim;
	double _velocity; 
      public:
	lid_driven(int dim, double velocity, double topcoord) : ufc::function() {
	  _dim = dim;
	  _velocity = velocity;
	  _topcoord = topcoord;
	}
	virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
	  if (coordinates[1] >= _topcoord) {
	    if (_dim == 2) {
	      values[0] = 4*(coordinates[0] - 1.)*coordinates[0]*_velocity;
	    } else if (_dim == 3) {
	      values[0] = 16*(coordinates[0] - 1.)*coordinates[0]*(coordinates[2] - 1.)*coordinates[2]*_velocity;
	    }
	  } else {
	    values[0] = 0.;
	  }
	}
      };
      void pinion_pressure(Obj<PETSC_MESH_TYPE> m, int pinned_dimension, double coord_thresh, std::string marker_name, int marker_num) {
	//mark a single vertex on the marker label with "marker_num"; we probably don't want to use "marker" due to the possibility of relabeling a boundary vertex
	const Obj<PETSC_MESH_TYPE::label_type> & marker = m->createLabel(marker_name);
	//first vertex -- nei! don't pin the lid!
	int dim = m->getDimension();
	//int depth = m->depth();
	//ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);
	
	const Obj<PETSC_MESH_TYPE::label_sequence>& cells = m->heightStratum(0);
	const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
	PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
	PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();

	ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);

	while (c_iter != c_iter_end) {
	  ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*m->getSieve(), *c_iter, pV);
	  const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
	  const int                          oSize   = pV.getSize();
	  for (int i = 0; i < oSize; i++) {
	    if (m->depth(oPoints[i]) == 0) {
	      const double * coords = coordinates->restrictPoint(oPoints[i]);
	      if (fabs(coords[0] - 0.5) < 0.01 && fabs(coords[1] - 0.5) < 0.01 && dim == 2) {
		m->setValue(marker, oPoints[i], marker_num);
		m->setValue(marker, *c_iter, marker_num + 1);
		PetscPrintf(m->comm(), "pinned %d\n", oPoints[i]);
	      } else if (fabs(coords[0] - 0.5) < 0.1 && fabs(coords[1] - 0.5) < 0.1 && fabs(coords[2] - 0.5) < 0.1 && dim == 3) {
		m->setValue(marker, oPoints[i], marker_num);
		m->setValue(marker, *c_iter, marker_num + 1);
	      }
	    }
	  }
	  pV.clear();
	  c_iter++;
	}
	marker->view(marker_name.c_str());
      }      
      void mark_frictionless(Obj<PETSC_MESH_TYPE> m, int constrained_dim, double lowcoord, double highcoord, std::string marker_name, int marker_num) {
	//mark a single vertex on the marker label with "marker_num"; we probably don't want to use "marker" due to the possibility of relabeling a boundary vertex
	
	const Obj<PETSC_MESH_TYPE::label_type> & marker = m->createLabel(marker_name);
	int dim = m->getDimension();
	//int depth = m->depth();
	//ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);
	
	const Obj<PETSC_MESH_TYPE::label_sequence>& cells = m->heightStratum(0);  //cells with unknowns on the topological boundary
	const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
	PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
	PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();

	ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);

	while (c_iter != c_iter_end) {
	  ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*m->getSieve(), *c_iter, pV);
	  const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
	  const int                          oSize   = pV.getSize();
	  for (int i = 0; i < oSize; i++) {
	    PETSC_MESH_TYPE::point_type p = oPoints[i];
	    if (m->getValue(m->getLabel("marker"), p) == 1) {
	      const double * coords = m->restrictClosure(coordinates, p);
	      //if any aren't at the threshhold don't constrain
	      bool flag = true;
	      for (int d = 0; d < m->depth(p)+1; d++) {
		if (coords[d*dim+constrained_dim] <= lowcoord || coords[d*dim+constrained_dim] >= highcoord) {
		  //m->setValue(marker, p, marker_num);
		  //m->setValue(marker, *c_iter, marker_num+1);
		} else {
		  flag = false;
		}
		if (flag) {
		  m->setValue(marker, p, marker_num);
		  m->setValue(marker, *c_iter, marker_num+1);
		}
	      }
	    }
	  }
	  pV.clear();
	  c_iter++;
	}
	marker->view(marker_name.c_str());
      }      

    };
    class Ex_UFC : ALE::ParallelObject {
    public:
    protected:
      Ex_UFCOptions         _options;
      DM                   _dm;
      Obj<PETSC_MESH_TYPE> _mesh;
      DMMG                *_dmmg;
      UFCHook            *_ufchook;                     // Basic interface to the UFC object
      GenericFormSubProblem *  _subproblem;              // Problem interface object.

    public:
      Ex_UFC(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
        PetscErrorCode ierr = this->processOptions(comm, &this->_options);CHKERRXX(ierr);
        this->_dm   = PETSC_NULL;
        this->_dmmg = PETSC_NULL;
	this->_subproblem = PETSC_NULL;
	this->_ufchook = PETSC_NULL;
	//	this->_options.form_type = POISSON;
      };
      ~Ex_UFC() {
        PetscErrorCode ierr;

        if (this->_dmmg) {ierr = DMMGDestroy(this->_dmmg);CHKERRXX(ierr);}
        //ierr = this->destroyExactSolution(this->_options.exactSol);CHKERRXX(ierr);
        //ierr = this->destroyExactSolution(this->_options.error);CHKERRXX(ierr);
        ierr = this->destroyMesh();CHKERRXX(ierr);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "Ex_UFCProcessOptions"
      PetscErrorCode processOptions(MPI_Comm comm, Ex_UFCOptions *options) {
        const char    *runTypes[3] = {"full", "test", "mesh"};
        const char    *bcTypes[2]  = {"neumann", "dirichlet"};
        const char    *asTypes[4]  = {"full", "stored", "calculated"};
	const char    *fTypes[4]   = {"poisson", "vectorpoisson", "stokes", "mixedpoisson"};
        ostringstream  filename;
        PetscInt       run, bc, as, f;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        options->debug            = 0;
        options->run              = RUN_FULL;
	options->form_type        = POISSON;
        options->dim              = 2;
        options->generateMesh     = PETSC_TRUE;
        options->interpolate      = PETSC_TRUE;
        options->refinementLimit  = 0.0;
        options->bcType           = DIRICHLET;
        options->operatorAssembly = ASSEMBLY_FULL;
        options->reentrantMesh    = PETSC_FALSE;
        options->circularMesh     = PETSC_FALSE;
        options->refineSingularity= PETSC_FALSE;

        ierr = PetscOptionsBegin(comm, "", "UFC Example Problem Options", "DMMG");CHKERRQ(ierr);
          ierr = PetscOptionsInt("-debug", "The debugging level", "bratu.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
          run = options->run;
	  ierr = PetscOptionsEList("-form", "The form type", "ex_ufc.cxx", fTypes, 3, fTypes[options->form_type], &f, PETSC_NULL);CHKERRQ(ierr);
	  options->form_type = (UFCFormType) f;
          ierr = PetscOptionsEList("-run", "The run type", "ex_ufc.cxx", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
          options->run = (RunType) run;
          ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "bratu.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-reentrant", "Make a reentrant-corner mesh", "bratu.cxx", options->reentrantMesh, &options->reentrantMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-circular_mesh", "Make a reentrant-corner mesh", "bratu.cxx", options->circularMesh, &options->circularMesh, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsTruth("-singularity", "Refine the mesh around a singularity with a priori poisson error estimation", "bratu.cxx", options->refineSingularity, &options->refineSingularity, PETSC_NULL);CHKERRQ(ierr);
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
        ierr = PetscOptionsEnd();

        this->setDebug(options->debug);
        PetscFunctionReturn(0);
      };
    public: // Accessors
      Ex_UFCOptions *getOptions() {return &this->_options;};
      int  dim() const {return this->_options.dim;};
      bool interpolated() const {return this->_options.interpolate;};
      void interpolated(const bool i) {this->_options.interpolate = (PetscTruth) i;};
      BCType bcType() const {return this->_options.bcType;};
      UFCHook * ufcHook() const {return this->_ufchook;};
      void ufcHook(UFCHook * uh) {this->_ufchook = uh;};
      void bcType(const BCType bc) {this->_options.bcType = bc;};
      AssemblyType opAssembly() const {return this->_options.operatorAssembly;};
      UFCFormType FormType() const {return this->_options.form_type;};
      void FormType(const UFCFormType f) {this->_options.form_type = f;};
      void opAssembly(const AssemblyType at) {this->_options.operatorAssembly = at;};
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
            ierr = ALE::DMBuilder::createBoxMesh(comm(), dim(), PETSC_FALSE, interpolated(), debug(), &this->_dm);CHKERRQ(ierr);
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
	ierr = MeshDestroy((::Mesh) this->_dm);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      };

      #include "../examples/tutorials/bratu_2d.h"
      #include "../examples/tutorials/bratu_3d.h"
      #include "../examples/tutorials/vector_poisson_2d.h"
      #include "../examples/tutorials/vector_poisson_3d.h"
      #include "../examples/tutorials/stokes_2d.h"
      #include "../examples/tutorials/stokes_3d.h"

    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateProblem"
      PetscErrorCode createProblem() {
	PetscErrorCode ierr;
        PetscFunctionBegin;
	
	//first step; set up a UFCDiscretization object for each element or subelement in the form, depending on what kind of form we're using.

	if (dim() == 2) {
	  if (FormType() == POISSON) {
	    //PetscPrintf(_mesh->comm(), "setting up the laplacian dirichlet problem.\n");
	    _ufchook = new UFCHook(new bratu_2dBilinearForm(), new bratu_2dLinearForm());
	    //PetscPrintf(_mesh->comm(), "created the UFC hook.");
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    //PetscPrintf(_mesh->comm(), "created the subproblem\n");
	    //set up the unknown discretization.
	    Obj<UFCDiscretization> disc = new UFCDiscretization(_mesh->comm(), _ufchook->_b_finite_element, _ufchook->_cell);
	    disc->setNumDof(0, 1);
	    disc->setNumDof(1, 1);
	    disc->setNumDof(2, 0);
	    //PetscPrintf(_mesh->comm(), "set up the discretization numdofs\n");
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 6);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 6, 1);
	    if (bcType() == DIRICHLET) {
	      Obj<UFCBoundaryCondition> bc;
	      if (_options.reentrantMesh) {
		disc->setRHSFunction(new UFCFunctions::constant_vector(1, 0.));
		bc = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::singularity_exact_2d_scalar(), new UFCFunctions::singularity_exact_2d_scalar(), _ufchook->_b_finite_element, _ufchook->_cell, "marker", 1);
	      } else {
		disc->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
		bc = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_2d_scalar(), new UFCFunctions::quadratic_2d_scalar(), _ufchook->_b_finite_element, _ufchook->_cell, "marker", 1);
	      }
	      PetscPrintf(_mesh->comm(), "set up the discretization BCs\n");
	      disc->setBoundaryCondition(bc);
	    }
	    _subproblem->setDiscretization(disc);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	  } else if (FormType() == VECTORPOISSON) {

	    _ufchook = new UFCHook(new vector_poisson_2dBilinearForm(), new vector_poisson_2dLinearForm());
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    //do it as one discretization for now.
	    //TODO rewrite evaluate_rhs to take into account vectors; it should just zero everything now.
	    Obj<UFCDiscretization> disc_x = new UFCDiscretization(_mesh->comm(), _ufchook->_b_sub_finite_elements[0], _ufchook->_cell);
	    disc_x->setNumDof(0, 1);
	    disc_x->setNumDof(1, 0);
	    disc_x->setNumDof(2, 0);
	    disc_x->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
	    Obj<UFCDiscretization> disc_y = new UFCDiscretization(_mesh->comm(), _ufchook->_b_sub_finite_elements[1], _ufchook->_cell);
	    disc_y->setNumDof(0, 1);
	    disc_y->setNumDof(1, 0);
	    disc_y->setNumDof(2, 0);
	    disc_y->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 6);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 6, 1);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	    if (bcType() == DIRICHLET) {
	      Obj<UFCBoundaryCondition> bc_x = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(2, 1), new UFCFunctions::quadratic_vector(2, 1), _ufchook->_b_sub_finite_elements[0], _ufchook->_cell, "marker", 1);

	      Obj<UFCBoundaryCondition> bc_y = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(2, 1), new UFCFunctions::quadratic_vector(2, 1), _ufchook->_b_sub_finite_elements[1], _ufchook->_cell, "marker", 1);
	      disc_x->setBoundaryCondition(bc_x);
	      disc_y->setBoundaryCondition(bc_y);
	    }
	    _subproblem->setDiscretization("1", disc_x);
	    _subproblem->setDiscretization("2", disc_y);

	  } else if (FormType() == STOKES) {
	    _ufchook = new UFCHook(new stokes_2dBilinearForm(), new stokes_2dLinearForm());
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    Obj<UFCDiscretization> vx = new UFCDiscretization(_mesh->comm(), new stokes_2dBilinearForm_finite_element_0_0_0(), _ufchook->_cell);
	    vx->setNumDof(0, 1);
	    vx->setNumDof(1, 1);
	    vx->setNumDof(2, 0);
	    //vx->setRHSFunction(new UFCFunctions::lid_driven(1, 0.93, 0, 0, 0.0));
	    vx->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 bulk force
	    Obj<UFCDiscretization> vy = new UFCDiscretization(_mesh->comm(), new stokes_2dBilinearForm_finite_element_0_0_1(), _ufchook->_cell);
	    vy->setNumDof(0, 1);
	    vy->setNumDof(1, 1);
	    vy->setNumDof(2, 0);
	    //vy->setRHSFunction(new UFCFunctions::lid_driven(1, 0.93, 0, 1, 0.0));
	    vy->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 bulk force
	    Obj<UFCDiscretization> w = new UFCDiscretization(_mesh->comm(), new stokes_2dBilinearForm_finite_element_0_1(), _ufchook->_cell);
	    w->setNumDof(0, 1);
	    w->setNumDof(1, 0);
	    w->setNumDof(2, 0);
	    w->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 divergence??
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 15);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 15, 1);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	    if (bcType() == DIRICHLET) {
	      //UFCFunctions::mark_frictionless(_mesh, 1, 0., 1., "ly", 1);
	      UFCFunctions::pinion_pressure(_mesh, 1, 0.01, std::string("w"), 1);
	      Obj<UFCBoundaryCondition> lx = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::lid_driven(2, 1., 0.99), new UFCFunctions::constant_vector(1, 0.), vx->getFiniteElement(), _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> ly = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::constant_vector(1, 0.), new UFCFunctions::constant_vector(1, 0.), vy->getFiniteElement(), _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> wp = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::constant_vector(1, 0.), new UFCFunctions::constant_vector(1, 0.), w->getFiniteElement(), _ufchook->_cell, "w", 1);

	      vx->setBoundaryCondition("lx", lx);
	      vy->setBoundaryCondition("ly", ly);
	      w->setBoundaryCondition("w", wp);
	    }
	    _subproblem->setDiscretization("vx", vx);
	    _subproblem->setDiscretization("vy", vy);
	    _subproblem->setDiscretization("w", w);
	  }
	} else if (dim() == 3) {
	  if (FormType() == POISSON) {
	    _ufchook = new UFCHook(new bratu_3dBilinearForm(), new bratu_3dLinearForm());
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    Obj<UFCDiscretization> disc = new UFCDiscretization(_mesh->comm(), _ufchook->_b_finite_element, _ufchook->_cell);
	    //second order
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 10);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 10, 1);
	    disc->setRHSFunction(new UFCFunctions::constant_scalar);
	    disc->setNumDof(0, 1);
	    disc->setNumDof(1, 1);
	    disc->setNumDof(2, 0);
	    disc->setNumDof(3, 0);
	    if (bcType() == DIRICHLET) {
	      Obj<UFCBoundaryCondition> bc = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(3, 1), new UFCFunctions::quadratic_vector(3, 1), _ufchook->_b_finite_element, _ufchook->_cell, "marker", 1);
	      disc->setBoundaryCondition(bc);
	    }
	    _subproblem->setDiscretization(disc);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	  } else if (FormType() == VECTORPOISSON) {

	    _ufchook = new UFCHook(new vector_poisson_3dBilinearForm(), new vector_poisson_3dLinearForm());
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    //do it as one discretization for now.
	    //TODO rewrite evaluate_rhs to take into account vectors; it should just zero everything now.
	    Obj<UFCDiscretization> disc_x = new UFCDiscretization(_mesh->comm(), _ufchook->_b_sub_finite_elements[0], _ufchook->_cell);
	    disc_x->setNumDof(0, 1);
	    disc_x->setNumDof(1, 1);
	    disc_x->setNumDof(2, 0);
	    disc_x->setNumDof(3, 0);
	    disc_x->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
	    Obj<UFCDiscretization> disc_y = new UFCDiscretization(_mesh->comm(), _ufchook->_b_sub_finite_elements[1], _ufchook->_cell);
	    disc_y->setNumDof(0, 1);
	    disc_y->setNumDof(1, 1);
	    disc_y->setNumDof(2, 0);
	    disc_y->setNumDof(3, 0);
	    disc_y->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
	    Obj<UFCDiscretization> disc_z = new UFCDiscretization(_mesh->comm(), _ufchook->_b_sub_finite_elements[2], _ufchook->_cell);
	    disc_z->setNumDof(0, 1);
	    disc_z->setNumDof(1, 1);
	    disc_z->setNumDof(2, 0);
	    disc_z->setNumDof(3, 0);
	    disc_z->setRHSFunction(new UFCFunctions::constant_vector(1, -4.));
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 30);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 30, 1);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	    if (bcType() == DIRICHLET) {
	      Obj<UFCBoundaryCondition> bc_x = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(3, 1, 1.), new UFCFunctions::coordinates(0), _ufchook->_b_sub_finite_elements[0], _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> bc_y = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(3, 1, 1.), new UFCFunctions::coordinates(1), _ufchook->_b_sub_finite_elements[1], _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> bc_z = UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::quadratic_vector(3, 1, 1.), new UFCFunctions::coordinates(2), _ufchook->_b_sub_finite_elements[1], _ufchook->_cell, "marker", 1);

	      disc_x->setBoundaryCondition(bc_x);
	      disc_y->setBoundaryCondition(bc_y);
	      disc_z->setBoundaryCondition(bc_z);
	    }
	    _subproblem->setDiscretization("ux", disc_x);
	    _subproblem->setDiscretization("uy", disc_y);
	    _subproblem->setDiscretization("uz", disc_z);

	  } else if (FormType() == STOKES) {
	    _ufchook = new UFCHook(new stokes_3dBilinearForm(), new stokes_3dLinearForm());
	    _subproblem = new UFCFormSubProblem(_mesh->comm(), _ufchook->_cell, _ufchook->_b_finite_element);
	    Obj<UFCDiscretization> vx = new UFCDiscretization(_mesh->comm(), new stokes_3dBilinearForm_finite_element_0_0_0(), _ufchook->_cell);
	    vx->setNumDof(0, 1);
	    vx->setNumDof(1, 1);
	    vx->setNumDof(2, 0);
	    vx->setNumDof(3, 0);
	    vx->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 bulk force
	    Obj<UFCDiscretization> vy = new UFCDiscretization(_mesh->comm(), new stokes_3dBilinearForm_finite_element_0_0_1(), _ufchook->_cell);
	    vy->setNumDof(0, 1);
	    vy->setNumDof(1, 1);
	    vy->setNumDof(2, 0);
	    vy->setNumDof(3, 0);
	    vy->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 bulk force
	    Obj<UFCDiscretization> vz = new UFCDiscretization(_mesh->comm(), new stokes_3dBilinearForm_finite_element_0_0_2(), _ufchook->_cell);
	    vz->setNumDof(0, 1);
	    vz->setNumDof(1, 1);
	    vz->setNumDof(2, 0);
	    vz->setNumDof(3, 0);
	    vz->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 bulk force
	    Obj<UFCDiscretization> w = new UFCDiscretization(_mesh->comm(), new stokes_3dBilinearForm_finite_element_0_1(), _ufchook->_cell);
	    w->setNumDof(0, 1);
	    w->setNumDof(1, 0);
	    w->setNumDof(2, 0);
	    w->setNumDof(3, 0);
	    w->setRHSFunction(new UFCFunctions::constant_vector(1, 0.0));  //0 divergence??
	    Obj<UFCCellIntegral> jac_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_b_cell_integrals[0], _ufchook->_cell, "height", 0, 2, 34);
	    Obj<UFCCellIntegral> rhs_integral = new UFCCellIntegral(_mesh->comm(), _ufchook->_l_cell_integrals[0], _ufchook->_cell, "height", 0, 1, 34, 1);
	    _subproblem->setIntegral("jac_integral", jac_integral);
	    _subproblem->setIntegral("rhs_integral", rhs_integral);
	    if (bcType() == DIRICHLET) {
	      UFCFunctions::pinion_pressure(_mesh, 1, 0.01, "w", 1);
	      Obj<UFCBoundaryCondition> lx = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::lid_driven(3, 10., 0.99), new UFCFunctions::constant_vector(1, 0.), vx->getFiniteElement(), _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> ly = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::constant_vector(1, 0.), new UFCFunctions::constant_vector(1, 0.), vy->getFiniteElement(), _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> lz = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::constant_vector(1, 0.), new UFCFunctions::constant_vector(1, 0.), vz->getFiniteElement(), _ufchook->_cell, "marker", 1);
	      Obj<UFCBoundaryCondition> wp = new UFCBoundaryCondition(_mesh->comm(), new UFCFunctions::constant_vector(1, 0.), new UFCFunctions::constant_vector(1, 0.), w->getFiniteElement(), _ufchook->_cell, "w", 1);

	      vx->setBoundaryCondition("lx", lx);
	      vy->setBoundaryCondition("ly", ly);
	      vz->setBoundaryCondition("lz", lz);
	      w->setBoundaryCondition("w", wp);
	    }
	    _subproblem->setDiscretization("vx", vx);
	    _subproblem->setDiscretization("vy", vy);
	    _subproblem->setDiscretization("vz", vz);
	    _subproblem->setDiscretization("w", w);
	    
	  }
	}
	PetscPrintf(_mesh->comm(), "done setting up the problem\n");
	const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("default");
	s->setDebug(debug());
	_subproblem->setupField(this->_mesh, s);
	PetscTruth flag;
	ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
	if (flag) {s->view("Exact Solution");}
	ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &flag);CHKERRQ(ierr);
	if (flag) {_mesh->view("setup mesh");}
        PetscFunctionReturn(0);
      };
    public:

      //Note to self: the exact solution function should include all discretizations in it and therefore go over the TOTAL fiberdimension of the form
      //later do it per-discretization for easier mixed and otherwise problems.

      #undef __FUNCT__
      #define __FUNCT__ "CreateExactSolution"
      PetscErrorCode createExactSolution() {
        PetscTruth     flag;
        PetscErrorCode ierr;

        PetscFunctionBegin;
	::Mesh mesh = (::Mesh) this->_dm;
	
	ierr = MeshGetSectionReal(mesh, "exactSolution", &this->_options.exactSol.section);CHKERRQ(ierr);
	const Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("exactSolution");
	//this->_mesh->setupField(s);
	this->_subproblem->setupField(this->_mesh, s, 2, false, true);

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
        PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "DestroyExactSolution"
      PetscErrorCode destroyExactSolution(ALE::Problem::ExactSolType sol) {
        PetscErrorCode ierr;

        PetscFunctionBegin;
	ierr = SectionRealDestroy(sol.section);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "CreateSolver"
      PetscErrorCode createSolver() {
        PetscErrorCode ierr;

        PetscFunctionBegin;
        ierr = DMMGCreate(this->comm(), 1, this->_subproblem, &this->_dmmg);CHKERRQ(ierr);
        ierr = DMMGSetDM(this->_dmmg, this->_dm);CHKERRQ(ierr);
	if (opAssembly() == ALE::Problem::ASSEMBLY_FULL) {
	  //PetscPrintf(MPI_COMM_WORLD, "setting local.\n");
	  ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::RHS_FEMProblem, ALE::Problem::Jac_FEMProblem, 0, 0);CHKERRQ(ierr);
#if 0
	} else if (opAssembly() == ALE::Problem::ASSEMBLY_CALCULATED) {
            ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
            ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::UFCFunctions::Rhs_Unstructured, ALE::Problem::UFCFunctions::Jac_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
	} else if (opAssembly() == ALE::Problem::ASSEMBLY_STORED) {
	  ierr = DMMGSetMatType(this->_dmmg, MATSHELL);CHKERRQ(ierr);
	  ierr = DMMGSetSNESLocal(this->_dmmg, ALE::Problem::UFCFunctions::Rhs_Unstructured, ALE::Problem::UFCFunctions::Jac_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
#endif
	} else {
	  SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", opAssembly());
	}
	ierr = DMMGSetFromOptions(this->_dmmg);CHKERRQ(ierr);
	if (bcType() == ALE::Problem::NEUMANN) {
	  // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
	  ierr = DMMGSetNullSpace(this->_dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
	}
	PetscFunctionReturn(0);
      };
      #undef __FUNCT__
      #define __FUNCT__ "Ex_UFCSolve"
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
        ierr = PetscPrintf(comm(), "Number of nonlinear iterations = %D\n", its);CHKERRQ(ierr);
        ierr = PetscPrintf(comm(), "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
        if (flag) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
        ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
        if (flag && dim() == 2) {ierr = VecView(DMMGGetx(this->_dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
	const Obj<PETSC_MESH_TYPE::real_section_type>& sol = this->_mesh->getRealSection("default");
	SectionReal solution;
	double      error;
	
	ierr = MeshGetSectionReal((::Mesh) this->_dm, "default", &solution);CHKERRQ(ierr);
	ierr = SectionRealToVec(solution, (::Mesh) this->_dm, SCATTER_REVERSE, DMMGGetx(this->_dmmg));CHKERRQ(ierr);
	ierr = this->calculateError(solution, &error);CHKERRQ(ierr);
	ierr = PetscPrintf(comm(), "Total error: %g\n", error);CHKERRQ(ierr);
	ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
	if (flag) {
	  PetscViewer viewer;
	  
	  ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
	  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
	  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
	  ierr = PetscViewerFileSetName(viewer, "sol.vtk");CHKERRQ(ierr);
	  ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
	  if (FormType() == POISSON) {
	    ierr = SectionRealView(solution, viewer);CHKERRQ(ierr);
	  } else {
	    ierr = ALE::Problem::SubProblemView(solution, "velocity", viewer, 0, dim()-1);
	    if (FormType() == STOKES) {
	      //ierr = ALE::Problem::SubProblemView(solution, "pressure", viewer, dim(), dim());
	    }
	  }
	  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
	  
	  ierr = PetscViewerCreate(comm(), &viewer);CHKERRQ(ierr);
	  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
	  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
	  ierr = PetscViewerFileSetName(viewer, "error.vtk");CHKERRQ(ierr);
	  ierr = MeshView((::Mesh) this->_dm, viewer);CHKERRQ(ierr);
	  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
	  ierr = SectionRealView(this->_options.error.section, viewer);CHKERRQ(ierr);
	  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
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
        //PetscScalar  (*func)(const double *) = this->_options.exactFunc;
        PetscErrorCode ierr;
        PetscFunctionBegin;
        //const int dim = this->_mesh->getDimension();
        double  localError = 0.0;
        // Loop over cells
        const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = this->_mesh->heightStratum(0);
        for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          PetscScalar *x;
          //double       elemError = 0.0;

          //this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          if (debug()) {
            //std::cout << "Element " << *c_iter << " v0: (" << v0[0]<<","<<v0[1]<<")" << "J " << J[0]<<","<<J[1]<<","<<J[2]<<","<<J[3] << " detJ " << detJ << std::endl;
          }
          ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
          // Loop over quadrature points
	  /*TODO rewrite with form
          for(int q = 0; q < numQuadPoints; ++q) {
            for(int d = 0; d < dim; d++) {
              coords[d] = v0[d];
              for(int e = 0; e < dim; e++) {
                coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
              }
              if (debug()) {std::cout << "q: "<<q<<"  coords["<<d<<"] " << coords[d] << std::endl;}
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
          ierr = SectionRealUpdateAdd(this->_options.error.section, *c_iter, &elemError);CHKERRQ(ierr);
          localError += elemError;
	  */
        }
        ierr = MPI_Allreduce(&localError, error, 1, MPI_DOUBLE, MPI_SUM, comm());CHKERRQ(ierr);
        //ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
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
	ierr = this->calculateError(sol.section, &norm);CHKERRQ(ierr);
	ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
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
	::Mesh      mesh = (::Mesh) this->_dm;
	SectionReal residual;
	
	ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
	ierr = ALE::Problem::RHS_FEMProblem(mesh, sol.section, residual, this->_subproblem);CHKERRQ(ierr);
	if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
	ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
	ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
	ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
        PetscPrintf(comm(), "Residual for trial solution %s: %g\n", name, norm);
        PetscFunctionReturn(0);
      };
    };
  }
}

#endif
