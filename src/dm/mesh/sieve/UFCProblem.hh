/*
  UFC Problem class for sieve; allows for UFC-assembled problems to be "bundled"

 */


#include <Mesh.hh>
#include <FEMProblem.hh>
#include <ufc.h>

namespace ALE {
  namespace Problem {

    class UFCHook { //initialize all the data from the UFC forms so we don't need to keep redoing it
    public:

      //the model for this is of course b(u, v) = (l(v), f)... because it's UFC.
      //if we need to extend this make these into sets.

      ufc::form * _lform;
      ufc::form * _bform;

      //perturbation for the closure that comes from the sieve.... SHOULD be set-able on a per-thing basis.
      int * s2u_closure;
      int * u2s_closure;

      //finite elements

      ufc::finite_element * _b_finite_element; //the thing upon which the cell integral is defined; however we want to crack it open for discretizations
      ufc::finite_element * _l_finite_element; //ditto

      std::map<int, ufc::finite_element *> _b_sub_finite_elements; 
      std::map<int, ufc::finite_element *> _l_sub_finite_elements;
      
      ufc::shape _cell_shape;

      ufc::cell * _cell; //assume, for now, that the mesh has one cell type

      //cell integrals

      std::map<int, ufc::cell_integral *> _b_cell_integrals;
      std::map<int, ufc::cell_integral *> _l_cell_integrals;

      //interior facet integrals

      std::map<int, ufc::interior_facet_integral *> _b_interior_facet_integrals;
      std::map<int, ufc::interior_facet_integral *> _l_interior_facet_integrals;

      //exterior facet integrals

      std::map<int, ufc::exterior_facet_integral *> _b_exterior_facet_integrals;
      std::map<int, ufc::exterior_facet_integral *> _l_exterior_facet_integrals;

    public:
      
      UFCHook(ufc::form * bform, ufc::form * lform) {

	_bform = bform;
	_lform = lform;
	
	//set up the cell, the finite_elements, the cell integrals
	
	//create the cell integrals
	//_num_l_cell_integrals = lform->num_cell_integrals();
	//_l_cell_integrals = new ufc::cell_integral *[_num_l_cell_integrals];
	for (unsigned int i = 0; i < lform->num_cell_integrals(); i++) {
	  _l_cell_integrals[i] = _lform->create_cell_integral(i);
	}

	if (_l_cell_integrals.size() == 0) throw Exception("No linear integrals in the form");
	
	//_num_b_cell_integrals = bform->num_cell_integrals();
	//_b_cell_integrals = new ufc::cell_integral *[_num_l_cell_integrals];
	for (unsigned int i = 0; i < bform->num_cell_integrals(); i++) {
	  _b_cell_integrals[i] = bform->create_cell_integral(i);
	}
	
	if (_b_cell_integrals.size() == 0) {
	  throw Exception("No bilinear integrals in the form");
	}
	
	//create the finite elements for the bform and lform; they should be the same I do believe

	_b_finite_element = _bform->create_finite_element(0);	//the other appears to be for quadrature

	//_num_finite_elements = _bform->num_coefficients() + _bform->rank();//_bform->num_finite_elements();
	//_finite_elements = new ufc::finite_element *[_num_finite_elements];

	if (_b_finite_element->num_sub_elements() > 1) for (unsigned int i = 0; i < _b_finite_element->num_sub_elements(); i++) {
	  _b_sub_finite_elements[i] = _bform->create_finite_element(i);
	}
	
	/*
	if (_b_sub_finite_elements.size() == 0) {
	  throw Exception("No finite elements in the form");
	}
	*/

	_l_finite_element = _lform->create_finite_element(0);

	//_num_coefficient_elements = _lform->num_coefficients() + _lform->rank();
	//_coefficient_elements = new ufc::finite_element *[_num_coefficient_elements];
	if (_l_finite_element->num_sub_elements() > 1) for (unsigned int i = 0; i < _l_finite_element->num_sub_elements(); i++) {
	  _l_sub_finite_elements[i] = lform->create_finite_element(i);
	}

	/*
	if (_l_sub_finite_elements.size() == 0) {
	  throw Exception("No coefficient elements in the form");
	}
	*/
	
	int dim, embedDim;
	//derive these from the mesh and/or the finite element;
	_cell = new ufc::cell;
	
	
	_cell_shape = _l_finite_element->cell_shape();


	if (_cell_shape == ufc::interval) {
	  dim = 1;
	  embedDim = 1;
	  _cell->entity_indices = new unsigned int *[dim+1];
	  _cell->entity_indices[0] = new unsigned int[2];
	  _cell->entity_indices[1] = new unsigned int[1];
	} else if (_cell_shape == ufc::triangle) {
	  dim = 2;
	  embedDim = 2;
	  _cell->entity_indices = new unsigned int *[dim+1];
	  _cell->entity_indices[0] = new unsigned int[3];
	  _cell->entity_indices[1] = new unsigned int[3];
	  _cell->entity_indices[2] = new unsigned int[1];
	} else if (_cell_shape == ufc::tetrahedron) {
	  dim = 3;
	  embedDim = 3;
	  _cell->entity_indices = new unsigned int *[dim+1];
	  _cell->entity_indices[0] = new unsigned int[4];
	  _cell->entity_indices[1] = new unsigned int[6];
	  _cell->entity_indices[2] = new unsigned int[4];
	  _cell->entity_indices[3] = new unsigned int[1];
	} else {
	  throw Exception("Unknown cell shape");
	}
	//create the cell
	_cell->topological_dimension = dim;
	_cell->geometric_dimension = embedDim;
	double * tmpCoords = new double[(dim+1)*embedDim];
	_cell->coordinates = new double *[dim+1];
	for (int i = 0; i < dim+1; i++) {
	  _cell->coordinates[i] = &tmpCoords[i*embedDim];
	}
	
      }
      
      ~UFCHook() {
	//remove all this stuff
	delete _lform;
	delete _bform;
	std::map<int, ufc::cell_integral *>::iterator i_iter = _l_cell_integrals.begin();
	std::map<int, ufc::cell_integral *>::iterator i_iter_end = _l_cell_integrals.end();
	while (i_iter != i_iter_end) {
	  delete i_iter->second;
	  i_iter++;
	}
	i_iter = _b_cell_integrals.begin();
	i_iter_end = _b_cell_integrals.end();
	while (i_iter != i_iter_end) {
	  delete i_iter->second;
	  i_iter++;
	}
	std::map<int, ufc::finite_element *>::iterator f_iter = _l_sub_finite_elements.begin();
	std::map<int, ufc::finite_element *>::iterator f_iter_end = _l_sub_finite_elements.end();
	while (f_iter != f_iter_end) {
	  delete f_iter->second;
	  f_iter++;
	}	 
	delete _cell->coordinates[0];
	delete _cell->coordinates;
      }

      //accessors
      

      void setCell(Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type c) {
	//setup the cell object such that it contains the mesh cell given
	
	const Obj<PETSC_MESH_TYPE::sieve_type> s = m->getSieve();
	Obj<PETSC_MESH_TYPE::real_section_type> coordinates = m->getRealSection("coordinates");
	int dim = m->getDimension();
	if ((int)_cell->topological_dimension != m->getDimension() - m->height(c)) throw ALE::Exception("Wrong element dimension for this UFC form");
	//int depth = m->depth();
	//the entity indices should be FIXED because we don't care to use any of the DoFMap stuff
	ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);
	ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*m->getSieve(), c, pV);

	const PETSC_MESH_TYPE::point_type * oPoints = pV.getPoints();
	int num_oPoints = pV.getSize();
	int vertex_index = 0;
	for (int t = 0; t < num_oPoints; t++) {
	  if (m->depth(oPoints[t]) == 0) {
	    // _cell->entity_indices[0][vertex_index] = oPoints[t];
	      const double * tmpcoords = coordinates->restrictPoint(oPoints[t]);
	      for (int i = 0; i < dim; i++) {
		_cell->coordinates[vertex_index][i] = tmpcoords[i];
	      }
	      vertex_index++;
	  }
	} 
      }
    };
    
    class UFCBoundaryCondition : public GeneralBoundaryCondition {
    private:
      //exact function

      ufc::function * _function;     //this will be evaluated to set the conditions.
      ufc::function * _exactSol;     //this will be evaluated in the case that there is an analytic solution to the associated problem.
      ufc::finite_element * _finite_element; //we have to evaluate both the function and the exactsol at some point
      ufc::cell * _cell;                     //ugh

    public:
      UFCBoundaryCondition(MPI_Comm comm, int debug = 0) : GeneralBoundaryCondition(comm, debug) {
	_function = PETSC_NULL;
	_exactSol = PETSC_NULL;
      }

      UFCBoundaryCondition(MPI_Comm comm, ufc::function * function, ufc::function * exactsol, ufc::finite_element * element, ufc::cell * cell, const std::string& name, const int marker, int debug = 0) : GeneralBoundaryCondition(comm, debug){
	_function = function;
	_exactSol = exactsol;
	_finite_element = element;
	_cell = cell;
	this->setLabelName(name);
	this->setMarker(marker);
      }

      void setFiniteElement(ufc::finite_element * finite_element) {
	_finite_element = finite_element;
      }
      
      ufc::finite_element * getFiniteElement() {
	return _finite_element;
      }
      
      void setCell(ufc::cell * cell) {
	_cell = cell;
      }

      ufc::cell * getCell() {
	return _cell;
      }

      void setFunction(ufc::function * function) {
	_function = function;
      }

      ufc::function * getFunction() {
	return _function;
      }


      //yeargh!  do we ever use evaluate explicitly?
      virtual double integrateDual(unsigned int dof) const {
	if (_cell == PETSC_NULL) throw Exception("UFCBoundaryCondition->integrateDual: cell is not initialized");
	//TODO reordering
	//just evaluate the function using the coordinates; assume the order doesn't get that screwed up
	return _finite_element->evaluate_dof(dof, *_function, *_cell);
      }
    };
      
    class UFCDiscretization : public GeneralDiscretization {
      //implementation of the discretization class that provides UFC hooks.
    private:
      //Each form will have the finite element -- not true; you have to associate by which part of the SPACE the integral exists over.  So, we must assign each integral to
      //the appropriate space (as we might have TWO integrals over say, the interior and then a boundary integral
      //NOT TRUE AGAIN! all the integrals have to happen over the overall form; they must be done a class up.  This should just be used to segment the space
      //
      ufc::cell           * _cell;
      ufc::finite_element * _finite_element;  //this will typically be a subelement.
      ufc::function * _rhs_function;          //TODO: generalize this further


    public:
      UFCDiscretization(MPI_Comm comm, int debug = 1) : GeneralDiscretization(comm, debug) {
      }

      UFCDiscretization(MPI_Comm comm, ufc::finite_element * element, ufc::cell * cell, int debug = 1) : GeneralDiscretization(comm, debug){
	_finite_element = element;
	_cell = cell;
	//TODO: set the numdof object up through use of the finite element; right now this feature doesn't exist.
      }

      ~UFCDiscretization() {
      }

      void setRHSFunction(ufc::function * function) {
	_rhs_function = function;
      }

      ufc::function * getRHSFunction() {
	return _rhs_function;
      }

      virtual double evaluateRHS(int dof) {
	return _finite_element->evaluate_dof(dof, *_rhs_function, *_cell);
      }

      int size() {
	return _finite_element->space_dimension();
      }

      ufc::finite_element * getFiniteElement() {return _finite_element;};
      void setFiniteElement(ufc::finite_element * element) {_finite_element = element;};
      ufc::cell * getCell() {return _cell;};
      void setCell(ufc::cell * cell) {_cell = cell;};
      
    };

    class UFCCellIntegral : public GeneralIntegral {
    private:
      ufc::cell_integral * _integral;
      ufc::cell * _cell;
      const double ** _coefficients;
    public:
      UFCCellIntegral(MPI_Comm comm, int debug = 1) : GeneralIntegral(comm, debug) {
	_integral = PETSC_NULL;
	_cell = PETSC_NULL;
	_coefficients = PETSC_NULL;
      }

      UFCCellIntegral(MPI_Comm comm, ufc::cell_integral * integral, ufc::cell * cell, name_type name, int label, int rank, int dimension, int ncoefficients = 0, int debug = 0) : GeneralIntegral(comm, name, label, ncoefficients, debug){
	this->_integral = integral;
	this->_cell = cell;
	this->setTensorRank(rank);
	this->setSpaceDimension(dimension);
	this->setNumCoefficients(ncoefficients);
	if (ncoefficients) {
	  this->_coefficients = new const double*[dimension];
	} else this->_coefficients = PETSC_NULL;
      }

      virtual ~UFCCellIntegral() {
	delete _coefficients;
      };

      virtual ufc::cell * getCell() {
	return _cell;
      }

      virtual void setCell(ufc::cell * cell) {
	_cell = cell;
      }

      virtual void setIntegral(ufc::cell_integral * integral) {
	_integral = integral;
      }
      
      virtual ufc::cell_integral * getIntegral () {
	return _integral;
      }


      //there will eventually be some reordering occuring here, but for now just assume we've done the right thing.
      virtual void tabulateTensor(double * tensor, const double * coefficients = PETSC_NULL) {
	//just run the frickin' tabulate_tensor routine from the form on what gets spit out by restrictClosure right now; rebuild or reorder later.
	if (_coefficients)
	for (int i = 0; i < this->getSpaceDimension(); i++) this->_coefficients[i] = &coefficients[i*this->getNumCoefficients()];
	_integral->tabulate_tensor(tensor, _coefficients, *_cell);
      }
    };

    //TODO: interior; exterior integral forms; these will have their own reordering notion.

    class UFCFormSubProblem : public GenericFormSubProblem {
      
      //Data and helper functions:

    private:

      ufc::cell * _cell;                     //we'll have to pass this on to all the children of this class;
      ufc::finite_element * _full_finite_element;  //the big one with all unknowns (even mixed) described.

    public:

      UFCFormSubProblem(MPI_Comm comm, int debug = 0) : GenericFormSubProblem(comm, debug) {
	//yeah
	_cell = PETSC_NULL;
	_full_finite_element = PETSC_NULL;
      }

      UFCFormSubProblem(MPI_Comm comm, ufc::cell * cell, ufc::finite_element * finite_element, int debug = 0) : GenericFormSubProblem(comm, debug) {
	_cell = cell;  //yeargh!
	_full_finite_element = finite_element;
      }
      
      ufc::cell * getCell() {
	return _cell;
      }

      ufc::finite_element * getFiniteElement() {
	return _full_finite_element;
      }

      virtual int localSpaceDimension() {
	return _full_finite_element->space_dimension();
      }


      virtual void setCell(Obj<PETSC_MESH_TYPE> mesh, PETSC_MESH_TYPE::point_type c) const {
	if (_cell == PETSC_NULL) throw Exception("UFCFormSubProblem: Uninitialized cell");
	Obj<PETSC_MESH_TYPE::real_section_type> coords = mesh->getRealSection("coordinates");
	const double * cellcoords = mesh->restrictClosure(coords, c);
	int index = 0;
	for (unsigned int i = 0; i < _cell->topological_dimension+1; i++) {
	  for (unsigned int j = 0; j < _cell->geometric_dimension; j++) {
	    _cell->coordinates[i][j] = cellcoords[index];
	    index++;
	  }
	}
      }

      ~UFCFormSubProblem(){};
      
    private:
      
      //helper functions
      void reorderRestrictToUFC (double * reordered, const double * values) {
	//considering the constrained number of geometries supported by UFC, do this casewise by the finite element shape
	
      }
      void reorderUFCToRestrict(double * reordered, const double * values) {
	//considering the constrained number of geometries supported by UFC< do this casewise by the finite element shape
      }
    };
  }
}
