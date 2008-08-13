#ifndef __UFCProblem
#define __UFCProblem

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
	  _b_sub_finite_elements[i] = _b_finite_element->create_sub_element(i);
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

#if 0

    class UFCCell : GeneralCell {
    private:
      ufc::cell * _cell; //the UFC cell.
    public:
      UFCCell (ufc::shape shape, int embed_dimension) : GeneralCell() {
	_embedded_dimension = embed_dimension;
	if (shape == ufc::interval) {
	  //allocate coordinates
	  _num_vertices = 2;
	  _coordinates = new double [_num_vertices*_embedded_dimension];
	  //setup the closure reorder
	  //setup the ufc::cell
	} else if (shape == ufc::triangle) {
	  //allocate coordinates
	  _num_vertices = 3;
	  _coordinates = new double [_num_vertices*_embedded_dimension];
	  //setup the closure reorder
	  //setup the ufc::cell
	} else if (shape == ufc::tetrahedron) {
	  //allocate coordinates
	  _num_vertices = 4;
	  _coordinates = new double [_num_vertices*_embedded_dimension];
	  //setup the closure reorder
	  //setup the ufc::cell
	} else throw Exception("UFCCell: Unsupported Shape");
      }
    };
    
#endif

    class UFCBoundaryCondition : public GeneralBoundaryCondition {
    private:
      //exact function

      ufc::function * _function;     //this will be evaluated to set the conditions.
      ufc::function * _exactSol;     //this will be evaluated in the case that there is an analytic solution to the associated problem.
      ufc::finite_element * _finite_element; //we have to evaluate both the function and the exactsol at some point
      ufc::cell * _cell;                     //ugh
      int * _closure2data;                    //map of LOCAL DISCRETIZATION closure dof indices to data dof indices; used for evaluation


    public:
      UFCBoundaryCondition(MPI_Comm comm, int debug = 0) : GeneralBoundaryCondition(comm, debug) {
	_function = PETSC_NULL;
	_exactSol = PETSC_NULL;
      }

      UFCBoundaryCondition(MPI_Comm comm, ufc::function * function, ufc::function * exactsol, ufc::finite_element * element, ufc::cell * cell, const std::string& name, const int marker, int debug = 0) : GeneralBoundaryCondition(comm, debug){
	_function = function;
	_exactSol = exactsol;
	this->setFiniteElement(element);
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

      //SHOULD be the same as the closure reorder for the overlying discretization.   This is just for library interface of course.

      virtual void setReorder(int * reorder) {
	this->_closure2data = reorder;
      }

      virtual const int * getReorder() {
	return this->_closure2data;
      }


      //yeargh!  do we ever use evaluate explicitly?
      virtual double integrateDual(unsigned int dof) const {
	int ufc_dof;
	ufc_dof = _closure2data[dof];
	//just evaluate the function using the coordinates; assume the order doesn't get that screwed up
	return this->_finite_element->evaluate_dof(ufc_dof, *_function, *_cell);
      }
    };
      
    class UFCDiscretization : public GeneralDiscretization {
      //implementation of the discretization class that provides UFC hooks.
    private:
      //Each form will have the finite element -- not true; you have to associate by which part of the SPACE the
      //integral exists over.  So, we must assign each integral to the appropriate space (as we might have TWO integrals
      //over say, the interior and then a boundary integral NOT TRUE AGAIN! all the integrals have to happen over the
      //overall form; they must be done a class up.  This should just be used to segment the space
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
      }

      ~UFCDiscretization() {
      }

      virtual void createReorder() {
	
	//create the forward and backward maps to and from UFC data layout to closure data layout.
	//do this case-by-case with info from the discretization
	
	int * reorder = new int[this->size()];
	std::map<int, int> closure_offsets;
	int offset = 0;
	if (this->getFiniteElement()->cell_shape() == ufc::interval) {
	  //find the indices corresponding to every point in the topology and assume they're ordered right; if not then we redo this later.
	  closure_offsets[0] = 0;
	  offset += this->getNumDof(1);
	  closure_offsets[1] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[2] = offset;
	  offset += this->getNumDof(0);
	  //we know the reorder; do it!
	  int offset = 0;
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[1] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[2] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[0] + i] = offset; 
	} else if (this->getFiniteElement()->cell_shape() == ufc::triangle) {
	  closure_offsets[0] = 0;
	  offset += this->getNumDof(2);
	  closure_offsets[1] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[2] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[3] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[4] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[5] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[6] = offset;
	  offset += this->getNumDof(0);
	  //we know the reorder; do it!
	  int offset = 0;
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[4] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[5] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[6] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[3] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[2] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[1] + i] = offset;
 	  for (int i = 0; i < this->getNumDof(2); i++, offset++) reorder[closure_offsets[0] + i] = offset;
	  //TET!
	} else if (this->getFiniteElement()->cell_shape() == ufc::tetrahedron) {
	  closure_offsets[0] = 0;
	  offset += this->getNumDof(3);
	  closure_offsets[1] = offset;
	  offset += this->getNumDof(2);
	  closure_offsets[2] = offset;
	  offset += this->getNumDof(2);
	  closure_offsets[3] = offset;
	  offset += this->getNumDof(2);
	  closure_offsets[4] = offset;
	  offset += this->getNumDof(2);
	  closure_offsets[5] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[6] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[7] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[8] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[9] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[10] = offset;
	  offset += this->getNumDof(1);
	  closure_offsets[11] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[12] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[13] = offset;
	  offset += this->getNumDof(0);
	  closure_offsets[14] = offset;
	  offset += this->getNumDof(0);
	  //we know the reorder; do it!
	  int offset = 0;
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[11] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[12] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[13] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(0); i++, offset++) reorder[closure_offsets[14] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[10] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[9] + i] = offset;
 	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[8] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[7] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[6] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(1); i++, offset++) reorder[closure_offsets[5] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(2); i++, offset++) reorder[closure_offsets[4] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(2); i++, offset++) reorder[closure_offsets[3] + i] = offset; 
	  for (int i = 0; i < this->getNumDof(2); i++, offset++) reorder[closure_offsets[2] + i] = offset;
 	  for (int i = 0; i < this->getNumDof(2); i++, offset++) reorder[closure_offsets[1] + i] = offset; 
 	  for (int i = 0; i < this->getNumDof(3); i++, offset++) reorder[closure_offsets[0] + i] = offset; 
	} else {
	  throw Exception("UFCDiscretization->createReordering(): unsupported cell geometry");
	}
	this->setReorder(reorder);
	//set this as the reorder of all boundary conditions involved here.
	Obj<names_type> bcs = this->getBoundaryConditions();
	names_type::iterator bc_iter = bcs->begin();
	names_type::iterator bc_iter_end = bcs->end();
	while (bc_iter != bc_iter_end) {
	  Obj<GeneralBoundaryCondition> bc = this->getBoundaryCondition(*bc_iter);
	  bc->setReorder(reorder);
	  bc_iter++;
	}
      }
	
      virtual void setReorder(const int * _reordering) {
	_closure2data = _reordering;
      }

      virtual const int * getReorder(const int * _reordering) {
	return _closure2data;
      }
 
      void setRHSFunction(ufc::function * function) {
	_rhs_function = function;
      }

      ufc::function * getRHSFunction() {
	return _rhs_function;
      }

      virtual double evaluateRHS(int dof) {

	//our notion of a degree of freedom in this case is a little different from the one used by UFC; they have the whole dimension of the vector space as 
	//one DOF with multiple coefficients; for the sake of being able to loop independent of the vector space here we split it up.

	int ufc_dof;
	if (_cell == PETSC_NULL) throw Exception("UFCBoundaryCondition->integrateDual: cell is not initialized");
	//TODO reordering 
	ufc_dof = _closure2data[dof];
	//switch the order

	//TODO: reorder based upon the cell attributes -- switch the dof_index to be the screwy indices as used by the
	//cells... this requires knowledge of the dofs per dim here

	//just evaluate the function using the coordinates; assume the order doesn't get that screwed up

	return _finite_element->evaluate_dof(ufc_dof, *_rhs_function, *_cell);
      }

      int size() {
	return _finite_element->space_dimension();
      }

      ufc::finite_element * getFiniteElement() {return _finite_element;};
      void setFiniteElement(ufc::finite_element * element) {_finite_element = element;};
      ufc::cell * getCell() {return _cell;};
      void setCell(ufc::cell * cell) {_cell = cell;};

#if 0

      virtual void orderTensor(int rank, int dim, double * reordered_tensor, double * original_tensor) {
	//reorder the tensor of rank n such that it can be effectively fed to UFC
	//precalculate the perturbation for a single tensor dimension here
	
	int entries = 1;
	int * coords = new int[rank];
	
	for (int r = 0; r < rank; r++) {
	  
	}
	delete coords;
      }

      virtual void setupReOrder() {
	
	//we need to loop over the closure, going over the fiberdimensions per topological depth to come up with forward and backwards index maps to and from UFC

	/*

        v0      v3
	o-------o
	|\     :|
	| e2  : |
	|  \ :  |
      e0|f0 \   |
	|  : \  |
        | :   \ |
        |:     \|
 	o-------o
        v1 e1   v2

	strategy: reorder maps!  create a map between the first index in the closure (multiplied once over for a space rank)

	 */


	unsigned int dim = _cell->topological_dimension();
	int space_size;
	int tensor_dimension = 1;
	for (int i = 0; i < _finite_element->space_rank(); i++) {
	  tensor_dimension = tensor_dimension * _finite_element->space_dimension(i);
	}
	if (dim == 1) {
	  int fdim_vertices = this->getNumDof(0);
	  int fdim_cells = this->getNumDof(1);
	  space_size = fdim_vertices*2+fdim_cells;
	  int index = 0;
	  std::map<int, int> closure_map; //use this to build the DoF maps based on fiberdimensions
	  std::map<int, int> sieve_map;
	  std::map<int, int> ufc_map;

	  //go through each topological object in each order and reassign it.

	  /*
	    the closure ordering will be:
	    sieve: e0 v1 v2
	    ufc  : v1 v2 e0

	    BUT, each field will have to be broken out by the space_dimension, as sieve puts all the points on the same 

	   */

	  //map from sieve ordering to UFC ordering

	  //edge
	  closure_map[0] = 2;
	  //first vertex
	  closure_map[1] = 0;
	  //second vertex
	  closure_map[2] = 1;

	  //map from closure sieve restrict indices to sieve closure indices -- takes into account fiberdimension.
	  //and the same map for ufc data entries to topological item indices; this will allow for us to construct forward and back maps
	  index = 0;
	  sieve_map[0] = index;
	  sieve_map[1] = index+fdim_cells;
	  sieve_map[2] = index+fdim_vertices;
	  
	  for (int i = 0; i < tensor_dimension; i++) { //loop over tensor dimension
	    //just enumerate the topological objects in order
	    
	  }

	  
	} else if (dim == 2) {
	  int fdim_vertices = this->getNumDof(0);
	  int fdim_edges = this->getNumDof(1);
	  int fdim_cells = this->getNumDof(2);
	  space_size = fdim_vertices*3 + fdim_edges*3 + fdim_cells;
	  /*
	    the closure ordering will be:
	    sieve: f0 e0 e1 e2 v0 v1 v2 
	    ufc  : v0 v1 v2 e2 e1 e0 f0
	   */

	  //face
	  closure_map[0] = 6;
	  //edge 1
	  closure_map[1] = 5;
	  //edge 2
	  closure_map[2] = 4;
	  //edge 3
	  closure_map[3] = 3;
	  //vertex 1
	  closure_map[4] = 0;
	  //vertex 2
	  closure_map[5] = 1;
	  //vertex 3
	  closure_map[6] = 2;


	    
	  int index = 0;
	  
	} else if (dim == 3) {
	  int fdim_vertices = this->getNumDof(0);
	  int fdim_edges = this->getNumDof(1);
	  int fdim_faces = this->getNumDof(2);
	  int fdim_cells = this->getNumDof(3);
	  _order = new int[fdim_vertices*4+ fdim_edges*6 + fdim_faces*4 + fdim_cells];
	  _reorder = new int[fdim_vertices*4+ fdim_edges*6 + fdim_faces*4 + fdim_cells];
	  /*
	    closure ordering will be:
	    sieve: c0 f0 f1 f2 f3 e0 e1 e2 e3 e4 e5 e6 v0 v1 v2 v3
	    ufc  : v0 v1 v2 v3 e6 e5 e4 e3 e2 e1 e0 f3 f2 f1 f0 c0
	   */

	  //tet

	  //this could be wrong; reorder if necessary later

	  closure_map[0] = 15;
	  //face 1
	  closure_map[1] = 14;
	  //face 2
	  closure_map[2] = 13;
	  //face 3
	  closure_map[3] = 12;
	  //face 4
	  closure_map[4] = 11;
	  //edge 1
	  closure_map[5] = 10;
	  //edge 2
	  closure_map[6] = 9;
	  //edge 3
	  closure_map[7] = 8;
	  //edge 4
	  closure_map[8] = 7;
	  //edge 5
	  closure_map[9] = 6;
	  //edge 6
	  closure_map[10] = 5;
	  //vertex 1
	  closure_map[11] = 0;
	  //vertex 2
	  closure_map[12] = 1;
	  //vertex 3
	  closure_map[13] = 2;
	  //vertex 4
	  closure_map[14] = 3;

	  int index = 0;
	} else {
	  throw Exception("UFCDiscretization->setupOrder(): Unsupported topological dimension");
	}
      }

#endif

    };

    class UFCCellIntegral : public GeneralIntegral {
    private:
      ufc::cell_integral * _integral;
      ufc::cell * _cell;
      const double ** _coefficients;
      double * _tmp_tensor;               //allows for us to reorder on the fly.
      double * _tmp_coefficients;
    public:
      UFCCellIntegral(MPI_Comm comm, int debug = 1) : GeneralIntegral(comm, debug) {
	this->_tmp_tensor = PETSC_NULL;
	this->_tmp_coefficients = PETSC_NULL;
	this->_integral = PETSC_NULL;
	this->_cell = PETSC_NULL;
	this->_coefficients = PETSC_NULL;
      }

      UFCCellIntegral(MPI_Comm comm, ufc::cell_integral * integral, ufc::cell * cell, name_type name, int label, int rank, int dimension, int ncoefficients = 0, int debug = 0) : GeneralIntegral(comm, name, label, ncoefficients, debug){
	this->_integral = integral;
	this->_cell = cell;
	this->setTensorRank(rank);
	this->setSpaceDimension(dimension);
	this->setNumCoefficients(ncoefficients);
	if (rank == 1) {
	  this->_tmp_tensor = new double[dimension];
	} else if (rank == 2) {
	  this->_tmp_tensor = new double[dimension*dimension];
	} else {
	  throw Exception("UFCCellIntegral::UFCCellIntegral(): unsupported tensor rank");
	}
	if (ncoefficients) {
	  this->_tmp_coefficients = new double[dimension*ncoefficients];
	  this->_coefficients = new const double*[dimension];
	} else this->_coefficients = PETSC_NULL;
      }

      virtual ~UFCCellIntegral() {
	delete _coefficients;
	if (_tmp_tensor) delete _tmp_tensor;
	if (_tmp_coefficients) delete _tmp_coefficients;
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
	//the coefficients have come in in
	  if (this->_coefficients)
	    for (int i = 0; i < this->getSpaceDimension(); i++) this->_coefficients[this->getReorder()[i]] = &coefficients[i*this->getNumCoefficients()];
	//the coefficients should already be reordered from the closure order to the normal order based upon the set
	_integral->tabulate_tensor(_tmp_tensor, _coefficients, *_cell);
	//reorder the tabulated tensor
	if (this->getTensorRank() == 1) {
	  for (int f = 0; f < this->getSpaceDimension(); f++) {
	    tensor[f] = _tmp_tensor[this->getReorder()[f]];
	  }
	}else if (this->getTensorRank() == 2) {
	  for (int f = 0; f < this->getSpaceDimension(); f++) {
	    for (int g = 0; g < this->getSpaceDimension(); g++) {
	      tensor[f*this->getSpaceDimension()+g] = _tmp_tensor[this->getReorder()[f]*this->getSpaceDimension()+this->getReorder()[g]]; //this should reorder it right.
	    }
	  }
	}
      }
    };

    //TODO: interior; exterior integral forms; these will have their own reordering notion.

    class UFCFormSubProblem : public GenericFormSubProblem {
      
      //Data and helper functions:

    private:

      ufc::cell * _cell;                     //we'll have to pass this on to all the children of this class;
      ufc::finite_element * _full_finite_element;  //the big one with all unknowns (even mixed) described.  all integrals are assumed to be defined on it

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
      
      UFCFormSubProblem(MPI_Comm comm, ufc::form * b_form, ufc::form * l_form, int debug = 0) : GenericFormSubProblem(comm, debug){
	//set up the discretizations, integrals, and other things; discretization-level things like boundary conditions must be set separately of course.
	//set the main finite element, the sub-elements, and 
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

      virtual void createReorder () {
	/*
	  GOAL: keep all interface-level things (in the other file) in *closure* order.
	  

	  The UFC ordering is assumed to go: 
	  1. discretization 1 entry 0 reordered per-cell like UFC
	     discretization 1 entry 1 ...
	  2. discretization 2 reordered per-cell like UFC
	  ... etc
	  
	  So, we first pull apart each discretization and reorder (based upon the indices of that discretization and the rank of the form, then based upon the discretization

	  Then pass the overall reorder to each integral to make all of this go smoothly.

	 */
	
	int * reorder = new int[this->localSpaceDimension()];
	//PetscPrintf(PETSC_COMM_WORLD, "starting the reorder calculation\n");
	//loop over discretizations, creating the local reorders
	Obj<names_type> discs = this->getDiscretizations();
	int offset = 0;
	for (names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); d_iter++) {
	  const Obj<GeneralDiscretization> disc = this->getDiscretization(*d_iter);
	  disc->createReorder();
	  const int * disc_indices = disc->getIndices();
	  const int * disc_reorder = disc->getReorder();
	  int disc_size = disc->size();
	  //PetscPrintf(PETSC_COMM_WORLD, "disc %s size: %d\n", (*d_iter).c_str(), disc_size);
	  for (int i = 0; i < disc_size; i++) {
	    reorder[disc_indices[i]] = disc_reorder[i] + offset;
	    //PetscPrintf(PETSC_COMM_WORLD, "map closure dof index %d to %d\n", disc_indices[i], reorder[disc_indices[i]]);
	  }
	  offset += disc_size;
	//use the discretizations closure indices to reorder the whole-field indices
	}	
	//set the reorder and set all the cell integrals to have this reorder as well.  Unfortunately this will take some hacking for the interior facet integrals.
	this->setReorder(reorder);
	Obj<names_type> integrals = this->getIntegrals();
	names_type::iterator i_iter = integrals->begin();
	names_type::iterator i_iter_end = integrals->end();
	while (i_iter != i_iter_end) {
	  Obj<GeneralIntegral> integral = this->getIntegral(*i_iter);
	  integral->setReorder(reorder);
	  i_iter++;
	}
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
      
    };
  }
}

#endif
