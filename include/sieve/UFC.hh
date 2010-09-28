/*

Routines for linking UFC to Sieve and PETSc

*/

#include "Mesh.hh"
#include <petscmat.h>
#include <ufc.h>

//we SHOULD have some overlying "problem" object.  Let's do that here!

#if 0

namespace ALE {
  class sieve_mesh_wrapper : ufc::mesh {

  public:
    Obj<PETSC_MESH_TYPE> m;
    sieve_mesh_wrapper(Obj<PETSC_MESH_TYPE> mesh) {
      m = mesh;
      Obj<PETSC_MESH_TYPE::real_section_type> coordinates = m->getRealSection("coordinates");
      int dim = m->getDimension();
      topological_dimension = dim;
      geometric_dimension = m->getFiberDimension(*m->depthStratum(0)->begin());
      num_entities = new unsigned int[dim+1];
      int depth = m->depth();
      for (int i = 0; i < depth; i++) {
	num_entities[i] = 0;
      }
      if (depth == 1) {
	num_entities[0] = m->depthStratum(0);
	num_entities[dim] = m->heightStratum(0);
      } else {
	if (depth != dim+1) throw Exception("Cannot handle partially interpolated sieves.");
	for (int i = 0; i < dim+1; i++) {
	  num_entities[i] = m->getDepthStratum(i)->size();
	}
	
      }
    }
    
  };

  class sieve_cell_wrapper : ufc::cell {
  public:
    Obj<PETSC_MESH_TYPE> m;                    // the presently associated mesh
    int num_corners;                           //for the sake of clarity; the number of vertices
    PETSC_MESH_TYPE::point_type current_point; // the presently associated sieve cell


    sieve_cell_wrapper() {
    }
    sieve_cell_wrapper(ufc::finite_element finite_element) {
      //properly initialize the cell from the form
      int geometric_dimension = m->getRealSection("coordinates")->getFiberDimension(m->depthStratum(0)->begin());
      if (finite_element->cell_shape() == ufc::interval) {
	num_corners = 2;
	topological_dimension = m->getDimension(); //we must clarify what this means in the mesh library
	element_entities = new int *[2];
	element_entities[0] = new int[2]; //vertices
	element_entities[1] = new int[1]; //line
	double * tmpcoords = new double[coorddim*2];
        coordinates = new double *[2];
	for (int i = 0; i < 2; i++) {
	  coordinates[i] = &tmpcoords[coorddim*i];
	}
      } else if (finite_element->cell_shape() == ufc::triangle) {
	num_corners = 3;
	element_entities = new int *[3];
	element_entities[0] = new int[3]; //vertices
	element_entities[1] = new int[3]; //edges
	element_entities[2] = new int[1]; //cell
	double * tmpcoords = new double[coorddim*3];
        coordinates = new double *[3];
	for (int i = 0; i < 3; i++) {
	  coordinates[i] = &tmpcoords[coorddim*i];
	}
      } else if (finite_element->cell_shape() == ufc::tetrahedron) {
	num_corners = 4;
	element_entities = new int *[2];
	element_entities[0] = new int[4]; //vertices
	element_entities[1] = new int[6]; //edges
	element_entities[2] = new int[4]; //faces
	element_entities[3] = new int[1]; //tetrahedron
 	double * tmpcoords = new double[coorddim*4];
        coordinates = new double *[4];
	for (int i = 0; i < 4; i++) {
	  coordinates[i] = &tmpcoords[coorddim*i];
	}
      } else throw Exception("Unsupported geometry");
    }
    ~sieve_cell_wrapper() {
    }
    void setMesh(Obj<PETSC_MESH_TYPE> mesh) {
      m = mesh;
    }
    Obj<PETSC_MESH_TYPE> getMesh() {
      return m;
    }
    void setCell(PETSC_MESH_TYPE::point_type p) {
      if (m->height(p) != 0) throw Exception("sieve_cell_wrapper: the point must be a cell");
      current_point = p;
      //copy over the coordinates through the restrictClosure and copy
      const double * coords = m->;
      for (int i = 0; i < num_corners; i++) {
	for (int j = 0; j < dim; j++) {
	  coordinates[i][j] = coords[i*coordim+j];
	}
      }
      //copy over the entity indices such that they're consistent with the 
    }
    void reorientSieveCell(PETSC_MESH_TYPE::point_type p) {
      //changing the orientation will be hard; keep it the same for now as it should orient normals right
      //this will not be so bad in the long run
      
    }
    Obj<PETSC_MESH_TYPE::sieve_type::oConeArray> reorderSieveClosure(PETSC_MESH_TYPE::point_type p) {
      //return m->getSieve()->closure(p); //null reorder for now. NEI! the oriented closure!
      return PETSC_MESH_TYPE::sieve_alg_type::orientedClosure(m, m->getArrowSection("orientation"), p);
    }
    void reorderSieveRestrictClosure(const double * coefficients) {
      //order the restricted closure given in coefficients based upon the cell mapping here
    }
    
  };

  class sieve_function_wrapper : ufc::function {
  public:
    (*_func)(double * values, const double * coords);  //the simple function that computes the value based on the coordinates.
    int _rank;                                       //the number of values returned by the function

    sieve_function_wrapper(){};
    sieve_function_wrapper(const double * (*func)(const double * coords), int rank) {
      _func = func;
      _rank = rank;
    }
    const double * (*)(const double *) getFunction() {
      return _func;
    }
    void setFunction(const double * (*func)(const double * coords), int rank) {
      _func = func;
    }
    void evaluate(double * values, const double * coordinates, ufc::cell &cell) {
      _func(values, coordinates);
    }
  };

  //this class takes coordinates in a cell and produces output based upon those coordinates and the section over the mesh in that cell
  //might come in handy for generalized MG, etc.
  class section_wrapper_function : ufc::function {
  public:

    ALE::Obj<PETSC_MESH_TYPE> _m;
    ALE::Obj<PETSC_MESH_TYPE::section_type> _s;
    //PETSC_MESH_TYPE::point_type c;

    ufc::form * _form;
    ufc::finite_element * _finite_element;
    ufc::cell_integral * _cell_integral;

    section_wrapper_function(){};
    section_wrapper_function(Obj<PETSC_MESH_TYPE> m, Obj<PETSC_MESH_TYPE::section_type> s, ufc::form form, ufc::finite_element finite_element, ufc::cell_integral::cell_integral){
      _m = m;
      _s = s;
      _form = form;
      _finite_element = finite_element;
      _cell_integral = cell_integral;
    }
    
    void evaluate(double * values, const double * coordinates, ufc::cell &cell)
    {
      //evaluate the degrees of freedom on the interior of the given cell; note that this requires that coordinates be within the cell to really make sense.
      //note for the future: we could wrap this easily into the multigrid routines
      //we need an additional aggregation array of size value_rank;
      const double * coefficients = m->restrictClosure(_s, cell->current_point);
      const double * tmp_values = new double[_finite_element->value_rank()];
      for (int i = 0; i < _finite_element->value_rank(); i++) {
	values[i] = 0.;
      }
      for (int i = 0; i < _finite_element->value_dimension(); i++) { //loop over basis functions
	_finite_element->evaluate_basis(i, tmp_values, coordinates, cell);
	for (int j = 0; j < _finite_element->value_rank()+1; i++) {
	  values[i] += coefficients[i*(finite_element->value_rank()+1)+j]*tmp_values[j];
	}
      }
      delete tmp_values;
    }
  };

  class boundary_condition {
  public:
    PetscBool  (*_func)(PETSC_MESH_TYPE::point_type, const double *);
    int marker;
    int cellmarker;

    boundary_condition(){};
    boundary_condition (PetscBool  (*func)(PETSC_MESH_TYPE::point_type, const double *), int mark = 1, int cellmark = 2) {
      
    }
    void applyBC(Obj<PETSC_MESH_TYPE> m) {
    }
  };
  
  PetscBool  Scalar_Dirichlet_Pred(PETSC_MESH_TYPE::point_type p, const double * coords) {
    //set up the marker
    //if anything in the star of the thing has support size 1 but not height 0 then the thing's on the boundary! mark it!
    Obj<PETSC_MESH_TYPE::sieve_type::supportArray> star = 
  }

  class UFCProblem : ALE::ParallelObject {
  public:
    //sieve parts
    Obj<PETSC_MESH_TYPE> _mesh;
    
    //UFC parts
    //Forms:
    ufc::form * _bform;
    ufc::form * _lform;
    //Coefficients:
    int _num_bcoefficients;
    double * b_w;
    int _num_lcoefficients;
    double * l_w;
    //Finite Elements:
    int _num_finite_elements;
    ufc::finite_element ** _finite_elements;
    //"Coefficient" elements from the RHS linear forms
    int _num_coefficient_elements;
    ufc::finite_element ** _coefficient_elements;
    //Cell types; perhaps not
    int _num_cell_types;
    ufc::cell ** _cell;
    //Cell integrals
    int _num_cell_integrals;
    ufc::cell_integral * _cell_integrals;
    //Functions:
    // - The RHS Function 
    ufc::function * _rhs_funct;
    // - The "Exact Solution"
    ufc::function * _exact_solution;
    //We also need to define some sort of predicate system for the boundary like dolfin;
    //This will involve some setting of the boundary marker, not necessarily based upon the topological boundary, but usually that.
    //Initialization
    UFCProblem(){};
    //give it a mesh, give it a form, mark boundary segments through some f
    UFCProblem(Obj<PETSC_MESH_TYPE> m, ufc::form * bform, ufc::form * lform){
      _mesh = m;
      _bform = bform;
      _lform = lform;
      //set up the bilinear form finite elements, and cell integral.
      _num_finite_elements = bform->num_finite_elements();
      _finite_elements = new ufc::finite_element *[_num_finite_elements];
      for (int i = 0; i < _num_finite_elements; i++) {
	
      }
      //set up the linear form finite elements and cell integrals.

      //set up the 

    }
    ~UFCProblem(){};
    //Accessors
    void setMesh(Obj<PETSC_MESH_TYPE> m){mesh = m;}
    Obj<PETSC_MESH_TYPE> getMesh() {return mesh;}
    Mesh getPetscMesh() {
      return PETSC_NULL;
    }
    void setForm(ufc::form * f) {form = f;}
    ufc::form * getForm() {}
    void setRHSFunction(ufc::function * f){rhs_funct = f;}
    ufc::function * getRHSFunction();
    void setExactSolution(ufc::function * f) {exact_solution = f;}
    ufc::function * getExactSolution() {
      return exact_solution;
    }
    

    //Misc
    void setupUFC(){
      //initialize the cell, function, and finite element structures for this given problem
      finite_element = form->create_finite_element(0);
      dof_map = form->create_dof_map(0);
      cell_integrals = form->create_cell_integral(0);
      
      cell = new ();
      
    };
    void setCell(PETSC_MESH_TYPE::point_type c) {
      
    }
    void setupFields(Obj<PETSC_MESH_TYPE> m, Obj<PETSC_MESH_TYPE::real_section_type> s, ufc::form form, ufc::function function){
      setCell
    };
    void assembleMatrix() {
      //use the bform to create the matrix
      
      //partial assembly?
    }
    void assembleRHS() {
      //use the lform on the interior and the bform on the boundaries to assemble the RHS
      
    }
    void setFieldfromFunction(Obj<PETSC_MESH_TYPE> m, Obj<PETSC_MESH_TYPE::real_section_type> s, ufc::form form, ufc::function function) {
      
    }
  };
}

#endif


/*
Wrapper to ufc::function for double * func(double * coords)
 */

class function_wrapper_scalar : public ufc::function {
private:
  PetscScalar (*function)(const double * coords);

public:
  void setFunction(PetscScalar (*func)(const double *)) {
    function = func;
  }
  virtual void evaluate(double * values, const double * coordinates, const ufc::cell &c) const {
    values[0] = (*function)(coordinates);
  }
};

/*
Do we even need this one if we're not going to be assembling ourselves?
*/

#undef __FUNCT__
#define __FUNCT__ "Map_SieveCell_UFCCell"

void Map_SieveCell_UFCCell(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type c, ufc::form * form, ufc::cell * cell, const PETSC_MESH_TYPE::point_type * _oPoints = PETSC_NULL, const int _num_oPoints= -1) {
  //set up the ufc cell to be equivalent to the sieve cell given by c;  Assume that the # of dofs is constant
  //PetscErrorCode ierr;
  
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type> s = m->getSieve();
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> coordinates = m->getRealSection("coordinates");
  int dim = m->getDimension();
  PetscPrintf(m->comm(), "cell of dimension: %d\n", cell->topological_dimension);
  if ((int)cell->topological_dimension != m->getDimension() - m->height(c)) throw ALE::Exception("Wrong element dimension for this UFC form");
  //  ALE::Obj<PETSC_MESH_TYPE::oConeArray> cell_closure = PETSC_MESH_TYPE::sieve_alg_type::orientedClosure(m, m->getArrowSection("orientation"), c);
  const PETSC_MESH_TYPE::point_type * oPoints = _oPoints;
  int num_oPoints = _num_oPoints;
  if (oPoints == PETSC_NULL) {
    ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> oC((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);
    ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*s, c, oC);
    PetscPrintf(m->comm(), "Got the orientedClosure\n");
    oPoints = oC.getPoints();
    num_oPoints = oC.getSize();
  }
  
  //PETSC_MESH_TYPE::oConeArray::iterator cc_iter = cell_closure->begin();
  //PETSC_MESH_TYPE::oConeArray::iterator cc_iter_end = cell_closure->end();
  int vertex_index = 0;
  
  for (int t = 0; t < num_oPoints; t++) {
    //PetscPrintf(PETSC_COMM_WORLD, "%d is in the closure\n", oPoints[t]);
  //while (cc_iter != cc_iter_end) {
    //FOR NOW: first order lagrange; if you have vertices then put 'em in.  This should be ordered
    // (and declare victory!)
    if (m->depth(oPoints[t]) == 0) {
      //"entities"
      //PetscPrintf(m->comm(), "%d is vertex %d\n", cc_iter->first, vertex_index);
      cell->entity_indices[0][vertex_index] = oPoints[t];
      //PetscPrintf(m->comm(), "%d: ", cc_iter->first);
      //and coordinates
      const double * tmpcoords = coordinates->restrictPoint(oPoints[t]);
      for (int i = 0; i < dim; i++) {
	cell->coordinates[vertex_index][i] = tmpcoords[i];
      }
      vertex_index++;
    }
  }
  PetscPrintf(m->comm(), "done with cell map\n");
}


#undef __FUNCT__
#define __FUNCT__ "Assemble_Mat_UFC"

PetscErrorCode Assemble_Mat_UFC(Mesh mesh, SectionReal section, Mat A, ufc::form * form) {
  PetscErrorCode ierr;
  //get, from the mesh, the assorted structures we need to do this. (numberings)
  PetscFunctionBegin;

  Obj<PETSC_MESH_TYPE::real_section_type> s;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);

  Obj<PETSC_MESH_TYPE> m;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  PetscPrintf(m->comm(), "Beginning Matrix assembly.\n");

  ALE::Obj<PETSC_MESH_TYPE::real_section_type> coordinates = m->getRealSection("coordinates");
  ALE::Obj<PETSC_MESH_TYPE::label_sequence> cells = m->heightStratum(0);
  const Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", s);
  int dim = m->getDimension();

  ufc::cell cell;
  ufc::finite_element * finite_element = form->create_finite_element(0);
  
  //initialize the ufc infrastructure
  cell.geometric_dimension = dim; //might be different; check the fiberdimension of the coordinates
  cell.topological_dimension = dim;
  cell.entity_indices = new unsigned int *[dim+1];
  cell.entity_indices[0] = new unsigned int[dim+1];
  double * tmpcellcoords = new double [(dim+1)*dim];
  int space_dimension = finite_element->space_dimension();
  //allow both our functions and theirs to use it!
  double * localTensor = new double[space_dimension*space_dimension];
  //double ** localTensor = new double*[space_dimension];
  //for(int i = 0; i < space_dimension; i++) {
  //  localTensor[i] = &localTensor_pointer[space_dimension*i];
  //}
  cell.coordinates = new double *[dim+1];
  
  for (int i = 0; i < dim+1; i++) {
    cell.coordinates[i] = &tmpcellcoords[i*dim];
  }
  ufc::cell_integral** cell_integrals;
  cell_integrals = new ufc::cell_integral*[form->num_cell_integrals()];
  if (form->num_cell_integrals() <= 0) throw ALE::Exception("Number of cell integrals in UFC form is 0.");
  for (unsigned int i = 0; i < form->num_cell_integrals(); i++){
    cell_integrals[i] = form->create_cell_integral(i);
  }
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
  PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
  while (c_iter != c_iter_end) {
    Map_SieveCell_UFCCell(m, *c_iter, form, &cell);
    //for now just do the first cell integral.  Fix when you talk to someone about what exactly having more than one means.
    //todo: coefficients.... ask if they're global and if yes ask why.
    cell_integrals[0]->tabulate_tensor(localTensor, (double * const *)PETSC_NULL, cell);
    //see what the local tensor coming out looks like:
    if (1) {
      //maybe print the local tensor?
    }
    ierr = updateOperator(A, m, s, order, *c_iter, localTensor, ADD_VALUES);CHKERRQ(ierr);
    c_iter++;
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  if (1) {
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  //throw ALE::Exception("Finished the jacobian assembly for UFC; aborting for now in case it's messed up.");
  PetscFunctionReturn(0);
}


#if 0
//GET THE NEW RHS_UNSTRUCTURED FOR THIS


PetscErrorCode Assemble_RHS_UFC(Mesh mesh, ufc::form * bform, ufc::form * lform, SectionReal X, SectionReal section, PetscScalar (*exactFunc)(const double *)) {
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  //const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  ufc::finite_element * finite_element = lform->create_finite_element(0);
  ufc::cell cell;
  cell.geometric_dimension = dim;
  cell.topological_dimension = dim;
  cell.entity_indices = new unsigned int *[dim+1];
  cell.entity_indices[0] = new unsigned int[dim];
  cell.coordinates = new double *[dim+1];
  double * tmpcellcoords = new double[dim*(dim+1)];
  for (int i = 0; i < dim+1; i++) {
    cell.coordinates[i] = &tmpcellcoords[i*dim];
  }
  ufc::cell_integral** cell_integrals;
  cell_integrals = new ufc::cell_integral*[bform->num_cell_integrals()];
  if (bform->num_cell_integrals() <= 0) throw ALE::Exception("Number of cell integrals in UFC form is 0.");
  for (unsigned int i = 0; i < bform->num_cell_integrals(); i++){
    cell_integrals[i] = bform->create_cell_integral(i);
  }

  ufc::cell_integral** cell_integrals_linear = new ufc::cell_integral*[lform->num_cell_integrals()];
  for (unsigned int i = 0; i < lform->num_cell_integrals(); i++) {
    cell_integrals_linear[i] = lform->create_cell_integral(i);
  }

  const int numBasisFuncs = finite_element->space_dimension();

  //double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat;

  ierr = SectionRealZero(section);CHKERRQ(ierr);
  ierr = PetscMalloc2(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  //  ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  Obj<PETSC_MESH_TYPE::real_section_type> xSection;
  Obj<PETSC_MESH_TYPE::real_section_type> fSection;
  int c = 0;
  double ** w = new double *[lform->num_coefficients()];
  function_wrapper_scalar sf;
  sf.setFunction(exactFunc);

  for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter, ++c) {
    ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    //set up the weight vector to be 0.
    //three steps for this:

    //build B in the finite element space
    //  involves calling the 
    //construct A local to the boundary 
    //subtract AX from the boundary

    //create the "neumann" RHS and put it in the vector
    //m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    Map_SieveCell_UFCCell(m, *c_iter, bform, &cell);

    PetscScalar *x;

    ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);

    for (int f = 0; f < numBasisFuncs; f++) {
       elemVec[f] = 0. - finite_element->evaluate_dof(f, sf, cell);
       //PetscPrintf(m->comm(), "Elemvec[f](before): %f\n", elemVec[f]);
    }
    for(unsigned int i = 0; i < lform->num_coefficients(); i++) {
      w[i] = new double[numBasisFuncs];
      for (int j = 0; j < numBasisFuncs; j++){
	w[i][j] = elemVec[j];
      }
    }

    cell_integrals_linear[0]->tabulate_tensor(elemVec, w, cell);
    cell_integrals[0]->tabulate_tensor(elemMat, w, cell);

    for(int f = 0; f < numBasisFuncs; ++f) {
      for(int g = 0; g < numBasisFuncs; ++g) {
	elemVec[f] += elemMat[f*numBasisFuncs+g]*x[g];
      }
      //PetscPrintf(m->comm(), "x[f]: %f\n", x[f]);
      //PetscPrintf(m->comm(), "elemVec[f]: %f\n", elemVec[f]);
    }
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);
    //m->updateAdd(fSection, c, elemVec);
  }
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  //ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  // Subtract the constant
  if (m->hasRealSection("constant")) {
    const Obj<PETSC_MESH_TYPE::real_section_type>& constant = m->getRealSection("constant");
    Obj<PETSC_MESH_TYPE::real_section_type>        s;
    
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    s->axpy(-1.0, constant);
  }
  PetscFunctionReturn(0);
}

#endif


PetscErrorCode Assemble_RHS_UFC(Mesh mesh, ufc::form * bform, ufc::form * lform, SectionReal X, SectionReal section, PetscScalar (*exactFunc)(const double *)) {

  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const Obj<PETSC_MESH_TYPE::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  PetscScalar *elemVec, *elemMat;

  ufc::finite_element * finite_element = lform->create_finite_element(0);
  ufc::cell cell;
  cell.geometric_dimension = dim;
  cell.topological_dimension = dim;
  cell.entity_indices = new unsigned int *[dim+1];
  cell.entity_indices[0] = new unsigned int[dim];
  cell.coordinates = new double *[dim+1];
  double * tmpcellcoords = new double[dim*(dim+1)];
  for (int i = 0; i < dim+1; i++) {
    cell.coordinates[i] = &tmpcellcoords[i*dim];
  }
  ufc::cell_integral** cell_integrals;
  cell_integrals = new ufc::cell_integral*[bform->num_cell_integrals()];
  if (bform->num_cell_integrals() <= 0) throw ALE::Exception("Number of cell integrals in UFC form is 0.");
  for (unsigned int i = 0; i < bform->num_cell_integrals(); i++){
    cell_integrals[i] = bform->create_cell_integral(i);
  }

  ufc::cell_integral** cell_integrals_linear = new ufc::cell_integral*[lform->num_cell_integrals()];
  for (unsigned int i = 0; i < lform->num_cell_integrals(); i++) {
    cell_integrals_linear[i] = lform->create_cell_integral(i);
  }
  double ** w = new double *[lform->num_coefficients()];
  function_wrapper_scalar sf;
    sf.setFunction(exactFunc);
  const int numBasisFuncs = finite_element->space_dimension();


  ierr = SectionRealZero(section);CHKERRQ(ierr);
  ierr = PetscMalloc2(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  //ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    //m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    Map_SieveCell_UFCCell(m, *c_iter, bform, &cell);
    //if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);

    PetscScalar *x;

    ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);

    for (int f = 0; f < numBasisFuncs; f++) {
       elemVec[f] = 0. - finite_element->evaluate_dof(f, sf, cell);
       //PetscPrintf(m->comm(), "Elemvec[f](before): %f\n", elemVec[f]);
    }
    for(unsigned int i = 0; i < lform->num_coefficients(); i++) {
      w[i] = new double[numBasisFuncs];
      for (int j = 0; j < numBasisFuncs; j++){
	w[i][j] = elemVec[j];
      }
    }

    cell_integrals_linear[0]->tabulate_tensor(elemVec, w, cell);
    cell_integrals[0]->tabulate_tensor(elemMat, w, cell);

    for(int f = 0; f < numBasisFuncs; ++f) {
      for(int g = 0; g < numBasisFuncs; ++g) {
	elemVec[f] += elemMat[f*numBasisFuncs+g]*x[g];
      }
      //PetscPrintf(m->comm(), "x[f]: %f\n", x[f]);
      //PetscPrintf(m->comm(), "elemVec[f]: %f\n", elemVec[f]);
    }
    
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
  }
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  //ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  // Subtract the constant
  if (m->hasRealSection("constant")) {
    const Obj<PETSC_MESH_TYPE::real_section_type>& constant = m->getRealSection("constant");
    Obj<PETSC_MESH_TYPE::real_section_type>        s;

    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    s->axpy(-1.0, constant);
  }
  PetscFunctionReturn(0);
}


//Integrator function based upon a given UFC:
//Takes a mesh, cell, and a UFC and integrate for all the unknowns on the cell


#undef __FUNCT__
#define __FUNCT__ "IntegrateDualBasis_UFC"

PetscErrorCode IntegrateDualBasis_UFC(ALE::Obj<PETSC_MESH_TYPE> m, PETSC_MESH_TYPE::point_type c, ufc::form & f) {
  
}

//you still have to wrap this one as the fields are set up on the basis of the discretizations; you have to set up the discretization as it would be from the form, so we have to at least fill in the fiberdimension parts of the discretization type such that setupFields can do its work.  This will be equivalent to the CreateProblem_gen_0 stuff that FIAT + Generator spits out.

//CreateProblem_UFC
//Takes a UFC form and generates the entire problem from it.  This involves building a discretization object within the mesh corresponding to what is sent to UFC.  Unfortunately UFC handles all the element/vectorish stuff on its own, but 
PetscErrorCode CreateProblem_UFC(DM dm, const char * name, ufc::form * form,  const int numBC, const int *markers, double (**bcFuncs)(const double * coords), double(*exactFunc)(const double * coords)) {
  Mesh mesh = (Mesh) dm;
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr = 0;
  //you need some finite element information from the form.
  ufc::finite_element * finite_element = form->create_finite_element(0);
  //needed information from the form.
  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  //int dim = m->getDimension();
  const ALE::Obj<ALE::Discretization>& d = new ALE::Discretization(m->comm(), m->debug()); //create the UFC
  //for now handle only vertex unknowns; complain about the fact that Dofs per dimension isn't in the release version of UFC.
  d->setNumDof(0, 1);
  /*
    for (int i = 0; i < dim+1; i++) {
    //for each element level; find the fiberdimension from the discretization and set it in the discretization.
    d->setNumDof(
    }
  */
  d->setQuadratureSize(finite_element->space_dimension());
  //boundary conditions
  for (int c = 0; c < numBC; c++) {
    const ALE::Obj<ALE::BoundaryCondition>& b = new ALE::BoundaryCondition(m->comm(), m->debug());
    ostringstream n;
    b->setLabelName("marker");
    b->setMarker(markers[c]);
    b->setFunction(bcFuncs[c]);
    //b->setDualIntegrator(IntegrateDualBasis_gen_2);
    n << c;
    d->setBoundaryCondition(n.str(), b);
    if (exactFunc) {
      const ALE::Obj<ALE::BoundaryCondition>& e = new ALE::BoundaryCondition(m->comm(), m->debug());
      e->setLabelName("marker");
      e->setFunction(exactFunc);
      e->setDualIntegrator(PETSC_NULL); //TODO
      d->setExactSolution(e); 
    }
  }
  m->setDiscretization(name, d);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization_UFC"

void SetupDiscretization_UFC(ALE::Obj<PETSC_MESH_TYPE> m, ufc::form * form) {
  ALE::Obj<PETSC_MESH_TYPE::sieve_type> s = m->getSieve();
  //we will treat the thing arising from the UFC form as a SINGLE discretization such that the separation of forms is handled transparent of sieve;
  //watch out if this screws up stuff involving the eventual output of a solution; we may have to reprocess or something silly like that.
  //also, where there are multiple forms what should we do?
  ALE::Obj<ALE::Discretization> d = m->getDiscretization("ufc_u");
  
}

//Comment: we shouldn't need to do this!  Tie this in properly with the discretization object and then setupfield with noupdate; write a separate routine for setting the boundary values given a wrapped function.

#undef __FUNCT__
#define __FUNCT__ "SetupField_UFC"

/*
  This is essentially a copy of m->setupField(s) such that it can use the UFC dualintegrator from the associated form.
 */

#if 0

 void SetupField_UFC(ALE::Obj<PETSC_MESH_TYPE> m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s, ufc::form * form, const int cellMarker = 2, const bool noUpdate = false){

   const ALE::Obj<PETSC_MESH_TYPE::names_type>& discs  = m->getDiscretizations();
   const int              debug  = s->debug();
   PETSC_MESH_TYPE::names_type  bcLabels;
   int                    maxDof;

   //setup the necessary UFC structures here
   ufc::finite_element * finite_element = form->create_finite_element(0);
   function_wrapper_scalar sf;
   ufc::cell cell;
   int dim = m->getDimension();
   int embeddim = m->getRealSection("coordinates")->getFiberDimension(*m->depthStratum(0)->begin());
   cell.geometric_dimension = embeddim;
   cell.topological_dimension = dim;
   cell.entity_indices = new unsigned int *[dim+1];
   cell.entity_indices[0] = new unsigned int[dim];
   cell.coordinates = new double *[dim+1];
   
   double * coordpointer = new double[(dim+1)*dim];
   for (int i = 0; i < dim+1; i++) {
     cell.coordinates[i] = &coordpointer[dim*i];
   }
   
   s->setChart(m->getSieve()->getChart());
   PetscPrintf(m->comm(), "Set the chart\n");
   maxDof = m->setFiberDimensions(s, discs, bcLabels);
   PetscPrintf(m->comm(), "Set the max dof\n");
   m->calculateIndices();
   PetscPrintf(m->comm(), "Calculated the indices\n");
   m->calculateIndicesExcluded(s, discs);
   PetscPrintf(m->comm(), "Calculated the excluded indices\n");
   m->allocate(s);
   PetscPrintf(m->comm(), "Allocated\n");
   s->defaultConstraintDof();
   PetscPrintf(m->comm(), "Set the default constraint DOF");
   const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellExclusion = m->getLabel("cellExclusion");
   PetscPrintf(m->comm(), "Cell exclusion\n");
   if (debug > 1) {std::cout << "Setting boundary values" << std::endl;}
   PetscPrintf(m->comm(), "At the boundary condition loop\n");
   for(PETSC_MESH_TYPE::names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
     const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&     boundaryCells = m->getLabelStratum(*n_iter, cellMarker);
     //     const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates   = m->getRealSection("coordinates");
     const ALE::Obj<PETSC_MESH_TYPE::names_type>&         discs         = m->getDiscretizations();
     ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = m->getSieve();
     const PETSC_MESH_TYPE::point_type               firstCell     = *boundaryCells->begin();
     const int                      numFields     = discs->size();
     PETSC_MESH_TYPE::real_section_type::value_type *values        = new PETSC_MESH_TYPE::real_section_type::value_type[m->sizeWithBC(s, firstCell)];
     int                           *dofs          = new int[maxDof];
     int                           *v             = new int[numFields];
     //double                        *v0            = new double[m->getDimension()];
     //double                        *J             = new double[m->getDimension()*m->getDimension()];
     //double                         detJ;
     ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> oC((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);

     for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
    
       //const Obj<PETSC_MESH_TYPE::coneArray>      closure = PETSC_MESH_TYPE::sieve_alg_type::closure(m, m->getArrowSection("orientation"), *c_iter);
       //const PETSC_MESH_TYPE::coneArray::iterator end     = closure->end();
       
       if (debug > 1) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
       ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(sieve, *c_iter, oC);
       const PETSC_MESH_TYPE::point_type * oPoints = oC.getPoints();
       const int num_oPoints = oC.getSize();
       Map_SieveCell_UFCCell(m, *c_iter, form, &cell, oPoints, num_oPoints);
       PetscPrintf(m->comm(), "successfully quit the cell map\n");
       //m->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
       for(int f = 0; f < numFields; ++f) v[f] = 0;
       for(int t = 0; t < num_oPoints; t++) {
	 const int cDim = s->getConstraintDimension(oPoints[t]);
	 int       off  = 0;
	 int       f    = 0;
	 int       i    = -1;
	 if (debug > 1) {std::cout << "    point " << oPoints[t] << std::endl;}
	 if (cDim) {
	   if (debug > 1) {std::cout << "      constrained excMarker: " << m->getValue(cellExclusion, *c_iter) << std::endl;}
	   for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	     const ALE::Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	     const ALE::Obj<PETSC_MESH_TYPE::names_type> bcs = disc->getBoundaryConditions();
	     const int                       fDim    = s->getFiberDimension(oPoints[t], f);//disc->getNumDof(m->depth(*cl_iter));
	     const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	     int                             b       = 0;
	     
	     for(PETSC_MESH_TYPE::names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter, ++b) {
	       const ALE::Obj<ALE::BoundaryCondition>& bc    = disc->getBoundaryCondition(*bc_iter);
	       const int                          value = m->getValue(m->getLabel(bc->getLabelName()), oPoints[t]);
	       
	       if (b > 0) v[f] -= fDim;
	       
	       if (value == bc->getMarker()) {
		 if (debug > 1) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
		 //instead, we use the form's dual integrator (evaluation
		 sf.setFunction(bc->getFunction());
		 /*
		   for(int d = 0; d < fDim; ++d, ++v[f]) {
		   dofs[++i] = off+d;
		   if (!noUpdate) values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
		   if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
		   }
		 */
		    
		 for (int d = 0; d < fDim; ++d, ++v[f]) {
		   dofs[++i] = off+d;
		   if (!noUpdate) {
		     values[indices[v[f]]] = finite_element->evaluate_dof(v[f], sf, cell);
		     PetscPrintf(m->comm(), "evaluated DOF %d\n", f);
		   }
		   
		 }
		 ++b;
		 break;
	       } else {
		 if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
		 for(int d = 0; d < fDim; ++d, ++v[f]) {
		   values[indices[v[f]]] = 0.0;
		   if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
		 }
	       }
	     }
	     if (b == 0) {
	       if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
	       for(int d = 0; d < fDim; ++d, ++v[f]) {
		 values[indices[v[f]]] = 0.0;
		 if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
	       }
	     }
	     off += fDim;
	   }
	   if (i != cDim-1) {throw ALE::Exception("Invalid constraint initialization");}
	   s->setConstraintDof(oPoints[t], dofs);
	 } else {
	   if (debug > 1) {std::cout << "      unconstrained" << std::endl;}
	   for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	     const Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	     const int                       fDim    = s->getFiberDimension(oPoints[t], f);//disc->getNumDof(m->depth(*cl_iter));
	     const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	     
	     if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
	     for(int d = 0; d < fDim; ++d, ++v[f]) {
	       values[indices[v[f]]] = 0.0;
	       if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
	     }
	   }
	 }
       }
#if 0
       if (debug > 1) {
	 const Obj<PETSC_MESH_TYPE::sieve_type::coneArray>      closure = PETSC_MESH_TYPE::sieve_alg_type::closure(m, m->getArrowSection("orientation"), *c_iter);
	 const PETSC_MESH_TYPE::sieve_type::coneArray::iterator end     = closure->end();
	 
	 for(int f = 0; f < numFields; ++f) v[f] = 0;
	 for(PETSC_MESH_TYPE::sieve_type::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
	   int f = 0;
	   for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	     const Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	     const int                       fDim    = s->getFiberDimension(*cl_iter, f);
	     const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	     
	     for(int d = 0; d < fDim; ++d, ++v[f]) {
	       std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
	     }
	   }
	 }
       }
#endif
       if (!noUpdate) {
	 m->updateAll(s, *c_iter, values);
       }
     }
     PetscPrintf(m->comm(), "Done with the cell loop.\n");
     delete [] dofs;
     delete [] values;
   }
   if (debug > 1) {s->view("");}
 }

#endif

void SetupField_UFC(ALE::Obj<PETSC_MESH_TYPE> m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s, ufc::form * form, const int cellMarker = 2, const bool noUpdate = false) {
  typedef ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
  const ALE::Obj<PETSC_MESH_TYPE::names_type>& discs  = m->getDiscretizations();
  const int              debug  = s->debug();
  PETSC_MESH_TYPE::names_type             bcLabels;
  
  s->setChart(m->getSieve()->getChart());
  int maxdof = m->setFiberDimensions(s, discs, bcLabels);
  m->calculateIndices();
  m->calculateIndicesExcluded(s, discs);
  m->allocate(s);
  s->defaultConstraintDof();
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellExclusion = m->getLabel("cellExclusion");

   ufc::finite_element * finite_element = form->create_finite_element(0);
   function_wrapper_scalar sf;
   ufc::cell cell;
   int dim = m->getDimension();
   int embeddim = m->getRealSection("coordinates")->getFiberDimension(*m->depthStratum(0)->begin());
   cell.geometric_dimension = embeddim;
   cell.topological_dimension = dim;
   cell.entity_indices = new unsigned int *[dim+1];
   cell.entity_indices[0] = new unsigned int[dim];
   cell.coordinates = new double *[dim+1];
   
   double * coordpointer = new double[(dim+1)*dim];
   for (int i = 0; i < dim+1; i++) {
     cell.coordinates[i] = &coordpointer[dim*i];
   }


  if (debug > 1) {std::cout << "Setting boundary values" << std::endl;}
  for(PETSC_MESH_TYPE::names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
    function_wrapper_scalar sf;
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&     boundaryCells = m->getLabelStratum(*n_iter, cellMarker);
    const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates   = m->getRealSection("coordinates");
    const ALE::Obj<PETSC_MESH_TYPE::names_type>&         discs         = m->getDiscretizations();
    const PETSC_MESH_TYPE::point_type               firstCell     = *boundaryCells->begin();
    const int                      numFields     = discs->size();
    PETSC_MESH_TYPE::real_section_type::value_type *values        = new PETSC_MESH_TYPE::real_section_type::value_type[m->sizeWithBC(s, firstCell)];
    int                           *dofs          = new int[maxdof];
    int                           *v             = new int[numFields];
    double                        *v0            = new double[m->getDimension()];
    double                        *J             = new double[m->getDimension()*m->getDimension()];
    double                         detJ;
    Visitor pV((int) pow(m->getSieve()->getMaxConeSize(), m->depth())+1, true);
    
    for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*m->getSieve(), *c_iter, pV);
      const Visitor::point_type *oPoints = pV.getPoints();
      const int                  oSize   = pV.getSize();
      
      if (debug > 1) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
      m->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
      for(int f = 0; f < numFields; ++f) v[f] = 0;
      for(int cl = 0; cl < oSize; ++cl) {
	const int cDim = s->getConstraintDimension(oPoints[cl]);
	int       off  = 0;
	int       f    = 0;
	int       i    = -1;
	
	if (debug > 1) {std::cout << "    point " << oPoints[cl] << std::endl;}
	if (cDim) {
	  if (debug > 1) {std::cout << "      constrained excMarker: " << m->getValue(cellExclusion, *c_iter) << std::endl;}
	  for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	    const Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	    const Obj<PETSC_MESH_TYPE::names_type>           bcs     = disc->getBoundaryConditions();
	    const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
	    const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	    int                             b       = 0;
	    
	    for(PETSC_MESH_TYPE::names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter, ++b) {
	      const Obj<ALE::BoundaryCondition>& bc    = disc->getBoundaryCondition(*bc_iter);
	      const int                          value = m->getValue(m->getLabel(bc->getLabelName()), oPoints[cl]);
	      sf.setFunction(bc->getFunction());
	      if (b > 0) v[f] -= fDim;
	      if (value == bc->getMarker()) {
		if (debug > 1) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
		for (int d = 0; d < fDim; ++d, ++v[f]) {
		  dofs[++i] = off+d;
		  if (!noUpdate) {
		    values[indices[v[f]]] = finite_element->evaluate_dof(v[f], sf, cell);
		    PetscPrintf(m->comm(), "evaluated DOF %d\n", f);
		  }
		}
		/*
		for(int d = 0; d < fDim; ++d, ++v[f]) {
		  dofs[++i] = off+d;
		  if (!noUpdate) values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
		  if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
		}
		*/
		// Allow only one condition per point
		++b;
		break;
	      } else {
		if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
		for(int d = 0; d < fDim; ++d, ++v[f]) {
		  values[indices[v[f]]] = 0.0;
		  if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
		}
	      }
	    }
	    if (b == 0) {
	      if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
	      for(int d = 0; d < fDim; ++d, ++v[f]) {
		values[indices[v[f]]] = 0.0;
		if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
	      }
	    }
	    off += fDim;
	  }
	  if (i != cDim-1) {throw ALE::Exception("Invalid constraint initialization");}
	  s->setConstraintDof(oPoints[cl], dofs);
	} else {
	  if (debug > 1) {std::cout << "      unconstrained" << std::endl;}
	  for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	    const Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	    const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
	    const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	    
	    if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
	    for(int d = 0; d < fDim; ++d, ++v[f]) {
	      values[indices[v[f]]] = 0.0;
	      if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
	    }
	  }
	}
      }
      if (debug > 1) {
	for(int f = 0; f < numFields; ++f) v[f] = 0;
	for(int cl = 0; cl < oSize; ++cl) {
	  int f = 0;
	  for(PETSC_MESH_TYPE::names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	    const Obj<ALE::Discretization>& disc    = m->getDiscretization(*f_iter);
	    const int                       fDim    = s->getFiberDimension(oPoints[cl], f);
	    const int                      *indices = disc->getIndices(m->getValue(cellExclusion, *c_iter));
	    
	    for(int d = 0; d < fDim; ++d, ++v[f]) {
	      std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
	    }
	  }
	}
      }
      if (!noUpdate) {
	m->updateAll(s, *c_iter, values);
      }
      pV.clear();
    }
    delete [] dofs;
    delete [] values;
    delete [] v0;
    delete [] J;
  }
  if (debug > 1) {s->view("");}
}


#undef __FUNCT__
#define __FUNCT__ "CreateExactSolution_UFC"
 PetscErrorCode CreateExactSolution_UFC(Obj<PETSC_MESH_TYPE> m, Obj<PETSC_MESH_TYPE::real_section_type> s, ufc::form * form, PetscScalar (*exactSolution)(const double *))
 {
   const int      dim = m->getDimension();
   //PetscBool      flag;
   //PetscErrorCode ierr;
   
   PetscFunctionBegin;
   SetupField_UFC(m, s, form);
   ufc::finite_element * finite_element = form->create_finite_element(0);
   ufc::cell cell;
   cell.geometric_dimension = dim;
   cell.topological_dimension = dim;
   cell.entity_indices = new unsigned int *[dim+1];
   cell.entity_indices[0] = new unsigned int[dim];
   cell.coordinates = new double *[dim+1];
   double * tmpcellcoords = new double[dim*(dim+1)];
   for (int i = 0; i < dim+1; i++) {
    cell.coordinates[i] = &tmpcellcoords[i*dim];
  }
   const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = m->heightStratum(0);
   //const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = m->getRealSection("coordinates");
   const int                                 localDof    = m->sizeWithBC(s, *cells->begin());
   PETSC_MESH_TYPE::real_section_type::value_type *values      = new PETSC_MESH_TYPE::real_section_type::value_type[localDof];
   function_wrapper_scalar sf;
   sf.setFunction(exactSolution);
   for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
     const Obj<PETSC_MESH_TYPE::coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(m, *c_iter);
     const PETSC_MESH_TYPE::coneArray::iterator end     = closure->end();
     int                                  v       = 0;
     
     //m->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
     Map_SieveCell_UFCCell(m, *c_iter, form, &cell);
     for(PETSC_MESH_TYPE::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
       const int pointDim = s->getFiberDimension(*cl_iter);
       //FOR NOW: keep this, only get rid of the integration routine.
       if (pointDim) {
	 for(int d = 0; d < pointDim; ++d, ++v) {
	   values[v] = finite_element->evaluate_dof(v, sf, cell);
	   //values[v] = (*options->integrate)(v0, J, v, options->exactFunc);
	 }
       }
     }
     m->updateAll(s, *c_iter, values);
   }
   s->view("setup field");
   PetscFunctionReturn(0);
 }

