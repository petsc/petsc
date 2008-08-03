/*
  
  Framework for solving a FEM problem using sieve.
  
  The Discretization objects are WAY too embedded into the way things are done; we will have to create ways of 

This includes derived types doing what indicesExcluded does for all things marked with a boundary marker.

*/

#include <Mesh.hh>

namespace ALE { 
  namespace Problem {

    /*
      This class defines a basic interface to a subproblem; all data the type needs will be set at initialization and be members of the derived types

      The recommended use for a given problem is to define a subproblem class for doing the data initialization for the form, as well as the postprocessing
      one for doing the boundary handling and setting, and one for dealing with the process of the solve.  Work can be broken up as needed; however for the 
      sake of simplicity one should probably have various forms in various subproblems to assemble.  This could of course include facet integrals or problems
      over part of the domain.  Look at the examples provided in UFCProblem, which use the UFC form compiler to assemble subproblems of a form.

     */

    class SubProblem : public ParallelObject {
    public:

      typedef std::string name_type;      

      SubProblem(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
      virtual ~SubProblem() {};

    };

    /*
      Below is a helper derived class of subproblem; the persistently limiting and annoying discretization object in Sieve must be excised;
      Here are subproblem objects containing the helper functions that are presently in the monster called PETSC_MESH_TYPE, but with references to the 
      discretization object stolen.  virtual functions to be filled in in further derived classes are marked.  This is meant to be a stepping stone
      towards generalized use of problem creation objects with multiple fields and interesting boundary conditions.  This type can be branched into

    */

    //creates a discretization-like thing for this particular case; from this we can use the helper functions in the generalformsubproblem to set up the discretization
    //across the mesh as one might expect in the case of a 

    class GeneralBoundaryCondition : ParallelObject {
    protected:

      std::string     _labelName;
      int             _marker;

    public:
      GeneralBoundaryCondition(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
      virtual ~GeneralBoundaryCondition() {};
    public:
      virtual const std::string& getLabelName() const {return this->_labelName;};
      virtual void setLabelName(const std::string& name) {this->_labelName = name;};
      virtual int getMarker() const {return this->_marker;};
      virtual void setMarker(const int marker) {this->_marker = marker;};
      
    public:

      virtual double integrateDual(unsigned int dof) const {
	//when implementing this one should have some notion of what the dof number is built into the BC; this would constitute having some cell or something known
	//by the object such that that cell can be set and integrated within.
	throw Exception("GeneralBoundaryCondition->integrateDual: Nonimplemented base-class version called.");
	return 3.;
      };
    };
    

    //we almost, ALMOST need an overall view of a local reference topology for this kind of stuff (and the boundary conditions).

    class GeneralIntegral : public ParallelObject {
    public:
      typedef std::string name_type;
    private: 
      //the integral should only apply to the labeled points in some way, probably over the closure(support(point));


      //the integrals will have some set of cells or cell-like objects they act on; this includes potentially internal faces.
      //But, for cell integrals these should be set to "height" and "0"

      name_type _label_name;
      int _label_marker;     //We could use this if there's only a certain label and marker that the given integral applies to

      int _num_coefficients; //for the linear forms

      int _space_dimension;       //the number of DoFs we're dealing with here.
      int _tensor_rank;           //the tensor rank; it BETTER be one or two.
      int _topological_dimension; //the topological dimension of the given mesh item -- tells us if it's a cell or facet integral

    public:
      
      GeneralIntegral (MPI_Comm comm, int debug = 0) : ParallelObject(comm, debug) {
	_tensor_rank = 0;
	_space_dimension = 0;
	_topological_dimension = 0;
	_label_marker = 0;
      }

      GeneralIntegral (MPI_Comm comm, name_type labelname, int labelmarker, int nCoefficients = 0, int debug = 0) : ParallelObject(comm, debug) {
	_num_coefficients = nCoefficients;
	_label_name = labelname;
	_label_marker = labelmarker;
      }

      virtual ~GeneralIntegral () {};

            
      int getNumCoefficients() {
	return _num_coefficients;
      }

      void setNumCoefficients(int nc) {
	_num_coefficients = nc;
      }


      name_type getLabelName() {
	return _label_name;
      }

      int getLabelMarker() {
	return _label_marker;
      }

      void setLabelName(name_type newName) {
	_label_name = newName;
      }

      void setLabelMarker(int newMarker) {
	_label_marker = newMarker;
      }
      

      virtual void setSpaceDimension(int dim) {
	_space_dimension = dim;
      }
      
      virtual int getSpaceDimension() {
	return _space_dimension;
      }

      virtual void setTensorRank(int rank) {
	_tensor_rank = rank;
      }
      
      virtual int getTensorRank() {
	return _tensor_rank; 
      }

      //use the UFC lingo here
      //have some notion of the cell initialized and in-state in the eventual implementation.
      virtual void tabulateTensor(double * tensor, const double * coefficients = PETSC_NULL) {
	throw Exception("GeneralIntegral->tabulateTensor: Nonimplemented Base class version called.");
	return;
      }
    };

    class GeneralDiscretization : ParallelObject { //this should largely resemble the old form of the discretizations, only with specifics of evaluation to FIAT removed.
      typedef std::map<std::string, Obj<GeneralBoundaryCondition> > boundaryConditions_type;
    protected:
      boundaryConditions_type _boundaryConditions;
      Obj<GeneralBoundaryCondition> _exactSolution;
      std::map<int,int> _dim2dof;
      std::map<int,int> _dim2class;
      int           _quadSize;
      int           _basisSize;
      const int *         _indices;
      std::map<int, const int *> _exclusionIndices;
    public:
      GeneralDiscretization(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _quadSize(0), _basisSize(0), _indices(NULL) {};
      virtual ~GeneralDiscretization() {
	if (this->_indices) {delete [] this->_indices;}
	for(std::map<int, const int *>::iterator i_iter = _exclusionIndices.begin(); i_iter != _exclusionIndices.end(); ++i_iter) {
	  delete [] i_iter->second;
	}
      };
    public:
      virtual const bool hasBoundaryCondition() {return (this->_boundaryConditions.find("default") != this->_boundaryConditions.end());};
      virtual Obj<GeneralBoundaryCondition> getBoundaryCondition() {return this->getBoundaryCondition("default");};
      virtual void setBoundaryCondition(const Obj<GeneralBoundaryCondition>& boundaryCondition) {this->setBoundaryCondition("default", boundaryCondition);};
      virtual Obj<GeneralBoundaryCondition> getBoundaryCondition(const std::string& name) {return this->_boundaryConditions[name];};
      virtual void setBoundaryCondition(const std::string& name, const Obj<GeneralBoundaryCondition>& boundaryCondition) {this->_boundaryConditions[name] = boundaryCondition;};
      virtual Obj<std::set<std::string> > getBoundaryConditions() const {
	Obj<std::set<std::string> > names = std::set<std::string>();
	for(boundaryConditions_type::const_iterator d_iter = this->_boundaryConditions.begin(); d_iter != this->_boundaryConditions.end(); ++d_iter) {
	  names->insert(d_iter->first);
	}
	return names;
      };

      //eh?
      //virtual const Obj<BoundaryCondition>& getExactSolution() {return this->_exactSolution;};
      //virtual void setExactSolution(const Obj<GeneralBoundaryCondition>& exactSolution) {this->_exactSolution = exactSolution;};
      virtual const int     getQuadratureSize() {return this->_quadSize;};
      virtual void          setQuadratureSize(const int size) {this->_quadSize = size;};
      virtual const int     getBasisSize() {return this->_basisSize;};
      virtual void          setBasisSize(const int size) {this->_basisSize = size;};

      //eh, until they get this together in UFC keep these around.

      virtual int           getNumDof(const int dim) {return this->_dim2dof[dim];};
      virtual void          setNumDof(const int dim, const int numDof) {this->_dim2dof[dim] = numDof;};
      virtual int           getDofClass(const int dim) {return this->_dim2class[dim];};
      virtual void          setDofClass(const int dim, const int dofClass) {this->_dim2class[dim] = dofClass;};
    public:

      //Yeah... not messing with this part.

      virtual const int *getIndices() {return this->_indices;};
      virtual const int *getIndices(const int marker) {

	if (!marker) return this->getIndices();
	return this->_exclusionIndices[marker];
      };
      virtual void       setIndices(const int *indices) {this->_indices = indices;};
      virtual void       setIndices(const int *indices, const int marker) {
	if (!marker) this->_indices = indices;
	this->_exclusionIndices[marker] = indices;
      };

      //return the size of the space
      virtual int size() {
	throw Exception("GeneralDiscretization->size(): Nonimplemented base class function called.");
	return 0;
      }
      virtual double evaluateRHS(int dof) {
	throw Exception("GeneralDiscretization->evaluateRHS: Nonimplemented base class function called.");
      }
      
    };

    //The GenericFormSubProblem shown here should basically contain the whole problem as it does the discretization-like setup, which might break
    //if you define multiple ones right now.  Set the multiple different spaces using different discretizations.

    class GenericFormSubProblem : SubProblem { //takes information from some abstract notion of a problem and sets up the thing.
    public:
      typedef std::map<std::string, Obj<GeneralDiscretization> > discretizations_type;
      typedef std::map<std::string, Obj<GeneralIntegral> > integral_type;
      typedef std::set<std::string> names_type;

      GenericFormSubProblem(MPI_Comm comm, int debug = 1) : SubProblem(comm, debug) {};

      ~GenericFormSubProblem(){};

    private:
      
      name_type _solutionSectionName;
      name_type _forcingSectionName;
      discretizations_type _discretizations;
      integral_type _integrals;
      Obj<GeneralBoundaryCondition> _exactSolution; //evaluates a function over all unknowns on the form containing an exact solution.  Per discretization later.

      //helper functions, stolen directly from Mesh.hh with slight modifications to remove FIAT dependence and instead use stuff from GeneralDiscretization

    public: 
      
      const Obj<GeneralDiscretization>& getDiscretization() {return this->getDiscretization("default");};
      const Obj<GeneralDiscretization>& getDiscretization(const std::string& name) {return this->_discretizations[name];};
      
      void setDiscretization(const Obj<GeneralDiscretization>& disc) {this->setDiscretization("default", disc);};
      void setDiscretization(const std::string& name, const Obj<GeneralDiscretization>& disc) {this->_discretizations[name] = disc;};

      const Obj<GeneralIntegral>& getIntegral() {return this->getIntegral("default");};
      const Obj<GeneralIntegral>& getIntegral(const std::string& name) {return this->_integrals[name];};

      void setIntegral(const Obj<GeneralIntegral>& integral) {this->setIntegral("default", integral);};
      void setIntegral(const std::string& name, const Obj<GeneralIntegral> integral) {this->_integrals[name] = integral;};

      //FAIL! this won't be used.
      void setExactSolution(Obj<GeneralBoundaryCondition> exactsol) {
	_exactSolution = exactsol;
      }

      Obj<GeneralBoundaryCondition> getExactSolution(){return _exactSolution;};

      Obj<names_type> getDiscretizations() const {
	Obj<names_type> names = names_type();
	
	for(discretizations_type::const_iterator d_iter = this->_discretizations.begin(); d_iter != this->_discretizations.end(); ++d_iter) {
	  names->insert(d_iter->first);
	}
	return names;
      };

      Obj<names_type> getIntegrals() const {
	Obj<names_type> names = names_type();
	for (integral_type::const_iterator i_iter = this->_integrals.begin(); i_iter != this->_integrals.end(); ++i_iter) {
	  names->insert(i_iter->first);
	}
	return names;
      }

      //TODO: Implement a cell interface class towards flexible handling of reordering between various parts of this thing.

      virtual void setCell(Obj<PETSC_MESH_TYPE> mesh, PETSC_MESH_TYPE::point_type c) const { //implementations should set some state of the derived type to the present cell
	throw Exception("Using the unimplemented base class version of setCell in GeneralDiscretization.");
	return;
      };

      virtual int localSpaceDimension() {
	throw Exception("GeneralDiscretization->localSpaceDimension(): using the unimplemented base class version.");
	return 0;
      }
      

      //TODO: Set up different discretization for the 

      int setFiberDimensions(const Obj<PETSC_MESH_TYPE> mesh, const Obj<PETSC_MESH_TYPE::real_section_type> s, const Obj<names_type>& discs, names_type& bcLabels) {
	const int debug  = this->debug();
	int       maxDof = 0;
	
	for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
	  s->addSpace();
	}
	for(int d = 0; d <= mesh->getDimension(); ++d) {
	  int numDof = 0;
	  int f      = 0;
	  
	  for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	    const Obj<GeneralDiscretization>& disc = this->getDiscretization(*f_iter);
	    const int                       sDof = disc->getNumDof(d);
	    
	    numDof += sDof;
	    if (sDof) s->setFiberDimension(mesh->depthStratum(d), sDof, f);
	  }
	  if (numDof) s->setFiberDimension(mesh->depthStratum(d), numDof);
	  maxDof = std::max(maxDof, numDof);
	}
	// Process exclusions
	typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
	int f = 0;
	
	for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	  const Obj<GeneralDiscretization>& disc      = this->getDiscretization(*f_iter);
	  std::string                     labelName = "exclude-"+*f_iter;
	  std::set<PETSC_MESH_TYPE::point_type>            seen;
	  Visitor pV((int) pow(mesh->getSieve()->getMaxConeSize(), mesh->depth()), true);
	  
	  if (mesh->hasLabel(labelName)) {
	    const Obj<PETSC_MESH_TYPE::label_type>&         label     = mesh->getLabel(labelName);
	    const Obj<PETSC_MESH_TYPE::label_sequence>&     exclusion = mesh->getLabelStratum(labelName, 1);
	    const PETSC_MESH_TYPE::label_sequence::iterator end       = exclusion->end();
	    if (debug > 1) {label->view(labelName.c_str());}
	    
	    for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
	      ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), *e_iter, pV);
	      const Visitor::point_type *oPoints = pV.getPoints();
	      const int                  oSize   = pV.getSize();
	      
	      for(int cl = 0; cl < oSize; ++cl) {
		if (seen.find(oPoints[cl]) != seen.end()) continue;
		if (mesh->getValue(label, oPoints[cl]) == 1) {
		  seen.insert(oPoints[cl]);
		  s->setFiberDimension(oPoints[cl], 0, f);
		  s->addFiberDimension(oPoints[cl], -disc->getNumDof(mesh->depth(oPoints[cl])));
		  if (debug > 1) {std::cout << "  point: " << oPoints[cl] << " dim: " << disc->getNumDof(mesh->depth(oPoints[cl])) << std::endl;}
		}
	      }
	      pV.clear();
	    }
	  }
	}
	// Process constraints
	f = 0;
	for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
	  const Obj<GeneralDiscretization>&    disc        = this->getDiscretization(*f_iter);
	  const Obj<std::set<std::string> >  bcs         = disc->getBoundaryConditions();
	  std::string                        excludeName = "exclude-"+*f_iter;
	  
	  for(std::set<std::string>::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
	    const Obj<GeneralBoundaryCondition>& bc       = disc->getBoundaryCondition(*bc_iter);
	    const Obj<PETSC_MESH_TYPE::label_sequence>&         boundary = mesh->getLabelStratum(bc->getLabelName(), bc->getMarker());
	    
	    bcLabels.insert(bc->getLabelName());
	    if (mesh->hasLabel(excludeName)) {
	      const Obj<PETSC_MESH_TYPE::label_type>& label = mesh->getLabel(excludeName);
	      
	      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
		if (!mesh->getValue(label, *e_iter)) {
		  const int numDof = disc->getNumDof(mesh->depth(*e_iter));
		  
		  if (numDof) s->addConstraintDimension(*e_iter, numDof);
		  if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
		}
	      }
	    } else {
	      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
		const int numDof = disc->getNumDof(mesh->depth(*e_iter));
		
		if (numDof) s->addConstraintDimension(*e_iter, numDof);
		if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
	      }
	    }
	  }
	}
	return maxDof;
      };
      void calculateIndices(Obj<PETSC_MESH_TYPE> mesh) {
	typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
	// Should have an iterator over the whole tree
	Obj<names_type> discs = this->getDiscretizations();
	const int       debug = this->debug();
	std::map<std::string, std::pair<int, int*> > indices;
	
	for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
	  const Obj<GeneralDiscretization>& disc = this->getDiscretization(*d_iter);
	  indices[*d_iter] = std::pair<int, int*>(0, new int[disc->size()]);  //size isn't a function of the mesh it's a function of the local function space
	  disc->setIndices(indices[*d_iter].second);
	}
	const Obj<PETSC_MESH_TYPE::label_sequence>& cells   = mesh->heightStratum(0);
	Visitor pV((int) pow(mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, true);
	ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), *cells->begin(), pV);
	const Visitor::point_type *oPoints = pV.getPoints();
	const int                  oSize   = pV.getSize();
	int                        offset  = 0;
	
	if (debug > 1) {std::cout << "Closure for first element" << std::endl;}
	for(int cl = 0; cl < oSize; ++cl) {
	  const int dim = mesh->depth(oPoints[cl]);
	  
	  if (debug > 1) {std::cout << "  point " << oPoints[cl] << " depth " << dim << std::endl;}
	  for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
	    const Obj<GeneralDiscretization>& disc = this->getDiscretization(*d_iter);
	    const int                  num  = disc->getNumDof(dim);
	    
	    //if (debug > 1) {std::cout << "    disc " << disc->getName() << " numDof " << num << std::endl;}
	    for(int o = 0; o < num; ++o) {
	      indices[*d_iter].second[indices[*d_iter].first++] = offset++;
	    }
	  }
	}
	pV.clear();
	if (debug > 1) {
	  for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
	    //const Obj<GeneralDiscretization>& disc = this->getDiscretization(*d_iter);
	    
	    //std::cout << "Discretization " << disc->getName() << " indices:";
	    for(int i = 0; i < indices[*d_iter].first; ++i) {
	      std::cout << " " << indices[*d_iter].second[i];
	    }
	    std::cout << std::endl;
	  }
	}
      };

      void calculateIndicesExcluded(const Obj<PETSC_MESH_TYPE> mesh, const Obj<PETSC_MESH_TYPE::real_section_type>& s, const Obj<names_type>& discs) {
	typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
	typedef std::map<std::string, std::pair<int, indexSet> > indices_type;
	const Obj<PETSC_MESH_TYPE::label_type>& indexLabel = mesh->createLabel("cellExclusion");
	const int debug  = this->debug();
	int       marker = 0;
	std::map<indices_type, int> indexMap;
	indices_type                indices;
	Visitor pV((int) pow(mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, true);
	
	for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
	  const Obj<GeneralDiscretization>& disc = this->getDiscretization(*d_iter);
	  const int                  size = disc->size();
	  
	  indices[*d_iter].second.resize(size);
	}
	const names_type::const_iterator dBegin = discs->begin();
	const names_type::const_iterator dEnd   = discs->end();
	std::set<PETSC_MESH_TYPE::point_type> seen;
	int f = 0;
	
	for(names_type::const_iterator f_iter = dBegin; f_iter != dEnd; ++f_iter, ++f) {
	  std::string labelName = "exclude-"+*f_iter;
	  
	  if (mesh->hasLabel(labelName)) {
	    const Obj<PETSC_MESH_TYPE::label_sequence>&     exclusion = mesh->getLabelStratum(labelName, 1);
	    const PETSC_MESH_TYPE::label_sequence::iterator end       = exclusion->end();
	    
	    if (debug > 1) {std::cout << "Processing exclusion " << labelName << std::endl;}
	    for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
	      if (mesh->height(*e_iter)) continue;
	      ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), *e_iter, pV);
	      const Visitor::point_type *oPoints = pV.getPoints();
	      const int                  oSize   = pV.getSize();
	      int                        offset  = 0;
	      
	      if (debug > 1) {std::cout << "  Closure for cell " << *e_iter << std::endl;}
	      for(int cl = 0; cl < oSize; ++cl) {
		int g = 0;
		
		if (debug > 1) {std::cout << "    point " << oPoints[cl] << std::endl;}
		for(names_type::const_iterator g_iter = dBegin; g_iter != dEnd; ++g_iter, ++g) {
		  const int fDim = s->getFiberDimension(oPoints[cl], g);
		  
		  if (debug > 1) {std::cout << "      disc " << *g_iter << " numDof " << fDim << std::endl;}
		  for(int d = 0; d < fDim; ++d) {
		    indices[*g_iter].second[indices[*g_iter].first++] = offset++;
		  }
		}
	      }
	      pV.clear();
	      const std::map<indices_type, int>::iterator entry = indexMap.find(indices);
	      
	      if (debug > 1) {
		for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
		  for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
		    std::cout << "Discretization (" << i_iter->second << ") " << *g_iter << " indices:";
		    for(int i = 0; i < ((indices_type) i_iter->first)[*g_iter].first; ++i) {
		      std::cout << " " << ((indices_type) i_iter->first)[*g_iter].second[i];
		    }
		    std::cout << std::endl;
		  }
		  std::cout << "Comparison: " << (indices == i_iter->first) << std::endl;
		}
		for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
		  std::cout << "Discretization " << *g_iter << " indices:";
		  for(int i = 0; i < indices[*g_iter].first; ++i) {
		    std::cout << " " << indices[*g_iter].second[i];
		  }
		  std::cout << std::endl;
		}
	      }
	      if (entry != indexMap.end()) {
		mesh->setValue(indexLabel, *e_iter, entry->second);
		if (debug > 1) {std::cout << "  Found existing indices with marker " << entry->second << std::endl;}
	      } else {
		indexMap[indices] = ++marker;
		mesh->setValue(indexLabel, *e_iter, marker);
		if (debug > 1) {std::cout << "  Created new indices with marker " << marker << std::endl;}
	      }
	      for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
		indices[*g_iter].first  = 0;
		for(unsigned int i = 0; i < indices[*g_iter].second.size(); ++i) indices[*g_iter].second[i] = 0;
	      }
	    }
	  }
	}
	if (debug > 1) {indexLabel->view("cellExclusion");}
	for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
	  if (debug > 1) {std::cout << "Setting indices for marker " << i_iter->second << std::endl;}
	  for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
	    const Obj<GeneralDiscretization>& disc = this->getDiscretization(*g_iter);
	    const indexSet  indSet   = ((indices_type) i_iter->first)[*g_iter].second;
	    const int       size     = indSet.size();
	    int            *_indices = new int[size];
	    
	    if (debug > 1) {std::cout << "  field " << *g_iter << std::endl;}
	    for(int i = 0; i < size; ++i) {
	      _indices[i] = indSet[i];
	      if (debug > 1) {std::cout << "    indices["<<i<<"] = " << _indices[i] << std::endl;}
	    }
	    disc->setIndices(_indices, i_iter->second);
	  }
	}
      };
    public:
      void setupField(const Obj<PETSC_MESH_TYPE> mesh, const Obj<PETSC_MESH_TYPE::real_section_type>& s, const int cellMarker = 2, const bool noUpdate = false, const bool setAll = false) {
	typedef ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Visitor;
	const Obj<names_type>& discs  = this->getDiscretizations();
	const int              debug  = s->debug();
	names_type             bcLabels;
	
	s->setChart(mesh->getSieve()->getChart());
	int maxDof = this->setFiberDimensions(mesh, s, discs, bcLabels);
	this->calculateIndices(mesh);
	this->calculateIndicesExcluded(mesh, s, discs);
	mesh->allocate(s);
	s->defaultConstraintDof();
	const Obj<PETSC_MESH_TYPE::label_type>& cellExclusion = mesh->getLabel("cellExclusion");
	
	if (debug > 1) {std::cout << "Setting boundary values" << std::endl;}
	for(names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
	  Obj<PETSC_MESH_TYPE::label_sequence> boundaryCells = mesh->getLabelStratum(*n_iter, cellMarker);
	  if (setAll) boundaryCells = mesh->heightStratum(0);
	  //const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates   = mesh->getRealSection("coordinates");
	  const Obj<names_type>&         discs         = this->getDiscretizations();
	  const PETSC_MESH_TYPE::point_type               firstCell     = *boundaryCells->begin();
	  const int                      numFields     = discs->size();
	  PETSC_MESH_TYPE::real_section_type::value_type *values        = new PETSC_MESH_TYPE::real_section_type::value_type[mesh->sizeWithBC(s, firstCell)];
	  int                           *dofs          = new int[maxDof];
	  int                           *v             = new int[numFields];
	  Visitor pV((int) pow(mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, true);
	  
	  for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
	    ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), *c_iter, pV);
	    const Visitor::point_type *oPoints = pV.getPoints();
	    const int                  oSize   = pV.getSize();
	    
	    if (debug > 1) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
	    //mesh->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
	    this->setCell(mesh, *c_iter);
	    for(int f = 0; f < numFields; ++f) v[f] = 0;
	    for(int cl = 0; cl < oSize; ++cl) {
	      const int cDim = s->getConstraintDimension(oPoints[cl]);
	      int       off  = 0;
	      int       f    = 0;
	      int       i    = -1;
	      
	      if (debug > 1) {std::cout << "    point " << oPoints[cl] << std::endl;}
	      if (cDim || setAll) {
		if (debug > 1) {std::cout << "      constrained excMarker: " << mesh->getValue(cellExclusion, *c_iter) << std::endl;}
		for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
		  const Obj<GeneralDiscretization>& disc    = this->getDiscretization(*f_iter);
		  const Obj<names_type>           bcs     = disc->getBoundaryConditions();
		  const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
		  const int                      *indices = disc->getIndices(mesh->getValue(cellExclusion, *c_iter));
		  int                             b       = 0;
		  
		  for(names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter, ++b) {
		    const Obj<GeneralBoundaryCondition>& bc    = disc->getBoundaryCondition(*bc_iter);
		    const int                          value = mesh->getValue(mesh->getLabel(bc->getLabelName()), oPoints[cl]);
		    if (b > 0) v[f] -= fDim;
		    if (value == bc->getMarker()) {
		      if (debug > 1) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
		      for(int d = 0; d < fDim; ++d, ++v[f]) {
			dofs[++i] = off+d;
			//if (!noUpdate) values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
			if (!noUpdate) {
			  values[indices[v[f]]] = bc->integrateDual(v[f]);
			}
			if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
		      }
		      // Allow only one condition per point
		      ++b;
		      break;
		    } else {
		      if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
		      for(int d = 0; d < fDim; ++d, ++v[f]) {
			if (!setAll) {
			  values[indices[v[f]]] = 0.0;
			} else {
			  //TODO: do strides of the rank of the unknown space; we need a natural way of handling vectors in a single discretization.
			  values[indices[v[f]]] = bc->integrateDual(v[f]);
			}
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
		for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
		  const Obj<GeneralDiscretization>& disc    = this->getDiscretization(*f_iter);
		  const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
		  const int                      *indices = disc->getIndices(mesh->getValue(cellExclusion, *c_iter));
		  
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
		for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
		  const Obj<GeneralDiscretization>& disc    = this->getDiscretization(*f_iter);
		  const int                       fDim    = s->getFiberDimension(oPoints[cl], f);
		  const int                      *indices = disc->getIndices(mesh->getValue(cellExclusion, *c_iter));
		  
		  for(int d = 0; d < fDim; ++d, ++v[f]) {
		    std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
		  }
		}
	      }
	    }
	    if (!noUpdate) {
	      mesh->updateAll(s, *c_iter, values);
	    }
	    pV.clear();
	  }
	  delete [] dofs;
	  delete [] values;
	}
	if (debug > 1) {s->view("");}
      };
    };
    /*

    Jacobian and RHS assembly functions to pass to DMMG which depend on a GenericFormSubProblem rather than on some options.
    

    */

#undef __FUNCT__
#define __FUNCT__ "RHS_FEMProblem"

    PetscErrorCode RHS_FEMProblem(::Mesh mesh, SectionReal X, SectionReal section,  void * ctx) {
      GenericFormSubProblem * subproblem = (GenericFormSubProblem *)ctx;
      Obj<PETSC_MESH_TYPE> m;
      PetscErrorCode ierr;
      
      PetscFunctionBegin;
      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      ierr = SectionRealZero(section);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE::real_section_type> s;
      ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
      // Loop over cells
      //loop over integrals;
      
      GenericFormSubProblem::names_type integral_names = subproblem->getIntegrals();
      GenericFormSubProblem::names_type::const_iterator n_iter = integral_names.begin();
      GenericFormSubProblem::names_type::const_iterator n_iter_end = integral_names.end();
      //const Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", s);
      while (n_iter != n_iter_end) {
	Obj<GeneralIntegral> cur_integral = subproblem->getIntegral(*n_iter);
	//get the integral's topological objects.
	std::string cur_marker_name = cur_integral->getLabelName(); 
	int cur_marker_num = cur_integral->getLabelMarker();
	int cur_rank = cur_integral->getTensorRank();
	int cur_dimension = cur_integral->getSpaceDimension();
	Obj<PETSC_MESH_TYPE::label_sequence> integral_cells = m->getLabelStratum(cur_marker_name, cur_marker_num);
	PETSC_MESH_TYPE::label_sequence::iterator ic_iter = integral_cells->begin();
	PETSC_MESH_TYPE::label_sequence::iterator ic_iter_end = integral_cells->end();
	if (cur_rank == 1) {
	  PetscScalar * values;
	  PetscMalloc(cur_dimension*sizeof(PetscScalar), &values);
	  PetscScalar * elemVec;	  
	  PetscMalloc(cur_dimension*sizeof(PetscScalar), &elemVec);
	  Obj<GenericFormSubProblem::names_type> discs = subproblem->getDiscretizations();
	  //loop over cells
	  while (ic_iter != ic_iter_end) {
	    subproblem->setCell(m, *ic_iter);
	    //loop over discretizations
	    GenericFormSubProblem::names_type::iterator d_iter = discs->begin();
	    GenericFormSubProblem::names_type::iterator d_iter_end = discs->end();
	    while (d_iter != d_iter_end) {
	      Obj<GeneralDiscretization> disc = subproblem->getDiscretization(*d_iter);
	      //evaluate the RHS at each index point
	      for (int i = 0; i < disc->size(); i++) {
		values[disc->getIndices()[i]] = -1.*disc->evaluateRHS(i);
	      }
	      d_iter++;
	    }
	    cur_integral->tabulateTensor(elemVec, values);
	    ierr = SectionRealUpdateAdd(section, *ic_iter, elemVec);CHKERRQ(ierr);
	    ic_iter++;
	  }
	  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
	} else if (cur_rank == 2) {
	  //cancel out BCs if necessary... this doesn't require any knowledge of the discretization form (if handled right).
	  PetscScalar * full_tensor;
	  PetscMalloc(cur_dimension*cur_dimension*sizeof(PetscScalar), &full_tensor);
	  PetscScalar * elemVec;	  
	  PetscMalloc(cur_dimension*sizeof(PetscScalar), &elemVec);
	  while (ic_iter != ic_iter_end) {
	    subproblem->setCell(m, *ic_iter);
	    cur_integral->tabulateTensor(full_tensor);
	    //create the linear contribution from the BCs
	    PetscScalar * xValues;
	    ierr = SectionRealRestrict(X, *ic_iter, &xValues);CHKERRQ(ierr); //get the coefficients -- BUG: setCell restricts as well; static
	    for(int f = 0; f < cur_dimension; f++) {
	      elemVec[f] = 0.;
	      for(int g = 0; g < cur_dimension; g++) {
		elemVec[f] += full_tensor[f*cur_dimension+g]*xValues[g];
	      }
	    } 
	    ierr = SectionRealUpdateAdd(section, *ic_iter, elemVec);CHKERRQ(ierr);
	    ic_iter++;
	  }
	  //delete full_tensor;
	  //delete elemVec;
	} else {
	  throw Exception("RHS_FEMProblem: Unsupported tensor rank");
	}
	n_iter++;
	ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
      }
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

#undef __FUNCT__
#define __FUNCT__ "Jac_FEMProblem"

    PetscErrorCode Jac_FEMProblem(::Mesh mesh, SectionReal section, Mat A, void * ctx) {
      PetscFunctionBegin;
      GenericFormSubProblem * subproblem = (GenericFormSubProblem *)ctx;
      Obj<PETSC_MESH_TYPE::real_section_type> s;
      Obj<PETSC_MESH_TYPE> m;
      PetscErrorCode ierr;
      ierr = MatZeroEntries(A);CHKERRQ(ierr);
      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
      //loop over integrals; for now.
      GenericFormSubProblem::names_type integral_names = subproblem->getIntegrals();
      GenericFormSubProblem::names_type::iterator n_iter = integral_names.begin();
      GenericFormSubProblem::names_type::iterator n_iter_end = integral_names.end();
      const Obj<PETSC_MESH_TYPE::order_type>&        order         = m->getFactory()->getGlobalOrder(m, "default", s);
      while (n_iter != n_iter_end) {
	Obj<GeneralIntegral> cur_integral = subproblem->getIntegral(*n_iter);
	//get the integral's topological objects.
	std::string cur_marker_name = cur_integral->getLabelName(); 
	int cur_marker_num = cur_integral->getLabelMarker();
	int cur_rank = cur_integral->getTensorRank();
	int cur_dimension = cur_integral->getSpaceDimension();
	//GOOD LORD >:(
	m->setMaxDof(subproblem->localSpaceDimension());

	//divide here; ignore integral ranks that aren't 2 in the jacobian construction.
	if (cur_rank == 2) {
	  Obj<PETSC_MESH_TYPE::label_sequence> integral_cells = m->getLabelStratum(cur_marker_name, cur_marker_num);
	  PETSC_MESH_TYPE::label_sequence::iterator ic_iter = integral_cells->begin();
	  PETSC_MESH_TYPE::label_sequence::iterator ic_iter_end = integral_cells->end();
	  PetscScalar * tensor;
	  PetscMalloc(cur_dimension*cur_dimension*sizeof(PetscScalar), &tensor);
	  while (ic_iter != ic_iter_end) {
	    subproblem->setCell(m, *ic_iter);
	    cur_integral->tabulateTensor(tensor);
	    ierr = updateOperator(A, m, s, order, *ic_iter, tensor, ADD_VALUES);CHKERRQ(ierr);
	    ic_iter++;
	  }
	}
	n_iter++;
      }
      ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatView(A, PETSC_VIEWER_STDOUT_SELF);
      PetscFunctionReturn(0);	
    }
    
  }  //namespace Problem
} //namespace ALE
