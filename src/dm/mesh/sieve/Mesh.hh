#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif
#ifndef  included_ALE_ParDelta_hh
#include <ParDelta.hh>
#endif
#ifndef  included_ALE_Partitioner_hh
#include <Partitioner.hh>
#endif

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>
#endif
#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#endif

#include <petscvec.h>

namespace ALE {
    // Forward declaration
    class Partitioner;

    class Mesh {
    public:
      typedef ALE::Point point_type;
      typedef std::vector<point_type> PointArray;
      typedef ALE::Sieve<point_type,int,int> sieve_type;
      typedef point_type patch_type;
      typedef CoSifter<sieve_type, patch_type, point_type, int> bundle_type;
      typedef CoSifter<sieve_type, patch_type, point_type, double> field_type;
      typedef CoSifter<sieve_type, ALE::pair<patch_type,int>, point_type, double> foliation_type;
      typedef std::map<std::string, Obj<field_type> > FieldContainer;
      typedef std::map<int, Obj<bundle_type> > BundleContainer;
      int debug;
    private:
      Obj<sieve_type> topology;
      Obj<field_type> coordinates;
      Obj<field_type> boundary;
      Obj<foliation_type> boundaries;
      FieldContainer  fields;
      BundleContainer bundles;
      MPI_Comm        _comm;
      int             _commRank;
      int             _commSize;
      int             dim;
      //FIX:
    public:
      bool            distributed;
    public:
      Mesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
        this->setComm(comm);
        this->topology    = sieve_type(comm, debug);
        this->coordinates = field_type(comm, debug);
        this->boundary    = field_type(comm, debug);
        this->boundaries  = foliation_type(comm, debug);
        this->distributed = false;
        this->coordinates->setTopology(this->topology);
        this->boundary->setTopology(this->topology);
        this->boundaries->setTopology(this->topology);
      };

      MPI_Comm        comm() const {return this->_comm;};
      void            setComm(MPI_Comm comm) {this->_comm = comm; MPI_Comm_rank(comm, &this->_commRank); MPI_Comm_size(comm, &this->_commSize);};
      int             commRank() const {return this->_commRank;};
      int             commSize() const {return this->_commSize;};
      Obj<sieve_type> getTopology() const {return this->topology;};
      void            setTopology(const Obj<sieve_type>& topology) {this->topology = topology;};
      int             getDimension() const {return this->dim;};
      void            setDimension(int dim) {this->dim = dim;};
      Obj<field_type> getCoordinates() const {return this->coordinates;};
      void            setCoordinates(const Obj<field_type>& coordinates) {this->coordinates = coordinates;};
      Obj<field_type> getBoundary() const {return this->boundary;};
      void            setBoundary(const Obj<field_type>& boundary) {this->boundary = boundary;};
      Obj<foliation_type> getBoundaries() const {return this->boundaries;};
      #undef __FUNCT__
      #define __FUNCT__ "Mesh::getBundle"
      Obj<bundle_type> getBundle(const int dim) {
        ALE_LOG_EVENT_BEGIN;
        if (this->bundles.find(dim) == this->bundles.end()) {
          Obj<bundle_type> bundle = bundle_type(this->comm(), debug);

          // Need to globalize indices (that is what we might use the value ints for)
          std::cout << "Creating new bundle for dim " << dim << std::endl;
          bundle->setTopology(this->topology);
          bundle->setPatch(this->topology->leaves(), bundle_type::patch_type());
          bundle->setFiberDimensionByDepth(bundle_type::patch_type(), dim, 1);
          bundle->orderPatches();
          if (this->distributed) {
            bundle->createGlobalOrder();
          }
          // "element" reorder is in vertexBundle by default, and intermediate bundles could be handled by a cell tuple
          this->bundles[dim] = bundle;
        } else {
          if (this->distributed && this->bundles[dim]->getGlobalOffsets() == NULL) {
            this->bundles[dim]->createGlobalOrder();
          }
        }
        ALE_LOG_EVENT_END;
        return this->bundles[dim];
      };
      Obj<field_type> getField(const std::string& name) {
        if (this->fields.find(name) == this->fields.end()) {
          Obj<field_type> field = field_type(this->comm(), debug);

          std::cout << "Creating new field " << name << std::endl;
          field->setTopology(this->topology);
          this->fields[name] = field;
        }
        return this->fields[name];
      };
      bool hasField(const std::string& name) {
        return(this->fields.find(name) != this->fields.end());
      };
      Obj<std::set<std::string> > getFields() {
        Obj<std::set<std::string> > names = std::set<std::string>();

        for(FieldContainer::iterator f_iter = this->fields.begin(); f_iter != this->fields.end(); ++f_iter) {
          names->insert(f_iter->first);
        }
        return names;
      }
      // For a hex, there are 2d faces
      void buildHexFaces(int dim, std::map<int, int*> *curSimplex, PointArray *boundary, point_type& simplex) {
        PointArray *faces = NULL;

        if (debug > 1) {std::cout << "  Building hex faces for boundary of " << simplex << " (size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 3) {
          throw ALE::Exception("Cannot do hexes of dimension greater than three");
        } else if (dim > 2) {
          int nodes[24] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                           1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7};
          faces = new PointArray();

          for(int b = 0; b < 6; b++) {
            PointArray faceBoundary = PointArray();
            point_type face;

            for(int c = 0; c < 4; c++) {
              faceBoundary.push_back((*boundary)[nodes[b*4+c]]);
            }
            if (debug > 1) {std::cout << "    boundary point " << (*boundary)[b] << std::endl;}
            this->buildHexFaces(dim-1, curSimplex, &faceBoundary, face);
            faces->push_back(face);
          }
        } else if (dim > 1) {
          int boundarySize = (int) boundary->size();
          faces = new PointArray();

          for(int b = 0; b < boundarySize; b++) {
            PointArray faceBoundary = PointArray();
            point_type face;

            for(int c = 0; c < 2; c++) {
              faceBoundary.push_back((*boundary)[(b+c)%boundarySize]);
            }
            if (debug > 1) {std::cout << "    boundary point " << (*boundary)[b] << std::endl;}
            this->buildHexFaces(dim-1, curSimplex, &faceBoundary, face);
            faces->push_back(face);
          }
        } else {
          if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
          faces = boundary;
        }
        if (debug > 1) {
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            std::cout << "  face point " << *f_itor << std::endl;
          }
        }
        // We always create the toplevel, so we could short circuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        PointArray::iterator f_itor = faces->begin();
        point_type           start = *f_itor;
        f_itor++;
        point_type           next = *f_itor;
        Obj<sieve_type::supportSet> preElement = this->topology->nJoin(start, next, 1);

        if (preElement->size() > 0) {
          simplex = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old simplex " << simplex << std::endl;}
        } else {
          int color = 0;

          simplex = point_type(0, (*(*curSimplex)[dim])++);
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            this->topology->addArrow(*f_itor, simplex, color++);
          }
          if (debug > 1) {std::cout << "  Added simplex " << simplex << " dim " << dim << std::endl;}
        }
        if (dim > 1) {
          delete faces;
        }
      };
      void buildFaces(int dim, std::map<int, int*> *curSimplex, PointArray *boundary, point_type& simplex) {
        PointArray *faces = NULL;

        if (debug > 1) {std::cout << "  Building faces for boundary of " << simplex << " (size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 1) {
          faces = new PointArray();

          // Use the cone construction
          for(PointArray::iterator b_itor = boundary->begin(); b_itor != boundary->end(); ++b_itor) {
            PointArray faceBoundary = PointArray();
            point_type face;

            for(PointArray::iterator i_itor = boundary->begin(); i_itor != boundary->end(); ++i_itor) {
              if (i_itor != b_itor) {
                faceBoundary.push_back(*i_itor);
              }
            }
            if (debug > 1) {std::cout << "    boundary point " << *b_itor << std::endl;}
            this->buildFaces(dim-1, curSimplex, &faceBoundary, face);
            faces->push_back(face);
          }
        } else {
          if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
          faces = boundary;
        }
        if (debug > 1) {
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            std::cout << "  face point " << *f_itor << std::endl;
          }
        }
        // We always create the toplevel, so we could short circuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        PointArray::iterator f_itor = faces->begin();
        point_type           start = *f_itor;
        f_itor++;
        point_type           next = *f_itor;
        Obj<sieve_type::supportSet> preElement = this->topology->nJoin(start, next, 1);

        if (preElement->size() > 0) {
          simplex = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old simplex " << simplex << std::endl;}
        } else {
          int color = 0;

          simplex = point_type(0, (*(*curSimplex)[dim])++);
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            this->topology->addArrow(*f_itor, simplex, color++);
          }
          if (debug > 1) {std::cout << "  Added simplex " << simplex << " dim " << dim << std::endl;}
        }
        if (dim > 1) {
          delete faces;
        }
      };

      #undef __FUNCT__
      #define __FUNCT__ "Mesh::buildTopology"
      // Build a topology from a connectivity description
      //   (0, 0)            ... (0, numSimplices-1):  dim-dimensional simplices
      //   (0, numSimplices) ... (0, numVertices):     vertices
      // The other simplices are numbered as they are requested
      void buildTopology(int numSimplices, int simplices[], int numVertices, bool interpolate = true, int corners = -1) {
        ALE_LOG_EVENT_BEGIN;
        if (this->commRank() != 0) {
          ALE_LOG_EVENT_END;
          return;
        }
        // Create a map from dimension to the current element number for that dimension
        std::map<int,int*> curElement = std::map<int,int*>();
        int                curSimplex = 0;
        int                curVertex  = numSimplices;
        int                newElement = numSimplices+numVertices;
        PointArray         boundary   = PointArray();

        if (corners < 0) corners = this->dim+1;
        curElement[0]         = &curVertex;
        curElement[this->dim] = &curSimplex;
        for(int d = 1; d < this->dim; d++) {
          curElement[d] = &newElement;
        }
        for(int s = 0; s < numSimplices; s++) {
          point_type simplex(0, s);

          // Build the simplex
          if (interpolate) {
            boundary.clear();
            for(int b = 0; b < corners; b++) {
              point_type vertex(0, simplices[s*corners+b]+numSimplices);

              if (debug > 1) {std::cout << "Adding boundary node " << vertex << std::endl;}
              boundary.push_back(vertex);
            }
            if (debug) {std::cout << "simplex " << s << " boundary size " << boundary.size() << std::endl;}

            if (corners != this->dim+1) {
              this->buildHexFaces(this->dim, &curElement, &boundary, simplex);
            } else {
              this->buildFaces(this->dim, &curElement, &boundary, simplex);
            }
          } else {
            for(int b = 0; b < corners; b++) {
              point_type p(0, simplices[s*corners+b]+numSimplices);

              this->topology->addArrow(p, simplex, b);
            }
          }
        }
        ALE_LOG_EVENT_END;
      };

      #undef __FUNCT__
      #define __FUNCT__ "Mesh::createVertBnd"
      void createVertexBundle(int numSimplices, int simplices[], int elementOffset = 0, int corners = -1) {
        ALE_LOG_STAGE_BEGIN;
        Obj<bundle_type> vertexBundle = this->getBundle(0);
        Obj<sieve_type::traits::heightSequence> elements = this->topology->heightStratum(0);
        std::string orderName("element");
        int vertexOffset;

        ALE_LOG_EVENT_BEGIN;
        if (elementOffset) {
          vertexOffset = 0;
        } else {
          vertexOffset = numSimplices;
        }
        if (corners < 0) corners = this->dim+1;
        for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          // setFiberDimensionByDepth() does not work here since we only want it to apply to the patch cone
          //   What we really need is the depthStratum relative to the patch
          Obj<PointArray> patch = PointArray();

          for(int b = 0; b < corners; b++) {
            patch->push_back(point_type(0, simplices[((*e_iter).index - elementOffset)*corners+b]+vertexOffset));
          }
          vertexBundle->setPatch(orderName, patch, *e_iter);
          for(PointArray::iterator p_iter = patch->begin(); p_iter != patch->end(); ++p_iter) {
            vertexBundle->setFiberDimension(orderName, *e_iter, *p_iter, 1);
          }
        }
        if (elements->size() == 0) {
          vertexBundle->setPatch(orderName, elements, bundle_type::patch_type());
        }
        ALE_LOG_EVENT_END;
        vertexBundle->orderPatches(orderName);
        ALE_LOG_STAGE_END;
      };

      #undef __FUNCT__
      #define __FUNCT__ "Mesh::createSerCoords"
      void createSerialCoordinates(int embedDim, int numSimplices, double coords[]) {
        ALE_LOG_EVENT_BEGIN;
        patch_type patch;

        this->coordinates->setTopology(this->topology);
        this->coordinates->setPatch(this->topology->leaves(), patch);
        this->coordinates->setFiberDimensionByDepth(patch, 0, embedDim);
        this->coordinates->orderPatches();
        Obj<sieve_type::traits::depthSequence> vertices = this->topology->depthStratum(0);
        for(sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
          this->coordinates->update(patch, *v_itor, &coords[((*v_itor).index - numSimplices)*embedDim]);
        }
        Obj<bundle_type> vertexBundle = this->getBundle(0);
        Obj<sieve_type::traits::heightSequence> elements = this->topology->heightStratum(0);
        std::string orderName("element");

        for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
          // setFiberDimensionByDepth() does not work here since we only want it to apply to the patch cone
          //   What we really need is the depthStratum relative to the patch
          Obj<bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_iter);

          this->coordinates->setPatch(orderName, cone, *e_iter);
          for(bundle_type::order_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            this->coordinates->setFiberDimension(orderName, *e_iter, *c_iter, embedDim);
          }
        }
        if (elements->size() == 0) {
          this->coordinates->setPatch(orderName, elements, field_type::patch_type());
        }
        this->coordinates->orderPatches(orderName);
        ALE_LOG_EVENT_END;
      };
    private:
      template<typename IntervalSequence>
      int *__expandIntervals(Obj<IntervalSequence> intervals) {
        int *indices;
        int  k = 0;

        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          k += std::abs(i_iter.color().index);
        }
        std::cout << "Allocated indices of size " << k << std::endl;
        indices = new int[k];
        k = 0;
        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          for(int i = i_iter.color().prefix; i < i_iter.color().prefix + std::abs(i_iter.color().index); i++) {
            std::cout << "  indices[" << k << "] = " << i << std::endl;
            indices[k++] = i;
          }
        }
        return indices;
      };
      template<typename IntervalSequence,typename Field>
      int *__expandCanonicalIntervals(Obj<IntervalSequence> intervals, Obj<Field> field) {
        typename Field::patch_type patch;
        int *indices;
        int  k = 0;

        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          k += std::abs(field->getFiberDimension(patch, *i_iter));
        }
        std::cout << "Allocated indices of size " << k << std::endl;
        indices = new int[k];
        k = 0;
        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          int dim = field->getFiberDimension(patch, *i_iter);
          int offset = field->getFiberOffset(patch, *i_iter);

          for(int i = offset; i < offset + std::abs(dim); i++) {
            std::cout << "  indices[" << k << "] = " << i << std::endl;
            indices[k++] = i;
          }
        }
        return indices;
      };
    public:
      #undef __FUNCT__
      #define __FUNCT__ "Mesh::parVertReOrd"
      // This is not right, we should not have to copy everything to the new order first
      void createParallelVertexReorder(Obj<bundle_type> serialVertexBundle) {
        ALE_LOG_EVENT_BEGIN;
        Obj<bundle_type> vertexBundle = this->getBundle(0);
        std::string orderName("element");

        if (!this->commRank()) {
          Obj<bundle_type::order_type::baseSequence> patches = serialVertexBundle->getPatches(orderName);

          for(bundle_type::order_type::baseSequence::iterator e_iter = patches->begin(); e_iter != patches->end(); ++e_iter) {
            Obj<bundle_type::order_type::coneSequence> patch = serialVertexBundle->getPatch(orderName, *e_iter);

            vertexBundle->setPatch(orderName, patch, *e_iter);
            for(bundle_type::order_type::coneSequence::iterator p_iter = patch->begin(); p_iter != patch->end(); ++p_iter) {
              vertexBundle->setFiberDimension(orderName, *e_iter, *p_iter, 1);
            }
          }
        } else {
          Obj<sieve_type::traits::heightSequence> elements = this->topology->heightStratum(0);
          Obj<bundle_type::order_type> reorder = vertexBundle->__getOrder(orderName);

          for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
            reorder->addBasePoint(*e_iter);
          }
        }
        vertexBundle->orderPatches(orderName);
        vertexBundle->partitionOrder(orderName);
        ALE_LOG_EVENT_END;
      };
      template<typename OverlapType>
      void createParallelCoordinates(int embedDim, Obj<field_type> serialCoordinates, Obj<OverlapType> partitionOverlap);

      // Create a serial mesh
      void populate(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true, int corners = -1) {
        this->topology->setStratification(false);
        this->buildTopology(numSimplices, simplices, numVertices, interpolate, corners);
        this->topology->stratify();
        this->topology->setStratification(true);
        this->createVertexBundle(numSimplices, simplices, 0, corners);
        this->createSerialCoordinates(this->dim, numSimplices, coords);
      };
      void populateBd(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true, int corners = -1) {
        this->topology->setStratification(false);
        this->buildTopology(numSimplices, simplices, numVertices, interpolate, corners);
        this->topology->stratify();
        this->topology->setStratification(true);
        this->createVertexBundle(numSimplices, simplices, 0, corners);
        this->createSerialCoordinates(this->dim+1, numSimplices, coords);
      };

      // Partition and distribute a serial mesh
      Obj<Mesh> distribute();
      // Collect a distributed mesh on process 0
      Obj<Mesh> unify();
    };

    class Generator {
#ifdef PETSC_HAVE_TRIANGLE
      static void initInput_Triangle(struct triangulateio *inputCtx) {
        inputCtx->numberofpoints = 0;
        inputCtx->numberofpointattributes = 0;
        inputCtx->pointlist = NULL;
        inputCtx->pointattributelist = NULL;
        inputCtx->pointmarkerlist = NULL;
        inputCtx->numberofsegments = 0;
        inputCtx->segmentlist = NULL;
        inputCtx->segmentmarkerlist = NULL;
        inputCtx->numberoftriangleattributes = 0;
        inputCtx->numberofholes = 0;
        inputCtx->holelist = NULL;
        inputCtx->numberofregions = 0;
        inputCtx->regionlist = NULL;
      };
      static void initOutput_Triangle(struct triangulateio *outputCtx) {
        outputCtx->pointlist = NULL;
        outputCtx->pointattributelist = NULL;
        outputCtx->pointmarkerlist = NULL;
        outputCtx->trianglelist = NULL;
        outputCtx->triangleattributelist = NULL;
        outputCtx->neighborlist = NULL;
        outputCtx->segmentlist = NULL;
        outputCtx->segmentmarkerlist = NULL;
        outputCtx->edgelist = NULL;
        outputCtx->edgemarkerlist = NULL;
      };
      static void finiOutput_Triangle(struct triangulateio *outputCtx) {
        free(outputCtx->pointmarkerlist);
        free(outputCtx->edgelist);
        free(outputCtx->edgemarkerlist);
        free(outputCtx->trianglelist);
        free(outputCtx->neighborlist);
      };
      #undef __FUNCT__
      #define __FUNCT__ "generate_Triangle"
      static Obj<Mesh> generate_Triangle(Obj<Mesh> boundary, bool interpolate) {
        struct triangulateio   in;
        struct triangulateio   out;
        int                    dim = 2;
        Obj<Mesh>              m = Mesh(boundary->comm(), dim, boundary->debug);
        Obj<Mesh::sieve_type>  bdTopology = boundary->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = boundary->getBundle(0);
        Obj<Mesh::bundle_type> edgeBundle = boundary->getBundle(1);
        PetscMPIInt            rank;
        PetscErrorCode         ierr;

        ierr = MPI_Comm_rank(boundary->comm(), &rank);
        initInput_Triangle(&in);
        initOutput_Triangle(&out);
        if (rank == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          Obj<Mesh::sieve_type::traits::depthSequence> vertices = bdTopology->depthStratum(0);
          Mesh::field_type::patch_type         patch;

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            Obj<Mesh::field_type> coordinates = boundary->getCoordinates();

            ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
            ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
            for(Mesh::sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
              const Mesh::field_type::index_type& interval = coordinates->getIndex(patch, *v_itor);
              const Mesh::field_type::value_type *array = coordinates->restrict(patch, *v_itor);

              for(int d = 0; d < interval.index; d++) {
                in.pointlist[interval.prefix + d] = array[d];
              }
              const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *v_itor);
              in.pointmarkerlist[vInterval.prefix] = v_itor.marker();
            }
          }

          Obj<Mesh::sieve_type::traits::depthSequence> edges = bdTopology->depthStratum(1);

          in.numberofsegments = edges->size();
          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            for(Mesh::sieve_type::traits::depthSequence::iterator e_itor = edges->begin(); e_itor != edges->end(); e_itor++) {
              const Mesh::field_type::index_type& interval = edgeBundle->getIndex(patch, *e_itor);
              Obj<Mesh::sieve_type::coneSequence> cone = bdTopology->cone(*e_itor);
              int                                 p = 0;
        
              for(Mesh::sieve_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *c_itor);

                in.segmentlist[interval.prefix * 2 + (p++)] = vInterval.prefix;
              }
              in.segmentmarkerlist[interval.prefix] = e_itor.marker();
            }
          }

          in.numberofholes = 0;
          if (in.numberofholes > 0) {
            ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);
          }
          if (createConvexHull) {
            args += "c";
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);

          ierr = PetscFree(in.pointlist);
          ierr = PetscFree(in.pointmarkerlist);
          ierr = PetscFree(in.segmentlist);
          ierr = PetscFree(in.segmentmarkerlist);
        }
        m->populate(out.numberoftriangles, out.trianglelist, out.numberofpoints, out.pointlist, interpolate);

        // Make boundary
        Obj<Mesh::sieve_type> topology = m->getTopology();
        Obj<Mesh::field_type> mBoundary = m->getBoundary();
        Mesh::sieve_type::markerSet markers;

        if (rank == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              markers.insert(out.pointmarkerlist[v]);
            }
          }
          for(int e = 0; e < out.numberofedges; e++) {
            if (out.edgemarkerlist[e]) {
              markers.insert(out.edgemarkerlist[e]);
            }
          }
          mBoundary->setTopology(topology);
          for(Mesh::sieve_type::markerSet::iterator m_iter = markers.begin(); m_iter != markers.end(); ++m_iter) {
            mBoundary->setPatch(topology->leaves(), Mesh::field_type::patch_type(0, *m_iter));
          }
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              mBoundary->setFiberDimension(Mesh::field_type::patch_type(0, out.pointmarkerlist[v]), Mesh::point_type(0, v+out.numberoftriangles), 1);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type vertexA(0, out.edgelist[e*2+0]+out.numberoftriangles);
                Mesh::point_type vertexB(0, out.edgelist[e*2+1]+out.numberoftriangles);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(vertexA, vertexB, 1);

                mBoundary->setFiberDimension(Mesh::field_type::patch_type(0, out.edgemarkerlist[e]), *(join->begin()), 1);
              }
            }
          }
        }
        mBoundary->orderPatches();

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> generate_TetGen(Obj<Mesh> boundary, bool interpolate) {
        ::tetgenio             in;
        ::tetgenio             out;
        int                    dim = 3;
        Obj<Mesh>              m = Mesh(boundary->comm(), dim, boundary->debug);
        Obj<Mesh::sieve_type>  bdTopology = boundary->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = boundary->getBundle(0);
        Obj<Mesh::bundle_type> facetBundle = boundary->getBundle(bdTopology->depth());
        PetscMPIInt            rank;
        PetscErrorCode         ierr;

        ierr = MPI_Comm_rank(boundary->comm(), &rank);

        if (rank == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          Obj<Mesh::sieve_type::traits::depthSequence> vertices = bdTopology->depthStratum(0);
          Mesh::field_type::patch_type         patch;

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            Obj<Mesh::field_type> coordinates = boundary->getCoordinates();

            in.pointlist       = new double[in.numberofpoints*dim];
            in.pointmarkerlist = new int[in.numberofpoints];
            for(Mesh::sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
              const Mesh::field_type::index_type& interval = coordinates->getIndex(patch, *v_itor);
              const Mesh::field_type::value_type *array = coordinates->restrict(patch, *v_itor);

              for(int d = 0; d < interval.index; d++) {
                in.pointlist[interval.prefix + d] = array[d];
              }
              const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *v_itor);
              in.pointmarkerlist[vInterval.prefix] = v_itor.marker();
            }
          }

          Obj<Mesh::sieve_type::traits::heightSequence> facets = bdTopology->heightStratum(0);

          in.numberoffacets = facets->size();
          if (in.numberoffacets > 0) {
            in.facetlist       = new tetgenio::facet[in.numberoffacets];
            in.facetmarkerlist = new int[in.numberoffacets];
            for(Mesh::sieve_type::traits::heightSequence::iterator f_itor = facets->begin(); f_itor != facets->end(); ++f_itor) {
              const Mesh::field_type::index_type& interval = facetBundle->getIndex(patch, *f_itor);
              Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch("element", *f_itor);

              in.facetlist[interval.prefix].numberofpolygons = 1;
              in.facetlist[interval.prefix].polygonlist = new tetgenio::polygon[in.facetlist[interval.prefix].numberofpolygons];
              in.facetlist[interval.prefix].numberofholes = 0;
              in.facetlist[interval.prefix].holelist = NULL;

              tetgenio::polygon *poly = in.facetlist[interval.prefix].polygonlist;
              int                c = 0;

              poly->numberofvertices = cone->size();
              poly->vertexlist = new int[poly->numberofvertices];
              // The "element" reorder should be fused with the structural order
              for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
                const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *c_itor);

                poly->vertexlist[c++] = vInterval.prefix;
              }
              in.facetmarkerlist[interval.prefix] = f_itor.marker();
            }
          }

          in.numberofholes = 0;
          if (createConvexHull) args += "c";
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist, interpolate);
  
        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(0, v + out.numberoftetrahedra), out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            if (out.edgemarkerlist) {
              for(int e = 0; e < out.numberofedges; e++) {
                if (out.edgemarkerlist[e]) {
                  Mesh::point_type endpointA(0, out.edgelist[e*2+0] + out.numberoftetrahedra);
                  Mesh::point_type endpointB(0, out.edgelist[e*2+1] + out.numberoftetrahedra);
                  Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(endpointA, endpointB, 1);

                  topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
                }
              }
            }
            if (out.trifacemarkerlist) {
              for(int f = 0; f < out.numberoftrifaces; f++) {
                if (out.trifacemarkerlist[f]) {
                  Obj<Mesh::sieve_type::supportSet> point = Mesh::sieve_type::supportSet();
                  Obj<Mesh::sieve_type::supportSet> edge = Mesh::sieve_type::supportSet();
                  Mesh::point_type cornerA(0, out.trifacelist[f*3+0] + out.numberoftetrahedra);
                  Mesh::point_type cornerB(0, out.trifacelist[f*3+1] + out.numberoftetrahedra);
                  Mesh::point_type cornerC(0, out.trifacelist[f*3+2] + out.numberoftetrahedra);
                  point->insert(cornerA);
                  edge->insert(cornerB);
                  edge->insert(cornerC);
                  Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(point, edge, 2);

                  topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
                }
              }
            }
          }
        }
        return m;
      };
#endif
    public:
      static Obj<Mesh> generate(Obj<Mesh> boundary, bool interpolate = true) {
        Obj<Mesh> mesh;
        int       dim = boundary->getDimension();

        if (dim == 1) {
#ifdef PETSC_HAVE_TRIANGLE
          mesh = generate_Triangle(boundary, interpolate);
#else
          throw ALE::Exception("Mesh generation currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 2) {
#ifdef PETSC_HAVE_TETGEN
          mesh = generate_TetGen(boundary, interpolate);
#else
          throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
        }
        return mesh;
      };
    private:
#ifdef PETSC_HAVE_TRIANGLE
      static Obj<Mesh> refine_Triangle(Obj<Mesh> serialMesh, const double maxAreas[], bool interpolate) {
        struct triangulateio in;
        struct triangulateio out;
        int                  dim = 2;
        Obj<Mesh>            m = Mesh(serialMesh->comm(), dim, serialMesh->debug);
        PetscInt             numElements = serialMesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt          rank;
        PetscErrorCode       ierr;

        ierr = MPI_Comm_rank(serialMesh->comm(), &rank);
        initInput_Triangle(&in);
        initOutput_Triangle(&out);
        if (rank == 0) {
          ierr = PetscMalloc(numElements * sizeof(double), &in.trianglearealist);
          for(int i = 0; i < numElements; i++) {
            in.trianglearealist[i] = maxAreas[i];
          }
        }

        Obj<Mesh::sieve_type>  serialTopology = serialMesh->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = serialMesh->getBundle(0);

        if (rank == 0) {
          std::string args("pqenzQra");
          Mesh::field_type::patch_type          patch;
          std::string                           orderName("element");
          Obj<Mesh::sieve_type::traits::heightSequence> faces = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::traits::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::field_type>                 coordinates = serialMesh->getCoordinates();
          const double                         *array = coordinates->restrict(patch);
          int                                   f = 0;

          in.numberofpoints = vertices->size();
          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
          for(int v = 0; v < (int) vertices->size(); ++v) {
            for(int d = 0; d < 2; d++) {
              in.pointlist[v*2 + d] = array[v*2 + d];
              in.pointmarkerlist[v] = 0;
            }
          }

          in.numberofcorners = 3;
          in.numberoftriangles = faces->size();
          ierr = PetscMalloc(in.numberoftriangles * in.numberofcorners * sizeof(int), &in.trianglelist);
          for(Mesh::sieve_type::traits::heightSequence::iterator f_itor = faces->begin(); f_itor != faces->end(); f_itor++) {
            Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *f_itor);
            int                                              v = 0;

            for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              in.trianglelist[f * in.numberofcorners + v++] = vertexBundle->getIndex(patch, *c_itor).prefix;
            }
            f++;
          }

          Obj<Mesh::field_type> boundary = serialMesh->getBoundary();
          Obj<Mesh::sieve_type::traits::depthSequence> segments = serialTopology->depthStratum(1);
          Obj<Mesh::field_type::order_type::baseSequence> patches = boundary->getPatches();

          in.numberofsegments = 0;
          for(Mesh::field_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
            for(Mesh::sieve_type::traits::depthSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
              if (boundary->getIndex(*p_iter, *s_itor).index > 0) {
                in.numberofsegments++;
              }
            }
          }

          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            int s = 0;
            for(Mesh::field_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
              for(Mesh::sieve_type::traits::depthSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
                const Mesh::field_type::index_type& interval = boundary->getIndex(*p_iter, *s_itor);

                if (interval.index > 0) {
                  Obj<Mesh::sieve_type::traits::coneSequence> cone = serialTopology->cone(*s_itor);
                  int                                         p    = 0;

                  for(Mesh::sieve_type::traits::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                    int v = vertexBundle->getIndex(patch, *c_itor).prefix;

                    in.segmentlist[s*2 + (p++)] = v;
                    in.pointmarkerlist[v] = (*p_iter).index;
                  }
                  in.segmentmarkerlist[s++] = (*p_iter).index;
                }
              }
            }
          }

          in.numberofholes = 0;
          if (in.numberofholes > 0) {
            ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);
          ierr = PetscFree(in.trianglearealist);
          ierr = PetscFree(in.pointlist);
          ierr = PetscFree(in.pointmarkerlist);
          ierr = PetscFree(in.segmentlist);
          ierr = PetscFree(in.segmentmarkerlist);
        }
        m->populate(out.numberoftriangles, out.trianglelist, out.numberofpoints, out.pointlist, interpolate);

        // Make boundary
        Obj<Mesh::sieve_type> topology = m->getTopology();
        Obj<Mesh::field_type> boundary = m->getBoundary();
        Mesh::sieve_type::markerSet markers;

        if (rank == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              markers.insert(out.pointmarkerlist[v]);
            }
          }
          for(int e = 0; e < out.numberofedges; e++) {
            if (out.edgemarkerlist[e]) {
              markers.insert(out.edgemarkerlist[e]);
            }
          }
          boundary->setTopology(topology);
          for(Mesh::sieve_type::markerSet::iterator m_iter = markers.begin(); m_iter != markers.end(); ++m_iter) {
            boundary->setPatch(topology->leaves(), Mesh::field_type::patch_type(0, *m_iter));
          }
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              boundary->setFiberDimension(Mesh::field_type::patch_type(0, out.pointmarkerlist[v]), Mesh::point_type(0, v+out.numberoftriangles), 1);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type vertexA(0, out.edgelist[e*2+0]+out.numberoftriangles);
                Mesh::point_type vertexB(0, out.edgelist[e*2+1]+out.numberoftriangles);
                Obj<Mesh::sieve_type::supportSet> join = topology->join(vertexA, vertexB);

                boundary->setFiberDimension(Mesh::field_type::patch_type(0, out.edgemarkerlist[e]), *(join->begin()), 1);
              }
            }
          }
        }
        boundary->orderPatches();

        m = m->distribute();

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> refine_TetGen(Obj<Mesh> mesh, const double maxAreas[], bool interpolate) {
        ::tetgenio     in;
        ::tetgenio     out;
        int            dim = 3;
        Obj<Mesh>      m = Mesh(mesh->comm(), dim, mesh->debug);
        // FIX: Need to globalize
        PetscInt       numElements = mesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt    rank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(mesh->comm(), &rank);

        if (rank == 0) {
          in.tetrahedronvolumelist = new double[numElements];
        }
        {
          // Scatter in local area constraints
#ifdef PARALLEL
          Vec        locAreas;
          VecScatter areaScatter;

          ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, numElements, areas, &locAreas);CHKERRQ(ierr);
          ierr = MeshCreateMapping(oldMesh, elementBundle, partitionTypes, serialElementBundle, &areaScatter);CHKERRQ(ierr);
          ierr = VecScatterBegin(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
          ierr = VecScatterEnd(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
          ierr = VecDestroy(locAreas);CHKERRQ(ierr);
          ierr = VecScatterDestroy(areaScatter);CHKERRQ(ierr);
#else
          for(int i = 0; i < numElements; i++) {
            in.tetrahedronvolumelist[i] = maxAreas[i];
          }
#endif
        }

#ifdef PARALLEL
        Obj<Mesh> serialMesh = this->unify(mesh);
#else
        Obj<Mesh> serialMesh = mesh;
#endif
        Obj<Mesh::sieve_type>  serialTopology = serialMesh->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = serialMesh->getBundle(0);

        if (rank == 0) {
          std::string args("qenzQra");
          Obj<Mesh::sieve_type::traits::heightSequence> cells = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::traits::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::field_type>                 coordinates = serialMesh->getCoordinates();
          Mesh::field_type::patch_type          patch;
          int                                   c = 0;

          in.numberofpoints = vertices->size();
          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(Mesh::sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
            const Mesh::field_type::index_type& interval = coordinates->getIndex(patch, *v_itor);
            const Mesh::field_type::value_type *array = coordinates->restrict(patch, *v_itor);

            for(int d = 0; d < interval.index; d++) {
              in.pointlist[interval.prefix + d] = array[d];
            }
            const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *v_itor);
            in.pointmarkerlist[vInterval.prefix] = v_itor.marker();
          }

          in.numberofcorners = 4;
          in.numberoftetrahedra = cells->size();
          in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
          for(Mesh::sieve_type::traits::heightSequence::iterator c_itor = cells->begin(); c_itor != cells->end(); ++c_itor) {
            Obj<Mesh::field_type::IndexArray> intervals = vertexBundle->getIndices("element", *c_itor);
            int                               v = 0;

            for(Mesh::field_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
              if (i_itor->index) {
                in.tetrahedronlist[c * in.numberofcorners + v++] = i_itor->prefix;
              }
            }
            c++;
          }

          in.numberofholes = 0;
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist, interpolate);
  
        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(0, v + out.numberoftetrahedra), out.pointmarkerlist[v]);
            }
          }
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type endpointA(0, out.edgelist[e*2+0] + out.numberoftetrahedra);
                Mesh::point_type endpointB(0, out.edgelist[e*2+1] + out.numberoftetrahedra);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                Obj<Mesh::sieve_type::supportSet> point = Mesh::sieve_type::supportSet();
                Obj<Mesh::sieve_type::supportSet> edge = Mesh::sieve_type::supportSet();
                Mesh::point_type cornerA(0, out.edgelist[f*3+0] + out.numberoftetrahedra);
                Mesh::point_type cornerB(0, out.edgelist[f*3+1] + out.numberoftetrahedra);
                Mesh::point_type cornerC(0, out.edgelist[f*3+2] + out.numberoftetrahedra);
                point->insert(cornerA);
                edge->insert(cornerB);
                edge->insert(cornerC);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(point, edge, 2);

                topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
              }
            }
          }
        }
        m = m->distribute();
        return m;
      };
#endif
      static Obj<Mesh::field_type> getSerialConstraints(Obj<Mesh> serialMesh, Obj<Mesh> parallelMesh, Obj<Mesh::field_type> parallelConstraints);
    public:
      static Obj<Mesh> refine(Obj<Mesh> mesh, double maxArea, bool interpolate = true) {
        Obj<Mesh::field_type> constraints = Mesh::field_type(mesh->comm(), mesh->debug);
        int             numElements = mesh->getTopology()->heightStratum(0)->size();
        Mesh::field_type::patch_type patch;

        constraints->setTopology(mesh->getTopology());
        constraints->setPatch(mesh->getTopology()->leaves(), patch);
        constraints->setFiberDimensionByHeight(patch, 0, 1);
        constraints->orderPatches();

        double *maxAreas = new double[numElements];
        for(int e = 0; e < numElements; e++) {
          maxAreas[e] = maxArea;
        }
        constraints->update(patch, maxAreas);
        delete maxAreas;
        Obj<Mesh> refinedMesh = refine(mesh, constraints, interpolate);

        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, double (*maxArea)(const double centroid[], void *ctx), void *ctx, bool interpolate = true) {
        Obj<Mesh::sieve_type>                         topology = mesh->getTopology();
        Obj<Mesh::field_type>                         constraints = Mesh::field_type(mesh->comm(), mesh->debug);
        Obj<Mesh::field_type>                         coordinates = mesh->getCoordinates();
        Obj<Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
        Mesh::field_type::patch_type                  patch;
        int                                           corners = topology->nCone(*elements->begin(), topology->depth())->size();
        int                                           embedDim = coordinates->getFiberDimension(patch, *topology->depthStratum(0)->begin());
        double                                       *centroid = new double[embedDim];
        std::string                                   orderName("element");

        constraints->setTopology(topology);
        constraints->setPatch(topology->leaves(), patch);
        constraints->setFiberDimensionByHeight(patch, 0, 1);
        constraints->orderPatches();

        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          const double *coords = coordinates->restrict(orderName, *e_itor);

          for(int d = 0; d < embedDim; d++) {
            centroid[d] = 0.0;
            for(int c = 0; c < corners; c++) {
              centroid[d] += coords[c*embedDim+d];
            }
            centroid[d] /= corners;
          }
          double area = maxArea(centroid, ctx);
          constraints->update(patch, *e_itor, &area);
        }
        delete [] centroid;
        Obj<Mesh> refinedMesh = refine(mesh, constraints, interpolate);

        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, Obj<Mesh::field_type> parallelConstraints, bool interpolate = true) {
        Obj<Mesh> refinedMesh;
        Obj<Mesh> serialMesh;
        int       dim = mesh->getDimension();

        if (mesh->distributed) {
          serialMesh = mesh->unify();
        } else {
          serialMesh = mesh;
        }
        Obj<Mesh::field_type> serialConstraints = getSerialConstraints(serialMesh, mesh, parallelConstraints);
        Mesh::field_type::patch_type patch;

        if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
          refinedMesh = refine_Triangle(serialMesh, serialConstraints->restrict(patch), interpolate);
#else
          throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
          refinedMesh = refine_TetGen(serialMesh, serialConstraints->restrict(patch), interpolate);
#else
          throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
        }
        return refinedMesh;
      };
    };

    // Creation
    class PyLithBuilder {
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f) {
        while((fgets(buf, bufSize, f) != NULL) && (buf[0] == '#')) {}
      };

      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, bool useZeroBase, int& numElements, int *vertices[], int *materials[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       maxCells = 1024, cellCount = 0;
        PetscInt      *verts;
        PetscInt      *mats;
        char           buf[2048];
        PetscInt       c;
        PetscInt       commRank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(comm, &commRank);
        if (commRank != 0) return;
        ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
        ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
        ierr = PetscViewerFileSetName(viewer, filename.c_str());
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        /* Ignore comments */
        ignoreComments(buf, 2048, f);
        do {
          const char *v = strtok(buf, " ");
          int         elementType;

          if (cellCount == maxCells) {
            PetscInt *vtmp, *mtmp;

            vtmp = verts;
            mtmp = mats;
            ierr = PetscMalloc2(maxCells*2*corners,PetscInt,&verts,maxCells*2,PetscInt,&mats);
            ierr = PetscMemcpy(verts, vtmp, maxCells*corners * sizeof(PetscInt));
            ierr = PetscMemcpy(mats,  mtmp, maxCells         * sizeof(PetscInt));
            ierr = PetscFree2(vtmp,mtmp);
            maxCells *= 2;
          }
          /* Ignore cell number */
          v = strtok(NULL, " ");
          /* Get element type */
          elementType = atoi(v);
          if (elementType == 1) {
            corners = 8;
          } else if (elementType == 5) {
            corners = 4;
          } else {
            ostringstream msg;

            msg << "We do not accept element type " << elementType << " right now";
            throw ALE::Exception(msg.str().c_str());
          }
          if (cellCount == 0) {
            ierr = PetscMalloc2(maxCells*corners,PetscInt,&verts,maxCells,PetscInt,&mats);
          }
          v = strtok(NULL, " ");
          /* Store material type */
          mats[cellCount] = atoi(v);
          v = strtok(NULL, " ");
          /* Ignore infinite domain element code */
          v = strtok(NULL, " ");
          for(c = 0; c < corners; c++) {
            int vertex = atoi(v);
        
            if (!useZeroBase) vertex -= 1;
            verts[cellCount*corners+c] = vertex;
            v = strtok(NULL, " ");
          }
          cellCount++;
        } while(fgets(buf, 2048, f) != NULL);
        ierr = PetscViewerDestroy(viewer);
        numElements = cellCount;
        *vertices   = verts;
        *materials  = mats;
      };
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       maxVerts = 1024, vertexCount = 0;
        PetscScalar   *coords;
        char           buf[2048];
        PetscInt       c;
        PetscInt       commRank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(comm, &commRank);
        if (commRank == 0) {
          ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
          ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
          ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
          ierr = PetscViewerFileSetName(viewer, filename.c_str());
          ierr = PetscViewerASCIIGetPointer(viewer, &f);
          /* Ignore comments and units line */
          ignoreComments(buf, 2048, f);
          ierr = PetscMalloc(maxVerts*dim * sizeof(PetscScalar), &coords);
          /* Ignore comments */
          ignoreComments(buf, 2048, f);
          do {
            const char *x = strtok(buf, " ");

            if (vertexCount == maxVerts) {
              PetscScalar *ctmp;

              ctmp = coords;
              ierr = PetscMalloc(maxVerts*2*dim * sizeof(PetscScalar), &coords);
              ierr = PetscMemcpy(coords, ctmp, maxVerts*dim * sizeof(PetscScalar));
              ierr = PetscFree(ctmp);
              maxVerts *= 2;
            }
            /* Ignore vertex number */
            x = strtok(NULL, " ");
            for(c = 0; c < dim; c++) {
              coords[vertexCount*dim+c] = atof(x);
              x = strtok(NULL, " ");
            }
            vertexCount++;
          } while(fgets(buf, 2048, f) != NULL);
          ierr = PetscViewerDestroy(viewer);
          numVertices = vertexCount;
          *coordinates = coords;
        }
      };
      static void readSplit(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numSplit, int *splitInd[], double *splitValues[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       maxSplit = 1024, splitCount = 0;
        PetscInt      *splitId;
        PetscScalar   *splitVal;
        char           buf[2048];
        PetscInt       c;
        PetscInt       commRank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(comm, &commRank);
        if (dim != 3) {
          throw ALE::Exception("PyLith only works in 3D");
        }
        if (commRank != 0) return;
        ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
        ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
        ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename.c_str()), PETSC_ERR_FILE_OPEN);
        if (PetscExceptionValue(ierr)) {
          // this means that a caller above me has also tryed this exception so I don't handle it here, pass it up
        } else if (PetscExceptionCaught(ierr,PETSC_ERR_FILE_OPEN)) {
          // File does not exist
          return;
        } 
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        /* Ignore comments */
        ignoreComments(buf, 2048, f);
        ierr = PetscMalloc2(maxSplit*2,PetscInt,&splitId,maxSplit*dim,PetscScalar,&splitVal);
        do {
          const char *s = strtok(buf, " ");

          if (splitCount == maxSplit) {
            PetscInt    *sitmp;
            PetscScalar *svtmp;

            sitmp = splitId;
            svtmp = splitVal;
            ierr = PetscMalloc2(maxSplit*2*2,PetscInt,&splitId,maxSplit*dim*2,PetscScalar,&splitVal);
            ierr = PetscMemcpy(splitId,  sitmp, maxSplit*2   * sizeof(PetscInt));
            ierr = PetscMemcpy(splitVal, svtmp, maxSplit*dim * sizeof(PetscScalar));
            ierr = PetscFree2(sitmp,svtmp);
            maxSplit *= 2;
          }
          /* Get element number */
          int elem = atoi(s);
          if (!useZeroBase) elem -= 1;
          splitId[splitCount*2+0] = elem;
          s = strtok(NULL, " ");
          /* Get node number */
          int node = atoi(s);
          if (!useZeroBase) node -= 1;
          splitId[splitCount*2+1] = node;
          s = strtok(NULL, " ");
          /* Ignore load history number */
          s = strtok(NULL, " ");
          /* Get split values */
          for(c = 0; c < dim; c++) {
            splitVal[splitCount*dim+c] = atof(s);
            s = strtok(NULL, " ");
          }
          splitCount++;
        } while(fgets(buf, 2048, f) != NULL);
        ierr = PetscViewerDestroy(viewer);
        numSplit     = splitCount;
        *splitInd    = splitId;
        *splitValues = splitVal;
      };
      static void createMaterialField(int numElements, int materials[], Obj<Mesh> mesh, Obj<Mesh::field_type> matField) {
        Obj<Mesh::sieve_type::traits::heightSequence> elements = mesh->getTopology()->heightStratum(0);
        Mesh::field_type::patch_type patch;

        matField->setPatch(elements, patch);
        matField->setFiberDimensionByHeight(patch, 0, 1);
        matField->orderPatches();
        for(int e = 0; e < numElements; e++) {
          double mat = (double) materials[e];
          matField->update(patch, Mesh::point_type(0, e), &mat);
        }
      };
      static void createSplitField(int numSplit, int splitInd[], double splitVals[], Obj<Mesh> mesh, Obj<Mesh::field_type> splitField) {
        Obj<Mesh::sieve_type::traits::heightSequence> elements = mesh->getTopology()->heightStratum(0);
        std::map<Mesh::point_type, std::set<int> > elem2vertIndex;
        Obj<std::list<Mesh::point_type> > vertices = std::list<Mesh::point_type>();
        Mesh::field_type::patch_type patch;
        int numElements = elements->size();
        int dim = 3;

        for(int e = 0; e < numSplit; e++) {
          elem2vertIndex[Mesh::point_type(0, splitInd[e*2+0])].insert(e);
        }
        for(std::map<Mesh::point_type, std::set<int> >::iterator e_iter = elem2vertIndex.begin(); e_iter != elem2vertIndex.end(); ++e_iter) {
          vertices->clear();
          for(std::set<int>::iterator v_iter = e_iter->second.begin(); v_iter != e_iter->second.end(); ++v_iter) {
            vertices->push_back(Mesh::point_type(0, numElements+splitInd[*v_iter*2+1]));
          }
          splitField->setPatch(vertices, e_iter->first);

          for(std::list<Mesh::point_type>::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
            splitField->setFiberDimension(e_iter->first, *v_iter, dim);
          }
        }
        splitField->orderPatches();
        for(std::map<Mesh::point_type, std::set<int> >::iterator e_iter = elem2vertIndex.begin(); e_iter != elem2vertIndex.end(); ++e_iter) {
          for(std::set<int>::iterator v_iter = e_iter->second.begin(); v_iter != e_iter->second.end(); ++v_iter) {
            splitField->update(e_iter->first, Mesh::point_type(0, numElements+splitInd[*v_iter*2+1]), &splitVals[*v_iter*dim]);
          }
        }
      };
    public:
      PyLithBuilder() {};
      virtual ~PyLithBuilder() {};

      static Obj<Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, bool interpolate = true, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices = NULL;
        int      *materials = NULL;
        double   *coordinates = NULL;
        int      *splitInd = NULL;
        double   *splitValues = NULL;
        int       numElements = 0, numVertices = 0, numCorners = dim+1, numSplit = 0, hasSplit;
        PetscErrorCode ierr;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".connect", numCorners, useZeroBase, numElements, &vertices, &materials);
        readCoordinates(comm, baseFilename+".coord", dim, numVertices, &coordinates);
        readSplit(comm, baseFilename+".split", dim, useZeroBase, numSplit, &splitInd, &splitValues);
        mesh->populate(numElements, vertices, numVertices, coordinates, interpolate, numCorners);
        createMaterialField(numElements, materials, mesh, mesh->getField("material"));
        ierr = MPI_Allreduce(&numSplit, &hasSplit, 1, MPI_INT, MPI_MAX, comm);
        if (hasSplit) {
          createSplitField(numSplit, splitInd, splitValues, mesh, mesh->getField("split"));
        }
        ierr = PetscFree2(vertices, materials);
        ierr = PetscFree(coordinates);
        return mesh;
      };
    };

    class PCICEBuilder {
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, bool useZeroBase, int& numElements, int *vertices[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       numCells, cellCount = 0;
        PetscInt      *verts;
        char           buf[2048];
        PetscInt       c;
        PetscInt       commRank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(comm, &commRank);

        if (commRank != 0) return;
        ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
        ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
        ierr = PetscViewerFileSetName(viewer, filename.c_str());
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        if (fgets(buf, 2048, f) == NULL) {
          throw ALE::Exception("Invalid connectivity file: Missing number of elements");
        }
        const char *sizes = strtok(buf, " ");
        numCells = atoi(sizes);
        sizes = strtok(NULL, " ");
        if (sizes != NULL) {
          corners = atoi(sizes);
          std::cout << "Reset corners to " << corners << std::endl;
        }
        ierr = PetscMalloc(numCells*corners * sizeof(PetscInt), &verts);
        while(fgets(buf, 2048, f) != NULL) {
          const char *v = strtok(buf, " ");
      
          /* Ignore cell number */
          v = strtok(NULL, " ");
          for(c = 0; c < corners; c++) {
            int vertex = atoi(v);
        
            if (!useZeroBase) vertex -= 1;
            verts[cellCount*corners+c] = vertex;
            v = strtok(NULL, " ");
          }
          cellCount++;
        }
        ierr = PetscViewerDestroy(viewer);
        numElements = numCells;
        *vertices = verts;
      };
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       numVerts, vertexCount = 0;
        PetscScalar   *coords;
        char           buf[2048];
        PetscInt       c;
        PetscInt       commRank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(comm, &commRank);

        if (commRank != 0) return;
        ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
        ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
        ierr = PetscViewerFileSetName(viewer, filename.c_str());
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        numVerts = atoi(fgets(buf, 2048, f));
        ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);
        while(fgets(buf, 2048, f) != NULL) {
          const char *x = strtok(buf, " ");
      
          /* Ignore vertex number */
          x = strtok(NULL, " ");
          for(c = 0; c < dim; c++) {
            coords[vertexCount*dim+c] = atof(x);
            x = strtok(NULL, " ");
          }
          vertexCount++;
        }
        ierr = PetscViewerDestroy(viewer);
        numVertices = numVerts;
        *coordinates = coords;
      };
    public:
      PCICEBuilder() {};
      virtual ~PCICEBuilder() {};

      static Obj<ALE::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<ALE::Mesh> mesh = ALE::Mesh(comm, dim, debug);
        int      *vertices;
        double   *coordinates;
        int       numElements = 0, numVertices = 0, numCorners = dim+1;

        readConnectivity(comm, baseFilename+".lcon", numCorners, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates, true, numCorners);
        return mesh;
      };

      static Obj<ALE::Mesh> createNewBd(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<ALE::Mesh> mesh = ALE::Mesh(comm, dim, debug);
        int      *vertices = NULL;
        double   *coordinates = NULL;
        int       numElements = 0, numVertices = 0;
        PetscErrorCode ierr;

        readConnectivity(comm, baseFilename+".lcon", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim+1, numVertices, &coordinates);
        mesh->populateBd(numElements, vertices, numVertices, coordinates);
        ierr = PetscFree(vertices);
        ierr = PetscFree(coordinates);
        return mesh;
      };
    };

    class Partitioner {
    public:
//       typedef ParDelta<Mesh::sieve_type>;
      typedef RightSequenceDuplicator<ConeArraySequence<Mesh::sieve_type::traits::arrow_type> > fuser;
      typedef ParConeDelta<Mesh::sieve_type, fuser,
                           Mesh::sieve_type::rebind<fuser::fusion_source_type,
                                                    fuser::fusion_target_type,
                                                    fuser::fusion_color_type,
                                                    Mesh::sieve_type::traits::cap_container_type::rebind<fuser::fusion_source_type, Mesh::sieve_type::traits::sourceRec_type::rebind<fuser::fusion_source_type, Mesh::sieve_type::marker_type>::type>::type,
                                                    Mesh::sieve_type::traits::base_container_type::rebind<fuser::fusion_target_type, Mesh::sieve_type::traits::targetRec_type::rebind<fuser::fusion_target_type, Mesh::sieve_type::marker_type>::type>::type
      >::type> coneDelta_type;
      typedef ParSupportDelta<Mesh::sieve_type, fuser,
                              Mesh::sieve_type::rebind<fuser::fusion_source_type,
                                                       fuser::fusion_target_type,
                                                       fuser::fusion_color_type,
                                                       Mesh::sieve_type::traits::cap_container_type::rebind<fuser::fusion_source_type, Mesh::sieve_type::traits::sourceRec_type::rebind<fuser::fusion_source_type, Mesh::sieve_type::marker_type>::type>::type,
                                                       Mesh::sieve_type::traits::base_container_type::rebind<fuser::fusion_target_type, Mesh::sieve_type::traits::targetRec_type::rebind<fuser::fusion_target_type, Mesh::sieve_type::marker_type>::type>::type
      >::type> supportDelta_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "createMappingStoP"
      template<typename FieldType, typename OverlapType>
      static VecScatter createMappingStoP(Obj<FieldType> serialSifter, Obj<FieldType> parallelSifter, Obj<OverlapType> overlap, bool doExchange = false) {
        VecScatter scatter;
        Obj<typename OverlapType::traits::baseSequence> neighbors = overlap->base();
        MPI_Comm comm = serialSifter->comm();
        int      rank = serialSifter->commRank();
        int      debug = serialSifter->debug;
        typename FieldType::patch_type patch;
        Vec        serialVec, parallelVec;
        PetscErrorCode ierr;

        if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "Creating mapping\n");}
        // Use an MPI vector for the serial data since it has no overlap
        if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating serial indices\n");}
        if (serialSifter->debug) {
          serialSifter->view("SerialSifter");
          overlap->view("Partition Overlap");
        }
        ierr = VecCreateMPIWithArray(serialSifter->comm(), serialSifter->getSize(patch), PETSC_DETERMINE, serialSifter->restrict(patch), &serialVec);CHKERROR(ierr, "Error in VecCreate");
        // Use individual serial vectors for each of the parallel domains
        if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating parallel indices\n");}
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, parallelSifter->getSize(patch), parallelSifter->restrict(patch), &parallelVec);CHKERROR(ierr, "Error in VecCreate");

        int NeighborCountA = 0, NeighborCountB = 0;
        for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
          Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              NeighborCountA++;
              break;
            }
          }
          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if ((*p_iter).first == 1) {
              NeighborCountB++;
              break;
            }
          } 
        }

        int *NeighborsA, *NeighborsB; // Neighbor processes
        int *SellSizesA, *BuySizesA;  // Sizes of the A cones to transmit and B cones to receive
        int *SellSizesB, *BuySizesB;  // Sizes of the B cones to transmit and A cones to receive
        int *SellConesA = PETSC_NULL;
        int *SellConesB = PETSC_NULL;
        int nA, nB, offsetA, offsetB;
        ierr = PetscMalloc2(NeighborCountA,int,&NeighborsA,NeighborCountB,int,&NeighborsB);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(NeighborCountA,int,&SellSizesA,NeighborCountA,int,&BuySizesA);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(NeighborCountB,int,&SellSizesB,NeighborCountB,int,&BuySizesB);CHKERROR(ierr, "Error in PetscMalloc");

        nA = 0;
        nB = 0;
        for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
          Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              NeighborsA[nA] = *neighbor;
              BuySizesA[nA] = 0;
              SellSizesA[nA] = 0;
              nA++;
              break;
            }
          }
          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if ((*p_iter).first == 1) {
              NeighborsB[nB] = *neighbor;
              BuySizesB[nB] = 0;
              SellSizesB[nB] = 0;
              nB++;
              break;
            }
          } 
        }
        if ((nA != NeighborCountA) || (nB != NeighborCountB)) {
          throw ALE::Exception("Invalid neighbor count");
        }

        nA = 0;
        offsetA = 0;
        nB = 0;
        offsetB = 0;
        for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
          Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);
          int foundA = 0, foundB = 0;

          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              // Assume the same index sizes
              int idxSize = serialSifter->getIndex(patch, (*p_iter).second).index;

              BuySizesA[nA] += idxSize;
              SellSizesA[nA] += idxSize;
              offsetA += idxSize;
              foundA = 1;
            } else {
              // Assume the same index sizes
              int idxSize = parallelSifter->getIndex(patch, (*p_iter).second).index;

              BuySizesB[nB] += idxSize;
              SellSizesB[nB] += idxSize;
              offsetB += idxSize;
              foundB = 1;
            }
          }
          if (foundA) nA++;
          if (foundB) nB++;
        }

        ierr = PetscMalloc2(offsetA,int,&SellConesA,offsetB,int,&SellConesB);CHKERROR(ierr, "Error in PetscMalloc");
        offsetA = 0;
        offsetB = 0;
        for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
          Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

          for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const Point& p = (*p_iter).second;

            if ((*p_iter).first == 0) {
              const typename FieldType::index_type& idx = serialSifter->getIndex(patch, p);

              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Packing A index " << idx << " for " << *neighbor << std::endl;
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              for(int i = idx.prefix; i < idx.prefix+idx.index; ++i) {
                SellConesA[offsetA++] = i;
              }
            } else {
              const typename FieldType::index_type& idx = parallelSifter->getIndex(patch, p);

              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Packing B index " << idx << " for " << *neighbor << std::endl;
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              for(int i = idx.prefix; i < idx.prefix+idx.index; ++i) {
                SellConesB[offsetB++] = i;
              }
            }
          }
        }
        if (debug) {
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        ierr = VecScatterCreateEmpty(comm, &scatter);CHKERROR(ierr, "Error in VecScatterCreate");
        scatter->from_n = serialSifter->getSize(patch);
        scatter->to_n = parallelSifter->getSize(patch);
        ierr = VecScatterCreateLocal_PtoS(NeighborCountA, SellSizesA, NeighborsA, SellConesA, NeighborCountB, SellSizesB, NeighborsB, SellConesB, 1, scatter);CHKERROR(ierr, "Error in VecScatterCreate");

        if (doExchange) {
          if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Exchanging data\n");}
          ierr = VecScatterBegin(serialVec, parallelVec, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERROR(ierr, "Error in VecScatter");
          ierr = VecScatterEnd(serialVec, parallelVec, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERROR(ierr, "Error in VecScatter");
        }

        ierr = VecDestroy(serialVec);CHKERROR(ierr, "Error in VecDestroy");
        ierr = VecDestroy(parallelVec);CHKERROR(ierr, "Error in VecDestroy");
        return scatter;
      };
    };
    inline
    Obj<Mesh::field_type> Generator::getSerialConstraints(Obj<Mesh> serialMesh, Obj<Mesh> parallelMesh, Obj<Mesh::field_type> parallelConstraints) {
      Obj<Mesh::field_type> serialConstraints = Mesh::field_type(parallelMesh->comm(), parallelMesh->debug);

      if (parallelMesh->distributed) {
        Mesh::field_type::patch_type patch;
        PetscErrorCode ierr;

        serialConstraints->setTopology(serialMesh->getTopology());
        serialConstraints->setPatch(serialMesh->getTopology()->leaves(), patch);
        serialConstraints->setFiberDimensionByHeight(patch, 0, 1);
        serialConstraints->orderPatches();

        Obj<ALE::Partitioner::coneDelta_type::bioverlap_type> partitionOverlap = ALE::Partitioner::coneDelta_type::overlap(serialMesh->getTopology(), parallelMesh->getTopology());
        Obj<Flip<ALE::Partitioner::coneDelta_type::bioverlap_type> > overlapFlip = Flip<ALE::Partitioner::coneDelta_type::bioverlap_type>(partitionOverlap);
        VecScatter scatter = ALE::Partitioner::createMappingStoP(serialConstraints, parallelConstraints, overlapFlip, false);
        Vec        serialVec, parallelVec;

        ierr = VecCreateMPIWithArray(serialMesh->comm(), serialConstraints->getSize(patch), PETSC_DETERMINE, serialConstraints->restrict(patch), &serialVec);CHKERROR(ierr, "Error in VecCreate");
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, parallelConstraints->getSize(patch), parallelConstraints->restrict(patch), &parallelVec);CHKERROR(ierr, "Error in VecCreate");
        ierr = VecScatterBegin(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
        ierr = VecScatterEnd(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
        ierr = VecDestroy(serialVec);CHKERROR(ierr, "Error in VecDestroy");
        ierr = VecDestroy(parallelVec);CHKERROR(ierr, "Error in VecDestroy");
        ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");
      } else {
        serialConstraints = parallelConstraints;
      }
      return serialConstraints;
    };
    #undef __FUNCT__
    #define __FUNCT__ "Mesh::createParCoords"
    template<typename OverlapType>
    void Mesh::createParallelCoordinates(int embedDim, Obj<field_type> serialCoordinates, Obj<OverlapType> partitionOverlap) {
      if (this->debug) {
        serialCoordinates->view("Serial coordinates");
        this->topology->view("Parallel topology");
      }
      ALE_LOG_EVENT_BEGIN;
      patch_type patch;

      this->coordinates->setTopology(this->topology);
      this->coordinates->setPatch(this->topology->leaves(), patch);
      this->coordinates->setFiberDimensionByDepth(patch, 0, embedDim);
      this->coordinates->orderPatches();
      if (this->debug) {
        this->coordinates->view("New parallel coordinates");
      }
      this->coordinates->createGlobalOrder();

      VecScatter scatter = ALE::Partitioner::createMappingStoP(serialCoordinates, this->coordinates, partitionOverlap, true);
      PetscErrorCode ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");

      Obj<bundle_type> vertexBundle = this->getBundle(0);
      Obj<sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      std::string orderName("element");

      for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
        this->coordinates->setPatch(orderName, vertexBundle->getPatch(orderName, *e_iter), *e_iter);
      }
      ALE_LOG_EVENT_END;
    };
    #undef __FUNCT__
    #define __FUNCT__ "Mesh::distribute"
    inline
    Obj<Mesh> Mesh::distribute() {
      ALE_LOG_EVENT_BEGIN;
      // Partition the topology
      Obj<Mesh> parallelMesh = Mesh(this->comm(), this->getDimension(), this->debug);
      ALE::MeshPartitioner<Mesh>::partition(*this, parallelMesh);
      // Calculate the bioverlap
      if (this->debug) {
        this->topology->view("Serial topology");
        parallelMesh->topology->view("Parallel topology");
        parallelMesh->getBoundary()->view("Parallel boundary");
      }
      Obj<Partitioner::supportDelta_type::bioverlap_type> partitionOverlap = Partitioner::supportDelta_type::overlap(this->topology, parallelMesh->topology);
      // Need to deal with boundary
      parallelMesh->createParallelVertexReorder(this->getBundle(0));
      parallelMesh->createParallelCoordinates(this->dim, this->coordinates, partitionOverlap);
      parallelMesh->distributed = true;
      ALE_LOG_EVENT_END;
      return parallelMesh;
    };

    #undef __FUNCT__
    #define __FUNCT__ "Mesh::unify"
    inline
    Obj<Mesh> Mesh::unify() {
      ALE_LOG_EVENT_BEGIN;
      Obj<Mesh> serialMesh = Mesh(this->comm(), this->getDimension(), this->debug);
      Obj<Mesh::sieve_type> topology = serialMesh->getTopology();
      Mesh::patch_type patch;
      PetscErrorCode ierr;

      topology->setStratification(false);
      // Partition the topology
      ALE::MeshPartitioner<Mesh>::unify(*this, serialMesh);
      topology->stratify();
      topology->setStratification(true);
      if (serialMesh->debug) {
        topology->view("Serial mesh");
      }
      // Need to deal with boundary
      serialMesh->bundles.clear();
      serialMesh->fields.clear();

      serialMesh->coordinates->setTopology(serialMesh->getTopology());
      serialMesh->coordinates->setPatch(serialMesh->getTopology()->leaves(), patch);
      serialMesh->coordinates->setFiberDimensionByDepth(patch, 0, this->dim);
      serialMesh->coordinates->orderPatches();

      Obj<Partitioner::supportDelta_type::bioverlap_type> partitionOverlap = Partitioner::supportDelta_type::overlap(serialMesh->topology, this->topology);
      VecScatter scatter = ALE::Partitioner::createMappingStoP(serialMesh->coordinates, this->coordinates, partitionOverlap, false);
      Vec        serialVec, parallelVec;

      ierr = VecCreateMPIWithArray(serialMesh->comm(), serialMesh->coordinates->getSize(patch), PETSC_DETERMINE, serialMesh->coordinates->restrict(patch), &serialVec);CHKERROR(ierr, "Error in VecCreate");
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, this->coordinates->getSize(patch), this->coordinates->restrict(patch), &parallelVec);CHKERROR(ierr, "Error in VecCreate");
      ierr = VecScatterBegin(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
      ierr = VecScatterEnd(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
      ierr = VecDestroy(serialVec);CHKERROR(ierr, "Error in VecDestroy");
      ierr = VecDestroy(parallelVec);CHKERROR(ierr, "Error in VecDestroy");
      ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");

      std::string orderName("element");
      Obj<bundle_type> vertexBundle = this->getBundle(0);
      Obj<bundle_type> serialVertexBundle = serialMesh->getBundle(0);
      Obj<bundle_type::order_type::baseSequence> patches = vertexBundle->getPatches(orderName);

      for(bundle_type::order_type::baseSequence::iterator e_iter = patches->begin(); e_iter != patches->end(); ++e_iter) {
        Obj<bundle_type::order_type::coneSequence> patch = vertexBundle->getPatch(orderName, *e_iter);

        serialVertexBundle->setPatch(orderName, patch, *e_iter);
        for(bundle_type::order_type::coneSequence::iterator p_iter = patch->begin(); p_iter != patch->end(); ++p_iter) {
          serialVertexBundle->setFiberDimension(orderName, *e_iter, *p_iter, 1);
        }
      }
      if (!this->commRank()) {
        Obj<sieve_type::traits::heightSequence> elements = serialMesh->getTopology()->heightStratum(0);
        Obj<bundle_type::order_type> reorder = serialVertexBundle->__getOrder(orderName);

        for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
          reorder->addBasePoint(*e_iter);
        }
      }
      serialVertexBundle->orderPatches(orderName);
      serialVertexBundle->partitionOrder(orderName);

      serialMesh->distributed = false;
      ALE_LOG_EVENT_END;
      return serialMesh;
    }
} // namespace ALE

#endif
