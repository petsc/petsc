#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif
#ifndef  included_ALE_ParDelta_hh
#include <ParDelta.hh>
#endif

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>
#endif
#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#endif

#include <petscvec.h>

namespace ALE {
  namespace Two {
    // Forward declaration
    class Partitioner;

    class Mesh {
    public:
      typedef ALE::Point point_type;
      typedef std::vector<point_type> PointArray;
      typedef ALE::Three::Sieve<point_type,int,int> sieve_type;
      typedef point_type patch_type;
      typedef CoSifter<sieve_type, patch_type, point_type, int> bundle_type;
      typedef CoSifter<sieve_type, patch_type, point_type, double> field_type;
      typedef CoSifter<sieve_type, std::pair<patch_type,int>, point_type, double> foliation_type;
      int debug;
    public:
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const std::pair<patch_type,int>& p) {
        os << "<" << p.first << ", "<< p.second << ">";
        return os;
      };
    private:
      Obj<sieve_type> topology;
      Obj<field_type> coordinates;
      Obj<field_type> boundary;
      Obj<foliation_type> boundaries;
      std::map<int, Obj<bundle_type> > bundles;
      std::map<std::string, Obj<field_type> > fields;
      MPI_Comm        _comm;
      int             _commRank;
      int             _commSize;
      int             dim;
      //FIX:
      bool            distributed;
    public:
      Mesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
        this->setComm(comm);
        this->topology    = sieve_type(comm, debug);
        this->coordinates = field_type(comm, debug);
        this->boundary    = field_type(comm, debug);
        this->boundaries  = foliation_type(comm, debug);
        this->distributed = false;
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
      }

      void buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointArray> boundary, point_type& simplex) {
        Obj<PointArray> faces = PointArray();

        if (debug > 1) {std::cout << "  Building faces for boundary(size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 1) {
          // Use the cone construction
          for(PointArray::iterator b_itor = boundary->begin(); b_itor != boundary->end(); ++b_itor) {
            Obj<PointArray> faceBoundary = PointArray();
            point_type    face;

            for(PointArray::iterator i_itor = boundary->begin(); i_itor != boundary->end(); ++i_itor) {
              if (i_itor != b_itor) {
                faceBoundary->push_back(*i_itor);
              }
            }
            if (debug > 1) {std::cout << "    boundary point " << *b_itor << std::endl;}
            this->buildFaces(dim-1, curSimplex, faceBoundary, face);
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
        // We always create the toplevel, so we could shortcircuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        PointArray::iterator f_itor = faces->begin();
        point_type           start = *f_itor;
        f_itor++;
        point_type           next = *f_itor;
        Obj<ALE::PointSet> preElement = this->topology->nJoin(start, next, 1);

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
      };

      #undef __FUNCT__
      #define __FUNCT__ "Mesh::buildTopology"
      // Build a topology from a connectivity description
      //   (0, 0)            ... (0, numSimplices-1):  dim-dimensional simplices
      //   (0, numSimplices) ... (0, numVertices):     vertices
      // The other simplices are numbered as they are requested
      void buildTopology(int numSimplices, int simplices[], int numVertices, bool interpolate = true) {
        ALE_LOG_EVENT_BEGIN;
        // Create a map from dimension to the current element number for that dimension
        std::map<int,int*> curElement = std::map<int,int*>();
        int                curSimplex = 0;
        int                curVertex  = numSimplices;
        int                newElement = numSimplices+numVertices;
        Obj<PointArray>    boundary   = PointArray();

        curElement[0]   = &curVertex;
        curElement[dim] = &curSimplex;
        for(int d = 1; d < dim; d++) {
          curElement[d] = &newElement;
        }
        for(int s = 0; s < numSimplices; s++) {
          point_type simplex(0, s);

          // Build the simplex
          if (interpolate) {
            boundary->clear();
            for(int b = 0; b < dim+1; b++) {
              point_type vertex(0, simplices[s*(dim+1)+b]+numSimplices);

              if (debug > 1) {std::cout << "Adding boundary node " << vertex << std::endl;}
              boundary->push_back(vertex);
            }
            if (debug) {std::cout << "simplex " << s << " boundary size " << boundary->size() << std::endl;}

            this->buildFaces(this->dim, &curElement, boundary, simplex);
          } else {
            for(int b = 0; b < dim+1; b++) {
              point_type p(0, simplices[s*(dim+1)+b]+numSimplices);

              this->topology->addArrow(p, simplex, b);
            }
          }
        }
        ALE_LOG_EVENT_END;
      };

      #undef __FUNCT__
      #define __FUNCT__ "Mesh::createVertBnd"
      void createVertexBundle(int numSimplices, int simplices[]) {
        ALE_LOG_STAGE_BEGIN;
        Obj<bundle_type> vertexBundle = this->getBundle(0);
        Obj<sieve_type::traits::heightSequence> elements = this->topology->heightStratum(0);
        std::string orderName("element");

        ALE_LOG_EVENT_BEGIN;
        for(sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          // setFiberDimensionByDepth() does not work here since we only want it to apply to the patch cone
          //   What we really need is the depthStratum relative to the patch
          Obj<PointArray> patch = PointArray();

          for(int b = 0; b < dim+1; b++) {
            patch->push_back(point_type(0, simplices[(*e_iter).index*(this->dim+1)+b]+numSimplices));
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
          Obj<bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(*e_iter);

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
      #define __FUNCT__ "Mesh::createParCoords"
      void createParallelCoordinates(int embedDim, Obj<bundle_type> serialVertexBundle, Obj<field_type> serialCoordinates) {
        if (this->debug) {
          serialCoordinates->view("Serial coordinates");
          this->topology->view("Parallel topology");
        }
        ALE_LOG_EVENT_BEGIN;
        // Create vertex bundle
        std::string orderName("element");
        Obj<bundle_type> vertexBundle = this->getBundle(0);

        if (!this->_commRank) {
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
        // Create coordinates
        patch_type patch;

        this->coordinates->setTopology(this->topology);
        this->coordinates->setPatch(this->topology->leaves(), patch);
        this->coordinates->setFiberDimensionByDepth(patch, 0, embedDim);
        this->coordinates->orderPatches();
        if (this->debug) {
          this->coordinates->view("New parallel coordinates");
        }
        this->coordinates->createGlobalOrder();

        VecScatter scatter;
        Vec serialVec, globalVec;
        IS  serialIS, globalIS;

        if (this->commRank()) {
          VecCreateMPIWithArray(this->comm(), 0, PETSC_DETERMINE, PETSC_NULL, &serialVec);
        } else {
          VecCreateMPIWithArray(this->comm(), serialCoordinates->getSize(patch), PETSC_DETERMINE, serialCoordinates->restrict(patch), &serialVec);
        }
        int *indices = this->__expandIntervals(this->coordinates->getGlobalOrder()->getPatch(patch));
        ISCreateGeneral(PETSC_COMM_SELF, this->coordinates->getSize(patch), indices, &serialIS);
        delete [] indices;
        VecCreateSeqWithArray(PETSC_COMM_SELF, this->coordinates->getSize(patch), this->coordinates->restrict(patch), &globalVec);
        //ISCreateStride(PETSC_COMM_SELF, this->coordinates->getSize(patch), 0, 1, &globalIS);
        indices = this->__expandCanonicalIntervals(this->coordinates->getGlobalOrder()->getPatch(patch), this->coordinates);
        ISCreateGeneral(PETSC_COMM_SELF, this->coordinates->getSize(patch), indices, &globalIS);
        delete [] indices;
        VecScatterCreate(serialVec, serialIS, globalVec, globalIS, &scatter);
        ISDestroy(serialIS);
        ISDestroy(globalIS);
        VecScatterBegin(serialVec, globalVec, INSERT_VALUES, SCATTER_FORWARD, scatter);
        VecScatterEnd(serialVec, globalVec, INSERT_VALUES, SCATTER_FORWARD, scatter);
        VecDestroy(serialVec);
        VecDestroy(globalVec);
        VecScatterDestroy(scatter);
        ALE_LOG_EVENT_END;
      };

      // Create a serial mesh
      void populate(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true) {
        this->topology->setStratification(false);
        if (this->commRank() == 0) {
          this->buildTopology(numSimplices, simplices, numVertices, interpolate);
        }
        this->topology->stratify();
        this->topology->setStratification(true);
        this->createVertexBundle(numSimplices, simplices);
        this->createSerialCoordinates(this->dim, numSimplices, coords);
      };

      // Create a serial mesh
      void distribute();
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

        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(0, v + out.numberoftriangles), out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type endpointA(0, out.edgelist[e*2+0] + out.numberoftriangles);
                Mesh::point_type endpointB(0, out.edgelist[e*2+1] + out.numberoftriangles);
                Obj<ALE::PointSet> join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
        }

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
                  Obj<ALE::PointSet> join = topology->nJoin(endpointA, endpointB, 1);

                  topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
                }
              }
            }
            if (out.trifacemarkerlist) {
              for(int f = 0; f < out.numberoftrifaces; f++) {
                if (out.trifacemarkerlist[f]) {
                  Obj<ALE::PointSet> point = ALE::PointSet();
                  Obj<ALE::PointSet> edge = ALE::PointSet();
                  Mesh::point_type cornerA(0, out.trifacelist[f*3+0] + out.numberoftetrahedra);
                  Mesh::point_type cornerB(0, out.trifacelist[f*3+1] + out.numberoftetrahedra);
                  Mesh::point_type cornerC(0, out.trifacelist[f*3+2] + out.numberoftetrahedra);
                  point->insert(cornerA);
                  edge->insert(cornerB);
                  edge->insert(cornerC);
                  Obj<ALE::PointSet> join = topology->nJoin(point, edge, 2);

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
      static Obj<Mesh> refine_Triangle(Obj<Mesh> mesh, double maxAreas[], bool interpolate) {
        struct triangulateio in;
        struct triangulateio out;
        int                  dim = 2;
        Obj<Mesh>            m = Mesh(mesh->comm(), dim, mesh->debug);
        // FIX: Need to globalize
        PetscInt             numElements = mesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt          rank;
        PetscErrorCode       ierr;

        ierr = MPI_Comm_rank(mesh->comm(), &rank);
        initInput_Triangle(&in);
        initOutput_Triangle(&out);
        if (rank == 0) {
          ierr = PetscMalloc(numElements * sizeof(double), &in.trianglearealist);
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
            in.trianglearealist[i] = maxAreas[i];
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
          std::string args("pqenzQra");
          Obj<Mesh::sieve_type::traits::heightSequence> faces = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::traits::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::field_type>                 coordinates = serialMesh->getCoordinates();
          Mesh::field_type::patch_type          patch;
          int                                   f = 0;

          in.numberofpoints = vertices->size();
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

          in.numberofcorners = 3;
          in.numberoftriangles = faces->size();
          ierr = PetscMalloc(in.numberoftriangles * in.numberofcorners * sizeof(int), &in.trianglelist);
          for(Mesh::sieve_type::traits::heightSequence::iterator f_itor = faces->begin(); f_itor != faces->end(); f_itor++) {
            Obj<Mesh::field_type::IndexArray> intervals = vertexBundle->getIndices("element", *f_itor);
            int                               v = 0;

            for(Mesh::field_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
              if (i_itor->index) {
                in.trianglelist[f * in.numberofcorners + v++] = i_itor->prefix;
              }
            }
            f++;
          }

          Obj<Mesh::sieve_type::traits::depthSequence> segments = serialTopology->depthStratum(1, 1);
          Obj<Mesh::bundle_type> segmentBundle = Mesh::bundle_type();

          segmentBundle->setTopology(serialTopology);
          segmentBundle->setPatch(segments, patch);
          segmentBundle->setFiberDimensionByDepth(patch, 1, 1);
          segmentBundle->orderPatches();
          in.numberofsegments = segments->size();
          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            for(Mesh::sieve_type::traits::depthSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
              const Mesh::field_type::index_type& interval = segmentBundle->getIndex(patch, *s_itor);
              Obj<Mesh::sieve_type::traits::coneSequence> cone = serialTopology->cone(*s_itor);
              int                                 p = 0;
        
              for(Mesh::sieve_type::traits::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *c_itor);

                in.segmentlist[interval.prefix * 2 + (p++)] = vInterval.prefix;
              }
              in.segmentmarkerlist[interval.prefix] = s_itor.marker();
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
        m->distribute();

        // Need to make boundary

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> refine_TetGen(Obj<Mesh> mesh, double maxAreas[], bool interpolate) {
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
                Obj<ALE::PointSet> join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                Obj<ALE::PointSet> point = ALE::PointSet();
                Obj<ALE::PointSet> edge = ALE::PointSet();
                Mesh::point_type cornerA(0, out.edgelist[f*3+0] + out.numberoftetrahedra);
                Mesh::point_type cornerB(0, out.edgelist[f*3+1] + out.numberoftetrahedra);
                Mesh::point_type cornerC(0, out.edgelist[f*3+2] + out.numberoftetrahedra);
                point->insert(cornerA);
                edge->insert(cornerB);
                edge->insert(cornerC);
                Obj<ALE::PointSet> join = topology->nJoin(point, edge, 2);

                topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
              }
            }
          }
        }
        m->distribute();
        return m;
      };
#endif
    public:
      static Obj<Mesh> refine(Obj<Mesh> mesh, double maxArea, bool interpolate = true) {
        int       numElements = mesh->getTopology()->heightStratum(0)->size();
        double   *maxAreas = new double[numElements];
        for(int e = 0; e < numElements; e++) {
          maxAreas[e] = maxArea;
        }
        Obj<Mesh> refinedMesh = refine(mesh, maxAreas, interpolate);

        delete [] maxAreas;
        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, double maxAreas[], bool interpolate = true) {
        Obj<Mesh> refinedMesh;
        int       dim = mesh->getDimension();

        if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
          refinedMesh = refine_Triangle(mesh, maxAreas, interpolate);
#else
          throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
          refinedMesh = refine_TetGen(mesh, maxAreas, interpolate);
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

      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]) {
        PetscViewer    viewer;
        FILE          *f;
        PetscInt       maxCells = 1024, cellCount = 0;
        PetscInt      *verts;
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
        ierr = PetscViewerFileSetName(viewer, filename.c_str());
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        /* Ignore comments */
        ignoreComments(buf, 2048, f);
        ierr = PetscMalloc(maxCells*(dim+1) * sizeof(PetscInt), &verts);
        do {
          const char *v = strtok(buf, " ");
          int         elementType;

          if (cellCount == maxCells) {
            PetscInt *vtmp;

            vtmp = verts;
            ierr = PetscMalloc(maxCells*2*(dim+1) * sizeof(PetscInt), &verts);
            ierr = PetscMemcpy(verts, vtmp, maxCells*(dim+1) * sizeof(PetscInt));
            ierr = PetscFree(vtmp);
            maxCells *= 2;
          }
          /* Ignore cell number */
          v = strtok(NULL, " ");
          /* Verify element type is linear tetrahedron */
          elementType = atoi(v);
          if (elementType != 5) {
            throw ALE::Exception("We only accept linear tetrahedra right now");
          }
          v = strtok(NULL, " ");
          /* Ignore material type */
          v = strtok(NULL, " ");
          /* Ignore infinite domain element code */
          v = strtok(NULL, " ");
          for(c = 0; c <= dim; c++) {
            int vertex = atoi(v);
        
            if (!useZeroBase) vertex -= 1;
            verts[cellCount*(dim+1)+c] = vertex;
            v = strtok(NULL, " ");
          }
          cellCount++;
        } while(fgets(buf, 2048, f) != NULL);
        ierr = PetscViewerDestroy(viewer);
        numElements = cellCount;
        *vertices = verts;
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
    public:
      PyLithBuilder() {};
      virtual ~PyLithBuilder() {};

      static Obj<ALE::Two::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, bool interpolate = true, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<ALE::Two::Mesh> mesh = ALE::Two::Mesh(comm, dim);
        int      *vertices = NULL;
        double   *coordinates = NULL;
        int       numElements = 0, numVertices = 0;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".connect", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".coord", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates, interpolate);
        return mesh;
      };
    };

    class PCICEBuilder {
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]) {
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
        numCells = atoi(fgets(buf, 2048, f));
        ierr = PetscMalloc(numCells*(dim+1) * sizeof(PetscInt), &verts);
        while(fgets(buf, 2048, f) != NULL) {
          const char *v = strtok(buf, " ");
      
          /* Ignore cell number */
          v = strtok(NULL, " ");
          for(c = 0; c <= dim; c++) {
            int vertex = atoi(v);
        
            if (!useZeroBase) vertex -= 1;
            verts[cellCount*(dim+1)+c] = vertex;
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

      static Obj<ALE::Two::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<ALE::Two::Mesh> mesh = ALE::Two::Mesh(comm, dim, debug);
        int      *vertices;
        double   *coordinates;
        int       numElements = 0, numVertices = 0;

        readConnectivity(comm, baseFilename+".lcon", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
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
      #define __FUNCT__ "partition_Sieve"
      static void partition_Sieve(const Obj<Mesh>& mesh, bool localize = true) {
        ALE_LOG_EVENT_BEGIN;
        Obj<Mesh::sieve_type> topology = mesh->getTopology();
        Obj<ALE::PointSet> localBase = ALE::PointSet();
        const char *name = NULL;

        // Construct a Delta object and a base overlap object
        coneDelta_type::setDebug(mesh->debug);
        Obj<coneDelta_type::overlap_type> overlap = coneDelta_type::overlap(topology);
        // Cone complete to move the partitions to the other processors
        Obj<coneDelta_type::fusion_type>  fusion  = coneDelta_type::fusion(topology, overlap);
        // Merge in the completion
        topology->add(fusion);
        if (mesh->debug) {
          topology->view("After merging inital fusion");
        }
        if(localize) {
          Obj<Mesh::sieve_type::coneSequence> cone = topology->cone(Mesh::point_type(-1, mesh->commRank()));

          localBase->insert(cone->begin(), cone->end());
          for(int p = 0; p < mesh->commSize(); ++p) {
            topology->removeBasePoint(Mesh::point_type(-1, p));
          }
        }
        // Support complete to build the local topology
        //supportDelta_type::setDebug(mesh->debug);
        Obj<supportDelta_type::overlap_type> overlap2 = supportDelta_type::overlap(topology);
        Obj<supportDelta_type::fusion_type>  fusion2  = supportDelta_type::fusion(topology, overlap2);
        topology->add(fusion2);
        if (mesh->debug) {
          fusion2->view("Second fusion");
          topology->view("After merging second fusion");
        }
        // Unless explicitly prohibited, restrict to the local partition
        if(localize) {
          if (mesh->debug) {
            std::cout << "["<<mesh->commRank()<<"]: Restricting base to";
            for(ALE::PointSet::iterator b_iter = localBase->begin(); b_iter != localBase->end(); ++b_iter) {
              std::cout << " " << *b_iter;
            }
            std::cout << std::endl;
          }
          topology->restrictBase(localBase);
          //FIX: The contains() method does not work
          //topology->restrictBase(topology->cone(Mesh::point_type(-1, mesh->commRank())));
          Obj<Mesh::sieve_type::capSequence> cap = topology->cap();

          for(Mesh::sieve_type::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
            if (topology->support(*c_iter)->size() == 0) {
              topology->removeCapPoint(*c_iter);
            }
          }
          if (mesh->debug) {
            ostringstream label5;
            if(name != NULL) {
              label5 << "Localized parallel version of '" << name << "'";
            } else {
              label5 << "Localized parallel sieve";
            }
            topology->view(label5.str().c_str());
          }
        }
        ALE_LOG_EVENT_END;
      };

      #undef __FUNCT__
      #define __FUNCT__ "partition_Simple"
      static void partition_Simple(const ALE::Obj<ALE::Two::Mesh> mesh) {
        ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
        PetscInt numLeaves = topology->leaves()->size();
        const char *name = NULL;

        PetscFunctionBegin;
        ALE_LOG_STAGE_BEGIN;
        ALE_LOG_EVENT_BEGIN;
        if (topology->commRank() == 0) {
          int size = topology->commSize();

          for(int p = 0; p < size; p++) {
            ALE::Two::Mesh::point_type partitionPoint(-1, p);

            for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
              topology->addCone(topology->closure(ALE::Two::Mesh::point_type(0, l)), partitionPoint);
            }
          }
        } else {
          ALE::Two::Mesh::point_type partitionPoint(-1, topology->commRank());
          topology->addBasePoint(partitionPoint);
        }
        if (mesh->debug) {
          ostringstream label1;
          label1 << "Partition of sieve ";
          if(name != NULL) {
            label1 << "'" << name << "'";
          }
          label1 << "\n";
          topology->view(label1.str().c_str());
        }
        ALE_LOG_EVENT_END;
        ALE_LOG_STAGE_END;
      };

#ifdef PETSC_HAVE_CHACO
      static void partition_Chaco(const ALE::Obj<ALE::Two::Mesh> mesh) {
        int size;
        MPI_Comm_size(mesh->getComm(), &size);

        /* arguments for Chaco library */
        int nvtxs;                              /* number of vertices in full graph */
        int *start;                             /* start of edge list for each vertex */
        int *adjacency;                         /* = adj -> j; edge list data  */
        int *vwgts = NULL;                      /* weights for all vertices */
        float *ewgts = NULL;                    /* weights for all edges */
        float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
        char *outassignname = NULL;             /*  name of assignment output file */
        char *outfilename = NULL;               /* output file name */
        short *assignment;                      /* set number of each vtx (length n) */
        int architecture = 1;                   /* 0 => hypercube, d => d-dimensional mesh */
        int ndims_tot = 0;                      /* total number of cube dimensions to divide */
        int *mesh_dims = size;                  /* dimensions of mesh of processors */
        double *goal = NULL;                    /* desired set sizes for each set */
        int global_method = 1;                  /* global partitioning algorithm */
        int local_method = 1;                   /* local partitioning algorithm */
        int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
        int vmax = 200;                         /* how many vertices to coarsen down to? */
        int ndims = 1;                          /* number of eigenvectors (2^d sets) */
        double eigtol = 0.001;                  /* tolerance on eigenvectors */
        long seed = 123636512;                  /* for random graph mutations */

        nvtxs = mesh->getTopology()->heightStratum(0)->size();
        start = new int[nvtxs+1];

        Obj<sieve_type::heightSequence> faces = mesh->getTopology()->heightStratum(1);
        Obj<bundle_type> vertexBundle = mesh->getBundle(0);
        Obj<bundle_type> elementBundle = mesh->getBundle(mesh->getTopology()->depth());
        ierr = PetscMemzero(start, (nvtxs+1) * sizeof(int));CHKERRQ(ierr);
        for(sieve_type::heightSequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
          Obj<sieve_type::supportSequence> cells = mesh->getTopology()->support(*f_iter);

          if (cells->size() == 2) {
            start[elementBundle->getIndex(*cells->begin()).prefix+1]++;
            start[elementBundle->getIndex(*(++cells->begin())).prefix+1]++;
          }
        }
        for(int v = 1; v <= nvtxs; v++) {
          start[v] += start[v-1];
        }
        adjacency = new int[start[nvtxs]];
        for(sieve_type::heightSequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
          Obj<sieve_type::supportSequence> cells = mesh->getTopology()->support(*f_iter);

          if (cells->size()) {
            int cellA = elementBundle->getIndex(*cells->begin()).prefix;
            int cellB = elementBundle->getIndex(*(++cells->begin())).prefix;

            adjacency[cellA+1] = cellB;
            adjacency[cellB+1] = cellA;
          }
        }

        assignment = new int[nvtxs];

        /* redirect output to buffer: chaco -> msgLog */
#ifdef PETSC_HAVE_UNISTD_H
        char *msgLog;
        int fd_stdout, fd_pipe[2], count;

        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
        msgLog = new char[16284];
#endif

        ierr = interface(nvtxs, start, adjacency, vwgts, ewgts, x, y, z,
                         outassignname, outfilename, assignment, architecture, ndims_tot,
                         mesh_dims, goal, global_method, local_method, rqi_flag, vmax, ndims,
                         eigtol, seed);

#ifdef PETSC_HAVE_UNISTD_H
        fflush(stdout);
        count = read(fd_pipe[0], msgLog, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0) count = 0;
        msgLog[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
        std::cout << msgLog << std::endl;
        delete [] msgLog;
#endif

        delete [] assignment;
        delete [] adjacency;
        delete [] start;
      };
#endif
    public:
      static void partition(const Obj<Mesh> mesh) {
        int dim = mesh->getDimension();

        if (dim == 2) {
#ifdef PETSC_HAVE_CHACO
          partition_Chaco(mesh);
#else
          partition_Simple(mesh);
          //throw ALE::Exception("Mesh partitioning currently requires Chaco to be installed. Use --download-chaco during configure.");
#endif
        } else {
          partition_Simple(mesh);
        }
        partition_Sieve(mesh);
      };
    };

    void Mesh::distribute() {
      ALE_LOG_EVENT_BEGIN;
      this->topology->setStratification(false);
      // Partition the topology
      ALE::Two::Partitioner::partition(*this);
      this->topology->stratify();
      this->topology->setStratification(true);
      Obj<Mesh::sieve_type::baseSequence> base = topology->base();
      int dim = this->getDimension();

      for(Mesh::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        if (b_iter.depth() + b_iter.height() != dim) {
          topology->removeBasePoint(*b_iter);
        }
      }
      if (this->debug) {
        this->topology->view("Parallel mesh");
      }
      // Need to deal with boundary
      Obj<bundle_type> vertexBundle = this->getBundle(0);
      Obj<field_type>  coordinates  = this->coordinates;
      this->coordinates = field_type(this->comm(), this->debug);
      this->bundles.clear();
      this->fields.clear();
      this->createParallelCoordinates(this->dim, vertexBundle, coordinates);
      this->distributed = true;
      ALE_LOG_EVENT_END;
    };
  } // namespace Two
} // namespace ALE

#endif
