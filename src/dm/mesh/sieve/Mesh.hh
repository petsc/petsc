#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif

namespace ALE {

  namespace def {
    // A computational mesh
    class Mesh {
    public:
      typedef Sieve<Point,int> sieve_type;
      typedef CoSieve<sieve_type, int, Point, double> coordinate_type;
      typedef CoSieve<sieve_type, int, Point, int> bundle_type;
      int debug;
    private:
      sieve_type      topology;
      sieve_type      orientation;
      coordinate_type coordinates;
      coordinate_type boundary;
      std::map<std::string, Obj<coordinate_type> > fields;
      std::map<int, Obj<bundle_type> > bundles;
      MPI_Comm        comm;
      int             dim;
    public:
      Mesh(MPI_Comm c, int dimension) : debug(0), comm(c), dim(dimension) {};

      MPI_Comm             getComm() {return this->comm;};
      Obj<sieve_type>      getTopology() {return this->topology;};
      Obj<sieve_type>      getOrientation() {return this->orientation;};
      int                  getDimension() {return this->dim;};
      Obj<coordinate_type> getCoordinates() {return this->coordinates;};
      Obj<coordinate_type> getBoundary() {return this->boundary;};
      Obj<coordinate_type> getField() {return this->fields.begin()->second;};
      Obj<coordinate_type> getField(const std::string& name) {return this->fields[name];};
      void                 setField(const std::string& name, const Obj<coordinate_type>& field) {this->fields[name] = field;};
      Obj<bundle_type>     getBundle(const int dim) {
        if (this->bundles.find(dim) == this->bundles.end()) {
          Obj<ALE::def::Mesh::bundle_type> vertexBundle = ALE::def::Mesh::bundle_type();

          // Need to globalize indices (that is what we might use the value ints for)
          std::cout << "Creating new bundle for dim " << dim << std::endl;
          vertexBundle->setTopology(this->topology);
          vertexBundle->setPatch(this->topology.leaves(), 0);
          vertexBundle->setIndexDimensionByDepth(dim, 1);
          vertexBundle->orderPatches();
          this->bundles[dim] = vertexBundle;
        }
        return this->bundles[dim];
      }

      void buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, Point& simplex) {
        Obj<PointSet> faces = PointSet();

        if (debug > 1) {std::cout << "  Building faces for boundary(size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 1) {
          // Use the cone construction
          for(PointSet::iterator b_itor = boundary->begin(); b_itor != boundary->end(); ++b_itor) {
            Obj<PointSet> faceBoundary = PointSet();
            Point         face;

            faceBoundary.copy(boundary);
            if (debug > 1) {std::cout << "    boundary point " << *b_itor << std::endl;}
            faceBoundary->erase(*b_itor);
            this->buildFaces(dim-1, curSimplex, faceBoundary, face);
            faces->insert(face);
          }
        } else {
          if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
          faces = boundary;
        }
        if (debug > 1) {
          for(PointSet::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            std::cout << "  face point " << *f_itor << std::endl;
          }
        }
        // We always create the toplevel, so we could shortcircuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        PointSet::iterator f_itor = faces->begin();
        Point              start = *f_itor;
        f_itor++;
        Point              next = *f_itor;
        Obj<PointSet>      preElement = this->topology.nJoin(start, next, 1);

        if (preElement->size() > 0) {
          simplex = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old simplex " << simplex << std::endl;}
        } else {
          simplex = Point(0, (*(*curSimplex)[dim])++);
          this->topology.addCone(faces, simplex);
          if (debug > 1) {std::cout << "  Added simplex " << simplex << " dim " << dim << std::endl;}
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "Mesh::buildTopology"
      // Build a topology from a connectivity description
      //   (0, 0)            ... (0, numSimplices-1):  dim-dimensional simplices
      //   (0, numSimplices) ... (0, numVertices):     vertices
      // The other simplices are numbered as they are requested
      void buildTopology(int numSimplices, int simplices[], int numVertices) {
        ALE_LOG_EVENT_BEGIN;
        // Create a map from dimension to the current element number for that dimension
        std::map<int,int*> curElement = std::map<int,int*>();
        int                curSimplex = 0;
        int                curVertex  = numSimplices;
        int                newElement = numSimplices+numVertices;
        Obj<PointSet>      boundary   = PointSet();
        Obj<PointSet>      cellTuple  = PointSet();

        curElement[0]   = &curVertex;
        curElement[dim] = &curSimplex;
        for(int d = 1; d < dim; d++) {
          curElement[d] = &newElement;
        }
        for(int s = 0; s < numSimplices; s++) {
          Point simplex(0, s);

          // Build the simplex
          boundary->clear();
          for(int b = 0; b < dim+1; b++) {
            Point vertex(0, simplices[s*(dim+1)+b]+numSimplices);

            if (debug > 1) {std::cout << "Adding boundary node " << vertex << std::endl;}
            boundary->insert(vertex);
          }
          if (debug) {std::cout << "simplex " << s << " boundary size " << boundary->size() << std::endl;}
          this->buildFaces(this->dim, &curElement, boundary, simplex);
          // Orient the simplex
          Point element(0, simplices[s*(dim+1)+0]+numSimplices);
          cellTuple->clear();
          cellTuple->insert(element);
          for(int b = 1; b < dim+1; b++) {
            Point next(0, simplices[s*(dim+1)+b]+numSimplices);
            Obj<PointSet> join = this->topology.nJoin(element, next, b);

            if (join->size() == 0) {
              if (debug) {std::cout << "element " << element << " next " << next << std::endl;}
              throw ALE::Exception("Invalid join");
            }
            element =  *join->begin();
            cellTuple->insert(element);
          }
          this->orientation.addCone(cellTuple, simplex);
        }
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "Mesh::createSerialCoordinates"
      void createSerialCoordinates(int numElements, double coords[]) {
        int dim = this->dim;

        ALE_LOG_EVENT_BEGIN;
        std::cout << "Creating coordinates" << std::endl;
        this->topology.debug = this->debug;
        this->coordinates.setTopology(this->topology);
        std::cout << "  setting patch" << std::endl;
        this->coordinates.setPatch(this->topology.leaves(), 0);
        std::cout << "  setting index dimensions" << std::endl;
        this->coordinates.setIndexDimensionByDepth(0, dim);
        std::cout << "  ordering patches" << std::endl;
        this->coordinates.orderPatches();
        std::cout << "  setting coordinates" << std::endl;
        Obj<Sieve<Point,int>::depthSequence> vertices = this->topology.depthStratum(0);
        for(Sieve<Point,int>::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
          if ((*v_itor).index%100 == 0) {std::cout << "Fiber index over vertex " << *v_itor << " is " << *this->coordinates.getIndices(0, *v_itor)->begin() << std::endl;}
          this->coordinates.update(0, *v_itor, &coords[((*v_itor).index - numElements)*dim]);
        }
        ALE_LOG_EVENT_END;
      };
      // I think that the boundary shuold be marked in the Sieve
      //   It could be done with point markers like depth/height, but is that right?
      // Should also put in boundary edges
      void createBoundary(int numBoundaryVertices, int numBoundaryComponents, int boundaryVertices[], double boundaryValues[]) {
        //FIX: Need to globalize
        int numElements = this->topology.heightStratum(0)->size();

        this->boundary.debug = this->debug;
        this->boundary.setTopology(this->topology);
        this->boundary.setPatch(this->topology.leaves(), 0);
        // Reverse order allows newer conditions to override older, as required by PyLith
        for(int v = numBoundaryVertices-1; v >= 0; v--) {
          sieve_type::point_type vertex(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);

          if (this->boundary.getIndexDimension(0, vertex) == 0) {
            for(int c = 0; c < numBoundaryComponents; c++) {
              if (boundaryVertices[v*(numBoundaryComponents+1)+c+1]) {
                this->boundary.setIndexDimension(0, vertex, c+1, 1);
              }
            }
          }
        }
        this->boundary.orderPatches();
        for(int v = 0; v < numBoundaryVertices; v++) {
          sieve_type::point_type vertex(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);

          this->boundary.update(0, vertex, &boundaryValues[v*numBoundaryComponents]);
        }
      };
      void populate(int numSimplices, int simplices[], int numVertices, double coords[]) {
        PetscMPIInt rank;

        MPI_Comm_rank(this->comm, &rank);
        /* Create serial sieve */
        this->topology.debug = this->debug;
        this->topology.setStratification(false);
        if (rank == 0) {
          this->buildTopology(numSimplices, simplices, numVertices);
        }
        this->topology.stratify();
        this->topology.setStratification(true);
        this->createSerialCoordinates(numSimplices, coords);
      };
    };

    // Creation
    class PyLithBuilder {
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f) {
        while((fgets(buf, bufSize, f) != NULL) && (buf[0] == '#')) {}
      };

      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
    public:
      PyLithBuilder() {};
      virtual ~PyLithBuilder() {};

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".connect", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".coord", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
        return mesh;
      };
    };

    class PCICEBuilder {
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
    public:
      PCICEBuilder() {};
      virtual ~PCICEBuilder() {};

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".lcon", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
        return mesh;
      };
    };
  } // namespace def
} // namespace ALE

#endif
