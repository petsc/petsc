#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>
#endif
#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#endif

namespace ALE {

  namespace Two {
    class Mesh {
    public:
      typedef ALE::def::Point point_type;
      typedef std::vector<point_type> PointArray;
      typedef ALE::def::Sieve<point_type,int> sieve_type;
      typedef point_type patch_type;
      typedef CoSifter<sieve_type, patch_type, point_type, int> bundle_type;
      typedef CoSifter<sieve_type, patch_type, point_type, double> field_type;
      int debug;
    private:
      Obj<sieve_type> topology;
      Obj<field_type> coordinates;
      std::map<int, Obj<bundle_type> > bundles;
      MPI_Comm        comm;
      int             rank;
      int             dim;
    public:
      Mesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
        this->setComm(comm);
        this->topology    = sieve_type(debug);
        this->coordinates = field_type(debug);
      };

      MPI_Comm        getComm() {return this->comm;};
      void            setComm(MPI_Comm comm) {this->comm = comm; MPI_Comm_rank(comm, &this->rank);};
      int             getRank() {return this->comm;};
      Obj<sieve_type> getTopology() {return this->topology;};
      void            setTopology(const Obj<sieve_type>& topology) {this->topology = topology;};
      int             getDimension() {return this->dim;};
      void            setDimension(int dim) {this->dim = dim;};
      Obj<field_type> getCoordinates() {return this->coordinates;};
      void            setCoordinates(const Obj<field_type>& coordinates) {this->coordinates = coordinates;};
      Obj<bundle_type> getBundle(const int dim) {
        if (this->bundles.find(dim) == this->bundles.end()) {
          Obj<bundle_type> bundle = bundle_type(debug);

          // Need to globalize indices (that is what we might use the value ints for)
          std::cout << "Creating new bundle for dim " << dim << std::endl;
          bundle->setTopology(this->topology);
          bundle->setPatch(this->topology->leaves(), bundle_type::patch_type());
          bundle->setFiberDimensionByDepth(bundle_type::patch_type(), dim, 1);
          bundle->orderPatches();
          this->bundles[dim] = bundle;
        }
        return this->bundles[dim];
      };

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
        Obj<ALE::def::PointSet> preElement = this->topology->nJoin(start, next, 1);

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
      #define __FUNCT__ "Mesh::createSerCoords"
      void createSerialCoordinates(int embedDim, int numElements, double coords[]) {
        ALE_LOG_EVENT_BEGIN;
        patch_type patch;

        this->coordinates->setTopology(this->topology);
        this->coordinates->setPatch(this->topology->leaves(), patch);
        this->coordinates->setFiberDimensionByDepth(patch, 0, embedDim);
        this->coordinates->orderPatches();
        Obj<sieve_type::depthSequence> vertices = this->topology->depthStratum(0);
        for(sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
          this->coordinates->update(patch, *v_itor, &coords[((*v_itor).index - numElements)*embedDim]);
        }
        ALE_LOG_EVENT_END;
      };

      // Create a serial mesh
      void populate(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true) {
        this->topology->debug = this->debug;
        this->topology->setStratification(false);
        if (this->getRank() == 0) {
          this->buildTopology(numSimplices, simplices, numVertices, interpolate);
        }
        this->topology->stratify();
        this->topology->setStratification(true);
        this->createSerialCoordinates(this->dim, numSimplices, coords);
      };
    };
  }

  namespace def {
    // A computational mesh
    class Mesh {
    public:
      typedef Point point_type;
      typedef Sieve<point_type,int> sieve_type;
      typedef CoSieve<sieve_type, point_type, int, int> ordering_type;
      typedef CoSieve<sieve_type, int, point_type, double> coordinate_type;
      typedef CoSieve<sieve_type, int, point_type, int> bundle_type;
      int debug;
    private:
      Obj<sieve_type>      topology;
      Obj<sieve_type>      orientation;
      Obj<ordering_type>   ordering;
      Obj<coordinate_type> coordinates;
      Obj<coordinate_type> boundary;
      std::map<std::string, Obj<coordinate_type> > fields;
      std::map<int, Obj<bundle_type> > bundles;
      MPI_Comm        comm;
      int             dim;
    public:
      Mesh(MPI_Comm c, int dimension, int debug = 0) : debug(debug), comm(c), dim(dimension) {
        this->topology    = sieve_type(debug);
        this->orientation = sieve_type(debug);
        this->ordering    = ordering_type(debug);
        this->coordinates = coordinate_type(debug);
        this->boundary    = coordinate_type(debug);
      };

      MPI_Comm             getComm() {return this->comm;};
      Obj<sieve_type>      getTopology() {return this->topology;};
      Obj<sieve_type>      getOrientation() {return this->orientation;};
      Obj<ordering_type>   getOrdering() {return this->ordering;};
      int                  getDimension() {return this->dim;};
      Obj<coordinate_type> getCoordinates() {return this->coordinates;};
      Obj<coordinate_type> getBoundary() {return this->boundary;};
      Obj<coordinate_type> getField() {return this->fields.begin()->second;};
      Obj<coordinate_type> getField(const std::string& name) {return this->fields[name];};
      void                 setField(const std::string& name, const Obj<coordinate_type>& field) {this->fields[name] = field;};
      Obj<bundle_type>     getBundle(const int dim) {
        if (this->bundles.find(dim) == this->bundles.end()) {
          Obj<ALE::def::Mesh::bundle_type> bundle = ALE::def::Mesh::bundle_type();

          // Need to globalize indices (that is what we might use the value ints for)
          std::cout << "Creating new bundle for dim " << dim << std::endl;
          bundle->debug = this->debug;
          bundle->setTopology(this->topology);
          bundle->setPatch(this->topology->leaves(), 0);
          bundle->setIndexDimensionByDepth(dim, 1);
          bundle->orderPatches();
          this->bundles[dim] = bundle;
        }
        return this->bundles[dim];
      }

      void buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, point_type& simplex) {
        Obj<PointSet> faces = PointSet();

        if (debug > 1) {std::cout << "  Building faces for boundary(size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 1) {
          // Use the cone construction
          for(PointSet::iterator b_itor = boundary->begin(); b_itor != boundary->end(); ++b_itor) {
            Obj<PointSet> faceBoundary = PointSet();
            point_type    face;

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
        point_type         start = *f_itor;
        f_itor++;
        point_type         next = *f_itor;
        Obj<PointSet>      preElement = this->topology->nJoin(start, next, 1);

        if (preElement->size() > 0) {
          simplex = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old simplex " << simplex << std::endl;}
        } else {
          simplex = point_type(0, (*(*curSimplex)[dim])++);
          this->topology->addCone(faces, simplex);
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
        Obj<PointSet>      boundary   = PointSet();
        Obj<PointSet>      cellTuple  = PointSet();

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
              boundary->insert(vertex);
            }
            if (debug) {std::cout << "simplex " << s << " boundary size " << boundary->size() << std::endl;}

            this->buildFaces(this->dim, &curElement, boundary, simplex);

            // Orient the simplex
            point_type element(0, simplices[s*(dim+1)+0]+numSimplices);
            cellTuple->clear();
            cellTuple->insert(element);
            for(int b = 1; b < dim+1; b++) {
              point_type next(0, simplices[s*(dim+1)+b]+numSimplices);
              Obj<PointSet> join = this->topology->nJoin(element, next, b);

              if (join->size() == 0) {
                if (debug) {std::cout << "element " << element << " next " << next << std::endl;}
                throw ALE::Exception("Invalid join");
              }
              element =  *join->begin();
              cellTuple->insert(element);
            }
            this->orientation->addCone(cellTuple, simplex);
          } else {
            for(int b = 0; b < dim+1; b++) {
              point_type p(0, simplices[s*(dim+1)+b]+numSimplices);

              this->topology->addArrow(p, simplex);
              this->orientation->addArrow(p, simplex, b);
            }
          }
        }
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "Mesh::createSerCoords"
      void createSerialCoordinates(int embedDim, int numElements, double coords[]) {
        ALE_LOG_EVENT_BEGIN;
        this->topology->debug = this->debug;
        this->coordinates->setTopology(this->topology);
        this->coordinates->setPatch(this->topology->leaves(), 0);
        this->coordinates->setIndexDimensionByDepth(0, embedDim);
        this->coordinates->orderPatches();
        Obj<sieve_type::depthSequence> vertices = this->topology->depthStratum(0);
        for(sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
          this->coordinates->update(0, *v_itor, &coords[((*v_itor).index - numElements)*embedDim]);
        }
        ALE_LOG_EVENT_END;
      };
      // I think that the boundary should be marked in the Sieve
      //   It could be done with point markers like depth/height, but is that right?
      // Should also put in boundary edges
      void createBoundary(int numBoundaryVertices, int numBoundaryComponents, int boundaryVertices[], double boundaryValues[]) {
        //FIX: Need to globalize
        int numElements = this->topology->heightStratum(0)->size();

        this->boundary->debug = this->debug;
        this->boundary->setTopology(this->topology);
        this->boundary->setPatch(this->topology->leaves(), 0);
        // Reverse order allows newer conditions to override older, as required by PyLith
        for(int v = numBoundaryVertices-1; v >= 0; v--) {
          point_type vertex(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);

          if (this->boundary->getIndexDimension(0, vertex) == 0) {
            for(int c = 0; c < numBoundaryComponents; c++) {
              if (boundaryVertices[v*(numBoundaryComponents+1)+c+1]) {
                this->boundary->setIndexDimension(0, vertex, c+1, 1);
              }
            }
          }
        }
        this->boundary->orderPatches();
        for(int v = 0; v < numBoundaryVertices; v++) {
          point_type vertex(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);

          this->boundary->update(0, vertex, &boundaryValues[v*numBoundaryComponents]);
        }
      };
      void populate(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true) {
        PetscMPIInt rank;

        MPI_Comm_rank(this->comm, &rank);
        /* Create serial sieve */
        this->topology->debug = this->debug;
        this->topology->setStratification(false);
        if (rank == 0) {
          this->buildTopology(numSimplices, simplices, numVertices, interpolate);
        }
        this->topology->stratify();
        this->topology->setStratification(true);
        this->createSerialCoordinates(this->dim, numSimplices, coords);
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

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, bool interpolate = true, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".connect", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".coord", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates, interpolate);
        return mesh;
      };

      static Obj<ALE::Two::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, bool interpolate = true, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<ALE::Two::Mesh> mesh = ALE::Two::Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

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

      static Obj<ALE::Two::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<ALE::Two::Mesh> mesh = ALE::Two::Mesh(comm, dim);
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
      static Obj<Mesh> generate_Triangle(Obj<Mesh> boundary) {
        struct triangulateio  in;
        struct triangulateio  out;
        int                   dim = 2;
        Obj<Mesh>             m = Mesh(boundary->getComm(), dim);
        Obj<Mesh::sieve_type> bdTopology = boundary->getTopology();
        Obj<Mesh::sieve_type> bdOrientation = boundary->getOrientation();
        PetscMPIInt           rank;
        PetscErrorCode        ierr;

        ierr = MPI_Comm_rank(boundary->getComm(), &rank);
        initInput_Triangle(&in);
        initOutput_Triangle(&out);
        if (rank == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          Obj<Mesh::sieve_type::depthSequence> vertices = bdTopology->depthStratum(0);
          Obj<Mesh::bundle_type>               vertexBundle = boundary->getBundle(0);

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            Obj<Mesh::coordinate_type> coordinates = boundary->getCoordinates();

            ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
            ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
            for(Mesh::sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
              const Mesh::coordinate_type::index_type& interval = coordinates->getIndex(0, *v_itor);
              const Mesh::coordinate_type::value_type *array = coordinates->restrict(0, *v_itor);

              for(int d = 0; d < interval.index; d++) {
                in.pointlist[interval.prefix + d] = array[d];
              }
              const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *v_itor);
              in.pointmarkerlist[vInterval.prefix] = v_itor.getMarker();
            }
          }

          Obj<Mesh::sieve_type::depthSequence> edges = bdTopology->depthStratum(1);
          Obj<Mesh::bundle_type>               edgeBundle = boundary->getBundle(1);

          in.numberofsegments = edges->size();
          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            for(Mesh::sieve_type::depthSequence::iterator e_itor = edges->begin(); e_itor != edges->end(); e_itor++) {
              const Mesh::coordinate_type::index_type& interval = edgeBundle->getIndex(0, *e_itor);
              Obj<Mesh::sieve_type::coneSequence>      cone = bdTopology->cone(*e_itor);
              int                                      p = 0;
        
              for(Mesh::sieve_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *c_itor);

                in.segmentlist[interval.prefix * 2 + (p++)] = vInterval.prefix;
              }
              in.segmentmarkerlist[interval.prefix] = e_itor.getMarker();
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
        m->populate(out.numberoftriangles, out.trianglelist, out.numberofpoints, out.pointlist);

        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(0, v + out.numberoftriangles), out.pointmarkerlist[v]);
            }
          }
          for(int e = 0; e < out.numberofedges; e++) {
            if (out.edgemarkerlist[e]) {
              Mesh::point_type endpointA(0, out.edgelist[e*2+0] + out.numberoftriangles);
              Mesh::point_type endpointB(0, out.edgelist[e*2+1] + out.numberoftriangles);
              Obj<PointSet>    join = topology->nJoin(endpointA, endpointB, 1);

              topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
            }
          }
        }

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      #undef __FUNCT__
      #define __FUNCT__ "generate_TetGen"
      static Obj<Mesh> generate_TetGen(Obj<Mesh> boundary) {
        ::tetgenio            in;
        ::tetgenio            out;
        int                   dim = 3;
        Obj<Mesh>             m = Mesh(boundary->getComm(), dim);
        Obj<Mesh::sieve_type> bdTopology = boundary->getTopology();
        Obj<Mesh::sieve_type> bdOrientation = boundary->getOrientation();
        Obj<Mesh::ordering_type> bdOrdering = boundary->getOrdering();
        PetscMPIInt           rank;
        PetscErrorCode        ierr;

        ierr = MPI_Comm_rank(boundary->getComm(), &rank);

        if (rank == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          Obj<Mesh::sieve_type::depthSequence> vertices = bdTopology->depthStratum(0);
          Obj<Mesh::bundle_type>               vertexBundle = boundary->getBundle(0);

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            Obj<Mesh::coordinate_type> coordinates = boundary->getCoordinates();

            in.pointlist       = new double[in.numberofpoints*dim];
            in.pointmarkerlist = new int[in.numberofpoints];
            for(Mesh::sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
              const Mesh::coordinate_type::index_type& interval = coordinates->getIndex(0, *v_itor);
              const Mesh::coordinate_type::value_type *array = coordinates->restrict(0, *v_itor);

              for(int d = 0; d < interval.index; d++) {
                in.pointlist[interval.prefix + d] = array[d];
              }
              const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *v_itor);
              in.pointmarkerlist[vInterval.prefix] = v_itor.getMarker();
            }
          }

          Obj<Mesh::sieve_type::heightSequence> facets = bdTopology->heightStratum(0);
          Obj<Mesh::bundle_type>                facetBundle = boundary->getBundle(bdTopology->depth());

          in.numberoffacets = facets->size();
          if (in.numberoffacets > 0) {
            in.facetlist       = new tetgenio::facet[in.numberoffacets];
            in.facetmarkerlist = new int[in.numberoffacets];
            for(Mesh::sieve_type::heightSequence::iterator f_itor = facets->begin(); f_itor != facets->end(); ++f_itor) {
              const Mesh::coordinate_type::index_type& interval = facetBundle->getIndex(0, *f_itor);
              Obj<Mesh::ordering_type::patches_type::coneSequence> cone = bdOrdering->getPatch(*f_itor);

              in.facetlist[interval.prefix].numberofpolygons = 1;
              in.facetlist[interval.prefix].polygonlist = new tetgenio::polygon[in.facetlist[interval.prefix].numberofpolygons];
              in.facetlist[interval.prefix].numberofholes = 0;
              in.facetlist[interval.prefix].holelist = NULL;

              tetgenio::polygon *poly = in.facetlist[interval.prefix].polygonlist;
              int                c = 0;

              poly->numberofvertices = cone->size();
              poly->vertexlist = new int[poly->numberofvertices];
              for(Mesh::ordering_type::patches_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
                const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *c_itor);

                poly->vertexlist[c++] = vInterval.prefix;
              }
              in.facetmarkerlist[interval.prefix] = f_itor.getMarker();
            }
          }

          in.numberofholes = 0;
          if (createConvexHull) args += "c";
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist);
  
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
                Obj<PointSet>    join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                Obj<PointSet>    point = PointSet();
                Obj<PointSet>    edge = PointSet();
                Mesh::point_type cornerA(0, out.trifacelist[f*3+0] + out.numberoftetrahedra);
                Mesh::point_type cornerB(0, out.trifacelist[f*3+1] + out.numberoftetrahedra);
                Mesh::point_type cornerC(0, out.trifacelist[f*3+2] + out.numberoftetrahedra);
                point->insert(cornerA);
                edge->insert(cornerB);
                edge->insert(cornerC);
                Obj<PointSet>    join = topology->nJoin(point, edge, 2);

                topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
              }
            }
          }
        }
        return m;
      };
#endif
    public:
      static Obj<Mesh> generate(Obj<Mesh> boundary) {
        Obj<Mesh> mesh;
        int       dim = boundary->getDimension();

        if (dim == 1) {
#ifdef PETSC_HAVE_TRIANGLE
          mesh = generate_Triangle(boundary);
#else
          throw ALE::Exception("Mesh generation currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 2) {
#ifdef PETSC_HAVE_TETGEN
          mesh = generate_TetGen(boundary);
#else
          throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
        }
        return mesh;
      };
    private:
#ifdef PETSC_HAVE_TRIANGLE
      static Obj<Mesh> refine_Triangle(Obj<Mesh> mesh, double maxAreas[]) {
        struct triangulateio in;
        struct triangulateio out;
        int                  dim = 2;
        Obj<Mesh>            m = Mesh(mesh->getComm(), dim);
        // FIX: Need to globalize
        PetscInt             numElements = mesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt          rank;
        PetscErrorCode       ierr;

        ierr = MPI_Comm_rank(mesh->getComm(), &rank);
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
        Obj<Mesh::sieve_type> serialTopology = serialMesh->getTopology();
        Obj<Mesh::sieve_type> serialOrientation = serialMesh->getOrientation();

        if (rank == 0) {
          std::string args("pqenzQra");
          Obj<Mesh::sieve_type::heightSequence> faces = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::bundle_type>                vertexBundle = serialMesh->getBundle(0);
          Obj<Mesh::coordinate_type>            coordinates = serialMesh->getCoordinates();
          int                                   f = 0;

          in.numberofpoints = vertices->size();
          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
          for(Mesh::sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
            const Mesh::coordinate_type::index_type& interval = coordinates->getIndex(0, *v_itor);
            const Mesh::coordinate_type::value_type *array = coordinates->restrict(0, *v_itor);

            for(int d = 0; d < interval.index; d++) {
              in.pointlist[interval.prefix + d] = array[d];
            }
            const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *v_itor);
            in.pointmarkerlist[vInterval.prefix] = v_itor.getMarker();
          }

          in.numberofcorners = 3;
          in.numberoftriangles = faces->size();
          ierr = PetscMalloc(in.numberoftriangles * in.numberofcorners * sizeof(int), &in.trianglelist);
          for(Mesh::sieve_type::heightSequence::iterator f_itor = faces->begin(); f_itor != faces->end(); f_itor++) {
            Obj<Mesh::coordinate_type::IndexArray> intervals = vertexBundle->getOrderedIndices(0, serialOrientation->cone(*f_itor));
            int                                    v = 0;

            for(Mesh::coordinate_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
              in.trianglelist[f * in.numberofcorners + v++] = i_itor->prefix;
            }
            f++;
          }

          Obj<Mesh::sieve_type::depthMarkerSequence> segments = serialTopology->depthStratum(1, 1);
          Obj<Mesh::bundle_type> segmentBundle = Mesh::bundle_type();

          segmentBundle->setTopology(serialTopology);
          segmentBundle->setPatch(segments, 0);
          segmentBundle->setIndexDimensionByDepth(1, 1);
          segmentBundle->orderPatches();
          in.numberofsegments = segments->size();
          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            for(Mesh::sieve_type::depthMarkerSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
              const Mesh::coordinate_type::index_type& interval = segmentBundle->getIndex(0, *s_itor);
              Obj<Mesh::sieve_type::coneSequence>      cone = serialTopology->cone(*s_itor);
              int                                      p = 0;
        
              for(Mesh::sieve_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *c_itor);

                in.segmentlist[interval.prefix * 2 + (p++)] = vInterval.prefix;
              }
              in.segmentmarkerlist[interval.prefix] = s_itor.getMarker();
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
        m->populate(out.numberoftriangles, out.trianglelist, out.numberofpoints, out.pointlist);
        //m->distribute(m);

        // Need to make boundary

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> refine_TetGen(Obj<Mesh> mesh, double maxAreas[]) {
        ::tetgenio     in;
        ::tetgenio     out;
        int            dim = 3;
        Obj<Mesh>      m = Mesh(mesh->getComm(), dim);
        // FIX: Need to globalize
        PetscInt       numElements = mesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt    rank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(mesh->getComm(), &rank);

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
        Obj<Mesh::sieve_type> serialTopology = serialMesh->getTopology();
        Obj<Mesh::sieve_type> serialOrientation = serialMesh->getOrientation();
        Obj<Mesh::ordering_type> serialOrdering = serialMesh->getOrdering();

        if (rank == 0) {
          std::string args("qenzQra");
          Obj<Mesh::sieve_type::heightSequence> cells = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::bundle_type>                vertexBundle = serialMesh->getBundle(0);
          Obj<Mesh::coordinate_type>            coordinates = serialMesh->getCoordinates();
          int                                   c = 0;

          in.numberofpoints = vertices->size();
          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(Mesh::sieve_type::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
            const Mesh::coordinate_type::index_type& interval = coordinates->getIndex(0, *v_itor);
            const Mesh::coordinate_type::value_type *array = coordinates->restrict(0, *v_itor);

            for(int d = 0; d < interval.index; d++) {
              in.pointlist[interval.prefix + d] = array[d];
            }
            const Mesh::coordinate_type::index_type& vInterval = vertexBundle->getIndex(0, *v_itor);
            in.pointmarkerlist[vInterval.prefix] = v_itor.getMarker();
          }

          in.numberofcorners = 4;
          in.numberoftetrahedra = cells->size();
          in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
          for(Mesh::sieve_type::heightSequence::iterator c_itor = cells->begin(); c_itor != cells->end(); ++c_itor) {
            Obj<Mesh::coordinate_type::IndexArray> intervals = vertexBundle->getOrderedIndices(0, serialOrientation->cone(*c_itor));
            int                                    v = 0;

            for(Mesh::coordinate_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
              in.tetrahedronlist[c * in.numberofcorners + v++] = i_itor->prefix;
            }
            c++;
          }

          in.numberofholes = 0;
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist);
  
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
                Obj<PointSet>    join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                Obj<PointSet>    point = PointSet();
                Obj<PointSet>    edge = PointSet();
                Mesh::point_type cornerA(0, out.edgelist[f*3+0] + out.numberoftetrahedra);
                Mesh::point_type cornerB(0, out.edgelist[f*3+1] + out.numberoftetrahedra);
                Mesh::point_type cornerC(0, out.edgelist[f*3+2] + out.numberoftetrahedra);
                point->insert(cornerA);
                edge->insert(cornerB);
                edge->insert(cornerC);
                Obj<PointSet>    join = topology->nJoin(point, edge, 2);

                topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
              }
            }
          }
        }
        return m;
      };
#endif
    public:
      static Obj<Mesh> refine(Obj<Mesh> mesh, double maxArea) {
        int       numElements = mesh->getTopology()->heightStratum(0)->size();
        double   *maxAreas = new double[numElements];
        for(int e = 0; e < numElements; e++) {
          maxAreas[e] = maxArea;
        }
        Obj<Mesh> refinedMesh = refine(mesh, maxAreas);

        delete [] maxAreas;
        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, double maxAreas[]) {
        Obj<Mesh> refinedMesh;
        int       dim = mesh->getDimension();

        if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
          refinedMesh = refine_Triangle(mesh, maxAreas);
#else
          throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
          refinedMesh = refine_TetGen(mesh, maxAreas);
#else
          throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
        }
        return refinedMesh;
      };
    };
  } // namespace def
} // namespace ALE

#endif
