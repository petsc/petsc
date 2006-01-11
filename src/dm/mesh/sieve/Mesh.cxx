#include <Mesh.hh>
#include <petscmesh.h>

namespace ALE {
  namespace def {

    void Mesh::buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, Point& simplex) {
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

    // Build a topology from a connectivity description
    //   (0, 0)            ... (0, numSimplices-1):  dim-dimensional simplices
    //   (0, numSimplices) ... (0, numVertices):     vertices
    // The other simplices are numbered as they are requested
    void Mesh::buildTopology(int numSimplices, int simplices[], int numVertices) {
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
    };

    void Mesh::createSerialCoordinates(int numElements, double coords[]) {
      int dim = this->dim;

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
    };

    // I think that the boundary shuold be marked in the Sieve
    //   It could be done with point markers like depth/height, but is that right?
    // Should also put in boundary edges
    void Mesh::createBoundary(int numBoundaryVertices, int numBoundaryComponents, int boundaryVertices[], double boundaryValues[]) {
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
    }

    void Mesh::populate(int numSimplices, int simplices[], int numVertices, double coords[]) {
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
      std::cout << "Before coordinates" << std::endl;
      this->createSerialCoordinates(numSimplices, coords);
      std::cout << "After coordinates" << std::endl;
    };

    void PyLithBuilder::readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]) {
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

    void PyLithBuilder::readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]) {
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

    void PCICEBuilder::readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]) {
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

    void PCICEBuilder::readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]) {
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
  }
}
