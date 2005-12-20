#include <Sifter.hh>
#include <petscmesh.h>

namespace ALE {
  namespace def {

    void Mesh::buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, Point& simplex) {
      Obj<PointSet> faces = PointSet();

      std::cout << "  Building faces for boundary(size " << boundary->size() << "), dim " << dim << std::endl;
      if (dim > 1) {
        // Use the cone construction
        for(PointSet::iterator b_itor = boundary->begin(); b_itor != boundary->end(); ++b_itor) {
          Obj<PointSet> faceBoundary = PointSet();
          Point         face;

          faceBoundary.copy(boundary);
          std::cout << "    boundary point " << *b_itor << std::endl;
          faceBoundary->erase(*b_itor);
          this->buildFaces(dim-1, curSimplex, faceBoundary, face);
          faces->insert(face);
        }
      } else {
        std::cout << "  Just set faces to boundary in 1d" << std::endl;
        faces = boundary;
      }
      for(PointSet::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
        std::cout << "  face point " << *f_itor << std::endl;
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
        std::cout << "  Found old simplex " << simplex << std::endl;
      } else {
        simplex = Point(0, (*(*curSimplex)[dim])++);
        this->topology.addCone(faces, simplex);
        std::cout << "  Added simplex " << simplex << " dim " << dim << std::endl;
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

          std::cout << "Adding boundary node " << vertex << std::endl;
          boundary->insert(vertex);
        }
        std::cout << "simplex " << s << " boundary size " << boundary->size() << std::endl;
        this->buildFaces(this->dim, &curElement, boundary, simplex);
        // Orient the simplex
        Point element(0, simplices[s*(dim+1)+0]+numSimplices);
        cellTuple->clear();
        cellTuple->insert(element);
        for(int b = 1; b < dim+1; b++) {
          Point next(0, simplices[s*(dim+1)+b]+numSimplices);
          Obj<PointSet> join = this->topology.nJoin(element, next, b);

          if (join->size() == 0) {
            std::cout << "element " << element << " next " << next << std::endl;
            throw ALE::Exception("Invalid join");
          }
          element =  *join->begin();
          cellTuple->insert(element);
        }
        this->orientation.addCone(cellTuple, simplex);
      }
    };

    void Mesh::populate(int numSimplices, int simplices[], int numVertices, double coords[]) {
      PetscMPIInt rank;

      MPI_Comm_rank(this->comm, &rank);
      //         if (debug) {
      //           topology->setVerbosity(11);
      //           orientation->setVerbosity(11);
      //           boundary->setVerbosity(11);
      //         }
      /* Create serial sieve */
      this->topology.setStratification(false);
      if (rank == 0) {
        this->buildTopology(numSimplices, simplices, numVertices);
      }
      //         if (debug) {
      //           topology->view("Serial Simplicializer topology");
      //           orientation->view("Serial Simplicializer orientation");
      //         }
      this->topology.stratify();
      this->topology.setStratification(true);
      //createSerialCoordinates(numElements, coords);
    };

    void PyLithBuilder::readConnectivityPyLith(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       maxCells = 1024, cellCount = 0;
      PetscInt      *verts;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      MPI_Comm_rank(comm, &commRank);
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
      this->ignoreCommentsPyLith(buf, 2048, f);
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

    void PyLithBuilder::readCoordinatesPyLith(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]) {
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
        this->ignoreCommentsPyLith(buf, 2048, f);
        ierr = PetscMalloc(maxVerts*dim * sizeof(PetscScalar), &coords);
        /* Ignore comments */
        this->ignoreCommentsPyLith(buf, 2048, f);
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

    Obj<Mesh> PyLithBuilder::createPyLith(MPI_Comm comm, const std::string& baseFilename) {
      int       dim = 3;
      bool      useZeroBase = false;
      Obj<Mesh> mesh = Mesh(comm, dim);
      int      *vertices;
      double   *coordinates;
      int       numElements, numVertices;

      this->readConnectivityPyLith(comm, baseFilename+".connect", dim, useZeroBase, numElements, &vertices);
      this->readCoordinatesPyLith(comm, baseFilename+".coord", dim, numVertices, &coordinates);
      mesh->populate(numElements, vertices, numVertices, coordinates);
      return mesh;
    };
  }
}
