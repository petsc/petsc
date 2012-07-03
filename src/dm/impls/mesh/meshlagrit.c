#include <petscdmmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#if 0

void FlipCellOrientation(pylith::int_array * const cells, const int numCells, const int numCorners, const int meshDim) {
  if (3 == meshDim && 4 == numCorners) {
    for(int iCell = 0; iCell < numCells; ++iCell) {
      const int i1 = iCell*numCorners+1;
      const int i2 = iCell*numCorners+2;
      const int tmp = (*cells)[i1];
      (*cells)[i1] = (*cells)[i2];
      (*cells)[i2] = tmp;
    }
  }
}
      //Obj<PETSC_MESH_TYPE> m = ALE::LaGriT::Builder::readMesh(comm, 3, options->baseFilename, options->interpolate, options->debug);'
      Obj<PETSC_MESH_TYPE>             m     = new PETSC_MESH_TYPE(comm, options->dim, options->debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm, options->debug);
      bool                 flipEndian = false;
      int                  dim;
      pylith::int_array    cells;
      pylith::double_array coordinates;
      pylith::int_array    materialIds;
      int                  numCells = 0, numVertices = 0, numCorners = 0;

      if (!m->commRank()) {
        if (pylith::meshio::GMVFile::isAscii(options->baseFilename)) {
          pylith::meshio::GMVFileAscii filein(options->baseFilename);
          filein.read(&coordinates, &cells, &materialIds, &dim, &dim, &numVertices, &numCells, &numCorners);
          if (options->interpolate) {
            FlipCellOrientation(&cells, numCells, numCorners, dim);
          }
        } else {
          pylith::meshio::GMVFileBinary filein(options->baseFilename, flipEndian);
          filein.read(&coordinates, &cells, &materialIds, &dim, &dim, &numVertices, &numCells, &numCorners);
          if (!options->interpolate) {
            FlipCellOrientation(&cells, numCells, numCorners, dim);
          }
        } // if/else
      }
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildTopology(sieve, dim, numCells, const_cast<int*>(&cells[0]), numVertices, options->interpolate, numCorners, -1, m->getArrowSection("orientation"));
      m->setSieve(sieve);
      m->stratify();
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(m, dim, const_cast<double*>(&coordinates[0]));

      ierr = DMMeshCreate(comm, &mesh);CHKERRQ(ierr);
      ierr = DMMeshSetMesh(mesh, m);CHKERRQ(ierr);
      ierr = DMMeshIDBoundary(mesh);CHKERRQ(ierr);
#endif

namespace ALE {
  namespace LaGriT {
    void Builder::readInpFile(MPI_Comm comm, const std::string& filename, const int dim, const int numCorners, int& numElements, int *vertices[], int& numVertices, PetscReal *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscReal     *coords;
      PetscInt      *verts;
      PetscInt       commRank;
      char           buf[2048];
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);CHKERRXX(ierr);
      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRXX(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRXX(ierr);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());CHKERRXX(ierr);
      ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRXX(ierr);
      // Read header
      if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
      // Number of vertices
      const char *x = strtok(buf, " ");
      numVertices = atoi(x);
      // Number of elements
      x = strtok(NULL, " ");
      numElements = atoi(x);
      // Element type
      x = strtok(NULL, " ");
      // ???
      x = strtok(NULL, " ");
      // ???
      x = strtok(NULL, " ");
      ierr = PetscMalloc(numVertices*dim * sizeof(PetscReal), &coords);CHKERRXX(ierr);
      for(PetscInt i = 0; i < numVertices; ++i) {
        if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
        x = strtok(buf, " ");
        // Ignore vertex number
        x = strtok(NULL, " ");
        for(int c = 0; c < dim; c++) {
          coords[i*dim+c] = atof(x);
          x = strtok(NULL, " ");
        }
      }
      *coordinates = coords;
      ierr = PetscMalloc(numElements*numCorners * sizeof(PetscInt), &verts);CHKERRXX(ierr);
      for(PetscInt i = 0; i < numElements; ++i) {
        if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
        x = strtok(buf, " ");
        // Ignore element number
        x = strtok(NULL, " ");
        // Ignore ??? (material)
        x = strtok(NULL, " ");
        // Ignore element type
        x = strtok(NULL, " ");
        for(int c = 0; c < numCorners; c++) {
          verts[i*numCorners+c] = atoi(x) - 1;
          x = strtok(NULL, " ");
        }
      }
      *vertices = verts;
      ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
    };
    Obj<Builder::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& filename, const bool interpolate = false, const int debug = 0) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      Obj<Mesh>       mesh  = new Mesh(comm, dim, debug);
      Obj<sieve_type> sieve = new sieve_type(comm, debug);
      Obj<FlexMesh>             m = new FlexMesh(comm, dim, debug);
      Obj<FlexMesh::sieve_type> s = new FlexMesh::sieve_type(comm, debug);
      std::map<Mesh::point_type,Mesh::point_type> renumbering;
      int       *cells;
      PetscReal *coordinates;
      int        numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode ierr;

      mesh->setSieve(sieve);
      Builder::readInpFile(comm, filename, dim, numCorners, numCells, &cells, numVertices, &coordinates);
      ALE::SieveBuilder<FlexMesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
      m->setSieve(s);
      m->stratify();
      ALE::SieveBuilder<FlexMesh>::buildCoordinates(m, dim+1, coordinates);
      ierr = PetscFree(cells);CHKERRXX(ierr);
      ierr = PetscFree(coordinates);CHKERRXX(ierr);
      ALE::ISieveConverter::convertMesh(*m, *mesh, renumbering, false);
      return mesh;
    };
    void Builder::readFault(Obj<Builder::Mesh> mesh, const std::string& filename) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
#if 0
    void Builder::readFault(Obj<Builder::Mesh> mesh, const std::string& filename) {
      const int      numCells = mesh->heightStratum(0)->size();
      PetscViewer    viewer;
      FILE          *f;
      char           buf[2048];
      PetscInt       numPsets;
      PetscErrorCode ierr;

      if (mesh->commRank() != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRXX(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRXX(ierr);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());CHKERRXX(ierr);
      ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRXX(ierr);
      // Read header
      if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
      // Check file type
      const char *x = strtok(buf, " ");
      std::string fileType("pset");
      if (fileType != x) throw ALE::Exception("Invalid file type");
      // Ignore set type
      x = strtok(NULL, " ");
      // Number of psets
      x = strtok(NULL, " ");
      numPsets = atoi(x);
      for(PetscInt p = 0; p < numPsets; ++p) {
        if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
        // Read name
        x = strtok(buf, " ");
        const Obj<Mesh::int_section_type>& fault = mesh->getIntSection(x);
        // Vertices per line
        x = strtok(NULL, " ");
        const PetscInt vertsPerLine = atoi(x);
        // Total vertices
        x = strtok(NULL, " ");
        const PetscInt totalVerts = atoi(x);

        for(PetscInt v = 0; v < totalVerts; ++v) {
          if (v%vertsPerLine == 0) {
            if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
            x = strtok(buf, " ");
          } else {
            x = strtok(NULL, " ");
          }
          const int vv = atoi(x) + numCells - 1;

          fault->setFiberDimension(vv, 1);
        }
        fault->allocatePoint();
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
    };
#endif
  }
}
