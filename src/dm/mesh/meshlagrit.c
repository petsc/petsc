#include <petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace LaGriT {
    void Builder::readInpFile(MPI_Comm comm, const std::string& filename, const int dim, const int numCorners, int& numElements, int *vertices[], int& numVertices, double *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscScalar   *coords;
      PetscInt      *verts;
      PetscInt       commRank;
      char           buf[2048];
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);
      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      // Read header
      if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
      // Number of elements
      const char *x = strtok(buf, " ");
      numElements = atoi(x);
      // Number of vertices
      x = strtok(NULL, " ");
      numVertices = atoi(x);
      // Element type
      x = strtok(NULL, " ");
      // ???
      x = strtok(NULL, " ");
      // ???
      x = strtok(NULL, " ");
      ierr = PetscMalloc(numVertices*dim * sizeof(PetscScalar), &coords);
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
      ierr = PetscMalloc(numElements*numCorners * sizeof(PetscInt), &verts);
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
      ierr = PetscViewerDestroy(viewer);
    };
    Obj<Builder::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& filename, const bool interpolate = false, const int debug = 0) {
      Obj<Mesh>       mesh  = new Mesh(comm, dim, debug);
      Obj<sieve_type> sieve = new sieve_type(comm, debug);
      int    *cells;
      double *coordinates;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;

      Builder::readInpFile(comm, filename, dim, numCorners, numCells, &cells, numVertices, &coordinates);
      ALE::SieveBuilder<Mesh>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
      mesh->setSieve(sieve);
      mesh->stratify();
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coordinates);
      return mesh;
    };
    void Builder::readFault(Obj<Builder::Mesh> mesh, const std::string& filename) {
      PetscViewer    viewer;
      FILE          *f;
      char           buf[2048];
      PetscInt       numPsets;
      PetscErrorCode ierr;

      if (mesh->commRank() != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
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
        const Obj<Mesh::label_type>& label = mesh->createLabel(x);
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
          mesh->setValue(label, atoi(x), 1);
        }
      }
      ierr = PetscViewerDestroy(viewer);
    };
  }
}
