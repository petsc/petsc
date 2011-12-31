#include <petscdmmesh_formats.hh>   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace Bardhan {
    void Builder::readInpFile(MPI_Comm comm, const std::string& filename, const int dim, const int numCorners, int& numElements, int *vertices[], int& numVertices, PetscReal *coordinates[], PetscReal *faceNormals[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscReal     *coords;
      PetscReal     *normals;
      PetscInt      *verts;
      PetscInt       commRank;
      char           buf[2048];
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);
      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      // Read header
      if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
      // Number of vertices
      const char *x = strtok(buf, " ");
      numVertices = atoi(x);
      // Number of elements
      x = strtok(NULL, " ");
      numElements = atoi(x);
      ierr = PetscMalloc(numVertices*(dim+1) * sizeof(PetscReal), &coords);CHKERRXX(ierr);
      ierr = PetscMalloc(numVertices*(dim+1) * sizeof(PetscReal), &normals);CHKERRXX(ierr);
      for(PetscInt i = 0; i < numVertices; ++i) {
        if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
        x = strtok(buf, " ");
        for(int c = 0; c < dim+1; c++) {
          coords[i*(dim+1)+c] = atof(x);
          x = strtok(NULL, " ");
        }
        for(int c = 0; c < dim+1; c++) {
          normals[i*(dim+1)+c] = atof(x);
          x = strtok(NULL, " ");
        }
        // Ignore ???
        x = strtok(NULL, " ");
        // Ignore ???
        x = strtok(NULL, " ");
        // Ignore ???
        x = strtok(NULL, " ");
      }
      *coordinates = coords;
      *faceNormals = normals;
      ierr = PetscMalloc(numElements*numCorners * sizeof(PetscInt), &verts);CHKERRXX(ierr);
      for(PetscInt i = 0; i < numElements; ++i) {
        if (fgets(buf, 2048, f) == NULL) throw ALE::Exception("File ended prematurely");
        x = strtok(buf, " ");
        for(int c = 0; c < numCorners; c++) {
          verts[i*numCorners+c] = atoi(x) - 1;
          x = strtok(NULL, " ");
        }
        // Ignore ???
        x = strtok(NULL, " ");
        // Ignore ???
        x = strtok(NULL, " ");
      }
      *vertices = verts;
      ierr = PetscViewerDestroy(&viewer);
    };
    Obj<Builder::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& filename, const bool interpolate = false, const int debug = 0) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      Obj<Mesh>       mesh  = new Mesh(comm, dim, debug);
      Obj<sieve_type> sieve = new sieve_type(comm, debug);
      Obj<FlexMesh>             m = new FlexMesh(comm, dim, debug);
      Obj<FlexMesh::sieve_type> s = new FlexMesh::sieve_type(comm, debug);
      std::map<Mesh::point_type,Mesh::point_type> renumbering;
      int       *cells;
      PetscReal *coordinates, *normals;
      int        numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode ierr;

      mesh->setSieve(sieve);
      Builder::readInpFile(comm, filename, dim, numCorners, numCells, &cells, numVertices, &coordinates, &normals);
      ALE::SieveBuilder<FlexMesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
      m->setSieve(s);
      m->stratify();
      ALE::SieveBuilder<FlexMesh>::buildCoordinates(m, dim+1, coordinates);
      ierr = PetscFree(cells);CHKERRXX(ierr);
      ierr = PetscFree(coordinates);CHKERRXX(ierr);
      ierr = PetscFree(normals);CHKERRXX(ierr);
      ALE::ISieveConverter::convertMesh(*m, *mesh, renumbering, false);
      return mesh;
    };
  }
}
