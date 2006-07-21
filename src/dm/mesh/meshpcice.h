#ifndef included_ALE_PCICE_hh
#define included_ALE_PCICE_hh

#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace PCICE {
    class Builder {
    public:
      Builder() {};
      virtual ~Builder() {};
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
      static Obj<ALE::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase, bool interpolate, int debug);
      static Obj<ALE::Mesh> createNewBd(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase, int debug);
    };

    class Viewer {
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer);
    };
  };
};

#endif
