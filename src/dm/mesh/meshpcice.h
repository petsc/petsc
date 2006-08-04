#ifndef included_ALE_PCICE_hh
#define included_ALE_PCICE_hh

#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace PCICE {
    class Builder {
    public:
      typedef ALE::Sieve<int, int, int>             sieve_type;
      typedef ALE::New::Topology<int, sieve_type>   topology_type;
      typedef ALE::New::Atlas<topology_type, Point> atlas_type;
      typedef ALE::New::Section<atlas_type, double> section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static void buildCoordinates(const Obj<section_type>& coords, const int embedDim, const double coordinates[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
      static Obj<Mesh> createNewBd(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase, int debug);
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
