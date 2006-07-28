#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

#include <CoSieve.hh>

namespace ALE {
  namespace PyLith {
    class Builder {
    public:
      typedef ALE::Sieve<Point, int, int>           sieve_type;
      typedef ALE::New::Topology<int, sieve_type>   topology_type;
      typedef ALE::New::Atlas<topology_type, Point> atlas_type;
      typedef ALE::New::Section<atlas_type, double> section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    protected:
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f);
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[], int *materials[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static void readSplit(MPI_Comm comm, const std::string& filename, const int dim, const bool useZeroBase, int& numSplit, int *splitInd[], double *splitValues[]);
      static void createSplitField(int numSplit, int splitInd[], double splitVals[], Obj<Mesh> mesh, Obj<Mesh::field_type> splitField);
      static void buildCoordinates(const Obj<section_type>& coords, const int embedDim, const double coordinates[]);
      static void buildMaterials(const Obj<Mesh::section_type>& matField, const int materials[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
    };

    class Viewer {
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(Obj<ALE::Mesh> mesh, PetscViewer viewer);
      static PetscErrorCode writeVerticesLocal(Obj<ALE::Mesh> mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(Obj<ALE::Mesh> mesh, PetscViewer viewer);
      static PetscErrorCode writeElementsLocal(Obj<ALE::Mesh> mesh, PetscViewer viewer);
      static PetscErrorCode writeSplitLocal(Obj<ALE::Mesh> mesh, PetscViewer viewer);
    };
  };
};

