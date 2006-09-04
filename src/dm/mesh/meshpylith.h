#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

#include <CoSieve.hh>

namespace ALE {
  namespace PyLith {

    class Builder {
    public:
      typedef ALE::Sieve<int, int, int>                sieve_type;
      typedef ALE::New::Topology<int, sieve_type>      topology_type;
      typedef ALE::New::Section<topology_type, double> section_type;
      //typedef struct {double x, y, z;}               split_value;
      typedef ALE::Mesh::split_value                   split_value;
      typedef ALE::New::Section<topology_type, ALE::pair<sieve_type::point_type, split_value> > split_section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    protected:
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f);
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[], int *materials[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static void readSplit(MPI_Comm comm, const std::string& filename, const int dim, const bool useZeroBase, int& numSplit, int *splitInd[], double *splitValues[]);
      static void buildCoordinates(const Obj<section_type>& coords, const int embedDim, const double coordinates[]);
      static void buildMaterials(const Obj<ALE::Mesh::section_type>& matField, const int materials[]);
      static void buildSplit(const Obj<split_section_type>& splitField, int numCells, int numSplit, int splitInd[], double splitVals[]);
      static Obj<ALE::Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
    };

    class Viewer {
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(const Obj<ALE::Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeVerticesLocal(const Obj<ALE::Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(const Obj<ALE::Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeElementsLocal(const Obj<ALE::Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeSplitLocal(const Obj<ALE::Mesh>& mesh, const Obj<Builder::split_section_type>& splitField, PetscViewer viewer);
    };
  };
};

