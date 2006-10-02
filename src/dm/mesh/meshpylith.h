#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

#include <CoSieve.hh>

namespace ALE {
  namespace PyLith {

    class Builder {
    public:
      typedef ALE::Mesh::topology_type     topology_type;
      typedef topology_type::sieve_type    sieve_type;
      typedef ALE::Mesh::real_section_type real_section_type;
      typedef ALE::Mesh::int_section_type  int_section_type;
      typedef ALE::Mesh::pair_section_type pair_section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    protected:
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f);
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[], int *materials[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static void readSplit(MPI_Comm comm, const std::string& filename, const int dim, const bool useZeroBase, int& numSplit, int *splitInd[], double *splitValues[]);
      static void readTractions(MPI_Comm comm, const std::string& filename, const int dim, const int& corners, const bool useZeroBase, int& numTractions, int& vertsPerFace, int *tractionVertices[], double *tractionValues[]);
      static void buildMaterials(const Obj<int_section_type>& matField, const int materials[]);
      static void buildSplit(const Obj<pair_section_type>& splitField, int numCells, int numSplit, int splitInd[], double splitVals[]);
      static void buildTractions(const Obj<real_section_type>& tractionField, const Obj<topology_type>& boundaryTopology, int numCells, int numTractions, int vertsPerFace, int tractionVertices[], double tractionValues[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
      static Obj<pair_section_type> createSplit(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase);
      static Obj<real_section_type> createTraction(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase);
    };

    class Viewer {
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer);
      static PetscErrorCode writeElementsLocal(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer);
      static PetscErrorCode writeSplitLocal(const Obj<Mesh>& mesh, const Obj<Builder::pair_section_type>& splitField, PetscViewer viewer);
      static PetscErrorCode writeTractionsLocal(const Obj<Mesh>& mesh, const Obj<Builder::real_section_type>& tractionField, PetscViewer viewer);
    };
  };
};

