#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

#include <CoSieve.hh>

namespace ALE {
  namespace PyLith {
    struct vertexOutput {
      ALE::Obj<ALE::Mesh::field_type> coordinates;
      PetscViewer viewer;
      int dim;
    public:
      vertexOutput(PetscViewer viewer, Obj<ALE::Mesh::field_type> coordinates, int dim);

      bool operator()(const ALE::Mesh::point_type& p) const;
    };
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
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, bool useZeroBase, int& numElements, int *vertices[], int *materials[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
      static void readSplit(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numSplit, int *splitInd[], double *splitValues[]);
      static void createMaterialField(int numElements, int materials[], Obj<Mesh> mesh, Obj<Mesh::field_type> matField);
      static void createSplitField(int numSplit, int splitInd[], double splitVals[], Obj<Mesh> mesh, Obj<Mesh::field_type> splitField);
      static Obj<ALE::Mesh> createNew(MPI_Comm comm, const std::string& baseFilename, bool interpolate, int debug);
      static void buildCoordinates(Obj<section_type> coords, const int embedDim, const double coordinates[]);
      static void buildMaterials(Obj<Mesh::section_type> matField, int materials[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, std::string basename, bool useZeroBase, bool interpolate, const int debug);
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

