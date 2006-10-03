#ifndef included_ALE_PCICE_hh
#define included_ALE_PCICE_hh

#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace PCICE {
    class Builder {
    public:
      typedef ALE::Sieve<int, int, int>                sieve_type;
      typedef ALE::New::Topology<int, sieve_type>      topology_type;
      typedef ALE::New::Section<topology_type, double> section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase, const bool interpolate, const int debug);
      static void readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename, const int numBdFaces, const int numBdVertices);
      static void outputVerticesLocal(const Obj<Mesh>& mesh, int *numVertices, int *dim, double *coordinates[], bool columnMajor);
      static void outputElementsLocal(const Obj<Mesh>& mesh, int *numElements, int *numCorners, int *vertices[], bool columnMajor);
    };

    typedef struct {
      Mesh::point_type                    vertex;
      Mesh::real_section_type::value_type veln_x;
      Mesh::real_section_type::value_type veln_y;
      Mesh::real_section_type::value_type pn;
      Mesh::real_section_type::value_type tn;
    } RestartType;

    class Viewer {
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeRestart(const Obj<Mesh>& mesh, PetscViewer viewer);
    };
  };
};

#endif
