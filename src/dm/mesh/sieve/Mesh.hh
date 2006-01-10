#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif

namespace ALE {

  namespace def {
    // A computational mesh
    class Mesh {
    public:
      typedef Sieve<Point,int> sieve_type;
      typedef CoSieve<sieve_type, int, Point, double> coordinate_type;
      typedef CoSieve<sieve_type, int, Point, int> bundle_type;
      int debug;
    private:
      sieve_type      topology;
      sieve_type      orientation;
      coordinate_type coordinates;
      coordinate_type boundary;
      MPI_Comm        comm;
      int             dim;
    public:
      Mesh(MPI_Comm c, int dimension) : debug(0), comm(c), dim(dimension) {};

      MPI_Comm             getComm() {return this->comm;};
      Obj<sieve_type>      getTopology() {return this->topology;};
      Obj<sieve_type>      getOrientation() {return this->orientation;};
      int                  getDimension() {return this->dim;};
      Obj<coordinate_type> getCoordinates() {return this->coordinates;};
      Obj<coordinate_type> getBoundary() {return this->boundary;};

      void buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, Point& simplex);
      void buildTopology(int numSimplices, int simplices[], int numVertices);
      void createSerialCoordinates(int numElements, double coords[]);
      void createBoundary(int numBoundaryVertices, int numBoundaryComponents, int boundaryVertices[], double boundaryValues[]);
      void populate(int numSimplices, int simplices[], int numVertices, double coords[]);
    };

    // Creation
    class PyLithBuilder {
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f) {
        while((fgets(buf, bufSize, f) != NULL) && (buf[0] == '#')) {}
      };

      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
    public:
      PyLithBuilder() {};
      virtual ~PyLithBuilder() {};

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, int debug = 0) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".connect", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".coord", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
        return mesh;
      };
    };

    class PCICEBuilder {
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int dim, bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, int dim, int& numVertices, double *coordinates[]);
    public:
      PCICEBuilder() {};
      virtual ~PCICEBuilder() {};

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false, int debug = 0) {
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        mesh->debug = debug;
        readConnectivity(comm, baseFilename+".lcon", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
        return mesh;
      };
    };
  } // namespace def
} // namespace ALE

#endif
