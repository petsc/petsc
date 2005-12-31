#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSifter_hh
#include <CoSifter.hh>
#endif

namespace ALE {

  namespace def {
    // A computational mesh
    class Mesh {
      Sieve<Point,int> topology;
      Sieve<Point,int> orientation;
      CoSieve<Sieve<Point,int>, int, Point, double> coordinates;
      MPI_Comm         comm;
      int              dim;
      int              debug;

    public:
      Mesh(MPI_Comm c, int dimension) : comm(c), dim(dimension), debug(0) {};

      Obj<Sieve<Point,int> > getTopology() {return this->topology;};
      Obj<Sieve<Point,int> > getOrientation() {return this->orientation;};

      void buildFaces(int dim, std::map<int, int*> *curSimplex, Obj<PointSet> boundary, Point& simplex);
      void buildTopology(int numSimplices, int simplices[], int numVertices);
      void createSerialCoordinates(int numElements, double coords[]);
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

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename) {
        int       dim = 3;
        bool      useZeroBase = false;
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

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

      static Obj<Mesh> create(MPI_Comm comm, const std::string& baseFilename, int dim, bool useZeroBase = false) {
        Obj<Mesh> mesh = Mesh(comm, dim);
        int      *vertices;
        double   *coordinates;
        int       numElements, numVertices;

        readConnectivity(comm, baseFilename+".lcon", dim, useZeroBase, numElements, &vertices);
        readCoordinates(comm, baseFilename+".nodes", dim, numVertices, &coordinates);
        mesh->populate(numElements, vertices, numVertices, coordinates);
        return mesh;
      };
    };
  } // namespace def
} // namespace ALE

#endif
