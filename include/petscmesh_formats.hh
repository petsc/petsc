#if !defined(__PETSCMESH_FORMATS_HH)
#define __PETSCMESH_FORMATS_HH

#include <petscmesh.hh>

namespace ALE {
  namespace PyLith {

    class Builder {
    public:
      typedef PETSC_MESH_TYPE               Mesh;
      typedef Mesh::sieve_type        sieve_type;
      typedef Mesh::real_section_type real_section_type;
      typedef Mesh::int_section_type  int_section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    protected:
      static inline void ignoreComments(char *buf, PetscInt bufSize, FILE *f);
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[], int *materials[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static void readSplit(MPI_Comm comm, const std::string& filename, const int dim, const bool useZeroBase, int& numSplit, int *splitInd[], int *loadHistory[], double *splitValues[]);
      static void readTractions(MPI_Comm comm, const std::string& filename, const int dim, const int& corners, const bool useZeroBase, int& numTractions, int& vertsPerFace, int *tractionVertices[], double *tractionValues[]);
      static void buildMaterials(const Obj<Mesh>& mesh, const Obj<int_section_type>& matField, const int materials[]);
#if 0
      static void buildSplit(const Obj<pair_section_type>& splitField, const Obj<int_section_type>& loadField, int numCells, int numSplit, int splitInd[], int loadHist[], double splitVals[]);
#endif
      static void buildTractions(const Obj<real_section_type>& tractionField, const Obj<Mesh>& boundaryMesh, int numCells, int numTractions, int vertsPerFace, int tractionVertices[], double tractionValues[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
#if 0
      static Obj<pair_section_type> createSplit(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase);
#endif
      static Obj<Mesh> createTraction(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase);
      static void createCohesiveElements(const Obj<Mesh>& mesh, const std::set<Mesh::point_type>& faultVertices);
    };

    class Viewer {
    public:
      typedef PETSC_MESH_TYPE Mesh;
    public:
      Viewer() {};
      virtual ~Viewer() {};
    public:
      static PetscErrorCode writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer);
      static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer);
      static PetscErrorCode writeElementsLocal(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer);
#if 0
      static PetscErrorCode writeSplitLocal(const Obj<Mesh>& mesh, const Obj<Builder::pair_section_type>& splitField, PetscViewer viewer);
#endif
      static PetscErrorCode writeTractionsLocal(const Obj<Mesh>& mesh, const Obj<Mesh>& tractionMesh, const Obj<Builder::real_section_type>& tractionField, PetscViewer viewer);
    };
  };
};

namespace ALE {
  namespace LaGriT {
    class Builder {
    public:
      typedef PETSC_MESH_TYPE             Mesh;
      typedef PETSC_MESH_TYPE::sieve_type sieve_type;
    public:
      static void readInpFile(MPI_Comm comm, const std::string& filename, const int dim, const int numCorners, int& numElements, int *vertices[], int& numVertices, double *coordinates[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& filename, const bool interpolate, const int debug);
      static void readFault(Obj<Mesh> mesh, const std::string& filename);
    };
  }
}

namespace ALE {
  namespace Bardhan {
    class Builder {
    public:
      typedef PETSC_MESH_TYPE             Mesh;
      typedef PETSC_MESH_TYPE::sieve_type sieve_type;
    public:
      static void readInpFile(MPI_Comm comm, const std::string& filename, const int dim, const int numCorners, int& numElements, int *vertices[], int& numVertices, double *coordinates[], double *faceNormals[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& filename, const bool interpolate, const int debug);
      static void readFault(Obj<Mesh> mesh, const std::string& filename);
    };
  }
}

namespace ALE {
  namespace PCICE {
    void fuseBoundary(const ALE::Obj<PETSC_MESH_TYPE>& mesh);

    class Builder {
    public:
      typedef PETSC_MESH_TYPE               Mesh;
      typedef Mesh::sieve_type        sieve_type;
      typedef Mesh::real_section_type section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase, const bool interpolate, const int debug);
      static void readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename);
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
      typedef PETSC_MESH_TYPE Mesh;
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

#ifdef PETSC_HAVE_LIBXML2

#include <libxml/parser.h>

namespace ALE {
  namespace Dolfin {
    class XMLObject {
    public:
      XMLObject() {};
      virtual ~XMLObject() {};
    public:
      /// Callback for start of XML element
      virtual void startElement(const xmlChar* name, const xmlChar** attrs) = 0;
      /// Callback for end of XML element
      virtual void endElement(const xmlChar* name) = 0;
      /// Callback for start of XML file (optional)
      virtual void open(const std::string& filename) {};
      /// Callback for end of XML file, should return true iff data is ok (optional)
      virtual bool close() {return true;};
    protected:
      int parseInt(const xmlChar* name, const xmlChar** attrs, const char *attribute);
      unsigned int parseUnsignedInt(const xmlChar* name, const xmlChar** attrs, const char *attribute);
      double parseReal(const xmlChar* name, const xmlChar** attrs, const char* attribute);
      std::string parseString(const xmlChar* name, const xmlChar** attrs, const char* attribute);
      bool parseBool(const xmlChar* name, const xmlChar** attrs, const char* attribute);
      void error(std::string msg, ...);
    };
    class XMLMesh : public XMLObject {
    private:
      enum ParserState {OUTSIDE, INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS, DONE};
    private:
      const Obj<PETSC_MESH_TYPE>& mesh;
      ParserState  state;
      double      *coords;   // The vertex coordinates
      unsigned int embedDim; // The embedding dimension
      unsigned int numCells; // The number of cells (co-dim 0)
    public:
      XMLMesh(const Obj<PETSC_MESH_TYPE>& mesh) : XMLObject(), mesh(mesh), state(OUTSIDE), coords(NULL), embedDim(0), numCells(0) {};
      virtual ~XMLMesh() {};
    public:    
      void startElement (const xmlChar* name, const xmlChar** attrs);
      void endElement   (const xmlChar* name);
      void open(std::string filename) {};
      bool close() {return state == DONE;};
    private:
      void readMesh        (const xmlChar* name, const xmlChar** attrs);
      void readVertices    (const xmlChar* name, const xmlChar** attrs);
      void readCells       (const xmlChar* name, const xmlChar** attrs);
      void readVertex      (const xmlChar* name, const xmlChar** attrs);
      void readInterval    (const xmlChar* name, const xmlChar** attrs);
      void readTriangle    (const xmlChar* name, const xmlChar** attrs);
      void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
    
      void closeMesh();
    };
    class Builder {
    protected:
      static void sax_start_document(void *ctx) {};
      static void sax_end_document(void *ctx) {};
      static void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs) {
        ((ALE::Dolfin::XMLObject *) ctx)->startElement(name, attrs);
      };
      static void sax_end_element(void *ctx, const xmlChar *name) {
        ((ALE::Dolfin::XMLObject *) ctx)->endElement(name);
      };
      static void sax_warning(void *ctx, const char *msg, ...) {
        char buffer[2048];
        va_list args;

        va_start(args, msg);
        vsnprintf(buffer, 2048, msg, args);
        std::cout << "Incomplete XML data: " << buffer << std::endl;
        va_end(args);
      };
      static void sax_error(void *ctx, const char *msg, ...) {
        char buffer[2048];
        va_list args;

        va_start(args, msg);
        vsnprintf(buffer, 2048, msg, args);
        std::cerr << "Incomplete XML data: " << buffer << std::endl;
        va_end(args);
      };
      static void sax_fatal_error(void *ctx, const char *msg, ...) {
        char buffer[2048];
        va_list args;

        va_start(args, msg);
        vsnprintf(buffer, 2048, msg, args);
        std::cerr << "Illegal XML data: " << buffer << std::endl;
        va_end(args);
      };
    public:
      static void parseSAX(const std::string& filename, ALE::Dolfin::XMLObject *xmlObject) {
        // Set up the sax handler. Note that it is important that we initialise
        // all (24) fields, even the ones we don't use!
        xmlSAXHandler sax = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
        // Set up handlers for parser events
        sax.startDocument = sax_start_document;
        sax.endDocument   = sax_end_document;
        sax.startElement  = sax_start_element;
        sax.endElement    = sax_end_element;
        sax.warning       = sax_warning;
        sax.error         = sax_error;
        sax.fatalError    = sax_fatal_error;

        // Parse file
        xmlSAXUserParseFile(&sax, (void *) xmlObject, filename.c_str());
      };
      static void readMesh(const Obj<PETSC_MESH_TYPE>& mesh, const std::string& filename) {
        ALE::Dolfin::XMLObject *xmlObject;

        xmlObject = new ALE::Dolfin::XMLMesh(mesh);
        xmlObject->open(filename);
        // Parse file using the SAX interface
        parseSAX(filename, xmlObject);
        if (!xmlObject->close()) {std::cerr << "Unable to find data in XML file." << std::endl;};
        delete xmlObject;
      };
    };
  }
}

#endif // PETSC_HAVE_LIBXML2

namespace ALE {
  namespace PFLOTRAN {
    void fuseBoundary(const ALE::Obj<PETSC_MESH_TYPE>& mesh);

    class Builder {
    public:
      typedef PETSC_MESH_TYPE               Mesh;
      typedef Mesh::sieve_type        sieve_type;
      typedef Mesh::real_section_type section_type;
    public:
      Builder() {};
      virtual ~Builder() {};
    public:
      static void readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]);
      static void readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase, const bool interpolate, const int debug);
      static Obj<Mesh> readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase, const bool interpolate, const int debug);
      static void readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename);
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
      typedef PETSC_MESH_TYPE Mesh;
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

#endif // __PETSCMESH_FORMATS_HH
