#ifndef included_ALE_Mesh_PyLith_hh
#define included_ALE_Mesh_PyLith_hh

#include "private/meshimpl.h"   /*I      "petscmesh.h"   I*/

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
      const Obj<ALE::Mesh>& mesh;
      ParserState  state;
      double      *coords;   // The vertex coordinates
      unsigned int embedDim; // The embedding dimension
    public:
      XMLMesh(const Obj<ALE::Mesh>& mesh) : XMLObject(), mesh(mesh), state(OUTSIDE), coords(NULL) {};
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
  }
}

#endif // PETSC_HAVE_LIBXML2

#endif
