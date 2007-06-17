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
      unsigned int numCells; // The number of cells (co-dim 0)
    public:
      XMLMesh(const Obj<ALE::Mesh>& mesh) : XMLObject(), mesh(mesh), state(OUTSIDE), coords(NULL), embedDim(0), numCells(0) {};
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
      static void readMesh(const Obj<ALE::Mesh>& mesh, const std::string& filename) {
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

#endif
