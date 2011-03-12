#include <petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#ifdef PETSC_HAVE_LIBXML2

namespace ALE {
  namespace Dolfin {
    void XMLObject::error(std::string msg, ...) {
      static char buffer[2048];
      va_list aptr;

      va_start(aptr, msg);
      vsnprintf(buffer, 2048, msg.c_str(), aptr);
      va_end(aptr);
      std::cerr << buffer << std::endl;
    }
    int XMLObject::parseInt(const xmlChar* name, const xmlChar** attrs, const char* attribute) {
      // Check that we got the data
      if (!attrs) error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      // Parse data
      for (uint i = 0; attrs[i]; i++) {
        // Check for attribute
        if (xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0) {
          if (!attrs[i+1]) error("Value for attribute \"%s\" of <%s> missing in XML file.", attribute, name);
          return atoi((const char *) (attrs[i+1]));
        }
      }
      // Didn't get the value
      error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      return 0;
    };
    unsigned int XMLObject::parseUnsignedInt(const xmlChar* name, const xmlChar** attrs, const char* attribute) {
      // Check that we got the data
      if (!attrs) error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      // Parse data
      for (uint i = 0; attrs[i]; i++) {
        // Check for attribute
        if (xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0) {
          if (!attrs[i+1]) error("Value for attribute \"%s\" of <%s> missing in XML file.", attribute, name);
          int value = atoi((const char *) (attrs[i+1]));

          if (value < 0) error("Value for attribute \"%s\" of <%s> is negative.", attribute, name);
          return static_cast<uint>(value);
        }
      }
      // Didn't get the value
      error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      return 0;
    };
    double XMLObject::parseReal(const xmlChar* name, const xmlChar** attrs, const char* attribute) {
      // Check that we got the data
      if (!attrs) error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      // Parse data
      for (uint i = 0; attrs[i]; i++) {
        // Check for attribute
        if (xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0) {
          if (!attrs[i+1]) error("Value for attribute \"%s\" of <%s> missing in XML file.", attribute, name);
          return atof((const char *) (attrs[i+1]));
        }
      }
      // Didn't get the value
      error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      return 0.0;
    };
    std::string XMLObject::parseString(const xmlChar* name, const xmlChar** attrs, const char* attribute) {
      // Check that we got the data
      if (!attrs) error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      // Parse data
      for (uint i = 0; attrs[i]; i++) {
        // Check for attribute
        if (xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0) {
          if (!attrs[i+1]) error("Value for attribute \"%s\" of <%s> missing in XML file.", attribute, name);
          return (const char *) (attrs[i+1]);
        }
      }
      // Didn't get the value
      error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      return "";
    };
    bool XMLObject::parseBool(const xmlChar* name, const xmlChar** attrs, const char* attribute) {
      // Check that we got the data
      if (!attrs) error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      // Parse data
      for (uint i = 0; attrs[i]; i++) {
        // Check for attribute
        if (xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0) {
          if (!attrs[i+1]) error("Value for attribute \"%s\" of <%s> missing in XML file.", attribute, name);
          std::string value = (const char *) (attrs[i+1]);
          if (strcmp(value.c_str(), "true") == 0 or strcmp(value.c_str(), "1") == 0)
            return true;
          if (strcmp(value.c_str(), "false") == 0 or strcmp(value.c_str(), "0") == 0)
            return false;
          error("Cannot convert \"%s\" for attribute \"%s\" in <%s> to bool.", value.c_str(), attribute, name);
          return false;
        }
      }
      // Didn't get the value
      error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
      return false;
    };

    void XMLMesh::startElement(const xmlChar *name, const xmlChar **attrs) {
      switch (state) {
      case OUTSIDE:
        if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0) {
          readMesh(name, attrs);
          state = INSIDE_MESH;
        }
        break;
      case INSIDE_MESH:
        if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0) {
          readVertices(name, attrs);
          state = INSIDE_VERTICES;
        }
        else if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0) {
          readCells(name, attrs);
          state = INSIDE_CELLS;
        }
        break;
      case INSIDE_VERTICES:
        if (xmlStrcasecmp(name, (xmlChar *) "vertex") == 0)
          readVertex(name, attrs);
        break;
      case INSIDE_CELLS:
        if (xmlStrcasecmp(name, (xmlChar *) "interval") == 0) {
          readInterval(name, attrs);
        } else if (xmlStrcasecmp(name, (xmlChar *) "triangle") == 0) {
          readTriangle(name, attrs);
        } else if (xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0) {
          readTetrahedron(name, attrs);
        }
        break;
      default:
        break;
      }
    };
    void XMLMesh::endElement(const xmlChar *name) {
      switch (state) {
      case INSIDE_MESH:
        if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0) {
          closeMesh();
          state = DONE;
        }
        break;
      case INSIDE_VERTICES:
        if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
          state = INSIDE_MESH;
        break;
      case INSIDE_CELLS:
        if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0)
          state = INSIDE_MESH;
        break;
      default:
        break;
      }
    };
    void XMLMesh::readMesh(const xmlChar *name, const xmlChar **attrs) {
      // Parse values
      std::string type = parseString(name, attrs, "celltype");
      this->embedDim = parseUnsignedInt(name, attrs, "dim");
      int tdim = 0;

      if (type == "interval") {
        tdim = 1;
      } else if (type == "triangle") {
        tdim = 2;
      } else if (type == "tetrahedron") {
        tdim = 3;
      }
      mesh->setDimension(tdim);
    };
    void XMLMesh::readVertices(const xmlChar *name, const xmlChar **attrs) {
      // Parse values
      unsigned int num_vertices = parseUnsignedInt(name, attrs, "size");
      // Set number of vertices
      this->coords = new double[num_vertices*this->embedDim];
    };
    void XMLMesh::readCells(const xmlChar *name, const xmlChar **attrs) {
      // Parse values
      this->numCells = parseUnsignedInt(name, attrs, "size");
    };
    void XMLMesh::readVertex(const xmlChar *name, const xmlChar **attrs) {
      // Read index
      uint v = parseUnsignedInt(name, attrs, "index");

      switch (this->embedDim) {
      case 3:
        this->coords[v*this->embedDim+2] = parseReal(name, attrs, "z");
      case 2:
        this->coords[v*this->embedDim+1] = parseReal(name, attrs, "y");
      case 1:
        this->coords[v*this->embedDim+0] = parseReal(name, attrs, "x");
        break;
      default:
        error("Dimension of mesh must be 1, 2 or 3.");
      }
    };
    void XMLMesh::readInterval(const xmlChar *name, const xmlChar **attrs) {
      // Check dimension
      if (mesh->getDimension() != 1)
        error("Mesh entity (interval) does not match dimension of mesh (%d).", mesh->getDimension());
      // Parse values
      unsigned int c  = parseUnsignedInt(name, attrs, "index");
      unsigned int v0 = parseUnsignedInt(name, attrs, "v0") + this->numCells;
      unsigned int v1 = parseUnsignedInt(name, attrs, "v1") + this->numCells;
      // Add cell
      mesh->getSieve()->addArrow(v0, c, 0);
      mesh->getSieve()->addArrow(v1, c, 1);
    };
    void XMLMesh::readTriangle(const xmlChar *name, const xmlChar **attrs) {
      // Check dimension
      if (mesh->getDimension() != 2)
        error("Mesh entity (triangle) does not match dimension of mesh (%d).", mesh->getDimension());
      // Parse values
      unsigned int c  = parseUnsignedInt(name, attrs, "index");
      unsigned int v0 = parseUnsignedInt(name, attrs, "v0") + this->numCells;
      unsigned int v1 = parseUnsignedInt(name, attrs, "v1") + this->numCells;
      unsigned int v2 = parseUnsignedInt(name, attrs, "v2") + this->numCells;
      // Add cell
      mesh->getSieve()->addArrow(v0, c, 0);
      mesh->getSieve()->addArrow(v1, c, 1);
      mesh->getSieve()->addArrow(v2, c, 2);
    };
    void XMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs) {
      // Check dimension
      if (mesh->getDimension() != 3)
        error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).", mesh->getDimension());
      // Parse values
      unsigned int c  = parseUnsignedInt(name, attrs, "index");
      unsigned int v0 = parseUnsignedInt(name, attrs, "v0") + this->numCells;
      unsigned int v1 = parseUnsignedInt(name, attrs, "v1") + this->numCells;
      unsigned int v2 = parseUnsignedInt(name, attrs, "v2") + this->numCells;
      unsigned int v3 = parseUnsignedInt(name, attrs, "v3") + this->numCells;
      // Add cell
      mesh->getSieve()->addArrow(v0, c, 0);
      mesh->getSieve()->addArrow(v1, c, 1);
      mesh->getSieve()->addArrow(v2, c, 2);
      mesh->getSieve()->addArrow(v3, c, 3);
    };
    void XMLMesh::closeMesh() {
      mesh->stratify();
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, this->embedDim, this->coords);
      delete [] this->coords;
    };
  }
}

#endif // PETSC_HAVE_LIBXML2
