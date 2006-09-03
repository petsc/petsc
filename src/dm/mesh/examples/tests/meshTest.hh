#include <Mesh.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sectionTest.hh"

using ALE::Obj;

namespace ALE {
  namespace Test {
    class MeshProcessor {
    public:
      typedef ALE::Mesh::section_type::value_type value_type; 
      typedef ALE::Mesh::section_type::patch_type patch_type;
   public:
      static std::string printMatrix(const std::string& name, const int rows, const int cols, const value_type matrix[], const int rank = -1)
      {
        ostringstream output;
        ostringstream rankStr;

        if (rank >= 0) {
          rankStr << "[" << rank << "]";
        }
        output << rankStr.str() << name << " = " << std::endl;
        for(int r = 0; r < rows; r++) {
          if (r == 0) {
            output << rankStr.str() << " /";
          } else if (r == rows-1) {
            output << rankStr.str() << " \\";
          } else {
            output << rankStr.str() << " |";
          }
          for(int c = 0; c < cols; c++) {
            output << " " << matrix[r*cols+c];
          }
          if (r == 0) {
            output << " \\" << std::endl;
          } else if (r == rows-1) {
            output << " /" << std::endl;
          } else {
            output << " |" << std::endl;
          }
        }
        return output.str();
      }
      static std::string printElement(const ALE::Mesh::section_type::point_type& e, const int dim, const value_type coords[], const int rank = -1) {
        ostringstream output;
        ostringstream r;

        if (rank >= 0) {
          r << "[" << rank << "]";
        }
        output << r.str() << "Element " << e << std::endl;
        output << r.str() << "Coordinates: " << e << std::endl << r.str() << "  ";
        for(int f = 0; f <= dim; f++) {
          output << " (";
          for(int d = 0; d < dim; d++) {
            if (d > 0) output << ", ";
            output << coords[f*dim+d];
          }
          output << ")";
        }
        output << std::endl;
        return output.str();
      };
      static void computeElementGeometry(const Obj<ALE::Mesh::section_type>& coordinates, int dim, const sieve_type::point_type& e, value_type v0[], value_type J[], value_type invJ[], value_type& detJ)
      {
        const patch_type  patch  = 0;
        const value_type *coords = coordinates->restrict(patch, e);
        value_type        invDet;

        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        if (dim == 1) {
          detJ = J[0];
        } else if (dim == 2) {
          detJ = J[0]*J[3] - J[1]*J[2];
        } else if (dim == 3) {
          detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
            J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
            J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        }
        invDet = 1.0/detJ;
        if (dim == 2) {
          invJ[0] =  invDet*J[3];
          invJ[1] = -invDet*J[1];
          invJ[2] = -invDet*J[2];
          invJ[3] =  invDet*J[0];
        } else if (dim == 3) {
          // FIX: This may be wrong
          invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
          invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
          invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
          invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
          invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
          invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
          invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
          invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
          invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        }
      };
    };
  };
};
