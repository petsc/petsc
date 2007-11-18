#include <Mesh.hh>

using ALE::Obj;

namespace ALE {
  namespace Test {
    class MeshProcessor {
    public:
      typedef ALE::Mesh::real_section_type::value_type value_type; 
   public:
      static std::string printElement(const ALE::Mesh::real_section_type::point_type& e, const int dim, const value_type coords[], const int rank = -1) {
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
    };
  };
};
