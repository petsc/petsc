#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
    class Mesh {
    public:
      typedef int point_type;
      typedef std::vector<point_type> PointArray;
      typedef ALE::Sieve<point_type,int,int> sieve_type;
      typedef ALE::Point patch_type;
      typedef ALE::New::Topology<int, sieve_type>        topology_type;
      typedef ALE::New::Atlas<topology_type, ALE::Point> atlas_type;
      typedef ALE::New::Section<atlas_type, double>      section_type;
      typedef std::map<std::string, Obj<section_type> >  SectionContainer;
      typedef ALE::New::Section<atlas_type, ALE::pair<int,double> > foliated_section_type;
      int debug;
    private:
      Obj<sieve_type>            topology;
      SectionContainer           sections;
      Obj<topology_type>         _topology;
      Obj<foliated_section_type> _boundaries;
      MPI_Comm        _comm;
      int             _commRank;
      int             _commSize;
      int             dim;
      //FIX:
    public:
      bool            distributed;
    public:
      Mesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
        this->setComm(comm);
        this->topology    = new sieve_type(comm, debug);
        this->_boundaries = new foliated_section_type(comm, debug);
        this->distributed = false;
      };

      MPI_Comm        comm() const {return this->_comm;};
      void            setComm(MPI_Comm comm) {this->_comm = comm; MPI_Comm_rank(comm, &this->_commRank); MPI_Comm_size(comm, &this->_commSize);};
      int             commRank() const {return this->_commRank;};
      int             commSize() const {return this->_commSize;};
      Obj<sieve_type> getTopology() const {return this->topology;};
      void            setTopology(const Obj<sieve_type>& topology) {this->topology = topology;};
      int             getDimension() const {return this->dim;};
      void            setDimension(int dim) {this->dim = dim;};
      const Obj<foliated_section_type>& getBoundariesNew() const {return this->_boundaries;};
      Obj<section_type> getSection(const std::string& name) {
        if (this->sections.find(name) == this->sections.end()) {
          Obj<section_type> section = new section_type(this->_comm, this->debug);
          section->getAtlas()->setTopology(this->_topology);

          std::cout << "Creating new section: " << name << std::endl;
          this->sections[name] = section;
        }
        return this->sections[name];
      };
      Obj<std::set<std::string> > getSections() {
        Obj<std::set<std::string> > names = std::set<std::string>();

        for(SectionContainer::iterator s_iter = this->sections.begin(); s_iter != this->sections.end(); ++s_iter) {
          names->insert(s_iter->first);
        }
        return names;
      }
      bool hasSection(const std::string& name) const {
        return(this->sections.find(name) != this->sections.end());
      };
      const Obj<topology_type>& getTopologyNew() {return this->_topology;};
      void setTopologyNew(const Obj<topology_type>& topology) {this->_topology = topology;};
    private:
      template<typename IntervalSequence>
      int *__expandIntervals(Obj<IntervalSequence> intervals) {
        int *indices;
        int  k = 0;

        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          k += std::abs(i_iter.color().index);
        }
        std::cout << "Allocated indices of size " << k << std::endl;
        indices = new int[k];
        k = 0;
        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          for(int i = i_iter.color().prefix; i < i_iter.color().prefix + std::abs(i_iter.color().index); i++) {
            std::cout << "  indices[" << k << "] = " << i << std::endl;
            indices[k++] = i;
          }
        }
        return indices;
      };
      template<typename IntervalSequence,typename Field>
      int *__expandCanonicalIntervals(Obj<IntervalSequence> intervals, Obj<Field> field) {
        typename Field::patch_type patch;
        int *indices;
        int  k = 0;

        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          k += std::abs(field->getFiberDimension(patch, *i_iter));
        }
        std::cout << "Allocated indices of size " << k << std::endl;
        indices = new int[k];
        k = 0;
        for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
          int dim = field->getFiberDimension(patch, *i_iter);
          int offset = field->getFiberOffset(patch, *i_iter);

          for(int i = offset; i < offset + std::abs(dim); i++) {
            std::cout << "  indices[" << k << "] = " << i << std::endl;
            indices[k++] = i;
          }
        }
        return indices;
      };
    public:
      // Create a serial mesh
      void populate(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true, int corners = -1) {
        this->topology->setStratification(false);
        ALE::New::SieveBuilder<sieve_type>::buildTopology(this->topology, this->dim, numSimplices, simplices, numVertices, interpolate, corners);
        this->topology->stratify();
        this->topology->setStratification(true);
      };
      void populateBd(int numSimplices, int simplices[], int numVertices, double coords[], bool interpolate = true, int corners = -1) {
        this->topology->setStratification(false);
        ALE::New::SieveBuilder<sieve_type>::buildTopology(this->topology, this->dim, numSimplices, simplices, numVertices, interpolate, corners);
        this->topology->stratify();
        this->topology->setStratification(true);
      };
    };
} // namespace ALE

#endif
