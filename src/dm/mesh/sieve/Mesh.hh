#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

namespace ALE {
    class Mesh {
    public:
      typedef int point_type;
      typedef std::vector<point_type> PointArray;
      typedef ALE::Sieve<point_type,int,int> sieve_type;
      typedef ALE::Point patch_type;
      typedef ALE::New::Topology<int, sieve_type>         topology_type;
      typedef ALE::New::Atlas<topology_type, ALE::Point>  atlas_type;
      typedef ALE::New::Section<atlas_type, double>       section_type;
      typedef std::map<std::string, Obj<section_type> >   SectionContainer;
      typedef ALE::New::Numbering<topology_type>          numbering_type;
      typedef std::map<int, Obj<numbering_type> >         NumberingContainer;
      typedef std::map<std::string, Obj<numbering_type> > OrderContainer;
      typedef ALE::New::Section<atlas_type, ALE::pair<int,double> > foliated_section_type;
      int debug;
    private:
      Obj<sieve_type>            topology;
      SectionContainer           sections;
      NumberingContainer         localNumberings;
      NumberingContainer         numberings;
      OrderContainer             orders;
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
      const Obj<section_type>& getSection(const std::string& name) {
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
      const Obj<numbering_type>& getNumbering(const int depth) {
        if (this->numberings.find(depth) == this->numberings.end()) {
          Obj<numbering_type> numbering = new numbering_type(this->getTopologyNew(), "depth", depth);
          numbering->construct();

          std::cout << "Creating new numbering: " << depth << std::endl;
          this->numberings[depth] = numbering;
        }
        return this->numberings[depth];
      };
      const Obj<numbering_type>& getLocalNumbering(const int depth) {
        if (this->localNumberings.find(depth) == this->localNumberings.end()) {
          Obj<numbering_type> numbering = new numbering_type(this->getTopologyNew(), "depth", depth);
          numbering->constructLocalOrder();

          std::cout << "Creating new local numbering: " << depth << std::endl;
          this->localNumberings[depth] = numbering;
        }
        return this->localNumberings[depth];
      };
      const Obj<numbering_type>& getGlobalOrder(const std::string& name) {
        if (this->orders.find(name) == this->orders.end()) {
          Obj<numbering_type> numbering = ALE::New::GlobalOrder::createIndices(this->getSection(name)->getAtlas(), this->getNumbering(0));

          std::cout << "Creating new global order: " << name << std::endl;
          this->orders[name] = numbering;
        }
        return this->orders[name];
      };
      const Obj<topology_type>& getTopologyNew() {return this->_topology;};
      void setTopologyNew(const Obj<topology_type>& topology) {this->_topology = topology;};
    };
} // namespace ALE

#endif
