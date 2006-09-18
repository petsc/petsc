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
    typedef ALE::New::Section<topology_type, double>    section_type;
    typedef section_type::atlas_type                    atlas_type;
    typedef std::map<std::string, Obj<section_type> >   SectionContainer;
    typedef ALE::New::Numbering<topology_type>          numbering_type;
    typedef std::map<int, Obj<numbering_type> >                 NumberingContainer;
    typedef std::map<int, std::map<int, Obj<numbering_type> > > NewNumberingContainer;
    typedef ALE::New::GlobalOrder<topology_type, section_type::atlas_type> order_type;
    typedef std::map<std::string, Obj<order_type> >          OrderContainer;
    typedef ALE::New::Section<topology_type, ALE::pair<int,double> > foliated_section_type;
    typedef struct {double x, y, z;}                                           split_value;
    typedef ALE::New::Section<topology_type, ALE::pair<point_type, split_value> > split_section_type;
    typedef ALE::New::Completion<topology_type, point_type>::send_overlap_type send_overlap_type;
    typedef ALE::New::Completion<topology_type, point_type>::recv_overlap_type recv_overlap_type;
    typedef ALE::New::Completion<topology_type, point_type>::topology_type     comp_topology_type;
    typedef ALE::New::OverlapValues<send_overlap_type, comp_topology_type, point_type> send_section_type;
    typedef ALE::New::OverlapValues<recv_overlap_type, comp_topology_type, point_type> recv_section_type;
    // PCICE: Big fucking hack
    typedef ALE::New::Section<topology_type, int>        bc_section_type;
    typedef std::map<std::string, Obj<bc_section_type> > BCSectionContainer;
    typedef struct {double rho,u,v,p;}                   bc_value_type;
    typedef std::map<int, bc_value_type>                 bc_values_type;
    int debug;
  private:
    Obj<sieve_type>            topology;
    SectionContainer           sections;
    NewNumberingContainer      localNumberings;
    NumberingContainer         numberings;
    OrderContainer             orders;
    Obj<topology_type>         _topology;
    Obj<foliated_section_type> _boundaries;
    Obj<split_section_type>    _splitField;
    Obj<send_overlap_type>     _vertexSendOverlap;
    Obj<recv_overlap_type>     _vertexRecvOverlap;
    MPI_Comm        _comm;
    int             _commRank;
    int             _commSize;
    int             dim;
    // PCICE: Big fucking hack
    BCSectionContainer bcSections;
    bc_values_type     bcValues;
  public:
    bool distributed;
  public:
    Mesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
      this->setComm(comm);
      this->topology    = new sieve_type(comm, debug);
      this->_boundaries = NULL;
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
    const Obj<foliated_section_type>& getBoundariesNew() {
      if (this->_boundaries.isNull()) {
        this->_boundaries = new foliated_section_type(this->getTopologyNew());
      }
      return this->_boundaries;
    };
    const Obj<section_type>& getSection(const std::string& name) {
      if (this->sections.find(name) == this->sections.end()) {
        Obj<section_type> section = new section_type(this->_topology);

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
    const Obj<numbering_type>& getLocalNumbering(const int depth, const topology_type::patch_type& patch = 0) {
      if ((this->localNumberings.find(depth) == this->localNumberings.end()) ||
          (this->localNumberings[depth].find(patch) == this->localNumberings[depth].end())) {
        Obj<numbering_type> numbering = new numbering_type(this->getTopologyNew(), "depth", depth);
        numbering->constructLocalOrder(numbering->getSendOverlap(), patch);

        std::cout << "Creating new local numbering: depth " << depth << " patch " << patch << std::endl;
        this->localNumberings[depth][patch] = numbering;
      }
      return this->localNumberings[depth][patch];
    };
    const Obj<order_type>& getGlobalOrder(const std::string& name) {
      if (this->orders.find(name) == this->orders.end()) {
        Obj<order_type> order = new order_type(this->getSection(name)->getAtlas(), this->getNumbering(0));
        order->construct();

        std::cout << "Creating new global order: " << name << std::endl;
        this->orders[name] = order;
      }
      return this->orders[name];
    };
    const Obj<topology_type>& getTopologyNew() const {return this->_topology;};
    void setTopologyNew(const Obj<topology_type>& topology) {this->_topology = topology;};
    const Obj<split_section_type>& getSplitSection() const {return this->_splitField;};
    void                           setSplitSection(const Obj<split_section_type>& splitField) {this->_splitField = splitField;};
    const Obj<send_overlap_type>&  getVertexSendOverlap() const {return this->_vertexSendOverlap;};
    void                           setVertexSendOverlap(const Obj<send_overlap_type>& vertexOverlap) {this->_vertexSendOverlap = vertexOverlap;};
    const Obj<recv_overlap_type>&  getVertexRecvOverlap() const {return this->_vertexRecvOverlap;};
    void                           setVertexRecvOverlap(const Obj<recv_overlap_type>& vertexOverlap) {this->_vertexRecvOverlap = vertexOverlap;};
    // PCICE: Big fucking hack
    const Obj<bc_section_type>& getBCSection(const std::string& name) {
      if (this->bcSections.find(name) == this->bcSections.end()) {
        Obj<bc_section_type> section = new bc_section_type(this->_topology);

        std::cout << "Creating new bc section: " << name << std::endl;
        this->bcSections[name] = section;
      }
      return this->bcSections[name];
    };
    const bc_value_type& getBCValue(const int bcFunc) {
      return this->bcValues[bcFunc];
    };
    void setBCValue(const int bcFunc, const bc_value_type& value) {
      this->bcValues[bcFunc] = value;
    };
    bc_values_type& getBCValues() {
      return this->bcValues;
    };
    // Printing
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const split_value& v) {
      os << "(" << v.x << "," << v.y << "," << v.z << ")";
      return os;
    };
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Mesh\n");
      } else {
        PetscPrintf(comm, "viewing Mesh '%s'\n", name.c_str());
      }
      this->getTopologyNew()->view("mesh topology", comm);
      this->getSection("coordinates")->view("mesh coordinates", comm);
    };
  };
} // namespace ALE

#endif
