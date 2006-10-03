#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

namespace ALE {
  template<typename Topology_>
  class Bundle : public ALE::ParallelObject {
  public:
    typedef Topology_                                      topology_type;
    typedef typename topology_type::point_type             point_type;
    typedef ALE::New::Section<topology_type, double>       real_section_type;
    typedef ALE::New::Section<topology_type, int>          int_section_type;
    typedef struct {double x, y, z;}                       split_value;
    typedef ALE::pair<point_type, split_value>             pair_type;
    typedef ALE::New::Section<topology_type, pair_type>    pair_section_type;
    typedef std::map<std::string, Obj<real_section_type> > real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >  int_sections_type;
    typedef std::map<std::string, Obj<pair_section_type> > pair_sections_type;
    typedef typename topology_type::send_overlap_type      send_overlap_type;
    typedef typename topology_type::recv_overlap_type      recv_overlap_type;
    typedef typename ALE::New::Completion<topology_type, point_type>::topology_type             comp_topology_type;
    typedef typename ALE::New::OverlapValues<send_overlap_type, comp_topology_type, point_type> send_section_type;
    typedef typename ALE::New::OverlapValues<recv_overlap_type, comp_topology_type, point_type> recv_section_type;
  protected:
    Obj<topology_type> _topology;
    bool               _distributed;
    real_sections_type _realSections;
    int_sections_type  _intSections;
    pair_sections_type _pairSections;
  public:
    Bundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _distributed(false) {};
    Bundle(const Obj<topology_type>& topology) : ALE::ParallelObject(topology->comm(), topology->debug()), _topology(topology), _distributed(false) {};
  public: // Accessors
    bool getDistributed() const {return this->_distributed;};
    void setDistributed(const bool distributed) {this->_distributed = distributed;};
    const Obj<topology_type>& getTopology() const {return this->_topology;};
    void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
  public:
    bool hasRealSection(const std::string& name) {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (this->_realSections.find(name) == this->_realSections.end()) {
        Obj<real_section_type> section = new real_section_type(this->_topology);

        std::cout << "Creating new real section: " << name << std::endl;
        this->_realSections[name] = section;
      }
      return this->_realSections[name];
    };
    void setRealSection(const std::string& name, const Obj<real_section_type>& section) {
      this->_realSections[name] = section;
    };
    Obj<std::set<std::string> > getRealSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename real_sections_type::const_iterator s_iter = this->_realSections.begin(); s_iter != this->_realSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasIntSection(const std::string& name) {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (this->_intSections.find(name) == this->_intSections.end()) {
        Obj<int_section_type> section = new int_section_type(this->_topology);

        std::cout << "Creating new int section: " << name << std::endl;
        this->_intSections[name] = section;
      }
      return this->_intSections[name];
    };
    void setIntSection(const std::string& name, const Obj<int_section_type>& section) {
      this->_intSections[name] = section;
    };
    Obj<std::set<std::string> > getIntSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename int_sections_type::const_iterator s_iter = this->_intSections.begin(); s_iter != this->_intSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasPairSection(const std::string& name) {
      return this->_pairSections.find(name) != this->_pairSections.end();
    };
    const Obj<pair_section_type>& getPairSection(const std::string& name) {
      if (this->_pairSections.find(name) == this->_pairSections.end()) {
        Obj<pair_section_type> section = new pair_section_type(this->_topology);

        std::cout << "Creating new pair section: " << name << std::endl;
        this->_pairSections[name] = section;
      }
      return this->_pairSections[name];
    };
    void setPairSection(const std::string& name, const Obj<pair_section_type>& section) {
      this->_pairSections[name] = section;
    };
    Obj<std::set<std::string> > getPairSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename pair_sections_type::const_iterator s_iter = this->_pairSections.begin(); s_iter != this->_pairSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
  public:
    // Printing
    friend std::ostream& operator<<(std::ostream& os, const split_value& s) {
      os << "(" << s.x << ", "<< s.y << ", "<< s.z << ")";
      return os;
    };
  };

  class Mesh : public Bundle<ALE::New::Topology<int, ALE::Sieve<int,int,int> > > {
  public:
    typedef int                                       point_type;
    typedef ALE::Sieve<point_type,int,int>            sieve_type;
    typedef ALE::New::Topology<int, sieve_type>       topology_type;
    typedef topology_type::patch_type                 patch_type;
    typedef Bundle<topology_type>                     base_type;
    typedef ALE::New::NumberingFactory<topology_type> NumberingFactory;
    typedef NumberingFactory::numbering_type          numbering_type;
    typedef NumberingFactory::order_type              order_type;
    typedef base_type::send_overlap_type              send_overlap_type;
    typedef base_type::recv_overlap_type              recv_overlap_type;
    typedef base_type::send_section_type              send_section_type;
    typedef base_type::recv_section_type              recv_section_type;
    // PCICE BC
    typedef struct {double rho,u,v,p;}                bc_value_type;
    typedef std::map<int, bc_value_type>              bc_values_type;
    // PyLith BC
    typedef ALE::New::Section<topology_type, ALE::pair<int,double> > foliated_section_type;
  protected:
    int                   _dim;
    Obj<NumberingFactory> _factory;
    // PCICE BC
    bc_values_type        _bcValues;
    // PyLith BC
    Obj<foliated_section_type> _boundaries;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : Bundle<ALE::New::Topology<int, ALE::Sieve<int,int,int> > >(comm, debug), _dim(dim) {
      this->_factory = NumberingFactory::singleton(debug);
      this->_boundaries = NULL;
    };
    Mesh(const Obj<topology_type>& topology, int dim) : Bundle<ALE::New::Topology<int, ALE::Sieve<int,int,int> > >(topology), _dim(dim) {
      this->_factory = NumberingFactory::singleton(topology->debug());
      this->_boundaries = NULL;
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    const Obj<NumberingFactory>& getFactory() {return this->_factory;};
  public: // BC values for PCICE
    const bc_value_type& getBCValue(const int bcFunc) {
      return this->_bcValues[bcFunc];
    };
    void setBCValue(const int bcFunc, const bc_value_type& value) {
      this->_bcValues[bcFunc] = value;
    };
    bc_values_type& getBCValues() {
      return this->_bcValues;
    };
    void distributeBCValues() {
      int size = this->_bcValues.size();

      MPI_Bcast(&size, 1, MPI_INT, 0, this->comm()); 
      if (this->commRank()) {
        for(int bc = 0; bc < size; ++bc) {
          int           funcNum;
          bc_value_type funcVal;

          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
          this->_bcValues[funcNum] = funcVal;
        }
      } else {
        for(bc_values_type::iterator bc_iter = this->_bcValues.begin(); bc_iter != this->_bcValues.end(); ++bc_iter) {
          const int&           funcNum = bc_iter->first;
          const bc_value_type& funcVal = bc_iter->second;
          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
        }
      }
    };
  public: // BC values for PyLith
    const Obj<foliated_section_type>& getBoundariesNew() {
      if (this->_boundaries.isNull()) {
        this->_boundaries = new foliated_section_type(this->getTopology());
      }
      return this->_boundaries;
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Mesh\n");
      } else {
        PetscPrintf(comm, "viewing Mesh '%s'\n", name.c_str());
      }
      this->getTopology()->view("mesh topology", comm);
      Obj<std::set<std::string> > sections = this->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
      }
      sections = this->getPairSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getPairSection(*name)->view(*name);
      }
    };
  };

  class OldMesh {
  public:
    typedef int                                       point_type;
    typedef ALE::Sieve<point_type,int,int>            sieve_type;
    typedef ALE::New::Topology<int, sieve_type>       topology_type;
    typedef topology_type::patch_type                 patch_type;
    typedef ALE::New::Section<topology_type, double>  section_type;
    typedef section_type::atlas_type                  atlas_type;
    typedef std::map<std::string, Obj<section_type> > SectionContainer;
    typedef ALE::New::NumberingFactory<topology_type> NumberingFactory;
    typedef NumberingFactory::numbering_type          numbering_type;
    typedef NumberingFactory::order_type              order_type;
    typedef ALE::New::Section<topology_type, ALE::pair<int,double> >              foliated_section_type;
    typedef struct {double x, y, z;}                                              split_value;
    typedef ALE::New::Section<topology_type, ALE::pair<point_type, split_value> > pair_section_type;
    typedef ALE::New::Completion<topology_type, point_type>::send_overlap_type    send_overlap_type;
    typedef ALE::New::Completion<topology_type, point_type>::recv_overlap_type    recv_overlap_type;
    typedef ALE::New::Completion<topology_type, point_type>::topology_type        comp_topology_type;
    typedef ALE::New::OverlapValues<send_overlap_type, comp_topology_type, point_type> send_section_type;
    typedef ALE::New::OverlapValues<recv_overlap_type, comp_topology_type, point_type> recv_section_type;
    // PCICE: Big fucking hack
    typedef ALE::New::Section<topology_type, int>         int_section_type;
    typedef std::map<std::string, Obj<int_section_type> > BCSectionContainer;
    typedef struct {double rho,u,v,p;}                    bc_value_type;
    typedef std::map<int, bc_value_type>                  bc_values_type;
    int debug;
  private:
    SectionContainer           sections;
    Obj<topology_type>         _topology;
    Obj<foliated_section_type> _boundaries;
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
    OldMesh(MPI_Comm comm, int dimension, int debug = 0) : debug(debug), dim(dimension) {
      this->setComm(comm);
      this->_boundaries = NULL;
      this->distributed = false;
    };

    MPI_Comm        comm() const {return this->_comm;};
    void            setComm(MPI_Comm comm) {this->_comm = comm; MPI_Comm_rank(comm, &this->_commRank); MPI_Comm_size(comm, &this->_commSize);};
    int             commRank() const {return this->_commRank;};
    int             commSize() const {return this->_commSize;};
    int             getDimension() const {return this->dim;};
    void            setDimension(int dim) {this->dim = dim;};
    const Obj<foliated_section_type>& getBoundariesNew() {
      if (this->_boundaries.isNull()) {
        this->_boundaries = new foliated_section_type(this->getTopology());
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
    void setSection(const std::string& name, const Obj<section_type>& section) {
      this->sections[name] = section;
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
    const Obj<topology_type>& getTopology() const {return this->_topology;};
    void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
    // PCICE: Big fucking hack
    Obj<std::set<std::string> > getBCSections() {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(BCSectionContainer::iterator s_iter = this->bcSections.begin(); s_iter != this->bcSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    }
    const Obj<int_section_type>& getBCSection(const std::string& name) {
      if (this->bcSections.find(name) == this->bcSections.end()) {
        Obj<int_section_type> section = new int_section_type(this->_topology);

        std::cout << "Creating new bc section: " << name << std::endl;
        this->bcSections[name] = section;
      }
      return this->bcSections[name];
    };
    void setBCSection(const std::string& name, const Obj<int_section_type>& section) {
      this->bcSections[name] = section;
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
    void distributeBCValues() {
      int size = this->bcValues.size();

      MPI_Bcast(&size, 1, MPI_INT, 0, this->comm()); 
      if (this->commRank()) {
        for(int bc = 0; bc < size; ++bc) {
          int           funcNum;
          bc_value_type funcVal;

          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
          this->bcValues[funcNum] = funcVal;
        }
      } else {
        for(bc_values_type::iterator bc_iter = this->bcValues.begin(); bc_iter != this->bcValues.end(); ++bc_iter) {
          const int&           funcNum = bc_iter->first;
          const bc_value_type& funcVal = bc_iter->second;
          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
        }
      }
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
      this->getTopology()->view("mesh topology", comm);
      this->getSection("coordinates")->view("mesh coordinates", comm);
    };
  };
} // namespace ALE

#endif
