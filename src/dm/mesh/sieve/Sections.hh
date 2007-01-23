#ifndef included_ALE_Sections_hh
#define included_ALE_Sections_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  namespace New {
    // This section takes an existing section, and reports instead the fiber dimensions as values
    //   Should not need the _patch variable
    template<typename Section_>
    class SizeSection : public ALE::ParallelObject {
    public:
      typedef Section_                          section_type;
      typedef typename section_type::patch_type patch_type;
      typedef typename section_type::point_type point_type;
      typedef int                               value_type;
    protected:
      Obj<section_type> _section;
      const patch_type  _patch;
      value_type        _size;
    public:
      SizeSection(const Obj<section_type>& section, const patch_type& patch) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section), _patch(patch) {};
      virtual ~SizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_section->hasPoint(patch, point);
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getFiberDimension(this->_patch, p); // Could be size()
        return &this->_size;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getFiberDimension(this->_patch, p);
        return &this->_size;
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        this->_section->view(name, comm);
      };
    };

    // This section reports as values the size of the partition associated with the partition point
    template<typename Topology_, typename MeshTopology_, typename Marker_>
    class PartitionSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type> _topology;
      sizes_type         _sizes;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        // Should check for patch 0
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(0, 0);
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(0);
        std::map<patch_type, std::set<point_type> >             points;

        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSet>& closure = sieve->closure(*e_iter);

          points[partition[*e_iter]].insert(closure->begin(), closure->end());
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_sizes[p_iter->first] = p_iter->second.size();
        }
      };
    public:
      PartitionSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology) {
        this->_init(meshTopology, numElements, partition);
      };
      virtual ~PartitionSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return &this->_sizes[p];
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a PartitionSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_>
    class PartitionDomain {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
    public:
      PartitionDomain() {};
      ~PartitionDomain() {};
    public:
      int count(const point_type& point) const {return 1;};
    };

    // This section returns the points in each partition
    template<typename Topology_, typename MeshTopology_, typename Marker_>
    class PartitionSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,point_type*>   points_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type> _topology;
      points_type        _points;
      chart_type         _domain;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        // Should check for patch 0
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(0, 0);
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(0);
        std::map<patch_type, std::set<point_type> >             points;
        std::map<patch_type, int>                               offsets;

        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSet>& closure = sieve->closure(*e_iter);

          points[partition[*e_iter]].insert(closure->begin(), closure->end());
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_points[p_iter->first] = new point_type[p_iter->second.size()];
          offsets[p_iter->first] = 0;
          for(typename std::set<point_type>::const_iterator s_iter = p_iter->second.begin(); s_iter != p_iter->second.end(); ++s_iter) {
            this->_points[p_iter->first][offsets[p_iter->first]++] = *s_iter;
          }
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          if (offsets[p_iter->first] != (int) p_iter->second.size()) {
            ostringstream txt;
            txt << "Invalid offset for partition " << p_iter->first << ": " << offsets[p_iter->first] << " should be " << p_iter->second.size();
            throw ALE::Exception(txt.str().c_str());
          }
        }
      };
    public:
      PartitionSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology) {this->_init(meshTopology, numElements, partition);};
      virtual ~PartitionSection() {
        for(typename points_type::iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          delete [] p_iter->second;
        }
      };
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return this->_points[p];
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a PartitionSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };
  }
}
#endif
