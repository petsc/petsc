#ifndef included_ALE_Sections_hh
#define included_ALE_Sections_hh

#ifndef  included_ALE_Numbering_hh
#include <Numbering.hh>
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
      typedef typename ALE::New::NumberingFactory<mesh_topology_type> NumberingFactory;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type> _topology;
      sizes_type         _sizes;
      int                _height;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        // Should check for patch 0
        const typename mesh_topology_type::patch_type           patch = 0;
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(patch);
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(patch, this->_height);
        const Obj<typename NumberingFactory::numbering_type>&   cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch) - this->_height);
        std::map<patch_type, std::set<point_type> >             points;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSet>& closure = sieve->closure(*e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename mesh_topology_type::sieve_type::supportSet>& star = sieve->star(*e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_sizes[p_iter->first] = p_iter->second.size();
        }
      };
    public:
      PartitionSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _height(elementHeight) {
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
        for(typename sizes_type::const_iterator s_iter = this->_sizes.begin(); s_iter != this->_sizes.end(); ++s_iter) {
          const patch_type& patch = s_iter->first;
          const value_type  size  = s_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << " size " << size << std::endl;
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
      typedef ALE::New::NumberingFactory<mesh_topology_type> NumberingFactory;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,point_type*>   points_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type> _topology;
      points_type        _points;
      chart_type         _domain;
      int                _height;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        // Should check for patch 0
        const typename mesh_topology_type::patch_type           patch = 0;
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(patch);
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(patch, this->_height);
        const Obj<typename NumberingFactory::numbering_type>&   cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch) - this->_height);
        std::map<patch_type, std::set<point_type> >             points;
        std::map<patch_type, int>                               offsets;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSet>& closure = sieve->closure(*e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename mesh_topology_type::sieve_type::supportSet>& star = sieve->star(*e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
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
      PartitionSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _height(elementHeight) {
        this->_init(meshTopology, numElements, partition);
      };
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
        for(typename points_type::const_iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          const patch_type& patch  = p_iter->first;
          //const point_type *points = p_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
        }
        if (this->_points.size() == 0) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename MeshTopology_, typename Sieve_>
    class ConeSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             cone_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<mesh_topology_type> _meshTopology;
      Obj<cone_sieve_type>    _sieve;
      value_type              _size;
      int                     _minHeight;
    public:
      ConeSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const Obj<cone_sieve_type>& sieve, int minimumHeight = 0) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _meshTopology(meshTopology), _sieve(sieve), _minHeight(minimumHeight) {};
      virtual ~ConeSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if ((this->_minHeight == 0) || (this->_meshTopology->height(patch, p) >= this->_minHeight)) {
          this->_size = this->_sieve->cone(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
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
            txt << "viewing a ConeSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename Sieve_>
    class ConeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             cone_sieve_type;
      typedef point_type                         value_type;
      typedef std::map<patch_type,int>           sizes_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<cone_sieve_type>    _sieve;
      int                     _coneSize;
      value_type             *_cone;
      chart_type              _domain;
      void ensureCone(const int size) {
        if (size > this->_coneSize) {
          if (this->_cone) delete [] this->_cone;
          this->_coneSize = size;
          this->_cone     = new value_type[this->_coneSize];
        }
      };
    public:
      ConeSection(MPI_Comm comm, const Obj<cone_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve), _coneSize(-1), _cone(NULL) {
        this->_topology = new topology_type(comm, debug);
      };
      ConeSection(const Obj<topology_type>& topology, const Obj<cone_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve), _coneSize(-1), _cone(NULL) {};
      virtual ~ConeSection() {if (this->_cone) delete [] this->_cone;};
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        const Obj<typename cone_sieve_type::traits::coneSequence>& cone = this->_sieve->cone(p);
        int c = 0;

        this->ensureCone(cone->size());
        for(typename cone_sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          this->_cone[c++] = *c_iter;
        }
        return this->_cone;
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
            txt << "viewing a ConeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename MeshTopology_, typename Sieve_>
    class SupportSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             support_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<mesh_topology_type> _meshTopology;
      Obj<support_sieve_type> _sieve;
      value_type              _size;
      int                     _minDepth;
    public:
      SupportSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const Obj<support_sieve_type>& sieve, int minimumDepth = 0) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _meshTopology(meshTopology), _sieve(sieve), _minDepth(minimumDepth) {};
      virtual ~SupportSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if ((this->_minDepth == 0) || (this->_meshTopology->depth(patch, p) >= this->_minDepth)) {
          this->_size = this->_sieve->support(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
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
            txt << "viewing a SupportSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename Sieve_>
    class SupportSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             support_sieve_type;
      typedef point_type                         value_type;
      typedef std::map<patch_type,int>           sizes_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<support_sieve_type> _sieve;
      int                     _supportSize;
      value_type             *_support;
      chart_type              _domain;
      void ensureSupport(const int size) {
        if (size > this->_supportSize) {
          if (this->_support) delete [] this->_support;
          this->_supportSize = size;
          this->_support     = new value_type[this->_supportSize];
        }
      };
    public:
      SupportSection(MPI_Comm comm, const Obj<support_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve), _supportSize(-1), _support(NULL) {
        this->_topology = new topology_type(comm, debug);
      };
      SupportSection(const Obj<topology_type>& topology, const Obj<support_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve), _supportSize(-1), _support(NULL) {};
      virtual ~SupportSection() {if (this->_support) delete [] this->_support;};
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        const Obj<typename support_sieve_type::traits::supportSequence>& support = this->_sieve->support(p);
        int s = 0;

        this->ensureSupport(support->size());
        for(typename support_sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
          this->_support[s++] = *s_iter;
        }
        return this->_support;
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
            txt << "viewing a SupportSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };
  }
}
#endif
