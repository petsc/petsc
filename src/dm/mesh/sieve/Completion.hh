#ifndef included_ALE_Completion_hh
#define included_ALE_Completion_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

extern PetscErrorCode PetscCommSynchronizeTags(MPI_Comm);

namespace ALE {
  namespace New {
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

    template<typename Section_>
    class PatchlessSection : public ALE::ParallelObject {
    public:
      typedef Section_                          section_type;
      typedef typename section_type::patch_type patch_type;
      typedef typename section_type::sieve_type sieve_type;
      typedef typename section_type::point_type point_type;
      typedef typename section_type::value_type value_type;
      typedef typename section_type::chart_type chart_type;
    protected:
      Obj<section_type> _section;
      const patch_type  _patch;
    public:
      PatchlessSection(const Obj<section_type>& section, const patch_type& patch) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section), _patch(patch) {};
      virtual ~PatchlessSection() {};
    public:
      const chart_type& getPatch(const patch_type& patch) {
        return this->_section->getAtlas()->getPatch(this->_patch);
      };
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_section->hasPoint(patch, point);
      };
      const value_type *restrict(const patch_type& patch) {
        return this->_section->restrict(this->_patch);
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        return this->_section->restrict(this->_patch, p);
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return this->_section->restrictPoint(this->_patch, p);
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->update(this->_patch, p, v);
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->updateAdd(this->_patch, p, v);
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->updatePoint(this->_patch, p, v);
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        this->_section->update(this->_patch, p, v);
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        this->_section->view(name, comm);
      };
    };

    template<typename Topology_, typename Marker_>
    class PartitionSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type> _topology;
      sizes_type         _sizes;
      void _init(const int numElements, const marker_type partition[]) {
        for(int e = 0; e < numElements; e++) {
          this->_sizes[partition[e]]++;
        }
      };
    public:
      PartitionSizeSection(MPI_Comm comm, const int numElements, const marker_type *partition, const int debug = 0) : ParallelObject(comm, debug) {
        this->_topology = new topology_type(comm, debug);
        this->_init(numElements, partition);
      };
      PartitionSizeSection(const Obj<topology_type>& topology, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology) {this->_init(numElements, partition);};
      virtual ~PartitionSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a PartitionSizeSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return &this->_sizes[p];
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSizeSection");
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSizeSection");
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSizeSection");
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        throw ALE::Exception("Cannot update a PartitionSizeSection");
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
        std::map<patch_type,int> sizes;
        std::map<patch_type,int> offsets;

        for(int e = 0; e < numElements; e++) {
          sizes[partition[e]]++;
        }
        for(typename std::map<patch_type,int>::iterator p_iter = sizes.begin(); p_iter != sizes.end(); ++p_iter) {
          this->_points[p_iter->first] = new point_type[p_iter->second];
          offsets[p_iter->first] = 0;
        }
        int e = 0;

        if (topology->hasPatch(0)) {
          const Obj<typename topology_type::label_sequence>& cells = topology->heightStratum(0, 0);

          for(typename topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
            this->_points[partition[e]][offsets[partition[e]]++] = *e_iter;
            e++;
          }
        }
        for(typename std::map<patch_type,int>::iterator p_iter = sizes.begin(); p_iter != sizes.end(); ++p_iter) {
          if (offsets[p_iter->first] != sizes[p_iter->first]) {
            ostringstream txt;
            txt << "Invalid offset for partition " << p_iter->first << ": " << offsets[p_iter->first] << " should be " << sizes[p_iter->first];
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

    template<typename Topology_, typename Sieve_>
    class ConeSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             cone_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>   _topology;
      Obj<cone_sieve_type> _sieve;
      value_type           _size;
    public:
      ConeSizeSection(MPI_Comm comm, const Obj<cone_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve) {
        this->_topology = new topology_type(comm, debug);
      };
      ConeSizeSection(const Obj<topology_type>& topology, const Obj<cone_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve) {};
      virtual ~ConeSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a ConeSizeSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->_size = this->_sieve->cone(p)->size();
        return &this->_size;
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSizeSection");
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSizeSection");
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSizeSection");
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        throw ALE::Exception("Cannot update a ConeSizeSection");
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
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a ConeSection");
      };
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
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSection");
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSection");
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConeSection");
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        throw ALE::Exception("Cannot update a ConeSection");
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

    template<typename Topology_, typename Sieve_>
    class SupportSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             support_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<support_sieve_type> _sieve;
      value_type              _size;
    public:
      SupportSizeSection(MPI_Comm comm, const Obj<support_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve) {
        this->_topology = new topology_type(comm, debug);
      };
      SupportSizeSection(const Obj<topology_type>& topology, const Obj<support_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve) {};
      virtual ~SupportSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->_size = this->_sieve->support(p)->size();
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

    template<typename Topology_, typename Value_>
    class Completion {
    public:
      typedef int                                                                         point_type;
      typedef Value_                                                                      value_type;
      typedef Topology_                                                                   mesh_topology_type;
      typedef typename mesh_topology_type::sieve_type                                     sieve_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                                dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                               topology_type;
      typedef typename ALE::Sifter<int, point_type, point_type>                           send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, int>     send_sizer_type;
      typedef typename ALE::Sifter<point_type, int, point_type>                           recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, int>     recv_sizer_type;
      typedef typename ALE::New::ConstantSection<topology_type, int>                      constant_sizer;
      typedef typename ALE::New::ConstantSection<topology_type, value_type>               constant_section;
      typedef typename ALE::New::ConeSizeSection<topology_type, sieve_type>               cone_size_section;
      typedef typename ALE::New::ConeSection<topology_type, sieve_type>                   cone_section;
    public:
      // Creates a DiscreteTopology with the overlap information
      static Obj<topology_type> createSendTopology(const Obj<send_overlap_type>& sendOverlap) {
        const Obj<send_overlap_type::traits::baseSequence> ranks = sendOverlap->base();
        Obj<topology_type> topology = new topology_type(sendOverlap->comm(), sendOverlap->debug());

        for(send_overlap_type::traits::baseSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(*r_iter));
          topology->setPatch(*r_iter, sendSieve);
        }
        topology->stratify();
        return topology;
      };
      template<typename Sizer, typename Section>
      static void setupSend(const Obj<send_overlap_type>& sendOverlap, const Obj<Sizer>& sendSizer, const Obj<Section>& sendSection) {
        // Here we should just use the overlap as the topology (once it is a new-style sieve)
        sendSection->clear();
        sendSection->setTopology(ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap));
        if (sendSection->debug() > 10) {sendSection->getTopology()->view("Send topology after setup", MPI_COMM_SELF);}
        sendSection->construct(sendSizer);
        sendSection->allocate();
        sendSection->constructCommunication(Section::SEND);
      };
      template<typename Filler, typename Section>
      static void fillSend(const Filler& sendFiller, const Obj<Section>& sendSection) {
        const topology_type::sheaf_type& ranks = sendSection->getTopology()->getPatches();
        const topology_type::patch_type  patch = 0; // FIX: patch should come from overlap

        for(topology_type::sheaf_type::const_iterator p_iter = ranks.begin(); p_iter != ranks.end(); ++p_iter) {
          const int&                                          rank = p_iter->first;
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            if (sendFiller->hasPoint(patch, *b_iter)) {
              sendSection->update(rank, *b_iter, sendFiller->restrict(patch, *b_iter));
            }
          }
        }
      };
      template<typename Sizer, typename Section>
      static void setupReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<Section>& recvSection) {
        // Create section
        const Obj<recv_overlap_type::traits::capSequence> ranks = recvOverlap->cap();

        recvSection->clear();
        for(recv_overlap_type::traits::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> recvSieve = new dsieve_type();
          const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(*r_iter);

          // Want to replace this loop with a slice through color
          for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            const dsieve_type::point_type& point = p_iter.color();

            recvSieve->addPoint(point);
          }
          recvSection->getTopology()->setPatch(*r_iter, recvSieve);
        }
        recvSection->getTopology()->stratify();
        recvSection->construct(recvSizer);
        recvSection->allocate();
        recvSection->constructCommunication(Section::RECEIVE);
      };
      template<typename SizerFiller, typename Filler, typename SendSection, typename RecvSection>
      static void completeSection(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<SizerFiller>& sizerFiller, const Filler& filler, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        Obj<send_sizer_type> sendSizer     = new send_sizer_type(sendSection->comm(), sendSection->debug());
        Obj<recv_sizer_type> recvSizer     = new recv_sizer_type(recvSection->comm(), sendSizer->getTag(), recvSection->debug());
        Obj<constant_sizer>  constantSizer = new constant_sizer(recvSection->comm(), 1, sendSection->debug());

        // 1) Create the sizer sections
        ALE::New::Completion<mesh_topology_type, int>::setupSend(sendOverlap, constantSizer, sendSizer);
        ALE::New::Completion<mesh_topology_type, int>::setupReceive(recvOverlap, constantSizer, recvSizer);
        // 2) Fill the sizer section and communicate
        ALE::New::Completion<mesh_topology_type,int>::fillSend(sizerFiller, sendSizer);
        if (sendSizer->debug()) {sendSizer->view("Send Sizer in Completion", MPI_COMM_SELF);}
        sendSizer->startCommunication();
        recvSizer->startCommunication();
        sendSizer->endCommunication();
        recvSizer->endCommunication();
        if (recvSizer->debug()) {recvSizer->view("Receive Sizer in Completion", MPI_COMM_SELF);}
        // No need to update a global section since the receive sizes are all on the interface
        // 3) Create the send and receive sections
        ALE::New::Completion<mesh_topology_type,value_type>::setupSend(sendOverlap, sendSizer, sendSection);
        ALE::New::Completion<mesh_topology_type,value_type>::setupReceive(recvOverlap, recvSizer, recvSection);
        // 4) Fill up send section and communicate
        ALE::New::Completion<mesh_topology_type,int>::fillSend(filler, sendSection);
        if (sendSection->debug()) {sendSection->view("Send Section in Completion", MPI_COMM_SELF);}
        sendSection->startCommunication();
        recvSection->startCommunication();
        sendSection->endCommunication();
        recvSection->endCommunication();
        if (recvSection->debug()) {recvSection->view("Receive Section in Completion", MPI_COMM_SELF);}
      };
      template<typename PartitionType>
      static void scatterSieve(const Obj<mesh_topology_type>& topology, const Obj<sieve_type>& sieve, const int dim, const Obj<sieve_type>& sieveNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int numCells, const PartitionType assignment[]) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        int rank  = sieve->commRank();
        int debug = sieve->debug();

        // Create local sieve
        const Obj<topology_type::label_sequence>& cells = topology->heightStratum(0, 0);
        int e = 0;

        for(topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          if (assignment[e] == rank) {
            const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);

            for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              sieveNew->addArrow(*c_iter, *e_iter, c_iter.color());
            }
          }
          e++;
        }
        sieveNew->stratify();
        // Complete sizer section
        typedef typename ALE::New::PartitionSizeSection<topology_type, PartitionType>                 partition_size_section;
        typedef typename ALE::New::PartitionSection<topology_type, mesh_topology_type, PartitionType> partition_section;
        Obj<topology_type>          secTopology          = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, numCells, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, topology, numCells, assignment);
        Obj<send_section_type>      sendSection          = new send_section_type(sieve->comm(), sieve->debug());
        Obj<recv_section_type>      recvSection          = new recv_section_type(sieve->comm(), sendSection->getTag(), sieve->debug());

        ALE::New::Completion<mesh_topology_type,value_type>::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // Unpack the section into the overlap
        sendOverlap->clear();
        recvOverlap->clear();
        const topology_type::sheaf_type& sendPatches = sendSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = sendPatches.begin(); p_iter != sendPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
            int size = sendSection->size(p_iter->first, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        const topology_type::sheaf_type& recvPatches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = recvPatches.begin(); p_iter != recvPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
        if (debug) {
          sendOverlap->view(std::cout, "Send overlap for points");
          recvOverlap->view(std::cout, "Receive overlap for points");
        }
        // Receive the point section
        ALE::New::Completion<mesh_topology_type,value_type>::scatterCones(sieve, sieveNew, sendOverlap, recvOverlap);
        sieveNew->stratify();
      };
      template<typename SifterType>
      static void scatterCones(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typedef typename ALE::New::ConeSizeSection<topology_type, SifterType>                  cone_size_section;
        typedef typename ALE::New::ConeSection<topology_type, SifterType>                      cone_section;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        Obj<topology_type>     secTopology     = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(secTopology, sifter);
        Obj<cone_section>      coneSection     = new cone_section(secTopology, sifter);
        Obj<send_section_type> sendSection     = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type> recvSection     = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        ALE::New::Completion<mesh_topology_type,value_type>::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(points[p], *b_iter, c++);
            }
          }
        }
      };
      template<typename SifterType>
      static void scatterSupports(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typedef typename ALE::New::SupportSizeSection<topology_type, SifterType>               support_size_section;
        typedef typename ALE::New::SupportSection<topology_type, SifterType>                   support_section;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        Obj<topology_type>        secTopology        = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<support_size_section> supportSizeSection = new support_size_section(secTopology, sifter);
        Obj<support_section>      supportSection     = new support_section(secTopology, sifter);
        Obj<send_section_type>    sendSection        = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type>    recvSection        = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        ALE::New::Completion<mesh_topology_type,value_type>::completeSection(sendOverlap, recvOverlap, supportSizeSection, supportSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(*b_iter, points[p], c++);
            }
          }
        }
      };
    };

    template<typename Value_>
    class ParallelFactory {
    public:
      typedef Value_ value_type;
    protected:
      int          _debug;
      MPI_Datatype _mpiType;
    protected:
      MPI_Datatype constructMPIType() {
        if (sizeof(value_type) == 4) {
          return MPI_INT;
        } else if (sizeof(value_type) == 8) {
          return MPI_DOUBLE;
        } else if (sizeof(value_type) == 28) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        } else if (sizeof(value_type) == 32) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_DOUBLE;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        }
        throw ALE::Exception("Cannot determine MPI type for value type");
      };
      ParallelFactory(const int debug) : _debug(debug) {
        this->_mpiType = this->constructMPIType();
      };
    public:
      ~ParallelFactory() {};
    public:
      static const Obj<ParallelFactory>& singleton(const int debug, bool cleanup = false) {
        static Obj<ParallelFactory> *_singleton = NULL;

        if (cleanup) {
          if (debug) {std::cout << "Destroying ParallelFactory" << std::endl;}
          if (_singleton) {delete _singleton;}
          _singleton = NULL;
        } else if (_singleton == NULL) {
          if (debug) {std::cout << "Creating new ParallelFactory" << std::endl;}
          _singleton  = new Obj<ParallelFactory>();
          *_singleton = new ParallelFactory(debug);
        }
        return *_singleton;
      };
    public: // Accessors
      int debug() const {return this->_debug;};
      MPI_Datatype getMPIType() const {return this->_mpiType;};
    };
  }
}
#endif
