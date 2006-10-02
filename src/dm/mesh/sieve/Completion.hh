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

    template<typename Topology_, typename Marker_>
    class PartitionSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
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
      void _init(const int numElements, const marker_type partition[]) {
        std::map<patch_type,int> sizes;

        for(int e = 0; e < numElements; e++) {
          sizes[partition[e]]++;
        }
        for(typename std::map<patch_type,int>::iterator p_iter = sizes.begin(); p_iter != sizes.end(); ++p_iter) {
          this->_points[p_iter->first] = new point_type[p_iter->second];
          sizes[p_iter->first] = 0;
        }
        for(int e = 0; e < numElements; e++) {
          this->_points[partition[e]][sizes[partition[e]]++] = e;
        }
      };
    public:
      PartitionSection(MPI_Comm comm, const int numElements, const marker_type *partition, const int debug = 0) : ParallelObject(comm, debug) {
        this->_topology = new topology_type(comm, debug);
        this->_init(numElements, partition);
      };
      PartitionSection(const Obj<topology_type>& topology, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology) {this->_init(numElements, partition);};
      virtual ~PartitionSection() {
        for(typename points_type::iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          delete [] p_iter->second;
        }
      };
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a PartitionSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return this->_points[p];
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSection");
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSection");
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a PartitionSection");
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        throw ALE::Exception("Cannot update a PartitionSection");
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
        Obj<topology_type> topology = new topology_type(sendOverlap->comm(), sendOverlap->debug);

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
        if (sendSection->debug() > 10) {sendSection->getAtlas()->getTopology()->view("Send topology after setup", MPI_COMM_SELF);}
        sendSection->construct(sendSizer);
        sendSection->allocate();
        sendSection->constructCommunication(Section::SEND);
      };
      template<typename Filler, typename Section>
      static void fillSend(const Filler& sendFiller, const Obj<Section>& sendSection) {
        const topology_type::sheaf_type& ranks = sendSection->getAtlas()->getTopology()->getPatches();
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
      // Partition a topology on process 0 and scatter to all processes
      static void scatterTopology(const Obj<mesh_topology_type>& topology, const int dim, const Obj<mesh_topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const std::string& partitioner) {
        if (partitioner == "chaco") {
#ifdef PETSC_HAVE_CHACO
          scatterTopology<ALE::New::Chaco::Partitioner<mesh_topology_type> >(topology, dim, topologyNew, sendOverlap, recvOverlap);
#else
          throw ALE::Exception("Chaco is not installed. Reconfigure with the flag --download-chaco");
#endif
        } else if (partitioner == "parmetis") {
#ifdef PETSC_HAVE_PARMETIS
          scatterTopology<ALE::New::ParMetis::Partitioner<mesh_topology_type> >(topology, dim, topologyNew, sendOverlap, recvOverlap);
#else
          throw ALE::Exception("ParMetis is not installed. Reconfigure with the flag --download-parmetis");
#endif
        } else {
          throw ALE::Exception("Unknown partitioner");
        }
      };
      template<typename Partitioner>
      static void scatterTopology(const Obj<mesh_topology_type>& topology, const int dim, const Obj<mesh_topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        typedef typename Partitioner::part_type part_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<send_sizer_type>   sendSizer     = new send_sizer_type(topology->comm(), topology->debug());
        Obj<recv_sizer_type>   recvSizer     = new recv_sizer_type(topology->comm(), sendSizer->getTag(), topology->debug());
        Obj<send_section_type> sendSection   = new send_section_type(topology->comm(), topology->debug());
        Obj<recv_section_type> recvSection   = new recv_section_type(topology->comm(), sendSection->getTag(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int numElements = topology->heightStratum(0, 0)->size();
        int rank        = topology->commRank();
        int debug       = topology->debug();

        // 1) Form partition point overlap a priori
        if (rank == 0) {
          for(int p = 1; p < sieve->commSize(); p++) {
            // The arrow is from local partition point p to remote partition point p on rank p
            sendOverlap->addCone(p, p, p);
          }
        } else {
          // The arrow is from remote partition point rank on rank 0 to local partition point rank
          recvOverlap->addCone(0, rank, rank);
        }
        if (debug) {
          sendOverlap->view("Send overlap for partition");
          recvOverlap->view("Receive overlap for partition");
        }
        // 2) Partition the mesh
        part_type *assignment;

        assignment = Partitioner::partitionSieve(topology, dim);
        // 3) Create local sieve
        for(int e = 0; e < numElements; e++) {
          if (assignment[e] == rank) {
            const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(e);

            for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              sieveNew->addArrow(*c_iter, e, c_iter.color());
            }
          }
        }
        sieveNew->stratify();
        // 2) Complete sizer section
        typedef typename ALE::New::PartitionSizeSection<topology_type, part_type> partition_size_section;
        typedef typename ALE::New::PartitionSection<topology_type, part_type>     partition_section;
        Obj<topology_type>          secTopology          = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, numElements, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, numElements, assignment);

        ALE::New::Completion<mesh_topology_type,value_type>::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // 3) Unpack the section into the overlap
        sendOverlap->clear();
        recvOverlap->clear();
        if (rank == 0) {
          const topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

          for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
            const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

            for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
              const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
              int size = sendSection->size(p_iter->first, *b_iter);

              for(int p = 0; p < size; p++) {
                sendOverlap->addArrow(points[p], p_iter->first, points[p]);
              }
            }
          }
        } else {
          const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

          for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
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
        }
        if (debug) {
          sendOverlap->view(std::cout, "Send overlap for points");
          recvOverlap->view(std::cout, "Receive overlap for points");
        }
        // 4) Receive the point section
        secTopology = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(secTopology, sieve);
        Obj<cone_section>      coneSection     = new cone_section(secTopology, sieve);

        ALE::New::Completion<mesh_topology_type,value_type>::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // 5) Unpack the section into the sieve
        const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sieveNew->addArrow(points[p], *b_iter, c++);
            }
          }
        }
        sieveNew->stratify();
        topologyNew->stratify();
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

    template<typename Topology_, typename Value_ = int>
    class NewNumbering : public UniformSection<Topology_, Value_> {
    public:
      typedef UniformSection<Topology_, Value_>  base_type;
      typedef Topology_                          topology_type;
      typedef Value_                             value_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
      typedef typename base_type::atlas_type     atlas_type;
    protected:
      int                       _localSize;
      int                      *_offsets;
      std::map<int, point_type> _invOrder;
    public:
      NewNumbering(const Obj<topology_type>& topology) : UniformSection<Topology_, Value_>(topology), _localSize(0) {
        this->_offsets    = new int[this->commSize()+1];
        this->_offsets[0] = 0;
      };
      ~NewNumbering() {
        delete [] this->_offsets;
      };
    public: // Sizes
      int        getLocalSize() const {return this->_localSize;};
      void       setLocalSize(const int size) {this->_localSize = size;};
      int        getGlobalSize() const {return this->_offsets[this->commSize()];};
      int        getGlobalOffset(const int p) const {return this->_offsets[p];};
      const int *getGlobalOffsets() const {return this->_offsets;};
      void       setGlobalOffsets(const int offsets[]) {
        for(int p = 0; p <= this->commSize(); ++p) {
          this->_offsets[p] = offsets[p];
        }
      };
    public: // Indices
      virtual int getIndex(const point_type& point) {
        return getIndex(0, point);
      };
      virtual int getIndex(const patch_type& patch, const point_type& point) {
        const value_type& idx = this->restrictPoint(patch, point)[0];
        if (idx >= 0) {
          return idx;
        }
        return -(idx+1);
      };
      virtual void setIndex(const point_type& point, const int index) {this->updatePoint(0, point, &index);};
      virtual bool isLocal(const point_type& point) {return this->restrictPoint(0, point)[0] >= 0;};
      virtual bool isRemote(const point_type& point) {return this->restrictPoint(0, point)[0] < 0;};
      point_type getPoint(const int& index) {return this->_invOrder[index];};
      void setPoint(const int& index, const point_type& point) {this->_invOrder[index] = point;};
    };

    template<typename Topology_>
    class NewGlobalOrder : public UniformSection<Topology_, ALE::Point> {
    public:
      typedef UniformSection<Topology_, ALE::Point> base_type;
      typedef Topology_                          topology_type;
      typedef ALE::Point                         value_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
      typedef typename base_type::atlas_type     atlas_type;
    protected:
      int                       _localSize;
      int                      *_offsets;
      //std::map<int, point_type> _invOrder;
    public:
      NewGlobalOrder(const Obj<topology_type>& topology) : UniformSection<Topology_, ALE::Point>(topology), _localSize(0) {
        this->_offsets    = new int[this->commSize()+1];
        this->_offsets[0] = 0;
      };
      ~NewGlobalOrder() {
        delete [] this->_offsets;
      };
    public: // Sizes
      int        getLocalSize() const {return this->_localSize;};
      void       setLocalSize(const int size) {this->_localSize = size;};
      int        getGlobalSize() const {return this->_offsets[this->commSize()];};
      int        getGlobalOffset(const int p) const {return this->_offsets[p];};
      const int *getGlobalOffsets() const {return this->_offsets;};
      void       setGlobalOffsets(const int offsets[]) {
        for(int p = 0; p <= this->commSize(); ++p) {
          this->_offsets[p] = offsets[p];
        }
      };
    public: // Indices
      virtual int getIndex(const point_type& point) {
        return getIndex(0, point);
      };
      virtual int getIndex(const patch_type& patch, const point_type& point) {
        if (this->restrictPoint(0, point)[0].prefix >= 0) {
          return this->restrictPoint(0, point)[0].prefix;
        }
        return -(this->restrictPoint(0, point)[0].prefix+1);
      };
      virtual void setIndex(const point_type& point, const int index) {
        const value_type idx(index, this->restrictPoint(0, point)[0].index);
        this->updatePoint(0, point, &idx);
      };
      virtual bool isLocal(const point_type& point) {return this->restrictPoint(0, point)[0].prefix >= 0;};
      virtual bool isRemote(const point_type& point) {return this->restrictPoint(0, point)[0].prefix < 0;};
    };

    // We have a dichotomy between \emph{types}, describing the structure of objects,
    //   and \emph{concepts}, describing the role these objects play in the algorithm.
    //   Below we identify concepts with potential implementing types.
    //
    //   Concept           Type
    //   -------           ----
    //   Overlap           Sifter
    //   Atlas             ConstantSection, UniformSection
    //   Numbering         UniformSection
    //   GlobalOrder       UniformSection
    //
    // We will use factory types to create objects which satisfy a given concept.
    template<typename Topology_, typename Value_ = int>
    class NumberingFactory {
    public:
      typedef Topology_                                                           topology_type;
      typedef Value_                                                              value_type;
      typedef typename topology_type::patch_type                                  patch_type;
      typedef typename topology_type::send_overlap_type                           send_overlap_type;
      typedef typename topology_type::recv_overlap_type                           recv_overlap_type;
      typedef typename ALE::New::NewNumbering<topology_type, value_type>          numbering_type;
      typedef std::map<int, Obj<numbering_type> >                                 depthMap_type;
      typedef std::map<patch_type, depthMap_type>                                 patchMap_type;
      typedef std::map<topology_type*, patchMap_type>                             numberings_type;
      typedef typename ALE::New::NewGlobalOrder<topology_type>                    order_type;
      typedef std::map<std::string, Obj<order_type> >                             sectionMap_type;
      typedef std::map<patch_type, sectionMap_type>                               oPatchMap_type;
      typedef std::map<topology_type*, oPatchMap_type>                            orders_type;
      typedef typename order_type::value_type                                     oValue_type;
    protected:
      int             _debug;
      numberings_type _localNumberings;
      numberings_type _numberings;
      orders_type     _orders;
      value_type      _unknownNumber;
      ALE::Point      _unknownOrder;
    protected:
      NumberingFactory(const int debug) : _debug(debug), _unknownNumber(-1), _unknownOrder(-1, 0) {};
    public:
      ~NumberingFactory() {};
    public:
      static const Obj<NumberingFactory>& singleton(const int debug, bool cleanup = false) {
        static Obj<NumberingFactory> *_singleton = NULL;

        if (cleanup) {
          if (debug) {std::cout << "Destroying NumberingFactory" << std::endl;}
          if (_singleton) {delete _singleton;}
          _singleton = NULL;
        } else if (_singleton == NULL) {
          if (debug) {std::cout << "Creating new NumberingFactory" << std::endl;}
          _singleton  = new Obj<NumberingFactory>();
          *_singleton = new NumberingFactory(debug);
        }
        return *_singleton;
      };
    public:
      const int debug() {return this->_debug;};
    public:
      // Number all local points
      //   points in the overlap are only numbered by the owner with the lowest rank
      template<typename Sequence>
      void constructLocalNumbering(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const patch_type& patch, const Obj<Sequence>& points) {
        int localSize = 0;

        numbering->setFiberDimension(patch, points, 1);
        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          value_type val;

          if (sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
            int minRank = sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < sendOverlap->commRank()) {
              val = this->_unknownNumber;
            } else {
              val = localSize++;
            }
          } else {
            val = localSize++;
          }
          numbering->updatePoint(patch, *l_iter, &val);
        }
        numbering->setLocalSize(localSize);
      };
      // Order all local points
      //   points in the overlap are only ordered by the owner with the lowest rank
      template<typename Sequence, typename Atlas>
      void constructLocalOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const patch_type& patch, const Obj<Sequence>& points, const Obj<Atlas>& atlas) {
        int localSize = 0;

        order->setFiberDimension(patch, points, 1);
        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          oValue_type val;

          if (sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
            int minRank = sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < sendOverlap->commRank()) {
              val = this->_unknownOrder;
            } else {
              val.prefix = localSize;
              val.index  = atlas->restrict(patch, *l_iter)[0].prefix;
            }
          } else {
            val.prefix = localSize;
            val.index  = atlas->restrict(patch, *l_iter)[0].prefix;
          }
          localSize += val.index;
          order->updatePoint(patch, *l_iter, &val);
        }
        order->setLocalSize(localSize);
      };
      // Calculate process offsets
      template<typename Numbering>
      void calculateOffsets(const Obj<Numbering>& numbering) {
        int  localSize = numbering->getLocalSize();
        int *offsets   = new int[numbering->commSize()+1];

        offsets[0] = 0;
        MPI_Allgather(&localSize, 1, MPI_INT, &(offsets[1]), 1, MPI_INT, numbering->comm());
        for(int p = 2; p <= numbering->commSize(); p++) {
          offsets[p] += offsets[p-1];
        }
        numbering->setGlobalOffsets(offsets);
        delete [] offsets;
      };
      // Update local offsets based upon process offsets
      template<typename Numbering, typename Sequence>
      void updateOrder(const Obj<Numbering>& numbering, const patch_type& patch, const Obj<Sequence>& points) {
        const typename Numbering::value_type val = numbering->getGlobalOffset(numbering->commRank());

        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (numbering->isLocal(*l_iter)) {
            numbering->updateAddPoint(patch, *l_iter, &val);
          }
        }
      };
      // Communicate numbers in the overlap
      void complete(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, bool allowDuplicates = false) {
        typedef typename Completion<topology_type, int>::topology_type topo_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topo_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topo_type, value_type> recv_section_type;
        typedef typename ALE::New::ConstantSection<topology_type, int> constant_sizer;
        const Obj<send_section_type> sendSection = new send_section_type(numbering->comm(), this->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(numbering->comm(), sendSection->getTag(), this->debug());
        //const Obj<constant_sizer>    sizer       = new constant_sizer(numbering->comm(), 1, this->debug());

        Completion<topology_type, int>::completeSection(sendOverlap, recvOverlap, numbering->getAtlas(), numbering, sendSection, recvSection);
        const typename recv_section_type::topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(typename recv_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename recv_section_type::patch_type& rPatch = p_iter->first;
          const typename recv_section_type::chart_type& points = recvSection->getPatch(rPatch);

          for(typename recv_section_type::chart_type::iterator r_iter = points.begin(); r_iter != points.end(); ++r_iter) {
            const typename recv_section_type::point_type& point  = *r_iter;
            const typename recv_section_type::value_type *values = recvSection->restrict(rPatch, point);

            if (recvSection->getFiberDimension(rPatch, point) == 0) continue;
            if (values[0] >= 0) {
              if (numbering->isLocal(point) && !allowDuplicates) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Multiple indices for point " << point << " from " << rPatch << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              if (numbering->getAtlas()->getFiberDimension(0, point) == 0) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Unexpected point " << point << " from " << rPatch << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              int val = -(values[0]+1);
              numbering->updatePoint(patch, point, &val);
            }
          }
        }
#if 0
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          numbering->setFiberDimension(0, *r_iter, 1);
        }
        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

            if (values[0] >= 0) {
              if (numbering->isLocal(*r_iter) && !allowDuplicates) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Multiple indices for point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              if (numbering->getAtlas()->getFiberDimension(0, *r_iter) == 0) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Unexpected point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              int val = -(values[0]+1);
              numbering->updatePoint(patch, *r_iter, &val);
            }
          }
        }
#endif
        PetscErrorCode ierr;
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
      };
      // Communicate (size,offset)s in the overlap
      void completeOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, bool allowDuplicates = false) {
        typedef typename Completion<topology_type, int>::topology_type topo_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topo_type, oValue_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topo_type, oValue_type> recv_section_type;
        typedef typename ALE::New::ConstantSection<topology_type, int> constant_sizer;
        const Obj<send_section_type> sendSection = new send_section_type(order->comm(), this->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(order->comm(), sendSection->getTag(), this->debug());
        const Obj<constant_sizer>    sizer       = new constant_sizer(order->comm(), 1, this->debug());

        Completion<topology_type, int>::completeSection(sendOverlap, recvOverlap, sizer, order, sendSection, recvSection);
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          if (!order->hasPoint(patch, *r_iter)) {
            order->setFiberDimension(patch, *r_iter, 1);
            order->update(patch, *r_iter, &this->_unknownOrder);
          }
        }
        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

            if (values[0].index == 0) continue;
            if (values[0].prefix >= 0) {
              if (order->isLocal(*r_iter)) {
                if (!allowDuplicates) {
                  ostringstream msg;
                  msg << "["<<order->commRank()<<"]Multiple indices for point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                  throw ALE::Exception(msg.str().c_str());
                }
                continue;
              }
              const oValue_type val(-(values[0].prefix+1), values[0].index);
              order->updatePoint(patch, *r_iter, &val);
            } else {
              if (order->isLocal(*r_iter)) continue;
              order->updatePoint(patch, *r_iter, values);
            }
          }
        }
      };
      // Construct a full global numbering
      template<typename Sequence>
      void construct(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, const Obj<Sequence>& points) {
        this->constructLocalNumbering(numbering, sendOverlap, patch, points);
        this->calculateOffsets(numbering);
        this->updateOrder(numbering, patch, points);
        this->complete(numbering, sendOverlap, recvOverlap, patch);
      };
      // Construct a full global order
      template<typename Sequence, typename Atlas>
      void constructOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, const Obj<Sequence>& points, const Obj<Atlas>& atlas) {
        this->constructLocalOrder(order, sendOverlap, patch, points, atlas);
        this->calculateOffsets(order);
        this->updateOrder(order, patch, points);
        this->completeOrder(order, sendOverlap, recvOverlap, patch);
      };
      // Construct the inverse map from numbers to points
      //   If we really need this, then we should consider using a label
      void constructInverseOrder(const Obj<numbering_type>& numbering) {
        const typename numbering_type::chart_type& patch = numbering->getAtlas()->getPatch(0);

        for(typename numbering_type::chart_type::iterator p_iter = patch.begin(); p_iter != patch.end(); ++p_iter) {
          numbering->setPoint(numbering->getIndex(*p_iter), *p_iter);
        }
      };
    public:
      const Obj<numbering_type>& getLocalNumbering(const Obj<topology_type>& topology, const patch_type& patch, const int depth) {
        if ((this->_localNumberings.find(topology.ptr()) == this->_localNumberings.end()) ||
            (this->_localNumberings[topology.ptr()].find(patch) == this->_localNumberings[topology.ptr()].end()) ||
            (this->_localNumberings[topology.ptr()][patch].find(depth) == this->_localNumberings[topology.ptr()][patch].end())) {
          Obj<numbering_type>    numbering   = new numbering_type(topology);
          // These go in the Topology soon
          Obj<send_overlap_type> sendOverlap = new send_overlap_type(topology->comm(), topology->debug());
          //Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(this->comm(), this->debug());

          this->constructLocalNumbering(numbering, sendOverlap, patch, topology->getLabelStratum(patch, "depth", depth));
          if (this->_debug) {std::cout << "Creating new local numbering: patch " << patch << " depth " << depth << std::endl;}
          this->_localNumberings[topology.ptr()][patch][depth] = numbering;
        }
        return this->_localNumberings[topology.ptr()][patch][depth];
      };
      const Obj<numbering_type>& getNumbering(const Obj<topology_type>& topology, const patch_type& patch, const int depth) {
        if ((this->_numberings.find(topology.ptr()) == this->_numberings.end()) ||
            (this->_numberings[topology.ptr()].find(patch) == this->_numberings[topology.ptr()].end()) ||
            (this->_numberings[topology.ptr()][patch].find(depth) == this->_numberings[topology.ptr()][patch].end())) {
          topology->constructOverlap(patch);
          Obj<numbering_type>    numbering   = new numbering_type(topology);
          Obj<send_overlap_type> sendOverlap = topology->getSendOverlap();
          Obj<recv_overlap_type> recvOverlap = topology->getRecvOverlap();

          this->construct(numbering, sendOverlap, recvOverlap, patch, topology->getLabelStratum(patch, "depth", depth));
          if (this->_debug) {std::cout << "Creating new numbering: patch " << patch << " depth " << depth << std::endl;}
          this->_numberings[topology.ptr()][patch][depth] = numbering;
        }
        return this->_numberings[topology.ptr()][patch][depth];
      };
      template<typename Atlas>
      const Obj<order_type>& getGlobalOrder(const Obj<topology_type>& topology, const patch_type& patch, const std::string& name, const Obj<Atlas>& atlas) {
        if ((this->_orders.find(topology.ptr()) == this->_orders.end()) ||
            (this->_orders[topology.ptr()].find(patch) == this->_orders[topology.ptr()].end()) ||
            (this->_orders[topology.ptr()][patch].find(name) == this->_orders[topology.ptr()][patch].end())) {
          topology->constructOverlap(patch);
          Obj<order_type>        order       = new order_type(topology);
          Obj<send_overlap_type> sendOverlap = topology->getSendOverlap();
          Obj<recv_overlap_type> recvOverlap = topology->getRecvOverlap();

          // FIX:
          //this->constructOrder(order, sendOverlap, recvOverlap, patch, atlas->getPatch(patch), atlas);
          this->constructOrder(order, sendOverlap, recvOverlap, patch, topology->depthStratum(patch, 0), atlas);
          if (this->_debug) {std::cout << "Creating new global order: patch " << patch << " name " << name << std::endl;}
          this->_orders[topology.ptr()][patch][name] = order;
        }
        return this->_orders[topology.ptr()][patch][name];
      };
    };
  }
}
#endif
