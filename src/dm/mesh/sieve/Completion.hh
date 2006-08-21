#ifndef included_ALE_Completion_hh
#define included_ALE_Completion_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

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
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getAtlas()->getFiberDimension(this->_patch, p); // Could be size()
        return &this->_size;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getAtlas()->getFiberDimension(this->_patch, p);
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
    protected:
      Obj<section_type> _section;
      const patch_type  _patch;
    public:
      PatchlessSection(const Obj<section_type>& section, const patch_type& patch) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section), _patch(patch) {};
      virtual ~PatchlessSection() {};
    public:
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
      void allocate() {};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a PartitionSizeSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if (patch != p) {
          throw ALE::Exception("Point must be identical to patch in a PartitionSizeSection");
        }
        return &this->_sizes[patch];
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
    protected:
      Obj<topology_type> _topology;
      points_type        _points;
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
      void allocate() {};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a PartitionSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if (patch != p) {
          throw ALE::Exception("Point must be identical to patch in a PartitionSection");
        }
        return this->_points[patch];
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
      void allocate() {};
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
    protected:
      Obj<topology_type>      _topology;
      Obj<cone_sieve_type>    _sieve;
      int                     _coneSize;
      value_type             *_cone;
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
      void allocate() {};
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
      typedef typename ALE::New::Atlas<topology_type, ALE::Point>                         atlas_type;
      typedef typename ALE::Sifter<int, point_type, point_type>                           send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, atlas_type, int>        send_sizer_type;
      typedef typename ALE::Sifter<point_type, int, point_type>                           recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, int>        recv_sizer_type;
      typedef typename ALE::New::ConstantSection<topology_type, int>                      constant_sizer;
      typedef typename ALE::New::ConstantSection<topology_type, value_type>               constant_section;
      typedef typename ALE::New::PartitionSizeSection<topology_type, short int>           partition_size_section;
      typedef typename ALE::New::PartitionSection<topology_type, short int>               partition_section;
      typedef typename ALE::New::ConeSizeSection<topology_type, sieve_type>               cone_size_section;
      typedef typename ALE::New::ConeSection<topology_type, sieve_type>                   cone_section;
    public:
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
        sendSection->getAtlas()->clear();
        sendSection->getAtlas()->setTopology(ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap));
        if (sendSection->debug() > 10) {sendSection->getAtlas()->getTopology()->view("Send topology after setup", MPI_COMM_SELF);}
        sendSection->construct(sendSizer);
        sendSection->getAtlas()->orderPatches();
        sendSection->allocate();
        sendSection->constructCommunication(Section::SEND);
      };
      template<typename Filler, typename Section>
      static void fillSend(const Obj<Filler>& sendFiller, const Obj<Section>& sendSection) {
        const topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();
        const topology_type::patch_type  patch   = 0; // FIX: patch should come from overlap

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            sendSection->update(p_iter->first, *b_iter, sendFiller->restrict(patch, *b_iter));
          }
        }
      };
      template<typename Filler, typename Section>
      static void completeSend(const Obj<Filler>& sendFiller, const Obj<Section>& sendSection) {
        // Fill section
        const topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            sendSection->update(p_iter->first, *b_iter, sendFiller->restrict(p_iter->first, *b_iter));
          }
        }
        if (sendSection->debug()) {sendSection->view("Send Section in Completion", MPI_COMM_SELF);}
        // Complete the section
        sendSection->startCommunication();
        sendSection->endCommunication();
      };
      template<typename SizerFiller, typename Filler, typename Section>
      static void sendSection(const Obj<send_overlap_type>& sendOverlap, const Obj<SizerFiller>& sizerFiller, const Obj<Filler>& filler, const Obj<Section>& sendSection) {
        Obj<send_sizer_type>   sendSizer     = new send_sizer_type(sendSection->comm(), sendSection->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, sendSection->debug());

        // 1) Create the sizer section
        ALE::New::Completion<mesh_topology_type,int>::setupSend(sendOverlap, constantSizer, sendSizer);
        // 2) Fill the sizer section and communicate
        ALE::New::Completion<mesh_topology_type,int>::completeSend(sizerFiller, sendSizer);
        // 3) Create the send section
        ALE::New::Completion<mesh_topology_type,value_type>::setupSend(sendOverlap, sendSizer, sendSection);
        // 4) Fill up send section and communicate
        ALE::New::Completion<mesh_topology_type,value_type>::completeSend(filler, sendSection);
      };
      #undef __FUNCT__
      #define __FUNCT__ "sendDistribution"
      static Obj<send_overlap_type> sendDistribution(const Obj<mesh_topology_type>& topology, const int dim, const Obj<mesh_topology_type>& topologyNew) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename ALE::New::OverlapValues<send_overlap_type, atlas_type, value_type> send_section_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<send_overlap_type> sendOverlap   = new send_overlap_type(topology->comm(), topology->debug());
        Obj<send_sizer_type>   sendSizer     = new send_sizer_type(topology->comm(), topology->debug());
        Obj<send_section_type> sendSection   = new send_section_type(topology->comm(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int numElements = topology->heightStratum(0, 0)->size();
        int rank        = topology->commRank();
        int debug       = topology->debug();

        // 1) Form partition point overlap a priori
        //      There are arrows to each rank whose color is the partition point (also the rank)
        for(int p = 1; p < sieve->commSize(); p++) {
          sendOverlap->addCone(p, p, p);
        }
        if (debug) {sendOverlap->view(std::cout, "Send overlap for partition");}
        // 2) Partition the mesh
        short *assignment = ALE::New::Partitioner<mesh_topology_type>::partitionSieve_Chaco(topology, dim);
        // 3) Create local sieve
        for(int e = 0; e < numElements; e++) {
          if (assignment[e] == rank) {
            const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(e);

            for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              sieveNew->addArrow(*c_iter, e, c_iter.color());
            }
          }
        }
        // 2) Send the point section
        Obj<topology_type>          secTopology          = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, numElements, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, numElements, assignment);
        ALE::New::Completion<mesh_topology_type,value_type>::sendSection(sendOverlap, partitionSizeSection, partitionSection, sendSection);
        // 3) Create point overlap
        // Could this potentially be the sendSection itself?
        sendOverlap->clear();
        const topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
            int size = sendSection->getAtlas()->size(p_iter->first, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        if (debug) {sendOverlap->view(std::cout, "Send overlap for points");}
        // 4) Send the point section
        secTopology = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(secTopology, sieve);
        Obj<cone_section>      coneSection     = new cone_section(secTopology, sieve);
        ALE::New::Completion<mesh_topology_type,value_type>::sendSection(sendOverlap, coneSizeSection, coneSection, sendSection);
        topologyNew->stratify();
        ALE_LOG_EVENT_END;
        return sendOverlap;
      };
      template<typename Sizer, typename Section>
      static void setupReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<Section>& recvSection) {
        // Create section
        const Obj<recv_overlap_type::traits::capSequence> ranks = recvOverlap->cap();

        recvSection->getAtlas()->clear();
        for(recv_overlap_type::traits::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> recvSieve = new dsieve_type();
          const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(*r_iter);

          // Want to replace this loop with a slice through color
          for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            recvSieve->addPoint(p_iter.color());
          }
          recvSection->getAtlas()->getTopology()->setPatch(*r_iter, recvSieve);
        }
        recvSection->getAtlas()->getTopology()->stratify();
        recvSection->construct(recvSizer);
        recvSection->getAtlas()->orderPatches();
        recvSection->allocate();
        recvSection->constructCommunication(Section::RECEIVE);
      };
      template<typename Section>
      static void completeReceive(const Obj<Section>& recvSection) {
        // Complete the section
        recvSection->startCommunication();
        recvSection->endCommunication();
        if (recvSection->debug()) {recvSection->view("Receive Section in Completion", MPI_COMM_SELF);}
        // Read out section values
      };
      template<typename Section>
      static void recvSection(const Obj<recv_overlap_type>& recvOverlap, const Obj<Section>& recvSection) {
        Obj<recv_sizer_type> recvSizer     = new recv_sizer_type(recvSection->comm(), recvSection->debug());
        Obj<constant_sizer>  constantSizer = new constant_sizer(MPI_COMM_SELF, 1, recvSection->debug());

        // 1) Create the sizer section
        ALE::New::Completion<mesh_topology_type,int>::setupReceive(recvOverlap, constantSizer, recvSizer);
        // 2) Communicate
        ALE::New::Completion<mesh_topology_type,int>::completeReceive(recvSizer);
        // 3) Update to the receive section
        ALE::New::Completion<mesh_topology_type,value_type>::setupReceive(recvOverlap, recvSizer, recvSection);
        // 4) Communicate
        ALE::New::Completion<mesh_topology_type,value_type>::completeReceive(recvSection);
      };
      static Obj<recv_overlap_type> receiveDistribution(const Obj<mesh_topology_type>& topology, const Obj<mesh_topology_type>& topologyNew) {
        typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, value_type> recv_section_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(topology->comm(), topology->debug());
        Obj<recv_sizer_type>   recvSizer   = new recv_sizer_type(topology->comm(), topology->debug());
        Obj<recv_section_type> recvSection = new recv_section_type(topology->comm(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int debug = topology->debug();

        // 1) Form partition point overlap a priori
        //      The arrow is from rank 0 with partition point 0
        recvOverlap->addCone(0, sieve->commRank(), sieve->commRank());
        if (debug) {recvOverlap->view(std::cout, "Receive overlap for partition");}
        // 2) Receive sizer section
        ALE::New::Completion<mesh_topology_type,value_type>::recvSection(recvOverlap, recvSection);
        // 3) Unpack the section into the overlap
        recvOverlap->clear();
        const topology_type::sheaf_type& patches = recvSection->getAtlas()->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
        if (debug) {recvOverlap->view(std::cout, "Receive overlap for points");}
        // 4) Receive the point section
        ALE::New::Completion<mesh_topology_type,value_type>::recvSection(recvOverlap, recvSection);
        // 5) Unpack the section into the sieve
        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                     rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);
            int c = 0;

            for(int p = 0; p < size; p++) {
              sieveNew->addArrow(points[p], *b_iter, c++);
            }
          }
        }
        topologyNew->stratify();
        return recvOverlap;
      };
      static Obj<send_overlap_type> sendDistribution2(const Obj<mesh_topology_type>& topology, const int dim, const Obj<mesh_topology_type>& topologyNew) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, atlas_type, value_type> send_section_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<send_overlap_type> sendOverlap   = new send_overlap_type(topology->comm(), topology->debug());
        Obj<send_sizer_type>   sendSizer     = new send_sizer_type(topology->comm(), topology->debug());
        Obj<send_section_type> sendSection   = new send_section_type(topology->comm(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int numElements = topology->heightStratum(0, 0)->size();
        int rank        = topology->commRank();
        int debug       = topology->debug();

        // 1) Form partition point overlap a priori
        //      The arrow is to rank 0 with partition point p
        sendOverlap->addCone(sieve->commRank(), 0, sieve->commRank());
        if (debug) {sendOverlap->view(std::cout, "Send overlap for partition");}
        // 2) Create simple partition
        short *assignment = new short[numElements];
        for(int e = 0; e < numElements; e++) assignment[e] = 0;
        // 2) Send the point section
        Obj<topology_type>          secTopology          = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, numElements, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, numElements, assignment);
        ALE::New::Completion<mesh_topology_type,value_type>::sendSection(sendOverlap, partitionSizeSection, partitionSection, sendSection);
        // 3) Create point overlap
        // Could this potentially be the sendSection itself?
        sendOverlap->clear();
        const topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
            int size = sendSection->getAtlas()->size(p_iter->first, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        if (debug) {sendOverlap->view(std::cout, "Send overlap for points");}
        // 4) Send the point section
        secTopology = ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(secTopology, sieve);
        Obj<cone_section>      coneSection     = new cone_section(secTopology, sieve);
        ALE::New::Completion<mesh_topology_type,value_type>::sendSection(sendOverlap, coneSizeSection, coneSection, sendSection);
        topologyNew->stratify();
        return sendOverlap;
      };
      static Obj<recv_overlap_type> receiveDistribution2(const Obj<mesh_topology_type>& topology, const Obj<mesh_topology_type>& topologyNew) {
        typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, value_type> recv_section_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(topology->comm(), topology->debug());
        Obj<recv_sizer_type>   recvSizer   = new recv_sizer_type(topology->comm(), topology->debug());
        Obj<recv_section_type> recvSection = new recv_section_type(topology->comm(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int debug = topology->debug();

        // 1) Form partition point overlap a priori
        //      There are arrows from each rank whose color is the partition point (also the rank)
        for(int p = 1; p < sieve->commSize(); p++) {
          recvOverlap->addCone(p, p, p);
        }
        recvOverlap->addCone(0, sieve->commRank(), sieve->commRank());
        if (debug) {recvOverlap->view(std::cout, "Receive overlap for partition");}
        // 2) Create local sieve
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(0, 0);

        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);

          for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            sieveNew->addArrow(*c_iter, *e_iter, c_iter.color());
          }
        }
        // 3) Receive sizer section
        ALE::New::Completion<mesh_topology_type,value_type>::recvSection(recvOverlap, recvSection);
        // 4) Unpack the section into the overlap
        recvOverlap->clear();
        const topology_type::sheaf_type& patches = recvSection->getAtlas()->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
        if (debug) {recvOverlap->view(std::cout, "Receive overlap for points");}
        // 4) Receive the point section
        ALE::New::Completion<mesh_topology_type,value_type>::recvSection(recvOverlap, recvSection);
        // 5) Unpack the section into the sieve
        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                     rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);
            int c = 0;

            for(int p = 0; p < size; p++) {
              sieveNew->addArrow(points[p], *b_iter, c++);
            }
          }
        }
        topologyNew->stratify();
        return recvOverlap;
      };
      template<typename SizerFiller, typename Filler, typename SendSection, typename RecvSection>
      static void completeSection(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<SizerFiller>& sizerFiller, const Obj<Filler>& filler, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
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
    };

    template<typename Atlas_>
    class NewNumbering : public Section<Atlas_, int> {
    public:
      typedef Atlas_                                          atlas_type;
      typedef typename atlas_type::topology_type              topology_type;
      typedef typename topology_type::patch_type              patch_type;
      typedef typename topology_type::point_type              point_type;
      typedef int                                             value_type;
      typedef typename Section<Atlas_, int>::values_type      values_type;
      typedef typename ALE::Sifter<int,point_type,point_type> send_overlap_type;
      typedef typename ALE::Sifter<point_type,int,point_type> recv_overlap_type;
    protected:
      std::string               _label;
      int                       _value;
      Obj<send_overlap_type>    _sendOverlap;
      Obj<recv_overlap_type>    _recvOverlap;
      int                       _localSize;
      int                      *_offsets;
      std::map<int, point_type> _invOrder;
    public:
      NewNumbering(const Obj<atlas_type>& atlas, const std::string& label, int value) : Section<Atlas_,int>(atlas), _label(label), _value(value), _localSize(0) {
        this->_sendOverlap = new send_overlap_type(this->comm(), this->debug());
        this->_recvOverlap = new recv_overlap_type(this->comm(), this->debug());
        this->_offsets     = new int[this->commSize()+1];
        this->_offsets[0]  = 0;
      };
      ~NewNumbering() {
        delete [] this->_offsets;
      };
    public: // Accessors
      std::string getLabel() const {return this->_label;};
      void        setLabel(const std::string& label) {this->_label = label;};
      int         getValue() const {return this->_value;};
      void        setValue(const int value) {this->_value = value;};
    public: // Sizes
      int        getLocalSize() const {return this->_localSize;};
      int        getGlobalSize() const {return this->_offsets[this->commSize()];};
      const int *getGlobalOffsets() const {return this->_offsets;};
    public: // Indices
      int getIndex(const point_type& point) {if (this->restrictPoint(0, point)[0] >= 0) return this->restrictPoint(0, point)[0]; else return -(this->restrictPoint(0, point)[0]+1);};
      point_type getPoint(const int& index) {return this->_invOrder[index];};
      bool isLocal(const point_type& point) {return this->restrictPoint(0, point)[0] >= 0;};
      bool isRemote(const point_type& point) {return this->restrictPoint(0, point)[0] < 0;};
    public:
      void constructOverlap() {
        const Obj<typename topology_type::label_sequence>& points = this->getAtlas()->getTopology()->getLabelStratum(0, this->_label, this->_value);

        point_type *sendBuf = new point_type[points->size()];
        int         size    = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          sendBuf[size++] = *l_iter;
        }
        int *sizes   = new int[this->commSize()];
        int *offsets = new int[this->commSize()+1];
        point_type *remotePoints = NULL;
        int        *remoteRanks  = NULL;

        // Change to Allgather() for the correct binning algorithm
        MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, this->comm());
        if (this->commRank() == 0) {
          offsets[0] = 0;
          for(int p = 1; p <= this->commSize(); p++) {
            offsets[p] = offsets[p-1] + sizes[p-1];
          }
          remotePoints = new point_type[offsets[this->commSize()]];
        }
        MPI_Gatherv(sendBuf, size, MPI_INT, remotePoints, sizes, offsets, MPI_INT, 0, this->comm());
        std::map<int, std::map<int, std::set<point_type> > > overlapInfo;

        if (this->commRank() == 0) {
          for(int p = 0; p < this->commSize(); p++) {
            std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]]);
          }
          for(int p = 0; p < this->commSize(); p++) {
            for(int q = p+1; q < this->commSize(); q++) {
              std::set_intersection(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]],
                                    &remotePoints[offsets[q]], &remotePoints[offsets[q+1]],
                                    std::insert_iterator<std::set<point_type> >(overlapInfo[p][q], overlapInfo[p][q].begin()));
              overlapInfo[q][p] = overlapInfo[p][q];
            }
            sizes[p]     = overlapInfo[p].size()*2;
            offsets[p+1] = offsets[p] + sizes[p];
          }
          remoteRanks = new int[offsets[this->commSize()]];
          int       k = 0;
          for(int p = 0; p < this->commSize(); p++) {
            for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
              remoteRanks[k*2]   = r_iter->first;
              remoteRanks[k*2+1] = r_iter->second.size();
              k++;
            }
          }
        }
        int numOverlaps;
        MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, this->comm());
        int *overlapRanks = new int[numOverlaps];
        MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, this->comm());
        if (this->commRank() == 0) {
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            sizes[p] = 0;
            for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
              sizes[p] += remoteRanks[k*2+1];
              k++;
            }
            offsets[p+1] = offsets[p] + sizes[p];
          }
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
              int rank = r_iter->first;
              for(typename std::set<point_type>::iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
                remotePoints[k++] = *p_iter;
              }
            }
          }
        }
        int numOverlapPoints = 0;
        for(int r = 0; r < numOverlaps/2; r++) {
          numOverlapPoints += overlapRanks[r*2+1];
        }
        point_type *overlapPoints = new point_type[numOverlapPoints];
        MPI_Scatterv(remotePoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints, MPI_INT, 0, this->comm());

        for(int r = 0, k = 0; r < numOverlaps/2; r++) {
          int rank = overlapRanks[r*2];

          for(int p = 0; p < overlapRanks[r*2+1]; p++) {
            point_type point = overlapPoints[k++];

            this->_sendOverlap->addArrow(point, rank, point);
            this->_recvOverlap->addArrow(rank, point, point);
          }
        }

        delete [] overlapPoints;
        delete [] overlapRanks;
        delete [] sizes;
        delete [] offsets;
        if (this->commRank() == 0) {
          delete [] remoteRanks;
          delete [] remotePoints;
        }
        if (this->debug()) {
          this->_sendOverlap->view("Send overlap");
          this->_recvOverlap->view("Receive overlap");
        }
      };
      void constructLocalOrder() {
        const patch_type patch = 0;
        const Obj<typename topology_type::label_sequence>& points = this->getAtlas()->getTopology()->getLabelStratum(patch, this->_label, this->_value);

        this->_atlas->setFiberDimensionByLabel(patch, this->_label, this->_value, 1);
        this->_atlas->orderPatches();
        this->allocate();
        this->_localSize = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          int val;

          if (this->_sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = this->_sendOverlap->support(*l_iter);
            int minRank = this->_sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < this->_sendOverlap->commRank()) {
              val = -1;
            } else {
              val = this->_localSize++;
            }
          } else {
            val = this->_localSize++;
          }
          this->update(patch, *l_iter, &val);
        }
      };
      void constructInverseOrder() {
        for(typename std::map<point_type, int>::iterator p_iter = this->_order.begin(); p_iter != this->_order.end(); ++p_iter) {
          this->_invOrder[this->getIndex(p_iter->first)] = p_iter->first;
        }
      };
      void calculateOffsets() {
        MPI_Allgather(&this->_localSize, 1, MPI_INT, &(this->_offsets[1]), 1, MPI_INT, this->comm());
        for(int p = 2; p <= this->commSize(); p++) {
          this->_offsets[p] += this->_offsets[p-1];
        }
      };
      void updateOrder() {
        const patch_type patch = 0;
        const Obj<typename topology_type::label_sequence>& points = this->getAtlas()->getTopology()->getLabelStratum(patch, this->_label, this->_value);
        const int val = this->_offsets[this->commRank()];

        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (this->isLocal(*l_iter)) {
            this->updateAdd(patch, *l_iter, &val);
          }
        }
      };
      void complete() {
        typedef typename Completion<topology_type, int>::atlas_type atlas_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, atlas_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, value_type> recv_section_type;
        typedef typename ALE::New::ConstantSection<topology_type, int> constant_sizer;
        const Obj<send_section_type> sendSection = new send_section_type(this->comm(), this->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(this->comm(), sendSection->getTag(), this->debug());
        const Obj<constant_sizer>    sizer       = new constant_sizer(this->comm(), 1, this->debug());
        Obj<NewNumbering<Atlas_> > filler(this);
        (*filler.refCnt)++;

        Completion<topology_type, int>::completeSection(this->_sendOverlap, this->_recvOverlap, sizer, filler, sendSection, recvSection);
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = this->_recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = this->_recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

            if (values[0] >= 0) {
              if (this->isLocal(*r_iter)) {
                ostringstream msg;
                msg << "Multiple indices for point " << *r_iter;
                throw ALE::Exception(msg.str().c_str());
              }
              int val = -(values[0]+1);
              this->update(0, *r_iter, &val);
            }
          }
        }
      };
      void construct() {
        this->constructOverlap();
        this->constructLocalOrder();
        this->calculateOffsets();
        this->updateOrder();
        this->complete();
      };
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
            txt << "viewing a Numbering" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing Numbering '" << name << "'" << std::endl;
          }
        }
        for(typename values_type::const_iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          const patch_type  patch = a_iter->first;
          const value_type *array = a_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          const typename atlas_type::chart_type& chart = this->_atlas->getChart(patch);

          for(typename atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename atlas_type::point_type& p   = c_iter->first;
            const typename atlas_type::index_type& idx = c_iter->second;

            if (idx.index != 0) {
              txt << "[" << this->commRank() << "]:   " << p << " --> ";
              if (array[idx.prefix] >= 0) {
                txt << array[idx.prefix];
              } else {
                txt << -(array[idx.prefix]+1) << " (global)";
              }
              txt << std::endl;
            }
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_>
    class Numbering : public ParallelObject {
    public:
      typedef Topology_                                                                           topology_type;
      typedef typename topology_type::point_type                                                  point_type;
      typedef typename topology_type::sieve_type                                                  sieve_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                                        dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                                       overlap_topology_type;
      typedef typename ALE::New::Atlas<overlap_topology_type, ALE::Point>                         overlap_atlas_type;
      typedef typename ALE::Sifter<int,point_type,point_type>                                     send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, overlap_atlas_type, point_type> send_section_type;
      typedef typename ALE::Sifter<point_type,int,point_type>                                     recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, overlap_atlas_type, point_type> recv_section_type;
    protected:
      Obj<topology_type>        _topology;
      std::string               _label;
      int                       _value;
      std::map<point_type, int> _order;
      std::map<int, point_type> _invOrder;
      Obj<send_overlap_type>    _sendOverlap;
      Obj<recv_overlap_type>    _recvOverlap;
      Obj<send_section_type>    _sendSection;
      Obj<recv_section_type>    _recvSection;
      int                       _localSize;
      int                      *_offsets;
    public:
      Numbering(const Obj<topology_type>& topology, const std::string& label, int value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _label(label), _value(value), _localSize(0) {
        this->_sendOverlap = new send_overlap_type(this->comm(), this->debug());
        this->_recvOverlap = new recv_overlap_type(this->comm(), this->debug());
        this->_sendSection = new send_section_type(this->comm(), this->debug());
        this->_recvSection = new recv_section_type(this->comm(), this->_sendSection->getTag(), this->debug());
        this->_offsets     = new int[this->commSize()+1];
        this->_offsets[0]  = 0;
      };
      ~Numbering() {
        delete [] this->_offsets;
      };
    public: // Accessors
      const Obj<topology_type>& getTopology() {return this->_topology;};
      void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
      std::string getLabel() {return this->_label;};
      void setLabel(const std::string& label) {this->_label = label;};
      int getValue() {return this->_value;};
      void setValue(const int value) {this->_value = value;};
    public: // Sizes
      int getLocalSize() const {return this->_localSize;};
      int getGlobalSize() const {return this->_offsets[this->commSize()];};
      const int *getGlobalOffsets() const {return this->_offsets;};
      int getIndex(const point_type& point) {if (this->_order[point] >= 0) return this->_order[point]; else return -(this->_order[point]+1);};
      point_type getPoint(const int& index) {return this->_invOrder[index];};
      bool isLocal(const point_type& point) {return this->_order[point] >= 0;};
      bool isRemote(const point_type& point) {return this->_order[point] < 0;};
      const Obj<send_overlap_type>& getSendOverlap() {return this->_sendOverlap;};
      void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
      const Obj<recv_overlap_type>& getRecvOverlap() {return this->_recvOverlap;};
      void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
      const Obj<send_section_type>& getSendSection() {return this->_sendSection;};
      void setSendSection(const Obj<send_section_type>& section) {this->_sendSection = section;};
      const Obj<recv_section_type>& getRecvSection() {return this->_recvSection;};
      void setRecvSection(const Obj<recv_section_type>& section) {this->_recvSection = section;};
    public: // Construction
      void setLocalSize(const int size) {this->_localSize = size;};
      void setIndex(const point_type& point, const int index) {this->_order[point] = index;};
      void calculateOffsets() {
        MPI_Allgather(&this->_localSize, 1, MPI_INT, &(this->_offsets[1]), 1, MPI_INT, this->comm());
        for(int p = 2; p <= this->commSize(); p++) {
          this->_offsets[p] += this->_offsets[p-1];
        }
      };
      void updateOrder() {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);

        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (this->_order[*l_iter] >= 0) {
            this->_order[*l_iter] += this->_offsets[this->commRank()];
          }
        }
      };
      void copyCommunication(const Obj<Numbering<topology_type> >& numbering) {
        this->setSendOverlap(numbering->getSendOverlap());
        this->setRecvOverlap(numbering->getRecvOverlap());
        this->setSendSection(numbering->getSendSection());
        this->setRecvSection(numbering->getRecvSection());
      }
      void constructOverlap() {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);

        point_type *sendBuf = new point_type[points->size()];
        int         size    = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          sendBuf[size++] = *l_iter;
        }
        int *sizes   = new int[this->commSize()];
        int *offsets = new int[this->commSize()+1];
        point_type *remotePoints = NULL;
        int        *remoteRanks  = NULL;

        // Change to Allgather() for the correct binning algorithm
        MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, this->comm());
        if (this->commRank() == 0) {
          offsets[0] = 0;
          for(int p = 1; p <= this->commSize(); p++) {
            offsets[p] = offsets[p-1] + sizes[p-1];
          }
          remotePoints = new point_type[offsets[this->commSize()]];
        }
        MPI_Gatherv(sendBuf, size, MPI_INT, remotePoints, sizes, offsets, MPI_INT, 0, this->comm());
        std::map<int, std::map<int, std::set<point_type> > > overlapInfo;

        if (this->commRank() == 0) {
          for(int p = 0; p < this->commSize(); p++) {
            std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]]);
          }
          for(int p = 0; p < this->commSize(); p++) {
            for(int q = p+1; q < this->commSize(); q++) {
              std::set_intersection(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]],
                                    &remotePoints[offsets[q]], &remotePoints[offsets[q+1]],
                                    std::insert_iterator<std::set<point_type> >(overlapInfo[p][q], overlapInfo[p][q].begin()));
              overlapInfo[q][p] = overlapInfo[p][q];
            }
            sizes[p]     = overlapInfo[p].size()*2;
            offsets[p+1] = offsets[p] + sizes[p];
          }
          remoteRanks = new int[offsets[this->commSize()]];
          int       k = 0;
          for(int p = 0; p < this->commSize(); p++) {
            for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
              remoteRanks[k*2]   = r_iter->first;
              remoteRanks[k*2+1] = r_iter->second.size();
              k++;
            }
          }
        }
        int numOverlaps;
        MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, this->comm());
        int *overlapRanks = new int[numOverlaps];
        MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, this->comm());
        if (this->commRank() == 0) {
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            sizes[p] = 0;
            for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
              sizes[p] += remoteRanks[k*2+1];
              k++;
            }
            offsets[p+1] = offsets[p] + sizes[p];
          }
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
              int rank = r_iter->first;
              for(typename std::set<point_type>::iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
                remotePoints[k++] = *p_iter;
              }
            }
          }
        }
        int numOverlapPoints = 0;
        for(int r = 0; r < numOverlaps/2; r++) {
          numOverlapPoints += overlapRanks[r*2+1];
        }
        point_type *overlapPoints = new point_type[numOverlapPoints];
        MPI_Scatterv(remotePoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints, MPI_INT, 0, this->comm());

        for(int r = 0, k = 0; r < numOverlaps/2; r++) {
          int rank = overlapRanks[r*2];

          for(int p = 0; p < overlapRanks[r*2+1]; p++) {
            point_type point = overlapPoints[k++];

            this->_sendOverlap->addArrow(point, rank, point);
            this->_recvOverlap->addArrow(rank, point, point);
          }
        }

        delete [] overlapPoints;
        delete [] overlapRanks;
        delete [] sizes;
        delete [] offsets;
        if (this->commRank() == 0) {
          delete [] remoteRanks;
          delete [] remotePoints;
        }
      };
      void constructLocalOrder() {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);

        this->_order.clear();
        this->_localSize = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (this->_sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = this->_sendOverlap->support(*l_iter);
            int minRank = this->_sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < this->_sendOverlap->commRank()) {
              this->_order[*l_iter] = -1;
            } else {
              this->_order[*l_iter] = this->_localSize++;
            }
          } else {
            this->_order[*l_iter] = this->_localSize++;
          }
        }
      };
      void constructInverseOrder() {
        for(typename std::map<point_type, int>::iterator p_iter = this->_order.begin(); p_iter != this->_order.end(); ++p_iter) {
          this->_invOrder[this->getIndex(p_iter->first)] = p_iter->first;
        }
      };
      void constructCommunication() {
        Obj<typename send_overlap_type::baseSequence> sendRanks = this->_sendOverlap->base();

        for(typename send_overlap_type::baseSequence::iterator r_iter = sendRanks->begin(); r_iter != sendRanks->end(); ++r_iter) {
          const Obj<typename send_overlap_type::coneSequence>& cone = this->_sendOverlap->cone(*r_iter);
          Obj<dsieve_type> sieve = new dsieve_type();

          for(typename send_overlap_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            sieve->addPoint(*c_iter);
          }
          this->_sendSection->getAtlas()->getTopology()->setPatch(*r_iter, sieve);
        }
        this->_sendSection->getAtlas()->getTopology()->stratify();
        Obj<typename recv_overlap_type::capSequence> recvRanks = this->_recvOverlap->cap();

        for(typename recv_overlap_type::capSequence::iterator r_iter = recvRanks->begin(); r_iter != recvRanks->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::supportSequence>& support = this->_recvOverlap->support(*r_iter);
          Obj<dsieve_type> sieve = new dsieve_type();

          for(typename recv_overlap_type::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
            sieve->addPoint(*s_iter);
          }
          this->_recvSection->getAtlas()->getTopology()->setPatch(*r_iter, sieve);
        }
        this->_recvSection->getAtlas()->getTopology()->stratify();
        // Setup sections
        this->_sendSection->construct(1);
        this->_recvSection->construct(1);
        this->_sendSection->getAtlas()->orderPatches();
        this->_recvSection->getAtlas()->orderPatches();
        this->_sendSection->allocate();
        this->_recvSection->allocate();
        this->_sendSection->constructCommunication(send_section_type::SEND);
        this->_recvSection->constructCommunication(recv_section_type::RECEIVE);
      };
      void fillSection() {
        Obj<typename send_overlap_type::traits::capSequence> sendPoints = this->_sendOverlap->cap();

        for(typename send_overlap_type::traits::capSequence::iterator s_iter = sendPoints->begin(); s_iter != sendPoints->end(); ++s_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = this->_sendOverlap->support(*s_iter);

          for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
            this->_sendSection->update(*p_iter, *s_iter, &(this->_order[*s_iter]));
          }
        }
      };
      void communicate() {
        this->_sendSection->startCommunication();
        this->_recvSection->startCommunication();
        this->_sendSection->endCommunication();
        this->_recvSection->endCommunication();
      };
      void fillOrder() {
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = this->_recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = this->_recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = this->_recvSection->restrict(*p_iter, *r_iter);

            if (values[0] >= 0) {
              if (this->_order[*r_iter] >= 0) {
                ostringstream msg;
                msg << "Multiple indices for point " << *r_iter;
                throw ALE::Exception(msg.str().c_str());
              }
              this->_order[*r_iter] = -(values[0]+1);
            }
          }
        }
      };
      void construct() {
        //const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap
        this->constructOverlap();
        this->constructLocalOrder();
        this->calculateOffsets();
        this->updateOrder();
        this->constructCommunication();
        this->fillSection();
        this->communicate();
        this->fillOrder();
      };
      void view(const std::string& name) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);
        ostringstream txt;

        if (name == "") {
          if(this->commRank() == 0) {
            txt << "viewing a Numbering" << std::endl;
          }
        } else {
          if(this->commRank() == 0) {
            txt << "viewing Numbering '" << name << "'" << std::endl;
          }
        }
        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          txt << "[" << this->commRank() << "] " << *p_iter << " --> ";
          if (this->_order[*p_iter] >= 0) {
            txt << this->_order[*p_iter];
          } else {
            txt << -(this->_order[*p_iter]+1) << " (global)";
          }
          txt << std::endl;
        }
        PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
        PetscSynchronizedFlush(this->comm());
      };
    };

    class GlobalOrder {
    public:
      template<typename Atlas, typename Numbering>
      static Obj<Numbering> createIndices(const Obj<Atlas>& atlas, const Obj<Numbering>& numbering) {
        Obj<Numbering> globalOffsets = new Numbering(numbering->getTopology(), numbering->getLabel(), numbering->getValue());
        typename Atlas::patch_type patch = 0;
        // FIX: I think we can just generalize Numbering to take a stride based upon an Atlas
        //        However, then we also want a lightweight way of creating one numbering from another I think (maybe constructor?)
        // Construct local offsets
        const Obj<typename Numbering::topology_type::label_sequence>& points = numbering->getTopology()->getLabelStratum(patch, numbering->getLabel(), numbering->getValue());
        int offset = 0;

        for(typename Numbering::topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (numbering->isLocal(*l_iter)) {
            globalOffsets->setIndex(*l_iter, offset);
            offset += atlas->getFiberDimension(patch, *l_iter);
          } else {
            globalOffsets->setIndex(*l_iter, -1);
          }
        }
        globalOffsets->setLocalSize(offset);
        globalOffsets->calculateOffsets();
        globalOffsets->updateOrder();
        globalOffsets->copyCommunication(numbering);
        globalOffsets->fillSection();
        globalOffsets->communicate();
        globalOffsets->fillOrder();
        return globalOffsets;
      };
    };
  }
}
#endif
