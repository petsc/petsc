#ifndef included_ALE_Completion_hh
#define included_ALE_Completion_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  namespace New {
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
      typedef typename ALE::New::OverlapValues<send_overlap_type, atlas_type, value_type> send_section_type;
      typedef typename ALE::Sifter<point_type, int, point_type>                           recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, int>        recv_sizer_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, atlas_type, value_type> recv_section_type;
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
      template<typename Sizer>
      static void setupSend(const Obj<send_overlap_type>& sendOverlap, const Obj<Sizer>& sendSizer, const Obj<send_section_type>& sendSection) {
        // Here we should just use the overlap as the topology (once it is a new-style sieve)
        sendSection->getAtlas()->clear();
        sendSection->getAtlas()->setTopology(ALE::New::Completion<mesh_topology_type,value_type>::createSendTopology(sendOverlap));
        if (sendSection->debug() > 10) {sendSection->getAtlas()->getTopology()->view("Send topology after setup", MPI_COMM_SELF);}
        sendSection->construct(sendSizer);
        sendSection->getAtlas()->orderPatches();
        sendSection->allocate();
        sendSection->constructCommunication(send_section_type::SEND);
      };
      template<typename Filler>
      static void completeSend(const Obj<Filler>& sendFiller, const Obj<send_section_type>& sendSection) {
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
      template<typename SizerFiller, typename Filler>
      static void sendSection(const Obj<send_overlap_type>& sendOverlap, const Obj<SizerFiller>& sizerFiller, const Obj<Filler>& filler, const Obj<send_section_type>& sendSection) {
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
      static Obj<send_overlap_type> sendDistribution(const Obj<mesh_topology_type>& topology, const int dim, const Obj<mesh_topology_type>& topologyNew) {
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
        return sendOverlap;
      };
      template<typename Sizer>
      static void setupReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<recv_section_type>& recvSection) {
        // Create section
        const Obj<recv_overlap_type::traits::capSequence> ranks = recvOverlap->cap();

        recvSection->getAtlas()->clear();
        for(recv_overlap_type::traits::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> recvSieve = new dsieve_type();
          const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(0);

          // Want to replace this loop with a slice through color
          for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            recvSieve->addPoint(p_iter.color());
          }
          recvSection->getAtlas()->getTopology()->setPatch(0, recvSieve);
        }
        recvSection->getAtlas()->getTopology()->stratify();
        recvSection->construct(recvSizer);
        recvSection->getAtlas()->orderPatches();
        recvSection->allocate();
        recvSection->constructCommunication(recv_section_type::RECEIVE);
      };
      static void completeReceive(const Obj<recv_section_type>& recvSection) {
        // Complete the section
        recvSection->startCommunication();
        recvSection->endCommunication();
        if (recvSection->debug()) {recvSection->view("Receive Section in Completion", MPI_COMM_SELF);}
        // Read out section values
      };
      static void recvSection(const Obj<recv_overlap_type>& recvOverlap, const Obj<recv_section_type>& recvSection) {
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
    };
  }
}
#endif
