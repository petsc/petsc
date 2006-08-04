#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "overlapTest.hh"

namespace ALE {
  namespace Test {
    template<typename Topology_, typename Index_, typename Marker_>
    class PartitionSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Index_                             index_type;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type> _topology;
      const int          _numElements;
      const marker_type *_partition;
      sizes_type         _sizes;
      void _init(const marker_type partition[]) {
        for(int e = 0; e < this->_numElements; e++) {
          this->_sizes[partition[e]]++;
        }
      };
    public:
      PartitionSection(MPI_Comm comm, const int numElements, const marker_type *partition, const int debug = 0) : ParallelObject(comm, debug), _numElements(numElements), _partition(partition) {
        this->_topology = new topology_type(comm, debug);
        this->_init(partition);
      };
      PartitionSection(const Obj<topology_type>& topology, const int numElements, const marker_type *partition) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _numElements(numElements), _partition(partition) {this->_init(partition);};
      virtual ~PartitionSection() {};
    public:
      void allocate() {};
      const value_type *restrict(const patch_type& patch) {
        throw ALE::Exception("Cannot restrict to a patch with a PartitionSection");
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if (patch == p) {
          throw ALE::Exception("Point must be identical to patch in a PartitionSection");
        }
        return &this->_sizes[patch];
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

    class Completion {
    public:
      typedef ALE::Test::OverlapTest::dsieve_type       dsieve_type;
      typedef ALE::Test::OverlapTest::send_overlap_type send_overlap_type;
      typedef ALE::Test::OverlapTest::send_section_type send_section_type;
      typedef ALE::Test::OverlapTest::recv_overlap_type recv_overlap_type;
      typedef ALE::Test::OverlapTest::recv_section_type recv_section_type;
    public:
      template<typename Sizer>
      static void setupSend(const Obj<send_overlap_type>& sendOverlap, const Obj<Sizer>& sendSizer, const Obj<send_section_type>& sendSection) {
        // Here we should just use the overlap as the topology (once it is a new-style sieve)
        const Obj<send_overlap_type::traits::baseSequence> ranks = sendOverlap->base();

        for(send_overlap_type::traits::baseSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(*r_iter));
          sendSection->getAtlas()->getTopology()->setPatch(*r_iter, sendSieve);
        }
        sendSection->getAtlas()->getTopology()->stratify();
        sendSection->construct(sendSizer);
        sendSection->getAtlas()->orderPatches();
        sendSection->allocate();
        sendSection->constructCommunication();
      };
      template<typename Filler>
      static void completeSend(const Obj<send_overlap_type>& sendOverlap, const Obj<Filler>& sendFiller, const Obj<send_section_type>& sendSection) {
        // Fill section
        const send_section_type::topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

        for(send_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<send_section_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(send_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            sendSection->update(p_iter->first, *b_iter, sendFiller->restrict(p_iter->first, *b_iter));
          }
        }
        sendSection->view("Send Section in Completion", MPI_COMM_SELF);
        // Complete the section
        sendSection->startCommunication();
        sendSection->endCommunication();
      };
      template<typename Sizer>
      static void setupReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<recv_section_type>& recvSection) {
        // Create section
        const Obj<recv_overlap_type::traits::capSequence> ranks = recvOverlap->cap();

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
        recvSection->constructCommunication();
      };
      static void completeReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<recv_section_type>& recvSection) {
        // Complete the section
        recvSection->startCommunication();
        recvSection->endCommunication();
        recvSection->view("Receive Section in Completion", MPI_COMM_SELF);
        // Read out section values
      };
    };
  };
};
