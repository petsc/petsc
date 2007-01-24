#ifndef included_ALE_SectionCompletion_hh
#define included_ALE_SectionCompletion_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  namespace New {
    template<typename Topology_, typename Value_>
    class SectionCompletion {
    public:
      typedef int                                                                     point_type;
      typedef Value_                                                                  value_type;
      typedef Topology_                                                               mesh_topology_type;
      typedef typename mesh_topology_type::sieve_type                                 sieve_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                            dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                           topology_type;
      typedef typename ALE::Sifter<int, point_type, point_type>                       send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, int> send_sizer_type;
      typedef typename ALE::Sifter<point_type, int, point_type>                       recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, int> recv_sizer_type;
      typedef typename ALE::New::ConstantSection<topology_type, int>                  constant_sizer;
      typedef typename ALE::New::SectionCompletion<mesh_topology_type, int>           int_completion;
      typedef typename ALE::New::SectionCompletion<mesh_topology_type, value_type>    completion;
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
        sendSection->setTopology(completion::createSendTopology(sendOverlap));
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
        int_completion::setupSend(sendOverlap, constantSizer, sendSizer);
        int_completion::setupReceive(recvOverlap, constantSizer, recvSizer);
        // 2) Fill the sizer section and communicate
        int_completion::fillSend(sizerFiller, sendSizer);
        if (sendSizer->debug()) {sendSizer->view("Send Sizer in Completion", MPI_COMM_SELF);}
        sendSizer->startCommunication();
        recvSizer->startCommunication();
        sendSizer->endCommunication();
        recvSizer->endCommunication();
        if (recvSizer->debug()) {recvSizer->view("Receive Sizer in Completion", MPI_COMM_SELF);}
        // No need to update a global section since the receive sizes are all on the interface
        // 3) Create the send and receive sections
        completion::setupSend(sendOverlap, sendSizer, sendSection);
        completion::setupReceive(recvOverlap, recvSizer, recvSection);
        // 4) Fill up send section and communicate
        completion::fillSend(filler, sendSection);
        if (sendSection->debug()) {sendSection->view("Send Section in Completion", MPI_COMM_SELF);}
        sendSection->startCommunication();
        recvSection->startCommunication();
        sendSection->endCommunication();
        recvSection->endCommunication();
        if (recvSection->debug()) {recvSection->view("Receive Section in Completion", MPI_COMM_SELF);}
      };
    };
  }
}

#endif
