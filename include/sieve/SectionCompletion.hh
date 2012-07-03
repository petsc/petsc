#ifndef included_ALE_SectionCompletion_hh
#define included_ALE_SectionCompletion_hh

#ifndef  included_ALE_Topology_hh
#include <sieve/Topology.hh>
#endif

#ifndef  included_ALE_Field_hh
#include <sieve/Field.hh>
#endif

namespace ALE {
  namespace New {
    template<typename Topology_, typename Value_, typename Alloc_ = typename Topology_::alloc_type>
    class SectionCompletion {
    public:
      typedef int                                                                              point_type;
      typedef Topology_                                                                        mesh_topology_type;
      typedef Value_                                                                           value_type;
      typedef Alloc_                                                                           alloc_type;
      typedef typename alloc_type::template rebind<point_type>::other                          point_alloc_type;
      typedef typename mesh_topology_type::sieve_type                                          sieve_type;
      typedef typename ALE::DiscreteSieve<point_type, point_alloc_type>                        dsieve_type;
      typedef typename ALE::Topology<int, dsieve_type, alloc_type>                             topology_type;
      // TODO: Fix this typedef typename ALE::Partitioner<>::part_type                         rank_type;
      typedef short int                                                                        rank_type;
      typedef typename ALE::New::SectionCompletion<mesh_topology_type, int, alloc_type>        int_completion;
      typedef typename ALE::New::SectionCompletion<mesh_topology_type, value_type, alloc_type> completion;
    public:
      // Creates a DiscreteTopology with the overlap information
      template<typename SendOverlap>
      static Obj<topology_type> createSendTopology(const Obj<SendOverlap>& sendOverlap) {
        const Obj<typename SendOverlap::baseSequence> ranks = sendOverlap->base();
        Obj<topology_type> topology = new topology_type(sendOverlap->comm(), sendOverlap->debug());

        for(typename SendOverlap::baseSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(*r_iter));
          topology->setPatch(*r_iter, sendSieve);
        }
        return topology;
      }
      template<typename RecvOverlap>
      static Obj<topology_type> createRecvTopology(const Obj<RecvOverlap>& recvOverlap) {
        const Obj<typename RecvOverlap::capSequence> ranks = recvOverlap->cap();
        Obj<topology_type> topology = new topology_type(recvOverlap->comm(), recvOverlap->debug());

        for(typename RecvOverlap::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> recvSieve = new dsieve_type();
          const Obj<typename RecvOverlap::supportSequence>& points  = recvOverlap->support(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            recvSieve->addPoint(p_iter.color());
          }
          topology->setPatch(*r_iter, recvSieve);
        }
        return topology;
      }
      template<typename Sizer, typename Section>
      static void setupSend(const Obj<topology_type>& sendChart, const Obj<Sizer>& sendSizer, const Obj<Section>& sendSection) {
        // Here we should just use the overlap as the topology (once it is a new-style sieve)
        sendSection->clear();
        sendSection->setTopology(sendChart);
        sendSection->construct(sendSizer);
        sendSection->allocate();
        if (sendSection->debug() > 10) {sendSection->view("Send section after setup", MPI_COMM_SELF);}
        sendSection->constructCommunication(Section::SEND);
      }
      template<typename Filler, typename Section>
      static void fillSend(const Filler& sendFiller, const Obj<Section>& sendSection) {
        const typename Section::sheaf_type& patches = sendSection->getPatches();

        for(typename Section::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename Section::section_type>&        section = p_iter->second;
          const typename Section::section_type::chart_type& chart   = section->getChart();

          for(typename Section::section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            if (sendFiller->hasPoint(*c_iter)) {
              section->updatePoint(*c_iter, sendFiller->restrictPoint(*c_iter));
            }
          }
        }
      }
      template<typename RecvOverlap, typename Sizer, typename Section>
      static void setupReceive(const Obj<RecvOverlap>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<Section>& recvSection) {
        // Create section
        const Obj<typename RecvOverlap::capSequence> ranks = recvOverlap->cap();

        recvSection->clear();
        for(typename RecvOverlap::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>& points  = recvOverlap->support(*r_iter);
          const Obj<typename Section::section_type>&        section = recvSection->getSection(*r_iter);

          // Want to replace this loop with a slice through color
          for(typename RecvOverlap::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            const typename dsieve_type::point_type& point = p_iter.color();

            section->setFiberDimension(point, 1);
          }
        }
        recvSection->construct(recvSizer);
        recvSection->allocate();
        recvSection->constructCommunication(Section::RECEIVE);
      }
      template<typename SendOverlap, typename RecvOverlap, typename SizerFiller, typename Filler, typename SendSection, typename RecvSection>
      static void completeSection(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SizerFiller>& sizerFiller, const Filler& filler, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        typedef typename alloc_type::template rebind<int>::other int_alloc_type;
        typedef typename ALE::Field<SendOverlap, int, ALE::Section<point_type, int, int_alloc_type> > send_sizer_type;
        typedef typename ALE::Field<RecvOverlap, int, ALE::Section<point_type, int, int_alloc_type> > recv_sizer_type;
        typedef typename ALE::Field<SendOverlap, int, ALE::ConstantSection<point_type, int> >         constant_send_sizer;
        typedef typename ALE::Field<RecvOverlap, int, ALE::ConstantSection<point_type, int> >         constant_recv_sizer;
        Obj<send_sizer_type>     sendSizer      = new send_sizer_type(sendSection->comm(), sendSection->debug());
        Obj<recv_sizer_type>     recvSizer      = new recv_sizer_type(recvSection->comm(), sendSizer->getTag(), recvSection->debug());
        Obj<constant_send_sizer> constSendSizer = new constant_send_sizer(sendSection->comm(), sendSection->debug());
        Obj<constant_recv_sizer> constRecvSizer = new constant_recv_sizer(recvSection->comm(), recvSection->debug());
        Obj<topology_type>       sendChart      = completion::createSendTopology(sendOverlap);
        Obj<topology_type>       recvChart      = completion::createRecvTopology(recvOverlap);

        // 1) Create the sizer sections
        constSendSizer->setTopology(sendChart);
        const typename topology_type::sheaf_type& sendRanks = sendChart->getPatches();
        for(typename topology_type::sheaf_type::const_iterator r_iter = sendRanks.begin(); r_iter != sendRanks.end(); ++r_iter) {
          const int rank = r_iter->first;
          const int one  = 1;
          constSendSizer->getSection(rank)->updatePoint(*r_iter->second->base()->begin(), &one);
        }
        constRecvSizer->setTopology(recvChart);
        const typename topology_type::sheaf_type& recvRanks = recvChart->getPatches();
        for(typename topology_type::sheaf_type::const_iterator r_iter = recvRanks.begin(); r_iter != recvRanks.end(); ++r_iter) {
          const int rank = r_iter->first;
          const int one  = 1;
          constRecvSizer->getSection(rank)->updatePoint(*r_iter->second->base()->begin(), &one);
        }
        int_completion::setupSend(sendChart, constSendSizer, sendSizer);
        int_completion::setupReceive(recvOverlap, constRecvSizer, recvSizer);
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
        completion::setupSend(sendChart, sendSizer, sendSection);
        completion::setupReceive(recvOverlap, recvSizer, recvSection);
        // 4) Fill up send section and communicate
        completion::fillSend(filler, sendSection);
        if (sendSection->debug()) {sendSection->view("Send Section in Completion", MPI_COMM_SELF);}
        sendSection->startCommunication();
        recvSection->startCommunication();
        sendSection->endCommunication();
        recvSection->endCommunication();
        if (recvSection->debug()) {recvSection->view("Receive Section in Completion", MPI_COMM_SELF);}
      }
    };
  }
}
#endif
