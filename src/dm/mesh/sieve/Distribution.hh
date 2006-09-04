#ifndef included_ALE_Distribution_hh
#define included_ALE_Distribution_hh

#ifndef  included_ALE_Mesh_hh
#include <Mesh.hh>
#endif

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

extern PetscErrorCode PetscCommSynchronizeTags(MPI_Comm);

// Attempt to unify all of the distribution mechanisms:
//   one to many  (distributeMesh)
//   many to one  (unifyMesh)
//   many to many (Numbering)
// as well as things being distributed
//   Section
//   Sieve        (This sends two sections, the points and cones)
//   Numbering    (Should be an integer section)
//   Global Order (should be an integer section with extra methods)
//
// 0) Create the new object to hold the communicated data
//
// 1) Create Overlap
//    There may be special ways to do this based upon what we know at the time
//
// 2) Create send and receive sections over the interface
//    These have a flat topology now, consisting only of the overlap nodes
//    We could make a full topology on the overlap (maybe it is necessary for higher order)
//
// 3) Communication section
//    Create sizer sections on interface (uses constant sizer)
//    Communicate sizes on interface (uses custom filler)
//      Fill send section
//      sendSection->startCommunication();
//      recvSection->startCommunication();
//      sendSection->endCommunication();
//      recvSection->endCommunication();
//
//    Create section on interface (uses previous sizer)
//    Communicate values on interface (uses custom filler)
//      Same stuff as above
//    
// 4) Update new section with old local values (can be done in between the communication?)
//    Loop over patches in new topology
//      Loop over chart from patch in old atlas
//        If this point is in the new sieve from patch
//          Set to old fiber dimension
//    Order and allocate new section
//    Repeat loop, but update values
//
// 5) Update new section with old received values
//    Loop over patches in discrete topology of receive section (these are ranks)
//      Loop over base of discrete sieve (we should transform this to a chart to match above)
//        Get new patch from overlap, or should the receive patches be <rank, patch>?
//        Guaranteed to be in the new sieve from patch (but we could check anyway)
//          Set to recevied fiber dimension
//    Order and allocate new section
//    Repeat loop, but update values
//
// 6) Synchronize PETSc tags (can I get around this?)
namespace ALE {
  namespace New {
    template<typename Topology_>
    class Distribution {
    public:
      typedef Topology_                                                       topology_type;
      typedef ALE::New::Completion<Topology_, Mesh::sieve_type::point_type>   sieveCompletion;
      typedef ALE::New::Completion<Topology_, Mesh::section_type::value_type> sectionCompletion;
      typedef typename sectionCompletion::send_overlap_type                   send_overlap_type;
      typedef typename sectionCompletion::recv_overlap_type                   recv_overlap_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "sendSection"
      template<typename Section, typename Sizer, typename Filler>
      static void sendSection(const Obj<send_overlap_type>& overlap, const Obj<Sizer>& sizer, const Obj<Filler>& filler, const Obj<Section>& serialSection, const Obj<Section>& parallelSection) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename ALE::New::OverlapValues<send_overlap_type, typename sectionCompletion::topology_type, typename Section::value_type> send_section_type;
        const Obj<send_section_type> sendSec = new send_section_type(serialSection->comm(), serialSection->debug());

        sectionCompletion::sendSection(overlap, sizer, filler, sendSec);
        const Obj<typename Section::atlas_type>&                      serialAtlas = serialSection->getAtlas();
        const Obj<typename Section::atlas_type>&                      atlas       = parallelSection->getAtlas();
        const Obj<typename Section::topology_type>&                   topology    = atlas->getTopology();
        const typename Mesh::section_type::topology_type::sheaf_type& patches     = topology->getPatches();

        for(typename Mesh::section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&                     patch = p_iter->first;
          const Obj<typename Section::topology_type::sieve_type>& sieve = p_iter->second;
          const typename Section::atlas_type::chart_type&         chart = serialAtlas->getPatch(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename Section::point_type& point = c_iter->first;

            if (sieve->hasPoint(point)) {
              atlas->setFiberDimension(patch, point, serialAtlas->getFiberDimension(patch, point));
            }
          }
        }
        atlas->orderPatches();
        //parallelSection->getAtlas()->copyByDepth(serialSection->getAtlas());
        parallelSection->allocate();
        for(typename Mesh::section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&             patch = p_iter->first;
          const typename Section::atlas_type::chart_type& chart = parallelSection->getAtlas()->getPatch(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename Section::point_type& point = *c_iter;

            parallelSection->update(patch, point, serialSection->restrict(patch, point));
          }
        }
        ALE_LOG_EVENT_END;
      };
      static void sendMesh(const Obj<Mesh>& serialMesh, const Obj<Mesh>& parallelMesh) {
        typedef ALE::New::SizeSection<Mesh::section_type>      SectionSizer;
        typedef ALE::New::PatchlessSection<Mesh::section_type> SectionFiller;
        const Obj<Mesh::topology_type> topology         = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type> parallelTopology = parallelMesh->getTopologyNew();
        const int dim   = serialMesh->getDimension();
        const int debug = serialMesh->debug;

        Obj<send_overlap_type> cellOverlap   = sieveCompletion::sendDistribution(topology, dim, parallelTopology);
        Obj<send_overlap_type> vertexOverlap = new send_overlap_type(serialMesh->comm(), debug);
        Obj<Mesh::sieve_type>  sieve         = topology->getPatch(0);
        const Obj<typename send_overlap_type::traits::capSequence> cap = cellOverlap->cap();

        parallelMesh->setVertexSendOverlap(vertexOverlap);
        for(typename send_overlap_type::traits::baseSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>& ranks = cellOverlap->support(*p_iter);

          for(typename send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              vertexOverlap->addArrow(*c_iter, *r_iter, *c_iter);
            }
          }
        }
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          const Mesh::section_type::patch_type patch         = 0;
          const Obj<Mesh::section_type>&       serialSection = serialMesh->getSection(*name);
          const Obj<SectionSizer>              sizer         = new SectionSizer(serialSection, patch);
          const Obj<SectionFiller>             filler        = new SectionFiller(serialSection, patch);
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            sendSection(cellOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          } else {
            sendSection(vertexOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          }
        }
        if (!serialMesh->getSplitSection().isNull()) {
          typedef ALE::New::SizeSection<Mesh::split_section_type>      SplitSizer;
          typedef ALE::New::PatchlessSection<Mesh::split_section_type> SplitFiller;
          const Mesh::section_type::patch_type patch              = 0;
          const Obj<Mesh::split_section_type>& serialSplitField   = serialMesh->getSplitSection();
          const Obj<SplitSizer>                sizer              = new SplitSizer(serialSplitField, patch);
          const Obj<SplitFiller>               filler             = new SplitFiller(serialSplitField, patch);
          Obj<Mesh::split_section_type>        parallelSplitField = new Mesh::split_section_type(serialMesh->comm(), debug);

          parallelSplitField->getAtlas()->setTopology(parallelMesh->getTopologyNew());
          sendSection(cellOverlap, sizer, filler, serialSplitField, parallelSplitField);
          parallelMesh->setSplitSection(parallelSplitField);
        }
      };
      template<typename Section>
      static void receiveSection(const Obj<recv_overlap_type>& overlap, const Obj<Section>& serialSection, const Obj<Section>& parallelSection) {
        typedef typename ALE::New::OverlapValues<recv_overlap_type, typename sectionCompletion::topology_type, typename Section::value_type> recv_section_type;
        const Obj<recv_section_type>         recvSec = new recv_section_type(serialSection->comm(), serialSection->debug());
        const Mesh::section_type::patch_type patch   = 0;

        sectionCompletion::recvSection(overlap, recvSec);
        const Obj<typename recv_section_type::atlas_type>&           serialAtlas = recvSec->getAtlas();
        const Obj<typename Section::atlas_type>&                     atlas       = parallelSection->getAtlas();
        const typename sectionCompletion::topology_type::sheaf_type& patches     = serialAtlas->getTopology()->getPatches();

        for(typename sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename sectionCompletion::topology_type::sieve_type::baseSequence>& base  = p_iter->second->base();

          for(typename sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            atlas->setFiberDimension(patch, *b_iter, serialAtlas->getFiberDimension(p_iter->first, *b_iter));
          }
        }
        atlas->orderPatches();
        //parallelSection->getAtlas()->copyByDepth(recvSec->getAtlas(), parallelSection->getAtlas()->getTopology());
        parallelSection->allocate();

        for(typename sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename sectionCompletion::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            parallelSection->update(patch, *b_iter, recvSec->restrict(p_iter->first, *b_iter));
          }
        }
      };
      static void receiveMesh(const Obj<Mesh>& serialMesh, const Obj<Mesh>& parallelMesh) {
        const Obj<Mesh::topology_type> topology         = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type> parallelTopology = parallelMesh->getTopologyNew();
        Obj<recv_overlap_type> cellOverlap   = sieveCompletion::receiveDistribution(topology, parallelTopology);
        Obj<recv_overlap_type> vertexOverlap = new recv_overlap_type(serialMesh->comm(), serialMesh->debug);
        Obj<Mesh::sieve_type>  parallelSieve = parallelTopology->getPatch(0);
        const Obj<typename recv_overlap_type::traits::baseSequence> base = cellOverlap->base();

        parallelMesh->setVertexRecvOverlap(vertexOverlap);
        for(typename recv_overlap_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& ranks = cellOverlap->cone(*p_iter);

          for(typename recv_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
            const Obj<typename ALE::Mesh::sieve_type::traits::coneSequence>& cone = parallelSieve->cone(*p_iter);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              vertexOverlap->addArrow(*r_iter, *c_iter, *c_iter);
            }
          }
        }
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            receiveSection(cellOverlap, serialMesh->getSection(*name), parallelMesh->getSection(*name));
          } else {
            receiveSection(vertexOverlap, serialMesh->getSection(*name), parallelMesh->getSection(*name));
          }
        }
        if (!serialMesh->getSplitSection().isNull()) {
          Obj<Mesh::split_section_type> parallelSplitField = new Mesh::split_section_type(serialMesh->comm(), serialMesh->debug);

          parallelSplitField->getAtlas()->setTopology(parallelMesh->getTopologyNew());
          receiveSection(cellOverlap, serialMesh->getSplitSection(), parallelSplitField);
          parallelMesh->setSplitSection(parallelSplitField);
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeMesh"
      static Obj<Mesh> distributeMesh(const Obj<Mesh>& serialMesh) {
        Obj<Mesh> parallelMesh = Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug);
        const Obj<Mesh::topology_type>& topology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve    = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug);
        PetscErrorCode                  ierr;

        if (serialMesh->distributed) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        topology->setPatch(0, sieve);
        parallelMesh->setTopologyNew(topology);
        if (serialMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          serialMesh->getTopologyNew()->view("Serial topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            serialMesh->getSection(*name)->view(*name);
          }
          if (!serialMesh->getSplitSection().isNull()) {
            serialMesh->getSplitSection()->view("Serial split field");
          }
        }
        if (serialMesh->commRank() == 0) {
          Distribution<topology_type>::sendMesh(serialMesh, parallelMesh);
        } else {
          Distribution<topology_type>::receiveMesh(serialMesh, parallelMesh);
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          parallelMesh->getTopologyNew()->view("Parallel topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            parallelMesh->getSection(*name)->view(*name);
          }
          if (!parallelMesh->getSplitSection().isNull()) {
            parallelMesh->getSplitSection()->view("Parallel split field");
          }
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "localUpdate"
      template<typename Section>
      static void localUpdate(const Obj<Section>& oldSection, const Obj<Section>& newSection)
      {
        const Obj<typename Section::atlas_type>&           oldAtlas    = oldSection->getAtlas();
        const Obj<typename Section::atlas_type>&           newAtlas    = newSection->getAtlas();
        const Obj<typename Section::topology_type>&        newTopology = newAtlas->getTopology();
        const typename Section::topology_type::sheaf_type& patches     = newTopology->getPatches();

        for(typename Section::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&                     patch    = p_iter->first;
          const Obj<typename Section::topology_type::sieve_type>& newSieve = p_iter->second;
          const typename Section::atlas_type::chart_type&         oldChart = oldAtlas->getChart(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
            const typename Section::point_type& point = c_iter->first;

            if (newSieve->hasPoint(point)) {
              newAtlas->setFiberDimension(patch, point, oldAtlas->getFiberDimension(patch, point));
            }
          }
        }
        newAtlas->orderPatches();
        newSection->allocate();
        for(typename Section::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&             patch    = p_iter->first;
          const typename Section::atlas_type::chart_type& newChart = newAtlas->getChart(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
            const typename Section::point_type& point = c_iter->first;

            newSection->update(patch, point, oldSection->restrict(patch, point));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "remoteUpdate"
      template<typename OldSection, typename NewSection>
      static void remoteUpdate(const Obj<OldSection>& oldSection, const Obj<NewSection>& newSection)
      {
        typedef typename OldSection::topology_type topology_type;
        const Obj<typename OldSection::atlas_type>& oldAtlas = oldSection->getAtlas();
        const Obj<typename NewSection::atlas_type>& newAtlas = newSection->getAtlas();
        const typename topology_type::sheaf_type&   patches  = oldAtlas->getTopology()->getPatches();
        // Patch should come from overlap
        const typename topology_type::patch_type    patch    = 0;

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            newAtlas->setFiberDimension(patch, *b_iter, oldAtlas->getFiberDimension(p_iter->first, *b_iter));
          }
        }
        newAtlas->orderPatches();
        newSection->allocate();
        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            newSection->update(patch, *b_iter, oldSection->restrict(p_iter->first, *b_iter));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "redistributeSection"
      template<typename Sizer, typename Filler, typename Section>
      static void redistributeSection(const Obj<send_overlap_type>& sendOverlap, const Obj<send_overlap_type>& recvOverlap, const Obj<Sizer>& sizer, const Obj<Filler>& filler, const Obj<Section>& oldSection, const Obj<Section>& newSection) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename Section::value_type              value_type;
        typedef typename sectionCompletion::topology_type topology_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        const Obj<send_section_type> sendSection = new send_section_type(oldSection->comm(), oldSection->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(oldSection->comm(), oldSection->debug());

        localUpdate(oldSection, newSection);
        sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
        remoteUpdate(recvSection, newSection);
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "createScatterOverlap"
      // This is the overlap for initially partitioned things where everyone communicates
      //   We also need 0 --> everybody
      //   An initial phase to figure out the selective overlap
      static void createScatterOverlap(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        const int rank = sendOverlap->commRank();

        for(int p = 0; p < sendOverlap->commSize(); p++) {
          if (p != rank) {
            // Send arrow:   local point --- (remote point) ---> remote rank 
            //   There are arrows to each rank whose color is the partition point (also the rank)
            sendOverlap->addCone(p, p, p);
            // Receive arrow:   remote rank --- (remote point) ---> local point
            recvOverlap->addCone(p, p, p);
          }
        }
        if (sendOverlap->debug) {sendOverlap->view(std::cout, "Send overlap for sieve scatter");}
        if (recvOverlap->debug) {recvOverlap->view(std::cout, "Receive overlap for sieve scatter");}
      };
      #undef __FUNCT__
      #define __FUNCT__ "localSieve"
      static void localSieve(const Obj<typename Mesh::sieve_type>& oldSieve, const int numElements, const short assignment[], const Obj<Mesh::sieve_type>& newSieve) {
        int rank = oldSieve->commRank();

        for(int e = 0; e < numElements; e++) {
          if (assignment[e] == rank) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = oldSieve->cone(e);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              newSieve->addArrow(*c_iter, e, c_iter.color());
            }
          }
        }
        newSieve->stratify();
      };
      #undef __FUNCT__
      #define __FUNCT__ "createPointOverlap"
      template<typename Overlap, typename Section>
      static void createPointOverlap(const Obj<Overlap>& overlap, const Obj<Section>& section) {
        overlap->clear();
        const typename Section::topology_type::sheaf_type& ranks = section->getAtlas()->getTopology()->getPatches();

        for(typename Section::topology_type::sheaf_type::const_iterator p_iter = ranks.begin(); p_iter != ranks.end(); ++p_iter) {
          const int                                                    rank = p_iter->first;
          const Obj<typename Section::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename Section::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename Section::value_type *points = section->restrict(rank, *b_iter);
            int size = section->getAtlas()->size(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              overlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        if (overlap->debug) {overlap->view(std::cout, "Overlap for points");}
      };
      #undef __FUNCT__
      #define __FUNCT__ "updateSieve"
      template<typename RecvSection>
      static void updateSieve(const Obj<RecvSection>& recvSection, const Obj<topology_type>& topology) {
        const typename RecvSection::patch_type                 patch = 0;
        const typename RecvSection::topology_type::sheaf_type& ranks = recvSection->getTopology()->getPatches();
        const Obj<typename topology_type::sieve_type>&         sieve = topology->getPatch(patch);

        for(typename RecvSection::topology_type::sheaf_type::const_iterator p_iter = ranks.begin(); p_iter != ranks.end(); ++p_iter) {
          int                                                              rank = p_iter->first;
          const Obj<typename RecvSection::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename RecvSection::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename RecvSection::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c = 0;

            for(int p = 0; p < size; p++) {
              //sieve->addArrow(points[p], *b_iter, c++);
              sieve->addArrow(points[p], *b_iter, c);
            }
          }
        }
      }
      #undef __FUNCT__
      #define __FUNCT__ "updateOverlap"
      template<typename SendSection, typename RecvSection>
      static void updateOverlap(const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        const typename SendSection::topology_type::sheaf_type& sendRanks = sendSection->getTopology()->getPatches();
        const typename RecvSection::topology_type::sheaf_type& recvRanks = recvSection->getTopology()->getPatches();

        for(typename SendSection::topology_type::sheaf_type::const_iterator p_iter = sendRanks.begin(); p_iter != sendRanks.end(); ++p_iter) {
          int                                                                       rank = p_iter->first;
          const Obj<typename SendSection::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename SendSection::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename SendSection::value_type *points = sendSection->restrict(rank, *b_iter);
            int size = sendSection->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], rank, points[p]);
            }
          }
        }
        for(typename RecvSection::topology_type::sheaf_type::const_iterator p_iter = recvRanks.begin(); p_iter != recvRanks.end(); ++p_iter) {
          int                                                                       rank = p_iter->first;
          const Obj<typename RecvSection::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename RecvSection::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename RecvSection::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
      }
      #undef __FUNCT__
      #define __FUNCT__ "scatterTopology"
      static void scatterTopology(const Obj<topology_type>& oldTopology, const int numElements, const short assignment[], Obj<topology_type>& newTopology) {
        if (oldTopology->commSize() == 1) {
          newTopology = oldTopology;
          return;
        }
        typedef typename Mesh::point_type value_type;
        typedef typename ALE::New::Completion<typename Mesh::topology_type,value_type> completion_type;
        typedef typename completion_type::topology_type topology_type;
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(oldTopology->comm(), oldTopology->debug());
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(oldTopology->comm(), oldTopology->debug());
        const Obj<typename topology_type::sieve_type> newSieve = new typename topology_type::sieve_type(newTopology->comm(), newTopology->debug());
        const typename topology_type::patch_type patch    = 0;

        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        const Obj<send_section_type> sendSection = new send_section_type(oldTopology->comm(), oldTopology->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(oldTopology->comm(), sendSection->getTag(), oldTopology->debug());

        // Create Overlap - Could be calculated from partition, but would need communication
        createScatterOverlap(sendOverlap, recvOverlap);
        // Create local sieve
        newTopology->setPatch(patch, newSieve);
        localSieve(oldTopology->getPatch(patch), numElements, assignment, newTopology->getPatch(patch));
        // Distribute points
        Obj<completion_type::topology_type>                   secTopology          = completion_type::createSendTopology(sendOverlap);
        Obj<typename completion_type::partition_size_section> partitionSizeSection = new typename completion_type::partition_size_section(secTopology, numElements, assignment);
        Obj<typename completion_type::partition_section>      partitionSection     = new typename completion_type::partition_section(secTopology, numElements, assignment);
        sectionCompletion::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // Create point overlap
        createPointOverlap(sendOverlap, sendSection);
        createPointOverlap(recvOverlap, recvSection);
        // Distribute cones
        secTopology = completion_type::createSendTopology(sendOverlap);
        Obj<typename completion_type::cone_size_section> coneSizeSection = new typename completion_type::cone_size_section(secTopology, oldTopology->getPatch(patch));
        Obj<typename completion_type::cone_section>      coneSection     = new typename completion_type::cone_section(secTopology, oldTopology->getPatch(patch));
        sectionCompletion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Update sieve
        updateSieve(recvSection, newTopology);
        newSieve->stratify();
        newTopology->stratify();
      };
      static void coneCompletion(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<topology_type>& topology) {
        typedef typename Mesh::point_type                                                      value_type;
        typedef typename ALE::New::Completion<typename Mesh::topology_type, value_type>        completion_type;
        typedef typename completion_type::topology_type                                        topology_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        const Obj<send_section_type> sendSection = new send_section_type(topology->comm(), topology->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(topology->comm(), sendSection->getTag(), topology->debug());

        coneCompletion(sendOverlap, recvOverlap, topology, sendSection, recvSection);
      };
      #undef __FUNCT__
      #define __FUNCT__ "coneCompletion"
      template<typename SendSection, typename RecvSection>
      static void coneCompletion(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<topology_type>& topology, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        if (sendOverlap->commSize() == 1) return;
        typedef typename ALE::New::Completion<typename Mesh::topology_type, typename Mesh::point_type> completion_type;

        // Distribute cones
        const typename topology_type::patch_type               patch           = 0;
        const Obj<completion_type::topology_type>              secTopology     = completion_type::createSendTopology(sendOverlap);
        const Obj<typename completion_type::cone_size_section> coneSizeSection = new typename completion_type::cone_size_section(secTopology, topology->getPatch(patch));
        const Obj<typename completion_type::cone_section>      coneSection     = new typename completion_type::cone_section(secTopology, topology->getPatch(patch));
        sectionCompletion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Update cones
        updateSieve(recvSection, topology);
      };
      // Creates a subordinate overlap
      //   If two points overlap in the original, the corresponding cones overlap here
      //     we assume identity of points, so only the cone of the new mesh is used
      #undef __FUNCT__
      #define __FUNCT__ "createConeOverlap"
      static void createConeOverlap(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<Mesh>& oldMesh, const Obj<Mesh>& newMesh) {
        Obj<send_overlap_type> coneSendOverlap = new send_overlap_type(newMesh->comm(), newMesh->debug);
        Obj<recv_overlap_type> coneRecvOverlap = new recv_overlap_type(newMesh->comm(), newMesh->debug);
        const Obj<Mesh::sieve_type>                                 oldSieve = oldMesh->getTopologyNew()->getPatch(0);
        const Obj<Mesh::sieve_type>                                 newSieve = newMesh->getTopologyNew()->getPatch(0);
        const Obj<typename send_overlap_type::traits::capSequence>  cap      = sendOverlap->cap();
        const Obj<typename recv_overlap_type::traits::baseSequence> base     = recvOverlap->base();

        for(typename send_overlap_type::traits::capSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>&     ranks = sendOverlap->support(*p_iter);
          const typename send_overlap_type::traits::supportSequence::iterator end   = ranks->end();

          for(typename send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != end; ++r_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>&     cone = oldSieve->cone(*p_iter);
            const typename Mesh::sieve_type::traits::coneSequence::iterator cEnd = cone->end();

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
              coneSendOverlap->addArrow(*c_iter, *r_iter, *c_iter);
            }
          }
        }
        for(typename recv_overlap_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>&     ranks = recvOverlap->cone(*p_iter);
          const typename recv_overlap_type::traits::coneSequence::iterator end   = ranks->end();

          for(typename send_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != end; ++r_iter) {
            const Obj<typename ALE::Mesh::sieve_type::traits::coneSequence>& cone = newSieve->cone(*p_iter);
            const typename Mesh::sieve_type::traits::coneSequence::iterator  cEnd = cone->end();

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
              coneRecvOverlap->addArrow(*r_iter, *c_iter, *c_iter);
            }
          }
        }
        if (sendOverlap->debug) {
          coneSendOverlap->view(std::cout, "Cone send overlap");
          coneRecvOverlap->view(std::cout, "Cone receive overlap");
        }
        newMesh->setVertexSendOverlap(coneSendOverlap);
        newMesh->setVertexRecvOverlap(coneRecvOverlap);
      };
      #undef __FUNCT__
      #define __FUNCT__ "updateSectionLocal"
      template<typename Section>
      static void updateSectionLocal(const Obj<Section>& oldSection, const Obj<Section>& newSection)
      {
        const typename Section::topology_type::sheaf_type& patches = newSection->getTopology()->getPatches();

        for(typename Section::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&                     patch    = p_iter->first;
          const Obj<typename Section::topology_type::sieve_type>& newSieve = p_iter->second;
          if (!oldSection->hasPatch(patch)) continue;
          const typename Section::atlas_type::chart_type&         oldChart = oldSection->getPatch(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
            if (newSieve->hasPoint(*c_iter)) {
              newSection->setFiberDimension(patch, *c_iter, oldSection->getFiberDimension(patch, *c_iter));
            }
          }
        }
        newSection->allocate();
        for(typename Section::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&             patch    = p_iter->first;
          if (!oldSection->hasPatch(patch)) continue;
          const typename Section::atlas_type::chart_type& newChart = newSection->getPatch(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
            newSection->update(patch, *c_iter, oldSection->restrict(patch, *c_iter));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "updateSectionRemote"
      template<typename RecvSection>
      static void updateSectionRemote(const Obj<recv_overlap_type>& recvOverlap, const Obj<RecvSection>& recvSection, const Obj<Mesh::section_type>& newSection) {
        const Mesh::section_type::patch_type                  patch      = 0;
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
          const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
            newSection->addPoint(patch, *r_iter, recvSection->getFiberDimension(*p_iter, *r_iter));
          }
        }
        newSection->reallocate();
        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
          const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
            newSection->update(patch, *r_iter, recvSection->restrict(*p_iter, *r_iter));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "redistributeMesh"
      static Obj<Mesh> redistributeMesh(const Obj<Mesh>& serialMesh) {
        Obj<Mesh> parallelMesh = Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug);
        const Obj<Mesh::topology_type>& serialTopology   = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type>& parallelTopology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve            = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug);
        PetscErrorCode                  ierr;

        if (serialMesh->distributed) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        parallelTopology->setPatch(0, sieve);
        parallelMesh->setTopologyNew(parallelTopology);
        if (serialMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          serialMesh->getTopologyNew()->view("Serial topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            serialMesh->getSection(*name)->view(*name);
          }
          if (!serialMesh->getSplitSection().isNull()) {
            serialMesh->getSplitSection()->view("Serial split field");
          }
        }

        // FIX Need to spread out cones
        Obj<send_overlap_type> cellSendOverlap   = new send_overlap_type(serialTopology->comm(), serialTopology->debug());
        Obj<recv_overlap_type> cellRecvOverlap   = new recv_overlap_type(serialTopology->comm(), serialTopology->debug());
        sieveCompletion::scatterTopology(serialTopology, serialMesh->getDimension(), parallelTopology, cellSendOverlap, cellRecvOverlap);
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {parallelMesh->getTopologyNew()->view("Parallel topology");}
        createConeOverlap(cellSendOverlap, cellRecvOverlap, serialMesh, parallelMesh);
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Mesh::section_type::value_type> send_section_type;
          typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Mesh::section_type::value_type> recv_section_type;
          typedef SizeSection<Mesh::section_type>      SectionSizer;
          typedef PatchlessSection<Mesh::section_type> SectionFiller;
          const Mesh::section_type::patch_type patch           = 0;
          const Obj<Mesh::section_type>&       serialSection   = serialMesh->getSection(*name);
          const Obj<Mesh::section_type>&       parallelSection = parallelMesh->getSection(*name);
          const Obj<send_section_type>         sendSection     = new send_section_type(serialMesh->comm(), serialMesh->debug);
          const Obj<recv_section_type>         recvSection     = new recv_section_type(serialMesh->comm(), sendSection->getTag(), serialMesh->debug);
          const Obj<SectionSizer>              sizer           = new SectionSizer(serialSection, patch);
          const Obj<SectionFiller>             filler          = new SectionFiller(serialSection, patch);
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            updateSectionLocal(serialSection, parallelSection);
            sieveCompletion::completeSection(cellSendOverlap, cellRecvOverlap, sizer, filler, sendSection, recvSection);
            updateSectionRemote(cellRecvOverlap, recvSection, parallelSection);
          } else {
            updateSectionLocal(serialSection, parallelSection);
            sieveCompletion::completeSection(parallelMesh->getVertexSendOverlap(), parallelMesh->getVertexRecvOverlap(), sizer, filler, sendSection, recvSection);
            updateSectionRemote(parallelMesh->getVertexRecvOverlap(), recvSection, parallelSection);
          }
        }

        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            parallelMesh->getSection(*name)->view(*name);
          }
          if (!parallelMesh->getSplitSection().isNull()) {
            parallelMesh->getSplitSection()->view("Parallel split field");
          }
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
    };
  }
}

#endif
