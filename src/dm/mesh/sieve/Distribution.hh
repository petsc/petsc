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
//   one to many  (redistributeMesh)
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
      // Creates a subordinate overlap
      //   If two points overlap in the original, the corresponding cones overlap here
      //     we assume identity of points, so only the cone of the new mesh is used
      #undef __FUNCT__
      #define __FUNCT__ "createConeOverlap"
      static void createConeOverlap(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<Mesh>& oldMesh, const Obj<Mesh>& newMesh) {
        const Obj<Mesh::sieve_type>                                 oldSieve = oldMesh->getTopology()->getPatch(0);
        const Obj<Mesh::sieve_type>                                 newSieve = newMesh->getTopology()->getPatch(0);
        const Obj<typename send_overlap_type::traits::capSequence>  cap      = sendOverlap->cap();
        const Obj<typename recv_overlap_type::traits::baseSequence> base     = recvOverlap->base();

        for(typename send_overlap_type::traits::capSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>&     ranks = sendOverlap->support(*p_iter);
          const typename send_overlap_type::traits::supportSequence::iterator end   = ranks->end();

          for(typename send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != end; ++r_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>&     cone = oldSieve->cone(*p_iter);
            const typename Mesh::sieve_type::traits::coneSequence::iterator cEnd = cone->end();

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
              sendOverlap->addArrow(*c_iter, *r_iter, *c_iter);
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
              recvOverlap->addArrow(*r_iter, *c_iter, *c_iter);
            }
          }
        }
        if (sendOverlap->debug) {
          sendOverlap->view(std::cout, "Cone send overlap");
          recvOverlap->view(std::cout, "Cone receive overlap");
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "updateOverlap"
      // This is just crappy. WE could introduce another phase to find out exactly what
      //   indices people do not have in the global order after communication
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
      };
      static void createLabelOverlap(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
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
            newSection->updatePoint(patch, *c_iter, oldSection->restrictPoint(patch, *c_iter));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "updateSectionRemote"
      template<typename RecvSection, typename Section>
      static void updateSectionRemote(const Obj<recv_overlap_type>& recvOverlap, const Obj<RecvSection>& recvSection, const Obj<Section>& newSection) {
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
            if (recvSection->getFiberDimension(*p_iter, *r_iter)) {
              newSection->updatePoint(patch, *r_iter, recvSection->restrictPoint(*p_iter, *r_iter));
            }
          }
        }
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
      #undef __FUNCT__
      #define __FUNCT__ "redistributeMesh"
      static Obj<Mesh> redistributeMesh(const Obj<Mesh>& serialMesh, const std::string& partitioner = "chaco") {
        Obj<Mesh> parallelMesh = Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug);
        const Obj<Mesh::topology_type>& serialTopology   = serialMesh->getTopology();
        const Obj<Mesh::topology_type>& parallelTopology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve            = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug);
        const int                       dim              = serialMesh->getDimension();
        PetscErrorCode                  ierr;

        if (serialMesh->distributed) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        // Why in the hell do I need this here????
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        parallelTopology->setPatch(0, sieve);
        parallelMesh->setTopology(parallelTopology);
        if (serialMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          serialMesh->getTopology()->view("Serial topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            serialMesh->getSection(*name)->view(*name);
          }
          if (!serialMesh->getSplitSection().isNull()) {
            serialMesh->getSplitSection()->view("Serial split field");
          }
          Obj<std::set<std::string> > bcSections = serialMesh->getBCSections();

          for(std::set<std::string>::iterator name = bcSections->begin(); name != bcSections->end(); ++name) {
            serialMesh->getBCSection(*name)->view(*name);
          }
        }

        // Distribute cones
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(serialTopology->comm(), serialTopology->debug());
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(serialTopology->comm(), serialTopology->debug());
        sieveCompletion::scatterTopology(serialTopology, dim, parallelTopology, sendOverlap, recvOverlap, partitioner);
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {parallelTopology->view("Parallel topology");}
        createConeOverlap(sendOverlap, recvOverlap, serialMesh, parallelMesh);
        parallelTopology->setDistSendOverlap(sendOverlap);
        parallelTopology->setDistRecvOverlap(recvOverlap);
#if 0
        // Distribute labels
        typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Mesh::point_type> send_section_type;
        typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Mesh::point_type> recv_section_type;
        const Obj<send_section_type>         sendSection = new send_section_type(serialMesh->comm(), serialMesh->debug);
        const Obj<recv_section_type>         recvSection = new recv_section_type(serialMesh->comm(), sendSection->getTag(), serialMesh->debug);
        const typename topology_type::labels_type& labels = serialMesh->getLabels();

        for(typename topology_type::labels_type::iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
          for(typename topology_type::label_type::iterator pl_iter = l_iter->second.begin(); pl_iter != l_iter->second.end(); ++pl_iter) {
            if (parallelTopology->hasLabel(pl_iter->first, l_iter->first)) continue;
            const typename topology_type::patch_label_type&      serialLabel   = pl_iter->second;
            const Obj<typename topology_type::patch_label_type>& parallelLabel = parallelTopology->createLabel(pl_iter->first, l_iter->first);
            // Make the label overlap, here we just cheat and use the vertex overlap

            updateSifterLocal(serialLabel, parallelLabel);
            sieveCompletion::completeSifter(parallelMesh->getVertexSendOverlap(), parallelMesh->getVertexRecvOverlap(), sizer, filler, sendSection, recvSection);
            updateSifterRemote(parallelMesh->getVertexRecvOverlap(), recvSection, parallelLabel);
          }
        }
#endif
        // Distribute sections
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          Obj<Mesh::section_type> parallelSection;

          parallelSection = distributeSection(serialMesh->getSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap);
          parallelMesh->setSection(*name, parallelSection);
        }
        // Distribute boundary condition sections
        sections = serialMesh->getBCSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Mesh::bc_section_type::value_type> send_section_type;
          typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Mesh::bc_section_type::value_type> recv_section_type;
          typedef SizeSection<Mesh::bc_section_type>      SectionSizer;
          typedef PatchlessSection<Mesh::bc_section_type> SectionFiller;
          const Mesh::bc_section_type::patch_type patch           = 0;
          const Obj<Mesh::bc_section_type>&       serialSection   = serialMesh->getBCSection(*name);
          const Obj<Mesh::bc_section_type>&       parallelSection = parallelMesh->getBCSection(*name);
          const Obj<send_section_type>            sendSection     = new send_section_type(serialMesh->comm(), serialMesh->debug);
          const Obj<recv_section_type>            recvSection     = new recv_section_type(serialMesh->comm(), sendSection->getTag(), serialMesh->debug);
          const Obj<SectionSizer>                 sizer           = new SectionSizer(serialSection, patch);
          const Obj<SectionFiller>                filler          = new SectionFiller(serialSection, patch);

          updateSectionLocal(serialSection, parallelSection);
          sieveCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
          updateSectionRemote(recvOverlap, recvSection, parallelSection);
        }
        if (!serialMesh->getSplitSection().isNull()) {
          typedef ALE::New::SizeSection<Mesh::split_section_type>      SplitSizer;
          typedef ALE::New::PatchlessSection<Mesh::split_section_type> SplitFiller;
          typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Mesh::split_section_type::value_type> send_section_type;
          typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Mesh::split_section_type::value_type> recv_section_type;
          const Mesh::section_type::patch_type patch              = 0;
          const Obj<Mesh::split_section_type>& serialSplitField   = serialMesh->getSplitSection();
          Obj<Mesh::split_section_type>        parallelSplitField = new Mesh::split_section_type(parallelMesh->getTopology());
          const Obj<send_section_type>         sendSection        = new send_section_type(serialMesh->comm(), serialMesh->debug);
          const Obj<recv_section_type>         recvSection        = new recv_section_type(serialMesh->comm(), sendSection->getTag(), serialMesh->debug);
          const Obj<SplitSizer>                sizer              = new SplitSizer(serialSplitField, patch);
          const Obj<SplitFiller>               filler             = new SplitFiller(serialSplitField, patch);

          updateSectionLocal(serialSplitField, parallelSplitField);
          sieveCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
          updateSectionRemote(recvOverlap, recvSection, parallelSplitField);
          parallelMesh->setSplitSection(parallelSplitField);
        }

        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {
          Obj<std::set<std::string> > sections = parallelMesh->getSections();

          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            parallelMesh->getSection(*name)->view(*name);
          }
          if (!parallelMesh->getSplitSection().isNull()) {
            parallelMesh->getSplitSection()->view("Parallel split field");
          }
        }
        sections = parallelMesh->getBCSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->getBCSection(*name)->view(*name);
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeSection"
      static Obj<Mesh::section_type> distributeSection(const Obj<Mesh::section_type>& serialSection, const Obj<Mesh::topology_type>& parallelTopology, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        if (serialSection->debug()) {
          serialSection->view("Serial Section");
        }
        typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Mesh::section_type::value_type> send_section_type;
        typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Mesh::section_type::value_type> recv_section_type;
        typedef SizeSection<Mesh::section_type>      SectionSizer;
        typedef PatchlessSection<Mesh::section_type> SectionFiller;
        Obj<Mesh::section_type>              parallelSection = new Mesh::section_type(parallelTopology);
        const Mesh::section_type::patch_type patch           = 0;
        const Obj<send_section_type>         sendSection     = new send_section_type(serialSection->comm(), serialSection->debug());
        const Obj<recv_section_type>         recvSection     = new recv_section_type(serialSection->comm(), sendSection->getTag(), serialSection->debug());
        const Obj<SectionSizer>              sizer           = new SectionSizer(serialSection, patch);
        const Obj<SectionFiller>             filler          = new SectionFiller(serialSection, patch);

        updateSectionLocal(serialSection, parallelSection);
        sieveCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
        updateSectionRemote(recvOverlap, recvSection, parallelSection);
        if (parallelSection->debug()) {
          parallelSection->view("Parallel Section");
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        PetscErrorCode ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        return parallelSection;
      };
    };
  }
}

#endif
