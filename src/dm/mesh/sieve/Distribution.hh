#ifndef included_ALE_Distribution_hh
#define included_ALE_Distribution_hh

#ifndef  included_ALE_Mesh_hh
#include <Mesh.hh>
#endif

#ifndef  included_ALE_Partitioner_hh
#include <Partitioner.hh>
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
      typedef Topology_                                                                   topology_type;
      typedef typename topology_type::sieve_type                                          sieve_type;
      typedef ALE::New::Completion<Topology_, Mesh::sieve_type::point_type>               sieveCompletion;
      typedef ALE::New::SectionCompletion<Topology_, Mesh::real_section_type::value_type> sectionCompletion;
      typedef typename sectionCompletion::send_overlap_type                               send_overlap_type;
      typedef typename sectionCompletion::recv_overlap_type                               recv_overlap_type;
    public:
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
        const Mesh::real_section_type::patch_type                  patch      = 0;
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
        typedef typename ALE::New::SectionCompletion<topology_type, typename sieve_type::point_type> completion;

        // Distribute cones
        const typename topology_type::patch_type               patch           = 0;
        const Obj<typename sieveCompletion::topology_type>     secTopology     = completion::createSendTopology(sendOverlap);
        const Obj<typename sieveCompletion::cone_size_section> coneSizeSection = new typename sieveCompletion::cone_size_section(secTopology, topology, topology->getPatch(patch));
        const Obj<typename sieveCompletion::cone_section>      coneSection     = new typename sieveCompletion::cone_section(secTopology, topology->getPatch(patch));
        completion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Update cones
        updateSieve(recvSection, topology);
      };
      #undef __FUNCT__
      #define __FUNCT__ "createAssignment"
      template<typename Partitioner>
      static typename Partitioner::part_type *createAssignment(const Obj<topology_type>& topology, const int dim, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {

        // 1) Form partition point overlap a priori
        if (topology->commRank() == 0) {
          for(int p = 1; p < topology->commSize(); p++) {
            // The arrow is from local partition point p to remote partition point p on rank p
            sendOverlap->addCone(p, p, p);
          }
        } else {
          // The arrow is from remote partition point rank on rank 0 to local partition point rank
          recvOverlap->addCone(0, topology->commRank(), topology->commRank());
        }
        if (topology->debug()) {
          sendOverlap->view("Send overlap for partition");
          recvOverlap->view("Receive overlap for partition");
        }
        // 2) Partition the mesh
        return Partitioner::partitionSieve(topology, dim);
      };
      #undef __FUNCT__
      #define __FUNCT__ "createAssignmentByFace"
      template<typename Partitioner>
      static typename Partitioner::part_type *createAssignmentByFace(const Obj<topology_type>& topology, const int dim, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {

        // 1) Form partition point overlap a priori
        if (topology->commRank() == 0) {
          for(int p = 1; p < topology->commSize(); p++) {
            // The arrow is from local partition point p to remote partition point p on rank p
            sendOverlap->addCone(p, p, p);
          }
        } else {
          // The arrow is from remote partition point rank on rank 0 to local partition point rank
          recvOverlap->addCone(0, topology->commRank(), topology->commRank());
        }
        if (topology->debug()) {
          sendOverlap->view("Send overlap for partition");
          recvOverlap->view("Receive overlap for partition");
        }
        // 2) Partition the mesh
        return Partitioner::partitionSieveByFace(topology, dim);
      };
      #undef __FUNCT__
      #define __FUNCT__ "scatterTopology"
      // Partition a topology on process 0 and scatter to all processes
      static void scatterTopology(const Obj<topology_type>& topology, const int dim, const Obj<topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const std::string& partitioner, const Obj<topology_type>& subTopology = NULL, const Obj<topology_type>& subTopologyNew = NULL) {
        if (partitioner == "chaco") {
#ifdef PETSC_HAVE_CHACO
          typedef typename ALE::New::Chaco::Partitioner<topology_type> Partitioner;
          typedef typename ALE::New::Partitioner<topology_type>        GenPartitioner;
          typedef typename Partitioner::part_type                      part_type;

          part_type *assignment = scatterTopology<Partitioner>(topology, dim, topologyNew, sendOverlap, recvOverlap);
          if (!subTopology.isNull() && !subTopologyNew.isNull()) {
            part_type *subAssignment = GenPartitioner::subordinatePartition(topology, 1, subTopology, assignment);
            const typename topology_type::patch_type patch      = 0;
            const Obj<sieve_type>&                   sieve      = subTopology->getPatch(patch);
            const Obj<sieve_type>&                   sieveNew   = subTopologyNew->getPatch(patch);
            const int                                numCells   = subTopology->heightStratum(patch, 0)->size();

            sieveCompletion::scatterSieve(subTopology, sieve, dim, sieveNew, sendOverlap, recvOverlap, numCells, subAssignment);
            subTopologyNew->stratify();
            delete [] subAssignment;
          }
          delete [] assignment;
#else
          throw ALE::Exception("Chaco is not installed. Reconfigure with the flag --download-chaco");
#endif
        } else if (partitioner == "parmetis") {
#ifdef PETSC_HAVE_PARMETIS
          typedef typename ALE::New::ParMetis::Partitioner<topology_type> Partitioner;
          typedef typename Partitioner::part_type                         part_type;

          part_type *assignment = scatterTopology<Partitioner>(topology, dim, topologyNew, sendOverlap, recvOverlap);
          delete [] assignment;
#else
          throw ALE::Exception("ParMetis is not installed. Reconfigure with the flag --download-parmetis");
#endif
        } else {
          throw ALE::Exception("Unknown partitioner");
        }
      };
      template<typename Partitioner>
      static typename Partitioner::part_type *scatterTopology(const Obj<topology_type>& topology, const int dim, const Obj<topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typename Partitioner::part_type         *assignment = createAssignment<Partitioner>(topology, dim, sendOverlap, recvOverlap);
        const typename topology_type::patch_type patch      = 0;
        const Obj<sieve_type>&                   sieve      = topology->getPatch(patch);
        const Obj<sieve_type>&                   sieveNew   = topologyNew->getPatch(patch);
        const int                                numCells   = topology->heightStratum(patch, 0)->size();

        sieveCompletion::scatterSieve(topology, sieve, dim, sieveNew, sendOverlap, recvOverlap, numCells, assignment);
        topologyNew->stratify();
        return assignment;
      };
      template<typename Partitioner>
      static typename Partitioner::part_type *scatterTopologyByFace(const Obj<topology_type>& topology, const int dim, const Obj<topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typename Partitioner::part_type         *assignment = createAssignmentByFace<Partitioner>(topology, dim, sendOverlap, recvOverlap);
        const typename topology_type::patch_type patch      = 0;
        const Obj<sieve_type>&                   sieve      = topology->getPatch(patch);
        const Obj<sieve_type>&                   sieveNew   = topologyNew->getPatch(patch);
        const int                                numFaces   = topology->heightStratum(patch, 1)->size();

        sieveCompletion::scatterSieveByFace(topology, sieve, dim, sieveNew, sendOverlap, recvOverlap, numFaces, assignment);
        topologyNew->stratify();
        return assignment;
      };
      static void unifyTopology(const Obj<topology_type>& topology, const int dim, const Obj<topology_type>& topologyNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, int> send_sizer_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, int> recv_sizer_type;
        typedef typename ALE::New::ConstantSection<topology_type, int>                  constant_sizer;
        typedef int part_type;
        const Obj<sieve_type>& sieve         = topology->getPatch(0);
        const Obj<sieve_type>& sieveNew      = topologyNew->getPatch(0);
        Obj<send_sizer_type>   sendSizer     = new send_sizer_type(topology->comm(), topology->debug());
        Obj<recv_sizer_type>   recvSizer     = new recv_sizer_type(topology->comm(), sendSizer->getTag(), topology->debug());
        Obj<constant_sizer>    constantSizer = new constant_sizer(MPI_COMM_SELF, 1, topology->debug());
        int rank  = topology->commRank();
        int debug = topology->debug();

        // 1) Form partition point overlap a priori
        if (rank == 0) {
          for(int p = 1; p < sieve->commSize(); p++) {
            // The arrow is from remote partition point 0 on rank p to local partition point 0
            recvOverlap->addCone(p, 0, 0);
          }
        } else {
          // The arrow is from local partition point 0 to remote partition point 0 on rank 0
          sendOverlap->addCone(0, 0, 0);
        }
        if (debug) {
          sendOverlap->view("Send overlap for partition");
          recvOverlap->view("Receive overlap for partition");
        }
        // 2) Partition the mesh
        int        numCells = topology->heightStratum(0, 0)->size();
        part_type *assignment = new part_type[numCells];

        for(int c = 0; c < numCells; ++c) {
          assignment[c] = 0;
        }
        // 3) Scatter the sieve
        sieveCompletion::scatterSieve(topology, sieve, dim, sieveNew, sendOverlap, recvOverlap, numCells, assignment);
        topologyNew->stratify();
        // 4) Cleanup
        delete [] assignment;
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeMesh"
      static Obj<Mesh> distributeMesh(const Obj<Mesh>& serialMesh, const std::string& partitioner = "chaco") {
        Obj<Mesh> parallelMesh = new Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug());
        const Obj<Mesh::topology_type>& serialTopology   = serialMesh->getTopology();
        const Obj<Mesh::topology_type>& parallelTopology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug());
        const Obj<Mesh::topology_type>& tractionTopology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug());
        const Obj<Mesh::sieve_type>&    sieve            = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug());
        const int                       dim              = serialMesh->getDimension();
        PetscErrorCode                  ierr;

        if (serialMesh->getDistributed()) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        // Why in the hell do I need this here????
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        parallelTopology->setPatch(0, sieve);
        parallelMesh->setTopology(parallelTopology);
        if (serialMesh->debug()) {
          serialMesh->view("Serial topology");
        }

        // Distribute cones
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(serialTopology->comm(), serialTopology->debug());
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(serialTopology->comm(), serialTopology->debug());
        if (serialMesh->hasRealSection("traction")) {
          const Obj<Mesh::real_section_type>& traction = serialMesh->getRealSection("traction");

          scatterTopology(serialTopology, dim, parallelTopology, sendOverlap, recvOverlap, partitioner, traction->getTopology(), tractionTopology);
        } else {
          scatterTopology(serialTopology, dim, parallelTopology, sendOverlap, recvOverlap, partitioner);
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        parallelTopology->setDistSendOverlap(sendOverlap);
        parallelTopology->setDistRecvOverlap(recvOverlap);

        // Distribute labels
        const typename topology_type::labels_type& labels = serialTopology->getLabels();

        for(typename topology_type::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
          for(typename topology_type::label_type::const_iterator pl_iter = l_iter->second.begin(); pl_iter != l_iter->second.end(); ++pl_iter) {
            if (parallelTopology->hasLabel(l_iter->first, pl_iter->first)) continue;
            const Obj<typename topology_type::patch_label_type>& serialLabel   = pl_iter->second;
            const Obj<typename topology_type::patch_label_type>& parallelLabel = parallelTopology->createLabel(pl_iter->first, l_iter->first);
            // Create local label
            const Obj<typename topology_type::patch_label_type::traits::baseSequence>& base = serialLabel->base();
            const Obj<typename topology_type::sieve_type>& parallelSieve = parallelTopology->getPatch(pl_iter->first);

            for(typename topology_type::patch_label_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
              if (parallelSieve->capContains(*b_iter) || parallelSieve->baseContains(*b_iter)) {
                parallelLabel->addArrow(*serialLabel->cone(*b_iter)->begin(), *b_iter);
              }
            }
            // Get remote labels
            sieveCompletion::scatterCones(serialLabel, parallelLabel, sendOverlap, recvOverlap);
          }
        }
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);

        // Distribute sections
        Obj<std::set<std::string> > sections = serialMesh->getRealSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          if (*name == "traction") {
            parallelMesh->setRealSection(*name, distributeSection(serialMesh->getRealSection(*name), tractionTopology, sendOverlap, recvOverlap));
          } else {
            parallelMesh->setRealSection(*name, distributeSection(serialMesh->getRealSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
          }
        }
        sections = serialMesh->getIntSections();
        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->setIntSection(*name, distributeSection(serialMesh->getIntSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
        }
        sections = serialMesh->getPairSections();
        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->setPairSection(*name, distributeSection(serialMesh->getPairSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
        }

        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug()) {parallelMesh->view("Parallel Mesh");}
        parallelMesh->setDistributed(true);
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeMeshByFace"
      static Obj<Mesh> distributeMeshByFace(const Obj<Mesh>& serialMesh, const std::string& partitioner = "chaco") {
        Obj<Mesh> parallelMesh = new Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug());
        const Obj<Mesh::topology_type>& serialTopology   = serialMesh->getTopology();
        const Obj<Mesh::topology_type>& parallelTopology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug());
        const Obj<Mesh::sieve_type>&    sieve            = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug());
        const int                       dim              = serialMesh->getDimension();
        PetscErrorCode                  ierr;

        if (serialMesh->getDistributed()) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        // Why in the hell do I need this here????
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        parallelTopology->setPatch(0, sieve);
        parallelMesh->setTopology(parallelTopology);
        if (serialMesh->debug()) {serialMesh->view("Serial topology");}

        // Distribute cones
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(serialTopology->comm(), serialTopology->debug());
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(serialTopology->comm(), serialTopology->debug());
        //scatterTopology(serialTopology, dim, parallelTopology, sendOverlap, recvOverlap, partitioner);
        {
          typedef ALE::New::ParMetis::Partitioner<topology_type> Partitioner;
          typedef typename Partitioner::part_type                part_type;

          part_type *assignment = scatterTopologyByFace<Partitioner>(serialTopology, dim, parallelTopology, sendOverlap, recvOverlap);
          delete [] assignment;
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        parallelTopology->setDistSendOverlap(sendOverlap);
        parallelTopology->setDistRecvOverlap(recvOverlap);

        // Distribute sections
        Obj<std::set<std::string> > sections = serialMesh->getRealSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->setRealSection(*name, distributeSection(serialMesh->getRealSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
        }
        sections = serialMesh->getIntSections();
        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->setIntSection(*name, distributeSection(serialMesh->getIntSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
        }
        sections = serialMesh->getPairSections();
        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          parallelMesh->setPairSection(*name, distributeSection(serialMesh->getPairSection(*name), parallelMesh->getTopology(), sendOverlap, recvOverlap));
        }

        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug()) {parallelMesh->view("Parallel Mesh");}
        parallelMesh->setDistributed(true);
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeSection"
      template<typename Section>
      static Obj<Section> distributeSection(const Obj<Section>& serialSection, const Obj<Mesh::topology_type>& parallelTopology, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        if (serialSection->debug()) {
          serialSection->view("Serial Section");
        }
        typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, typename Section::value_type> send_section_type;
        typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, typename Section::value_type> recv_section_type;
        typedef SizeSection<Section>      SectionSizer;
        // TEST THIS! I think this is unnecessary
        typedef PatchlessSection<Section> SectionFiller;
        Obj<Section>                       parallelSection = new Section(parallelTopology);
        const typename Section::patch_type patch           = 0;
        const Obj<send_section_type>       sendSection     = new send_section_type(serialSection->comm(), serialSection->debug());
        const Obj<recv_section_type>       recvSection     = new recv_section_type(serialSection->comm(), sendSection->getTag(), serialSection->debug());
        const Obj<SectionSizer>            sizer           = new SectionSizer(serialSection, patch);
        const Obj<SectionFiller>           filler          = new SectionFiller(serialSection, patch);

        updateSectionLocal(serialSection, parallelSection);
        sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
        updateSectionRemote(recvOverlap, recvSection, parallelSection);
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelSection->debug()) {
          parallelSection->view("Parallel Section");
        }
        return parallelSection;
      };
      #undef __FUNCT__
      #define __FUNCT__ "completeSection"
      template<typename Section>
      static void completeSection(const Obj<Section>& section) {
        typedef typename Section::topology_type                       topology_type;
        typedef typename Distribution<topology_type>::sieveCompletion sieveCompletion;
        typedef typename topology_type::send_overlap_type             send_overlap_type;
        typedef typename topology_type::recv_overlap_type             recv_overlap_type;
        typedef typename Section::value_type                          value_type;
        typedef OverlapValues<send_overlap_type, typename sieveCompletion::topology_type, value_type> send_section_type;
        typedef OverlapValues<recv_overlap_type, typename sieveCompletion::topology_type, value_type> recv_section_type;
        typedef SizeSection<Section>      SectionSizer;
        typedef PatchlessSection<Section> SectionFiller;
        const Obj<topology_type>&                topology = section->getTopology();
        const typename topology_type::patch_type patch    = 0;
        const int                                debug    = section->debug();
        topology->constructOverlap(patch);

        const Obj<send_overlap_type> sendOverlap = topology->getSendOverlap();
        const Obj<recv_overlap_type> recvOverlap = topology->getRecvOverlap();
        const Obj<send_section_type> sendSection = new send_section_type(section->comm(), section->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(section->comm(), sendSection->getTag(), section->debug());
        const Obj<SectionSizer>      sizer       = new SectionSizer(section, patch);
        const Obj<SectionFiller>     filler      = new SectionFiller(section, patch);

        sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
        // Update section with remote data
        const Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = topology->getRecvOverlap()->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
          const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
            if (recvSection->getFiberDimension(*p_iter, *r_iter)) {
              if (debug) {std::cout << "["<<section->commRank()<<"]Completed point " << *r_iter << std::endl;}
              section->updateAddPoint(patch, *r_iter, recvSection->restrictPoint(*p_iter, *r_iter));
            }
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "unifyMesh"
      static Obj<Mesh> unifyMesh(const Obj<Mesh>& parallelMesh) {
        Obj<Mesh> serialMesh = new Mesh(parallelMesh->comm(), parallelMesh->getDimension(), parallelMesh->debug());
        const Obj<Mesh::topology_type>& parallelTopology = parallelMesh->getTopology();
        const Obj<Mesh::topology_type>& serialTopology   = new Mesh::topology_type(parallelMesh->comm(), parallelMesh->debug());
        const Obj<Mesh::sieve_type>&    sieve            = new Mesh::sieve_type(parallelMesh->comm(), parallelMesh->debug());
        const int                       dim              = parallelMesh->getDimension();
        PetscErrorCode                  ierr;

        if (!parallelMesh->getDistributed()) return parallelMesh;
        ALE_LOG_EVENT_BEGIN;
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        serialTopology->setPatch(0, sieve);
        serialMesh->setTopology(serialTopology);
        if (parallelMesh->debug()) {
          parallelMesh->view("Parallel topology");
        }

        // Unify cones
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(serialTopology->comm(), serialTopology->debug());
        Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(serialTopology->comm(), serialTopology->debug());
        unifyTopology(parallelTopology, dim, serialTopology, sendOverlap, recvOverlap);
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        serialTopology->setDistSendOverlap(sendOverlap);
        serialTopology->setDistRecvOverlap(recvOverlap);

        // Unify labels
        const typename topology_type::labels_type& labels = parallelTopology->getLabels();

        for(typename topology_type::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
          for(typename topology_type::label_type::const_iterator pl_iter = l_iter->second.begin(); pl_iter != l_iter->second.end(); ++pl_iter) {
            if (serialTopology->hasLabel(l_iter->first, pl_iter->first)) continue;
            const Obj<typename topology_type::patch_label_type>& parallelLabel = pl_iter->second;
            const Obj<typename topology_type::patch_label_type>& serialLabel   = serialTopology->createLabel(pl_iter->first, l_iter->first);

            sieveCompletion::scatterCones(parallelLabel, serialLabel, sendOverlap, recvOverlap);
          }
        }
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);

        // Unify coordinates
        serialMesh->setRealSection("coordinates", distributeSection(parallelMesh->getRealSection("coordinates"), serialTopology, sendOverlap, recvOverlap));

        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (serialMesh->debug()) {serialMesh->view("Serial Mesh");}
        serialMesh->setDistributed(false);
        ALE_LOG_EVENT_END;
        return serialMesh;
      };
    };
  }
}

#endif
