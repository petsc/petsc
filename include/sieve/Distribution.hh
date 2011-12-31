#ifndef included_ALE_Distribution_hh
#define included_ALE_Distribution_hh

#ifndef  included_ALE_Mesh_hh
#include <sieve/Mesh.hh>
#endif

#ifndef  included_ALE_Completion_hh
#include <sieve/Completion.hh>
#endif

#ifndef  included_ALE_SectionCompletion_hh
#include <sieve/SectionCompletion.hh>
#endif

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
  template<typename Mesh, typename Partitioner = ALE::Partitioner<> >
  class DistributionNew {
  public:
    typedef Partitioner                                   partitioner_type;
    typedef typename Mesh::point_type                     point_type;
    typedef OrientedPoint<point_type>                     oriented_point_type;
    typedef typename Partitioner::part_type               rank_type;
    typedef ALE::ISection<rank_type, point_type>          partition_type;
    typedef ALE::Section<ALE::Pair<int, point_type>, point_type>          cones_type;
    typedef ALE::Section<ALE::Pair<int, point_type>, oriented_point_type> oriented_cones_type;
  public:
    template<typename Sieve, typename NewSieve, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<cones_type> completeCones(const Obj<Sieve>& sieve, const Obj<NewSieve>& newSieve, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      typedef ALE::ConeSection<Sieve> cones_wrapper_type;
      Obj<cones_wrapper_type> cones        = new cones_wrapper_type(sieve);
      Obj<cones_type>         overlapCones = new cones_type(sieve->comm(), sieve->debug());

      ALE::Pullback::SimpleCopy::copy(sendMeshOverlap, recvMeshOverlap, cones, overlapCones);
      if (sieve->debug()) {overlapCones->view("Overlap Cones");}
      // Inserts cones into parallelMesh (must renumber here)
      ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvMeshOverlap, renumbering, newSieve);
      return overlapCones;
    }
    template<typename Sieve, typename NewSieve, typename SendOverlap, typename RecvOverlap>
    static Obj<oriented_cones_type> completeConesV(const Obj<Sieve>& sieve, const Obj<NewSieve>& newSieve, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      typedef ALE::OrientedConeSectionV<Sieve> oriented_cones_wrapper_type;
      Obj<oriented_cones_wrapper_type> cones        = new oriented_cones_wrapper_type(sieve);
      Obj<oriented_cones_type>         overlapCones = new oriented_cones_type(sieve->comm(), sieve->debug());

      ALE::Pullback::SimpleCopy::copy(sendMeshOverlap, recvMeshOverlap, cones, overlapCones);
      if (sieve->debug()) {overlapCones->view("Overlap Cones");}
      ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvMeshOverlap, newSieve);
      return overlapCones;
    }
    template<typename Sieve, typename NewSieve, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<oriented_cones_type> completeConesV(const Obj<Sieve>& sieve, const Obj<NewSieve>& newSieve, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      typedef ALE::OrientedConeSectionV<Sieve> oriented_cones_wrapper_type;
      Obj<oriented_cones_wrapper_type> cones        = new oriented_cones_wrapper_type(sieve);
      Obj<oriented_cones_type>         overlapCones = new oriented_cones_type(sieve->comm(), sieve->debug());

      ALE::Pullback::SimpleCopy::copy(sendMeshOverlap, recvMeshOverlap, cones, overlapCones);
      if (sieve->debug()) {overlapCones->view("Overlap Cones");}
      // Inserts cones into parallelMesh (must renumber here)
      ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvMeshOverlap, renumbering, newSieve);
      return overlapCones;
    }
    // Given a partition of sieve points, copy the mesh pieces to each process and fuse into the new mesh
    //   Return overlaps for the cone communication
    template<typename Renumbering, typename NewMesh, typename SendOverlap, typename RecvOverlap>
    static void completeMesh(const Obj<Mesh>& mesh, const Obj<partition_type>& partition, Renumbering& renumbering, const Obj<NewMesh>& newMesh, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      typedef ALE::Sifter<rank_type,rank_type,rank_type> part_send_overlap_type;
      typedef ALE::Sifter<rank_type,rank_type,rank_type> part_recv_overlap_type;
      const Obj<part_send_overlap_type> sendOverlap = new part_send_overlap_type(partition->comm());
      const Obj<part_recv_overlap_type> recvOverlap = new part_recv_overlap_type(partition->comm());

      // Create overlap for partition points
      //   TODO: This needs to be generalized for multiple sources
      Partitioner::createDistributionPartOverlap(sendOverlap, recvOverlap);
      // Communicate partition pieces to processes
      Obj<partition_type> overlapPartition = new partition_type(partition->comm(), partition->debug());

      overlapPartition->setChart(partition->getChart());
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, partition, overlapPartition);
      // Create renumbering
      const int         rank           = partition->commRank();
      const point_type *localPoints    = partition->restrictPoint(rank);
      const int         numLocalPoints = partition->getFiberDimension(rank);

      for(point_type p = 0; p < numLocalPoints; ++p) {
        renumbering[localPoints[p]] = p;
      }
      const Obj<typename part_recv_overlap_type::traits::baseSequence> rPoints    = recvOverlap->base();
      point_type                                                       localPoint = numLocalPoints;

      for(typename part_recv_overlap_type::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
        const Obj<typename part_recv_overlap_type::coneSequence>& ranks           = recvOverlap->cone(*p_iter);
        const rank_type&                                          remotePartPoint = ranks->begin().color();
        const point_type                                         *points          = overlapPartition->restrictPoint(remotePartPoint);
        const int                                                 numPoints       = overlapPartition->getFiberDimension(remotePartPoint);

        for(int i = 0; i < numPoints; ++i) {
          renumbering[points[i]] = localPoint++;
        }
      }
      // Create mesh overlap from partition overlap
      //   TODO: Generalize to redistribution (receive from multiple sources)
      Partitioner::createDistributionMeshOverlap(partition, recvOverlap, renumbering, overlapPartition, sendMeshOverlap, recvMeshOverlap);
      // Send cones
      completeCones(mesh->getSieve(), newMesh->getSieve(), renumbering, sendMeshOverlap, recvMeshOverlap);
    }
    template<typename Renumbering, typename NewMesh, typename SendOverlap, typename RecvOverlap>
    static void completeBaseV(const Obj<Mesh>& mesh, const Obj<partition_type>& partition, Renumbering& renumbering, const Obj<NewMesh>& newMesh, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      typedef ALE::Sifter<rank_type,rank_type,rank_type> part_send_overlap_type;
      typedef ALE::Sifter<rank_type,rank_type,rank_type> part_recv_overlap_type;
      const Obj<part_send_overlap_type> sendOverlap = new part_send_overlap_type(partition->comm());
      const Obj<part_recv_overlap_type> recvOverlap = new part_recv_overlap_type(partition->comm());

      // Create overlap for partition points
      //   TODO: This needs to be generalized for multiple sources
      Partitioner::createDistributionPartOverlap(sendOverlap, recvOverlap);
      // Communicate partition pieces to processes
      Obj<partition_type> overlapPartition = new partition_type(partition->comm(), partition->debug());

      overlapPartition->setChart(partition->getChart());
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, partition, overlapPartition);
      // Create renumbering
      const int         rank           = partition->commRank();
      const point_type *localPoints    = partition->restrictPoint(rank);
      const int         numLocalPoints = partition->getFiberDimension(rank);

      for(point_type p = 0; p < numLocalPoints; ++p) {
        ///std::cout <<"["<<partition->commRank()<<"]: local renumbering " << localPoints[p] << " --> " << p << std::endl;
        renumbering[localPoints[p]] = p;
      }
      const Obj<typename part_recv_overlap_type::traits::baseSequence> rPoints    = recvOverlap->base();
      point_type                                                       localPoint = numLocalPoints;

      for(typename part_recv_overlap_type::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
        const Obj<typename part_recv_overlap_type::coneSequence>& ranks           = recvOverlap->cone(*p_iter);
        const rank_type&                                          remotePartPoint = ranks->begin().color();
        const point_type                                         *points          = overlapPartition->restrictPoint(remotePartPoint);
        const int                                                 numPoints       = overlapPartition->getFiberDimension(remotePartPoint);

        for(int i = 0; i < numPoints; ++i) {
          ///std::cout <<"["<<partition->commRank()<<"]: remote renumbering " << points[i] << " --> " << localPoint << std::endl;
          renumbering[points[i]] = localPoint++;
        }
      }
      newMesh->getSieve()->setChart(typename NewMesh::sieve_type::chart_type(0, renumbering.size()));
      // Create mesh overlap from partition overlap
      //   TODO: Generalize to redistribution (receive from multiple sources)
      Partitioner::createDistributionMeshOverlap(partition, recvOverlap, renumbering, overlapPartition, sendMeshOverlap, recvMeshOverlap);
    }
    template<typename NewMesh, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<partition_type> distributeMesh(const Obj<Mesh>& mesh, const Obj<NewMesh>& newMesh, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap, const int height = 0) {
      const Obj<partition_type> cellPartition = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());
      const Obj<partition_type> partition     = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());

      // Create the cell partition
      Partitioner::createPartition(mesh, cellPartition, height);
      if (mesh->debug()) {
        PetscViewer    viewer;
        PetscErrorCode ierr;

        cellPartition->view("Cell Partition");
        ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
        ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRXX(ierr);
        ///TODO ierr = MeshView_Sieve_Ascii(mesh, cellPartition, viewer);CHKERRXX(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
      }
      // Close the partition over sieve points
      Partitioner::createPartitionClosure(mesh, cellPartition, partition, height);
      if (mesh->debug()) {partition->view("Partition");}
      // Create the remote meshes
      completeMesh(mesh, partition, renumbering, newMesh, sendMeshOverlap, recvMeshOverlap);
      // Create the local mesh
      Partitioner::createLocalMesh(mesh, partition, renumbering, newMesh, height);
      newMesh->stratify();
      return partition;
    }
    template<typename NewMesh, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<partition_type> distributeMeshAndSections(const Obj<Mesh>& mesh, const Obj<NewMesh>& newMesh, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap, const int height = 0) {
      Obj<partition_type> partition = distributeMesh(mesh, newMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);

      // Distribute the coordinates
      const Obj<typename Mesh::real_section_type>& coordinates         = mesh->getRealSection("coordinates");
      const Obj<typename Mesh::real_section_type>& parallelCoordinates = newMesh->getRealSection("coordinates");

      newMesh->setupCoordinates(parallelCoordinates);
      distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
      // Distribute other sections
      if (mesh->getRealSections()->size() > 1) {
        Obj<std::set<std::string> > names = mesh->getRealSections();

        for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
          if (*n_iter == "coordinates")   continue;
          distributeSection(mesh->getRealSection(*n_iter), partition, renumbering, sendMeshOverlap, recvMeshOverlap, newMesh->getRealSection(*n_iter));
        }
      }
      if (mesh->getIntSections()->size() > 0) {
        Obj<std::set<std::string> > names = mesh->getIntSections();

        for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
          distributeSection(mesh->getIntSection(*n_iter), partition, renumbering, sendMeshOverlap, recvMeshOverlap, newMesh->getIntSection(*n_iter));
        }
      }
      if (mesh->getArrowSections()->size() > 1) {
        throw ALE::Exception("Need to distribute more arrow sections");
      }
      // Distribute labels
      const typename Mesh::labels_type& labels = mesh->getLabels();

      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (newMesh->hasLabel(l_iter->first)) continue;
        const Obj<typename Mesh::label_type>& origLabel = l_iter->second;
        const Obj<typename Mesh::label_type>& newLabel  = newMesh->createLabel(l_iter->first);
        // Get remote labels
        ALE::New::Completion<Mesh,typename Mesh::point_type>::scatterCones(origLabel, newLabel, sendMeshOverlap, recvMeshOverlap, renumbering);
        // Create local label
        newLabel->add(origLabel, newMesh->getSieve(), renumbering);
      }
      return partition;
    }
    template<typename NewMesh, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<partition_type> distributeMeshV(const Obj<Mesh>& mesh, const Obj<NewMesh>& newMesh, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap, const int height = 0) {
      const Obj<partition_type> cellPartition = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());
      const Obj<partition_type> partition     = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());

      PETSc::Log::Event("DistributeMesh").begin();
      // Create the cell partition
      Partitioner::createPartitionV(mesh, cellPartition, height);
      if (mesh->debug()) {
        PetscViewer    viewer;
        PetscErrorCode ierr;

        cellPartition->view("Cell Partition");
        ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRXX(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
        ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRXX(ierr);
        ///TODO ierr = MeshView_Sieve_Ascii(mesh, cellPartition, viewer);CHKERRXX(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
      }
      // Close the partition over sieve points
      Partitioner::createPartitionClosureV(mesh, cellPartition, partition, height);
      if (mesh->debug()) {partition->view("Partition");}
      // Create the remote bases
      completeBaseV(mesh, partition, renumbering, newMesh, sendMeshOverlap, recvMeshOverlap);
      // Size the local mesh
      Partitioner::sizeLocalMeshV(mesh, partition, renumbering, newMesh, height);
      // Create the remote meshes
      completeConesV(mesh->getSieve(), newMesh->getSieve(), renumbering, sendMeshOverlap, recvMeshOverlap);
      // Create the local mesh
      Partitioner::createLocalMeshV(mesh, partition, renumbering, newMesh, height);
      newMesh->getSieve()->symmetrize();
      newMesh->stratify();
      PETSc::Log::Event("DistributeMesh").end();
      return partition;
    }
    // distributeMeshV:
    //   createPartitionV (can be dumb)
    //   createPartitionClosureV (should be low memory)
    //   completeBaseV ( have not decided )
    //     Partitioner::createDistributionPartOverlap (low memory)
    //     copy points to partitions (uses small overlap and fake sections)
    //     renumber (map is potentially big, can measure)
    //     Partitioner::createDistributionMeshOverlap (should be large for distribution)
    //       sendMeshOverlap is localPoint--- remotePoint --->remoteRank
    //       recvMeshOverlap is remoteRank--- remotePoint --->localPoint
    //   sizeLocalMeshV (should be low memory)
    //   completeConesV ( have not decided )
    //   createLocalMesh (should be low memory)
    //   symmetrize
    //   stratify
    template<typename NewMesh>
    static void distributeMeshAndSectionsV(const Obj<Mesh>& mesh, const Obj<NewMesh>& newMesh) {
      typedef typename Mesh::point_type point_type;

      const Obj<typename Mesh::send_overlap_type> sendMeshOverlap = new typename Mesh::send_overlap_type(mesh->comm(), mesh->debug());
      const Obj<typename Mesh::recv_overlap_type> recvMeshOverlap = new typename Mesh::recv_overlap_type(mesh->comm(), mesh->debug());
      std::map<point_type,point_type>&            renumbering     = newMesh->getRenumbering();
      // Distribute the mesh
      Obj<partition_type> partition = distributeMeshV(mesh, newMesh, renumbering, sendMeshOverlap, recvMeshOverlap);
      if (mesh->debug()) {
        std::cout << "["<<mesh->commRank()<<"]: Mesh Renumbering:" << std::endl;
        for(typename Mesh::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
          std::cout << "["<<mesh->commRank()<<"]:   global point " << r_iter->first << " --> " << " local point " << r_iter->second << std::endl;
        }
      }
      // Distribute the coordinates
      PETSc::Log::Event("DistributeCoords").begin();
      const Obj<typename Mesh::real_section_type>& coordinates         = mesh->getRealSection("coordinates");
      const Obj<typename Mesh::real_section_type>& parallelCoordinates = newMesh->getRealSection("coordinates");

      newMesh->setupCoordinates(parallelCoordinates);
      distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
      PETSc::Log::Event("DistributeCoords").end();
      // Distribute other sections
      if (mesh->getRealSections()->size() > 1) {
        PETSc::Log::Event("DistributeRealSec").begin();
        Obj<std::set<std::string> > names = mesh->getRealSections();
        int                         n     = 0;

        for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
          if (*n_iter == "coordinates")   continue;
          std::cout << "ERROR: Did not distribute real section " << *n_iter << std::endl;
          ++n;
        }
        PETSc::Log::Event("DistributeRealSec").end();
        if (n) {throw ALE::Exception("Need to distribute more real sections");}
      }
      if (mesh->getIntSections()->size() > 0) {
        PETSc::Log::Event("DistributeIntSec").begin();
        Obj<std::set<std::string> > names = mesh->getIntSections();

        for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
          const Obj<typename Mesh::int_section_type>& section    = mesh->getIntSection(*n_iter);
          const Obj<typename Mesh::int_section_type>& newSection = newMesh->getIntSection(*n_iter);
          
          // We assume all integer sections are complete sections
          newSection->setChart(newMesh->getSieve()->getChart());
          distributeSection(section, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newSection);
        }
        PETSc::Log::Event("DistributeIntSec").end();
      }
      if (mesh->getArrowSections()->size() > 1) {
        throw ALE::Exception("Need to distribute more arrow sections");
      }
      // Distribute labels
      PETSc::Log::Event("DistributeLabels").begin();
      const typename Mesh::labels_type& labels = mesh->getLabels();

      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (newMesh->hasLabel(l_iter->first)) continue;
        const Obj<typename Mesh::label_type>& origLabel = l_iter->second;
        const Obj<typename Mesh::label_type>& newLabel  = newMesh->createLabel(l_iter->first);

#ifdef IMESH_NEW_LABELS
        newLabel->setChart(newMesh->getSieve()->getChart());
        // Size the local mesh
        Partitioner::sizeLocalSieveV(origLabel, partition, renumbering, newLabel);
        // Create the remote meshes
        completeConesV(origLabel, newLabel, renumbering, sendMeshOverlap, recvMeshOverlap);
        // Create the local mesh
        Partitioner::createLocalSieveV(origLabel, partition, renumbering, newLabel);
        newLabel->symmetrize();
#else
	distributeLabelV(newMesh->getSieve(), origLabel, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newLabel);
#endif
      }
      PETSc::Log::Event("DistributeLabels").end();
      // Create the parallel overlap
      PETSc::Log::Event("CreateOverlap").begin();
      Obj<typename Mesh::send_overlap_type> sendParallelMeshOverlap = newMesh->getSendOverlap();
      Obj<typename Mesh::recv_overlap_type> recvParallelMeshOverlap = newMesh->getRecvOverlap();
      //   Can I figure this out in a nicer way?
      ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

      ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
      newMesh->setCalculatedOverlap(true);
      PETSc::Log::Event("CreateOverlap").end();
    }
    template<typename Label, typename Partition, typename Renumbering, typename SendOverlap, typename RecvOverlap, typename NewLabel>
    static void distributeLabel(const Obj<typename Mesh::sieve_type>& sieve, const Obj<Label>& l, const Obj<Partition>& partition, Renumbering& renumbering, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<NewLabel>& newL) {
      Partitioner::createLocalSifter(l, partition, renumbering, newL);
      //completeCones(l, newL, renumbering, sendMeshOverlap, recvMeshOverlap);
      {
        typedef ALE::UniformSection<point_type, int>                cones_type;
        typedef ALE::LabelSection<typename Mesh::sieve_type, Label> cones_wrapper_type;
        Obj<cones_wrapper_type> cones        = new cones_wrapper_type(sieve, l);
        Obj<cones_type>         overlapCones = new cones_type(l->comm(), l->debug());

        ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, cones, overlapCones);
        if (l->debug()) {overlapCones->view("Overlap Label Values");}
        // Inserts cones into newL (must renumber here)
        //ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvOverlap, renumbering, newSieve);
        {
	  typedef typename cones_type::point_type overlap_point_type;
          const Obj<typename RecvOverlap::baseSequence>      rPoints = recvOverlap->base();
	  const typename RecvOverlap::baseSequence::iterator rEnd    = rPoints->end();

          for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
            const Obj<typename RecvOverlap::coneSequence>& points       = recvOverlap->cone(*p_iter);
            const typename RecvOverlap::target_type&       localPoint   = *p_iter;
            const typename cones_type::point_type&         remotePoint  = points->begin().color();
	    const overlap_point_type                       overlapPoint = overlap_point_type(remotePoint.second, remotePoint.first);
            const int                                      size         = overlapCones->getFiberDimension(overlapPoint);
            const typename cones_type::value_type         *values       = overlapCones->restrictPoint(overlapPoint);

            newL->clearCone(localPoint);
            for(int i = 0; i < size; ++i) {newL->addCone(values[i], localPoint);}
          }
        }
      }
    }
    template<typename Label, typename Partition, typename Renumbering, typename SendOverlap, typename RecvOverlap, typename NewLabel>
    static void distributeLabelV(const Obj<typename Mesh::sieve_type>& sieve, const Obj<Label>& l, const Obj<Partition>& partition, Renumbering& renumbering, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<NewLabel>& newL) {
      Partitioner::createLocalSifter(l, partition, renumbering, newL);
      //completeCones(l, newL, renumbering, sendMeshOverlap, recvMeshOverlap);
      {
	typedef typename Label::alloc_type::template rebind<int>::other alloc_type;
	typedef LabelBaseSectionV<typename Mesh::sieve_type, Label, alloc_type> atlas_type;
        typedef ALE::UniformSection<ALE::Pair<int, point_type>, int>            cones_type;
        typedef ALE::LabelSection<typename Mesh::sieve_type, Label, alloc_type, atlas_type> cones_wrapper_type;
        Obj<cones_wrapper_type> cones        = new cones_wrapper_type(sieve, l);
        Obj<cones_type>         overlapCones = new cones_type(l->comm(), l->debug());

        ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, cones, overlapCones);
        if (l->debug()) {overlapCones->view("Overlap Label Values");}
        // Inserts cones into newL (must renumber here)
        //ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvOverlap, renumbering, newSieve);
        {
	  typedef typename cones_type::point_type overlap_point_type;
          const typename RecvOverlap::capSequence::iterator rBegin = recvOverlap->capBegin();
          const typename RecvOverlap::capSequence::iterator rEnd   = recvOverlap->capEnd();

          for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
            const int                                             rank    = *r_iter;
            const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
            const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

            for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
              const typename RecvOverlap::target_type& localPoint   = *p_iter;
              const typename RecvOverlap::target_type& remotePoint  = p_iter.color();
              const overlap_point_type                 overlapPoint = overlap_point_type(rank, remotePoint);
              const int                                size         = overlapCones->getFiberDimension(overlapPoint);
              const typename cones_type::value_type   *values       = overlapCones->restrictPoint(overlapPoint);

              newL->clearCone(localPoint);
              for(int i = 0; i < size; ++i) {newL->addCone(values[i], localPoint);}
            }
          }
        }
      }
    }
    template<typename Section, typename Partition, typename Renumbering, typename SendOverlap, typename RecvOverlap, typename NewSection>
    static void distributeSection(const Obj<Section>& s, const Obj<Partition>& partition, Renumbering& renumbering, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<NewSection>& newS) {
      Partitioner::createLocalSection(s, partition, renumbering, newS);
      ALE::Completion::completeSection(sendOverlap, recvOverlap, s, newS);
    }
    template<typename NewMesh, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static Obj<partition_type> unifyMesh(const Obj<Mesh>& mesh, const Obj<NewMesh>& newMesh, Renumbering& renumbering, const Obj<SendOverlap>& sendMeshOverlap, const Obj<RecvOverlap>& recvMeshOverlap) {
      const Obj<partition_type> cellPartition = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());
      const Obj<partition_type> partition     = new partition_type(mesh->comm(), 0, mesh->commSize(), mesh->debug());
      const Obj<typename Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
      const typename Mesh::label_sequence::iterator cEnd   = cells->end();
      typename Mesh::point_type                    *values = new typename Mesh::point_type[cells->size()];
      int                                           c      = 0;

      cellPartition->setFiberDimension(0, cells->size());
      cellPartition->allocatePoint();
      for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter, ++c) {
        values[c] = *c_iter;
      }
      cellPartition->updatePoint(0, values);
      delete [] values;
      // Close the partition over sieve points
      Partitioner::createPartitionClosure(mesh, cellPartition, partition);
      // Create the remote meshes
      completeMesh(mesh, partition, renumbering, newMesh, sendMeshOverlap, recvMeshOverlap);
      // Create the local mesh
      Partitioner::createLocalMesh(mesh, partition, renumbering, newMesh);
      newMesh->stratify();
      newMesh->view("Unified mesh");
      return partition;
    }
    static Obj<Mesh> unifyMesh(const Obj<Mesh>& mesh) {
      typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
      typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
      const Obj<Mesh>                      newMesh         = new Mesh(mesh->comm(), mesh->getDimension(), mesh->debug());
      const Obj<typename Mesh::sieve_type> newSieve        = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      const Obj<mesh_send_overlap_type>    sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
      const Obj<mesh_recv_overlap_type>    recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
      std::map<point_type,point_type>      renumbering;

      newMesh->setSieve(newSieve);
      const Obj<partition_type> partition = unifyMesh(mesh, newMesh, renumbering, sendMeshOverlap, recvMeshOverlap);
      // Unify coordinates
      const Obj<typename Mesh::real_section_type>& coordinates    = mesh->getRealSection("coordinates");
      const Obj<typename Mesh::real_section_type>& newCoordinates = newMesh->getRealSection("coordinates");

      newMesh->setupCoordinates(newCoordinates);
      distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newCoordinates);
      // Unify labels
      const typename Mesh::labels_type& labels = mesh->getLabels();

      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (newMesh->hasLabel(l_iter->first)) continue;
        const Obj<typename Mesh::label_type>& label    = l_iter->second;
        const Obj<typename Mesh::label_type>& newLabel = newMesh->createLabel(l_iter->first);

        //completeCones(label, newLabel, renumbering, sendMeshOverlap, recvMeshOverlap);
        {
          typedef ALE::UniformSection<point_type, int> cones_type;
          typedef ALE::LabelSection<typename Mesh::sieve_type,typename Mesh::label_type> cones_wrapper_type;
          Obj<cones_wrapper_type> cones        = new cones_wrapper_type(mesh->getSieve(), label);
          Obj<cones_type>         overlapCones = new cones_type(label->comm(), label->debug());

          ALE::Pullback::SimpleCopy::copy(sendMeshOverlap, recvMeshOverlap, cones, overlapCones);
          if (label->debug()) {overlapCones->view("Overlap Label Values");}
          // Inserts cones into parallelMesh (must renumber here)
          //ALE::Pullback::InsertionBinaryFusion::fuse(overlapCones, recvMeshOverlap, renumbering, newSieve);
          {
            const Obj<typename mesh_recv_overlap_type::baseSequence> rPoints = recvMeshOverlap->base();

            for(typename mesh_recv_overlap_type::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
              const Obj<typename mesh_recv_overlap_type::coneSequence>& points      = recvMeshOverlap->cone(*p_iter);
              const typename mesh_recv_overlap_type::target_type&       localPoint  = *p_iter;
              const typename cones_type::point_type&                    remotePoint = points->begin().color();
              const int                                                 size        = overlapCones->getFiberDimension(remotePoint);
              const typename cones_type::value_type                    *values      = overlapCones->restrictPoint(remotePoint);

              newLabel->clearCone(localPoint);
              for(int i = 0; i < size; ++i) {newLabel->addCone(values[i], localPoint);}
            }
          }
        }
        //newLabel->add(label, newSieve);
        Partitioner::createLocalSifter(label, partition, renumbering, newLabel);
        newLabel->view(l_iter->first.c_str());
      }
      return newMesh;
    }
  };
  template<typename Bundle_>
  class Distribution {
  public:
    typedef Bundle_                                                                     bundle_type;
    typedef typename bundle_type::sieve_type                                            sieve_type;
    typedef typename bundle_type::point_type                                            point_type;
    typedef typename bundle_type::alloc_type                                            alloc_type;
    typedef typename bundle_type::send_overlap_type                                     send_overlap_type;
    typedef typename bundle_type::recv_overlap_type                                     recv_overlap_type;
    typedef typename ALE::New::Completion<bundle_type, typename sieve_type::point_type>                            sieveCompletion;
    typedef typename ALE::New::SectionCompletion<bundle_type, typename bundle_type::real_section_type::value_type> sectionCompletion;
  public:
    #undef __FUNCT__
    #define __FUNCT__ "createPartitionOverlap"
    static void createPartitionOverlap(const Obj<bundle_type>& bundle, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      const Obj<send_overlap_type>& topSendOverlap = bundle->getSendOverlap();
      const Obj<recv_overlap_type>& topRecvOverlap = bundle->getRecvOverlap();
      const Obj<typename send_overlap_type::traits::baseSequence> base = topSendOverlap->base();
      const Obj<typename recv_overlap_type::traits::capSequence>  cap  = topRecvOverlap->cap();
      const int rank = bundle->commRank();

      if (base->empty()) {
        if (rank == 0) {
          for(int p = 1; p < bundle->commSize(); p++) {
            // The arrow is from local partition point p (source) to remote partition point p (color) on rank p (target)
            sendOverlap->addCone(p, p, p);
          }
        }
      } else {
        for(typename send_overlap_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const int& p = *b_iter;
          // The arrow is from local partition point p (source) to remote partition point p (color) on rank p (target)
          sendOverlap->addCone(p, p, p);
        }
      }
      if (cap->empty()) {
        if (rank != 0) {
          // The arrow is from local partition point rank (color) on rank 0 (source) to remote partition point rank (target)
          recvOverlap->addCone(0, rank, rank);
        }
      } else {
        for(typename recv_overlap_type::traits::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          const int& p = *c_iter;
          // The arrow is from local partition point rank (color) on rank p (source) to remote partition point rank (target)
          recvOverlap->addCone(p, rank, rank);
        }
      }
    };
    #undef __FUNCT__
    #define __FUNCT__ "createAssignment"
    template<typename Partitioner>
    static typename Partitioner::part_type *createAssignment(const Obj<bundle_type>& bundle, const int dim, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int height = 0) {
      // 1) Form partition point overlap a priori
      createPartitionOverlap(bundle, sendOverlap, recvOverlap);
      if (bundle->debug()) {
        sendOverlap->view("Send overlap for partition");
        recvOverlap->view("Receive overlap for partition");
      }
      // 2) Partition the mesh
      if (height == 0) {
        return Partitioner::partitionSieve(bundle, dim);
      } else if (height == 1) {
        return Partitioner::partitionSieveByFace(bundle, dim);
      }
      throw ALE::Exception("Invalid partition height");
    }
    #undef __FUNCT__
    #define __FUNCT__ "scatterBundle"
    // Partition a bundle on process 0 and scatter to all processes
    static void scatterBundle(const Obj<bundle_type>& bundle, const int dim, const Obj<bundle_type>& bundleNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const std::string& partitioner, const int height = 0, const Obj<bundle_type>& subBundle = NULL, const Obj<bundle_type>& subBundleNew = NULL) {
      if (height == 0) {
        if (partitioner == "chaco") {
#ifdef PETSC_HAVE_CHACO
          typedef typename ALE::New::Chaco::Partitioner<bundle_type> Partitioner;
          typedef typename ALE::New::Partitioner<bundle_type>        GenPartitioner;
          typedef typename Partitioner::part_type                    part_type;

          part_type *assignment = scatterBundle<Partitioner>(bundle, dim, bundleNew, sendOverlap, recvOverlap, height);
          if (!subBundle.isNull() && !subBundleNew.isNull()) {
            part_type *subAssignment = GenPartitioner::subordinatePartition(bundle, 1, subBundle, assignment);
            const Obj<sieve_type>& sieve      = subBundle->getSieve();
            const Obj<sieve_type>& sieveNew   = new typename ALE::Mesh<PetscInt,PetscScalar>::sieve_type(subBundle->comm(), subBundle->debug());
            const int              numCells   = subBundle->heightStratum(height)->size();

            subBundleNew->setSieve(sieveNew);
            sieveCompletion::scatterSieve(subBundle, sieve, dim, sieveNew, sendOverlap, recvOverlap, height, numCells, subAssignment);
            subBundleNew->stratify();
            if (subAssignment != NULL) delete [] subAssignment;
          }
          if (assignment != NULL) delete [] assignment;
#else
          throw ALE::Exception("Chaco is not installed. Reconfigure with the flag --download-chaco");
#endif
        } else if (partitioner == "parmetis") {
#ifdef PETSC_HAVE_PARMETIS
          typedef typename ALE::New::ParMetis::Partitioner<bundle_type> Partitioner;
          typedef typename ALE::New::Partitioner<bundle_type>           GenPartitioner;
          typedef typename Partitioner::part_type                       part_type;

          part_type *assignment = scatterBundle<Partitioner>(bundle, dim, bundleNew, sendOverlap, recvOverlap, height);
          if (!subBundle.isNull() && !subBundleNew.isNull()) {
            part_type *subAssignment = GenPartitioner::subordinatePartition(bundle, 1, subBundle, assignment);
            const Obj<sieve_type>& sieve      = subBundle->getSieve();
            const Obj<sieve_type>& sieveNew   = new typename ALE::Mesh<PetscInt,PetscScalar>::sieve_type(subBundle->comm(), subBundle->debug());
            const int              numCells   = subBundle->heightStratum(height)->size();

            subBundleNew->setSieve(sieveNew);
            sieveCompletion::scatterSieve(subBundle, sieve, dim, sieveNew, sendOverlap, recvOverlap, height, numCells, subAssignment);
            subBundleNew->stratify();
            if (subAssignment != NULL) delete [] subAssignment;
          }
          if (assignment != NULL) delete [] assignment;
#else
          throw ALE::Exception("ParMetis is not installed. Reconfigure with the flag --download-parmetis");
#endif
        } else {
          throw ALE::Exception("Unknown partitioner");
        }
      } else if (height == 1) {
        if (partitioner == "zoltan") {
#ifdef PETSC_HAVE_ZOLTAN
          typedef typename ALE::New::Zoltan::Partitioner<bundle_type> Partitioner;
          typedef typename Partitioner::part_type                     part_type;

          part_type *assignment = scatterBundle<Partitioner>(bundle, dim, bundleNew, sendOverlap, recvOverlap, height);
          if (assignment != NULL) delete [] assignment;
#else
          throw ALE::Exception("Zoltan is not installed. Reconfigure with the flag --download-zoltan");
#endif
        } else if (partitioner == "parmetis") {
#ifdef PETSC_HAVE_PARMETIS
          typedef typename ALE::New::ParMetis::Partitioner<bundle_type> Partitioner;
          typedef typename Partitioner::part_type                       part_type;

          part_type *assignment = scatterBundle<Partitioner>(bundle, dim, bundleNew, sendOverlap, recvOverlap, height);
          if (assignment != NULL) delete [] assignment;
#else
          throw ALE::Exception("ParMetis is not installed. Reconfigure with the flag --download-parmetis");
#endif
        } else {
          throw ALE::Exception("Unknown partitioner");
        }
      } else {
        throw ALE::Exception("Invalid partition height");
      }
    }
    template<typename Partitioner>
    static typename Partitioner::part_type *scatterBundle(const Obj<bundle_type>& bundle, const int dim, const Obj<bundle_type>& bundleNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int height = 0) {
      typename Partitioner::part_type *assignment = createAssignment<Partitioner>(bundle, dim, sendOverlap, recvOverlap, height);
      const Obj<sieve_type>&           sieve      = bundle->getSieve();
      const Obj<sieve_type>&           sieveNew   = bundleNew->getSieve();
      const int                        numPoints  = bundle->heightStratum(height)->size();

      sieveCompletion::scatterSieve(bundle, sieve, dim, sieveNew, sendOverlap, recvOverlap, height, numPoints, assignment);
      bundleNew->stratify();
      return assignment;
    }
    #undef __FUNCT__
    #define __FUNCT__ "distributeMesh"
    static Obj<ALE::Mesh<PetscInt,PetscScalar> > distributeMesh(const Obj<ALE::Mesh<PetscInt,PetscScalar> >& serialMesh, const int height = 0, const std::string& partitioner = "chaco") {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      MPI_Comm                         comm          = serialMesh->comm();
      const int                        dim           = serialMesh->getDimension();
      Obj<FlexMesh>                    parallelMesh  = new FlexMesh(comm, dim, serialMesh->debug());
      const Obj<FlexMesh::sieve_type>& parallelSieve = new FlexMesh::sieve_type(comm, serialMesh->debug());

      ALE_LOG_EVENT_BEGIN;
      parallelMesh->setSieve(parallelSieve);
      if (serialMesh->debug()) {serialMesh->view("Serial mesh");}

      // Distribute cones
      Obj<send_overlap_type> sendOverlap = new send_overlap_type(comm, serialMesh->debug());
      Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(comm, serialMesh->debug());
      scatterBundle(serialMesh, dim, parallelMesh, sendOverlap, recvOverlap, partitioner, height);
      parallelMesh->setDistSendOverlap(sendOverlap);
      parallelMesh->setDistRecvOverlap(recvOverlap);

      // Distribute labels
      const typename bundle_type::labels_type& labels = serialMesh->getLabels();

      for(typename bundle_type::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (parallelMesh->hasLabel(l_iter->first)) continue;
        const Obj<typename bundle_type::label_type>& serialLabel   = l_iter->second;
        const Obj<typename bundle_type::label_type>& parallelLabel = parallelMesh->createLabel(l_iter->first);
        // Create local label
#define NEW_LABEL
#ifdef NEW_LABEL
        parallelLabel->add(serialLabel, parallelSieve);
#else
        const Obj<typename bundle_type::label_type::traits::baseSequence>& base = serialLabel->base();

        for(typename bundle_type::label_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (parallelSieve->capContains(*b_iter) || parallelSieve->baseContains(*b_iter)) {
            parallelLabel->addArrow(*serialLabel->cone(*b_iter)->begin(), *b_iter);
          }
        }
#endif
        // Get remote labels
        sieveCompletion::scatterCones(serialLabel, parallelLabel, sendOverlap, recvOverlap);
      }

      // Distribute sections
      Obj<std::set<std::string> > sections = serialMesh->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        parallelMesh->setRealSection(*name, distributeSection(serialMesh->getRealSection(*name), parallelMesh, sendOverlap, recvOverlap));
      }
      sections = serialMesh->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        parallelMesh->setIntSection(*name, distributeSection(serialMesh->getIntSection(*name), parallelMesh, sendOverlap, recvOverlap));
      }
      sections = serialMesh->getArrowSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        parallelMesh->setArrowSection(*name, distributeArrowSection(serialMesh->getArrowSection(*name), serialMesh, parallelMesh, sendOverlap, recvOverlap));
      }
      if (parallelMesh->debug()) {parallelMesh->view("Parallel Mesh");}
      ALE_LOG_EVENT_END;
      return parallelMesh;
    }
    #undef __FUNCT__
    #define __FUNCT__ "updateSectionLocal"
    template<typename Section>
    static void updateSectionLocal(const Obj<Section>& oldSection, const Obj<bundle_type>& newBundle, const Obj<Section>& newSection) {
      const Obj<typename bundle_type::sieve_type>&    newSieve = newBundle->getSieve();
      const typename Section::atlas_type::chart_type& oldChart = oldSection->getChart();

      for(typename Section::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        if (newSieve->capContains(*c_iter) || newSieve->baseContains(*c_iter)) {
          newSection->setFiberDimension(*c_iter, oldSection->getFiberDimension(*c_iter));
        }
      }
      newBundle->allocate(newSection);
      const typename Section::atlas_type::chart_type& newChart = newSection->getChart();

      for(typename Section::atlas_type::chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        newSection->updatePointAll(*c_iter, oldSection->restrictPoint(*c_iter));
      }
    }
    #undef __FUNCT__
    #define __FUNCT__ "updateSectionRemote"
    template<typename RecvSection, typename Section>
    static void updateSectionRemote(const Obj<recv_overlap_type>& recvOverlap, const Obj<RecvSection>& recvSection, const Obj<bundle_type>& newBundle, const Obj<Section>& newSection) {
      Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

      for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
        const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

        for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
          newSection->addPoint(*r_iter, recvSection->getSection(*p_iter)->getFiberDimension(*r_iter));
        }
      }
      newBundle->reallocate(newSection);
      for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
        const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

        for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
          if (recvSection->getSection(*p_iter)->getFiberDimension(*r_iter)) {
            newSection->updatePointAll(*r_iter, recvSection->getSection(*p_iter)->restrictPoint(*r_iter));
          }
        }
      }
    }
    #undef __FUNCT__
    #define __FUNCT__ "distributeSection"
    template<typename Section>
    static Obj<Section> distributeSection(const Obj<Section>& serialSection, const Obj<bundle_type>& parallelBundle, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      if (serialSection->debug()) {
        serialSection->view("Serial Section");
      }
      typedef typename alloc_type::template rebind<typename Section::value_type>::other value_alloc_type;
      typedef ALE::Field<send_overlap_type, int, ALE::Section<point_type, typename Section::value_type, value_alloc_type> > send_section_type;
      typedef ALE::Field<recv_overlap_type, int, ALE::Section<point_type, typename Section::value_type, value_alloc_type> > recv_section_type;
      typedef ALE::New::SizeSection<Section> SectionSizer;
      Obj<Section>                 parallelSection = new Section(serialSection->comm(), serialSection->debug());
      const Obj<send_section_type> sendSection     = new send_section_type(serialSection->comm(), serialSection->debug());
      const Obj<recv_section_type> recvSection     = new recv_section_type(serialSection->comm(), sendSection->getTag(), serialSection->debug());
      const Obj<SectionSizer>      sizer           = new SectionSizer(serialSection);

      updateSectionLocal(serialSection, parallelBundle, parallelSection);
      sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, serialSection, sendSection, recvSection);
      updateSectionRemote(recvOverlap, recvSection, parallelBundle, parallelSection);
      if (parallelSection->debug()) {
        parallelSection->view("Parallel Section");
      }
      return parallelSection;
    }
    #undef __FUNCT__
    #define __FUNCT__ "updateArrowSectionLocal"
    template<typename Section>
    static void updateArrowSectionLocal(const Obj<Section>& oldSection, const Obj<bundle_type>& newBundle, const Obj<Section>& newSection) {
      const Obj<typename bundle_type::sieve_type>&    newSieve = newBundle->getSieve();
      const typename Section::atlas_type::chart_type& oldChart = oldSection->getChart();

      for(typename Section::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        // Dmitry should provide a Sieve::contains(MinimalArrow) method
        if (newSieve->capContains(c_iter->source) && newSieve->baseContains(c_iter->target)) {
          newSection->setFiberDimension(*c_iter, oldSection->getFiberDimension(*c_iter));
        }
      }
      //newBundle->allocate(newSection);
      const typename Section::atlas_type::chart_type& newChart = newSection->getChart();

      for(typename Section::atlas_type::chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        newSection->updatePointAll(*c_iter, oldSection->restrictPoint(*c_iter));
      }
    }
    #undef __FUNCT__
    #define __FUNCT__ "updateArrowSectionRemote"
    template<typename RecvSection, typename Section>
    static void updateArrowSectionRemote(const Obj<recv_overlap_type>& recvOverlap, const Obj<RecvSection>& recvSection, const Obj<bundle_type>& newBundle, const Obj<Section>& newSection) {
      Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

      for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
        const Obj<typename bundle_type::sieve_type::traits::coneSequence>&     cone = newBundle->getSieve()->cone(*r_iter);
        const typename bundle_type::sieve_type::traits::coneSequence::iterator end  = cone->end();

        for(typename bundle_type::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
          newSection->setFiberDimension(typename Section::point_type(*c_iter, *r_iter), 1);
        }
      }
      //newBundle->reallocate(newSection);
      for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
        const typename recv_overlap_type::traits::coneSequence::iterator recvEnd     = recvPatches->end();

        for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvEnd; ++p_iter) {
          const Obj<typename RecvSection::section_type>& section = recvSection->getSection(*p_iter);

          if (section->getFiberDimension(*r_iter)) {
            const Obj<typename bundle_type::sieve_type::traits::coneSequence>&     cone    = newBundle->getSieve()->cone(*r_iter);
            const typename bundle_type::sieve_type::traits::coneSequence::iterator end     = cone->end();
            const typename RecvSection::value_type                                *values  = section->restrictPoint(*r_iter);
            int                                                                    c       = -1;

            for(typename bundle_type::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
              newSection->updatePoint(typename Section::point_type(*c_iter, *r_iter), &values[++c]);
            }
          }
        }
      }
    }
    #undef __FUNCT__
    #define __FUNCT__ "distributeArrowSection"
    template<typename Section>
    static Obj<Section> distributeArrowSection(const Obj<Section>& serialSection, const Obj<bundle_type>& serialBundle, const Obj<bundle_type>& parallelBundle, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      if (serialSection->debug()) {
        serialSection->view("Serial ArrowSection");
      }
      typedef typename alloc_type::template rebind<typename Section::value_type>::other value_alloc_type;
      typedef ALE::Field<send_overlap_type, int, ALE::Section<point_type, typename Section::value_type, value_alloc_type> > send_section_type;
      typedef ALE::Field<recv_overlap_type, int, ALE::Section<point_type, typename Section::value_type, value_alloc_type> > recv_section_type;
      typedef ALE::New::ConeSizeSection<bundle_type, sieve_type> SectionSizer;
      typedef ALE::New::ArrowSection<sieve_type, Section>        ArrowFiller;
      Obj<Section>                 parallelSection = new Section(serialSection->comm(), serialSection->debug());
      const Obj<send_section_type> sendSection     = new send_section_type(serialSection->comm(), serialSection->debug());
      const Obj<recv_section_type> recvSection     = new recv_section_type(serialSection->comm(), sendSection->getTag(), serialSection->debug());
      const Obj<SectionSizer>      sizer           = new SectionSizer(serialBundle, serialBundle->getSieve());
      const Obj<ArrowFiller>       filler          = new ArrowFiller(serialBundle->getSieve(), serialSection);

      updateArrowSectionLocal(serialSection, parallelBundle, parallelSection);
      sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, filler, sendSection, recvSection);
      updateArrowSectionRemote(recvOverlap, recvSection, parallelBundle, parallelSection);
      if (parallelSection->debug()) {
        parallelSection->view("Parallel ArrowSection");
      }
      return parallelSection;
    }
    static void unifyBundle(const Obj<bundle_type>& bundle, const int dim, const Obj<bundle_type>& bundleNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      typedef int part_type;
      const Obj<sieve_type>& sieve    = bundle->getSieve();
      const Obj<sieve_type>& sieveNew = bundleNew->getSieve();
      const int              rank     = bundle->commRank();
      const int              debug    = bundle->debug();

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
      int        numCells   = bundle->heightStratum(0)->size();
      part_type *assignment = new part_type[numCells];

      for(int c = 0; c < numCells; ++c) {
        assignment[c] = 0;
      }
      // 3) Scatter the sieve
      sieveCompletion::scatterSieve(bundle, sieve, dim, sieveNew, sendOverlap, recvOverlap, 0, numCells, assignment);
      bundleNew->stratify();
      // 4) Cleanup
      if (assignment != NULL) delete [] assignment;
    }
    #undef __FUNCT__
    #define __FUNCT__ "unifyMesh"
    static Obj<ALE::Mesh<PetscInt,PetscScalar> > unifyMesh(const Obj<ALE::Mesh<PetscInt,PetscScalar> >& parallelMesh) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      const int                        dim         = parallelMesh->getDimension();
      Obj<FlexMesh>                    serialMesh  = new FlexMesh(parallelMesh->comm(), dim, parallelMesh->debug());
      const Obj<FlexMesh::sieve_type>& serialSieve = new FlexMesh::sieve_type(parallelMesh->comm(), parallelMesh->debug());

      ALE_LOG_EVENT_BEGIN;
      serialMesh->setSieve(serialSieve);
      if (parallelMesh->debug()) {
        parallelMesh->view("Parallel topology");
      }

      // Unify cones
      Obj<send_overlap_type> sendOverlap = new send_overlap_type(serialMesh->comm(), serialMesh->debug());
      Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(serialMesh->comm(), serialMesh->debug());
      unifyBundle(parallelMesh, dim, serialMesh, sendOverlap, recvOverlap);
      serialMesh->setDistSendOverlap(sendOverlap);
      serialMesh->setDistRecvOverlap(recvOverlap);

      // Unify labels
      const typename bundle_type::labels_type& labels = parallelMesh->getLabels();

      for(typename bundle_type::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (serialMesh->hasLabel(l_iter->first)) continue;
        const Obj<typename bundle_type::label_type>& parallelLabel = l_iter->second;
        const Obj<typename bundle_type::label_type>& serialLabel   = serialMesh->createLabel(l_iter->first);

        sieveCompletion::scatterCones(parallelLabel, serialLabel, sendOverlap, recvOverlap);
      }

      // Unify coordinates
      Obj<std::set<std::string> > sections = parallelMesh->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        serialMesh->setRealSection(*name, distributeSection(parallelMesh->getRealSection(*name), serialMesh, sendOverlap, recvOverlap));
      }
      sections = parallelMesh->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        serialMesh->setIntSection(*name, distributeSection(parallelMesh->getIntSection(*name), serialMesh, sendOverlap, recvOverlap));
      }
      sections = parallelMesh->getArrowSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        serialMesh->setArrowSection(*name, distributeArrowSection(parallelMesh->getArrowSection(*name), parallelMesh, serialMesh, sendOverlap, recvOverlap));
      }
      if (serialMesh->debug()) {serialMesh->view("Serial Mesh");}
      ALE_LOG_EVENT_END;
      return serialMesh;
    }
  public: // Do not like these
    #undef __FUNCT__
    #define __FUNCT__ "updateOverlap"
    // This is just crappy. We could introduce another phase to find out exactly what
    //   indices people do not have in the global order after communication
    template<typename OrigSendOverlap, typename OrigRecvOverlap, typename SendSection, typename RecvSection>
    static void updateOverlap(const Obj<OrigSendOverlap>& origSendOverlap, const Obj<OrigRecvOverlap>& origRecvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      const typename SendSection::sheaf_type& sendRanks = sendSection->getPatches();
      const typename RecvSection::sheaf_type& recvRanks = recvSection->getPatches();

      for(typename SendSection::sheaf_type::const_iterator p_iter = sendRanks.begin(); p_iter != sendRanks.end(); ++p_iter) {
        const typename SendSection::patch_type&               rank    = p_iter->first;
        const Obj<typename SendSection::section_type>&        section = p_iter->second;
        const typename SendSection::section_type::chart_type& chart   = section->getChart();

        for(typename SendSection::section_type::chart_type::const_iterator b_iter = chart.begin(); b_iter != chart.end(); ++b_iter) {
          const typename SendSection::value_type *points = section->restrictPoint(*b_iter);
          const int                               size   = section->getFiberDimension(*b_iter);

          for(int p = 0; p < size; p++) {
            if (origSendOverlap->support(points[p])->size() == 0) {
              sendOverlap->addArrow(points[p], rank, points[p]);
            }
          }
        }
      }
      for(typename RecvSection::sheaf_type::const_iterator p_iter = recvRanks.begin(); p_iter != recvRanks.end(); ++p_iter) {
        const typename RecvSection::patch_type&               rank    = p_iter->first;
        const Obj<typename RecvSection::section_type>&        section = p_iter->second;
        const typename RecvSection::section_type::chart_type& chart   = section->getChart();

        for(typename RecvSection::section_type::chart_type::const_iterator b_iter = chart.begin(); b_iter != chart.end(); ++b_iter) {
          const typename RecvSection::value_type *points = section->restrictPoint(*b_iter);
          const int                               size   = section->getFiberDimension(*b_iter);

          for(int p = 0; p < size; p++) {
            if (origRecvOverlap->support(rank, points[p])->size() == 0) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
      }
    }
    #undef __FUNCT__
    #define __FUNCT__ "updateSieve"
    template<typename RecvOverlap, typename RecvSection>
    static void updateSieve(const Obj<RecvOverlap>& recvOverlap, const Obj<RecvSection>& recvSection, const Obj<sieve_type>& sieve) {
#if 1
      Obj<typename RecvOverlap::traits::baseSequence> recvPoints = recvOverlap->base();

      for(typename RecvOverlap::traits::baseSequence::iterator p_iter = recvPoints->begin(); p_iter != recvPoints->end(); ++p_iter) {
        const Obj<typename RecvOverlap::traits::coneSequence>& ranks      = recvOverlap->cone(*p_iter);
        const typename RecvOverlap::target_type&               localPoint = *p_iter;

        for(typename RecvOverlap::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const typename RecvOverlap::target_type&       remotePoint = r_iter.color();
          const int                                      rank        = *r_iter;
          const Obj<typename RecvSection::section_type>& section     = recvSection->getSection(rank);
          const typename RecvSection::value_type        *points      = section->restrictPoint(remotePoint);
          const int                                      size        = section->getFiberDimension(remotePoint);
          int                                            c           = 0;

          ///std::cout << "["<<recvSection->commRank()<<"]: Receiving " << size << " points from rank " << rank << std::endl;
          for(int p = 0; p < size; p++) {
            // rank -- remote point --> local point
            if (recvOverlap->support(rank, points[p])->size()) {
              sieve->addArrow(*recvOverlap->support(rank, points[p])->begin(), localPoint, c);
              ///std::cout << "["<<recvSection->commRank()<<"]:   1Adding arrow " << *recvOverlap->support(rank, points[p])->begin() << "("<<points[p]<<") --> " << localPoint << std::endl;
            } else {
              sieve->addArrow(points[p], localPoint, c);
              ///std::cout << "["<<recvSection->commRank()<<"]:   2Adding arrow " << points[p] << " --> " << localPoint << std::endl;
            }
          }
        }
      }
#else
      const typename RecvSection::sheaf_type& ranks = recvSection->getPatches();

      for(typename RecvSection::sheaf_type::const_iterator p_iter = ranks.begin(); p_iter != ranks.end(); ++p_iter) {
        const Obj<typename RecvSection::section_type>&        section = p_iter->second;
        const typename RecvSection::section_type::chart_type& chart   = section->getChart();

        for(typename RecvSection::section_type::chart_type::const_iterator b_iter = chart.begin(); b_iter != chart.end(); ++b_iter) {
          const typename RecvSection::value_type *points = section->restrictPoint(*b_iter);
          int                                     size   = section->getFiberDimension(*b_iter);
          int                                     c      = 0;

          std::cout << "["<<recvSection->commRank()<<"]: Receiving " << size << " points from rank " << p_iter->first << std::endl;
          for(int p = 0; p < size; p++) {
            //sieve->addArrow(points[p], *b_iter, c++);
            sieve->addArrow(points[p], *b_iter, c);
            std::cout << "["<<recvSection->commRank()<<"]:   Adding arrow " << points[p] << " --> " << *b_iter << std::endl;
          }
        }
      }
#endif
    }
    #undef __FUNCT__
    #define __FUNCT__ "coneCompletion"
    template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
    static void coneCompletion(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<bundle_type>& bundle, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
      if (sendOverlap->commSize() == 1) return;
      // Distribute cones
      const Obj<sieve_type>&                                 sieve           = bundle->getSieve();
      const Obj<typename sieveCompletion::topology_type>     secTopology     = sieveCompletion::completion::createSendTopology(sendOverlap);
      const Obj<typename sieveCompletion::cone_size_section> coneSizeSection = new typename sieveCompletion::cone_size_section(bundle, sieve);
      const Obj<typename sieveCompletion::cone_section>      coneSection     = new typename sieveCompletion::cone_section(sieve);
      sieveCompletion::completion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
      // Update cones
      updateSieve(recvOverlap, recvSection, sieve);
    }
    #undef __FUNCT__
    #define __FUNCT__ "completeSection"
    template<typename Section>
    static void completeSection(const Obj<bundle_type>& bundle, const Obj<Section>& section) {
      typedef typename Distribution<bundle_type>::sieveCompletion sieveCompletion;
      typedef typename bundle_type::send_overlap_type             send_overlap_type;
      typedef typename bundle_type::recv_overlap_type             recv_overlap_type;
      typedef typename Section::value_type                        value_type;
      typedef typename alloc_type::template rebind<typename Section::value_type>::other value_alloc_type;
      typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > send_section_type;
      typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > recv_section_type;
      typedef ALE::New::SizeSection<Section>                                SectionSizer;
      const int debug = section->debug();

      bundle->constructOverlap();
      const Obj<send_overlap_type> sendOverlap = bundle->getSendOverlap();
      const Obj<recv_overlap_type> recvOverlap = bundle->getRecvOverlap();
      const Obj<send_section_type> sendSection = new send_section_type(section->comm(), section->debug());
      const Obj<recv_section_type> recvSection = new recv_section_type(section->comm(), sendSection->getTag(), section->debug());
      const Obj<SectionSizer>      sizer       = new SectionSizer(section);

      sectionCompletion::completeSection(sendOverlap, recvOverlap, sizer, section, sendSection, recvSection);
      // Update section with remote data
      const Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = bundle->getRecvOverlap()->base();

      for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>&     recvPatches = recvOverlap->cone(*r_iter);
        const typename recv_overlap_type::traits::coneSequence::iterator end         = recvPatches->end();

        for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != end; ++p_iter) {
          if (recvSection->getSection(*p_iter)->getFiberDimension(p_iter.color())) {
            if (debug) {std::cout << "["<<section->commRank()<<"]Completed point " << *r_iter << std::endl;}
            section->updateAddPoint(*r_iter, recvSection->getSection(*p_iter)->restrictPoint(p_iter.color()));
          }
        }
      }
    }
  };
}
#endif
