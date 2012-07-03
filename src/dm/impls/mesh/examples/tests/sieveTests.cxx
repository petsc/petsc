static char help[] = "Sieve Package Tests.\n\n";

#include <petscmesh.hh>

using ALE::Obj;

typedef struct {
  PetscInt   debug;
  PetscInt   rank;
  PetscInt   size;
  // Classes
  PetscBool  overlap;       // Run the Overlap tests
  PetscBool  preallocation; // Run the Preallocation tests
  PetscBool  label;         // Run the Label tests
  // Mesh flags
  PetscBool  interpolate;   // Interpolate the mesh
  PetscReal  refine;        // The refinement limit
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  options->debug         = 0;
  options->overlap       = PETSC_FALSE;
  options->preallocation = PETSC_FALSE;
  options->label         = PETSC_FALSE;
  options->interpolate   = PETSC_FALSE;
  options->refine        = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "Debugging flag", "sieveTests", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-overlap", "Run Overlap tests", "sieveTests", options->overlap, &options->overlap, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-preallocation", "Run Preallocation tests", "sieveTests", options->preallocation, &options->preallocation, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-label", "Run Label tests", "sieveTests", options->label, &options->label, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "sieveTests", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refine", "The refinement limit", "sieveTests", options->refine, &options->refine, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &options->size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSquareMesh"
template<typename MeshT>
PetscErrorCode CreateSquareMesh(Obj<MeshT>& mesh, const Options *options)
{
  double lower[2] = {0.0, 0.0};
  double upper[2] = {1.0, 1.0};
  int    edges[2] = {2, 2};
  Obj<MeshT> imB;
  Obj<typename MeshT::sieve_type> isieve;

  PetscFunctionBegin;
  const Obj<ALE::Mesh> mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  imB    = new MeshT(mB->comm(), mB->getDimension(), mB->debug());
  isieve = new typename MeshT::sieve_type(mB->comm(), mB->debug());
  imB->setSieve(isieve);
  ALE::ISieveConverter::convertMesh(*mB, *imB, imB->getRenumbering(), false);
  mesh = ALE::Generator<MeshT>::generateMeshV(imB, options->interpolate);
  if (options->refine > 0.0) {
    mesh = ALE::Generator<MeshT>::refineMeshV(mesh, options->refine, options->interpolate);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
template<typename MeshT>
PetscErrorCode DistributeMesh(Obj<MeshT>& mesh, Obj<MeshT>& newMesh, const Options *options)
{
  typedef typename ALE::DistributionNew<MeshT>        DistributionType;
  typedef typename DistributionType::partitioner_type PartitionerType;
  typedef typename MeshT::point_type                  point_type;
  typedef typename DistributionType::partition_type   partition_type;

  PetscFunctionBegin;
  const Obj<typename MeshT::sieve_type>        newSieve        = new typename MeshT::sieve_type(mesh->comm(), mesh->debug());
  const Obj<typename MeshT::send_overlap_type> sendMeshOverlap = new typename MeshT::send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<typename MeshT::recv_overlap_type> recvMeshOverlap = new typename MeshT::recv_overlap_type(mesh->comm(), mesh->debug());

  newMesh = new MeshT(mesh->comm(), mesh->getDimension(), mesh->debug());
  newMesh->setSieve(newSieve);
  typename MeshT::renumbering_type&            renumbering     = newMesh->getRenumbering();
  // Distribute the mesh
  Obj<partition_type> partition = DistributionType::distributeMeshV(mesh, newMesh, renumbering, sendMeshOverlap, recvMeshOverlap);
#if 0
  if (mesh->debug()) {
    std::cout << "["<<mesh->commRank()<<"]: Mesh Renumbering:" << std::endl;
    for(typename MeshT::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
      std::cout << "["<<mesh->commRank()<<"]:   global point " << r_iter->first << " --> " << " local point " << r_iter->second << std::endl;
    }
  }
  // Check overlap
  int localSendOverlapSize = 0, sendOverlapSize;
  int localRecvOverlapSize = 0, recvOverlapSize;
  for(int p = 0; p < sendMeshOverlap->commSize(); ++p) {
    localSendOverlapSize += sendMeshOverlap->cone(p)->size();
    localRecvOverlapSize += recvMeshOverlap->support(p)->size();
  }
  MPI_Allreduce(&localSendOverlapSize, &sendOverlapSize, 1, MPI_INT, MPI_SUM, sendMeshOverlap->comm());
  MPI_Allreduce(&localRecvOverlapSize, &recvOverlapSize, 1, MPI_INT, MPI_SUM, recvMeshOverlap->comm());
  if(sendOverlapSize != recvOverlapSize) {
    std::cout <<"["<<sendMeshOverlap->commRank()<<"]: Size mismatch " << sendOverlapSize << " != " << recvOverlapSize << std::endl;
    sendMeshOverlap->view("Send Overlap");
    recvMeshOverlap->view("Recv Overlap");
    throw ALE::Exception("Invalid Overlap");
  }
#endif

  // Distribute the coordinates
  const Obj<typename MeshT::real_section_type>& coordinates         = mesh->getRealSection("coordinates");
  const Obj<typename MeshT::real_section_type>& parallelCoordinates = newMesh->getRealSection("coordinates");

  newMesh->setupCoordinates(parallelCoordinates);
  DistributionType::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
#if 0
  // Distribute other sections
  if (mesh->getRealSections()->size() > 1) {
    Obj<std::set<std::string> > names = mesh->getRealSections();
    int                         n     = 0;

    for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
      if (*n_iter == "coordinates")   continue;
      if (*n_iter == "replaced_cells") continue;
      std::cout << "ERROR: Did not distribute real section " << *n_iter << std::endl;
      ++n;
    }
    if (n) {throw ALE::Exception("Need to distribute more real sections");}
  }
  if (mesh->getIntSections()->size() > 0) {
    Obj<std::set<std::string> > names = mesh->getIntSections();

    for(std::set<std::string>::const_iterator n_iter = names->begin(); n_iter != names->end(); ++n_iter) {
      const Obj<MeshT::int_section_type>& origSection = mesh->getIntSection(*n_iter);
      const Obj<MeshT::int_section_type>& newSection  = (*newMesh)->getIntSection(*n_iter);

      // We assume all integer sections are complete sections
      newSection->setChart((*newMesh)->getSieve()->getChart());
      DistributionType::distributeSection(origSection, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newSection);
    }
  }
  if (mesh->getArrowSections()->size() > 1) {
    throw ALE::Exception("Need to distribute more arrow sections");
  }
#endif
  // Distribute labels
  const typename MeshT::labels_type& labels = mesh->getLabels();

  for(typename MeshT::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
    if (newMesh->hasLabel(l_iter->first)) continue;
    const Obj<typename MeshT::label_type>& origLabel = l_iter->second;
    const Obj<typename MeshT::label_type>& newLabel  = newMesh->createLabel(l_iter->first);

    newLabel->setChart(newMesh->getSieve()->getChart());
    // Size the local mesh
    PartitionerType::sizeLocalSieveV(origLabel, partition, renumbering, newLabel);
    // Create the remote meshes
    DistributionType::completeConesV(origLabel, newLabel, sendMeshOverlap, recvMeshOverlap);
    // Create the local mesh
    PartitionerType::createLocalLabelV(origLabel, partition, renumbering, newLabel);
    newLabel->symmetrize();
  }
  // Create the parallel overlap
  Obj<typename MeshT::send_overlap_type> sendParallelMeshOverlap = newMesh->getSendOverlap();
  Obj<typename MeshT::recv_overlap_type> recvParallelMeshOverlap = newMesh->getRecvOverlap();
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  newMesh->setCalculatedOverlap(true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputOverlap"
template<typename MeshT>
PetscErrorCode OutputOverlap(Obj<MeshT>& mesh, const Options *options)
{
  const Obj<typename MeshT::real_section_type> coordinates = mesh->getRealSection("coordinates");
  PetscViewer    viewer;
  ostringstream  sendName;
  ostringstream  recvName;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sendName << "sendOverlap_" << options->rank << "_" << options->size << ".py";
  ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, sendName.str().c_str(), &viewer);CHKERRQ(ierr);
  const Obj<typename MeshT::send_overlap_type::traits::capSequence>      sPoints = mesh->getSendOverlap()->cap();
  const typename MeshT::send_overlap_type::traits::capSequence::iterator sBegin  = sPoints->begin();
  const typename MeshT::send_overlap_type::traits::capSequence::iterator sEnd    = sPoints->end();

  ierr = PetscViewerASCIIPrintf(viewer, "sendOverlap = {\n");CHKERRQ(ierr);
  for(typename MeshT::send_overlap_type::traits::capSequence::iterator p_iter = sBegin; p_iter != sEnd; ++p_iter) {
    const typename MeshT::point_type&                                  localPoint = *p_iter;
    const Obj<typename MeshT::send_overlap_type::supportSequence>&     targets    = mesh->getSendOverlap()->support(localPoint);
    const typename MeshT::send_overlap_type::supportSequence::iterator tBegin     = targets->begin();
    const typename MeshT::send_overlap_type::supportSequence::iterator tEnd       = targets->end();

    if (p_iter != sBegin) {ierr = PetscViewerASCIIPrintf(viewer, ",\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "  %d: [", localPoint);CHKERRQ(ierr);
    for(typename MeshT::send_overlap_type::supportSequence::iterator r_iter = tBegin; r_iter != tEnd; ++r_iter) {
      const int                            rank        = *r_iter;
      const typename MeshT::point_type& remotePoint = r_iter.color();

      if (r_iter != tBegin) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "(%d, %d)", rank, remotePoint);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "]");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n}\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "coordinates = {\n");CHKERRQ(ierr);
  for(typename MeshT::send_overlap_type::traits::capSequence::iterator p_iter = sBegin; p_iter != sEnd; ++p_iter) {
    const typename MeshT::point_type&                    localPoint = *p_iter;
    const int                                            dim        = coordinates->getFiberDimension(localPoint);
    const typename MeshT::real_section_type::value_type *coords     = coordinates->restrictPoint(localPoint);

    if (p_iter != sBegin) {ierr = PetscViewerASCIIPrintf(viewer, ",\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "  %d: [", localPoint);CHKERRQ(ierr);
    for(int d = 0; d < dim; ++d) {
      if (d > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "%g", coords[d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "]");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n}\n");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  recvName << "recvOverlap_" << options->rank << "_" << options->size << ".py";
  ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, recvName.str().c_str(), &viewer);CHKERRQ(ierr);
  const Obj<typename MeshT::recv_overlap_type::traits::baseSequence>      rPoints = mesh->getRecvOverlap()->base();
  const typename MeshT::recv_overlap_type::traits::baseSequence::iterator rBegin  = rPoints->begin();
  const typename MeshT::recv_overlap_type::traits::baseSequence::iterator rEnd    = rPoints->end();

  ierr = PetscViewerASCIIPrintf(viewer, "recvOverlap = {\n");CHKERRQ(ierr);
  for(typename MeshT::recv_overlap_type::traits::baseSequence::iterator p_iter = rBegin; p_iter != rEnd; ++p_iter) {
    const typename MeshT::point_type&                               localPoint = *p_iter;
    const Obj<typename MeshT::recv_overlap_type::coneSequence>&     sources    = mesh->getRecvOverlap()->cone(localPoint);
    const typename MeshT::recv_overlap_type::coneSequence::iterator sBegin     = sources->begin();
    const typename MeshT::recv_overlap_type::coneSequence::iterator sEnd       = sources->end();

    if (p_iter != rBegin) {ierr = PetscViewerASCIIPrintf(viewer, ",\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "  %d: [", localPoint);CHKERRQ(ierr);
    for(typename MeshT::recv_overlap_type::coneSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
      const int                            rank        = *r_iter;
      const typename MeshT::point_type& remotePoint = r_iter.color();

      if (r_iter != sBegin) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "(%d, %d)", rank, remotePoint);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "]");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n}\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "coordinates = {\n");CHKERRQ(ierr);
  for(typename MeshT::recv_overlap_type::traits::baseSequence::iterator p_iter = rBegin; p_iter != rEnd; ++p_iter) {
    const typename MeshT::point_type&                    localPoint = *p_iter;
    const int                                            dim        = coordinates->getFiberDimension(localPoint);
    const typename MeshT::real_section_type::value_type *coords     = coordinates->restrictPoint(localPoint);

    if (p_iter != rBegin) {ierr = PetscViewerASCIIPrintf(viewer, ",\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "  %d: [", localPoint);CHKERRQ(ierr);
    for(int d = 0; d < dim; ++d) {
      if (d > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "%g", coords[d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "]");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n}\n");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckPreallocation"
template<typename MeshT>
PetscErrorCode CheckPreallocation(Obj<MeshT>& mesh, Mat A, const Options *options)
{
  typedef typename MeshT::real_section_type                                                           real_section_type;
  typedef ALE::ISieveVisitor::IndicesVisitor<real_section_type, typename MeshT::order_type, PetscInt> visitor_type;
  const Obj<typename MeshT::sieve_type>&         sieve       = mesh->getSieve();
  const Obj<real_section_type>&                  coordinates = mesh->getRealSection("coordinates");
  const Obj<typename MeshT::order_type>&         globalOrder = mesh->getFactory()->getGlobalOrder(mesh, "default", coordinates);
  const Obj<typename MeshT::label_sequence>&     cells       = mesh->heightStratum(0);
  const typename MeshT::label_sequence::iterator cEnd        = cells->end();
  const PetscInt                                 dim         = mesh->getDimension();
  PetscScalar                                   *elemMatrix  = new PetscScalar[PetscSqr((dim+1)*dim)];
  PetscErrorCode                                 ierr;
  visitor_type iV(*coordinates, *globalOrder, (int) pow(sieve->getMaxConeSize(), mesh->depth())*dim);

  // The easiest way to check preallocation is to fill up the entire matrix
  PetscFunctionBegin;
  for(PetscInt i = 0; i < PetscSqr((dim+1)*dim); ++i) {elemMatrix[i] = 1.0;}
  for(typename MeshT::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
    ierr = updateOperator(A, *sieve, iV, *c_iter, elemMatrix, ADD_VALUES);CHKERRQ(ierr);
    iV.clear();
  }
  delete [] elemMatrix;
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OverlapTests"
PetscErrorCode OverlapTests(const Options *options)
{
  typedef ALE::IMesh<> mesh_type;
  Obj<mesh_type> mesh;
  Obj<mesh_type> newMesh;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = CreateSquareMesh(mesh, options);CHKERRQ(ierr);
  ierr = DistributeMesh(mesh, newMesh, options);CHKERRQ(ierr);
  ierr = OutputOverlap(newMesh, options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PreallocationTests"
PetscErrorCode PreallocationTests(const Options *options)
{
  typedef ALE::IMesh<> mesh_type;
  Obj<mesh_type> mesh;
  Obj<mesh_type> newMesh;
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateSquareMesh(mesh, options);CHKERRQ(ierr);
  ierr = DistributeMesh(mesh, newMesh, options);CHKERRQ(ierr);
  const Obj<mesh_type::real_section_type>& coordinates = newMesh->getRealSection("coordinates");
  const Obj<mesh_type::order_type>&        globalOrder = newMesh->getFactory()->getGlobalOrder(newMesh, "default", coordinates);
  const PetscInt n   = globalOrder->getLocalSize();
  const PetscInt N   = globalOrder->getGlobalSize();
  const PetscInt fR  = globalOrder->getGlobalOffsets()[mesh->commRank()];
  const PetscInt bs  = newMesh->getDimension();
  PetscInt      *dnz = new PetscInt[n/bs];
  PetscInt      *onz = new PetscInt[n/bs];

  ierr = MatCreate(newMesh->comm(), &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(A, MATBAIJ);CHKERRQ(ierr);
  ierr = preallocateOperatorNew(newMesh, bs, coordinates->getAtlas(), globalOrder, dnz, onz, A);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_UNUSED_NONZERO_LOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  if (options->debug) {
    for(PetscInt r = 0; r < n/bs; ++r) {
      for(PetscInt b = 0; b < bs; ++b) {
        ierr = PetscSynchronizedPrintf(mesh->comm(), "dnz[%d]: %d  onz[%d]: %d\n", r*bs+b+fR, dnz[r], r*bs+b+fR, onz[r]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSynchronizedFlush(mesh->comm());CHKERRQ(ierr);
  ierr = CheckPreallocation(newMesh, A, options);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  delete [] dnz;
  delete [] onz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LabelTests"
PetscErrorCode LabelTests(const Options *options)
{
  typedef ALE::IMesh<> mesh_type;
  Obj<mesh_type> mesh;
  Obj<mesh_type> newMesh;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = CreateSquareMesh(mesh, options);CHKERRQ(ierr);
  const Obj<mesh_type::label_type>&        testLabel   = mesh->createLabel("test");
  const Obj<mesh_type::real_section_type>& coordinates = mesh->getRealSection("coordinates");
  const Obj<mesh_type::label_sequence>&    vertices    = mesh->depthStratum(0);
  int                                      base        = vertices->size();

  ierr = MPI_Bcast(&base, 1, MPI_INT, 0, mesh->comm());CHKERRQ(ierr);
  testLabel->setChart(mesh->getSieve()->getChart());
  for(mesh_type::label_sequence::const_iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    testLabel->setConeSize(*v_iter, 1);
  }
  if (vertices->size()) {testLabel->setSupportSize(0, vertices->size());}
  testLabel->allocate();
  for(mesh_type::label_sequence::const_iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const mesh_type::real_section_type::value_type *coords = coordinates->restrictPoint(*v_iter);
    double                                          value  = 0.0;
    int                                             label;

    for(int d = 0; d < mesh->getDimension(); ++d) {
      value += coords[d]*pow(10, d);
    }
    label = ((int) value)%base;
    mesh->setValue(testLabel, *v_iter, label);
  }
  testLabel->recalculateLabel();
  testLabel->view("Test Label");
  ierr = DistributeMesh(mesh, newMesh, options);CHKERRQ(ierr);
  const Obj<mesh_type::label_type>&        newTestLabel   = newMesh->getLabel("test");
  const Obj<mesh_type::real_section_type>& newCoordinates = newMesh->getRealSection("coordinates");
  const Obj<mesh_type::label_sequence>&    newVertices    = newMesh->depthStratum(0);

  newTestLabel->view("New Test Label");
  for(mesh_type::label_sequence::const_iterator v_iter = newVertices->begin(); v_iter != newVertices->end(); ++v_iter) {
    const mesh_type::real_section_type::value_type *coords = newCoordinates->restrictPoint(*v_iter);
    double                                          value  = 0.0;
    int                                             label;

    for(int d = 0; d < newMesh->getDimension(); ++d) {
      value += coords[d]*pow(10, d);
    }
    label = ((int) value)%base;
    if (label != newMesh->getValue(newTestLabel, *v_iter)) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "SieveTests: Invalid label value for vertex %d: %d should be %d", *v_iter, newMesh->getValue(newTestLabel, *v_iter), label);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->overlap)       {ierr = OverlapTests(options);CHKERRQ(ierr);}
  if (options->preallocation) {ierr = PreallocationTests(options);CHKERRQ(ierr);} 
  if (options->label)         {ierr = LabelTests(options);CHKERRQ(ierr);} 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  try {
    ierr = RunUnitTests(&options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, e.message());
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
