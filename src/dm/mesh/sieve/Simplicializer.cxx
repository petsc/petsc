#include <petscmesh.h>

int debug = 0;

#if 0

#undef __FUNCT__
#define __FUNCT__ "ComputePreSievePartition"
PetscErrorCode ComputePreSievePartition(ALE::PreSieve* presieve, ALE::Point_set leaves, const char *name = NULL)
{
  MPI_Comm       comm = presieve->getComm();
  PetscInt       numLeaves = leaves.size();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (rank == 0) {
    for(int p = 0; p < size; p++) {
      ALE::Point partPoint(-1, p);
      for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
        ALE::Point leaf(0, l);
        ALE::Point_set cone = presieve->cone(leaf);
        presieve->addCone(cone, partPoint);
      }
    }
  } else {
    ALE::Point partitionPoint(-1, rank);
    presieve->addBasePoint(partitionPoint);
  }
  if (debug) {
    ostringstream label1;
    label1 << "Partition of presieve ";
    if(name != NULL) {
      label1 << "'" << name << "'";
    }
    label1 << "\n";
    presieve->view(label1.str().c_str());
  }
  ALE_LOG_EVENT_END
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSievePartition"
PetscErrorCode ComputeSievePartition(ALE::Sieve* sieve, const char *name = NULL)
{
  MPI_Comm       comm = sieve->getComm();
  PetscInt       numLeaves = sieve->leaves().size();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_STAGE_BEGIN;
  ALE_LOG_EVENT_BEGIN
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (rank == 0) {
    for(int p = 0; p < size; p++) {
      ALE::Point partPoint(-1, p);
      for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
        sieve->addCone(sieve->closure(ALE::Point(0, l)), partPoint);
      }
    }
  } else {
    ALE::Point partitionPoint(-1, rank);
    sieve->addBasePoint(partitionPoint);
  }
  if (debug) {
    ostringstream label1;
    label1 << "Partition of sieve ";
    if(name != NULL) {
      label1 << "'" << name << "'";
    }
    label1 << "\n";
    sieve->view(label1.str().c_str());
  }
  ALE_LOG_EVENT_END
  ALE_LOG_STAGE_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionPreSieve"
PetscErrorCode PartitionPreSieve(ALE::PreSieve* presieve, const char *name = NULL, bool localize = 1, ALE::Obj<ALE::PreSieve> *pointTypes = NULL)
{
  MPI_Comm       comm = presieve->getComm();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  // Cone complete to move the partitions to the other processors
  ALE::Obj<ALE::Stack> completionStack = presieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone, NULL);
  ALE::Obj<ALE::PreSieve> completion = completionStack->top();
  if (debug) {
    ostringstream label1;
    label1 << "Completion";
    if(name != NULL) {
      label1 << " of '" << name << "'";
    }
    completion->view(label1.str().c_str());
  }
  // Create point type presieve
  if (pointTypes != NULL) {
    // Point types are as follows:
    //
    //   (rank, 0): Local point,  not shared with other processes
    //   (rank, 1): Leased point, owned but shared with other processes
    //     (otherRank, otherRank):    point leased to process otherRank
    //   (rank, 2): Rented point, not owned and shared with other processes
    //     (-otherRank-1, otherRank): point rented from process otherRank
    ALE::Obj<ALE::PreSieve> pTypes = ALE::PreSieve(comm);

    for(int p = 0; p < size; p++) {
      ALE::Point partitionPoint(-1, p);
      ALE::Point_set cone;

      if (presieve->baseContains(partitionPoint)) {
        cone = presieve->cone(partitionPoint);
      }

      if (p == rank) {
        ALE::Point point(rank, ALE::localPoint);
        pTypes->addCone(cone, point);

        if (completion->baseContains(partitionPoint)) {
          cone = completion->cone(partitionPoint);
        } else {
          cone.clear();
        }
        point = ALE::Point(rank, ALE::rentedPoint);
        pTypes->addCone(cone, point);
        for(ALE::Point_set::iterator e_itor = cone.begin(); e_itor != cone.end(); e_itor++) {
          ALE::Point e = *e_itor;
          ALE::Point f = *completionStack->support(e)->begin();

          point = ALE::Point(-f.index-1, f.index);
          pTypes->addCone(e, point);
        }
      } else {
        ALE::Point point;

        point = ALE::Point(rank, ALE::leasedPoint);
        pTypes->addCone(cone, point);
        point = ALE::Point(p, p);
        pTypes->addCone(cone, point);
      }
    }
    *pointTypes = pTypes;
    if (debug) {
      pTypes->view("Partition pointTypes");
    }
  }
  // Merge in the completion
  presieve->add(completion);
  // Move the cap to the base of the partition sieve
  ALE::Point partitionPoint(-1, rank);
  ALE::Point_set partition = presieve->cone(partitionPoint);
  for(ALE::Point_set::iterator p_itor = partition.begin(); p_itor != partition.end(); p_itor++) {
    ALE::Point p = *p_itor;
    presieve->addBasePoint(p);
  }
  if (debug) {
    ostringstream label2;
    if(name != NULL) {
      label2 << "Initial parallel state of '" << name << "'";
    } else {
      label2 << "Initial parallel presieve";
    }
    presieve->view(label2.str().c_str());
  }
  // Cone complete again to build the local topology
  completion = presieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone, NULL)->top();
  if (debug) {
    ostringstream label3;
    if(name != NULL) {
      label3 << "Completion of '" << name << "'";
    } else {
      label3 << "Completion";
    }
    completion->view(label3.str().c_str());
  }
  presieve->add(completion);
  if (debug) {
    ostringstream label4;
    if(name != NULL) {
      label4 << "Completed parallel version of '" << name << "'";
    } else {
      label4 << "Completed parallel presieve";
    }
    presieve->view(label4.str().c_str());
  }
  // Unless explicitly prohibited, restrict to the local partition
  if(localize) {
    presieve->restrictBase(partition);
    if (debug) {
      ostringstream label5;
      if(name != NULL) {
        label5 << "Localized parallel version of '" << name << "'";
      } else {
        label5 << "Localized parallel presieve";
      }
      presieve->view(label5.str().c_str());
    }
  }
  // Support complete to get the adjacency information
  ALE_LOG_EVENT_END
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval"
/* This is currently duplicated in ex33mesh.c */
inline void ExpandInterval(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.index; i++) {
    indices[(*indx)++] = interval.prefix + i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
/* This is currently duplicated in ex33mesh.c */
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::Point_array> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    ExpandInterval(*i_itor, indices, &k);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshComputeGlobalScatter"
PetscErrorCode MeshComputeGlobalScatter(ALE::Obj<ALE::IndexBundle> bundle, VecScatter *injection)
{
  VecScatter               sc;
  ALE::Obj<ALE::PreSieve>  globalIndices;
  ALE::Obj<ALE::PreSieve>  localIndices;
  ALE::Obj<ALE::Point_set> points;
  Vec                      l, g;
  IS                       localIS, globalIS;
  PetscInt                *localIdx, *globalIdx;
  PetscInt                 localSize, remoteSize, lcntr = 0, gcntr = 0;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  localSize = bundle->getLocalSize();
  remoteSize = bundle->getRemoteSize();
  ierr = VecCreateSeq(PETSC_COMM_SELF, localSize+remoteSize, &l);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD, localSize, PETSC_DETERMINE, &g);CHKERRQ(ierr);
  ierr = PetscMalloc((localSize+remoteSize) * sizeof(PetscInt), &localIdx);CHKERRQ(ierr);
  ierr = PetscMalloc((localSize+remoteSize) * sizeof(PetscInt), &globalIdx);CHKERRQ(ierr);
  localIndices  = bundle->getLocalIndices();
  globalIndices = bundle->getGlobalIndices();
  points = globalIndices->base();
  for(ALE::Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
    ALE::Point p = *p_itor;
    ALE::Point_set lCone = localIndices->cone(p);
    ALE::Point_set gCone = globalIndices->cone(p);

    if (lCone.size()) {
      ExpandInterval(*(lCone.begin()), localIdx, &lcntr);
    }
    if (gCone.size()) {
      ExpandInterval(*(gCone.begin()), globalIdx, &gcntr);
    }
    if (lcntr != gcntr) {
      SETERRQ2(PETSC_ERR_PLIB, "Inconsistent numbering, %d != %d", lcntr, gcntr);
    }
  }
  if (lcntr != localSize+remoteSize) {
    SETERRQ2(PETSC_ERR_PLIB, "Inconsistent numbering, %d should be %d", lcntr, localSize+remoteSize);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize+remoteSize, localIdx, &localIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize+remoteSize, globalIdx, &globalIS);CHKERRQ(ierr);
  ierr = PetscFree(localIdx);CHKERRQ(ierr);
  ierr = PetscFree(globalIdx);CHKERRQ(ierr);
  ierr = VecScatterCreate(l, localIS, g, globalIS, &sc);CHKERRQ(ierr);
  ierr = ISDestroy(localIS);CHKERRQ(ierr);
  ierr = ISDestroy(globalIS);CHKERRQ(ierr);
  *injection = sc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setFiberValues"
PetscErrorCode setFiberValues(Vec b, ALE::Point e, ALE::IndexBundle* bundle, PetscScalar array[], InsertMode mode)
{
  ALE_LOG_EVENT_BEGIN
  ALE::Point_set   ee(e), empty;
  ALE::Obj<ALE::Point_set> intervals = bundle->getFiberIndices(ee, empty)->cap();
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0, i = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_set::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  if (debug) {
    for(ALE::Point_set::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
    }
  }
  ExpandInterval(*intervals->begin(), indices, &i);
  if (debug) {
    for(int i = 0; i < numIndices; i++) {
      printf("indices[%d] = %d\n", i, indices[i]);
    }
  }
  ierr = VecSetValues(b, numIndices, indices, array, mode);CHKERRQ(ierr);
  ALE_LOG_EVENT_END
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setClosureValues"
PetscErrorCode setClosureValues(Vec b, ALE::Point e, ALE::IndexBundle* bundle, ALE::PreSieve* orientation, PetscScalar array[], InsertMode mode)
{
  ALE::Point_set   empty;
  ALE::Obj<ALE::Point_array> intervals = bundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> intervals = bundle->getOverlapOrderedIndices(orientation->cone(e), empty);
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
  }
  ierr = VecSetValues(b, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "restrictField"
PetscErrorCode restrictField(ALE::Obj<ALE::IndexBundle> bundle, ALE::Obj<ALE::PreSieve> orientation, PetscScalar *array, ALE::Point e, PetscScalar *values[])
{
  ALE::Obj<ALE::Point_array> intervals = bundle->getLocalOrderedClosureIndices(orientation->cone(e));
  /* This should be done by memory pooling by array size (we have a simple form below) */
  static PetscScalar *vals;
  static PetscInt     numValues = 0;
  static PetscInt    *indices = NULL;
  PetscInt            numIndices = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (numValues && (numValues != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
    ierr = PetscFree(vals); CHKERRQ(ierr);
    vals = NULL;
  }
  if (!indices) {
    numValues = numIndices;
    ierr = PetscMalloc(numValues * sizeof(PetscInt), &indices); CHKERRQ(ierr);
    ierr = PetscMalloc(numValues * sizeof(PetscScalar), &vals); CHKERRQ(ierr);
  }
  if (debug) {
    for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      printf("[%d]interval (%d, %d)\n", bundle->getCommRank(), (*i_itor).prefix, (*i_itor).index);
    }
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    if (debug) {printf("[%d]indices[%d] = %d  val: %g\n", bundle->getCommRank(), i, indices[i], array[indices[i]]);}
    vals[i] = array[indices[i]];
  }
  *values = vals;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleField"
PetscErrorCode assembleField(ALE::Obj<ALE::IndexBundle> bundle, ALE::Obj<ALE::PreSieve> orientation, Vec b, ALE::Point e, PetscScalar array[], InsertMode mode)
{
  ALE::Obj<ALE::Point_array> intervals = bundle->getGlobalOrderedClosureIndices(orientation->cone(e));
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  if (debug) {
    for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      printf("[%d]Element (%d, %d) interval (%d, %d)\n", bundle->getCommRank(), e.prefix, e.index, (*i_itor).prefix, (*i_itor).index);
    }
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  if (debug) {
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]indices[%d] = %d with value %g\n", bundle->getCommRank(), i, indices[i], array[i]);
    }
  }
  ierr = VecSetValues(b, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleOperator"
PetscErrorCode assembleOperator(ALE::Obj<ALE::IndexBundle> bundle, ALE::Obj<ALE::PreSieve> orientation, Mat A, ALE::Point e, PetscScalar array[], InsertMode mode)
{
  ALE::Obj<ALE::Point_array> intervals = bundle->getGlobalOrderedClosureIndices(orientation->cone(e));
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  if (debug) {
    for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      printf("[%d]interval (%d, %d)\n", bundle->getCommRank(), (*i_itor).prefix, (*i_itor).index);
    }
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  if (debug) {
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]indices[%d] = %d\n", bundle->getCommRank(), i, indices[i]);
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MeshCreateVector(Mesh, ALE::IndexBundle *, int, Vec *);
extern PetscErrorCode MeshCreateMapping(Mesh, ALE::Obj<ALE::IndexBundle>, ALE::Obj<ALE::PreSieve>, ALE::Obj<ALE::IndexBundle>, VecScatter *);

#undef __FUNCT__
#define __FUNCT__ "createParallelCoordinates"
PetscErrorCode createParallelCoordinates(Mesh mesh, int dim, ALE::Obj<ALE::PreSieve> partitionTypes)
{
  ALE::Obj<ALE::Sieve>       topology;
  ALE::Obj<ALE::IndexBundle> coordBundle;
  ALE::Obj<ALE::IndexBundle> serialCoordBundle;
  Vec                        coordinates, oldCoordinates, locCoordinates;
  VecScatter                 coordScatter;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, &serialCoordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &oldCoordinates);CHKERRQ(ierr);
  /* Create coordinate bundle */
  coordBundle = ALE::IndexBundle(topology);
  if (debug) {
    coordBundle->setVerbosity(11);
  }
  coordBundle->setFiberDimensionByDepth(0, dim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  coordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshCreateVector(mesh, coordBundle, debug, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(coordinates, &locCoordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(locCoordinates, dim);CHKERRQ(ierr);
  ierr = MeshSetCoordinateBundle(mesh, coordBundle);CHKERRQ(ierr);
  /* Setup mapping to partitioned storage */
  ierr = MeshCreateMapping(mesh, serialCoordBundle, partitionTypes, coordBundle, &coordScatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  ierr = VecScatterDestroy(coordScatter);CHKERRQ(ierr);
  ierr = VecDestroy(oldCoordinates);CHKERRQ(ierr);
  if (debug) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Coordinates\n===========================\n");CHKERRQ(ierr);
    ierr = VecView(coordinates, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Communicate ghosted coordinates */
  ierr = VecGhostUpdateBegin(coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ALE_LOG_EVENT_END
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateMapping"
PetscErrorCode MeshCreateMapping(Mesh mesh, ALE::Obj<ALE::IndexBundle> sourceBundle, ALE::Obj<ALE::PreSieve> pointTypes, ALE::Obj<ALE::IndexBundle> targetBundle, VecScatter *scatter)
{
  ALE::Obj<ALE::Stack>     mappingStack;
  ALE::Obj<ALE::PreSieve>  sourceIndices, targetIndices;
  ALE::Obj<ALE::Point_set> base;
  Vec                      sourceVec, targetVec;
  IS                       fromIS, toIS;
  PetscInt                *fromIndices, *toIndices;
  PetscInt                 fromIdx = 0, toIdx = 0;
  PetscInt                 locSourceSize = sourceBundle->getLocalSize();
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  mappingStack  = sourceBundle->computeMappingIndices(pointTypes, targetBundle);
  sourceIndices = mappingStack->top();
  targetIndices = mappingStack->bottom();
  base = sourceIndices->base();
#if 1
  locSourceSize = 0;
  for(ALE::Point_set::iterator e_itor = base->begin(); e_itor != base->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> sourceCone = sourceIndices->cone(*e_itor);
    ALE::Obj<ALE::Point_set> targetCone = targetIndices->cone(*e_itor);

    if (sourceCone->size() && targetCone->size()) {
      if (sourceCone->begin()->index != targetCone->begin()->index) {
        SETERRQ2(PETSC_ERR_PLIB, "Mismatch in index sizes %d and %d", sourceCone->begin()->index, targetCone->begin()->index);
      }
      locSourceSize += sourceCone->begin()->index;
    }
  }
#endif
  ierr = PetscMalloc(locSourceSize * sizeof(PetscInt), &fromIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(locSourceSize * sizeof(PetscInt), &toIndices);CHKERRQ(ierr);
  for(ALE::Point_set::iterator e_itor = base->begin(); e_itor != base->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> sourceCone = sourceIndices->cone(*e_itor);
    ALE::Obj<ALE::Point_set> targetCone = targetIndices->cone(*e_itor);

    if (sourceCone->size() && targetCone->size()) {
      ExpandInterval(*sourceCone->begin(), fromIndices, &fromIdx);
      ExpandInterval(*targetCone->begin(), toIndices,   &toIdx);
    }
  }
#if 0
  if ((fromIdx != locSourceSize) || (toIdx != locSourceSize)) {
    SETERRQ3(PETSC_ERR_PLIB, "Invalid index sizes %d, %d should be %d", fromIdx, toIdx, locSourceSize);
  }
#endif
  ierr = ISCreateGeneral(PETSC_COMM_SELF, locSourceSize, fromIndices, &fromIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, locSourceSize, toIndices,   &toIS);CHKERRQ(ierr);
  ierr = PetscFree(fromIndices);CHKERRQ(ierr);
  ierr = PetscFree(toIndices);CHKERRQ(ierr);
  ierr = MeshCreateVector(mesh, sourceBundle, debug, &sourceVec);CHKERRQ(ierr);
  ierr = MeshCreateVector(mesh, targetBundle, debug, &targetVec);CHKERRQ(ierr);
  ierr = VecScatterCreate(sourceVec, fromIS, targetVec, toIS, scatter);CHKERRQ(ierr);
  ierr = VecDestroy(sourceVec);CHKERRQ(ierr);
  ierr = VecDestroy(targetVec);CHKERRQ(ierr);
  ierr = ISDestroy(fromIS);CHKERRQ(ierr);
  ierr = ISDestroy(toIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshDistribute"
/*@
  MeshDistribute - 
*/
PetscErrorCode MeshDistribute(Mesh mesh)
{
  ALE::Obj<ALE::Sieve>       topology;
  ALE::Obj<ALE::PreSieve>    orientation;
  ALE::Obj<ALE::IndexBundle> elementBundle;
  ALE::Obj<ALE::PreSieve>    partitionTypes;
  MPI_Comm                   comm;
  PetscMPIInt                rank;
  PetscInt                   dim;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  ALE_LOG_EVENT_BEGIN
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, &orientation);CHKERRQ(ierr);
  dim = topology->diameter();
  topology->setStratificationPolicy(ALE::Sieve::stratificationPolicyOnLocking);
  /* Partition the topology and orientation */
  ierr = ComputePreSievePartition(orientation, topology->leaves(), "Orientation");CHKERRQ(ierr);
  ierr = PartitionPreSieve(orientation, "Orientation", 1);CHKERRQ(ierr);
  ierr = ComputeSievePartition(topology, "Topology");CHKERRQ(ierr);
  ierr = PartitionPreSieve(topology, "Topology", 1, &partitionTypes);CHKERRQ(ierr);
  topology->getLock();
  topology->releaseLock();
  topology->setStratificationPolicy(ALE::Sieve::stratificationPolicyOnMutation);
  /* Add the trivial vertex orientation */
  ALE::Obj<ALE::Point_set> roots = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = roots->begin(); vertex_itor != roots->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    orientation->addCone(v, v);
  }
  /* Create element bundle */
  elementBundle = ALE::IndexBundle(topology);
  if (debug) {
    elementBundle->setVerbosity(11);
  }
  elementBundle->setFiberDimensionByHeight(0, 1);
  elementBundle->computeOverlapIndices();
  elementBundle->computeGlobalIndices();
  elementBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetElementBundle(mesh, elementBundle);CHKERRQ(ierr);
  ierr = createParallelCoordinates(mesh, dim, partitionTypes);CHKERRQ(ierr);
  ALE_LOG_EVENT_END
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshUnify"
/*@
  MeshUnify - 
*/
PetscErrorCode MeshUnify(Mesh mesh, Mesh *serialMesh)
{
  ALE::Obj<ALE::Sieve>       topology;
  ALE::Obj<ALE::PreSieve>    orientation;
  ALE::Obj<ALE::IndexBundle> coordBundle;
  ALE::Obj<ALE::PreSieve>    partitionTypes;
  ALE::Obj<ALE::Sieve>       boundary;
  Vec               serialCoordinates, coordinates;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  PetscInt          dim;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  /* Unify the topology and orientation */
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, &orientation);CHKERRQ(ierr);
  ALE::Obj<ALE::Sieve>     serialTopology = ALE::Sieve(comm);
  ALE::Obj<ALE::PreSieve>  serialOrientation = ALE::PreSieve(comm);
  ALE::Obj<ALE::Point_set> base = topology->base();
  ALE::Obj<ALE::Point_set> orientationBase = orientation->base();
  ALE::Point               partition(-1, 0);

  ierr = MeshCreate(comm, serialMesh);CHKERRQ(ierr);
  serialOrientation->addCone(orientation->space(), partition);
  for(ALE::Point_set::iterator b_itor = orientationBase->begin(); b_itor != orientationBase->end(); b_itor++) {
    serialOrientation->addCone(orientation->cone(*b_itor), *b_itor);
  }
  ierr = PartitionPreSieve(serialOrientation, "Serial Orientation", 1);CHKERRQ(ierr);
  serialTopology->addCone(topology->space(), partition);
  for(ALE::Point_set::iterator b_itor = base->begin(); b_itor != base->end(); b_itor++) {
    serialTopology->addCone(topology->cone(*b_itor), *b_itor);
  }
  ierr = PartitionPreSieve(serialTopology, "Serial Topology", 1, &partitionTypes);CHKERRQ(ierr);
  ierr = MeshSetTopology(*serialMesh, serialTopology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(*serialMesh, serialOrientation);CHKERRQ(ierr);
  /* Unify boundary */
  ierr = MeshGetBoundary(mesh, &boundary);CHKERRQ(ierr);
  ALE::Obj<ALE::Sieve> serialBoundary = ALE::Sieve(comm);
  ALE::Obj<ALE::Point_set> boundaryBase = boundary->base();

  serialBoundary->addCone(boundary->space(), partition);
  for(ALE::Point_set::iterator b_itor = boundaryBase->begin(); b_itor != boundaryBase->end(); b_itor++) {
    serialBoundary->addCone(boundary->cone(*b_itor), *b_itor);
  }
  ierr = PartitionPreSieve(serialBoundary, "Serial Boundary", 1);CHKERRQ(ierr);
  ierr = MeshSetBoundary(*serialMesh, serialBoundary);CHKERRQ(ierr);
  /* Create vertex bundle */
  ALE::Obj<ALE::IndexBundle> serialVertexBundle = ALE::IndexBundle(serialTopology);
  serialVertexBundle->setFiberDimensionByDepth(0, 1);
  serialVertexBundle->computeOverlapIndices();
  serialVertexBundle->computeGlobalIndices();
  serialVertexBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetVertexBundle(*serialMesh, serialVertexBundle);CHKERRQ(ierr);
  /* Create element bundle */
  ALE::Obj<ALE::IndexBundle> serialElementBundle = ALE::IndexBundle(serialTopology);
  serialElementBundle->setFiberDimensionByHeight(0, 1);
  serialElementBundle->computeOverlapIndices();
  serialElementBundle->computeGlobalIndices();
  serialElementBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetElementBundle(*serialMesh, serialElementBundle);CHKERRQ(ierr);
  /* Create coordinate bundle and storage */
  ALE::Obj<ALE::IndexBundle> serialCoordBundle = ALE::IndexBundle(serialTopology);
  dim = topology->diameter();
  serialCoordBundle->setFiberDimensionByDepth(0, dim);
  serialCoordBundle->computeOverlapIndices();
  serialCoordBundle->computeGlobalIndices();
  serialCoordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshCreateVector(*serialMesh, serialCoordBundle, debug, &serialCoordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(serialCoordinates, dim);CHKERRQ(ierr);
  ierr = MeshSetCoordinateBundle(*serialMesh, serialCoordBundle);CHKERRQ(ierr);
  /* Setup mapping to unified storage */
  VecScatter coordScatter;

  ierr = MeshCreateMapping(mesh, coordBundle, partitionTypes, serialCoordBundle, &coordScatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(coordinates, serialCoordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(coordinates, serialCoordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(*serialMesh, serialCoordinates);CHKERRQ(ierr);
  ierr = VecScatterDestroy(coordScatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

ALE::Obj<ALE::Point_set> getLocal(MPI_Comm comm, ALE::Obj<ALE::Stack> spaceFootprint, ALE::Obj<ALE::Point_set> points)
{
  ALE::Obj<ALE::Point_set> localPoints(new ALE::Point_set);
  ALE::Point     proc(0, spaceFootprint->getCommRank());

  for(ALE::Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
    if (*spaceFootprint->cone(*p_itor)->begin() != proc) continue;
    localPoints->insert(*p_itor);
  }
  return localPoints;
}

PetscErrorCode MeshComputeOverlap(Mesh mesh)
{
  ALE::Obj<ALE::Sieve> topology;
  ALE::Obj<ALE::Stack> spaceFootprint;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MeshGetSpaceFootprint(mesh, &spaceFootprint);CHKERRQ(ierr);
  if (!spaceFootprint) {
    ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
    spaceFootprint = topology->spaceFootprint(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeSupport, NULL);
    ierr = MeshSetSpaceFootprint(mesh, spaceFootprint);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MeshGetDimLocalSize(Mesh mesh, int dim, PetscInt *size)
{
  MPI_Comm             comm;
  ALE::Obj<ALE::Sieve> topology;
  ALE::Obj<ALE::Stack> spaceFootprint;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidIntPointer(size, 3);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MeshComputeOverlap(mesh);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
  ierr = MeshGetSpaceFootprint(mesh, &spaceFootprint);CHKERRQ(ierr);
  *size = getLocal(comm, *spaceFootprint, topology->depthStratum(dim))->size();
  PetscFunctionReturn(0);
}

PetscErrorCode MeshGetDimLocalRanges(Mesh mesh, int dim, PetscInt starts[])
{
  MPI_Comm       comm;
  PetscInt       localSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidIntPointer(starts, 3);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MeshGetDimLocalSize(mesh, dim, &localSize);CHKERRQ(ierr);
  ierr = MPI_Allgather(&localSize, 1, MPI_INT, starts, 1, MPI_INT, comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshGetDimGlobalSize(Mesh mesh, int dim, PetscInt *size)
{
  MPI_Comm       comm;
  PetscInt       localSize, globalSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidIntPointer(size, 3);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MeshGetDimLocalSize(mesh, dim, &localSize);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&localSize, &globalSize, 1, MPI_INT, MPI_SUM, comm);CHKERRQ(ierr);
  *size = globalSize;
  PetscFunctionReturn(0);
}

#endif
