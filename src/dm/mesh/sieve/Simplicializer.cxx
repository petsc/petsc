#include <petscda.h>
#include <IndexBundle.hh>

int debug = 0;

#undef __FUNCT__
#define __FUNCT__ "BuildFaces"
PetscErrorCode BuildFaces(int dim, std::map<int, int*> curSimplex, ALE::Point_set boundary, ALE::Point& simplex, ALE::Sieve *topology)
{
  ALE::Point_set faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (debug) {PetscPrintf(PETSC_COMM_SELF, "  Building faces for boundary(size %u), dim %d\n", (unsigned int) boundary.size(), dim);}
  if (dim > 1) {
    // Use the cone construction
    for(ALE::Point_set::iterator b_itor = boundary.begin(); b_itor != boundary.end(); b_itor++) {
      ALE::Point_set faceBoundary(boundary);
      ALE::Point face;

      if (debug) {PetscPrintf(PETSC_COMM_SELF, "    boundary point (%d, %d)\n", (*b_itor).prefix, (*b_itor).index);}
      faceBoundary.erase(*b_itor);
      ierr = BuildFaces(dim-1, curSimplex, faceBoundary, face, topology); CHKERRQ(ierr);
      faces.insert(face);
    }
  } else {
    if (debug) {PetscPrintf(PETSC_COMM_SELF, "  Just set faces to boundary in 1d\n");}
    faces = boundary;
  }
  for(ALE::Point_set::iterator f_itor = faces.begin(); f_itor != faces.end(); f_itor++) {
    if (debug) {PetscPrintf(PETSC_COMM_SELF, "  face point (%d, %d)\n", (*f_itor).prefix, (*f_itor).index);}
  }
  // We always create the toplevel, so we could shortcircuit somehow
  // Should not have to loop here since the meet of just 2 boundary elements is an element
  ALE::Point_set::iterator f_itor = faces.begin();
  ALE::Point start = *f_itor;
  f_itor++;
  ALE::Point next = *f_itor;
  ALE::Obj<ALE::Point_set> preElement = topology->nJoin(start, next, 1);

  if (preElement->size() > 0) {
    simplex = *preElement->begin();
    if (debug) {PetscPrintf(PETSC_COMM_SELF, "  Found old simplex (%d, %d)\n", simplex.prefix, simplex.index);}
  } else {
    simplex = ALE::Point(0, (*curSimplex[dim])++);
    topology->addCone(faces, simplex);
    if (debug) {PetscPrintf(PETSC_COMM_SELF, "  Added simplex (%d, %d), dim %d\n", simplex.prefix, simplex.index, dim);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BuildTopology"
PetscErrorCode BuildTopology(int dim, PetscInt numSimplices, PetscInt *simplices, PetscInt numVertices, ALE::Sieve *topology, ALE::PreSieve *orientation)
{
  int            curVertex  = numSimplices;
  int            curSimplex = 0;
  int            newElement = numSimplices+numVertices;
  std::map<int,int*> curElement;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  curElement[0] = &curVertex;
  curElement[dim] = &curSimplex;
  for(int d = 1; d < dim; d++) {
    curElement[d] = &newElement;
  }
  for(int s = 0; s < numSimplices; s++) {
    ALE::Point simplex(0, s);
    ALE::Point_set cellTuple;
    ALE::Point_set boundary;

    /* Build the simplex */
    boundary.clear();
    for(int b = 0; b < dim+1; b++) {
      ALE::Point vertex(0, simplices[s*(dim+1)+b]+numSimplices);

      if (debug) {PetscPrintf(PETSC_COMM_SELF, "Adding boundary node (%d, %d)\n", vertex.prefix, vertex.index);}
      boundary.insert(vertex);
    }
    PetscPrintf(PETSC_COMM_SELF, "simplex %d\n", s);
    if (debug) {PetscPrintf(PETSC_COMM_SELF, "  simplex %d boundary size %u\n", s, (unsigned int) boundary.size());}
    ierr = BuildFaces(dim, curElement, boundary, simplex, topology); CHKERRQ(ierr);
    /* Orient the simplex */
    ALE::Point element = ALE::Point(0, simplices[s*(dim+1)+0]+numSimplices);
    cellTuple.clear();
    cellTuple.insert(element);
    for(int b = 1; b < dim+1; b++) {
      ALE::Point next = ALE::Point(0, simplices[s*(dim+1)+b]+numSimplices);
      ALE::Obj<ALE::Point_set> join = topology->nJoin(element, next, b);

      if (join->size() == 0) {
        PetscPrintf(PETSC_COMM_SELF, "element (%d, %d) next(%d, %d)\n", element.prefix, element.index, next.prefix, next.index);
        SETERRQ(PETSC_ERR_LIB, "Invalid join");
      }
      element = *join->begin();
      cellTuple.insert(element);
    }
    orientation->addCone(cellTuple, simplex);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePreSievePartition"
PetscErrorCode ComputePreSievePartition(ALE::Obj<ALE::PreSieve> presieve, ALE::Obj<ALE::Point_set> leaves, const char *name = NULL)
{
  MPI_Comm       comm = presieve->getComm();
  PetscInt       numLeaves = leaves->size();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSievePartition"
PetscErrorCode ComputeSievePartition(ALE::Obj<ALE::Sieve> sieve, const char *name = NULL)
{
  MPI_Comm       comm = sieve->getComm();
  PetscInt       numLeaves = sieve->leaves().size();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (rank == 0) {
    for(int p = 0; p < size; p++) {
      ALE::Point partPoint(-1, p);
      for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
        ALE::Point leaf(0, l);
        ALE::Point_set closure = sieve->closure(leaf);
        sieve->addCone(closure, partPoint);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionPreSieve"
PetscErrorCode PartitionPreSieve(ALE::Obj<ALE::PreSieve> presieve, const char *name = NULL, bool localize = 1, ALE::PreSieve **pointTypes = NULL)
{
  MPI_Comm       comm = presieve->getComm();
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    ALE::PreSieve *pTypes = new ALE::PreSieve(comm);

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
PetscErrorCode MeshComputeGlobalScatter(ALE::Sieve *topology, ALE::IndexBundle *bundle, VecScatter *injection)
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
  //ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  //ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
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
PetscErrorCode setFiberValues(Vec b, ALE::Point e, ALE::Obj<ALE::IndexBundle> bundle, PetscScalar array[], InsertMode mode)
{
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setClosureValues"
PetscErrorCode setClosureValues(Vec b, ALE::Point e, ALE::Obj<ALE::IndexBundle> bundle, ALE::Obj<ALE::PreSieve> orientation, PetscScalar array[], InsertMode mode)
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
PetscErrorCode restrictField(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, PetscScalar *array, ALE::Point e, PetscScalar *values[])
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
PetscErrorCode assembleField(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, Vec b, ALE::Point e, PetscScalar array[], InsertMode mode)
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
  ierr = VecSetValues(b, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleOperator"
PetscErrorCode assembleOperator(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, Mat A, ALE::Point e, PetscScalar array[], InsertMode mode)
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

#undef __FUNCT__
#define __FUNCT__ "MeshPopulate"
/*@C
  MeshPopulate - Takes an adjacency description of a simplicial mesh FROM PROCESS 0
                 and produces a partitioned, parallel, oriented Mesh with given topology.

  Input Parameters:
+ mesh - The PETSc mesh object
. dim - The dimension of the mesh
. numVertices - The number of vertices in the mesh
. numElements - The number of (highest dimensional) elements in the mesh
- elements - The indices of all vertices on each element

  Level: developer

.seealso MeshCreate(), MeshGetTopology(), MeshSetTopology()
*/
PetscErrorCode MeshPopulate(Mesh mesh, int dim, PetscInt numVertices, PetscInt numElements, PetscInt elements[], PetscScalar coords[])
{
  MPI_Comm comm;
  PetscObjectGetComm((PetscObject) mesh, &comm);
  ALE::Sieve       *topology = new ALE::Sieve(comm);
  ALE::Sieve       *boundary = new ALE::Sieve(comm);
  ALE::PreSieve    *orientation = new ALE::PreSieve(comm);
  ALE::IndexBundle *coordBundle = new ALE::IndexBundle(comm);
  Vec               coordinates;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  if (debug) {
    topology->setVerbosity(11);
    orientation->setVerbosity(11);
    boundary->setVerbosity(11);
  }
  /* Create serial sieve */
  if (rank == 0) {
    PetscValidIntPointer(elements,5);
    PetscValidScalarPointer(coords,6);
    ierr = BuildTopology(dim, numElements, elements, numVertices, topology, orientation);CHKERRQ(ierr);
  }
  if (debug) {
    topology->view("Serial Simplicializer topology");
    orientation->view("Serial Simplicializer orientation");
  }
  ierr = MeshSetTopology(mesh, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(mesh, (void *) orientation);CHKERRQ(ierr);
  /* Create the initial coordinate bundle and storage */
  coordBundle->setTopology(topology);
  coordBundle->setFiberDimensionByDepth(0, dim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  coordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  if (debug) {
    coordBundle->setVerbosity(11);
  }
  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  int localSize = coordBundle->getLocalSize();
  int globalSize = coordBundle->getGlobalSize();
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, localSize, globalSize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ALE::Obj<ALE::Point_set> vertices = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = vertices->begin(); vertex_itor != vertices->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    if (debug) {
      printf("Sizeof fiber over vertex (%d, %d) is %d\n", v.prefix, v.index, coordBundle->getFiberDimension(v));
#if !defined(PETSC_HAVE_COMPLEX)
      for(int c = 0; c < coordBundle->getFiberDimension(v); c++) {printf("  %c: %g\n", 'x'+c, coords[(v.index - numElements)*dim+c]);}
#endif
    }
    ierr = setFiberValues(coordinates, v, coordBundle, &coords[(v.index - numElements)*dim], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Serial coordinates\n====================\n");CHKERRQ(ierr);
    ierr = VecView(coordinates, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MeshCreateVector(Mesh, ALE::IndexBundle *, int, Vec *);

#undef __FUNCT__
#define __FUNCT__ "MeshCreateMapping"
PetscErrorCode MeshCreateMapping(Mesh mesh, ALE::IndexBundle *sourceBundle, ALE::PreSieve *pointTypes, ALE::IndexBundle *targetBundle, VecScatter *scatter)
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
  ierr = PetscMalloc(locSourceSize * sizeof(PetscInt), &fromIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(locSourceSize * sizeof(PetscInt), &toIndices);CHKERRQ(ierr);
  base = sourceIndices->base();
  for(ALE::Point_set::iterator e_itor = base->begin(); e_itor != base->end(); e_itor++) {
    ALE::Point               point = *e_itor;
    ALE::Obj<ALE::Point_set> sourceCone = sourceIndices->cone(point);
    ALE::Obj<ALE::Point_set> targetCone = targetIndices->cone(point);

    if (sourceCone->size() && targetCone->size()) {
      ExpandInterval(*sourceCone->begin(), fromIndices, &fromIdx);
      ExpandInterval(*targetCone->begin(), toIndices,   &toIdx);
    }
  }
  if ((fromIdx != locSourceSize) || (toIdx != locSourceSize)) {
    SETERRQ3(PETSC_ERR_PLIB, "Invalid index sizes %d, %d should be %d", fromIdx, toIdx, locSourceSize);
  }
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
  ALE::Sieve       *topology;
  ALE::PreSieve    *orientation;
  ALE::IndexBundle *elementBundle;
  ALE::IndexBundle *coordBundle;
  ALE::IndexBundle *serialCoordBundle;
  ALE::PreSieve    *partitionTypes;
  Vec               coordinates, oldCoordinates, locCoordinates;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  PetscInt          dim;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &serialCoordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &oldCoordinates);CHKERRQ(ierr);
  dim = topology->diameter();
  /* Partition the topology and orientation */
  ierr = ComputePreSievePartition(orientation, topology->leaves(), "Orientation");CHKERRQ(ierr);
  ierr = PartitionPreSieve(orientation, "Orientation", 1);CHKERRQ(ierr);
  ierr = ComputeSievePartition(topology, "Topology");CHKERRQ(ierr);
  ierr = PartitionPreSieve(topology, "Topology", 1, &partitionTypes);CHKERRQ(ierr);
  /* Add the trivial vertex orientation */
  ALE::Obj<ALE::Point_set> roots = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = roots->begin(); vertex_itor != roots->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    orientation->addCone(v, v);
  }
  /* Create element bundle */
  elementBundle = new ALE::IndexBundle(topology);
  if (debug) {
    elementBundle->setVerbosity(11);
  }
  elementBundle->setFiberDimensionByHeight(0, 1);
  elementBundle->computeOverlapIndices();
  elementBundle->computeGlobalIndices();
  elementBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetElementBundle(mesh, (void *) elementBundle);CHKERRQ(ierr);
  /* Create coordinate bundle and storage */
  coordBundle = new ALE::IndexBundle(topology);
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
  /* Setup mapping to partitioned storage */
  VecScatter coordScatter;

  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  ierr = MeshCreateMapping(mesh, serialCoordBundle, partitionTypes, coordBundle, &coordScatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  ierr = VecScatterDestroy(coordScatter);CHKERRQ(ierr);
  ierr = VecDestroy(oldCoordinates);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Parallel Coordinates\n===========================\n");CHKERRQ(ierr);
    ierr = VecView(coordinates, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  delete serialCoordBundle;
  /* Communicate ghosted coordinates */
  ierr = VecGhostUpdateBegin(coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshUnify"
/*@
  MeshUnify - 
*/
PetscErrorCode MeshUnify(Mesh mesh, Mesh *serialMesh)
{
  ALE::Sieve       *topology;
  ALE::PreSieve    *orientation;
  ALE::IndexBundle *coordBundle;
  ALE::PreSieve    *partitionTypes;
  ALE::Sieve       *boundary;
  Vec               serialCoordinates, coordinates;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  PetscInt          dim;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  dim = topology->diameter();
  /* Unify the topology and orientation */
  ALE::Sieve    *serialTopology = new ALE::Sieve(PETSC_COMM_SELF);
  ALE::PreSieve *serialOrientation = new ALE::PreSieve(PETSC_COMM_SELF);
  ALE::Point_set space = topology->space();
  ALE::Point_set base = topology->base();
  ALE::Point_set orderSpace = orientation->space();
  ALE::Point_set orderBase = orientation->base();
  ALE::Point     partition(-1, 0);

  ierr = MeshCreate(PETSC_COMM_SELF, serialMesh);CHKERRQ(ierr);
  serialOrientation->addCone(orderSpace, partition);
  for(ALE::Point_set::iterator b_itor = orderBase.begin(); b_itor != orderBase.end(); b_itor++) {
    ALE::Point     point = *b_itor;
    ALE::Point_set cone = orientation->cone(point);

    serialOrientation->addCone(cone, point);
  }
  ierr = PartitionPreSieve(serialOrientation, "Serial Orientation", 1);CHKERRQ(ierr);
  serialTopology->addCone(space, partition);
  for(ALE::Point_set::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++) {
    ALE::Point     point = *b_itor;
    ALE::Point_set cone = topology->cone(point);

    serialTopology->addCone(cone, point);
  }
  ierr = PartitionPreSieve(serialTopology, "Serial Topology", 1, &partitionTypes);CHKERRQ(ierr);
  ierr = MeshSetTopology(*serialMesh, (void *) serialTopology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(*serialMesh, (void *) serialOrientation);CHKERRQ(ierr);
  /* Unify boundary */
  ierr = MeshGetBoundary(mesh, (void **) &boundary);CHKERRQ(ierr);
  ALE::Sieve    *serialBoundary = new ALE::Sieve(PETSC_COMM_SELF);
  ALE::Point_set boundarySpace = boundary->space();
  ALE::Point_set boundaryBase = boundary->base();
  serialBoundary->addCone(boundarySpace, partition);
  for(ALE::Point_set::iterator b_itor = boundaryBase.begin(); b_itor != boundaryBase.end(); b_itor++) {
    ALE::Point     point = *b_itor;
    ALE::Point_set cone = boundary->cone(point);

    serialBoundary->addCone(cone, point);
  }
  ierr = PartitionPreSieve(serialBoundary, "Serial Boundary", 1);CHKERRQ(ierr);
  ierr = MeshSetBoundary(*serialMesh, (void *) serialBoundary);CHKERRQ(ierr);
  /* Create vertex bundle */
  ALE::IndexBundle *serialVertexBundle = new ALE::IndexBundle(serialTopology);
  serialVertexBundle->setFiberDimensionByDepth(0, 1);
  serialVertexBundle->computeOverlapIndices();
  serialVertexBundle->computeGlobalIndices();
  serialVertexBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetVertexBundle(*serialMesh, (void *) serialVertexBundle);CHKERRQ(ierr);
  /* Create element bundle */
  ALE::IndexBundle *serialElementBundle = new ALE::IndexBundle(serialTopology);
  serialElementBundle->setFiberDimensionByHeight(0, 1);
  serialElementBundle->computeOverlapIndices();
  serialElementBundle->computeGlobalIndices();
  serialElementBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshSetElementBundle(*serialMesh, (void *) serialElementBundle);CHKERRQ(ierr);
  /* Create coordinate bundle and storage */
  ALE::IndexBundle *serialCoordBundle = new ALE::IndexBundle(serialTopology);
  serialCoordBundle->setFiberDimensionByDepth(0, dim);
  serialCoordBundle->computeOverlapIndices();
  serialCoordBundle->computeGlobalIndices();
  serialCoordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  ierr = MeshCreateVector(*serialMesh, serialCoordBundle, debug, &serialCoordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(serialCoordinates, dim);CHKERRQ(ierr);
  ierr = MeshSetCoordinateBundle(*serialMesh, (void *) serialCoordBundle);CHKERRQ(ierr);
  /* Setup mapping to unified storage */
  VecScatter coordScatter;

  ierr = MeshCreateMapping(mesh, coordBundle, partitionTypes, serialCoordBundle, &coordScatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(coordinates, serialCoordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(coordinates, serialCoordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(*serialMesh, serialCoordinates);CHKERRQ(ierr);
  ierr = VecScatterDestroy(coordScatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateBoundary"
/*@C
  MeshCreateBoundary - Takes a description of the mesh boundary ON PROCESS 0.

  Input Parameters:
+ mesh - The PETSc mesh object
. numBoundaryVertices - The number of vertices in the mesh boundary
- boundaryVertices - The indices of all boundary vertices

  Output Parameters:
+ boundaryBundle - A numbering for the boundary values
- boundaryVec - The vector of boundary values

  Level: developer

.seealso MeshCreate(), MeshGetTopology(), MeshSetTopology()
*/
PetscErrorCode MeshCreateBoundary(Mesh mesh, PetscInt numBoundaryVertices, PetscInt numBoundaryComponents, PetscInt *boundaryVertices, PetscScalar *boundaryValues, void **boundaryBundle, Vec *boundaryVec)
{
  MPI_Comm          comm;
  PetscObjectGetComm((PetscObject) mesh, &comm);
  ALE::Sieve       *boundary = new ALE::Sieve(comm);
  ALE::Sieve       *topology;
  ALE::IndexBundle *elementBundle;
  ALE::IndexBundle *bdBundle;
  PetscScalar      *values;
  PetscInt          numElements;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  PetscValidIntPointer(boundaryVertices,3);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(mesh, (void **) &elementBundle);CHKERRQ(ierr);
  numElements = elementBundle->getGlobalSize();
  if (debug) {
    boundary->setVerbosity(11);
  }
  /* Should also put in boundary edges */
  bdBundle = new ALE::IndexBundle(topology);
  for(int v = numBoundaryVertices-1; v >= 0; v--) {
    ALE::Point vertex = ALE::Point(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);

    if (topology->capContains(vertex) && !boundary->capContains(vertex)) {
      bdBundle->setFiberDimension(vertex, numBoundaryComponents);
      for(int c = 0; c < numBoundaryComponents; c++) {
        ALE::Point boundaryPoint(c, boundaryVertices[v*(numBoundaryComponents+1)+c+1]);

        boundary->addCone(vertex, boundaryPoint);
      }
    }
  }
  ierr = MeshSetBoundary(mesh, (void *) boundary);CHKERRQ(ierr);
  if (debug) {
    boundary->view("Boundary sieve");
  }
  if (boundaryVec) {
    ierr = VecCreate(PETSC_COMM_SELF, boundaryVec);CHKERRQ(ierr);
    ierr = VecSetSizes(*boundaryVec, bdBundle->getFiberDimension(topology->depthStratum(0)), PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*boundaryVec);CHKERRQ(ierr);
    ierr = VecGetArray(*boundaryVec, &values);CHKERRQ(ierr);
    for(int v = 0; v < numBoundaryVertices; v++) {
      ALE::Point vertex = ALE::Point(0, boundaryVertices[v*(numBoundaryComponents+1)] + numElements);
      ALE::Point interval = bdBundle->getFiberInterval(vertex);

      for(int c = 0; c < numBoundaryComponents; c++) {
        values[interval.prefix+c] = boundaryValues[v*numBoundaryComponents+c];
      }
    }
    ierr = VecRestoreArray(*boundaryVec, &values);CHKERRQ(ierr);
  }
  if (boundaryBundle) {
    *boundaryBundle = bdBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetDimension"
/*@C
    MeshGetDimension - Return the topological dimension of the mesh, meaning the
                       maximum dimension of any element.

    Not collective

    Input Parameter:
.   mesh - the mesh object

    Output Parameter:
.   dimension - the topological dimension
 
    Level: advanced

.seealso MeshCreate(), MeshGetEmbedddingDimension()

@*/
PetscErrorCode MeshGetDimension(Mesh mesh, PetscInt *dimension)
{
  ALE::Sieve    *topology;
  PetscErrorCode ierr;

  PetscValidIntPointer(dimension,2);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  *dimension = topology->diameter();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetEmbeddingDimension"
/*@C
    MeshGetEmbeddingDimension - Return the dimension of the ambient space for the mesh.

    Not collective

    Input Parameter:
.   mesh - the mesh object

    Output Parameter:
.   dimension - the dimension of the ambient space
 
    Level: advanced

.seealso MeshCreate(), MeshGetDimension()

@*/
PetscErrorCode MeshGetEmbeddingDimension(Mesh mesh, PetscInt *dimension)
{
  ALE::Sieve       *topology;
  ALE::IndexBundle *coordBundle;
  PetscErrorCode    ierr;

  PetscValidIntPointer(dimension,2);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  *dimension = coordBundle->getFiberDimension(*topology->depthStratum(0).begin());
  PetscFunctionReturn(0);
}

ALE::Obj<ALE::Point_set> getLocal(MPI_Comm comm, ALE::Obj<ALE::Stack> spaceFootprint, ALE::Obj<ALE::Point_set> points)
{
  ALE::Obj<ALE::Point_set> localPoints(new ALE::Point_set);
  ALE::Point     proc(0, spaceFootprint->getCommRank());

  for(ALE::Point_set::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
    if (*spaceFootprint->cone(*p_itor).begin() != proc) continue;
    localPoints->insert(*p_itor);
  }
  return localPoints;
}

PetscErrorCode MeshComputeOverlap(Mesh mesh)
{
  ALE::Sieve    *topology;
  ALE::Stack    *spaceFootprint;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetSpaceFootprint(mesh, (void **) &spaceFootprint);CHKERRQ(ierr);
  if (!spaceFootprint) {
    ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
    spaceFootprint = topology->spaceFootprint(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeSupport, NULL);
    ierr = MeshSetSpaceFootprint(mesh, (void *) spaceFootprint);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MeshGetDimLocalSize(Mesh mesh, int dim, PetscInt *size)
{
  MPI_Comm       comm;
  ALE::Sieve    *topology;
  ALE::Stack    *spaceFootprint;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidIntPointer(size, 3);
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MeshComputeOverlap(mesh);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetSpaceFootprint(mesh, (void **) &spaceFootprint);CHKERRQ(ierr);
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
