#include <petscda.h>
#include <IndexBundle.hh>

#undef __FUNCT__
#define __FUNCT__ "BuildFaces"
PetscErrorCode BuildFaces(int dim, std::map<int, int*> curSimplex, ALE::Point_set boundary, ALE::Point& simplex, ALE::Sieve *topology)
{
  ALE::Point_set faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("  Building faces for boundary(%u), dim %d\n", (unsigned int) boundary.size(), dim);
  if (dim > 1) {
    // Use the cone construction
    for(ALE::Point_set::iterator b_itor = boundary.begin(); b_itor != boundary.end(); b_itor++) {
      ALE::Point_set faceBoundary(boundary);
      ALE::Point face;

      printf("    boundary point (%d, %d)\n", (*b_itor).prefix, (*b_itor).index);
      faceBoundary.erase(*b_itor);
      ierr = BuildFaces(dim-1, curSimplex, faceBoundary, face, topology); CHKERRQ(ierr);
      faces.insert(face);
    }
  } else {
    printf("  Just set faces to boundary in 1d\n");
    faces = boundary;
  }
  for(ALE::Point_set::iterator f_itor = faces.begin(); f_itor != faces.end(); f_itor++) {
    printf("  face point (%d, %d)\n", (*f_itor).prefix, (*f_itor).index);
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
    printf("  Found old simplex (%d, %d)\n", simplex.prefix, simplex.index);
  } else {
    simplex = ALE::Point(0, (*curSimplex[dim])++);
    topology->addCone(faces, simplex);
    printf("  Added simplex (%d, %d), dim %d\n", simplex.prefix, simplex.index, dim);
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
      printf("Adding boundary node (%d, %d)\n", 0, simplices[s*(dim+1)+b]+numSimplices);
      boundary.insert(ALE::Point(0, simplices[s*(dim+1)+b]+numSimplices));
    }
    printf("simplex boundary size %u\n", (unsigned int) boundary.size());
    ierr = BuildFaces(dim, curElement, boundary, simplex, topology); CHKERRQ(ierr);
    /* Orient the simplex */
    ALE::Point element = ALE::Point(0, simplices[s*(dim+1)+0]+numSimplices);
    cellTuple.clear();
    cellTuple.insert(element);
    for(int b = 1; b < dim+1; b++) {
      ALE::Point next = ALE::Point(0, simplices[s*(dim+1)+b]+numSimplices);
      ALE::Obj<ALE::Point_set> join = topology->nJoin(element, next, b);

      if (join->size() == 0) {
        printf("element (%d, %d) next(%d, %d)\n", element.prefix, element.index, next.prefix, next.index);
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
      for(int l = (numLeaves/size)*p + (numLeaves%size > p); l < (numLeaves/size)*(p+1) + (numLeaves%size > p+1); l++) {
        ALE::Point leaf(0, l);
        ALE::Point_set cone = presieve->cone(leaf);
        presieve->addCone(cone, partPoint);
      }
    }
  } else {
    ALE::Point partitionPoint(-1, rank);
    presieve->addBasePoint(partitionPoint);
  }
  //
  ostringstream label1;
  label1 << "Partition of presieve ";
  if(name != NULL) {
    label1 << "'" << name << "'";
  }
  label1 << "\n";
  presieve->view(label1.str().c_str());
  //
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
      for(int l = (numLeaves/size)*p + (numLeaves%size > p); l < (numLeaves/size)*(p+1) + (numLeaves%size > p+1); l++) {
        ALE::Point leaf(0, l);
        ALE::Point_set closure = sieve->closure(leaf);
        sieve->addCone(closure, partPoint);
      }
    }
  } else {
    ALE::Point partitionPoint(-1, rank);
    sieve->addBasePoint(partitionPoint);
  }
  //
  ostringstream label1;
  label1 << "Partition of sieve ";
  if(name != NULL) {
    label1 << "'" << name << "'";
  }
  label1 << "\n";
  sieve->view(label1.str().c_str());
  //
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
  //
  ostringstream label1;
  label1 << "Completion";
  if(name != NULL) {
    label1 << " of '" << name << "'";
  }
  completion->view(label1.str().c_str());
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
    pTypes->view("Partition pointTypes");
    *pointTypes = pTypes;
  }
  // Merge in the completion
  presieve->add(completion);
  //presieve->view("Completed partition");
  // Move the cap to the base of the partition sieve
  ALE::Point partitionPoint(-1, rank);
  ALE::Point_set partition = presieve->cone(partitionPoint);
  for(ALE::Point_set::iterator p_itor = partition.begin(); p_itor != partition.end(); p_itor++) {
    ALE::Point p = *p_itor;
    presieve->addBasePoint(p);
  }
  //
  ostringstream label2;
  if(name != NULL) {
    label2 << "Initial parallel state of '" << name << "'";
  }
  else {
    label2 << "Initial parallel presieve";
  }
  presieve->view(label2.str().c_str());
  //
  // Cone complete again to build the local topology
  completion = presieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone, NULL)->top();
  //
  ostringstream label3;
  if(name != NULL) {
    label3 << "Completion of '" << name << "'";
  }
  else {
    label3 << "Completion";
  }
  completion->view(label3.str().c_str());
  //
  presieve->add(completion);
  //
  ostringstream label4;
  if(name != NULL) {
    label4 << "Completed parallel version of '" << name << "'";
  }
  else {
    label4 << "Completed parallel presieve";
  }
  presieve->view(label4.str().c_str());
  //
  // Unless explicitly prohibited, restrict to the local partition
  if(localize) {
    presieve->restrictBase(partition);
    //
    ostringstream label5;
    if(name != NULL) {
      label5 << "Localized parallel version of '" << name << "'";
    }
    else {
      label5 << "Localized parallel presieve";
    }
    presieve->view(label5.str().c_str());
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
  for(ALE::Point_set::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ExpandInterval(*intervals->begin(), indices, &i);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
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
#define __FUNCT__ "assembleField"
/* This is currently present in ex33mesh.c, although the functionality and the calling sequences differ significantly. */
PetscErrorCode assembleField(Vec b, ALE::Obj<ALE::IndexBundle> bundle, ALE::Obj<ALE::PreSieve> orientation, InsertMode mode)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateSeq"
/*@C
  MeshCreateSeq - Takes an adjacency description of a simplicial mesh FROM PROCESS 0
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
PetscErrorCode MeshCreateSeq(Mesh mesh, int dim, PetscInt numVertices, PetscInt numElements, PetscInt *elements, PetscScalar coords[])
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
  PetscValidIntPointer(elements,4);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  topology->setVerbosity(11);
  orientation->setVerbosity(11);
  boundary->setVerbosity(11);
  /* Create serial sieve */
  if (rank == 0) {
    ierr = BuildTopology(dim, numElements, elements, numVertices, topology, orientation);CHKERRQ(ierr);
  }
  topology->view("Serial Simplicializer topology");
  orientation->view("Serial Simplicializer orientation");
  ierr = MeshSetTopology(mesh, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(mesh, (void *) orientation);CHKERRQ(ierr);
  /* Create the initial coordinate bundle and storage */
  coordBundle->setTopology(topology);
  coordBundle->setFiberDimensionByDepth(0, dim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  coordBundle->getLock();  // lock the bundle so that the overlap indices do not change
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
    printf("Sizeof fiber over vertex (%d, %d) is %d\n", v.prefix, v.index, coordBundle->getFiberDimension(v));
    ierr = setFiberValues(coordinates, v, coordBundle, &coords[(v.index - numElements)*dim], INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscPrintf(comm, "Serial coordinates\n====================\n");
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = VecView(coordinates, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
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
  ALE::IndexBundle *coordBundle;
  ALE::IndexBundle *serialCoordBundle;
  ALE::PreSieve    *partitionTypes;
  Vec               coordinates, oldCoordinates;
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
  ierr = ComputeSievePartition(topology, "Topology");CHKERRQ(ierr);
  ierr = PartitionPreSieve(topology, "Topology", 1, &partitionTypes);CHKERRQ(ierr);
  ierr = ComputePreSievePartition(orientation, topology->leaves(), "Orientation");CHKERRQ(ierr);
  ierr = PartitionPreSieve(orientation, "Orientation", 1);CHKERRQ(ierr);
  /* Add the trivial vertex orientation */
  ALE::Obj<ALE::Point_set> roots = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = roots->begin(); vertex_itor != roots->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    orientation->addCone(v, v);
  }
  /* Create coordinate bundle */
  coordBundle = new ALE::IndexBundle(topology);
  coordBundle->setFiberDimensionByDepth(0, dim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  coordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  /* Create ghosted coordinate storage */
  int localSize = coordBundle->getLocalSize();
  int globalSize = coordBundle->getGlobalSize();
  ALE::Obj<ALE::PreSieve> globalIndices = coordBundle->getGlobalIndices();
  ALE::Obj<ALE::PreSieve> pointTypes = coordBundle->getPointTypes();
  ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(coordBundle->getCommRank(), ALE::rentedPoint));
  int ghostSize = 0;
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ghostSize += interval.index;
    }
  }
  int *ghostIndices = new int[ghostSize];
  int ghostIdx = 0;
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ExpandInterval(interval, ghostIndices, &ghostIdx);
    }
  }
  PetscPrintf(comm, "Making an ordering over the vertices\n===============================\n");
  PetscSynchronizedPrintf(comm, "[%d]  global size: %d localSize: %d ghostSize: %d\n", rank, globalSize, localSize, ghostSize);
  PetscSynchronizedPrintf(comm, "[%d]  ghostIndices:", rank);
  for(int g = 0; g < ghostSize; g++) {
    PetscSynchronizedPrintf(comm, "[%d] %d\n", rank, ghostIndices[g]);
  }
  PetscSynchronizedPrintf(comm, "\n");
  PetscSynchronizedFlush(comm);
  ierr = VecCreateGhostBlock(comm,dim,localSize,globalSize,ghostSize,ghostIndices,&coordinates);CHKERRQ(ierr);
  /* Setup mapping to partitioned storage */
  ALE::Obj<ALE::Stack> mappingStack;
  ALE::Obj<ALE::PreSieve> sourceIndices, targetIndices;
  ALE::Obj<ALE::Point_set> base;
  IS         coordIS, oldCoordIS;
  VecScatter coordScatter;
  PetscInt   oldLocalSize = serialCoordBundle->getLocalSize();
  PetscInt  *oldIndices, *indices;
  PetscInt   oldIdx = 0, idx = 0;

  mappingStack = serialCoordBundle->computeMappingIndices(partitionTypes, coordBundle);
  sourceIndices = mappingStack->top();
  targetIndices = mappingStack->bottom();
  base = sourceIndices->base();
  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  ierr = PetscMalloc(oldLocalSize * sizeof(PetscInt), &oldIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(oldLocalSize * sizeof(PetscInt), &indices);CHKERRQ(ierr);
  for(ALE::Point_set::iterator e_itor = base->begin(); e_itor != base->end(); e_itor++) {
    ALE::Point point = *e_itor;

    ExpandInterval(*sourceIndices->cone(point).begin(), oldIndices, &oldIdx);
    ExpandInterval(*targetIndices->cone(point).begin(), indices,    &idx);
  }
  ierr = ISCreateGeneral(comm, oldLocalSize, oldIndices, &oldCoordIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, oldLocalSize, indices,    &coordIS);CHKERRQ(ierr);
  ierr = PetscFree(oldIndices);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecScatterCreate(oldCoordinates, oldCoordIS, coordinates, coordIS, &coordScatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(oldCoordinates, coordinates, INSERT_VALUES, SCATTER_FORWARD, coordScatter);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  ierr = VecScatterDestroy(coordScatter);CHKERRQ(ierr);
  ierr = ISDestroy(oldCoordIS);CHKERRQ(ierr);
  ierr = ISDestroy(coordIS);CHKERRQ(ierr);
  ierr = VecDestroy(oldCoordinates);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Parallel Coordinates\n===========================\n");CHKERRQ(ierr);
  ierr = VecView(coordinates, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  delete serialCoordBundle;
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

  Level: developer

.seealso MeshCreate(), MeshGetTopology(), MeshSetTopology()
*/
PetscErrorCode MeshCreateBoundary(Mesh mesh, PetscInt numBoundaryVertices, PetscInt *boundaryVertices)
{
  MPI_Comm comm;
  PetscObjectGetComm((PetscObject) mesh, &comm);
  ALE::Sieve    *boundary = new ALE::Sieve(comm);
  ALE::Sieve    *topology;
  PetscInt       numElements;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,DA_COOKIE,1);
  PetscValidIntPointer(boundaryVertices,3);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  numElements = topology->heightStratum(0).size();
  boundary->setVerbosity(11);
  /* Create boundary */
  if (rank == 0) {
    ALE::Point_set cone;
    ALE::Point boundaryPoint(0, 1);

    /* Should also put in boundary edges */
    for(int v = 0; v < numBoundaryVertices; v++) {
      ALE::Point vertex = ALE::Point(0, boundaryVertices[v] + numElements);

      cone.insert(vertex);
    }
    boundary->addCone(cone, boundaryPoint);
    ierr = MeshSetBoundary(mesh, (void *) boundary);CHKERRQ(ierr);
  }
  boundary->view("Simplicializer boundary topology");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateCoordinates"
PetscErrorCode MeshCreateCoordinates(Mesh mesh, PetscScalar coords[])
{
  ALE::IndexBundle      *coordBundle;
  ALE::Sieve              *topology;
  ALE::Sieve              *orientation;
  ALE::Obj<ALE::Point_set> vertices;
  ALE::Point_set           empty;
  Vec                      coordinates;
  PetscInt                 dim, numElements;
  MPI_Comm                 comm;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm); CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  dim = topology->diameter();
  /* Create bundle */
  coordBundle = new ALE::IndexBundle(comm);
  coordBundle->setTopology(topology);
  coordBundle->setFiberDimensionByDepth(0, dim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  coordBundle->getLock();  // lock the bundle so that the overlap indices do not change
  //
  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  /* Create coordinate storage */
  int localSize = coordBundle->getLocalSize();
  int globalSize = coordBundle->getGlobalSize();
  ALE::Obj<ALE::PreSieve> globalIndices = coordBundle->getGlobalIndices();
  globalIndices->view("Global Indices");
  ALE::Obj<ALE::PreSieve> pointTypes = coordBundle->getPointTypes();
  ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(coordBundle->getCommRank(), ALE::rentedPoint));
  int ghostSize = 0;
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ghostSize += interval.index;
    }
  }
  int *ghostIndices = new int[ghostSize];
  int idx = 0;
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ExpandInterval(interval, ghostIndices, &idx);
    }
  }
  /* Print shit */
  printf("Making an ordering over the vertices\n===============================\n");
  printf("  global size: %d ghostSize: %d\n", globalSize, ghostSize);
  printf("  ghostIndices:");
  for(int g = 0; g < ghostSize; g++) {
    printf(" %d\n", ghostIndices[g]);
  }
  printf("\n");
  // NO: Create a global ghosted vector to store a field
  ierr = VecCreateGhostBlock(comm,1,localSize,globalSize,ghostSize,ghostIndices,&coordinates);CHKERRQ(ierr);
  // Create a scatter from original coordinates to partitioned storage
  // Set local coordinates
  numElements = topology->heightStratum(0).size();
  vertices = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = vertices->begin(); vertex_itor != vertices->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    printf("Sizeof fiber over vertex (%d, %d) is %d\n", v.prefix, v.index, coordBundle->getFiberDimension(v));
    ierr = setFiberValues(coordinates, v,coordBundle, &coords[(v.index - numElements)*dim], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
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
  ALE::Sieve         *topology;
  ALE::IndexBundle *coordBundle;
  PetscErrorCode      ierr;

  PetscValidIntPointer(dimension,2);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  *dimension = coordBundle->getFiberDimension(*topology->depthStratum(0).begin());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "restrictField"
PetscErrorCode restrictField(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, PetscScalar *array, ALE::Point e, PetscScalar *values[])
{
  ALE::Point_set             empty;
  ALE::Obj<ALE::Point_array> intervals = bundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> intervals = bundle->getOverlapOrderedIndices(orientation->cone(e), empty);
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
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
    vals[i] = array[indices[i]];
  }
  *values = vals;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKVertices"
PetscErrorCode WriteVTKVertices(Mesh mesh, PetscViewer viewer)
{
  Vec            coordinates;
  PetscInt       dim, numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetDimension(mesh, &dim);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &numVertices);CHKERRQ(ierr);
  numVertices /= dim;
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n", numVertices);CHKERRQ(ierr);
  ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Sieve         *topology;
  ALE::IndexBundle  vertexBundle;
  ALE::IndexBundle  elementBundle;
  ALE::Point_set      elements;
  int                 dim, numElements, corners;
  MPI_Comm            comm;
  PetscMPIInt         rank, size;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  vertexBundle = ALE::IndexBundle(topology);
  vertexBundle.setFiberDimensionByDepth(0, 1);
  vertexBundle.computeOverlapIndices();
  vertexBundle.computeGlobalIndices();
  elementBundle = ALE::IndexBundle(topology);
  elementBundle.setFiberDimensionByHeight(0, 1);
  elementBundle.computeOverlapIndices();
  elementBundle.computeGlobalIndices();
  dim = topology->diameter();
  elements = topology->heightStratum(0);
  numElements = elementBundle.getGlobalSize();
  corners = topology->nCone(*elements.begin(), dim).size();
  ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
  if (rank == 0) {
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
      ALE::Point_set cone = topology->nCone(*e_itor, dim);
      for(ALE::Point_set::iterator c_itor = cone.begin(); c_itor != cone.end(); c_itor++) {
        ALE::Point index = vertexBundle.getGlobalFiberInterval(*c_itor);

        ierr = PetscViewerASCIIPrintf(viewer, " %d", index.prefix);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
    for(int p = 1; p < size; p++) {
      MPI_Status  status;
      int        *remoteVertices;
      int         numLocalElements;

      ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &remoteVertices);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteVertices, numLocalElements*corners, MPI_INT, 0, 1, comm, &status);CHKERRQ(ierr);
      for(int e = 0; e < numLocalElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
    }
  } else {
    int  numLocalElements = elements.size(), offset = 0;
    int *array;

    ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &array);CHKERRQ(ierr);
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ALE::Point_set cone = topology->nCone(*e_itor, dim);
      for(ALE::Point_set::iterator c_itor = cone.begin(); c_itor != cone.end(); c_itor++) {
        ALE::Point index = vertexBundle.getGlobalFiberInterval(*c_itor);

        array[offset++] = index.prefix;
      }
    }
    if (offset != numLocalElements*corners) {
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d shuold be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "CELL_TYPES %d\n", numElements);CHKERRQ(ierr);
  if (corners == 3) {
    // VTK_TRIANGLE
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "5\n");CHKERRQ(ierr);
    }
  } else if (corners == 4) {
    // VTK_TETRA
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "10\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode WriteVTKHeader(Mesh mesh, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Simplicial Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
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
