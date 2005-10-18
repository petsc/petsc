#include <petscda.h>
#include <petscdmmg.h>
#include <ALE.hh>
#include <Sieve.hh>
#include <ClosureBundle.hh>

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar   nu;
  BCType        bcType;
} UserContext;

#define NUM_QUADRATURE_POINTS 9

/* Quadrature points */
static double points[18] = {
  -0.794564690381,
  -0.822824080975,
  -0.866891864322,
  -0.181066271119,
  -0.952137735426,
  0.575318923522,
  -0.0885879595127,
  -0.822824080975,
  -0.409466864441,
  -0.181066271119,
  -0.787659461761,
  0.575318923522,
  0.617388771355,
  -0.822824080975,
  0.0479581354402,
  -0.181066271119,
  -0.623181188096,
  0.575318923522};

/* Quadrature weights */
static double weights[9] = {
  0.223257681932,
  0.2547123404,
  0.0775855332238,
  0.357212291091,
  0.407539744639,
  0.124136853158,
  0.223257681932,
  0.2547123404,
  0.0775855332238};

#define NUM_BASIS_FUNCTIONS 3

/* Nodal basis function evaluations */
static double Basis[27] = {
  0.808694385678,
  0.10271765481,
  0.0885879595127,
  0.52397906772,
  0.0665540678392,
  0.409466864441,
  0.188409405952,
  0.0239311322871,
  0.787659461761,
  0.455706020244,
  0.455706020244,
  0.0885879595127,
  0.29526656778,
  0.29526656778,
  0.409466864441,
  0.10617026912,
  0.10617026912,
  0.787659461761,
  0.10271765481,
  0.808694385678,
  0.0885879595127,
  0.0665540678392,
  0.52397906772,
  0.409466864441,
  0.0239311322871,
  0.188409405952,
  0.787659461761};

/* Nodal basis function derivative evaluations */
static double BasisDerivatives[54] = {
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5};

static double meshCoords[18] = {
  0.0, 0.0,
  1.0, 0.0,
  2.0, 0.0,
  2.0, 1.0,
  2.0, 2.0,
  1.0, 2.0,
  0.0, 2.0,
  0.0, 1.0,
  1.0, 1.0};

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::Point_array> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    for(int i = 0; i < (*i_itor).index; i++) {
      indices[k++] = (*i_itor).prefix + i;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandSetIntervals"
PetscErrorCode ExpandSetIntervals(ALE::Point_set intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Point_set::iterator i_itor = intervals.begin(); i_itor != intervals.end(); i_itor++) {
    for(int i = 0; i < (*i_itor).index; i++) {
      indices[k++] = (*i_itor).prefix + i;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePreSievePartition"
PetscErrorCode ComputePreSievePartition(ALE::Obj<ALE::PreSieve> presieve, ALE::Obj<ALE::Point_set> leaves)
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
  presieve->view("Partitioned presieve");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSievePartition"
PetscErrorCode ComputeSievePartition(ALE::Obj<ALE::Sieve> sieve)
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
  sieve->view("Partitioned sieve");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionPreSieve"
PetscErrorCode PartitionPreSieve(ALE::Obj<ALE::PreSieve> presieve)
{
  MPI_Comm       comm = presieve->getComm();
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  // Cone complete to move the partitions to the other processors
  ALE::Obj<ALE::PreSieve> completion = presieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone, NULL)->top();
  completion->view("Completion");
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
  presieve->view("Initial parallel presieve");
  // Cone complete again to build the local topology
  completion = presieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone, NULL)->top();
  completion->view("Completion");
  presieve->add(completion);
  presieve->view("Completed parallel presieve");
  // Restrict to the local partition
  presieve->restrictBase(partition);
  presieve->view("Restricted parallel presieve");
  // Support complete to get the adjacency information
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Simplicializer"
PetscErrorCode Simplicializer(MPI_Comm comm, PetscInt numFaces, PetscInt *faces, PetscInt numVertices, PetscScalar *vertices, PetscInt numBoundaryVertices, PetscInt *boundaryVertices, Mesh *mesh)
{
  Mesh           m;
  ALE::Sieve    *topology = new ALE::Sieve(comm);
  ALE::Sieve    *boundary = new ALE::Sieve(comm);
  ALE::PreSieve *orientation = new ALE::PreSieve(comm);
  PetscInt       curEdge = numFaces+numVertices;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MeshCreate(comm, &m);CHKERRQ(ierr);
  topology->setVerbosity(11);
  orientation->setVerbosity(11);
  boundary->setVerbosity(11);
  /* Create serial sieve */
  if (rank == 0) {
    for(int f = 0; f < numFaces; f++) {
      ALE::Point face(0, f);
      ALE::Point_set edges;
      ALE::Point_set cellTuple;

      edges.clear();
      cellTuple.clear();
      for(int e = 0; e < 3; e++) {
        ALE::Point vS = ALE::Point(0, faces[f*3+e]+numFaces);
        ALE::Point vE = ALE::Point(0, faces[f*3+((e+1)%3)]+numFaces);
        ALE::Obj<ALE::Point_set> preEdge = topology->support(vS);
        ALE::Point_set endpoints;
        ALE::Point edge;

        preEdge->meet(topology->support(vE));
        if (preEdge->size() > 0) {
          edge = *preEdge->begin();
        } else {
          endpoints.clear();
          endpoints.insert(vS);
          endpoints.insert(vE);
          edge = ALE::Point(0, curEdge++);
          topology->addCone(endpoints, edge);
        }
        edges.insert(edge);
        if (e == 0) {
          cellTuple.insert(vS);
          cellTuple.insert(edge);
        }
      }
      topology->addCone(edges, face);
      cellTuple.insert(face);
      orientation->addCone(cellTuple, face);
    }
  }
  topology->view("Serial Simplicializer topology");
  ierr = MeshSetTopology(m, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(m, (void *) orientation);CHKERRQ(ierr);
  /* Create boundary */
  if (rank == 0) {
    ALE::Point_set cone;
    ALE::Point boundaryPoint(0, 1);

    /* Should also put in boundary edges */
    for(int v = 0; v < numBoundaryVertices; v++) {
      ALE::Point vertex = ALE::Point(0, boundaryVertices[v] + numFaces);

      cone.insert(vertex);
    }
    boundary->addCone(cone, boundaryPoint);
    ierr = MeshSetBoundary(m, (void *) boundary);CHKERRQ(ierr);
  }
  topology->view("Simplicializer boundary topology");
  ierr = ComputeSievePartition(topology);CHKERRQ(ierr);
  ierr = PartitionPreSieve(topology);CHKERRQ(ierr);
  ierr = ComputePreSievePartition(orientation, topology->leaves());CHKERRQ(ierr);
  ierr = PartitionPreSieve(orientation);CHKERRQ(ierr);
  ALE::Obj<ALE::Point_set> roots = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = roots->begin(); vertex_itor != roots->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    orientation->addCone(v, v);
  }
  *mesh = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "restrictField"
PetscErrorCode restrictField(ALE::ClosureBundle *bundle, ALE::PreSieve *orientation, PetscScalar *array, ALE::Point e, PetscScalar *values[])
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
#define __FUNCT__ "assembleField"
PetscErrorCode assembleField(ALE::ClosureBundle *bundle, ALE::PreSieve *orientation, Vec b, ALE::Point e, PetscScalar array[], InsertMode mode)
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
#define __FUNCT__ "assembleOperator"
PetscErrorCode assembleOperator(ALE::ClosureBundle *bundle, ALE::PreSieve *orientation, Mat A, ALE::Point e, PetscScalar array[], InsertMode mode)
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
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshCoordinates"
PetscErrorCode CreateMeshCoordinates(Mesh mesh)
{
  ALE::ClosureBundle      *coordBundle;
  ALE::Sieve              *topology;
  ALE::Sieve              *orientation;
  ALE::Obj<ALE::Point_set> vertices;
  ALE::Point_set           empty;
  Vec                      coordinates;
  MPI_Comm                 comm;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm); CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  /* Create bundle */
  coordBundle = new ALE::ClosureBundle(comm);
  coordBundle->setTopology(topology);
  coordBundle->setFiberDimensionByDepth(0, 2);
  //coordBundle->computeOverlapIndices();
  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  /* Create coordinate storage */
  //ierr = MeshCreateGlobalVector(mesh, &coordinates);CHKERRQ(ierr);
  // Need number of ghost dof
  // numGhosts = coordBundle->getGlobalRemoteSize();
  PetscInt numGhosts = 0;
  // Need all global ghost indices
  // ghostIndices = coordBundle->getGlobalRemoteIndices().cap();
  PetscInt *ghostIndices = NULL;
  // Create a global ghosted vector to store a field
  ierr = VecCreateGhostBlock(comm,1,coordBundle->getBundleDimension(empty),PETSC_DETERMINE,numGhosts,ghostIndices,&coordinates);CHKERRQ(ierr);
  /* Set coordinates */
  vertices = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = vertices->begin(); vertex_itor != vertices->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    printf("Sizeof vertex (%d, %d) is %d\n", v.prefix, v.index, coordBundle->getFiberDimension(v));
    ierr = assembleField(coordBundle, orientation, coordinates, v, &meshCoords[(v.index - 8)*2], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTestMesh"
/*
  CreateTestMesh - Create a simple square mesh

         13
  14--28----31---12
    |\    |\    |
    2 2 5 2 3 7 3
    7  6  9  0  2
    | 4 \ | 6 \ |
    |    \|    \|
  15--20-16-24---11
    |\    |\    |
    1 1 1 2 2 3 2
    9  8  1  3  5
    | 0 \ | 2 \ |
    |    \|    \|
   8--17----22---10
          9
*/
PetscErrorCode CreateTestMesh(MPI_Comm comm, Mesh *mesh)
{
  ALE::ClosureBundle *bundle = new ALE::ClosureBundle(comm);
  ALE::Sieve         *topology;
  PetscInt            faces[24] = {
    0, 1, 7,
    8, 7, 1,
    1, 2, 8,
    3, 8, 2,
    7, 8, 6,
    5, 6, 8,
    8, 3, 5,
    4, 5, 3};
  PetscScalar         vertexCoords[18] = {
    0.0, 0.0,
    1.0, 0.0,
    2.0, 0.0,
    2.0, 1.0,
    2.0, 2.0,
    1.0, 2.0,
    0.0, 2.0,
    0.0, 1.0,
    1.0, 1.0};
  PetscInt            boundaryVertices[8] = {
    0, 1, 2, 3, 4, 5, 6, 7};
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = Simplicializer(comm, 8, faces, 9, vertexCoords, 8, boundaryVertices, mesh);CHKERRQ(ierr);
  /* Create field ordering */
  ierr = MeshGetTopology(*mesh, (void **) &topology);CHKERRQ(ierr);
  bundle->setTopology(topology);
  bundle->setFiberDimensionByDepth(0, 1);
  //bundle->computeGlobalIndices();
  ierr = MeshSetBundle(*mesh, (void *) bundle);CHKERRQ(ierr);
  /* Finish old-style DM construction */
  ALE::Point_set empty;
  ierr = MeshSetGhosts(*mesh, 1, bundle->getBundleDimension(empty), 0, NULL);
  /* Create coordinates */
  ierr = CreateMeshCoordinates(*mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
PetscErrorCode ElementGeometry(ALE::ClosureBundle *coordBundle, ALE::PreSieve *orientation, PetscScalar *coords, ALE::Point e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscScalar   *array;
  PetscReal      det, invDet;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = restrictField(coordBundle, orientation, coords, e, &array); CHKERRQ(ierr);
  if (v0) {
    v0[0] = array[0*2+0];
    v0[1] = array[0*2+1];
  }
  if (J) {
    J[0] = 0.5*(array[1*2+0] - array[0*2+0]);
    J[1] = 0.5*(array[2*2+0] - array[0*2+0]);
    J[2] = 0.5*(array[1*2+1] - array[0*2+1]);
    J[3] = 0.5*(array[2*2+1] - array[0*2+1]);
    printf("J = / %g %g \\\n    \\ %g %g /\n", J[0], J[1], J[2], J[3]);
    det = fabs(J[0]*J[3] - J[1]*J[2]);
    invDet = 1.0/det;
    if (detJ) {
      *detJ = det;
    }
    if (invJ) {
      invJ[0] =  invDet*J[3];
      invJ[1] = -invDet*J[1];
      invJ[2] = -invDet*J[2];
      invJ[3] =  invDet*J[0];
      printf("Jinv = / %g %g \\\n       \\ %g %g /\n", invJ[0], invJ[1], invJ[2], invJ[3]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscErrorCode ComputeRho(PetscReal x, PetscReal y, PetscScalar *rho)
{
  PetscFunctionBegin;
  if ((x > 1.0/3.0) && (x < 2.0/3.0) && (y > 1.0/3.0) && (y < 2.0/3.0)) {
    //*rho = 100.0;
    *rho = 1.0;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeBlock"
PetscErrorCode ComputeBlock(DMMG dmmg, Vec u, Vec r, ALE::Point_set block)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::PreSieve      *orientation;
  ALE::ClosureBundle *bundle;
  ALE::ClosureBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscScalar        *array;
  PetscScalar        *field;
  PetscReal           v0[2];
  PetscReal           Jac[4], Jinv[4];
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           linearVec[NUM_BASIS_FUNCTIONS];
  PetscReal           t_der[2], b_der[2];
  PetscReal           xi, eta, x_q, y_q, detJ, rho, funcValue;
  PetscInt            f, g, q;
  PetscErrorCode      ierr;

  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(u, &array); CHKERRQ(ierr);
  elements = topology->star(block);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, Jinv, &detJ); CHKERRQ(ierr);
    ierr = restrictField(bundle, orientation, array, e, &field); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      ierr = ComputeRho(x_q, y_q, &rho);CHKERRQ(ierr);
      funcValue = PetscExpScalar(-(x_q*x_q)/user->nu)*PetscExpScalar(-(y_q*y_q)/user->nu);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          linearVec[f] += rho*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*field[g];
        }
        elementVec[f] += (Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue - linearVec[f])*weights[q]*detJ;
      }
    }
    printf("elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);
    /* Assembly */
    ierr = assembleField(bundle, orientation, r, e, elementVec, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(u, &array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::PreSieve      *orientation;
  ALE::ClosureBundle *bundle;
  ALE::ClosureBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           v0[2];
  PetscReal           Jac[4];
  PetscReal           xi, eta, x_q, y_q, detJ, funcValue;
  PetscInt            f, q;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, PETSC_NULL, &detJ); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      funcValue = PetscExpScalar(-(x_q*x_q)/user->nu)*PetscExpScalar(-(y_q*y_q)/user->nu);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        elementVec[f] += Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue*weights[q]*detJ;
      }
    }
    printf("elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);
    /* Assembly */
    ierr = assembleField(bundle, orientation, b, e, elementVec, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = KSPGetNullSpace(dmmg->ksp,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
PetscErrorCode ComputeJacobian(DMMG dmmg, Mat J, Mat jac)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::Sieve         *boundary;
  ALE::PreSieve      *orientation;
  ALE::ClosureBundle *bundle;
  ALE::ClosureBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscInt            numElementIndices;
  PetscInt           *elementIndices = NULL;
  PetscReal           v0[2];
  PetscReal           elementMat[NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS];
  PetscReal           Jac[4], Jinv[4], t_der[2], b_der[2];
  PetscReal           xi, eta, x_q, y_q, detJ, rho;
  PetscInt            f, g, q;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetBoundary(mesh, (void **) &boundary);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, Jinv, &detJ); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementMat, NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      ierr = ComputeRho(x_q, y_q, &rho);CHKERRQ(ierr);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          elementMat[f*NUM_BASIS_FUNCTIONS+g] += rho*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*weights[q]*detJ;
        }
      }
    }
    printf("elementMat = [%g %g %g]\n             [%g %g %g]\n             [%g %g %g]\n",
           elementMat[0], elementMat[1], elementMat[2], elementMat[3], elementMat[4], elementMat[5], elementMat[6], elementMat[7], elementMat[8]);
    /* Assembly */
    ierr = assembleOperator(bundle, orientation, jac, e, elementMat, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(elementIndices);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    /* Zero out BC rows */
    ALE::Point id(0, 1);
    ALE::Point_set boundaryElements = boundary->cone(id);
    int numBoundaryIndices = bundle->getFiberDimension(boundaryElements);
    ALE::Point_set boundaryIntervals = bundle->getFiberIndices(boundaryElements, empty)->cap();
    PetscInt *boundaryIndices;

    ierr = PetscMalloc(numBoundaryIndices * sizeof(PetscInt), &boundaryIndices); CHKERRQ(ierr);
    ierr = ExpandSetIntervals(boundaryIntervals, boundaryIndices); CHKERRQ(ierr);
    for(int i = 0; i < numElementIndices; i++) {
      printf("boundaryIndices[%d] = %d\n", i, boundaryIndices[i]);
    }
    ierr = MatZeroRows(jac, numBoundaryIndices, boundaryIndices, 1.0);CHKERRQ(ierr);
    ierr = PetscFree(boundaryIndices);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
