extern "C" {
  #include <petscda.h>
  #include <petscdmmg.h>
}
#include <ALE/ALE.hh>
#include <ALE/Sieve.hh>
#include <ALE/ClosureBundle.hh>

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
  1.0, 0.0,
  0.0, 1.0,
  0.0, 0.0,
  1.0, 1.0,
  2.0, 0.0,
  2.0, 1.0,
  0.0, 2.0,
  1.0, 2.0,
  2.0, 2.0};

static double newMeshCoords[18] = {
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
#define __FUNCT__ "CreateTestMesh"
/*
  CreateTestMesh - Create a simple square mesh

         29
  30--19----22---28
    |\    |\    |
    1 1 5 2 2 7 2
    8  7  0  1  3
    | 4 \ | 6 \ |
    |    \|    \|
  31--11-32-15---27
    |\    |\    |
    | \ 1 1 1 3 1
    9  8  2  3  6
    | 0 \ | 2 \ |
    |    \|    \|
  24--10----14---26
         25
*/
extern "C" PetscErrorCode CreateTestMesh(Mesh mesh)
{
  ALE::Sieve *topology;
  ALE::Sieve *boundary;
  ALE::PreSieve *orientation;
  ALE::ClosureBundle *bundle;
  ALE::ClosureBundle *coordBundle;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  topology = new ALE::Sieve(comm);
  boundary = new ALE::Sieve(comm);
  orientation = new ALE::PreSieve(comm);
  bundle = new ALE::ClosureBundle(comm);
  coordBundle = new ALE::ClosureBundle(comm);
  topology->setVerbosity(11);
  boundary->setVerbosity(11);
  orientation->setVerbosity(11);
  if (rank == 0) {
    ALE::Point point;
    ALE::Point_set cone;
    ALE::Point boundaryPoint(0, 1);
    ALE::Point_set boundaryCone;

    /* Edges */
    point = ALE::Point(0, 8);
    cone.insert(ALE::Point(0, 25));
    cone.insert(ALE::Point(0, 31));
    topology->addCone(cone, point);
    point = ALE::Point(0, 9);
    cone.clear();
    cone.insert(ALE::Point(0, 31));
    cone.insert(ALE::Point(0, 24));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 10);
    cone.clear();
    cone.insert(ALE::Point(0, 24));
    cone.insert(ALE::Point(0, 25));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 11);
    cone.clear();
    cone.insert(ALE::Point(0, 32));
    cone.insert(ALE::Point(0, 31));
    topology->addCone(cone, point);
    point = ALE::Point(0, 12);
    cone.clear();
    cone.insert(ALE::Point(0, 25));
    cone.insert(ALE::Point(0, 32));
    topology->addCone(cone, point);
    point = ALE::Point(0, 13);
    cone.clear();
    cone.insert(ALE::Point(0, 26));
    cone.insert(ALE::Point(0, 32));
    topology->addCone(cone, point);
    point = ALE::Point(0, 14);
    cone.clear();
    cone.insert(ALE::Point(0, 25));
    cone.insert(ALE::Point(0, 26));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 15);
    cone.clear();
    cone.insert(ALE::Point(0, 27));
    cone.insert(ALE::Point(0, 32));
    topology->addCone(cone, point);
    point = ALE::Point(0, 16);
    cone.clear();
    cone.insert(ALE::Point(0, 26));
    cone.insert(ALE::Point(0, 27));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 17);
    cone.clear();
    cone.insert(ALE::Point(0, 32));
    cone.insert(ALE::Point(0, 30));
    topology->addCone(cone, point);
    point = ALE::Point(0, 18);
    cone.clear();
    cone.insert(ALE::Point(0, 30));
    cone.insert(ALE::Point(0, 31));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 19);
    cone.clear();
    cone.insert(ALE::Point(0, 29));
    cone.insert(ALE::Point(0, 30));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 20);
    cone.clear();
    cone.insert(ALE::Point(0, 32));
    cone.insert(ALE::Point(0, 29));
    topology->addCone(cone, point);
    point = ALE::Point(0, 21);
    cone.clear();
    cone.insert(ALE::Point(0, 27));
    cone.insert(ALE::Point(0, 29));
    topology->addCone(cone, point);
    point = ALE::Point(0, 22);
    cone.clear();
    cone.insert(ALE::Point(0, 28));
    cone.insert(ALE::Point(0, 29));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    point = ALE::Point(0, 23);
    cone.clear();
    cone.insert(ALE::Point(0, 27));
    cone.insert(ALE::Point(0, 28));
    boundaryCone.insert(point);
    boundaryCone.insert(cone.begin(), cone.end());
    topology->addCone(cone, point);
    /* Faces */
    point = ALE::Point(0, 0);
    cone.clear();
    cone.insert(ALE::Point(0, 8));
    cone.insert(ALE::Point(0, 9));
    cone.insert(ALE::Point(0, 10));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 24));
    cone.insert(ALE::Point(0, 10));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 1);
    cone.clear();
    cone.insert(ALE::Point(0, 8));
    cone.insert(ALE::Point(0, 12));
    cone.insert(ALE::Point(0, 11));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 32));
    cone.insert(ALE::Point(0, 11));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 2);
    cone.clear();
    cone.insert(ALE::Point(0, 13));
    cone.insert(ALE::Point(0, 12));
    cone.insert(ALE::Point(0, 14));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 25));
    cone.insert(ALE::Point(0, 14));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 3);
    cone.clear();
    cone.insert(ALE::Point(0, 13));
    cone.insert(ALE::Point(0, 16));
    cone.insert(ALE::Point(0, 15));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 27));
    cone.insert(ALE::Point(0, 15));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 4);
    cone.clear();
    cone.insert(ALE::Point(0, 17));
    cone.insert(ALE::Point(0, 18));
    cone.insert(ALE::Point(0, 11));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 31));
    cone.insert(ALE::Point(0, 11));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 5);
    cone.clear();
    cone.insert(ALE::Point(0, 17));
    cone.insert(ALE::Point(0, 20));
    cone.insert(ALE::Point(0, 19));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 29));
    cone.insert(ALE::Point(0, 19));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 6);
    cone.clear();
    cone.insert(ALE::Point(0, 21));
    cone.insert(ALE::Point(0, 20));
    cone.insert(ALE::Point(0, 15));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 32));
    cone.insert(ALE::Point(0, 15));
    cone.insert(point);
    orientation->addCone(cone, point);
    point = ALE::Point(0, 7);
    cone.clear();
    cone.insert(ALE::Point(0, 21));
    cone.insert(ALE::Point(0, 23));
    cone.insert(ALE::Point(0, 22));
    topology->addCone(cone, point);
    cone.clear();
    cone.insert(ALE::Point(0, 28));
    cone.insert(ALE::Point(0, 22));
    cone.insert(point);
    orientation->addCone(cone, point);

    boundary->addCone(boundaryCone, boundaryPoint);
  }
  topology->view("Simple mesh topology");
  //ALE::Stack completionStack = topology->coneCompletion(ALE::completionTypePoint, ALE::footprintTypeNone, NULL);
  /* Create orderings */
  bundle->setTopology(topology);
  bundle->setFiberDimensionByDepth(0, 1);
  //bundle->computeGlobalIndices();
  coordBundle->setTopology(topology);
  coordBundle->setFiberDimensionByDepth(0, 2);
  //coordBundle->computeOverlapIndices();
  ierr = MeshSetTopology(mesh, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetBoundary(mesh, (void *) boundary);CHKERRQ(ierr);
  ierr = MeshSetOrientation(mesh, (void *) orientation);CHKERRQ(ierr);
  ierr = MeshSetBundle(mesh, (void *) bundle);CHKERRQ(ierr);
  ierr = MeshSetCoordinateBundle(mesh, (void *) coordBundle);CHKERRQ(ierr);
  /* Finish old-style DM construction */
  ALE::Point_set empty;

  ierr = MeshSetGhosts(mesh, 1, bundle->getBundleDimension(empty), 0, NULL);
  /* Create coordinates */
  Vec coordinates;
  ALE::Point_set vertices = topology->depthStratum(0);
  PetscInt indices[2];

  //ierr = MeshCreateGlobalVector(mesh, &coordinates);CHKERRQ(ierr);
  ierr  = VecCreateGhostBlock(comm,1,coordBundle->getBundleDimension(empty),PETSC_DETERMINE,0,NULL,&coordinates);CHKERRQ(ierr);
  printf("Making coordinates\n");
  for(ALE::Point_set::iterator vertex_itor = vertices.begin(); vertex_itor != vertices.end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;

    printf("  vertex(%d, %d)\n", v.prefix, v.index);
    ierr = ExpandSetIntervals(coordBundle->getFiberIndices(ALE::Point_set(v), empty)->cap(), indices);CHKERRQ(ierr);
    //ierr = ExpandSetIntervals(coordBundle->getOverlapFiberIndices(v, rank).cap(), indices);CHKERRQ(ierr);
    ierr = VecSetValues(coordinates, 2, indices, &newMeshCoords[(v.index - 24)*2], INSERT_VALUES);CHKERRQ(ierr);
    printf("  (%d, %d) = (%g, %g)\n", indices[0], indices[1], newMeshCoords[(v.index - 24)*2], newMeshCoords[(v.index - 24)*2 + 1]);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(mesh, coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Simplicializer"
PetscErrorCode Simplicializer(MPI_Comm comm, PetscInt numFaces, PetscInt *faces, PetscInt numVertices, PetscScalar *vertices, PetscInt numBoundaryVertices, PetscInt *boundaryVertices, Mesh *mesh)
{
  Mesh m;
  ALE::Sieve *topology = new ALE::Sieve(comm);
  ALE::Sieve *boundary = new ALE::Sieve(comm);
  ALE::PreSieve *orientation = new ALE::PreSieve(comm);
  ALE::ClosureBundle *coordBundle = new ALE::ClosureBundle(comm);
  PetscInt curEdge = numFaces+numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, &m);CHKERRQ(ierr);
  topology->setVerbosity(11);
  orientation->setVerbosity(11);
  boundary->setVerbosity(11);
  coordBundle->setVerbosity(11);
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
        printf("  depth: %d\n", topology->depth(edge));
        printf("  depth: %d\n", topology->depth(vS));
        printf("  depth: %d\n", topology->depth(vE));
        printf("  maxDepth: %d\n", topology->maxDepth(endpoints));
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
  topology->view("Simplicializer topology");
  ierr = MeshSetTopology(m, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(m, (void *) orientation);CHKERRQ(ierr);
  {
    ALE::Point_set cone;
    ALE::Point boundaryPoint(0, 1);

    /* Should also put in boundary edges */
    for(int v = 0; v < numBoundaryVertices; v++) {
      ALE::Point vertex = ALE::Point(0, boundaryVertices[v] + numFaces);

      cone.insert(vertex);
    }
    boundary->addCone(cone, boundaryPoint);
  }
  ierr = MeshSetBoundary(m, (void *) boundary);CHKERRQ(ierr);
  *mesh = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTestMesh2"
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
extern "C" PetscErrorCode CreateTestMesh2(MPI_Comm comm)
{
  Mesh            mesh;
  PetscInt        faces[24] = {
    0, 1, 7,
    8, 7, 1,
    1, 2, 8,
    3, 8, 2,
    7, 8, 6,
    5, 6, 8,
    8, 3, 5,
    4, 5, 3};
  PetscScalar     vertices[18] = {
    0.0, 0.0,
    1.0, 0.0,
    2.0, 0.0,
    2.0, 1.0,
    2.0, 2.0,
    1.0, 2.0,
    0.0, 2.0,
    0.0, 1.0,
    1.0, 1.0};
  PetscInt        boundaryVertices[8] = {
    0, 1, 2, 3, 4, 5, 6, 7};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Simplicializer(comm, 8, faces, 9, vertices, 8, boundaryVertices, &mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
extern "C" PetscErrorCode ElementGeometry(ALE::ClosureBundle *coordBundle, ALE::PreSieve *orientation, PetscScalar *coords, ALE::Point e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  static PetscInt  coordSize = 0;
  static PetscInt *coordinateIndices = NULL;
  ALE::Point_set   empty;
  ALE::Obj<ALE::Point_array> coordinateIntervals = coordBundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> coordinateIntervals = coordBundle->getOverlapOrderedIndices(orientation->cone(e), empty);
  PetscInt         numCoordinateIndices = 0;
  //PetscScalar     *coords = meshCoords;
  PetscReal        det, invDet;
  PetscInt         c = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = coordinateIntervals->begin(); i_itor != coordinateIntervals->end(); i_itor++) {
    numCoordinateIndices += (*i_itor).index;
  }
  if (coordSize && (coordSize != numCoordinateIndices)) {
    ierr = PetscFree(coordinateIndices); CHKERRQ(ierr);
    coordinateIndices = NULL;
  }
  if (!coordinateIndices) {
    coordSize = numCoordinateIndices;
    ierr = PetscMalloc(coordSize * sizeof(PetscInt), &coordinateIndices); CHKERRQ(ierr);
  }
  for(ALE::Point_array::iterator i_itor = coordinateIntervals->begin(); i_itor != coordinateIntervals->end(); i_itor++) {
    printf("coordinateIndices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(coordinateIntervals, coordinateIndices); CHKERRQ(ierr);
  for(int i = 0; i < numCoordinateIndices; i++) {
    printf("coordinateIndices[%d] = %d\n", i, coordinateIndices[i]);
  }
  if (v0) {
    v0[0] = coords[coordinateIndices[0*2+0]];
    v0[1] = coords[coordinateIndices[0*2+1]];
  }
  if (J) {
    J[0] = 0.5*(coords[coordinateIndices[1*2+0]] - coords[coordinateIndices[0*2+0]]);
    J[1] = 0.5*(coords[coordinateIndices[2*2+0]] - coords[coordinateIndices[0*2+0]]);
    J[2] = 0.5*(coords[coordinateIndices[1*2+1]] - coords[coordinateIndices[0*2+1]]);
    J[3] = 0.5*(coords[coordinateIndices[2*2+1]] - coords[coordinateIndices[0*2+1]]);
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
extern "C" PetscErrorCode ComputeBlock(DMMG dmmg, Vec u, Vec r, ALE::Point_set block)
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
  PetscInt            numElementIndices;
  PetscInt           *elementIndices = NULL;
  PetscScalar        *array;
  PetscReal           v0[2];
  PetscReal           Jac[4], Jinv[4];
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           linearVec[NUM_BASIS_FUNCTIONS];
  PetscReal           field[NUM_BASIS_FUNCTIONS];
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
    /* Field */
    ALE::Point_array elementIntervals = bundle->getClosureIndices(orientation->cone(e), empty);

    if (!elementIndices) {
      numElementIndices = bundle->getBundleDimension(e);
      ierr = PetscMalloc(numElementIndices * sizeof(PetscInt), &elementIndices); CHKERRQ(ierr);
    }
    ierr = ExpandIntervals(elementIntervals, elementIndices); CHKERRQ(ierr);
    for(int i = 0; i < numElementIndices; i++) {
      printf("elementIndices[%d] = %d\n", i, elementIndices[i]);
    }
    for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
      field[f] = array[elementIndices[f]];
    }
    /* Do the integration */
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
    ierr = VecSetValues(r, numElementIndices, elementIndices, elementVec, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(u, &array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
extern "C" PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
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
  PetscInt            numElementIndices;
  PetscInt           *elementIndices = NULL;
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
    ALE::Point_array elementIntervals = bundle->getClosureIndices(orientation->cone(e), empty);

    if (!elementIndices) {
      numElementIndices = bundle->getBundleDimension(e);
      ierr = PetscMalloc(numElementIndices * sizeof(PetscInt), &elementIndices); CHKERRQ(ierr);
    }
    ierr = ExpandIntervals(elementIntervals, elementIndices); CHKERRQ(ierr);
    for(int i = 0; i < numElementIndices; i++) {
      printf("elementIndices[%d] = %d\n", i, elementIndices[i]);
    }
    ierr = VecSetValues(b, numElementIndices, elementIndices, elementVec, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(elementIndices);CHKERRQ(ierr);
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
extern "C" PetscErrorCode ComputeJacobian(DMMG dmmg, Mat J, Mat jac)
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
  PetscInt            e, f, g, q;
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
    ALE::Point_array elementIntervals = bundle->getClosureIndices(orientation->cone(e), empty);
    PetscInt idx = 0;

    if (!elementIndices) {
      numElementIndices = bundle->getBundleDimension(e);
      ierr = PetscMalloc(numElementIndices * sizeof(PetscInt), &elementIndices); CHKERRQ(ierr);
    }
    ierr = ExpandIntervals(elementIntervals, elementIndices); CHKERRQ(ierr);
    for(int i = 0; i < numElementIndices; i++) {
      printf("elementIndices[%d] = %d\n", i, elementIndices[i]);
    }
    ierr = MatSetValues(jac, numElementIndices, elementIndices, numElementIndices, elementIndices, elementMat, ADD_VALUES);CHKERRQ(ierr);
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
    int b = 0;

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
