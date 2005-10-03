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

static double meshCoords[48] = {
  0.0, 0.0,
  1.0, 0.0,
  0.0, 1.0,
  1.0, 1.0,
  0.0, 1.0,
  1.0, 0.0,

  1.0, 0.0,
  2.0, 0.0,
  1.0, 1.0,
  2.0, 1.0,
  1.0, 1.0,
  2.0, 0.0,

  0.0, 1.0,
  1.0, 1.0,
  0.0, 2.0,
  1.0, 2.0,
  0.0, 2.0,
  1.0, 1.0,

  1.0, 1.0,
  2.0, 1.0,
  1.0, 2.0,
  2.0, 2.0,
  1.0, 2.0,
  2.0, 1.0};

static PetscInt meshIndices[24] = {
  0, 1, 7,
  8, 7, 1,
  1, 2, 8,
  3, 7, 2,
  7, 8, 6,
  5, 6, 8,
  8, 3, 5,
  4, 5, 3};

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
  ALE::ClosureBundle *bundle;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  topology = new ALE::Sieve(comm);
  boundary = new ALE::Sieve(comm);
  bundle = new ALE::ClosureBundle(comm);
  topology->setVerbosity(11);
  boundary->setVerbosity(11);
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
    point = ALE::Point(0, 1);
    cone.clear();
    cone.insert(ALE::Point(0, 8));
    cone.insert(ALE::Point(0, 12));
    cone.insert(ALE::Point(0, 11));
    topology->addCone(cone, point);
    point = ALE::Point(0, 2);
    cone.clear();
    cone.insert(ALE::Point(0, 13));
    cone.insert(ALE::Point(0, 12));
    cone.insert(ALE::Point(0, 14));
    topology->addCone(cone, point);
    point = ALE::Point(0, 3);
    cone.clear();
    cone.insert(ALE::Point(0, 13));
    cone.insert(ALE::Point(0, 16));
    cone.insert(ALE::Point(0, 15));
    topology->addCone(cone, point);
    point = ALE::Point(0, 4);
    cone.clear();
    cone.insert(ALE::Point(0, 17));
    cone.insert(ALE::Point(0, 18));
    cone.insert(ALE::Point(0, 11));
    topology->addCone(cone, point);
    point = ALE::Point(0, 5);
    cone.clear();
    cone.insert(ALE::Point(0, 17));
    cone.insert(ALE::Point(0, 20));
    cone.insert(ALE::Point(0, 19));
    topology->addCone(cone, point);
    point = ALE::Point(0, 6);
    cone.clear();
    cone.insert(ALE::Point(0, 21));
    cone.insert(ALE::Point(0, 20));
    cone.insert(ALE::Point(0, 15));
    topology->addCone(cone, point);
    point = ALE::Point(0, 7);
    cone.clear();
    cone.insert(ALE::Point(0, 21));
    cone.insert(ALE::Point(0, 23));
    cone.insert(ALE::Point(0, 22));
    topology->addCone(cone, point);

    boundary->addCone(boundaryCone, boundaryPoint);
  }
  topology->view("Simple mesh topology");
  //ALE::Stack completionStack = topology->coneCompletion(ALE::completionTypePoint, ALE::footprintTypeNone, NULL);
  ierr = MeshSetTopology(mesh, (void *) topology);CHKERRQ(ierr);
  ierr = MeshSetBoundary(mesh, (void *) boundary);CHKERRQ(ierr);
  ierr = MeshSetBundle(mesh, (void *) bundle);CHKERRQ(ierr);
  //
  ALE::Obj<ALE::Point_set> stratum = topology->depthStratum(0);
  for(ALE::Point_set::iterator s_itor = stratum->begin(); s_itor != stratum->end(); s_itor++) {
    printf("prefix: %d index: %d\n", (*s_itor).prefix, (*s_itor).index);
  }
  // Should use the bundle here
  ALE::Point_set bottom;
  bundle->setTopology(topology);
  bundle->setFiberDimensionByDepth(0, 1);
  int dim = bundle->getBundleDimension(bottom);
  ierr = MeshSetGhosts(mesh, 1, dim, 0, NULL);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
extern "C" PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  Mesh               mesh = (Mesh) dmmg->dm;
  UserContext       *user = (UserContext *) dmmg->user;
  ALE::Sieve        *topology;
  ALE::ClosureBundle *bundle;
  ALE::Point_set     elements;
  ALE::Point_set     empty;
  PetscInt           numElementIndices;
  PetscInt          *elementIndices = NULL;
  PetscReal         *coords = meshCoords;
  PetscReal          elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal          Jac[4];
  PetscReal          xi, eta, x_q, y_q, detJ, funcValue;
  PetscInt           e, f, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;
    /* Element geometry */
    Jac[0] = 0.5*(coords[1*2+0] - coords[0*2+0]);
    Jac[1] = 0.5*(coords[2*2+0] - coords[0*2+0]);
    Jac[2] = 0.5*(coords[1*2+1] - coords[0*2+1]);
    Jac[3] = 0.5*(coords[2*2+1] - coords[0*2+1]);
    detJ = Jac[0]*Jac[3] - Jac[1]*Jac[2];
    printf("J = / %g %g \\\n    \\ %g %g /\n", Jac[0], Jac[1], Jac[2], Jac[3]);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + coords[0];
      y_q = Jac[2]*xi + Jac[3]*eta + coords[1];
      funcValue = PetscExpScalar(-(x_q*x_q)/user->nu)*PetscExpScalar(-(y_q*y_q)/user->nu);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        elementVec[f] += Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue*weights[q]*detJ;
      }
    }
    printf("elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);
    /* Assembly */
    ALE::Point_set elementIntervals = bundle->getBundleIndices(empty, ALE::Point_set(e));
    PetscInt idx = 0;

    if (!elementIndices) {
      numElementIndices = bundle->getBundleDimension(e);
      ierr = PetscMalloc(numElementIndices * sizeof(PetscInt), &elementIndices); CHKERRQ(ierr);
    }
    ierr = VecSetValues(b, numElementIndices, elementIndices, elementVec, ADD_VALUES);CHKERRQ(ierr);

    coords = coords + 6;
  }
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
#define __FUNCT__ "ComputeJacobian"
extern "C" PetscErrorCode ComputeJacobian(DMMG dmmg, Mat J, Mat jac)
{
  Mesh           mesh = (Mesh) dmmg->dm;
  UserContext   *user = (UserContext *) dmmg->user;
  ALE::Sieve    *topology;
  ALE::Sieve    *boundary;
  ALE::ClosureBundle *bundle;
  ALE::Point_set elements;
  ALE::Point_set empty;
  PetscInt       numElementIndices;
  PetscInt      *elementIndices = NULL;
  PetscReal     *coords = meshCoords;
  PetscReal      elementMat[NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS];
  PetscReal      Jac[4], Jinv[4], t_der[2], b_der[2];
  PetscReal      xi, eta, x_q, y_q, detJ, detJinv, rho;
  PetscInt       e, f, g, q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetBoundary(mesh, (void **) &boundary);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  topology->view("In ComputeJacobian");
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;
    /* Element geometry */
    Jac[0] = 0.5*(coords[1*2+0] - coords[0*2+0]);
    Jac[1] = 0.5*(coords[2*2+0] - coords[0*2+0]);
    Jac[2] = 0.5*(coords[1*2+1] - coords[0*2+1]);
    Jac[3] = 0.5*(coords[2*2+1] - coords[0*2+1]);
    detJ = Jac[0]*Jac[3] - Jac[1]*Jac[2];
    detJinv = 1.0/detJ;
    Jinv[0] =  detJinv*Jac[3];
    Jinv[1] = -detJinv*Jac[1];
    Jinv[2] = -detJinv*Jac[2];
    Jinv[3] =  detJinv*Jac[0];
    printf("J = / %g %g \\\n    \\ %g %g /\n", Jac[0], Jac[1], Jac[2], Jac[3]);
    printf("Jinv = / %g %g \\\n       \\ %g %g /\n", Jinv[0], Jinv[1], Jinv[2], Jinv[3]);
    /* Element integral */
    ierr = PetscMemzero(elementMat, NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + coords[0];
      y_q = Jac[2]*xi + Jac[3]*eta + coords[1];
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
    ALE::Point_set elementIntervals = bundle->getBundleIndices(empty, ALE::Point_set(e));
    PetscInt idx = 0;

    if (!elementIndices) {
      numElementIndices = bundle->getBundleDimension(e);
      ierr = PetscMalloc(numElementIndices * sizeof(PetscInt), &elementIndices); CHKERRQ(ierr);
    }
    for(ALE::Point_set::iterator e_itor = elementIntervals.begin(); e_itor != elementIntervals.end(); e_itor++) {
      for(int i = 0; i < (*e_itor).index; i++) {
        elementIndices[idx++] = (*e_itor).prefix + i;
      }
    }
    ierr = MatSetValues(jac, numElementIndices, elementIndices, numElementIndices, elementIndices, elementMat, ADD_VALUES);CHKERRQ(ierr);

    coords = coords + 6;
  }
  ierr = PetscFree(elementIndices);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    /* Zero out BC rows */
    ALE::Point id(0, 1);
    ALE::Point_set boundaryElements = boundary->cone(id);
    int numBoundaryIndices = bundle->getFiberDimension(boundaryElements);
    ALE::Point_set boundaryIntervals = bundle->getFiberIndices(empty, boundaryElements);
    PetscInt *boundaryIndices;
    int b = 0;

    ierr = PetscMalloc(numBoundaryIndices * sizeof(PetscInt), &boundaryIndices); CHKERRQ(ierr);
    for(ALE::Point_set::iterator b_itor = boundaryIntervals.begin(); b_itor != boundaryIntervals.end(); b_itor++) {
      for(int i = 0; i < (*b_itor).index; i++) {
        boundaryIndices[b++] = (*b_itor).prefix + i;
      }
    }
    ierr = MatZeroRows(jac, numBoundaryIndices, boundaryIndices, 1.0);CHKERRQ(ierr);
    ierr = PetscFree(boundaryIndices);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
