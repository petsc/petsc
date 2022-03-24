static char help[] = "Demonstration of creating and viewing DMFields objects.\n\n";

#include <petscdmfield.h>
#include <petscdmplex.h>
#include <petscdmda.h>

static PetscErrorCode ViewResults(PetscViewer viewer, PetscInt N, PetscInt dim, PetscScalar *B, PetscScalar *D, PetscScalar *H, PetscReal *rB, PetscReal *rD, PetscReal *rH)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"B:\n"));
  CHKERRQ(PetscScalarView(N,B,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"D:\n"));
  CHKERRQ(PetscScalarView(N*dim,D,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"H:\n"));
  CHKERRQ(PetscScalarView(N*dim*dim,H,viewer));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"rB:\n"));
  CHKERRQ(PetscRealView(N,rB,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"rD:\n"));
  CHKERRQ(PetscRealView(N*dim,rD,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"rH:\n"));
  CHKERRQ(PetscRealView(N*dim*dim,rH,viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluate(DMField field, PetscInt n, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  Vec            points;
  PetscScalar    *array;
  PetscViewer    viewer;
  MPI_Comm       comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  CHKERRQ(DMFieldGetNumComponents(field,&nc));
  CHKERRQ(DMFieldGetDM(field,&dm));
  CHKERRQ(DMGetDimension(dm,&dim));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)field),n * dim,PETSC_DETERMINE,&points));
  CHKERRQ(VecSetBlockSize(points,dim));
  CHKERRQ(VecGetArray(points,&array));
  for (i = 0; i < n * dim; i++) CHKERRQ(PetscRandomGetValue(rand,&array[i]));
  CHKERRQ(VecRestoreArray(points,&array));
  CHKERRQ(PetscMalloc6(n*nc,&B,n*nc,&rB,n*nc*dim,&D,n*nc*dim,&rD,n*nc*dim*dim,&H,n*nc*dim*dim,&rH));
  CHKERRQ(DMFieldEvaluate(field,points,PETSC_SCALAR,B,D,H));
  CHKERRQ(DMFieldEvaluate(field,points,PETSC_REAL,rB,rD,rH));
  viewer = PETSC_VIEWER_STDOUT_(comm);

  CHKERRQ(PetscObjectSetName((PetscObject)points,"Test Points"));
  CHKERRQ(VecView(points,viewer));
  CHKERRQ(ViewResults(viewer,n*nc,dim,B,D,H,rB,rD,rH));

  CHKERRQ(PetscFree6(B,rB,D,rD,H,rH));
  CHKERRQ(VecDestroy(&points));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluateFE(DMField field, PetscInt n, PetscInt cStart, PetscInt cEnd, PetscQuadrature quad, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc, nq;
  PetscInt       N;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  PetscInt       *cells;
  IS             cellIS;
  PetscViewer    viewer;
  MPI_Comm       comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  CHKERRQ(DMFieldGetNumComponents(field,&nc));
  CHKERRQ(DMFieldGetDM(field,&dm));
  CHKERRQ(DMGetDimension(dm,&dim));
  CHKERRQ(PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd));
  CHKERRQ(PetscMalloc1(n,&cells));
  for (i = 0; i < n; i++) {
    PetscReal rc;

    CHKERRQ(PetscRandomGetValueReal(rand,&rc));
    cells[i] = PetscFloorReal(rc);
  }
  CHKERRQ(ISCreateGeneral(comm,n,cells,PETSC_OWN_POINTER,&cellIS));
  CHKERRQ(PetscObjectSetName((PetscObject)cellIS,"FE Test Cells"));
  CHKERRQ(PetscQuadratureGetData(quad,NULL,NULL,&nq,NULL,NULL));
  N    = n * nq * nc;
  CHKERRQ(PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH));
  CHKERRQ(DMFieldEvaluateFE(field,cellIS,quad,PETSC_SCALAR,B,D,H));
  CHKERRQ(DMFieldEvaluateFE(field,cellIS,quad,PETSC_REAL,rB,rD,rH));
  viewer = PETSC_VIEWER_STDOUT_(comm);

  CHKERRQ(PetscObjectSetName((PetscObject)quad,"Test quadrature"));
  CHKERRQ(PetscQuadratureView(quad,viewer));
  CHKERRQ(ISView(cellIS,viewer));
  CHKERRQ(ViewResults(viewer,N,dim,B,D,H,rB,rD,rH));

  CHKERRQ(PetscFree6(B,rB,D,rD,H,rH));
  CHKERRQ(ISDestroy(&cellIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEvaluateFV(DMField field, PetscInt n, PetscInt cStart, PetscInt cEnd, PetscRandom rand)
{
  DM             dm;
  PetscInt       dim, i, nc;
  PetscInt       N;
  PetscScalar    *B, *D, *H;
  PetscReal      *rB, *rD, *rH;
  PetscInt       *cells;
  IS             cellIS;
  PetscViewer    viewer;
  MPI_Comm       comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  CHKERRQ(DMFieldGetNumComponents(field,&nc));
  CHKERRQ(DMFieldGetDM(field,&dm));
  CHKERRQ(DMGetDimension(dm,&dim));
  CHKERRQ(PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd));
  CHKERRQ(PetscMalloc1(n,&cells));
  for (i = 0; i < n; i++) {
    PetscReal rc;

    CHKERRQ(PetscRandomGetValueReal(rand,&rc));
    cells[i] = PetscFloorReal(rc);
  }
  CHKERRQ(ISCreateGeneral(comm,n,cells,PETSC_OWN_POINTER,&cellIS));
  CHKERRQ(PetscObjectSetName((PetscObject)cellIS,"FV Test Cells"));
  N    = n * nc;
  CHKERRQ(PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH));
  CHKERRQ(DMFieldEvaluateFV(field,cellIS,PETSC_SCALAR,B,D,H));
  CHKERRQ(DMFieldEvaluateFV(field,cellIS,PETSC_REAL,rB,rD,rH));
  viewer = PETSC_VIEWER_STDOUT_(comm);

  CHKERRQ(ISView(cellIS,viewer));
  CHKERRQ(ViewResults(viewer,N,dim,B,D,H,rB,rD,rH));

  CHKERRQ(PetscFree6(B,rB,D,rD,H,rH));
  CHKERRQ(ISDestroy(&cellIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode radiusSquared(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt       i;
  PetscReal      r2 = 0.;

  PetscFunctionBegin;
  for (i = 0; i < dim; i++) {r2 += PetscSqr(x[i]);}
  for (i = 0; i < Nf; i++) {
    u[i] = (i + 1) * r2;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestShellEvaluate(DMField field, Vec points, PetscDataType type, void *B, void *D, void *H)
{
  Vec                ctxVec = NULL;
  const PetscScalar *mult;
  PetscInt           dim;
  const PetscScalar *x;
  PetscInt           Nc, n, i, j, k, l;

  PetscFunctionBegin;
  CHKERRQ(DMFieldGetNumComponents(field, &Nc));
  CHKERRQ(DMFieldShellGetContext(field, &ctxVec));
  CHKERRQ(VecGetBlockSize(points, &dim));
  CHKERRQ(VecGetLocalSize(points, &n));
  n /= Nc;
  CHKERRQ(VecGetArrayRead(ctxVec, &mult));
  CHKERRQ(VecGetArrayRead(points, &x));
  for (i = 0; i < n; i++) {
    PetscReal r2 = 0.;

    for (j = 0; j < dim; j++) {r2 += PetscSqr(PetscRealPart(x[i * dim + j]));}
    for (j = 0; j < Nc; j++) {
      PetscReal m = PetscRealPart(mult[j]);
      if (B) {
        if (type == PETSC_SCALAR) {
          ((PetscScalar *)B)[i * Nc + j] = m * r2;
        } else {
          ((PetscReal *)B)[i * Nc + j] = m * r2;
        }
      }
      if (D) {
        if (type == PETSC_SCALAR) {
          for (k = 0; k < dim; k++) ((PetscScalar *)D)[(i * Nc + j) * dim + k] = 2. * m * x[i * dim + k];
        } else {
          for (k = 0; k < dim; k++) ((PetscReal   *)D)[(i * Nc + j) * dim + k] = 2. * m * PetscRealPart(x[i * dim + k]);
        }
      }
      if (H) {
        if (type == PETSC_SCALAR) {
          for (k = 0; k < dim; k++) for (l = 0; l < dim; l++) ((PetscScalar *)H)[((i * Nc + j) * dim + k) * dim + l] = (k == l) ? 2. * m : 0.;
        } else {
          for (k = 0; k < dim; k++) for (l = 0; l < dim; l++) ((PetscReal   *)H)[((i * Nc + j) * dim + k) * dim + l] = (k == l) ? 2. * m : 0.;
        }
      }
    }
  }
  CHKERRQ(VecRestoreArrayRead(points, &x));
  CHKERRQ(VecRestoreArrayRead(ctxVec, &mult));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestShellDestroy(DMField field)
{
  Vec                ctxVec = NULL;

  PetscFunctionBegin;
  CHKERRQ(DMFieldShellGetContext(field, &ctxVec));
  CHKERRQ(VecDestroy(&ctxVec));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm = NULL;
  MPI_Comm        comm;
  char            type[256] = DMPLEX;
  PetscBool       isda, isplex;
  PetscInt        dim = 2;
  DMField         field = NULL;
  PetscInt        nc = 1;
  PetscInt        cStart = -1, cEnd = -1;
  PetscRandom     rand;
  PetscQuadrature quad = NULL;
  PetscInt        pointsPerEdge = 2;
  PetscInt        numPoint = 0, numFE = 0, numFV = 0;
  PetscBool       testShell = PETSC_FALSE;
  PetscErrorCode  ierr;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMField Tutorial Options", "DM");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-dm_type","DM implementation on which to define field","ex1.c",DMList,type,type,256,NULL));
  CHKERRQ(PetscOptionsRangeInt("-dim","DM intrinsic dimension", "ex1.c", dim, &dim, NULL,1,3));
  CHKERRQ(PetscOptionsBoundedInt("-num_components","Number of components in field", "ex1.c", nc, &nc, NULL,1));
  CHKERRQ(PetscOptionsBoundedInt("-num_quad_points","Number of quadrature points per dimension", "ex1.c", pointsPerEdge, &pointsPerEdge, NULL,1));
  CHKERRQ(PetscOptionsBoundedInt("-num_point_tests", "Number of test points for DMFieldEvaluate()", "ex1.c", numPoint, &numPoint, NULL,0));
  CHKERRQ(PetscOptionsBoundedInt("-num_fe_tests", "Number of test cells for DMFieldEvaluateFE()", "ex1.c", numFE, &numFE, NULL,0));
  CHKERRQ(PetscOptionsBoundedInt("-num_fv_tests", "Number of test cells for DMFieldEvaluateFV()", "ex1.c", numFV, &numFV, NULL,0));
  CHKERRQ(PetscOptionsBool("-test_shell", "Test the DMFIELDSHELL implementation of DMField", "ex1.c", testShell, &testShell, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscCheckFalse(dim > 3,comm,PETSC_ERR_ARG_OUTOFRANGE,"This examples works for dim <= 3, not %D",dim);
  CHKERRQ(PetscStrncmp(type,DMPLEX,256,&isplex));
  CHKERRQ(PetscStrncmp(type,DMDA,256,&isda));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  if (isplex) {
    PetscInt  overlap = 0;
    Vec       fieldvec;
    PetscInt  cells[3] = {3,3,3};
    PetscBool simplex;
    PetscFE   fe;

    ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Options", "DM");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsBoundedInt("-overlap","DMPlex parallel overlap","ex1.c",overlap,&overlap,NULL,0));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (0) {
      CHKERRQ(DMPlexCreateBoxMesh(comm,2,PETSC_TRUE,cells,NULL,NULL,NULL,PETSC_TRUE,&dm));
    } else {
      CHKERRQ(DMCreate(comm, &dm));
      CHKERRQ(DMSetType(dm, DMPLEX));
      CHKERRQ(DMSetFromOptions(dm));
      CHKMEMQ;
    }
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexIsSimplex(dm, &simplex));
    if (simplex) {
      CHKERRQ(PetscDTStroudConicalQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad));
    } else {
      CHKERRQ(PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad));
    }
    CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
    if (testShell) {
      Vec ctxVec;
      PetscInt i;
      PetscScalar *array;

      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, nc, &ctxVec));
      CHKERRQ(VecSetUp(ctxVec));
      CHKERRQ(VecGetArray(ctxVec,&array));
      for (i = 0; i < nc; i++) array[i] = i + 1.;
      CHKERRQ(VecRestoreArray(ctxVec,&array));
      CHKERRQ(DMFieldCreateShell(dm, nc, DMFIELD_VERTEX, (void *) ctxVec, &field));
      CHKERRQ(DMFieldShellSetEvaluate(field, TestShellEvaluate));
      CHKERRQ(DMFieldShellSetDestroy(field, TestShellDestroy));
    } else {
      CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF,dim,nc,simplex,NULL,PETSC_DEFAULT,&fe));
      CHKERRQ(PetscFESetName(fe,"MyPetscFE"));
      CHKERRQ(DMSetField(dm,0,NULL,(PetscObject)fe));
      CHKERRQ(PetscFEDestroy(&fe));
      CHKERRQ(DMCreateDS(dm));
      CHKERRQ(DMCreateLocalVector(dm,&fieldvec));
      {
        PetscErrorCode (*func[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt, PetscScalar *,void *);
        void            *ctxs[1];

        func[0] = radiusSquared;
        ctxs[0] = NULL;

        CHKERRQ(DMProjectFunctionLocal(dm,0.0,func,ctxs,INSERT_ALL_VALUES,fieldvec));
      }
      CHKERRQ(DMFieldCreateDS(dm,0,fieldvec,&field));
      CHKERRQ(VecDestroy(&fieldvec));
    }
  } else if (isda) {
    PetscInt       i;
    PetscScalar    *cv;

    switch (dim) {
    case 1:
      CHKERRQ(DMDACreate1d(comm, DM_BOUNDARY_NONE, 3, 1, 1, NULL, &dm));
      break;
    case 2:
      CHKERRQ(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dm));
      break;
    default:
      CHKERRQ(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, NULL, &dm));
      break;
    }
    CHKERRQ(DMSetUp(dm));
    CHKERRQ(DMDAGetHeightStratum(dm,0,&cStart,&cEnd));
    CHKERRQ(PetscMalloc1(nc * (1 << dim),&cv));
    for (i = 0; i < nc * (1 << dim); i++) {
      PetscReal rv;

      CHKERRQ(PetscRandomGetValueReal(rand,&rv));
      cv[i] = rv;
    }
    CHKERRQ(DMFieldCreateDA(dm,nc,cv,&field));
    CHKERRQ(PetscFree(cv));
    CHKERRQ(PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad));
  } else SETERRQ(comm,PETSC_ERR_SUP,"This test does not run for DM type %s",type);

  CHKERRQ(PetscObjectSetName((PetscObject)dm,"mesh"));
  CHKERRQ(DMViewFromOptions(dm,NULL,"-dm_view"));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscObjectSetName((PetscObject)field,"field"));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)field,NULL,"-dmfield_view"));
  if (numPoint) CHKERRQ(TestEvaluate(field,numPoint,rand));
  if (numFE) CHKERRQ(TestEvaluateFE(field,numFE,cStart,cEnd,quad,rand));
  if (numFV) CHKERRQ(TestEvaluateFV(field,numFV,cStart,cEnd,rand));
  CHKERRQ(DMFieldDestroy(&field));
  CHKERRQ(PetscQuadratureDestroy(&quad));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: da
    requires: !complex
    args: -dm_type da -dim 2 -num_components 2 -num_point_tests 2 -num_fe_tests 2 -num_fv_tests 2 -dmfield_view

  test:
    suffix: da_1
    requires: !complex
    args: -dm_type da -dim 1  -num_fe_tests 2

  test:
    suffix: da_2
    requires: !complex
    args: -dm_type da -dim 2  -num_fe_tests 2

  test:
    suffix: da_3
    requires: !complex
    args: -dm_type da -dim 3  -num_fe_tests 2

  test:
    suffix: ds
    requires: !complex triangle
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3 -num_components 2 -num_point_tests 2 -num_fe_tests 2 -num_fv_tests 2 -dmfield_view -petscspace_degree 2 -num_quad_points 1

  test:
    suffix: ds_simplex_0
    requires: !complex triangle
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_degree 0

  test:
    suffix: ds_simplex_1
    requires: !complex triangle
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_degree 1

  test:
    suffix: ds_simplex_2
    requires: !complex triangle
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_degree 2

  test:
    suffix: ds_tensor_2_0
    requires: !complex
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 0 -dm_plex_simplex 0

  test:
    suffix: ds_tensor_2_1
    requires: !complex
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 1 -dm_plex_simplex 0

  test:
    suffix: ds_tensor_2_2
    requires: !complex
    args: -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 2 -dm_plex_simplex 0

  test:
    suffix: ds_tensor_3_0
    requires: !complex
    args: -dm_type plex -dm_plex_dim 3 -dm_plex_box_faces 3,3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 0 -dm_plex_simplex 0

  test:
    suffix: ds_tensor_3_1
    requires: !complex
    args: -dm_type plex -dm_plex_dim 3 -dm_plex_box_faces 3,3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 1 -dm_plex_simplex 0

  test:
    suffix: ds_tensor_3_2
    requires: !complex
    args: -dm_type plex -dm_plex_dim 3 -dm_plex_box_faces 3,3,3  -num_fe_tests 2  -petscspace_poly_tensor 1 -petscspace_degree 2 -dm_plex_simplex 0

  test:
    suffix: shell
    requires: !complex triangle
    args: -dm_coord_space 0 -dm_type plex -dm_plex_dim 2 -dm_plex_box_faces 3,3 -num_components 2 -num_point_tests 2 -num_fe_tests 2 -num_fv_tests 2 -dmfield_view -num_quad_points 1 -test_shell

TEST*/
