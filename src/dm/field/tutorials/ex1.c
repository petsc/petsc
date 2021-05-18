static char help[] = "Demonstration of creating and viewing DMFields objects.\n\n";

#include <petscdmfield.h>
#include <petscdmplex.h>
#include <petscdmda.h>

static PetscErrorCode ViewResults(PetscViewer viewer, PetscInt N, PetscInt dim, PetscScalar *B, PetscScalar *D, PetscScalar *H, PetscReal *rB, PetscReal *rD, PetscReal *rH)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"B:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N,B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim,D,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"H:\n");CHKERRQ(ierr);
  ierr = PetscScalarView(N*dim*dim,H,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"rB:\n");CHKERRQ(ierr);
  ierr = PetscRealView(N,rB,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rD:\n");CHKERRQ(ierr);
  ierr = PetscRealView(N*dim,rD,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rH:\n");CHKERRQ(ierr);
  ierr = PetscRealView(N*dim*dim,rH,viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)field),n * dim,PETSC_DETERMINE,&points);CHKERRQ(ierr);
  ierr = VecSetBlockSize(points,dim);CHKERRQ(ierr);
  ierr = VecGetArray(points,&array);CHKERRQ(ierr);
  for (i = 0; i < n * dim; i++) {ierr = PetscRandomGetValue(rand,&array[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(points,&array);CHKERRQ(ierr);
  ierr = PetscMalloc6(n*nc,&B,n*nc,&rB,n*nc*dim,&D,n*nc*dim,&rD,n*nc*dim*dim,&H,n*nc*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field,points,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field,points,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)points,"Test Points");CHKERRQ(ierr);
  ierr = VecView(points,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,n*nc,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = VecDestroy(&points);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&cells);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscReal rc;

    ierr = PetscRandomGetValueReal(rand,&rc);CHKERRQ(ierr);
    cells[i] = PetscFloorReal(rc);
  }
  ierr = ISCreateGeneral(comm,n,cells,PETSC_OWN_POINTER,&cellIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)cellIS,"FE Test Cells");CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,NULL,NULL);CHKERRQ(ierr);
  N    = n * nq * nc;
  ierr = PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFE(field,cellIS,quad,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFE(field,cellIS,quad,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = PetscObjectSetName((PetscObject)quad,"Test quadrature");CHKERRQ(ierr);
  ierr = PetscQuadratureView(quad,viewer);CHKERRQ(ierr);
  ierr = ISView(cellIS,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,N,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)field);
  ierr = DMFieldGetNumComponents(field,&nc);CHKERRQ(ierr);
  ierr = DMFieldGetDM(field,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,(PetscScalar) cStart, (PetscScalar) cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&cells);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscReal rc;

    ierr = PetscRandomGetValueReal(rand,&rc);CHKERRQ(ierr);
    cells[i] = PetscFloorReal(rc);
  }
  ierr = ISCreateGeneral(comm,n,cells,PETSC_OWN_POINTER,&cellIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)cellIS,"FV Test Cells");CHKERRQ(ierr);
  N    = n * nc;
  ierr = PetscMalloc6(N,&B,N,&rB,N*dim,&D,N*dim,&rD,N*dim*dim,&H,N*dim*dim,&rH);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFV(field,cellIS,PETSC_SCALAR,B,D,H);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFV(field,cellIS,PETSC_REAL,rB,rD,rH);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);

  ierr = ISView(cellIS,viewer);CHKERRQ(ierr);
  ierr = ViewResults(viewer,N,dim,B,D,H,rB,rD,rH);CHKERRQ(ierr);

  ierr = PetscFree6(B,rB,D,rD,H,rH);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMFieldGetNumComponents(field, &Nc);CHKERRQ(ierr);
  ierr = DMFieldShellGetContext(field, (void *) &ctxVec);CHKERRQ(ierr);
  ierr = VecGetBlockSize(points, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(points, &n);CHKERRQ(ierr);
  n /= Nc;
  ierr = VecGetArrayRead(ctxVec, &mult);CHKERRQ(ierr);
  ierr = VecGetArrayRead(points, &x);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(points, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctxVec, &mult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestShellDestroy(DMField field)
{
  Vec                ctxVec = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMFieldShellGetContext(field, (void *) &ctxVec);CHKERRQ(ierr);
  ierr = VecDestroy(&ctxVec);CHKERRQ(ierr);
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

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMField Tutorial Options", "DM");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type","DM implementation on which to define field","ex1.c",DMList,type,type,256,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim","DM intrinsic dimension", "ex1.c", dim, &dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_components","Number of components in field", "ex1.c", nc, &nc, NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_quad_points","Number of quadrature points per dimension", "ex1.c", pointsPerEdge, &pointsPerEdge, NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_point_tests", "Number of test points for DMFieldEvaluate()", "ex1.c", numPoint, &numPoint, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_fe_tests", "Number of test cells for DMFieldEvaluateFE()", "ex1.c", numFE, &numFE, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_fv_tests", "Number of test cells for DMFieldEvaluateFV()", "ex1.c", numFV, &numFV, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_shell", "Test the DMFIELDSHELL implementation of DMField", "ex1.c", testShell, &testShell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"This examples works for dim <= 3, not %D",dim);
  ierr = PetscStrncmp(type,DMPLEX,256,&isplex);CHKERRQ(ierr);
  ierr = PetscStrncmp(type,DMDA,256,&isda);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  if (isplex) {
    PetscInt  overlap = 0;
    Vec       fieldvec;
    PetscInt  cells[3] = {3,3,3};
    PetscBool simplex;
    PetscFE   fe;

    ierr = PetscOptionsBegin(comm, "", "DMField DMPlex Options", "DM");CHKERRQ(ierr);
    ierr = PetscOptionsBoundedInt("-overlap","DMPlex parallel overlap","ex1.c",overlap,&overlap,NULL,0);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (0) {
      ierr = DMPlexCreateBoxMesh(comm,2,PETSC_TRUE,cells,NULL,NULL,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
    } else {
      ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
      ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
      ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
      CHKMEMQ;
    }
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
    if (simplex) {
      ierr = PetscDTStroudConicalQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
    } else {
      ierr = PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
    }
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    if (testShell) {
      Vec ctxVec;
      PetscInt i;
      PetscScalar *array;

      ierr = VecCreateSeq(PETSC_COMM_SELF, nc, &ctxVec);CHKERRQ(ierr);
      ierr = VecSetUp(ctxVec);CHKERRQ(ierr);
      ierr = VecGetArray(ctxVec,&array);CHKERRQ(ierr);
      for (i = 0; i < nc; i++) array[i] = i + 1.;
      ierr = VecRestoreArray(ctxVec,&array);CHKERRQ(ierr);
      ierr = DMFieldCreateShell(dm, nc, DMFIELD_VERTEX, (void *) ctxVec, &field);CHKERRQ(ierr);
      ierr = DMFieldShellSetEvaluate(field, TestShellEvaluate);CHKERRQ(ierr);
      ierr = DMFieldShellSetDestroy(field, TestShellDestroy);CHKERRQ(ierr);
    } else {
      ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,nc,simplex,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
      ierr = PetscFESetName(fe,"MyPetscFE");CHKERRQ(ierr);
      ierr = DMSetField(dm,0,NULL,(PetscObject)fe);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
      ierr = DMCreateDS(dm);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dm,&fieldvec);CHKERRQ(ierr);
      {
        PetscErrorCode (*func[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt, PetscScalar *,void *);
        void            *ctxs[1];

        func[0] = radiusSquared;
        ctxs[0] = NULL;

        ierr = DMProjectFunctionLocal(dm,0.0,func,ctxs,INSERT_ALL_VALUES,fieldvec);CHKERRQ(ierr);
      }
      ierr = DMFieldCreateDS(dm,0,fieldvec,&field);CHKERRQ(ierr);
      ierr = VecDestroy(&fieldvec);CHKERRQ(ierr);
    }
  } else if (isda) {
    PetscInt       i;
    PetscScalar    *cv;

    switch (dim) {
    case 1:
      ierr = DMDACreate1d(comm, DM_BOUNDARY_NONE, 3, 1, 1, NULL, &dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    default:
      ierr = DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 3, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, NULL, &dm);CHKERRQ(ierr);
      break;
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(nc * (1 << dim),&cv);CHKERRQ(ierr);
    for (i = 0; i < nc * (1 << dim); i++) {
      PetscReal rv;

      ierr = PetscRandomGetValueReal(rand,&rv);CHKERRQ(ierr);
      cv[i] = rv;
    }
    ierr = DMFieldCreateDA(dm,nc,cv,&field);CHKERRQ(ierr);
    ierr = PetscFree(cv);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim, 1, pointsPerEdge, -1.0, 1.0, &quad);CHKERRQ(ierr);
  } else SETERRQ1(comm,PETSC_ERR_SUP,"This test does not run for DM type %s",type);

  ierr = PetscObjectSetName((PetscObject)dm,"mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)field,"field");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)field,NULL,"-dmfield_view");CHKERRQ(ierr);
  if (numPoint) {ierr = TestEvaluate(field,numPoint,rand);CHKERRQ(ierr);}
  if (numFE) {ierr = TestEvaluateFE(field,numFE,cStart,cEnd,quad,rand);CHKERRQ(ierr);}
  if (numFV) {ierr = TestEvaluateFV(field,numFV,cStart,cEnd,rand);CHKERRQ(ierr);}
  ierr = DMFieldDestroy(&field);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
