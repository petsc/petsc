/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Added at the request of Marc Garbey.

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscReal rho;
  PetscReal nu;
  BCType    bcType;
} UserContext;

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  PetscErrorCode ierr;
  PetscInt       bc,xn,j,col,*owner,i,k;
  Vec            x,y,x_shm,y_shm,lvec,y_loc,x_loc;
  PC             pc;
  Mat            C,A,B;
  PetscMPIInt    srank,size;
  PetscBool      flg;
  PetscScalar    *mem;
  MPI_Win        win;
  MPI_Comm       shmcomm;
  PetscScalar    *x_arr;
  const PetscInt *garray,*ranges;
  Mat_MPIAIJ     *c;
  MPI_Aint       sz;
  PetscInt       dsp_unit,idx_loc,it,its=100;
  PetscScalar    **optr,*lvec_arr;
#if defined(PETSC_USE_LOG)
  PetscLogStage stages[3];
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  MPI_Comm_split_type(PETSC_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  ierr = MPI_Comm_rank(shmcomm,&srank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(shmcomm,&size);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Setup",&stages[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MatMult MPI",&stages[1]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MatMult Shm",&stages[2]);CHKERRQ(ierr);

  ierr = KSPCreate(shmcomm,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(shmcomm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0,1,0,1,0,0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"Pressure");CHKERRQ(ierr);

  user.rho    = 1.0;
  user.nu     = 0.1;
  bc          = (PetscInt)DIRICHLET;
  user.bcType = (BCType)bc;

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&C,NULL);CHKERRQ(ierr);

  ierr = MatCreateVecs(C,&x,&y);CHKERRQ(ierr);

  /* (1) Create x_shm and y_shm */
  /*----------------------------*/
  ierr = VecGetLocalSize(x,&xn);CHKERRQ(ierr);
  MPI_Win_allocate_shared(2*xn*sizeof(PetscScalar), 1, MPI_INFO_NULL, shmcomm, &mem, &win);

  ierr = VecCreateMPIWithArray(shmcomm,1,xn,PETSC_DECIDE,(const PetscScalar*)mem,&x_shm);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(shmcomm,1,xn,PETSC_DECIDE,(const PetscScalar*)(mem+xn),&y_shm);CHKERRQ(ierr);

  ierr = VecGetArray(x_shm,&x_arr);CHKERRQ(ierr);
  for (i=0; i<xn; i++) x_arr[i] = (PetscScalar)(srank+1);
  ierr = VecRestoreArray(x_shm,&x_arr);CHKERRQ(ierr);

  /* Compute y = C*x_shm using MatMult() for comparison */
  ierr = VecCopy(x_shm,x);CHKERRQ(ierr);
  ierr = MPI_Barrier(shmcomm);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);
  for (i=0; i<its; i++) {
    ierr = MatMult(C,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* Create Mv context */
  c = (Mat_MPIAIJ*)C->data;
  lvec = c->lvec;

  ierr = MatMPIAIJGetSeqAIJ(C,&A,&B,&garray);CHKERRQ(ierr);
  ierr = PetscMalloc1(B->cmap->n,&owner);CHKERRQ(ierr);
  //printf("[%d] Bn %d\n",srank,B->cmap->n);
  ierr = MatGetOwnershipRangesColumn(C,&ranges);CHKERRQ(ierr);
  if (srank == 1000) {
    printf("ranges: ");
    for (j=0; j<=size; j++) printf(" %d,",ranges[j]);
    printf("\n ");
  }

  PetscInt nNeighbors = 0,nGhosts[size];
  j = 0;
  for (i=0; i<=size; i++) {
    nGhosts[i] = 0;
    while (j < B->cmap->n) {
      col = garray[j];
      if (col < ranges[i+1]) {
        nGhosts[i]++; owner[j++] = i;
      } else break;
    }
  }
  PetscInt sum=0;
  for (i=0; i<size; i++) {
    if (nGhosts[i]) {nNeighbors++; sum += nGhosts[i];}
  }
  if (sum != B->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "sum %d != Bn %d ",sum,B->cmap->n);
  //printf("[%d] nNeighbors %d\n",srank,nNeighbors);
  ierr = PetscMalloc1(nNeighbors,&optr);CHKERRQ(ierr);
  k = 0;
  for (i=0; i<size; i++) {
    if (nGhosts[i]) {
      MPI_Win_shared_query(win,i,&sz,&dsp_unit,&optr[k]);
      k++;
    }
  }

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,xn,(const PetscScalar *)mem,&x_loc);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,xn,(const PetscScalar *)(mem+xn),&y_loc);CHKERRQ(ierr);
  if (srank == 1000) {
    VecView(x_loc,0);
  }

  /* (2) Read my ghosts from others into lvec */
  /*------------------------------------------*/
  PetscScalar *optr1;
  PetscInt    ii,rstart;
  ierr = MPI_Barrier(shmcomm);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stages[2]);CHKERRQ(ierr);
  for (it=0; it<its; it++) {
    ierr = VecGetArray(lvec,&lvec_arr);CHKERRQ(ierr);
    j = 0;
    ii = 0;
    for (i=0; i<size; i++) {
      if (nGhosts[i]) { /* read ghost values from shared proc[i] */
        optr1  = optr[ii];
        rstart = ranges[i];
        for (k=0; k< nGhosts[i]; k++) {
          idx_loc = garray[j] - rstart;
          lvec_arr[j++] = optr1[idx_loc];
        }
        ii++;
      }
    }
    ierr = VecRestoreArray(lvec,&lvec_arr);CHKERRQ(ierr);

    /* (3) y_loc = A*x_loc + B*lvec */
    /*------------------------------*/
    /*y_loc  = B*lvec */
    ierr = MatMult(B,lvec,y_loc);CHKERRQ(ierr);

    /* y_loc = A*x_loc + y_loc */
    ierr = MatMultAdd(A,x_loc,y_loc,y_loc);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  //if (srank == 0) printf("y_shm:\n");
  //ierr = VecView(y_shm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* (4) Check y == y_shm */
  /*----------------------*/
  ierr = VecEqual(y,y_shm,&flg);CHKERRQ(ierr);
  if (!flg) printf("y != y_shm\n");

  /* Free spaces */
  ierr = VecDestroy(&y_loc);CHKERRQ(ierr);
  ierr = VecDestroy(&x_loc);CHKERRQ(ierr);
  ierr = PetscFree(optr);CHKERRQ(ierr);
  MPI_Win_free(&win);
  MPI_Comm_free(&shmcomm);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x_shm);CHKERRQ(ierr);
  ierr = VecDestroy(&y_shm);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      centerRho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];
  DM             da;

  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  centerRho = user->rho;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      ierr  = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j-1;
            numy++; num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;              col[num].i = i-1; col[num].j = j;
            numx++; num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;              col[num].i = i+1; col[num].j = j;
            numx++; num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j+1;
            numy++; num++;
          }
          v[num] = numx*rho*HydHx + numy*rho*HxdHy; col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires:  define(PETSC_HAVE_MPI_WIN_ALLOCATE_SHARED) define(PETSC_HAVE_MPI_WIN_SHARED_QUERY)

    test:

TEST*/
