/*
  Laplacian in 3D. Use for testing BAIJ matrix.
  Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

   with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

static char help[] = "Solves 3D Laplacian using wirebasket based multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(DM,Mat);
extern PetscErrorCode ComputeRHS(DM,Vec);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  DM             da;
  Mat            A,Atrans;
  PetscInt       dof=1,M=8;
  PetscBool      flg,trans=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL));

  CHKERRQ(DMDACreate(PETSC_COMM_WORLD,&da));
  CHKERRQ(DMSetDimension(da,3));
  CHKERRQ(DMDASetBoundaryType(da,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE));
  CHKERRQ(DMDASetStencilType(da,DMDA_STENCIL_STAR));
  CHKERRQ(DMDASetSizes(da,M,M,M));
  CHKERRQ(DMDASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(DMDASetDof(da,dof));
  CHKERRQ(DMDASetStencilWidth(da,1));
  CHKERRQ(DMDASetOwnershipRanges(da,NULL,NULL,NULL));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(DMCreateGlobalVector(da,&b));
  CHKERRQ(ComputeRHS(da,b));
  CHKERRQ(DMSetMatType(da,MATBAIJ));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(ComputeMatrix(da,A));

  /* A is non-symmetric. Make A = 0.5*(A + Atrans) symmetric for testing icc and cholesky */
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&Atrans));
  CHKERRQ(MatAXPY(A,1.0,Atrans,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatScale(A,0.5));
  CHKERRQ(MatDestroy(&Atrans));

  /* Test sbaij matrix */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-test_sbaij1", &flg,NULL));
  if (flg) {
    Mat       sA;
    PetscBool issymm;
    CHKERRQ(MatIsTranspose(A,A,0.0,&issymm));
    if (issymm) {
      CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
    } else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: A is non-symmetric\n"));
    CHKERRQ(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));
    CHKERRQ(MatDestroy(&A));
    A    = sA;
  }

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetDM(pc,(DM)da));

  if (trans) {
    CHKERRQ(KSPSolveTranspose(ksp,b,x));
  } else {
    CHKERRQ(KSPSolve(ksp,b,x));
  }

  /* check final residual */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-check_final_residual", &flg,NULL));
  if (flg) {
    Vec       b1;
    PetscReal norm;
    CHKERRQ(KSPGetSolution(ksp,&x));
    CHKERRQ(VecDuplicate(b,&b1));
    CHKERRQ(MatMult(A,x,b1));
    CHKERRQ(VecAXPY(b1,-1.0,b));
    CHKERRQ(VecNorm(b1,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Final residual %g\n",norm));
    CHKERRQ(VecDestroy(&b1));
  }

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(DM da,Vec b)
{
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  CHKERRQ(VecSet(b,h));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(DM da,Mat B)
{
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0));
  /* For simplicity, this example only works on mx=my=mz */
  PetscCheckFalse(mx != my || mx != mz,PETSC_COMM_SELF,PETSC_ERR_SUP,"This example only works with mx %D = my %D = mz %D",mx,my,mz);

  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;

  CHKERRQ(PetscMalloc1(2*dof*dof+1,&v));
  v_neighbor = v + dof*dof;
  CHKERRQ(PetscArrayzero(v,2*dof*dof+1));
  k3         = 0;
  for (k1=0; k1<dof; k1++) {
    for (k2=0; k2<dof; k2++) {
      if (k1 == k2) {
        v[k3]          = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
        v_neighbor[k3] = -HxHydHz;
      } else {
        v[k3]          = k1/(dof*dof);
        v_neighbor[k3] = k2/(dof*dof);
      }
      k3++;
    }
  }
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) { /* boundary points */
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&row,v,INSERT_VALUES));
        } else { /* interior points */
          /* center */
          col.i = i; col.j = j; col.k = k;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v,INSERT_VALUES));

          /* x neighbors */
          col.i = i-1; col.j = j; col.k = k;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i+1; col.j = j; col.k = k;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));

          /* y neighbors */
          col.i = i; col.j = j-1; col.k = k;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i; col.j = j+1; col.k = k;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));

          /* z neighbors */
          col.i = i; col.j = j; col.k = k-1;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i; col.j = j; col.k = k+1;
          CHKERRQ(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree(v));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ksp_monitor_short -dm_mat_type sbaij -ksp_monitor_short -pc_type cholesky -ksp_view

TEST*/
