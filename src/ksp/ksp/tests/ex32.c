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
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL);CHKERRQ(ierr);

  ierr = DMDACreate(PETSC_COMM_WORLD,&da);CHKERRQ(ierr);
  ierr = DMSetDimension(da,3);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da,DMDA_STENCIL_STAR);CHKERRQ(ierr);
  ierr = DMDASetSizes(da,M,M,M);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetDof(da,dof);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,1);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(da,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = ComputeRHS(da,b);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATBAIJ);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,A);CHKERRQ(ierr);

  /* A is non-symmetric. Make A = 0.5*(A + Atrans) symmetric for testing icc and cholesky */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&Atrans);CHKERRQ(ierr);
  ierr = MatAXPY(A,1.0,Atrans,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(A,0.5);CHKERRQ(ierr);
  ierr = MatDestroy(&Atrans);CHKERRQ(ierr);

  /* Test sbaij matrix */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-test_sbaij1", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Mat       sA;
    PetscBool issymm;
    ierr = MatIsTranspose(A,A,0.0,&issymm);CHKERRQ(ierr);
    if (issymm) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    } else {ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: A is non-symmetric\n");CHKERRQ(ierr);}
    ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = sA;
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDM(pc,(DM)da);CHKERRQ(ierr);

  if (trans) {
    ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }

  /* check final residual */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-check_final_residual", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Vec       b1;
    PetscReal norm;
    ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&b1);CHKERRQ(ierr);
    ierr = MatMult(A,x,b1);CHKERRQ(ierr);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Final residual %g\n",norm);CHKERRQ(ierr);
    ierr = VecDestroy(&b1);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(DM da,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  ierr = VecSet(b,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(DM da,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  /* For simplicity, this example only works on mx=my=mz */
  if (mx != my || mx != mz) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"This example only works with mx %D = my %D = mz %D\n",mx,my,mz);

  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;

  ierr       = PetscMalloc1(2*dof*dof+1,&v);CHKERRQ(ierr);
  v_neighbor = v + dof*dof;
  ierr       = PetscArrayzero(v,2*dof*dof+1);CHKERRQ(ierr);
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
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) { /* boundary points */
          ierr = MatSetValuesBlockedStencil(B,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else { /* interior points */
          /* center */
          col.i = i; col.j = j; col.k = k;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v,INSERT_VALUES);CHKERRQ(ierr);

          /* x neighbors */
          col.i = i-1; col.j = j; col.k = k;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
          col.i = i+1; col.j = j; col.k = k;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);

          /* y neighbors */
          col.i = i; col.j = j-1; col.k = k;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
          col.i = i; col.j = j+1; col.k = k;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);

          /* z neighbors */
          col.i = i; col.j = j; col.k = k-1;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
          col.i = i; col.j = j; col.k = k+1;
          ierr  = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ksp_monitor_short -dm_mat_type sbaij -ksp_monitor_short -pc_type cholesky -ksp_view

TEST*/
