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
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  DM             da;
  Mat            A,Atrans;
  PetscInt       dof=1,M=8;
  PetscBool      flg,trans=PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL));

  PetscCall(DMDACreate(PETSC_COMM_WORLD,&da));
  PetscCall(DMSetDimension(da,3));
  PetscCall(DMDASetBoundaryType(da,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE));
  PetscCall(DMDASetStencilType(da,DMDA_STENCIL_STAR));
  PetscCall(DMDASetSizes(da,M,M,M));
  PetscCall(DMDASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(DMDASetDof(da,dof));
  PetscCall(DMDASetStencilWidth(da,1));
  PetscCall(DMDASetOwnershipRanges(da,NULL,NULL,NULL));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(DMCreateGlobalVector(da,&b));
  PetscCall(ComputeRHS(da,b));
  PetscCall(DMSetMatType(da,MATBAIJ));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(ComputeMatrix(da,A));

  /* A is non-symmetric. Make A = 0.5*(A + Atrans) symmetric for testing icc and cholesky */
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&Atrans));
  PetscCall(MatAXPY(A,1.0,Atrans,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatScale(A,0.5));
  PetscCall(MatDestroy(&Atrans));

  /* Test sbaij matrix */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-test_sbaij1", &flg,NULL));
  if (flg) {
    Mat       sA;
    PetscBool issymm;
    PetscCall(MatIsTranspose(A,A,0.0,&issymm));
    if (issymm) {
      PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
    } else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: A is non-symmetric\n"));
    PetscCall(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));
    PetscCall(MatDestroy(&A));
    A    = sA;
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetDM(pc,(DM)da));

  if (trans) {
    PetscCall(KSPSolveTranspose(ksp,b,x));
  } else {
    PetscCall(KSPSolve(ksp,b,x));
  }

  /* check final residual */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-check_final_residual", &flg,NULL));
  if (flg) {
    Vec       b1;
    PetscReal norm;
    PetscCall(KSPGetSolution(ksp,&x));
    PetscCall(VecDuplicate(b,&b1));
    PetscCall(MatMult(A,x,b1));
    PetscCall(VecAXPY(b1,-1.0,b));
    PetscCall(VecNorm(b1,NORM_2,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Final residual %g\n",norm));
    PetscCall(VecDestroy(&b1));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(DM da,Vec b)
{
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  PetscCall(VecSet(b,h));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(DM da,Mat B)
{
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0));
  /* For simplicity, this example only works on mx=my=mz */
  PetscCheck(mx == my && mx == mz,PETSC_COMM_SELF,PETSC_ERR_SUP,"This example only works with mx %D = my %D = mz %D",mx,my,mz);

  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;

  PetscCall(PetscMalloc1(2*dof*dof+1,&v));
  v_neighbor = v + dof*dof;
  PetscCall(PetscArrayzero(v,2*dof*dof+1));
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
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) { /* boundary points */
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&row,v,INSERT_VALUES));
        } else { /* interior points */
          /* center */
          col.i = i; col.j = j; col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v,INSERT_VALUES));

          /* x neighbors */
          col.i = i-1; col.j = j; col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i+1; col.j = j; col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));

          /* y neighbors */
          col.i = i; col.j = j-1; col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i; col.j = j+1; col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));

          /* z neighbors */
          col.i = i; col.j = j; col.k = k-1;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
          col.i = i; col.j = j; col.k = k+1;
          PetscCall(MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ksp_monitor_short -dm_mat_type sbaij -ksp_monitor_short -pc_type cholesky -ksp_view

TEST*/
