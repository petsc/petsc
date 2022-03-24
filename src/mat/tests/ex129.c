
/*
  Laplacian in 3D. Use for testing MatSolve routines.
  Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

   with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

static char help[] = "This example is for testing different MatSolve routines :MatSolve(), MatSolveAdd(), MatSolveTranspose(), MatSolveTransposeAdd(), and MatMatSolve().\n\
Example usage: ./ex129 -mat_type aij -dof 2\n\n";

#include <petscdm.h>
#include <petscdmda.h>

extern PetscErrorCode ComputeMatrix(DM,Mat);
extern PetscErrorCode ComputeRHS(DM,Vec);
extern PetscErrorCode ComputeRHSMatrix(PetscInt,PetscInt,Mat*);

int main(int argc,char **args)
{
  PetscMPIInt    size;
  Vec            x,b,y,b1;
  DM             da;
  Mat            A,F,RHS,X,C1;
  MatFactorInfo  info;
  IS             perm,iperm;
  PetscInt       dof =1,M=8,m,n,nrhs;
  PetscScalar    one = 1.0;
  PetscReal      norm,tol = 1000*PETSC_MACHINE_EPSILON;
  PetscBool      InplaceLU=PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));

  CHKERRQ(DMDACreate(PETSC_COMM_WORLD,&da));
  CHKERRQ(DMSetDimension(da,3));
  CHKERRQ(DMDASetBoundaryType(da,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE));
  CHKERRQ(DMDASetStencilType(da,DMDA_STENCIL_STAR));
  CHKERRQ(DMDASetSizes(da,M,M,M));
  CHKERRQ(DMDASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(DMDASetDof(da,dof));
  CHKERRQ(DMDASetStencilWidth(da,1));
  CHKERRQ(DMDASetOwnershipRanges(da,NULL,NULL,NULL));
  CHKERRQ(DMSetMatType(da,MATBAIJ));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(DMCreateGlobalVector(da,&b));
  CHKERRQ(VecDuplicate(b,&y));
  CHKERRQ(ComputeRHS(da,b));
  CHKERRQ(VecSet(y,one));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(ComputeMatrix(da,A));
  CHKERRQ(MatGetSize(A,&m,&n));
  nrhs = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(ComputeRHSMatrix(m,nrhs,&RHS));
  CHKERRQ(MatDuplicate(RHS,MAT_DO_NOT_COPY_VALUES,&X));

  CHKERRQ(MatGetOrdering(A,MATORDERINGND,&perm,&iperm));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-inplacelu",&InplaceLU,NULL));
  CHKERRQ(MatFactorInfoInitialize(&info));
  if (!InplaceLU) {
    CHKERRQ(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F));
    info.fill = 5.0;
    CHKERRQ(MatLUFactorSymbolic(F,A,perm,iperm,&info));
    CHKERRQ(MatLUFactorNumeric(F,A,&info));
  } else { /* Test inplace factorization */
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&F));
    CHKERRQ(MatLUFactor(F,perm,iperm,&info));
  }

  CHKERRQ(VecDuplicate(y,&b1));

  /* MatSolve */
  CHKERRQ(MatSolve(F,b,x));
  CHKERRQ(MatMult(A,x,b1));
  CHKERRQ(VecAXPY(b1,-1.0,b));
  CHKERRQ(VecNorm(b1,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatSolve              : Error of norm %g\n",(double)norm));
  }

  /* MatSolveTranspose */
  CHKERRQ(MatSolveTranspose(F,b,x));
  CHKERRQ(MatMultTranspose(A,x,b1));
  CHKERRQ(VecAXPY(b1,-1.0,b));
  CHKERRQ(VecNorm(b1,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatSolveTranspose     : Error of norm %g\n",(double)norm));
  }

  /* MatSolveAdd */
  CHKERRQ(MatSolveAdd(F,b,y,x));
  CHKERRQ(MatMult(A,y,b1));
  CHKERRQ(VecScale(b1,-1.0));
  CHKERRQ(MatMultAdd(A,x,b1,b1));
  CHKERRQ(VecAXPY(b1,-1.0,b));
  CHKERRQ(VecNorm(b1,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatSolveAdd           : Error of norm %g\n",(double)norm));
  }

  /* MatSolveTransposeAdd */
  CHKERRQ(MatSolveTransposeAdd(F,b,y,x));
  CHKERRQ(MatMultTranspose(A,y,b1));
  CHKERRQ(VecScale(b1,-1.0));
  CHKERRQ(MatMultTransposeAdd(A,x,b1,b1));
  CHKERRQ(VecAXPY(b1,-1.0,b));
  CHKERRQ(VecNorm(b1,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatSolveTransposeAdd  : Error of norm %g\n",(double)norm));
  }

  /* MatMatSolve */
  CHKERRQ(MatMatSolve(F,RHS,X));
  CHKERRQ(MatMatMult(A,X,MAT_INITIAL_MATRIX,2.0,&C1));
  CHKERRQ(MatAXPY(C1,-1.0,RHS,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C1,NORM_FROBENIUS,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMatSolve           : Error of norm %g\n",(double)norm));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&b1));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&RHS));
  CHKERRQ(MatDestroy(&C1));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(DM da,Vec b)
{
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  CHKERRQ(VecSet(b,h));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHSMatrix(PetscInt m,PetscInt nrhs,Mat *C)
{
  PetscRandom    rand;
  Mat            RHS;
  PetscScalar    *array,rval;
  PetscInt       i,k;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&RHS));
  CHKERRQ(MatSetSizes(RHS,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  CHKERRQ(MatSetType(RHS,MATSEQDENSE));
  CHKERRQ(MatSetUp(RHS));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatDenseGetArray(RHS,&array));
  for (i=0; i<m; i++) {
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    array[i] = rval;
  }
  if (nrhs > 1) {
    for (k=1; k<nrhs; k++) {
      for (i=0; i<m; i++) {
        array[m*k+i] = array[i];
      }
    }
  }
  CHKERRQ(MatDenseRestoreArray(RHS,&array));
  CHKERRQ(MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY));
  *C   = RHS;
  CHKERRQ(PetscRandomDestroy(&rand));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(DM da,Mat B)
{
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy,r1,r2;
  MatStencil     row,col;
  PetscRandom    rand;

  PetscFunctionBegin;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetSeed(rand,1));
  CHKERRQ(PetscRandomSetInterval(rand,-.001,.001));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0));
  /* For simplicity, this example only works on mx=my=mz */
  PetscCheckFalse(mx != my || mx != mz,PETSC_COMM_SELF,PETSC_ERR_SUP,"This example only works with mx %" PetscInt_FMT " = my %" PetscInt_FMT " = mz %" PetscInt_FMT,mx,my,mz);

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
        CHKERRQ(PetscRandomGetValue(rand,&r1));
        CHKERRQ(PetscRandomGetValue(rand,&r2));

        v[k3]          = r1;
        v_neighbor[k3] = r2;
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
  CHKERRQ(PetscRandomDestroy(&rand));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -dm_mat_type aij -dof 1
      output_file: output/ex129.out

   test:
      suffix: 2
      args: -dm_mat_type aij -dof 1 -inplacelu
      output_file: output/ex129.out

TEST*/
