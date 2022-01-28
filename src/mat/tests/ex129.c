
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
  PetscErrorCode ierr;
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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only");
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);

  ierr = DMDACreate(PETSC_COMM_WORLD,&da);CHKERRQ(ierr);
  ierr = DMSetDimension(da,3);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da,DMDA_STENCIL_STAR);CHKERRQ(ierr);
  ierr = DMDASetSizes(da,M,M,M);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetDof(da,dof);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,1);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(da,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATBAIJ);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&y);CHKERRQ(ierr);
  ierr = ComputeRHS(da,b);CHKERRQ(ierr);
  ierr = VecSet(y,one);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  nrhs = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = ComputeRHSMatrix(m,nrhs,&RHS);CHKERRQ(ierr);
  ierr = MatDuplicate(RHS,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  ierr = MatGetOrdering(A,MATORDERINGND,&perm,&iperm);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-inplacelu",&InplaceLU,NULL);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  if (!InplaceLU) {
    ierr      = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    info.fill = 5.0;
    ierr      = MatLUFactorSymbolic(F,A,perm,iperm,&info);CHKERRQ(ierr);
    ierr      = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);
  } else { /* Test inplace factorization */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
    ierr = MatLUFactor(F,perm,iperm,&info);CHKERRQ(ierr);
  }

  ierr = VecDuplicate(y,&b1);CHKERRQ(ierr);

  /* MatSolve */
  ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  ierr = MatMult(A,x,b1);CHKERRQ(ierr);
  ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolve              : Error of norm %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* MatSolveTranspose */
  ierr = MatSolveTranspose(F,b,x);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,b1);CHKERRQ(ierr);
  ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolveTranspose     : Error of norm %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* MatSolveAdd */
  ierr = MatSolveAdd(F,b,y,x);CHKERRQ(ierr);
  ierr = MatMult(A,y,b1);CHKERRQ(ierr);
  ierr = VecScale(b1,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(A,x,b1,b1);CHKERRQ(ierr);
  ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolveAdd           : Error of norm %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* MatSolveTransposeAdd */
  ierr = MatSolveTransposeAdd(F,b,y,x);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,y,b1);CHKERRQ(ierr);
  ierr = VecScale(b1,-1.0);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A,x,b1,b1);CHKERRQ(ierr);
  ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolveTransposeAdd  : Error of norm %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* MatMatSolve */
  ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);
  ierr = MatMatMult(A,X,MAT_INITIAL_MATRIX,2.0,&C1);CHKERRQ(ierr);
  ierr = MatAXPY(C1,-1.0,RHS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C1,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMatSolve           : Error of norm %g\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  ierr = MatDestroy(&C1);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(DM da,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  ierr = VecSet(b,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHSMatrix(PetscInt m,PetscInt nrhs,Mat *C)
{
  PetscErrorCode ierr;
  PetscRandom    rand;
  Mat            RHS;
  PetscScalar    *array,rval;
  PetscInt       i,k;

  PetscFunctionBegin;
  ierr = MatCreate(PETSC_COMM_WORLD,&RHS);CHKERRQ(ierr);
  ierr = MatSetSizes(RHS,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(RHS,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(RHS);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatDenseGetArray(RHS,&array);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr     = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    array[i] = rval;
  }
  if (nrhs > 1) {
    for (k=1; k<nrhs; k++) {
      for (i=0; i<m; i++) {
        array[m*k+i] = array[i];
      }
    }
  }
  ierr = MatDenseRestoreArray(RHS,&array);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *C   = RHS;
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(DM da,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy,r1,r2;
  MatStencil     row,col;
  PetscRandom    rand;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand,1);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,-.001,.001);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  /* For simplicity, this example only works on mx=my=mz */
  PetscAssertFalse(mx != my || mx != mz,PETSC_COMM_SELF,PETSC_ERR_SUP,"This example only works with mx %" PetscInt_FMT " = my %" PetscInt_FMT " = mz %" PetscInt_FMT,mx,my,mz);

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
        ierr = PetscRandomGetValue(rand,&r1);CHKERRQ(ierr);
        ierr = PetscRandomGetValue(rand,&r2);CHKERRQ(ierr);

        v[k3]          = r1;
        v_neighbor[k3] = r2;
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
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
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
