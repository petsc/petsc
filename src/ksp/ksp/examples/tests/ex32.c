
/*
  Laplacian in 3D. Use for testing BAIJ matrix. 
  Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

   with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

static char help[] = "Solves 3D Laplacian using wirebasket based multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"

extern PetscErrorCode ComputeMatrix(DA,Mat);
extern PetscErrorCode ComputeRHS(DA,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  DA             da;
  Mat            A,Atrans;
  PetscInt       dof=1,M=-8;
  PetscTruth     flg,trans=PETSC_FALSE;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-trans",&trans,PETSC_NULL);CHKERRQ(ierr);

  ierr = DACreate(PETSC_COMM_WORLD,&da);CHKERRQ(ierr);
  ierr = DASetDim(da,3);CHKERRQ(ierr);
  ierr = DASetPeriodicity(da,DA_NONPERIODIC);CHKERRQ(ierr);
  ierr = DASetStencilType(da,DA_STENCIL_STAR);CHKERRQ(ierr);
  ierr = DASetSizes(da,M,M,M);CHKERRQ(ierr);
  ierr = DASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DASetDof(da,dof);CHKERRQ(ierr);
  ierr = DASetStencilWidth(da,1);CHKERRQ(ierr);
  ierr = DASetVertexDivision(da,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DASetFromOptions(da);CHKERRQ(ierr);

  ierr = DACreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = ComputeRHS(da,b);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATBAIJ,&A);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,A);CHKERRQ(ierr);


  /* A is non-symmetric. Make A = 0.5*(A + Atrans) symmetric for testing icc and cholesky */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&Atrans);CHKERRQ(ierr);
  ierr = MatAXPY(A,1.0,Atrans,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(A,0.5);CHKERRQ(ierr);
  ierr = MatDestroy(Atrans);CHKERRQ(ierr);

  /* Test sbaij matrix */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-test_sbaij1", &flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg){
    Mat sA;
    ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
    A = sA;
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDA(pc,da);CHKERRQ(ierr);
 
  if (trans) {
    ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }

  /* check final residual */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-check_final_residual", &flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg){
    Vec            b1;
    PetscReal      norm;
    ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&b1);CHKERRQ(ierr);
    ierr = MatMult(A,x,b1);CHKERRQ(ierr);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Final residual %g\n",norm);CHKERRQ(ierr);
    ierr = VecDestroy(b1);CHKERRQ(ierr);
  }
   
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DA da,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       mx,my,mz;
  PetscScalar    h;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  ierr = VecSet(b,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
    
#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DA da,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,dof,k1,k2,k3;
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col;
 
  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);CHKERRQ(ierr); 
  /* For simplicity, this example only works on mx=my=mz */
  if ( mx != my || mx != mz) SETERRQ3(1,"This example only works with mx %d = my %d = mz %d\n",mx,my,mz);

  Hx = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;

  ierr = PetscMalloc((2*dof*dof+1)*sizeof(PetscScalar),&v);CHKERRQ(ierr);
  v_neighbor = v + dof*dof;
  ierr = PetscMemzero(v,(2*dof*dof+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  k3 = 0;
  for (k1=0; k1<dof; k1++){
    for (k2=0; k2<dof; k2++){
      if (k1 == k2){
        v[k3]          = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
        v_neighbor[k3] = -HxHydHz;
      } else {
	v[k3] = k1/(dof*dof); ;
	v_neighbor[k3] = k2/(dof*dof);
      }	
      k3++;
    }
  }
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  
  for (k=zs; k<zs+zm; k++){
    for (j=ys; j<ys+ym; j++){
      for(i=xs; i<xs+xm; i++){
        row.i = i; row.j = j; row.k = k;
	if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1){ /* boudary points */	 
	  ierr = MatSetValuesBlockedStencil(B,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else { /* interior points */
          /* center */
          col.i = i; col.j = j; col.k = k;
          ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v,INSERT_VALUES);CHKERRQ(ierr);          
          
          /* x neighbors */
	  col.i = i-1; col.j = j; col.k = k;
          ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
	  col.i = i+1; col.j = j; col.k = k;
	  ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
	 
	  /* y neighbors */
	  col.i = i; col.j = j-1; col.k = k;
	  ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
	  col.i = i; col.j = j+1; col.k = k;
	  ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
	 
          /* z neighbors */
	  col.i = i; col.j = j; col.k = k-1;
	  ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
	  col.i = i; col.j = j; col.k = k+1;
          ierr = MatSetValuesBlockedStencil(B,1,&row,1,&col,v_neighbor,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

