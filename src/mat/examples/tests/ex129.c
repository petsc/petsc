
/*
  Laplacian in 3D. Use for testing MatSolve routines. 
  Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

   with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

static char help[] = "This example is for testing different MatSolve routines,namely MatSolve,MatSolveAdd,MatSolveTranspose and MatSolveTransposeAdd.\n\
Example usage: ./ex129 -solve_type 1 -dof 1\n\n";

#include "petscda.h"
#include "petscmg.h"

extern PetscErrorCode ComputeMatrix(DA,Mat);
extern PetscErrorCode ComputeRHS(DA,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  Vec               x,b,y,b1;
  DA                da;
  Mat               A,F;
  MatFactorInfo     info;
  IS                perm,iperm;
  PetscInt          dof=1,M=-8,solve_type=0;
  PetscScalar       one = 1.0;
  PetscReal         norm;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if(size != 1) SETERRQ(1,"This is a uniprocessor example only\n");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);

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
  ierr = VecDuplicate(b,&y);CHKERRQ(ierr);
  ierr = ComputeRHS(da,b);CHKERRQ(ierr);
  ierr = VecSet(y,one);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATBAIJ,&A);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,A);CHKERRQ(ierr);

  ierr = MatGetOrdering(A,MATORDERING_ND,&perm,&iperm);CHKERRQ(ierr);
  ierr = MatGetFactor(A,MAT_SOLVER_PETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  info.fill = 5.0;
  ierr = MatLUFactorSymbolic(F,A,perm,iperm,&info);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-solve_type",&solve_type,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&b1);CHKERRQ(ierr);
  switch (solve_type){
  case 1:
    /* MatSolve */
    ierr = MatSolve(F,b,x);CHKERRQ(ierr);
    ierr = MatMult(A,x,b1);CHKERRQ(ierr);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error of norm %g\n",norm);CHKERRQ(ierr);
    break;
  case 2:
    /* MatSolveTranspose */
    ierr = MatSolveTranspose(F,b,x);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,x,b1);CHKERRQ(ierr);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error of norm %g\n",norm);CHKERRQ(ierr);
    break;
  case 3:
    /* MatSolveAdd */
    ierr = MatSolveAdd(F,b,y,x);CHKERRQ(ierr);
    ierr = MatMult(A,y,b1);CHKERRQ(ierr);
    ierr = VecScale(b1,-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,b1,b1);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error or norm %g\n",norm);CHKERRQ(ierr);
    break;
  case 4:
    /* MatSolveTransposeAdd */
    ierr = MatSolveTransposeAdd(F,b,y,x);CHKERRQ(ierr); 
    ierr = MatMultTranspose(A,y,b1);CHKERRQ(ierr);
    ierr = VecScale(b1,-1.0);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,b1,b1);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error or norm %g\n",norm);CHKERRQ(ierr);
    break;
  }
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(b1);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(F);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);
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
  PetscScalar    *v,*v_neighbor,Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy,r1,r2;
  MatStencil     row,col;
  PetscRandom    rand; 

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rand,PETSCRAND);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand,1);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,-.001,.001);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

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
	ierr = PetscRandomGetValue(rand,&r1);CHKERRQ(ierr);
	ierr = PetscRandomGetValue(rand,&r2);CHKERRQ(ierr);
	v[k3] = r1;
	v_neighbor[k3] = r2;
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
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);
  /* ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */ 
  return 0;
}

