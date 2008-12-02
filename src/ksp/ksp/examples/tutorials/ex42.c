
/*
Laplacian in 3D. Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

with boundary conditions

   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

   This uses 2 level multigrid with the wirebasket based coarse problem to solve the linear system

*/

static char help[] = "Solves 3D Laplacian using wirebasket based multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"

extern PetscErrorCode ComputeMatrix(DA,Mat);
extern PetscErrorCode ComputeRHS(DA,Vec);
extern PetscErrorCode ComputeInterpolation(PC,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  DA             da;
  Mat            A;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-8,-8,-8,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);  
  ierr = DACreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = ComputeRHS(da,b);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATAIJ,&A);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,A);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCExoticSetDA(pc,da);CHKERRQ(ierr);
 
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
   
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
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    v[7],Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col[7];

  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  
  for (k=zs; k<zs+zm; k++){
    for (j=ys; j<ys+ym; j++){
      for(i=xs; i<xs+xm; i++){
        row.i = i; row.j = j; row.k = k;
	if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1){
          v[0] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
	  ierr = MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
	} else {
	  v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
	  v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
	  v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
	  v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
	  v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
	  v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
	  v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
	  ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

extern PetscErrorCode DAGetWireBasketInterpolation(DA,Mat,Mat*);

#undef __FUNCT__
#define __FUNCT__ "ComputeInterpolation"
PetscErrorCode ComputeInterpolation(PC pc,void *ida)
{
  PetscErrorCode ierr;
  DA             da = (DA)ida;
  Mat            A,P;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetWireBasketInterpolation(da,A,&P);CHKERRQ(ierr);
  ierr = PCMGSetInterpolation(pc,1,P);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


