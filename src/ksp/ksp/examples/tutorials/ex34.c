/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Laplacian in 3D. Modeled by the partial differential equation

   div  grad u = f,  0 < x,y,z < 1,

with pure Neumann boundary conditions

   u = 0 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

The functions are cell-centered

This uses multigrid to solve the linear system

       Contributed by Jianming Yang <jianming-yang@uiowa.edu>
*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\n";

#include <petscdmda.h>
#include <petscksp.h>
#include <petscpcmg.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,MatStructure*,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  BCType        bcType;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  PetscReal      norm;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscErrorCode ierr;
  PetscInt       bc;

  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ***array;
  Vec            x,b,r;
  Mat            J;

  PetscInitialize(&argc,&argv,(char *)0,help);
  
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,12,12,12,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);  
  ierr = DMDASetInterpolationType(da, DMDA_Q0);CHKERRQ(ierr);  

  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DM");
  bc          = (PetscInt)NEUMANN;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex34.c",bcTypes,2,bcTypes[0],&bc,PETSC_NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr = PetscOptionsEnd();
  
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,PETSC_NULL,&J,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);

  ierr = MatMult(J,x,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",norm);CHKERRQ(ierr); 
  
  ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, x, &array);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++){
    for (j=ys; j<ys+ym; j++){
      for(i=xs; i<xs+xm; i++){
	array[k][j][i] -= 
	  PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))*
	  PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))*
	  PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz));
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, x, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm);CHKERRQ(ierr); 
  ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz)));CHKERRQ(ierr); 
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz)));CHKERRQ(ierr); 

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ***array;
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (k=zs; k<zs+zm; k++){
    for (j=ys; j<ys+ym; j++){
      for(i=xs; i<xs+xm; i++){
	array[k][j][i] = 12*PETSC_PI*PETSC_PI
	  *PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))
	  *PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))
	  *PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz))
	  *Hx*Hy*Hz;
      }
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
    ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

    
#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(KSP ksp, Mat J,Mat jac,MatStructure *str, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
  PetscScalar    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
  MatStencil     row, col[7];
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx);
  Hy    = 1.0 / (PetscReal)(my);
  Hz    = 1.0 / (PetscReal)(mz);
  HyHzdHx = Hy*Hz/Hx;
  HxHzdHy = Hx*Hz/Hy;
  HxHydHz = Hx*Hy/Hz;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  for (k=zs; k<zs+zm; k++)
    {
      for (j=ys; j<ys+ym; j++)
	{
	  for(i=xs; i<xs+xm; i++)
	    {
	      row.i = i; row.j = j; row.k = k;
	      if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) 
		{
		  if (user->bcType == DIRICHLET) 
		    {
		      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Dirichlet boundary conditions not supported !\n");
		      v[0] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
		      ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
		    } 
		  else if (user->bcType == NEUMANN) 
		    {
		      num = 0; numi=0; numj=0; numk=0;
		      if (k!=0) 
			{
			  v[num] = -HxHydHz;              
			  col[num].i = i;   
			  col[num].j = j;
			  col[num].k = k-1;
			  num++; numk++;
			}
		      if (j!=0) 
			{
			  v[num] = -HxHzdHy;              
			  col[num].i = i;   
			  col[num].j = j-1;
			  col[num].k = k;
			  num++; numj++;
			}
		      if (i!=0) 
			{
			  v[num] = -HyHzdHx;              
			  col[num].i = i-1; 
			  col[num].j = j;
			  col[num].k = k;
			  num++; numi++;
			}
		      if (i!=mx-1) 
			{
			  v[num] = -HyHzdHx;              
			  col[num].i = i+1; 
			  col[num].j = j;
			  col[num].k = k;
			  num++; numi++;
			}
		      if (j!=my-1) 
			{
			  v[num] = -HxHzdHy;              
			  col[num].i = i;   
			  col[num].j = j+1;
			  col[num].k = k;
			  num++; numj++;
			}
		      if (k!=mz-1) 
			{
			  v[num] = -HxHydHz;              
			  col[num].i = i;   
			  col[num].j = j;
			  col[num].k = k+1;
			  num++; numk++;
			}
		      v[num]   = (PetscReal)(numk)*HxHydHz + (PetscReal)(numj)*HxHzdHy + (PetscReal)(numi)*HyHzdHx;
		      col[num].i = i;   col[num].j = j;   col[num].k = k;
		      num++;
		      ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
		    }
		} 
	      else 
		{
		  v[0] = -HxHydHz;                          col[0].i = i;   col[0].j = j;   col[0].k = k-1;
		  v[1] = -HxHzdHy;                          col[1].i = i;   col[1].j = j-1; col[1].k = k;
		  v[2] = -HyHzdHx;                          col[2].i = i-1; col[2].j = j;   col[2].k = k;
		  v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz); col[3].i = i;   col[3].j = j;   col[3].k = k;
		  v[4] = -HyHzdHx;                          col[4].i = i+1; col[4].j = j;   col[4].k = k;
		  v[5] = -HxHzdHy;                          col[5].i = i;   col[5].j = j+1; col[5].k = k;
		  v[6] = -HxHydHz;                          col[6].i = i;   col[6].j = j;   col[6].k = k+1;
		  ierr = MatSetValuesStencil(jac,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
		}
	    }
	}
    }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(jac,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

