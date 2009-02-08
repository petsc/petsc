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

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"
#include "petscdmmg.h"

extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  BCType        bcType;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  DA             da;
  UserContext    user;
  PetscReal      norm;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscErrorCode ierr;
  PetscInt       l,bc;

  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ***array;


  PetscInitialize(&argc,&argv,(char *)0,help);
  
  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);  
  ierr = DASetInterpolationType(da, DA_Q0);CHKERRQ(ierr);  

  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  } 
  
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMMG");
  bc          = (PetscInt)NEUMANN;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex34.c",bcTypes,2,bcTypes[0],&bc,PETSC_NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr = PetscOptionsEnd();
  
  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);
  if (user.bcType == NEUMANN) {
    ierr = DMMGSetNullSpace(dmmg,PETSC_TRUE,0,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  
  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(DMMGGetr(dmmg),-1.0,DMMGGetRHS(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",norm);CHKERRQ(ierr); 
  
  ierr = DAGetInfo(DMMGGetDA(dmmg), 0, &mx, &my, &mz, 0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DAGetCorners(DMMGGetDA(dmmg),&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAVecGetArray(DMMGGetDA(dmmg), DMMGGetx(dmmg), &array);CHKERRQ(ierr);

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
  ierr = DAVecRestoreArray(DMMGGetDA(dmmg), DMMGGetx(dmmg), &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(DMMGGetx(dmmg));CHKERRQ(ierr);

  ierr = VecNorm(DMMGGetx(dmmg),NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm);CHKERRQ(ierr); 
  ierr = VecNorm(DMMGGetx(dmmg),NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz)));CHKERRQ(ierr); 
  ierr = VecNorm(DMMGGetx(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz)));CHKERRQ(ierr); 

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  DA             da = (DA)dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ***array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, b, &array);CHKERRQ(ierr);
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
  ierr = DAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) 
    {
      MatNullSpace nullspace;
      
      ierr = KSPGetNullSpace(dmmg->ksp,&nullspace);CHKERRQ(ierr);
      ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
    }

  PetscFunctionReturn(0);
}

    
#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DMMG dmmg, Mat J,Mat jac)
{
  DA             da = (DA) dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
  PetscScalar    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
  MatStencil     row, col[7];

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx);
  Hy    = 1.0 / (PetscReal)(my);
  Hz    = 1.0 / (PetscReal)(mz);
  HyHzdHx = Hy*Hz/Hx;
  HxHzdHy = Hx*Hz/Hy;
  HxHydHz = Hx*Hy/Hz;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
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
		      SETERRQ(PETSC_ERR_SUP,"Dirichlet boundary conditions not supported !\n");
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

  PetscFunctionReturn(0);
}

