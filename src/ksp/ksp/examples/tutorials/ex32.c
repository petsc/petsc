/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Laplacian in 2D. Modeled by the partial differential equation

   div  grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-(1 - x)^2/\nu} e^{-(1 - y)^2/\nu}

with pure Neumann boundary conditions

The functions are cell-centered

This uses multigrid to solve the linear system

       Contributed by Andrei Draganescu <aidraga@sandia.gov>

Note the nice multigrid convergence despite the fact it is only using
peicewise constant interpolation/restriction. This is because cell-centered multigrid
does not need the same rule: 

    polynomial degree(interpolation) + polynomial degree(restriction) + 2 > degree of PDE

that vertex based multigrid needs.
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"
#include "petscdmmg.h"

extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar   nu;
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

  PetscInitialize(&argc,&argv,(char *)0,help);
  
  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);  
  ierr = DASetInterpolationType(da, DA_Q0);CHKERRQ(ierr);  
  
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  }
  
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMMG");
  user.nu     = 0.1;
  ierr        = PetscOptionsScalar("-nu", "The width of the Gaussian source", "ex29.c", 0.1, &user.nu, PETSC_NULL);CHKERRQ(ierr);
  bc          = (PetscInt)NEUMANN;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,PETSC_NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr = PetscOptionsEnd();
  
  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);
  if (user.bcType == NEUMANN) {
    ierr = DMMGSetNullSpace(dmmg,PETSC_TRUE,0,PETSC_NULL);CHKERRQ(ierr);
  }
  
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  
  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(DMMGGetRHS(dmmg),-1.0,DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",norm);CHKERRQ(ierr); */

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
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = PetscExpScalar(-(((PetscReal)i+0.5)*Hx)*(((PetscReal)i+0.5)*Hx)/user->nu)*PetscExpScalar(-(((PetscReal)j+0.5)*Hy)*(((PetscReal)j+0.5)*Hy)/user->nu)*Hx*Hy;
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
  PetscInt       i,j,mx,my,xm,ym,xs,ys,num, numi, numj;
  PetscScalar    v[5],Hx,Hy,HydHx,HxdHy;
  MatStencil     row, col[5];

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx);
  Hy    = 1.0 / (PetscReal)(my);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++)  {
    for(i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
	if (user->bcType == DIRICHLET) {
	  v[0] = 2.0*(HxdHy + HydHx);
	  ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
	  SETERRQ(PETSC_ERR_SUP,"Dirichlet boundary conditions not supported !\n");
	} else if (user->bcType == NEUMANN) {
	  num = 0; numi=0; numj=0;
	  if (j!=0)  {
	    v[num] = -HxdHy;              
	    col[num].i = i;   
	    col[num].j = j-1;
	    num++; numj++;
	  }
	  if (i!=0)   {
	    v[num] = -HydHx;              
	    col[num].i = i-1; 
	    col[num].j = j;
	    num++; numi++;
	  }
	  if (i!=mx-1) {
	    v[num] = -HydHx;              
	    col[num].i = i+1; 
	    col[num].j = j;
	    num++; numi++;
	  }
	  if (j!=my-1)  {
	    v[num] = -HxdHy;              
	    col[num].i = i;   
	    col[num].j = j+1;
	    num++; numj++;
	  }
	  v[num]   = (PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx; col[num].i = i;   col[num].j = j;
	  num++;
	  ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
	}
      } else   {
	v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
	v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
	v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
	v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
	v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
	ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
