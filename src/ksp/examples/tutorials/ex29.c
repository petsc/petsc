/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   div \kappa grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-(1 - x)^2/\nu} e^{-(1 - y)^2/\nu}

with boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"

extern int ComputeJacobian(DMMG,Mat);
extern int ComputeRHS(DMMG,Vec);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar nu;
  BCType      bcType;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG       *dmmg;
  DA          da;
  UserContext user;
  char        bcTypeName[256];
  PetscReal   norm;
  PetscScalar mone = -1.0;
  PetscTruth  isDirichlet, isNeumann, iopt;
  int         l, ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for(l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMMG");
  user.nu = 0.1;
  ierr = PetscOptionsScalar("-nu", "The width of the Gaussian source", "ex29.c", 0.1, &user.nu, &iopt);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bc_type", "Type of boundary condition", "ex29.c", "dirichlet", bcTypeName, 256, &iopt);CHKERRQ(ierr);
  ierr = PetscStrcasecmp(bcTypeName, "dirichlet", &isDirichlet);CHKERRQ(ierr);
  ierr = PetscStrcasecmp(bcTypeName, "neumann", &isNeumann);CHKERRQ(ierr);
  if (isDirichlet) {
    user.bcType = DIRICHLET;
  } else if (isNeumann) {
    user.bcType = NEUMANN;
  } else {
    user.bcType = DIRICHLET;
  }
  ierr = PetscOptionsEnd();

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeJacobian);CHKERRQ(ierr);
  if (user.bcType == NEUMANN) {
    MatNullSpace nullSpace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &nullSpace);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(DMMGGetKSP(dmmg), nullSpace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(nullSpace);CHKERRQ(ierr);
  }

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(&mone,DMMGGetb(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",norm);CHKERRQ(ierr); */

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
int ComputeRHS(DMMG dmmg, Vec b)
{
  DA          da = (DA)dmmg->dm;
  UserContext *user = (UserContext *) dmmg->user;
  int         ierr,i,j,mx,my,xm,ym,xs,ys;
  PetscScalar Hx,Hy;
  PetscScalar **array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = PetscExpScalar(-(i*Hx)*(i*Hx)/user->nu)*PetscExpScalar(-(j*Hy)*(j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
    
#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
int ComputeJacobian(DMMG dmmg, Mat jac)
{
  DA           da = (DA) dmmg->dm;
  UserContext *user = (UserContext *) dmmg->user;
  int          ierr,i,j,mx,my,xm,ym,xs,ys,num;
  PetscScalar  v[5],Hx,Hy,HydHx,HxdHy,rho;
  MatStencil   row, col[5];

  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx-1);
  Hy    = 1.0 / (PetscReal)(my-1);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      row.i = i; row.j = j;
      if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
        rho = 100.0;
      } else {
        rho = 1.0;
      }
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          num = 0;
          if (j!=0) {
            v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
            num++;
          }
          if (i!=0) {
            v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
            num++;
          }
          v[2]   = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
          num++;
          if (i!=mx-1) {
            v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
            num++;
          }
          if (j!=my-1) {
            v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
            num++;
          }
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}
