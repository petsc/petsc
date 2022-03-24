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

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar nu;
  BCType      bcType;
} UserContext;

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscErrorCode ierr;
  PetscInt       bc;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,12,12,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetInterpolationType(da, DMDA_Q0));

  CHKERRQ(KSPSetDM(ksp,da));

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DM");CHKERRQ(ierr);
  user.nu     = 0.1;
  CHKERRQ(PetscOptionsScalar("-nu", "The width of the Gaussian source", "ex29.c", 0.1, &user.nu, NULL));
  bc          = (PetscInt)NEUMANN;
  CHKERRQ(PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,NULL));
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(KSPSetComputeRHS(ksp,ComputeRHS,&user));
  CHKERRQ(KSPSetComputeOperators(ksp,ComputeMatrix,&user));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,NULL,NULL));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0));
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  CHKERRQ(DMDAVecGetArray(da, b, &array));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-(((PetscReal)i+0.5)*Hx)*(((PetscReal)i+0.5)*Hx)/user->nu)*PetscExpScalar(-(((PetscReal)j+0.5)*Hy)*(((PetscReal)j+0.5)*Hy)/user->nu)*Hx*Hy;
    }
  }
  CHKERRQ(DMDAVecRestoreArray(da, b, &array));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    CHKERRQ(MatNullSpaceRemove(nullspace,b));
    CHKERRQ(MatNullSpaceDestroy(&nullspace));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat J,Mat jac, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscInt       i,j,mx,my,xm,ym,xs,ys,num, numi, numj;
  PetscScalar    v[5],Hx,Hy,HydHx,HxdHy;
  MatStencil     row, col[5];
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0));
  Hx    = 1.0 / (PetscReal)(mx);
  Hy    = 1.0 / (PetscReal)(my);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*(HxdHy + HydHx);
          CHKERRQ(MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES));
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Dirichlet boundary conditions not supported !");
        } else if (user->bcType == NEUMANN) {
          num = 0; numi=0; numj=0;
          if (j!=0) {
            v[num] = -HxdHy;
            col[num].i = i;
            col[num].j = j-1;
            num++; numj++;
          }
          if (i!=0) {
            v[num]     = -HydHx;
            col[num].i = i-1;
            col[num].j = j;
            num++; numi++;
          }
          if (i!=mx-1) {
            v[num]     = -HydHx;
            col[num].i = i+1;
            col[num].j = j;
            num++; numi++;
          }
          if (j!=my-1) {
            v[num]     = -HxdHy;
            col[num].i = i;
            col[num].j = j+1;
            num++; numj++;
          }
          v[num] = (PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx; col[num].i = i;   col[num].j = j;
          num++;
          CHKERRQ(MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES));
        }
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        CHKERRQ(MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    CHKERRQ(MatSetNullSpace(J,nullspace));
    CHKERRQ(MatNullSpaceDestroy(&nullspace));
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -pc_mg_levels 3 -mg_coarse_pc_factor_shift_type nonzero

TEST*/
