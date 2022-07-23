/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Periodic boundary conditions
         p(x=0) = p(x=1)
       Neuman boundary conditions
         dp/dy = 0 for y = 0, y = 1.

       This example uses the DM_BOUNDARY_MIRROR to implement the Neumann boundary conditions, see the manual pages for DMBoundaryType

       Compare to ex50.c
*/

static char help[] = "Solves 2D Poisson equation,\n\
                      using mirrored boundaries to implement Neumann boundary conditions,\n\
                      using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

extern PetscErrorCode ComputeJacobian(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef struct {
  PetscScalar uu, tt;
} UserContext;

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(KSPSetDM(ksp,(DM)da));
  PetscCall(DMSetApplicationContext(da,&user));

  user.uu     = 1.0;
  user.tt     = 1.0;

  PetscCall(KSPSetComputeRHS(ksp,ComputeRHS,&user));
  PetscCall(KSPSetComputeOperators(ksp,ComputeJacobian,&user));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,NULL,NULL));

  PetscCall(DMDestroy(&da));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscInt       i,j,M,N,xm,ym,xs,ys;
  PetscScalar    Hx,Hy,pi,uu,tt;
  PetscScalar    **array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp,&da));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0));
  uu   = user->uu; tt = user->tt;
  pi   = 4*PetscAtanReal(1.0);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0)); /* Fine grid */
  PetscCall(DMDAVecGetArray(da, b, &array));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = -PetscCosScalar(uu*pi*((PetscReal)i+0.5)*Hx)*PetscCosScalar(tt*pi*((PetscReal)j+0.5)*Hy)*Hx*Hy;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, b, &array));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
  PetscCall(MatNullSpaceRemove(nullspace,b));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat jac,void *ctx)
{
  PetscInt       i, j, M, N, xm, ym, xs, ys;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp,&da));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0));
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
      v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
      v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
      v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
      v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
      PetscCall(MatSetValuesStencil(jac,1,&row,5,col,v,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
  PetscCall(MatSetNullSpace(J,nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type svd -ksp_view

   test:
      suffix: 2
      nsize: 4
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view

TEST*/
