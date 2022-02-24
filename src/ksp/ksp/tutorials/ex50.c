/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Neuman boundary conditions
        dp/dx = 0 for x = 0, x = 1.
        dp/dy = 0 for y = 0, y = 1.

     Contributed by Michael Boghosian <boghmic@iit.edu>, 2008,
         based on petsc/src/ksp/ksp/tutorials/ex29.c and ex32.c

     Compare to ex66.c

     Example of Usage:
          ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause -1
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor
          mpiexec -n 4 ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view
*/

static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";

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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(KSPSetDM(ksp,(DM)da));
  CHKERRQ(DMSetApplicationContext(da,&user));

  user.uu     = 1.0;
  user.tt     = 1.0;

  CHKERRQ(KSPSetComputeRHS(ksp,ComputeRHS,&user));
  CHKERRQ(KSPSetComputeOperators(ksp,ComputeJacobian,&user));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,NULL,NULL));

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
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
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0));
  uu   = user->uu; tt = user->tt;
  pi   = 4*atan(1.0);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0)); /* Fine grid */
  CHKERRQ(DMDAVecGetArray(da, b, &array));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = -PetscCosScalar(uu*pi*((PetscReal)i+0.5)*Hx)*PetscCosScalar(tt*pi*((PetscReal)j+0.5)*Hy)*Hx*Hy;
    }
  }
  CHKERRQ(DMDAVecRestoreArray(da, b, &array));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
  CHKERRQ(MatNullSpaceRemove(nullspace,b));
  CHKERRQ(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat jac,void *ctx)
{
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0));
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num=0; numi=0; numj=0;
        if (j!=0) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
          num++; numj++;
        }
        if (i!=0) {
          v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
          num++; numi++;
        }
        if (i!=M-1) {
          v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
          num++; numi++;
        }
        if (j!=N-1) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
          num++; numj++;
        }
        v[num] = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx); col[num].i = i;   col[num].j = j;
        num++;
        CHKERRQ(MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES));
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

  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
  CHKERRQ(MatSetNullSpace(J,nullspace));
  CHKERRQ(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type svd -ksp_view

   test:
      suffix: 2
      nsize: 4
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view

   test:
      suffix: 3
      nsize: 2
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 5 -mg_coarse_ksp_type cg -mg_coarse_ksp_converged_reason -mg_coarse_ksp_rtol 1e-2 -mg_coarse_ksp_max_it 5 -mg_coarse_pc_type none -pc_mg_levels 2 -ksp_type pipefgmres -ksp_pipefgmres_shift 1.5

   test:
      suffix: tut_1
      nsize: 1
      args: -da_grid_x 4 -da_grid_y 4 -mat_view

   test:
      suffix: tut_2
      requires: superlu_dist parmetis
      nsize: 4
      args: -da_grid_x 120 -da_grid_y 120 -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_view

   test:
      suffix: tut_3
      nsize: 4
      args: -da_grid_x 1025 -da_grid_y 1025 -pc_type mg -pc_mg_levels 9 -ksp_monitor

TEST*/
