static const char help[] = "Solves obstacle problem in 2D as a variational inequality\n\
or nonlinear complementarity problem.  This is a form of the Laplace equation in\n\
which the solution u is constrained to be above a given function psi.  In the\n\
problem here an exact solution is known.\n";

/*  On a square S = {-2<x<2,-2<y<2}, the PDE
    u_{xx} + u_{yy} = 0
is solved on the set where membrane is above obstacle (u(x,y) >= psi(x,y)).
Here psi is the upper hemisphere of the unit ball.  On the boundary of S
we have Dirichlet boundary conditions from the exact solution.  Uses centered
FD scheme.  This example contributed by Ed Bueler.

Example usage:
  * get help:
    ./ex9 -help
  * monitor run:
    ./ex9 -da_refine 2 -snes_vi_monitor
  * use other SNESVI type (default is SNESVINEWTONRSLS):
    ./ex9 -da_refine 2 -snes_vi_monitor -snes_type vinewtonssls
  * use FD evaluation of Jacobian by coloring, instead of analytical:
    ./ex9 -da_refine 2 -snes_fd_color
  * X windows visualizations:
    ./ex9 -snes_monitor_solution draw -draw_pause 1 -da_refine 4
    ./ex9 -snes_vi_monitor_residual -draw_pause 1 -da_refine 4
  * full-cycle multigrid:
    ./ex9 -snes_converged_reason -snes_grid_sequence 4 -pc_type mg
  * serial convergence evidence:
    for M in 3 4 5 6 7; do ./ex9 -snes_grid_sequence $M -pc_type mg; done
  * FIXME sporadic parallel bug:
    mpiexec -n 4 ./ex9 -snes_converged_reason -snes_grid_sequence 4 -pc_type mg
*/

#include <petsc.h>

/* z = psi(x,y) is the hemispherical obstacle, but made C^1 with "skirt" at r=r0 */
PetscReal psi(PetscReal x, PetscReal y)
{
    const PetscReal  r = x * x + y * y,r0 = 0.9,psi0 = PetscSqrtReal(1.0 - r0*r0),dpsi0 = - r0 / psi0;
    if (r <= r0) {
      return PetscSqrtReal(1.0 - r);
    } else {
      return psi0 + dpsi0 * (r - r0);
    }
}

/*  This exact solution solves a 1D radial free-boundary problem for the
Laplace equation, on the interval 0 < r < 2, with above obstacle psi(x,y).
The Laplace equation applies where u(r) > psi(r),
    u''(r) + r^-1 u'(r) = 0
with boundary conditions including free b.c.s at an unknown location r = a:
    u(a) = psi(a),  u'(a) = psi'(a),  u(2) = 0
The solution is  u(r) = - A log(r) + B   on  r > a.  The boundary conditions
can then be reduced to a root-finding problem for a:
    a^2 (log(2) - log(a)) = 1 - a^2
The solution is a = 0.697965148223374 (giving residual 1.5e-15).  Then
A = a^2*(1-a^2)^(-0.5) and B = A*log(2) are as given below in the code.  */
PetscReal u_exact(PetscReal x, PetscReal y)
{
    const PetscReal afree = 0.697965148223374,
                    A     = 0.680259411891719,
                    B     = 0.471519893402112;
    PetscReal  r;
    r = PetscSqrtReal(x * x + y * y);
    return (r <= afree) ? psi(x,y)  /* active set; on the obstacle */
                        : - A * PetscLogReal(r) + B; /* solves laplace eqn */
}

extern PetscErrorCode FormExactSolution(DMDALocalInfo*,Vec);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscReal**,PetscReal**,void*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscReal**,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode      ierr;
  SNES                snes;
  DM                  da, da_after;
  Vec                 u, u_exact;
  DMDALocalInfo       info;
  PetscReal           error1,errorinf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,5,5, /* 5x5 coarse grid; override with -da_grid_x,_y */
                      PETSC_DECIDE,PETSC_DECIDE,
                      1,1,  /* dof=1 and s = 1 (stencil extends out one cell) */
                      NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,0.0,1.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(DMDASNESFunction)FormFunctionLocal,NULL);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,(DMDASNESJacobian)FormJacobianLocal,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* solve nonlinear system */
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  /* DMDA after solve may be different, e.g. with -snes_grid_sequence */
  ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr); /* do not destroy u */
  ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
  ierr = FormExactSolution(&info,u_exact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,u_exact);CHKERRQ(ierr); /* u <-- u - u_exact */
  ierr = VecNorm(u,NORM_1,&error1);CHKERRQ(ierr);
  error1 /= (PetscReal)info.mx * (PetscReal)info.my; /* average error */
  ierr = VecNorm(u,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"errors on %D x %D grid:  av |u-uexact|  = %.3e,  |u-uexact|_inf = %.3e\n",info.mx,info.my,(double)error1,(double)errorinf);CHKERRQ(ierr);
  ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FormExactSolution(DMDALocalInfo *info, Vec u)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      **au, dx, dy, x, y;
  dx = 4.0 / (PetscReal)(info->mx-1);
  dy = 4.0 / (PetscReal)(info->my-1);
  ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = -2.0 + j * dy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = -2.0 + i * dx;
      au[j][i] = u_exact(x,y);
    }
  }
  ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu)
{
  PetscErrorCode ierr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i, j;
  PetscReal      **aXl, dx, dy, x, y;

  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = 4.0 / (PetscReal)(info.mx-1);
  dy = 4.0 / (PetscReal)(info.my-1);
  ierr = DMDAVecGetArray(da, Xl, &aXl);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = -2.0 + j * dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = -2.0 + i * dx;
      aXl[j][i] = psi(x,y);
    }
  }
  ierr = DMDAVecRestoreArray(da, Xl, &aXl);CHKERRQ(ierr);
  ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar **au, PetscScalar **af, void *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      dx,dy,x,y,ue,un,us,uw;

  PetscFunctionBeginUser;
  dx = 4.0 / (PetscReal)(info->mx-1);
  dy = 4.0 / (PetscReal)(info->my-1);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = -2.0 + j * dy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = -2.0 + i * dx;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        af[j][i] = 4.0 * (au[j][i] - u_exact(x,y));
      } else {
        uw = (i-1 == 0)          ? u_exact(x-dx,y) : au[j][i-1];
        ue = (i+1 == info->mx-1) ? u_exact(x+dx,y) : au[j][i+1];
        us = (j-1 == 0)          ? u_exact(x,y-dy) : au[j-1][i];
        un = (j+1 == info->my-1) ? u_exact(x,y+dy) : au[j+1][i];
        af[j][i] = - (dy/dx) * (uw - 2.0 * au[j][i] + ue) - (dx/dy) * (us - 2.0 * au[j][i] + un);
      }
    }
  }
  ierr = PetscLogFlops(12.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **au, Mat A, Mat jac, void *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j,n;
  MatStencil     col[5],row;
  PetscReal      v[5],dx,dy,oxx,oyy;

  PetscFunctionBeginUser;
  dx  = 4.0 / (PetscReal)(info->mx-1);
  dy  = 4.0 / (PetscReal)(info->my-1);
  oxx = dy / dx;
  oyy = dx / dy;
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      row.j = j; row.i = i;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) { /* boundary */
        v[0] = 4.0;
        ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
      } else { /* interior grid points */
        v[0] = 2.0 * (oxx + oyy);  col[0].j = j;  col[0].i = i;
        n = 1;
        if (i-1 > 0) {
          v[n] = -oxx;  col[n].j = j;  col[n++].i = i-1;
        }
        if (i+1 < info->mx-1) {
          v[n] = -oxx;  col[n].j = j;  col[n++].i = i+1;
        }
        if (j-1 > 0) {
          v[n] = -oyy;  col[n].j = j-1;  col[n++].i = i;
        }
        if (j+1 < info->my-1) {
          v[n] = -oyy;  col[n].j = j+1;  col[n++].i = i;
        }
        ierr = MatSetValuesStencil(jac,1,&row,n,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* Assemble matrix, using the 2-step process: */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != jac) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(2.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      requires: !single
      nsize: 1
      args: -da_refine 1 -snes_monitor_short -snes_type vinewtonrsls

   test:
      suffix: 2
      requires: !single
      nsize: 2
      args: -da_refine 1 -snes_monitor_short -snes_type vinewtonssls

   test:
      suffix: 3
      requires: !single
      nsize: 2
      args: -snes_grid_sequence 2 -snes_vi_monitor -snes_type vinewtonrsls

   test:
      suffix: mg
      requires: !single
      nsize: 4
      args: -snes_grid_sequence 3 -snes_converged_reason -pc_type mg

   test:
      suffix: 4
      nsize: 1
      args: -mat_is_symmetric

   test:
      suffix: 5
      nsize: 1
      args: -ksp_converged_reason -snes_fd_color

   test:
      suffix: 6
      requires: !single
      nsize: 2
      args: -snes_grid_sequence 2 -pc_type mg -snes_monitor_short -ksp_converged_reason

   test:
      suffix: 7
      nsize: 2
      args: -da_refine 1 -snes_monitor_short -snes_type composite -snes_composite_type multiplicative -snes_composite_sneses vinewtonrsls,vinewtonssls -sub_0_snes_vi_monitor -sub_1_snes_vi_monitor
      TODO: fix nasty memory leak in SNESCOMPOSITE

   test:
      suffix: 8
      nsize: 2
      args: -da_refine 1 -snes_monitor_short -snes_type composite -snes_composite_type additive -snes_composite_sneses vinewtonrsls -sub_0_snes_vi_monitor
      TODO: fix nasty memory leak in SNESCOMPOSITE

   test:
      suffix: 9
      nsize: 2
      args: -da_refine 1 -snes_monitor_short -snes_type composite -snes_composite_type additiveoptimal -snes_composite_sneses vinewtonrsls -sub_0_snes_vi_monitor
      TODO: fix nasty memory leak in SNESCOMPOSITE

TEST*/
