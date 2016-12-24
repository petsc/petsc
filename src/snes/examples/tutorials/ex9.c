static const char help[] = "Solves obstacle problem in 2D as a variational inequality.\n\
An elliptic problem with solution  u  constrained to be above a given function  psi. \n\
Exact solution is known.\n";

/*  Solve on a square R = {-2<x<2,-2<y<2}:
    u_{xx} + u_{yy} = 0
on the set where membrane is above obstacle.  Constraint is  u(x,y) >= psi(x,y).
Here psi is the upper hemisphere of the unit ball.  On the boundary of R
we have nonhomogenous Dirichlet boundary conditions coming from the exact solution.

Method is centered finite differences.

This example was contributed by Ed Bueler.  The exact solution is known for the
given psi and boundary values in question.  See
  https://github.com/bueler/fem-code-challenge/blob/master/obstacleDOC.pdf?raw=true.

Example usage follows.

Get help:
  ./ex9 -help

Monitor run:
  ./ex9 -snes_converged_reason -snes_monitor -snes_vi_monitor

Use finite difference evaluation of Jacobian by coloring, instead of analytical:
  ./ex9 -snes_fd_color

Graphical:
  ./ex9 -snes_monitor_solution draw -draw_pause 1 -da_refine 2

Convergence evidence:
  for M in 1 2 3 4 5; do mpiexec -n 4 ./ex9 -da_refine $M; done
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/* application context for obstacle problem solver */
typedef struct {
  Vec psi, uexact;
} ObsCtx;

extern PetscErrorCode FormPsiAndExactSoln(DM);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,ObsCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar**,Mat,Mat,ObsCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode      ierr;
  ObsCtx              user;
  SNES                snes;
  DM                  da;
  Vec                 u,     /* solution */
                      Xu;    /* upper bound */
  DMDALocalInfo       info;
  PetscReal           error1,errorinf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,/* default to 10x10 grid */
                      PETSC_DECIDE,PETSC_DECIDE, /* number of processors in each dimension */1,/* dof = 1 */1,/* s = 1; stencil extends out one cell */
                      NULL,NULL,/* do not specify processor decomposition */&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.uexact));CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.psi));CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  ierr = FormPsiAndExactSoln(da);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);

  /* set upper and lower bound constraints for VI */
  ierr = VecDuplicate(u,&Xu);CHKERRQ(ierr);
  ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,user.psi,Xu);CHKERRQ(ierr);
  ierr = VecDestroy(&Xu);CHKERRQ(ierr);

  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* report on setup */
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"setup done: grid  Mx,My = %D,%D  with spacing  dx,dy = %.4f,%.4f\n",
                     info.mx,info.my,(double)(4.0/(PetscReal)(info.mx-1)),(double)(4.0/(PetscReal)(info.my-1)));CHKERRQ(ierr);

  /* solve nonlinear system */
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

  /* compare to exact */
  ierr = VecAXPY(u,-1.0,user.uexact);CHKERRQ(ierr); /* u <- u - uexact */
  ierr = VecNorm(u,NORM_1,&error1);CHKERRQ(ierr);
  error1 /= (PetscReal)info.mx * (PetscReal)info.my;
  ierr = VecNorm(u,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"errors:     av |u-uexact|  = %.3e    |u-uexact|_inf = %.3e\n",(double)error1,(double)errorinf);CHKERRQ(ierr);

  /* Free work space.  */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&(user.psi));CHKERRQ(ierr);
  ierr = VecDestroy(&(user.uexact));CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormPsiAndExactSoln(DM da) {
  ObsCtx         *user;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i,j;
  DM             coordDA;
  Vec            coordinates;
  DMDACoor2d     **coords;
  PetscReal      **psi, **uexact, r;
  const PetscReal afree = 0.69797, A = 0.68026, B = 0.47152;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da, &coordDA);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da, &coordinates);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, user->psi, &psi);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, user->uexact, &uexact);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      r = PetscSqrtReal(PetscPowScalarInt(coords[j][i].x,2) + PetscPowScalarInt(coords[j][i].y,2));
      if (r <= 1.0) psi[j][i] = PetscSqrtReal(1.0 - r * r);
      else psi[j][i] = -1.0;
      if (r <= afree) uexact[j][i] = psi[j][i];  /* on the obstacle */
      else uexact[j][i] = - A * PetscLogReal(r) + B;   /* solves the laplace eqn */
    }
  }
  ierr = DMDAVecRestoreArray(da, user->psi, &psi);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, user->uexact, &uexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,ObsCtx *user) {
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      dx,dy,uxx,uyy;
  PetscReal      **uexact;  /* used for boundary values only */

  PetscFunctionBeginUser;
  dx = 4.0 / (PetscReal)(info->mx-1);
  dy = 4.0 / (PetscReal)(info->my-1);

  ierr = DMDAVecGetArray(info->da, user->uexact, &uexact);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = 4.0*(x[j][i] - uexact[j][i]);
      } else {
        uxx     = dy*(x[j][i-1] - 2.0 * x[j][i] + x[j][i+1]) / dx;
        uyy     = dx*(x[j-1][i] - 2.0 * x[j][i] + x[j+1][i]) / dy;
        f[j][i] = -uxx - uyy;
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, user->uexact, &uexact);CHKERRQ(ierr);

  ierr = PetscLogFlops(10.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* FormJacobianLocal - Evaluates Jacobian matrix on local process patch */
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar **x,Mat A,Mat jac, ObsCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
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
        v[0] = -oyy;                 col[0].j = j - 1;  col[0].i = i;
        v[1] = -oxx;                 col[1].j = j;      col[1].i = i - 1;
        v[2] = 2.0 * (oxx + oyy);    col[2].j = j;      col[2].i = i;
        v[3] = -oxx;                 col[3].j = j;      col[3].i = i + 1;
        v[4] = -oyy;                 col[4].j = j + 1;  col[4].i = i;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
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

