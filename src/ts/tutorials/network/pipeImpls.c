#include "pipe.h"

/* Initial Function for PIPE       */
/*-------------------------------- */
/*
     Q(x) = Q0 (constant)
     H(x) = H0 - (R/gA) Q0*|Q0|* x
 */
/* ----------------------------------- */
PetscErrorCode PipeComputeSteadyState(Pipe pipe,PetscScalar Q0,PetscScalar H0)
{
  DM             cda;
  PipeField      *x;
  PetscInt       i,start,n;
  Vec            local;
  PetscScalar    *coords,c=pipe->R/(GRAV*pipe->A);

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(pipe->da, &cda));
  PetscCall(DMGetCoordinatesLocal(pipe->da, &local));
  PetscCall(DMDAVecGetArray(pipe->da, pipe->x, &x));
  PetscCall(DMDAVecGetArrayRead(cda, local, &coords));
  PetscCall(DMDAGetCorners(pipe->da, &start, 0, 0, &n, 0, 0));

  for (i = start; i < start + n; i++) {
    x[i].q = Q0;
    x[i].h = H0 - c * Q0 * PetscAbsScalar(Q0) * coords[i];
  }

  PetscCall(DMDAVecRestoreArray(pipe->da, pipe->x, &x));
  PetscCall(DMDAVecRestoreArrayRead(cda, local, &coords));
  PetscFunctionReturn(0);
}

/* Function evalutions for PIPE    */
/*-------------------------------- */
/* consider using a one-sided higher order fd derivative at boundary. */
static inline PetscScalar dqdx(PipeField *x,PetscInt i,PetscInt ilast,PetscReal dx)
{
  if (i == 0) {
    return (x[i+1].q - x[i].q) / dx;
  } else if (i == ilast) {
    return (x[i].q - x[i-1].q) / dx;
  } else {
    return (x[i+1].q - x[i-1].q) / (2*dx);
  }
}

static inline PetscScalar dhdx(PipeField *x,PetscInt i,PetscInt ilast,PetscReal dx)
{
  if (i == 0) {
    return (x[i+1].h - x[i].h) / dx;
  } else if (i == ilast) {
    return (x[i].h - x[i-1].h) / dx;
  } else {
    return (x[i+1].h - x[i-1].h) / (2*dx);
  }
}

PetscErrorCode PipeIFunctionLocal_Lax(DMDALocalInfo *info,PetscReal ptime,PipeField *x,PipeField *xdot,PetscScalar *f,Pipe pipe)
{
  PetscInt       i,start,n,ilast;
  PetscReal      a=pipe->a,A=pipe->A,R=pipe->R,c=a*a/(GRAV*A);
  PetscReal      dx=pipe->length/(info->mx-1),dt=pipe->dt;
  PetscScalar    qavg,xold_i,ha,hb,qa,qb;
  PipeField      *xold=pipe->xold;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(pipe->da, &start, 0, 0, &n, 0, 0));

  /* interior and boundary */
  ilast = start + n - 1;
  for (i = start + 1; i < start + n - 1; i++) {
    qavg = (xold[i+1].q + xold[i-1].q)/2.0;
    qa   = xold[i-1].q; qb   = xold[i+1].q;
    ha   = xold[i-1].h; hb   = xold[i+1].h;

    /* xdot[i].q = (x[i].q - old_i)/dt */
    xold_i = 0.5*(qa+qb);
    f[2*(i - 1) + 2] = (x[i].q - xold_i) + dt * (GRAV * pipe->A * dhdx(xold, i, ilast, dx) + pipe->R * qavg * PetscAbsScalar(qavg));

    /* xdot[i].h = (x[i].h - xold_i)/dt */
    xold_i = 0.5*(ha+hb);
    f[2*(i - 1) + 3] =  (x[i].h - xold_i) + dt * c * dqdx(xold, i, ilast, dx);
  }

  /* Characteristic equations */
  f[start + 1] = x[start].q - xold[start + 1].q - ((GRAV * A) / a)*(x[start].h - xold[start + 1].h) + dt*R*xold[start + 1].q * PetscAbsScalar(xold[start + 1].q);
  f[2*ilast]   = x[ilast].q - xold[ilast - 1].q + ((GRAV * A) / a)*(x[ilast].h - xold[ilast - 1].h) + dt*R*xold[ilast - 1].q * PetscAbsScalar(xold[ilast - 1].q);
  PetscFunctionReturn(0);
}
