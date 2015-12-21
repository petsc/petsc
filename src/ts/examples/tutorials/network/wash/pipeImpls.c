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
  PetscErrorCode ierr;
  DM             cda;
  PipeField      *x;
  PetscInt       i,start,n;
  Vec            local;
  PetscScalar    *coords,c=pipe->R/(GRAV*pipe->A);

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(pipe->da, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(pipe->da, &local);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(pipe->da, pipe->x, &x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda, local, &coords);CHKERRQ(ierr);
  ierr = DMDAGetCorners(pipe->da, &start, 0, 0, &n, 0, 0);CHKERRQ(ierr);
  
  for (i = start; i < start + n; i++) {
    x[i].q = Q0;
    x[i].h = H0 - c * Q0 * PetscAbsScalar(Q0) * coords[i];
  }

  ierr = DMDAVecRestoreArray(pipe->da, pipe->x, &x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda, local, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Function evalutions for PIPE    */
/*-------------------------------- */
/* consider using a one-sided higher order fd derivative at boundary. */
PETSC_STATIC_INLINE PetscScalar dqdx(PipeField *x,PetscInt i,PetscInt ilast,PetscReal dx)
{
  if (i == 0) {
    return (x[i+1].q - x[i].q) / dx;
  } else if (i == ilast) {
    return (x[i].q - x[i-1].q) / dx;
  } else {
    return (x[i+1].q - x[i-1].q) / (2*dx);
  }
}

PETSC_STATIC_INLINE PetscScalar dhdx(PipeField *x,PetscInt i,PetscInt ilast,PetscReal dx)
{
  if (i == 0) {
    return (x[i+1].h - x[i].h) / dx;
  } else if (i == ilast) {
    return (x[i].h - x[i-1].h) / dx;
  } else {
    return (x[i+1].h - x[i-1].h) / (2*dx);
  }
}

PetscErrorCode PipeIFunctionLocal(DMDALocalInfo *info,PetscReal ptime,PipeField *x,PipeField *xdot,PipeField *f,Pipe pipe)
{
  PetscErrorCode ierr;
  PetscInt       i, start, n, ilast;
  PetscReal      c = (pipe->a * pipe->a) / (GRAV * pipe->A);
  PetscReal      dx = pipe->length / (info->mx-1);
  PetscScalar    qavg;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(pipe->da, &start, 0, 0, &n, 0, 0);CHKERRQ(ierr);
  
  /* interior and boundary */
  ilast = start + n -1;
  for (i = start; i < start + n; i++) {
    if (i == start || i == ilast) {
      qavg = x[i].q;
    } else {
      qavg = (x[i+1].q + x[i-1].q)/2.0; /* ok for single pipe with DM_BOUNDARY_GHOSTED, but mem corrupt for pipes! */
    }
    f[i].q = xdot[i].q + GRAV * pipe->A * dhdx(x, i, ilast, dx) + pipe->R * qavg * PetscAbsScalar(qavg);
    f[i].h = xdot[i].h + c * dqdx(x, i, ilast, dx);
  }

  /* up-stream boundary */
  if (info->xs == 0) {
    if (pipe->boundary.Q0 == PIPE_CHARACTERISTIC) {
      f[0].h = x[0].h - pipe->boundary.H0;
    } else {
      f[0].q = x[0].q - pipe->boundary.Q0;
    }
  }
  
  /* down-stream boundary */
  if (start + n == info->mx) {
    if (pipe->boundary.HL == PIPE_CHARACTERISTIC) {
      f[info->mx-1].q = x[info->mx-1].q - pipe->boundary.QL;
    } else {
      f[info->mx-1].h = x[info->mx-1].h - pipe->boundary.HL;
    }
  }
  PetscFunctionReturn(0);
}

