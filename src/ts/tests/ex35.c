static char help[] = "Test of Colorized Scatter Plot.\n";

#include "petscdraw.h"
#include "petscvec.h"
#include "petscis.h"

typedef struct {
  PetscInt  Np;          /* total number of particles */
  PetscInt  dim;
  PetscInt  dim_inp;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim  = 2;
  options->dim_inp = 2;
  options->Np   = 100;

  ierr = PetscOptionsBegin(comm, "", "Test of colorized scatter plot", "");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-Np", "Number of particles", "ex35.c", options->Np, &options->Np, PETSC_NULL));
  PetscCall(PetscOptionsInt("-dim", "Number of dimensions", "ex35.c", options->dim_inp, &options->dim_inp, PETSC_NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

/*
  ref: http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
*/
PetscReal erfinv(PetscReal x)
{
  PetscReal *ck, r   = 0.;
  PetscInt   maxIter = 100;

  PetscCall(PetscCalloc1(maxIter,&ck));
  ck[0] = 1;
  r = ck[0]*((PetscSqrtReal(PETSC_PI)/2.)*x);
  for (PetscInt k = 1; k < maxIter; ++k){
    for (PetscInt m = 0; m <= k-1; ++m){
      PetscReal denom = (m+1.)*(2.*m+1.);
      ck[k] += (ck[m]*ck[k-1-m])/denom;
    }
    PetscReal temp = 2.*k+1.;
    r += (ck[k]/temp)*PetscPowReal((PetscSqrtReal(PETSC_PI)/2.)*x,2.*k+1.);
  }
  PetscCallAbort(PETSC_COMM_SELF,PetscFree(ck));
  return r;
}

int main(int argc, char **argv)
{
  PetscInt          p, dim, Np;
  PetscScalar       *randVecNums;
  PetscReal         speed, value, *x, *v;
  PetscRandom       rngx, rng1, rng2;
  Vec               randVec, subvecvx, subvecvy;
  IS                isvx, isvy;
  AppCtx            user;
  PetscDrawAxis     axis;
  PetscDraw         positionDraw;
  PetscDrawSP       positionDrawSP;
  MPI_Comm          comm;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));

  Np = user.Np;
  dim = user.dim;

  PetscCall(PetscMalloc2(Np*dim, &x, Np*dim, &v));

  PetscCall(PetscRandomCreate(comm, &rngx));
  PetscCall(PetscRandomSetInterval(rngx, 0., 1.));
  PetscCall(PetscRandomSetFromOptions(rngx));
  PetscCall(PetscRandomSetSeed(rngx, 1034));
  PetscCall(PetscRandomSeed(rngx));

  PetscCall(PetscRandomCreate(comm, &rng1));
  PetscCall(PetscRandomSetInterval(rng1, 0., 1.));
  PetscCall(PetscRandomSetFromOptions(rng1));
  PetscCall(PetscRandomSetSeed(rng1, 3084));
  PetscCall(PetscRandomSeed(rng1));

  PetscCall(PetscRandomCreate(comm, &rng2));
  PetscCall(PetscRandomSetInterval(rng2, 0., 1.));
  PetscCall(PetscRandomSetFromOptions(rng2));
  PetscCall(PetscRandomSetSeed(rng2, 2397));
  PetscCall(PetscRandomSeed(rng2));

  /* Set particle positions and velocities */
  if (user.dim_inp == 1) {
    for (p = 0; p < Np; ++p) {
      PetscReal temp;
      PetscCall(PetscRandomGetValueReal(rngx, &value));
      x[p*dim] = value;
      x[p*dim+1] = 0.;
      temp = erfinv(2*value-1);
      v[p*dim] = temp;
      v[p*dim+1] = 0.;
    }
  } else if (user.dim_inp == 2) {
    /*
     Use Box-Muller to sample a distribution of velocities for the maxwellian.
     https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    */
    PetscCall(VecCreate(comm,&randVec));
    PetscCall(VecSetSizes(randVec,PETSC_DECIDE, Np*dim));
    PetscCall(VecSetFromOptions(randVec));

    PetscCall(ISCreateStride(comm, Np*dim/2, 0, 2, &isvx));
    PetscCall(ISCreateStride(comm, Np*dim/2, 1, 2, &isvy));
    PetscCall(VecGetSubVector(randVec, isvx, &subvecvx));
    PetscCall(VecGetSubVector(randVec, isvy, &subvecvy));
    PetscCall(VecSetRandom(subvecvx, rng1));
    PetscCall(VecSetRandom(subvecvy, rng2));
    PetscCall(VecRestoreSubVector(randVec, isvx, &subvecvx));
    PetscCall(VecRestoreSubVector(randVec, isvy, &subvecvy));
    PetscCall(VecGetArray(randVec, &randVecNums));

    for (p = 0; p < Np; ++p) {
      PetscReal u1, u2, mag, zx, zy;

      u1 = PetscRealPart(randVecNums[p*dim]);
      u2 = PetscRealPart(randVecNums[p*dim+1]);

      x[p*dim] = u1;
      x[p*dim+1] = u2;

      mag = PetscSqrtReal(-2.0 * PetscLogReal(u1));

      zx = mag * PetscCosReal(2*PETSC_PI*u2) + 0;
      zy = mag * PetscSinReal(2*PETSC_PI*u2) + 0;

      v[p*dim] = zx;
      v[p*dim+1] = zy;
    }
    PetscCall(ISDestroy(&isvx));
    PetscCall(ISDestroy(&isvy));
    PetscCall(VecDestroy(&subvecvx));
    PetscCall(VecDestroy(&subvecvy));
    PetscCall(VecDestroy(&randVec));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension %D", dim);

  PetscCall(PetscDrawCreate(comm, NULL, "monitor_particle_positions", 0,0,400,300, &positionDraw));
  PetscCall(PetscDrawSetFromOptions(positionDraw));
  PetscCall(PetscDrawSPCreate(positionDraw, 10, &positionDrawSP));
  PetscCall(PetscDrawSPSetDimension(positionDrawSP,1));
  PetscCall(PetscDrawSPGetAxis(positionDrawSP, &axis));
  PetscCall(PetscDrawSPReset(positionDrawSP));
  PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "y"));
  PetscCall(PetscDrawSetSave(positionDraw, "ex35_pos.ppm"));
  PetscCall(PetscDrawSPReset(positionDrawSP));
  PetscCall(PetscDrawSPSetLimits(positionDrawSP, 0, 1, 0, 1));
  for (p = 0; p < Np; ++p) {
    speed = PetscSqrtReal(PetscSqr(v[p*dim]) + PetscSqr(v[p*dim+1]));
    PetscCall(PetscDrawSPAddPointColorized(positionDrawSP, &x[p*dim], &x[p*dim+1], &speed));
  }
  PetscCall(PetscDrawSPDraw(positionDrawSP, PETSC_TRUE));
  PetscCall(PetscDrawSave(positionDraw));

  PetscCall(PetscFree2(x, v));
  PetscCall(PetscRandomDestroy(&rngx));
  PetscCall(PetscRandomDestroy(&rng1));
  PetscCall(PetscRandomDestroy(&rng2));

  PetscCall(PetscDrawSPDestroy(&positionDrawSP));
  PetscCall(PetscDrawDestroy(&positionDraw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
     suffix: 1D
     args: -Np 50\
     -dim 1
   test:
     suffix: 2D
     args: -Np 50\
     -dim 2
TEST*/
