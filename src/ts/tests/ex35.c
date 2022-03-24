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

  ierr = PetscOptionsBegin(comm, "", "Test of colorized scatter plot", "");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-Np", "Number of particles", "ex35.c", options->Np, &options->Np, PETSC_NULL));
  CHKERRQ(PetscOptionsInt("-dim", "Number of dimensions", "ex35.c", options->dim_inp, &options->dim_inp, PETSC_NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  ref: http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
*/
PetscReal erfinv(PetscReal x)
{
  PetscReal *ck, r   = 0.;
  PetscInt   maxIter = 100;

  CHKERRQ(PetscCalloc1(maxIter,&ck));
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
  CHKERRABORT(PETSC_COMM_SELF,PetscFree(ck));
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

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));

  Np = user.Np;
  dim = user.dim;

  CHKERRQ(PetscMalloc2(Np*dim, &x, Np*dim, &v));

  CHKERRQ(PetscRandomCreate(comm, &rngx));
  CHKERRQ(PetscRandomSetInterval(rngx, 0., 1.));
  CHKERRQ(PetscRandomSetFromOptions(rngx));
  CHKERRQ(PetscRandomSetSeed(rngx, 1034));
  CHKERRQ(PetscRandomSeed(rngx));

  CHKERRQ(PetscRandomCreate(comm, &rng1));
  CHKERRQ(PetscRandomSetInterval(rng1, 0., 1.));
  CHKERRQ(PetscRandomSetFromOptions(rng1));
  CHKERRQ(PetscRandomSetSeed(rng1, 3084));
  CHKERRQ(PetscRandomSeed(rng1));

  CHKERRQ(PetscRandomCreate(comm, &rng2));
  CHKERRQ(PetscRandomSetInterval(rng2, 0., 1.));
  CHKERRQ(PetscRandomSetFromOptions(rng2));
  CHKERRQ(PetscRandomSetSeed(rng2, 2397));
  CHKERRQ(PetscRandomSeed(rng2));

  /* Set particle positions and velocities */
  if (user.dim_inp == 1) {
    for (p = 0; p < Np; ++p) {
      PetscReal temp;
      CHKERRQ(PetscRandomGetValueReal(rngx, &value));
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
    CHKERRQ(VecCreate(comm,&randVec));
    CHKERRQ(VecSetSizes(randVec,PETSC_DECIDE, Np*dim));
    CHKERRQ(VecSetFromOptions(randVec));

    CHKERRQ(ISCreateStride(comm, Np*dim/2, 0, 2, &isvx));
    CHKERRQ(ISCreateStride(comm, Np*dim/2, 1, 2, &isvy));
    CHKERRQ(VecGetSubVector(randVec, isvx, &subvecvx));
    CHKERRQ(VecGetSubVector(randVec, isvy, &subvecvy));
    CHKERRQ(VecSetRandom(subvecvx, rng1));
    CHKERRQ(VecSetRandom(subvecvy, rng2));
    CHKERRQ(VecRestoreSubVector(randVec, isvx, &subvecvx));
    CHKERRQ(VecRestoreSubVector(randVec, isvy, &subvecvy));
    CHKERRQ(VecGetArray(randVec, &randVecNums));

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
    CHKERRQ(ISDestroy(&isvx));
    CHKERRQ(ISDestroy(&isvy));
    CHKERRQ(VecDestroy(&subvecvx));
    CHKERRQ(VecDestroy(&subvecvy));
    CHKERRQ(VecDestroy(&randVec));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension %D", dim);

  CHKERRQ(PetscDrawCreate(comm, NULL, "monitor_particle_positions", 0,0,400,300, &positionDraw));
  CHKERRQ(PetscDrawSetFromOptions(positionDraw));
  CHKERRQ(PetscDrawSPCreate(positionDraw, 10, &positionDrawSP));
  CHKERRQ(PetscDrawSPSetDimension(positionDrawSP,1));
  CHKERRQ(PetscDrawSPGetAxis(positionDrawSP, &axis));
  CHKERRQ(PetscDrawSPReset(positionDrawSP));
  CHKERRQ(PetscDrawAxisSetLabels(axis, "Particles", "x", "y"));
  CHKERRQ(PetscDrawSetSave(positionDraw, "ex35_pos.ppm"));
  CHKERRQ(PetscDrawSPReset(positionDrawSP));
  CHKERRQ(PetscDrawSPSetLimits(positionDrawSP, 0, 1, 0, 1));
  for (p = 0; p < Np; ++p) {
    speed = PetscSqrtReal(PetscSqr(v[p*dim]) + PetscSqr(v[p*dim+1]));
    CHKERRQ(PetscDrawSPAddPointColorized(positionDrawSP, &x[p*dim], &x[p*dim+1], &speed));
  }
  CHKERRQ(PetscDrawSPDraw(positionDrawSP, PETSC_TRUE));
  CHKERRQ(PetscDrawSave(positionDraw));

  CHKERRQ(PetscFree2(x, v));
  CHKERRQ(PetscRandomDestroy(&rngx));
  CHKERRQ(PetscRandomDestroy(&rng1));
  CHKERRQ(PetscRandomDestroy(&rng2));

  CHKERRQ(PetscDrawSPDestroy(&positionDrawSP));
  CHKERRQ(PetscDrawDestroy(&positionDraw));
  CHKERRQ(PetscFinalize());
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
