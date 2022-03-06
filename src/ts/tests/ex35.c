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
  ierr = PetscOptionsInt("-Np", "Number of particles", "ex35.c", options->Np, &options->Np, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Number of dimensions", "ex35.c", options->dim_inp, &options->dim_inp, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  ref: http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
*/
PetscReal erfinv(PetscReal x)
{
  PetscReal      *ck, r = 0.;
  PetscInt       k, m, maxIter=100;
  PetscErrorCode ierr;

  ierr = PetscCalloc1(maxIter,&ck);CHKERRQ(ierr);
  ck[0] = 1;
  r = ck[0]*((PetscSqrtReal(PETSC_PI)/2.)*x);
  for (k = 1; k < maxIter; ++k){
    for (m = 0; m <= k-1; ++m){
      PetscReal denom = (m+1.)*(2.*m+1.);
      ck[k] += (ck[m]*ck[k-1-m])/denom;
    }
    PetscReal temp = 2.*k+1.;
    r += (ck[k]/temp)*PetscPowReal((PetscSqrtReal(PETSC_PI)/2.)*x,2.*k+1.);
  }
  ierr = PetscFree(ck);
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
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  Np = user.Np;
  dim = user.dim;

  ierr = PetscMalloc2(Np*dim, &x, Np*dim, &v);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &rngx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rngx, 0., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rngx);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rngx, 1034);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rngx);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &rng1);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rng1, 0., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rng1);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rng1, 3084);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rng1);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &rng2);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rng2, 0., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rng2);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rng2, 2397);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rng2);CHKERRQ(ierr);

  /* Set particle positions and velocities */
  if (user.dim_inp == 1) {
    for (p = 0; p < Np; ++p) {
      PetscReal temp;
      ierr = PetscRandomGetValueReal(rngx, &value);CHKERRQ(ierr);
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
    ierr = VecCreate(comm,&randVec);
    ierr = VecSetSizes(randVec,PETSC_DECIDE, Np*dim);
    ierr = VecSetFromOptions(randVec);

    ierr = ISCreateStride(comm, Np*dim/2, 0, 2, &isvx);CHKERRQ(ierr);
    ierr = ISCreateStride(comm, Np*dim/2, 1, 2, &isvy);CHKERRQ(ierr);
    ierr = VecGetSubVector(randVec, isvx, &subvecvx);CHKERRQ(ierr);
    ierr = VecGetSubVector(randVec, isvy, &subvecvy);CHKERRQ(ierr);
    ierr = VecSetRandom(subvecvx, rng1);CHKERRQ(ierr);
    ierr = VecSetRandom(subvecvy, rng2);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(randVec, isvx, &subvecvx);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(randVec, isvy, &subvecvy);CHKERRQ(ierr);
    ierr = VecGetArray(randVec, &randVecNums);CHKERRQ(ierr);

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
    ierr = ISDestroy(&isvx);CHKERRQ(ierr);
    ierr = ISDestroy(&isvy);CHKERRQ(ierr);
    ierr = VecDestroy(&subvecvx);CHKERRQ(ierr);
    ierr = VecDestroy(&subvecvy);CHKERRQ(ierr);
    ierr = VecDestroy(&randVec);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension %D", dim);

  ierr = PetscDrawCreate(comm, NULL, "monitor_particle_positions", 0,0,400,300, &positionDraw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(positionDraw);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(positionDraw, 10, &positionDrawSP);CHKERRQ(ierr);
  ierr = PetscDrawSPSetDimension(positionDrawSP,1);CHKERRQ(ierr);
  ierr = PetscDrawSPGetAxis(positionDrawSP, &axis);CHKERRQ(ierr);
  ierr = PetscDrawSPReset(positionDrawSP);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis, "Particles", "x", "y");CHKERRQ(ierr);
  ierr = PetscDrawSetSave(positionDraw, "ex35_pos.ppm");CHKERRQ(ierr);
  ierr = PetscDrawSPReset(positionDrawSP);CHKERRQ(ierr);
  ierr = PetscDrawSPSetLimits(positionDrawSP, 0, 1, 0, 1);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    speed = PetscSqrtReal(PetscSqr(v[p*dim]) + PetscSqr(v[p*dim+1]));
    ierr = PetscDrawSPAddPointColorized(positionDrawSP, &x[p*dim], &x[p*dim+1], &speed);CHKERRQ(ierr);
  }
  ierr = PetscDrawSPDraw(positionDrawSP, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDrawSave(positionDraw);CHKERRQ(ierr);

  ierr = PetscFree2(x, v);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rngx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rng1);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rng2);CHKERRQ(ierr);

  ierr = PetscDrawSPDestroy(&positionDrawSP);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&positionDraw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
