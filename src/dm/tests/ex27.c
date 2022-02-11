static char help[] = "Test sequential USFFT interface on a uniform DMDA and compares the result to FFTW\n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc and the FFTW package, so configure
      must be run to enable these.

*/

#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>
int main(int argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char     *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  Mat            A, AA;
  PetscMPIInt    size;
  PetscInt       N,i, stencil=1,dof=1;
  PetscInt       dim[3] = {10,10,10}, ndim = 3;
  Vec            coords,x,y,z,xx,yy,zz;
  PetscReal      h[3];
  PetscScalar    s;
  PetscRandom    rdm;
  PetscReal      norm, enorm;
  PetscInt       func;
  FuncType       function = TANH;
  DM             da, coordsda;
  PetscBool      view_x = PETSC_FALSE, view_y = PETSC_FALSE, view_z = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "USFFT Options", "ex27");CHKERRQ(ierr);
  ierr     = PetscOptionsEList("-function", "Function type", "ex27", funcNames, NUM_FUNCS, funcNames[function], &func, NULL);CHKERRQ(ierr);
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);
  ierr     = PetscOptionsGetBool(NULL,NULL,"-view_x",&view_x,NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsGetBool(NULL,NULL,"-view_y",&view_y,NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsGetBool(NULL,NULL,"-view_z",&view_z,NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsGetIntArray(NULL,NULL,"-dim",dim,&ndim,NULL);CHKERRQ(ierr);

  ierr = DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,dim[0], dim[1], dim[2],
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,dof, stencil,NULL, NULL, NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /* Coordinates */
  ierr = DMGetCoordinateDM(da, &coordsda);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(coordsda, &coords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coords, "Grid coordinates");CHKERRQ(ierr);
  for (i = 0, N = 1; i < 3; i++) {
    h[i] = 1.0/dim[i];
    PetscScalar *a;
    ierr = VecGetArray(coords, &a);CHKERRQ(ierr);
    PetscInt j,k,n = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        for (k = 0; k < 3; ++k) {
          a[n] = j*h[i]; /* coordinate along the j-th point in the i-th dimension */
          ++n;
        }
      }
    }
    ierr = VecRestoreArray(coords, &a);CHKERRQ(ierr);

  }
  ierr = DMSetCoordinates(da, coords);CHKERRQ(ierr);

  /* Work vectors */
  ierr = DMGetGlobalVector(da, &x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &xx);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) xx, "Real space vector");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y, "USFFT frequency space vector");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &yy);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) yy, "FFTW frequency space vector");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &z);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) z, "USFFT reconstructed vector");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &zz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) zz, "FFTW reconstructed vector");CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF, "%3-D: USFFT on vector of ");CHKERRQ(ierr);
  for (i = 0, N = 1; i < 3; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "dim[%d] = %d ",i,dim[i]);CHKERRQ(ierr);
    N   *= dim[i];
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "; total size %d \n",N);CHKERRQ(ierr);

  if (function == RANDOM) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
    ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  } else if (function == CONSTANT) {
    ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  } else if (function == TANH) {
    PetscScalar *a;
    ierr = VecGetArray(x, &a);CHKERRQ(ierr);
    PetscInt j,k = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        a[k] = tanh((j - dim[i]/2.0)*(10.0/dim[i]));
        ++k;
      }
    }
    ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
  }
  if (view_x) {
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecCopy(x,xx);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|x|_2 = %g\n",norm);CHKERRQ(ierr);

  /* create USFFT object */
  ierr = MatCreateSeqUSFFT(coords,da,&A);CHKERRQ(ierr);
  /* create FFTW object */
  ierr = MatCreateSeqFFTW(PETSC_COMM_SELF,3,dim,&AA);CHKERRQ(ierr);

  /* apply USFFT and FFTW FORWARD "preemptively", so the fftw_plans can be reused on different vectors */
  ierr = MatMult(A,x,z);CHKERRQ(ierr);
  ierr = MatMult(AA,xx,zz);CHKERRQ(ierr);
  /* Now apply USFFT and FFTW forward several (3) times */
  for (i=0; i<3; ++i) {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = MatMult(AA,xx,yy);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
    ierr = MatMultTranspose(AA,yy,zz);CHKERRQ(ierr);
  }

  if (view_y) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "y = \n");CHKERRQ(ierr);
    ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "yy = \n");CHKERRQ(ierr);
    ierr = VecView(yy, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (view_z) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "z = \n");CHKERRQ(ierr);
    ierr = VecView(z, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "zz = \n");CHKERRQ(ierr);
    ierr = VecView(zz, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* compare x and z. USFFT computes an unnormalized DFT, thus z = N*x */
  s    = 1.0/(PetscReal)N;
  ierr = VecScale(z,s);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,z);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|x-z| = %g\n",enorm);CHKERRQ(ierr);

  /* compare xx and zz. FFTW computes an unnormalized DFT, thus zz = N*x */
  s    = 1.0/(PetscReal)N;
  ierr = VecScale(zz,s);CHKERRQ(ierr);
  ierr = VecAXPY(xx,-1.0,zz);CHKERRQ(ierr);
  ierr = VecNorm(xx,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|xx-zz| = %g\n",enorm);CHKERRQ(ierr);

  /* compare y and yy: USFFT and FFTW results*/
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,yy);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|y|_2 = %g\n",norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|y-yy| = %g\n",enorm);CHKERRQ(ierr);

  /* compare z and zz: USFFT and FFTW results*/
  ierr = VecNorm(z,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecAXPY(z,-1.0,zz);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|z|_2 = %g\n",norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "|z-zz| = %g\n",enorm);CHKERRQ(ierr);

  /* free spaces */
  ierr = DMRestoreGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&xx);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&y);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&yy);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&z);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&zz);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
