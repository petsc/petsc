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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "USFFT Options", "ex27");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-function", "Function type", "ex27", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_x",&view_x,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_y",&view_y,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_z",&view_z,NULL));
  CHKERRQ(PetscOptionsGetIntArray(NULL,NULL,"-dim",dim,&ndim,NULL));

  ierr = DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,dim[0], dim[1], dim[2],
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,dof, stencil,NULL, NULL, NULL,&da);CHKERRQ(ierr);
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /* Coordinates */
  CHKERRQ(DMGetCoordinateDM(da, &coordsda));
  CHKERRQ(DMGetGlobalVector(coordsda, &coords));
  CHKERRQ(PetscObjectSetName((PetscObject) coords, "Grid coordinates"));
  for (i = 0, N = 1; i < 3; i++) {
    h[i] = 1.0/dim[i];
    PetscScalar *a;
    CHKERRQ(VecGetArray(coords, &a));
    PetscInt j,k,n = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        for (k = 0; k < 3; ++k) {
          a[n] = j*h[i]; /* coordinate along the j-th point in the i-th dimension */
          ++n;
        }
      }
    }
    CHKERRQ(VecRestoreArray(coords, &a));

  }
  CHKERRQ(DMSetCoordinates(da, coords));

  /* Work vectors */
  CHKERRQ(DMGetGlobalVector(da, &x));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector"));
  CHKERRQ(DMGetGlobalVector(da, &xx));
  CHKERRQ(PetscObjectSetName((PetscObject) xx, "Real space vector"));
  CHKERRQ(DMGetGlobalVector(da, &y));
  CHKERRQ(PetscObjectSetName((PetscObject) y, "USFFT frequency space vector"));
  CHKERRQ(DMGetGlobalVector(da, &yy));
  CHKERRQ(PetscObjectSetName((PetscObject) yy, "FFTW frequency space vector"));
  CHKERRQ(DMGetGlobalVector(da, &z));
  CHKERRQ(PetscObjectSetName((PetscObject) z, "USFFT reconstructed vector"));
  CHKERRQ(DMGetGlobalVector(da, &zz));
  CHKERRQ(PetscObjectSetName((PetscObject) zz, "FFTW reconstructed vector"));

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%3-D: USFFT on vector of "));
  for (i = 0, N = 1; i < 3; i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "dim[%d] = %d ",i,dim[i]));
    N   *= dim[i];
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "; total size %d \n",N));

  if (function == RANDOM) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
    CHKERRQ(PetscRandomSetFromOptions(rdm));
    CHKERRQ(VecSetRandom(x, rdm));
    CHKERRQ(PetscRandomDestroy(&rdm));
  } else if (function == CONSTANT) {
    CHKERRQ(VecSet(x, 1.0));
  } else if (function == TANH) {
    PetscScalar *a;
    CHKERRQ(VecGetArray(x, &a));
    PetscInt j,k = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        a[k] = tanh((j - dim[i]/2.0)*(10.0/dim[i]));
        ++k;
      }
    }
    CHKERRQ(VecRestoreArray(x, &a));
  }
  if (view_x) {
    CHKERRQ(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(VecCopy(x,xx));

  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|x|_2 = %g\n",norm));

  /* create USFFT object */
  CHKERRQ(MatCreateSeqUSFFT(coords,da,&A));
  /* create FFTW object */
  CHKERRQ(MatCreateSeqFFTW(PETSC_COMM_SELF,3,dim,&AA));

  /* apply USFFT and FFTW FORWARD "preemptively", so the fftw_plans can be reused on different vectors */
  CHKERRQ(MatMult(A,x,z));
  CHKERRQ(MatMult(AA,xx,zz));
  /* Now apply USFFT and FFTW forward several (3) times */
  for (i=0; i<3; ++i) {
    CHKERRQ(MatMult(A,x,y));
    CHKERRQ(MatMult(AA,xx,yy));
    CHKERRQ(MatMultTranspose(A,y,z));
    CHKERRQ(MatMultTranspose(AA,yy,zz));
  }

  if (view_y) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "y = \n"));
    CHKERRQ(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "yy = \n"));
    CHKERRQ(VecView(yy, PETSC_VIEWER_STDOUT_WORLD));
  }

  if (view_z) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "z = \n"));
    CHKERRQ(VecView(z, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "zz = \n"));
    CHKERRQ(VecView(zz, PETSC_VIEWER_STDOUT_WORLD));
  }

  /* compare x and z. USFFT computes an unnormalized DFT, thus z = N*x */
  s    = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(z,s));
  CHKERRQ(VecAXPY(x,-1.0,z));
  CHKERRQ(VecNorm(x,NORM_1,&enorm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|x-z| = %g\n",enorm));

  /* compare xx and zz. FFTW computes an unnormalized DFT, thus zz = N*x */
  s    = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(zz,s));
  CHKERRQ(VecAXPY(xx,-1.0,zz));
  CHKERRQ(VecNorm(xx,NORM_1,&enorm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|xx-zz| = %g\n",enorm));

  /* compare y and yy: USFFT and FFTW results*/
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(VecAXPY(y,-1.0,yy));
  CHKERRQ(VecNorm(y,NORM_1,&enorm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|y|_2 = %g\n",norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|y-yy| = %g\n",enorm));

  /* compare z and zz: USFFT and FFTW results*/
  CHKERRQ(VecNorm(z,NORM_2,&norm));
  CHKERRQ(VecAXPY(z,-1.0,zz));
  CHKERRQ(VecNorm(z,NORM_1,&enorm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|z|_2 = %g\n",norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "|z-zz| = %g\n",enorm));

  /* free spaces */
  CHKERRQ(DMRestoreGlobalVector(da,&x));
  CHKERRQ(DMRestoreGlobalVector(da,&xx));
  CHKERRQ(DMRestoreGlobalVector(da,&y));
  CHKERRQ(DMRestoreGlobalVector(da,&yy));
  CHKERRQ(DMRestoreGlobalVector(da,&z));
  CHKERRQ(DMRestoreGlobalVector(da,&zz));
  CHKERRQ(VecDestroy(&coords));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
