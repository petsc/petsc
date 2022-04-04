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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "USFFT Options", "ex27");PetscCall(ierr);
  PetscCall(PetscOptionsEList("-function", "Function type", "ex27", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();PetscCall(ierr);
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_x",&view_x,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_y",&view_y,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_z",&view_z,NULL));
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,"-dim",dim,&ndim,NULL));

  ierr = DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,dim[0], dim[1], dim[2],
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,dof, stencil,NULL, NULL, NULL,&da);PetscCall(ierr);
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /* Coordinates */
  PetscCall(DMGetCoordinateDM(da, &coordsda));
  PetscCall(DMGetGlobalVector(coordsda, &coords));
  PetscCall(PetscObjectSetName((PetscObject) coords, "Grid coordinates"));
  for (i = 0, N = 1; i < 3; i++) {
    h[i] = 1.0/dim[i];
    PetscScalar *a;
    PetscCall(VecGetArray(coords, &a));
    PetscInt j,k,n = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        for (k = 0; k < 3; ++k) {
          a[n] = j*h[i]; /* coordinate along the j-th point in the i-th dimension */
          ++n;
        }
      }
    }
    PetscCall(VecRestoreArray(coords, &a));

  }
  PetscCall(DMSetCoordinates(da, coords));

  /* Work vectors */
  PetscCall(DMGetGlobalVector(da, &x));
  PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector"));
  PetscCall(DMGetGlobalVector(da, &xx));
  PetscCall(PetscObjectSetName((PetscObject) xx, "Real space vector"));
  PetscCall(DMGetGlobalVector(da, &y));
  PetscCall(PetscObjectSetName((PetscObject) y, "USFFT frequency space vector"));
  PetscCall(DMGetGlobalVector(da, &yy));
  PetscCall(PetscObjectSetName((PetscObject) yy, "FFTW frequency space vector"));
  PetscCall(DMGetGlobalVector(da, &z));
  PetscCall(PetscObjectSetName((PetscObject) z, "USFFT reconstructed vector"));
  PetscCall(DMGetGlobalVector(da, &zz));
  PetscCall(PetscObjectSetName((PetscObject) zz, "FFTW reconstructed vector"));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%3-D: USFFT on vector of "));
  for (i = 0, N = 1; i < 3; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "dim[%d] = %d ",i,dim[i]));
    N   *= dim[i];
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "; total size %d \n",N));

  if (function == RANDOM) {
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
    PetscCall(PetscRandomSetFromOptions(rdm));
    PetscCall(VecSetRandom(x, rdm));
    PetscCall(PetscRandomDestroy(&rdm));
  } else if (function == CONSTANT) {
    PetscCall(VecSet(x, 1.0));
  } else if (function == TANH) {
    PetscScalar *a;
    PetscCall(VecGetArray(x, &a));
    PetscInt j,k = 0;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < dim[i]; ++j) {
        a[k] = tanh((j - dim[i]/2.0)*(10.0/dim[i]));
        ++k;
      }
    }
    PetscCall(VecRestoreArray(x, &a));
  }
  if (view_x) {
    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecCopy(x,xx));

  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|x|_2 = %g\n",norm));

  /* create USFFT object */
  PetscCall(MatCreateSeqUSFFT(coords,da,&A));
  /* create FFTW object */
  PetscCall(MatCreateSeqFFTW(PETSC_COMM_SELF,3,dim,&AA));

  /* apply USFFT and FFTW FORWARD "preemptively", so the fftw_plans can be reused on different vectors */
  PetscCall(MatMult(A,x,z));
  PetscCall(MatMult(AA,xx,zz));
  /* Now apply USFFT and FFTW forward several (3) times */
  for (i=0; i<3; ++i) {
    PetscCall(MatMult(A,x,y));
    PetscCall(MatMult(AA,xx,yy));
    PetscCall(MatMultTranspose(A,y,z));
    PetscCall(MatMultTranspose(AA,yy,zz));
  }

  if (view_y) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "y = \n"));
    PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "yy = \n"));
    PetscCall(VecView(yy, PETSC_VIEWER_STDOUT_WORLD));
  }

  if (view_z) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "z = \n"));
    PetscCall(VecView(z, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "zz = \n"));
    PetscCall(VecView(zz, PETSC_VIEWER_STDOUT_WORLD));
  }

  /* compare x and z. USFFT computes an unnormalized DFT, thus z = N*x */
  s    = 1.0/(PetscReal)N;
  PetscCall(VecScale(z,s));
  PetscCall(VecAXPY(x,-1.0,z));
  PetscCall(VecNorm(x,NORM_1,&enorm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|x-z| = %g\n",enorm));

  /* compare xx and zz. FFTW computes an unnormalized DFT, thus zz = N*x */
  s    = 1.0/(PetscReal)N;
  PetscCall(VecScale(zz,s));
  PetscCall(VecAXPY(xx,-1.0,zz));
  PetscCall(VecNorm(xx,NORM_1,&enorm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|xx-zz| = %g\n",enorm));

  /* compare y and yy: USFFT and FFTW results*/
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(VecAXPY(y,-1.0,yy));
  PetscCall(VecNorm(y,NORM_1,&enorm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|y|_2 = %g\n",norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|y-yy| = %g\n",enorm));

  /* compare z and zz: USFFT and FFTW results*/
  PetscCall(VecNorm(z,NORM_2,&norm));
  PetscCall(VecAXPY(z,-1.0,zz));
  PetscCall(VecNorm(z,NORM_1,&enorm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|z|_2 = %g\n",norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "|z-zz| = %g\n",enorm));

  /* free spaces */
  PetscCall(DMRestoreGlobalVector(da,&x));
  PetscCall(DMRestoreGlobalVector(da,&xx));
  PetscCall(DMRestoreGlobalVector(da,&y));
  PetscCall(DMRestoreGlobalVector(da,&yy));
  PetscCall(DMRestoreGlobalVector(da,&z));
  PetscCall(DMRestoreGlobalVector(da,&zz));
  PetscCall(VecDestroy(&coords));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
