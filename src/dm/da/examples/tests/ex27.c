static char help[] = "Test sequential USFFT interface on a uniform DA and compares the result to FFTW\n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc and the FFTW package, so configure
      must be run to enable these.

*/

#include "petscmat.h"
#include "petscda.h"
#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char    *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  Mat            A, AA;    
  PetscMPIInt    size;
  PetscInt       n = 10,N,dim[3],i, stencil=1,dof=1;
  Vec            x,y,z,yy,zz;
  PetscScalar    s;  
  PetscRandom    rdm;
  PetscReal      enorm;
  PetscInt       func;
  FuncType       function = RANDOM;
  PetscTruth     view = PETSC_FALSE;
  DA             da;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_ERR_SUP, "This example requires complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "USFFT Options", "ex124");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-function", "Function type", "ex124", funcNames, NUM_FUNCS, funcNames[function], &func, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-vec_view_draw", "View the functions", "ex124", view, &view, PETSC_NULL);CHKERRQ(ierr);
    function = (FuncType) func;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  for(i = 0; i < 3; ++i){
    dim[i] = n;  /* size of transformation in the i-th dimension */
  }
  ierr = DACreate3d(PETSC_COMM_SELF,DA_NONPERIODIC,DA_STENCIL_STAR, 
                    dim[0], dim[1], dim[2], 
                    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 
                    dof, stencil,
                    PETSC_NULL, PETSC_NULL, PETSC_NULL,
                    &da); 
  CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da, &x); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da, &y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y, "USFFT frequency space vector");CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da, &yy); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) yy, "FFTW frequency space vector");CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da, &z); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) z, "USFFT reconstructed vector");CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da, &zz); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) zz, "FFTW reconstructed vector");CHKERRQ(ierr);

  for(i = 0, N = 1; i < 3; i++) {
    N *= dim[i];
  }

  ierr = PetscPrintf(PETSC_COMM_SELF, "%3-D: USFFT on vector of total size %d \n",N);CHKERRQ(ierr);

  
  if (function == RANDOM) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
    ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  } 
  else if (function == CONSTANT) {
    ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  } 
  else if (function == TANH) {
    PetscScalar *a;
    ierr = VecGetArray(x, &a);CHKERRQ(ierr);
    PetscInt j,k = 0;
    for(i = 0; i < 3; ++i) {
      for(j = 0; j < dim[i]; ++j) {
        a[k] = tanh((j - dim[i]/2.0)*(10.0/dim[i]));
        ++k;
      }
    }
    ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
  }
  if (view) {
    ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  
  /* create USFFT object */
  ierr = MatCreateSeqUSFFT(da,da,&A);CHKERRQ(ierr);
  /* create FFTW object */
  ierr = MatCreateSeqFFTW(PETSC_COMM_SELF,1,&N,&AA);CHKERRQ(ierr);
  
  /* apply USFFT_FORWARD several times, so the fftw_plan can be reused on different vectors */
  ierr = MatMult(A,x,z);CHKERRQ(ierr);
  for (i=0; i<3; i++){
    ierr = MatMult(A,x,y);CHKERRQ(ierr); 
    if (view && i == 0) {ierr = VecView(y, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    /* apply USFFT_BACKWARD several times */  
    ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
  }
  
  /* compare y and yy: USFFT and FFTW results*/
  ierr = VecAXPY(y,-1.0,yy);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_1,&enorm);CHKERRQ(ierr);
  if (enorm > 1.e-11){
    ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |y - yy| %A\n",enorm);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "|y-yy| = %g\n",enorm);CHKERRQ(ierr);
  
  /* compare z and zz: USFFT and FFTW results*/
  ierr = VecAXPY(z,-1.0,zz);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
  if (enorm > 1.e-11){
    ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |z - zz| %A\n",enorm);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "|z-zz| = %g\n",enorm);CHKERRQ(ierr);
  
  /* compare x and z. USFFT computes an unnormalized DFT, thus z = N*x */
  s = 1.0/(PetscReal)N;
  ierr = VecScale(z,s);CHKERRQ(ierr);
  if (view) {ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
  if (enorm > 1.e-11){
    ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %A\n",enorm);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "|x-z| = %g\n",enorm);CHKERRQ(ierr);

  /* free spaces */
  ierr = DARestoreGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&y);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&yy);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&z);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&zz);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
