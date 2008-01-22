static char help[] = "Test sequential FFTW interface \n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this

*/

#include "petscmat.h"
#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char    *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  Mat            A;    
  PetscMPIInt    size;
  PetscInt       n = 10,N,ndim=4,dim[4],DIM,i;
  Vec            x,y,z;
  PetscScalar    s;  
  PetscRandom    rdm;
  PetscReal      enorm;
  PetscInt       func;
  FuncType       function = RANDOM;
  PetscTruth     view = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_ERR_SUP, "This example requires complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "FFTW Options", "ex112");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-function", "Function type", "ex112", funcNames, NUM_FUNCS, funcNames[function], &func, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-vec_view_draw", "View the functions", "ex112", view, &view, PETSC_NULL);CHKERRQ(ierr);
    function = (FuncType) func;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  for(DIM = 0; DIM < ndim; DIM++){
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  for(DIM = 1; DIM < 5; DIM++){
    /* create vectors of length N=n^DIM */
    for(i = 0, N = 1; i < DIM; i++) N *= dim[i];
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);
    if (function == RANDOM) {
      ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    } else if (function == CONSTANT) {
      ierr = VecSet(x, 1.0);CHKERRQ(ierr);
    } else if (function == TANH) {
      PetscScalar *a;

      ierr = VecGetArray(x, &a);CHKERRQ(ierr);
      for(i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
    }
    if (view) {ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

    /* create FFTW object */
    ierr = MatCreateSeqFFTW(PETSC_COMM_SELF,DIM,dim,&A);CHKERRQ(ierr);

    /* apply FFTW_FORWARD several times, so the fftw_plan can be reused on different vectors */
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    for (i=0; i<3; i++){
      ierr = MatMult(A,x,y);CHKERRQ(ierr); 
      if (view && i == 0) {ierr = VecView(y, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
      /* apply FFTW_BACKWARD several times */  
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
    }
 
    /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    s = 1.0/(PetscReal)N;
    ierr = VecScale(z,s);CHKERRQ(ierr);
    if (view) {ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-11){
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %A\n",enorm);CHKERRQ(ierr);
    }

    /* free spaces */
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = VecDestroy(y);CHKERRQ(ierr);
    ierr = VecDestroy(z);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
