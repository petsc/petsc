static char help[] = "Test sequential r2c/c2r FFTW without PETSc interface \n\n";

/*
  Compiling the code:
      This code uses the real numbers version of PETSc
*/

#include <petscmat.h>
#include <fftw3.h>

int main(int argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char      *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  PetscMPIInt     size;
  int             n = 10,N,Ny,ndim=4,i,dim[4],DIM;
  Vec             x,y,z;
  PetscScalar     s;
  PetscRandom     rdm;
  PetscReal       enorm;
  PetscInt        func     = RANDOM;
  FuncType        function = RANDOM;
  PetscBool       view     = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscScalar     *x_array,*y_array,*z_array;
  fftw_plan       fplan,bplan;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex142");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-function", "Function type", "ex142", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  CHKERRQ(PetscOptionsBool("-vec_view draw", "View the functions", "ex142", view, &view, NULL));
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of real space vector in DIM-dimension */
  }
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  for (DIM = 1; DIM < 5; DIM++) {
    /* create vectors of length N=dim[0]*dim[1]* ...*dim[DIM-1] */
    /*----------------------------------------------------------*/
    N = Ny = 1;
    for (i = 0; i < DIM-1; i++) {
      N *= dim[i];
    }
    Ny = N; Ny *= 2*(dim[DIM-1]/2 + 1); /* add padding elements to output vector y */
    N *= dim[DIM-1];

    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));
    CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector"));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,Ny,&y));
    CHKERRQ(PetscObjectSetName((PetscObject) y, "Frequency space vector"));

    CHKERRQ(VecDuplicate(x,&z));
    CHKERRQ(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

    /* Set fftw plan                    */
    /*----------------------------------*/
    CHKERRQ(VecGetArray(x,&x_array));
    CHKERRQ(VecGetArray(y,&y_array));
    CHKERRQ(VecGetArray(z,&z_array));

    unsigned int flags = FFTW_ESTIMATE; /*or FFTW_MEASURE */
    /* The data in the in/out arrays is overwritten during FFTW_MEASURE planning, so such planning
     should be done before the input is initialized by the user. */
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"DIM: %d, N %d, Ny %d\n",DIM,N,Ny));

    switch (DIM) {
    case 1:
      fplan = fftw_plan_dft_r2c_1d(dim[0], (double*)x_array, (fftw_complex*)y_array, flags);
      bplan = fftw_plan_dft_c2r_1d(dim[0], (fftw_complex*)y_array, (double*)z_array, flags);
      break;
    case 2:
      fplan = fftw_plan_dft_r2c_2d(dim[0],dim[1],(double*)x_array, (fftw_complex*)y_array,flags);
      bplan = fftw_plan_dft_c2r_2d(dim[0],dim[1],(fftw_complex*)y_array,(double*)z_array,flags);
      break;
    case 3:
      fplan = fftw_plan_dft_r2c_3d(dim[0],dim[1],dim[2],(double*)x_array, (fftw_complex*)y_array,flags);
      bplan = fftw_plan_dft_c2r_3d(dim[0],dim[1],dim[2],(fftw_complex*)y_array,(double*)z_array,flags);
      break;
    default:
      fplan = fftw_plan_dft_r2c(DIM,(int*)dim,(double*)x_array, (fftw_complex*)y_array,flags);
      bplan = fftw_plan_dft_c2r(DIM,(int*)dim,(fftw_complex*)y_array,(double*)z_array,flags);
      break;
    }

    CHKERRQ(VecRestoreArray(x,&x_array));
    CHKERRQ(VecRestoreArray(y,&y_array));
    CHKERRQ(VecRestoreArray(z,&z_array));

    /* Initialize Real space vector x:
       The data in the in/out arrays is overwritten during FFTW_MEASURE planning, so planning
       should be done before the input is initialized by the user.
    --------------------------------------------------------*/
    if (function == RANDOM) {
      CHKERRQ(VecSetRandom(x, rdm));
    } else if (function == CONSTANT) {
      CHKERRQ(VecSet(x, 1.0));
    } else if (function == TANH) {
      CHKERRQ(VecGetArray(x, &x_array));
      for (i = 0; i < N; ++i) {
        x_array[i] = tanh((i - N/2.0)*(10.0/N));
      }
      CHKERRQ(VecRestoreArray(x, &x_array));
    }
    if (view) {
      CHKERRQ(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    }

    /* FFT - also test repeated transformation   */
    /*-------------------------------------------*/
    CHKERRQ(VecGetArray(x,&x_array));
    CHKERRQ(VecGetArray(y,&y_array));
    CHKERRQ(VecGetArray(z,&z_array));
    for (i=0; i<4; i++) {
      /* FFTW_FORWARD */
      fftw_execute(fplan);

      /* FFTW_BACKWARD: destroys its input array 'y_array' even for out-of-place transforms! */
      fftw_execute(bplan);
    }
    CHKERRQ(VecRestoreArray(x,&x_array));
    CHKERRQ(VecRestoreArray(y,&y_array));
    CHKERRQ(VecRestoreArray(z,&z_array));

    /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    /*------------------------------------------------------------------*/
    s    = 1.0/(PetscReal)N;
    CHKERRQ(VecScale(z,s));
    if (view) CHKERRQ(VecView(x, PETSC_VIEWER_DRAW_WORLD));
    if (view) CHKERRQ(VecView(z, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(VecAXPY(z,-1.0,x));
    CHKERRQ(VecNorm(z,NORM_1,&enorm));
    if (enorm > 1.e-11) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm));
    }

    /* free spaces */
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&z));
  }
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: fftw !complex

   test:
     output_file: output/ex142.out

TEST*/
