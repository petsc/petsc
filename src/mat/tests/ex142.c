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

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex142");CHKERRQ(ierr);
  ierr     = PetscOptionsEList("-function", "Function type", "ex142", funcNames, NUM_FUNCS, funcNames[function], &func, NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsBool("-vec_view draw", "View the functions", "ex142", view, &view, NULL);CHKERRQ(ierr);
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of real space vector in DIM-dimension */
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  for (DIM = 1; DIM < 5; DIM++) {
    /* create vectors of length N=dim[0]*dim[1]* ...*dim[DIM-1] */
    /*----------------------------------------------------------*/
    N = Ny = 1;
    for (i = 0; i < DIM-1; i++) {
      N *= dim[i];
    }
    Ny = N; Ny *= 2*(dim[DIM-1]/2 + 1); /* add padding elements to output vector y */
    N *= dim[DIM-1];

    ierr = PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,Ny,&y);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);

    ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

    /* Set fftw plan                    */
    /*----------------------------------*/
    ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
    ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
    ierr = VecGetArray(z,&z_array);CHKERRQ(ierr);

    unsigned int flags = FFTW_ESTIMATE; /*or FFTW_MEASURE */
    /* The data in the in/out arrays is overwritten during FFTW_MEASURE planning, so such planning
     should be done before the input is initialized by the user. */
    ierr = PetscPrintf(PETSC_COMM_SELF,"DIM: %d, N %d, Ny %d\n",DIM,N,Ny);CHKERRQ(ierr);

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

    ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(z,&z_array);CHKERRQ(ierr);

    /* Initialize Real space vector x:
       The data in the in/out arrays is overwritten during FFTW_MEASURE planning, so planning
       should be done before the input is initialized by the user.
    --------------------------------------------------------*/
    if (function == RANDOM) {
      ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    } else if (function == CONSTANT) {
      ierr = VecSet(x, 1.0);CHKERRQ(ierr);
    } else if (function == TANH) {
      ierr = VecGetArray(x, &x_array);CHKERRQ(ierr);
      for (i = 0; i < N; ++i) {
        x_array[i] = tanh((i - N/2.0)*(10.0/N));
      }
      ierr = VecRestoreArray(x, &x_array);CHKERRQ(ierr);
    }
    if (view) {
      ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* FFT - also test repeated transformation   */
    /*-------------------------------------------*/
    ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
    ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
    ierr = VecGetArray(z,&z_array);CHKERRQ(ierr);
    for (i=0; i<4; i++) {
      /* FFTW_FORWARD */
      fftw_execute(fplan);

      /* FFTW_BACKWARD: destroys its input array 'y_array' even for out-of-place transforms! */
      fftw_execute(bplan);
    }
    ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(z,&z_array);CHKERRQ(ierr);

    /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    /*------------------------------------------------------------------*/
    s    = 1.0/(PetscReal)N;
    ierr = VecScale(z,s);CHKERRQ(ierr);
    if (view) {ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    if (view) {ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-11) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm);CHKERRQ(ierr);
    }

    /* free spaces */
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: fftw !complex

   test:
     output_file: output/ex142.out

TEST*/
