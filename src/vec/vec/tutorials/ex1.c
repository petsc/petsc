
static char help[] = "Basic vector routines.\n\n";

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/

#include <petscvec.h>

int main(int argc,char **argv)
{
  Vec            x,y,w;               /* vectors */
  Vec            *z;                    /* array of vectors */
  PetscReal      norm,v,v1,v2,maxval;
  PetscInt       n = 20,maxind;
  PetscScalar    one = 1.0,two = 2.0,three = 3.0,dots[3],dot;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(), the vector format
     (currently parallel, shared, or sequential) is determined at runtime.  Also, the
     parallel partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate(), VecSetSizes() and VecSetFromOptions() the option -vec_type mpi or
     -vec_type shared causes the particular type of vector to be formed.

  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecDuplicate(x,&w));

  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
  PetscCall(VecDuplicateVecs(x,3,&z));
  /*
     Set the vectors to entries to a constant value.
  */
  PetscCall(VecSet(x,one));
  PetscCall(VecSet(y,two));
  PetscCall(VecSet(z[0],one));
  PetscCall(VecSet(z[1],two));
  PetscCall(VecSet(z[2],three));
  /*
     Demonstrate various basic vector routines.
  */
  PetscCall(VecDot(x,y,&dot));
  PetscCall(VecMDot(x,3,z,dots));

  /*
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vector length %" PetscInt_FMT "\n",n));
  PetscCall(VecMax(x,&maxind,&maxval));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecMax %g, VecInd %" PetscInt_FMT "\n",(double)maxval,maxind));

  PetscCall(VecMin(x,&maxind,&maxval));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecMin %g, VecInd %" PetscInt_FMT "\n",(double)maxval,maxind));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n"));

  PetscCall(VecScale(x,two));
  PetscCall(VecNorm(x,NORM_2,&norm));
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecScale %g\n",(double)v));

  PetscCall(VecCopy(x,w));
  PetscCall(VecNorm(w,NORM_2,&norm));
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",(double)v));

  PetscCall(VecAXPY(y,three,x));
  PetscCall(VecNorm(y,NORM_2,&norm));
  v    = norm-8.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",(double)v));

  PetscCall(VecAYPX(y,two,x));
  PetscCall(VecNorm(y,NORM_2,&norm));
  v    = norm-18.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecAYPX %g\n",(double)v));

  PetscCall(VecSwap(x,y));
  PetscCall(VecNorm(y,NORM_2,&norm));
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",(double)v));
  PetscCall(VecNorm(x,NORM_2,&norm));
  v = norm-18.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",(double)v));

  PetscCall(VecWAXPY(w,two,x,y));
  PetscCall(VecNorm(w,NORM_2,&norm));
  v    = norm-38.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %g\n",(double)v));

  PetscCall(VecPointwiseMult(w,y,x));
  PetscCall(VecNorm(w,NORM_2,&norm));
  v    = norm-36.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %g\n",(double)v));

  PetscCall(VecPointwiseDivide(w,x,y));
  PetscCall(VecNorm(w,NORM_2,&norm));
  v    = norm-9.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %g\n",(double)v));

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;

  PetscCall(VecSet(x,one));
  PetscCall(VecMAXPY(x,3,dots,z));
  PetscCall(VecNorm(z[0],NORM_2,&norm));
  v    = norm-PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(VecNorm(z[1],NORM_2,&norm));
  v1   = norm-2.0*PetscSqrtReal((PetscReal)n); if (v1 > -PETSC_SMALL && v1 < PETSC_SMALL) v1 = 0.0;
  PetscCall(VecNorm(z[2],NORM_2,&norm));
  v2   = norm-3.0*PetscSqrtReal((PetscReal)n); if (v2 > -PETSC_SMALL && v2 < PETSC_SMALL) v2 = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %g %g %g \n",(double)v,(double)v1,(double)v2));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroyVecs(3,&z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/ex1_1.out
    # This is a test where the exact numbers are critical
    diff_args: -j

    test:

    test:
        suffix: cuda
        args: -vec_type cuda
        requires: cuda

    test:
        suffix: kokkos
        args: -vec_type kokkos
        requires: kokkos_kernels

    test:
        suffix: hip
        args: -vec_type hip
        requires: hip

    test:
        suffix: 2
        nsize: 2

    test:
        suffix: 2_cuda
        nsize: 2
        args: -vec_type cuda
        requires: cuda

    test:
        suffix: 2_kokkos
        nsize: 2
        args: -vec_type kokkos
        requires: kokkos_kernels

    test:
        suffix: 2_hip
        nsize: 2
        args: -vec_type hip
        requires: hip

TEST*/
