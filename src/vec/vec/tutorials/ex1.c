
static char help[] = "Basic vector routines.\n\n";

/*T
   Concepts: vectors^basic routines;
   Processors: n
T*/



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
  PetscErrorCode ierr;
  PetscScalar    one = 1.0,two = 2.0,three = 3.0,dots[3],dot;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

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
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);

  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
  ierr = VecDuplicateVecs(x,3,&z);CHKERRQ(ierr);
  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  ierr = VecSet(z[0],one);CHKERRQ(ierr);
  ierr = VecSet(z[1],two);CHKERRQ(ierr);
  ierr = VecSet(z[2],three);CHKERRQ(ierr);
  /*
     Demonstrate various basic vector routines.
  */
  ierr = VecDot(x,y,&dot);CHKERRQ(ierr);
  ierr = VecMDot(x,3,z,dots);CHKERRQ(ierr);

  /*
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector length %D\n",n);CHKERRQ(ierr);
  ierr = VecMax(x,&maxind,&maxval);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecMax %g, VecInd %D\n",(double)maxval,maxind);CHKERRQ(ierr);

  ierr = VecMin(x,&maxind,&maxval);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecMin %g, VecInd %D\n",(double)maxval,maxind);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n");CHKERRQ(ierr);


  ierr = VecScale(x,two);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecScale %g\n",(double)v);CHKERRQ(ierr);


  ierr = VecCopy(x,w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecAXPY(y,three,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-8.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecAYPX(y,two,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-18.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecAYPX %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecSwap(x,y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-2.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",(double)v);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-18.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecWAXPY(w,two,x,y);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-38.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecPointwiseMult(w,y,x);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-36.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %g\n",(double)v);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(w,x,y);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-9.0*PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %g\n",(double)v);CHKERRQ(ierr);

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;

  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecMAXPY(x,3,dots,z);CHKERRQ(ierr);
  ierr = VecNorm(z[0],NORM_2,&norm);CHKERRQ(ierr);
  v    = norm-PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  ierr = VecNorm(z[1],NORM_2,&norm);CHKERRQ(ierr);
  v1   = norm-2.0*PetscSqrtReal((PetscReal)n); if (v1 > -PETSC_SMALL && v1 < PETSC_SMALL) v1 = 0.0;
  ierr = VecNorm(z[2],NORM_2,&norm);CHKERRQ(ierr);
  v2   = norm-3.0*PetscSqrtReal((PetscReal)n); if (v2 > -PETSC_SMALL && v2 < PETSC_SMALL) v2 = 0.0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %g %g %g \n",(double)v,(double)v1,(double)v2);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroyVecs(3,&z);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex1_1.out

   test:
      suffix: 2_cuda
      nsize: 2
      args: -vec_type cuda
      output_file: output/ex1_1.out
      requires: cuda

   test:
      suffix: cuda
      args: -vec_type cuda
      output_file: output/ex1_1.out
      requires: cuda

TEST*/
