
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates various vector routines.\n\n";

/*T
   Concepts: vectors^basic routines;
   Processors: n
T*/

/* 

   This uses the PETSc _ error checking routines. Put _ before the PETSc function call
  and __ after the call (or ___ in a subroutine, not the main program). This is equivalent
  to using the ierr = ... CHKERRQ(ierr); macros


  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#define PETSC_UNDERSCORE_CHKERR

#include "petscvec.h"

#if defined(PETSC_USE_SCALAR_SINGLE)
#define PETSC_EPS 1.e-5
#else
#define PETSC_EPS 1.e-10
#endif

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec         x, y, w;               /* vectors */
  Vec         *z;                    /* array of vectors */
  PetscReal   norm, v, v1, v2;
  PetscInt    n = 20;
  PetscTruth  flg;
  PetscScalar one = 1.0, two = 2.0, three = 3.0, dots[3], dot;

_ PetscInitialize(&argc,&argv,(char*)0,help);___
_ PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);___

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
_ VecCreate(PETSC_COMM_WORLD,&x);___
_ VecSetSizes(x,PETSC_DECIDE,n);___
_ VecSetFromOptions(x);___

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
_ VecDuplicate(x,&y);___
_ VecDuplicate(x,&w);___

  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
_ VecDuplicateVecs(x,3,&z);___ 

  /*
     Set the vectors to entries to a constant value.
  */
_ VecSet(x,one);___
_ VecSet(y,two);___
_ VecSet(z[0],one);___
_ VecSet(z[1],two);___
_ VecSet(z[2],three);___

  /*
     Demonstrate various basic vector routines.
  */
_ VecDot(x,x,&dot);___
_ VecMDot(x,3,z,dots);___

  /* 
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */
#if defined(PETSC_USE_COMPLEX)
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %D\n", (int) (PetscRealPart(dot)));___
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %D %D %D\n",(PetscInt)PetscRealPart(dots[0]),
                             (PetscInt)PetscRealPart(dots[1]),(PetscInt)PetscRealPart(dots[2]));___
#else
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %D\n",(PetscInt) dot);___
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %D %D %D\n",(PetscInt)dots[0],
                             (PetscInt)dots[1],(PetscInt)dots[2]);___
#endif

_ PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n");___

_ VecScale(x,two);___
_ VecNorm(x,NORM_2,&norm);___
  v = norm-2.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecScale %G\n",v);___

_ VecCopy(x,w);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-2.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %G\n",v);___

_ VecAXPY(y,three,x);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-8.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %G\n",v);___

_ VecAYPX(y,two,x);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-18.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecAYPX %G\n",v);___

_ VecSwap(x,y);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-2.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %G\n",v);___
_ VecNorm(x,NORM_2,&norm);___
  v = norm-18.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %G\n",v);___

_ VecWAXPY(w,two,x,y);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-38.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %G\n",v);___

_ VecPointwiseMult(w,y,x);___
_ VecNorm(w,NORM_2,&norm);___ 
  v = norm-36.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %G\n",v);___

_ VecPointwiseDivide(w,x,y);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-9.0*sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %G\n",v);___

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;
_ VecSet(x,one);___
_ VecMAXPY(x,3,dots,z);___
_ VecNorm(z[0],NORM_2,&norm);___
  v = norm-sqrt((PetscReal) n); if (v > -PETSC_EPS && v < PETSC_EPS) v = 0.0; 
_ VecNorm(z[1],NORM_2,&norm);___
  v1 = norm-2.0*sqrt((PetscReal) n); if (v1 > -PETSC_EPS && v1 < PETSC_EPS) v1 = 0.0; 
_ VecNorm(z[2],NORM_2,&norm);___
  v2 = norm-3.0*sqrt((PetscReal) n); if (v2 > -PETSC_EPS && v2 < PETSC_EPS) v2 = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %G %G %G \n",v,v1,v2);___

  /* 
     Test whether vector has been corrupted (just to demonstrate this
     routine) not needed in most application codes.
  */
_ VecValid(x,&flg);___
  if (!flg) SETERRQ(1,"Corrupted vector.");

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
_ VecDestroy(x);___
_ VecDestroy(y);___
_ VecDestroy(w);___
_ VecDestroyVecs(z,3);___
_ PetscFinalize();___
  return 0;
}
 
