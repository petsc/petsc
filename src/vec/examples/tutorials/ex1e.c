/*$Id: ex1e.c,v 1.10 2001/03/23 23:21:37 balay Exp balay $*/

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

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
     petsc.h       - base PETSc routines   petscis.h     - index sets
     petscsys.h    - system routines       petscviewer.h - viewers
*/

#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec         x, y, w;               /* vectors */
  Vec         *z;                    /* array of vectors */
  double      norm, v, v1, v2;
  int         n = 20;
  PetscTruth  flg;
  PetscScalar one = 1.0, two = 2.0, three = 3.0, dots[3], dot;

_ PetscInitialize(&argc,&argv,(char*)0,help);___
_ PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);___

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel,
     shared, or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate() and VecSetFromOptions() the option -vec_type mpi or -vec_type shared causes the 
     particular type of vector to be formed.

  */
_ VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);___
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
_ VecSet(&one,x);___
_ VecSet(&two,y);___
_ VecSet(&one,z[0]);___
_ VecSet(&two,z[1]);___
_ VecSet(&three,z[2]);___

  /*
     Demonstrate various basic vector routines.
  */
_ VecDot(x,x,&dot);___
_ VecMDot(3,x,z,dots);___

  /* 
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */
#if defined(PETSC_USE_COMPLEX)
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %d\n", int (PetscRealPart(dot)));___
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %d %d %d\n",(int)PetscRealPart(dots[0]),
                             (int)PetscRealPart(dots[1]),(int)PetscRealPart(dots[2]));___
#else
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %d\n",(int) dot);___
_ PetscPrintf(PETSC_COMM_WORLD,"Vector length %d %d %d\n",(int)dots[0],
                             (int)dots[1],(int)dots[2]);___
#endif

_ PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n");___

_ VecScale(&two,x);___
_ VecNorm(x,NORM_2,&norm);___
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecScale %g\n",v);___

_ VecCopy(x,w);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",v);___

_ VecAXPY(&three,x,y);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-8.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",v);___

_ VecAYPX(&two,x,y);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",v);___

_ VecSwap(x,y);___
_ VecNorm(y,NORM_2,&norm);___
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);___
_ VecNorm(x,NORM_2,&norm);___
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);___

_ VecWAXPY(&two,x,y,w);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-38.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %g\n",v);___

_ VecPointwiseMult(y,x,w);___
_ VecNorm(w,NORM_2,&norm);___ 
  v = norm-36.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %g\n",v);___

_ VecPointwiseDivide(x,y,w);___
_ VecNorm(w,NORM_2,&norm);___
  v = norm-9.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %g\n",v);___

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;
_ VecSet(&one,x);___
_ VecMAXPY(3,dots,x,z);___
_ VecNorm(z[0],NORM_2,&norm);___
  v = norm-sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
_ VecNorm(z[1],NORM_2,&norm);___
  v1 = norm-2.0*sqrt((double) n); if (v1 > -1.e-10 && v1 < 1.e-10) v1 = 0.0; 
_ VecNorm(z[2],NORM_2,&norm);___
  v2 = norm-3.0*sqrt((double) n); if (v2 > -1.e-10 && v2 < 1.e-10) v2 = 0.0; 
_ PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %g %g %g \n",v,v1,v2);___

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
 
