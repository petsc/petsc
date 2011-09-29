
static char help[] = "Tests basic vector routines for vec_type pthread.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

#if defined(PETSC_HAVE_PTHREADCLASSES)
  PetscInt       n = 10;
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y,w;
  PetscScalar    xdoty;
  PetscInt       nthreads=2;
  PetscScalar    alpha=2.0;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nthreads",&nthreads,PETSC_NULL);CHKERRQ(ierr);
  /* Can create pthread vector directly*/
  ierr = VecCreateSeqPThread(PETSC_COMM_SELF,n,nthreads,&x);CHKERRQ(ierr);
  /* Or by setting number of threads by calling VecSeqPThreadSetNThreads() */
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,n);CHKERRQ(ierr);
  ierr = VecSetType(y,"seqpthread");CHKERRQ(ierr);
  ierr = VecSeqPThreadSetNThreads(y,nthreads);CHKERRQ(ierr);
  /* or by setting the number of threads at run time */
  ierr = VecCreate(PETSC_COMM_WORLD,&w);CHKERRQ(ierr);
  ierr = VecSetSizes(w,n,n);CHKERRQ(ierr);
  ierr = VecSetType(w,"seqpthread");CHKERRQ(ierr);
  ierr = VecSetFromOptions(w);CHKERRQ(ierr); /* use runtime option -vec_threads option to set number of threads associated with the vector */

  /* Test VecSet */
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  /* Test WAXPY */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"w = alpha*x + y\n");CHKERRQ(ierr);
  ierr = VecWAXPY(w,alpha,x,y);CHKERRQ(ierr);
  ierr = VecView(w,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /* Test VecScale */
  ierr = VecScale(y,0.5);CHKERRQ(ierr);

  /* Test VecDot */
  ierr = VecDot(x,y,&xdoty);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," x.y = %4.3f\n",xdoty);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
#endif
  ierr = PetscFinalize();
  return 0;
}
 
