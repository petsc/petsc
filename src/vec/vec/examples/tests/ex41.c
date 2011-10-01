
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
  Vec*           yvecs;
  PetscScalar    xdoty,xnorm;
  PetscInt       nthreads=2;
  PetscScalar    alpha=2.0;
  PetscScalar    mdot[2],alphas[2] ={1.0,2.0};

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

  ierr = VecDuplicateVecs(x,2,&yvecs);CHKERRQ(ierr);
  /* Test VecSet */
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  /* Test WAXPY w = 2.0*1.0 + 2.0 = 4.0 */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"w = 2.0*x + y\n");CHKERRQ(ierr);
  ierr = VecWAXPY(w,alpha,x,y);CHKERRQ(ierr);
  ierr = VecView(w,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* Test VecScale y = 2.0*0.5 = 1.0 */
  ierr = VecScale(y,0.5);CHKERRQ(ierr);

  /* Test VecDot */
  ierr = VecDot(x,y,&xdoty);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," x.y = %4.3f\n",xdoty);CHKERRQ(ierr);
  /* Test VecNorm */
  ierr = VecNorm(x,NORM_2,&xnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"||x|| = %4.3f\n",xnorm);CHKERRQ(ierr);
  /* Test AXPY y = 1.0 + 2.0*1.0 = 3.0 */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"y = y + 2.0*x\n");CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /* Test AYPX y = 2.0*3.0 + 1.0 = 7.0 */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"y = 2.0*y + x\n");CHKERRQ(ierr);
  ierr = VecAYPX(y,alpha,x);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /* Test VecCopy */
  ierr = VecCopy(x,yvecs[0]);CHKERRQ(ierr);
  ierr = VecCopy(y,yvecs[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"yvec[0]\n");CHKERRQ(ierr);
  ierr = VecView(yvecs[0],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"yvec[1]\n");CHKERRQ(ierr);
  ierr = VecView(yvecs[1],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /* Test VecMDot */
  ierr = VecMDot(x,2,yvecs,mdot);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"x.yvec[0] = %4.3f x.yvec2 = %4.3f\n",mdot[0],mdot[1]);CHKERRQ(ierr);
  /* Test VecMAXPY w = 1.0*yvec[0] + 2.0*yvec[2] */
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,2,(const PetscScalar*)alphas,yvecs);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"w = 1.0*.yvec[0] + 2.0*yvec[1]\n");CHKERRQ(ierr);
  ierr = VecView(w,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2,&yvecs);CHKERRQ(ierr);
#endif
  ierr = PetscFinalize();
  return 0;
}
 
