static char help[] = "Test to demonstrate interface for thread reductions and passing scalar values.\n\n";

/*T
   Concepts: PetscThreadComm^basic example: Threaded reductions and passing scalar values
T*/

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>

PetscInt    *trstarts;

PetscErrorCode set_kernel(PetscInt myrank,PetscScalar *a,PetscScalar *alphap)
{
  PetscScalar alpha=*alphap;
  PetscInt    i;

  for (i=trstarts[myrank];i < trstarts[myrank+1];i++) a[i] = alpha+(PetscScalar)i;

  return 0;
}

PetscErrorCode sum_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  PetscScalar my_sum=0.0;
  PetscInt    i;

  for (i=trstarts[myrank];i < trstarts[myrank+1];i++) my_sum += a[i];

  PetscThreadReductionKernelPost(myrank,red,&my_sum);

  return 0;
}

PetscErrorCode max_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  PetscScalar my_max=a[trstarts[myrank]];
  PetscInt    i;

  for (i=trstarts[myrank]+1;i < trstarts[myrank+1];i++) {
    if (PetscRealPart(a[i]) > PetscRealPart(my_max)) my_max = a[i];
  }

  PetscThreadReductionKernelPost(myrank,red,&my_max);

  return 0;
}

PetscErrorCode min_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  PetscScalar my_min=a[trstarts[myrank]];
  PetscInt    i;

  for (i=trstarts[myrank]+1;i < trstarts[myrank+1];i++) {
    if (PetscRealPart(a[i]) < PetscRealPart(my_min)) my_min = a[i];
  }

  PetscThreadReductionKernelPost(myrank,red,&my_min);

  return 0;
}

PetscErrorCode maxloc_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  PetscScalar my_maxloc[2];
  PetscInt    i;

  my_maxloc[0]=a[trstarts[myrank]];
  my_maxloc[1] = trstarts[myrank];
  for (i=trstarts[myrank]+1;i < trstarts[myrank+1];i++) {
    if (PetscRealPart(a[i]) > PetscRealPart(my_maxloc[0])) { my_maxloc[0] = a[i]; my_maxloc[1] = i;}
  }

  PetscThreadReductionKernelPost(myrank,red,my_maxloc);

  return 0;
}

PetscErrorCode minloc_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  PetscScalar my_minloc[2];
  PetscInt    i;

  my_minloc[0]=a[trstarts[myrank]];
  my_minloc[1] = trstarts[myrank];
  for (i=trstarts[myrank]+1;i < trstarts[myrank+1];i++) {
    if (PetscRealPart(a[i]) < PetscRealPart(my_minloc[0])) { my_minloc[0] = a[i]; my_minloc[1] = i;}
  }

  PetscThreadReductionKernelPost(myrank,red,my_minloc);

  return 0;
}

PetscErrorCode mult_reds_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommReduction red)
{
  minloc_kernel(myrank,a,red);
  maxloc_kernel(myrank,a,red);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               N=8;
  PetscScalar           *a,sum=0.0,alpha=0.0,*scalar,max=0.0,min=N,maxloc[2],minloc[2];
  PetscThreadCommReduction  red;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscScalar),&a);CHKERRQ(ierr);

  /* Set thread ownership ranges for the array */
  ierr = PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,N,&trstarts);CHKERRQ(ierr);

  /* Set a[i] = 1.0 .. i = 1,N */
  /* Get location to store the scalar value alpha from threadcomm */
  ierr = PetscThreadCommGetScalars(PETSC_COMM_WORLD,&scalar,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  *scalar = alpha;
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)set_kernel,2,a,scalar);CHKERRQ(ierr);

  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_SUM,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)sum_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,&sum);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Sum(x) = %f\n",sum);CHKERRQ(ierr);

  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MAX,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)max_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,&max);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Max = %f\n",PetscRealPart(max));CHKERRQ(ierr);

  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MIN,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)min_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,&min);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Min = %f\n",PetscRealPart(min));CHKERRQ(ierr);

  /*  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MAXLOC,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)maxloc_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,maxloc);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Max = %f, location = %d\n",PetscRealPart(maxloc[0]),(PetscInt)maxloc[1]);CHKERRQ(ierr);

  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MINLOC,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)minloc_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,minloc);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Min = %f, location = %d\n",PetscRealPart(minloc[0]),(PetscInt)minloc[1]);CHKERRQ(ierr);
*/
  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MINLOC,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_MAXLOC,PETSC_SCALAR,1,&red);CHKERRQ(ierr);

  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)mult_reds_kernel,2,a,red);CHKERRQ(ierr);

  ierr = PetscThreadReductionEnd(red,minloc);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,maxloc);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Min = %f, location = %d\n",PetscRealPart(minloc[0]),(PetscInt)minloc[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Max = %f, location = %d\n",PetscRealPart(maxloc[0]),(PetscInt)maxloc[1]);CHKERRQ(ierr);


  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(trstarts);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
