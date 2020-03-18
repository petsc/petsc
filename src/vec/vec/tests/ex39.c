
static char help[] = "This example is intended for showing how subvectors can\n\
                      share the pointer with the main vector using VecGetArray()\n\
                      and VecPlaceArray() routines so that vector operations done\n\
                      on the subvectors automatically modify the values in the main vector.\n\n";

#include <petscvec.h>

/* This example shares the array pointers of vectors X,Y,and F with subvectors
   X1,X2,Y1,Y2,F1,F2 and does vector addition on the subvectors F1 = X1 + Y1, F2 = X2 + Y2 so
   that F gets updated as a result of sharing the pointers.
 */

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  PetscInt          N = 10,i;
  Vec               X,Y,F,X1,Y1,X2,Y2,F1,F2;
  PetscScalar       value,zero=0.0;
  const PetscScalar *x,*y;
  PetscScalar       *f;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create vectors X,Y and F and set values in it*/
  ierr = VecCreate(PETSC_COMM_SELF,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,N,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  for (i=0; i < N; i++) {
    value = i;
    ierr  = VecSetValues(X,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
    value = 100 + i;
    ierr  = VecSetValues(Y,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecSet(F,zero);CHKERRQ(ierr);

  /* Create subvectors X1,X2,Y1,Y2,F1,F2 */
  ierr = VecCreate(PETSC_COMM_SELF,&X1);CHKERRQ(ierr);
  ierr = VecSetSizes(X1,N/2,N/2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X1);CHKERRQ(ierr);
  ierr = VecDuplicate(X1,&X2);CHKERRQ(ierr);
  ierr = VecDuplicate(X1,&Y1);CHKERRQ(ierr);
  ierr = VecDuplicate(X1,&Y2);CHKERRQ(ierr);
  ierr = VecDuplicate(X1,&F1);CHKERRQ(ierr);
  ierr = VecDuplicate(X1,&F2);CHKERRQ(ierr);

  /* Get array pointers for X,Y,F */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  /* Share X,Y,F array pointers with subvectors */
  ierr = VecPlaceArray(X1,x);CHKERRQ(ierr);
  ierr = VecPlaceArray(X2,x+N/2);CHKERRQ(ierr);
  ierr = VecPlaceArray(Y1,y);CHKERRQ(ierr);
  ierr = VecPlaceArray(Y2,y+N/2);CHKERRQ(ierr);
  ierr = VecPlaceArray(F1,f);CHKERRQ(ierr);
  ierr = VecPlaceArray(F2,f+N/2);CHKERRQ(ierr);

  /* Do subvector addition */
  ierr = VecWAXPY(F1,1.0,X1,Y1);CHKERRQ(ierr);
  ierr = VecWAXPY(F2,1.0,X2,Y2);CHKERRQ(ierr);

  /* Reset subvectors */
  ierr = VecResetArray(X1);CHKERRQ(ierr);
  ierr = VecResetArray(X2);CHKERRQ(ierr);
  ierr = VecResetArray(Y1);CHKERRQ(ierr);
  ierr = VecResetArray(Y2);CHKERRQ(ierr);
  ierr = VecResetArray(F1);CHKERRQ(ierr);
  ierr = VecResetArray(F2);CHKERRQ(ierr);

  /* Restore X,Y,and F */
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"F = X + Y\n");CHKERRQ(ierr);
  ierr = VecView(F,0);CHKERRQ(ierr);
  /* Destroy vectors */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&Y1);CHKERRQ(ierr);
  ierr = VecDestroy(&F1);CHKERRQ(ierr);
  ierr = VecDestroy(&X2);CHKERRQ(ierr);
  ierr = VecDestroy(&Y2);CHKERRQ(ierr);
  ierr = VecDestroy(&F2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

