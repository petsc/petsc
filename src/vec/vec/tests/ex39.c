
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
  PetscInt          N = 10,i;
  Vec               X,Y,F,X1,Y1,X2,Y2,F1,F2;
  PetscScalar       value,zero=0.0;
  const PetscScalar *x,*y;
  PetscScalar       *f;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vectors X,Y and F and set values in it*/
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&X));
  CHKERRQ(VecSetSizes(X,N,N));
  CHKERRQ(VecSetFromOptions(X));
  CHKERRQ(VecDuplicate(X,&Y));
  CHKERRQ(VecDuplicate(X,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  for (i=0; i < N; i++) {
    value = i;
    CHKERRQ(VecSetValues(X,1,&i,&value,INSERT_VALUES));
    value = 100 + i;
    CHKERRQ(VecSetValues(Y,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecSet(F,zero));

  /* Create subvectors X1,X2,Y1,Y2,F1,F2 */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&X1));
  CHKERRQ(VecSetSizes(X1,N/2,N/2));
  CHKERRQ(VecSetFromOptions(X1));
  CHKERRQ(VecDuplicate(X1,&X2));
  CHKERRQ(VecDuplicate(X1,&Y1));
  CHKERRQ(VecDuplicate(X1,&Y2));
  CHKERRQ(VecDuplicate(X1,&F1));
  CHKERRQ(VecDuplicate(X1,&F2));

  /* Get array pointers for X,Y,F */
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Y,&y));
  CHKERRQ(VecGetArray(F,&f));
  /* Share X,Y,F array pointers with subvectors */
  CHKERRQ(VecPlaceArray(X1,x));
  CHKERRQ(VecPlaceArray(X2,x+N/2));
  CHKERRQ(VecPlaceArray(Y1,y));
  CHKERRQ(VecPlaceArray(Y2,y+N/2));
  CHKERRQ(VecPlaceArray(F1,f));
  CHKERRQ(VecPlaceArray(F2,f+N/2));

  /* Do subvector addition */
  CHKERRQ(VecWAXPY(F1,1.0,X1,Y1));
  CHKERRQ(VecWAXPY(F2,1.0,X2,Y2));

  /* Reset subvectors */
  CHKERRQ(VecResetArray(X1));
  CHKERRQ(VecResetArray(X2));
  CHKERRQ(VecResetArray(Y1));
  CHKERRQ(VecResetArray(Y2));
  CHKERRQ(VecResetArray(F1));
  CHKERRQ(VecResetArray(F2));

  /* Restore X,Y,and F */
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Y,&y));
  CHKERRQ(VecRestoreArray(F,&f));

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"F = X + Y\n"));
  CHKERRQ(VecView(F,0));
  /* Destroy vectors */
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(VecDestroy(&X1));
  CHKERRQ(VecDestroy(&Y1));
  CHKERRQ(VecDestroy(&F1));
  CHKERRQ(VecDestroy(&X2));
  CHKERRQ(VecDestroy(&Y2));
  CHKERRQ(VecDestroy(&F2));

  CHKERRQ(PetscFinalize());
  return 0;
}
