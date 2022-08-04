
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vectors X,Y and F and set values in it*/
  PetscCall(VecCreate(PETSC_COMM_SELF,&X));
  PetscCall(VecSetSizes(X,N,N));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecDuplicate(X,&Y));
  PetscCall(VecDuplicate(X,&F));
  PetscCall(PetscObjectSetName((PetscObject)F,"F"));
  for (i=0; i < N; i++) {
    value = i;
    PetscCall(VecSetValues(X,1,&i,&value,INSERT_VALUES));
    value = 100 + i;
    PetscCall(VecSetValues(Y,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecSet(F,zero));

  /* Create subvectors X1,X2,Y1,Y2,F1,F2 */
  PetscCall(VecCreate(PETSC_COMM_SELF,&X1));
  PetscCall(VecSetSizes(X1,N/2,N/2));
  PetscCall(VecSetFromOptions(X1));
  PetscCall(VecDuplicate(X1,&X2));
  PetscCall(VecDuplicate(X1,&Y1));
  PetscCall(VecDuplicate(X1,&Y2));
  PetscCall(VecDuplicate(X1,&F1));
  PetscCall(VecDuplicate(X1,&F2));

  /* Get array pointers for X,Y,F */
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Y,&y));
  PetscCall(VecGetArray(F,&f));
  /* Share X,Y,F array pointers with subvectors */
  PetscCall(VecPlaceArray(X1,x));
  PetscCall(VecPlaceArray(X2,x+N/2));
  PetscCall(VecPlaceArray(Y1,y));
  PetscCall(VecPlaceArray(Y2,y+N/2));
  PetscCall(VecPlaceArray(F1,f));
  PetscCall(VecPlaceArray(F2,f+N/2));

  /* Do subvector addition */
  PetscCall(VecWAXPY(F1,1.0,X1,Y1));
  PetscCall(VecWAXPY(F2,1.0,X2,Y2));

  /* Reset subvectors */
  PetscCall(VecResetArray(X1));
  PetscCall(VecResetArray(X2));
  PetscCall(VecResetArray(Y1));
  PetscCall(VecResetArray(Y2));
  PetscCall(VecResetArray(F1));
  PetscCall(VecResetArray(F2));

  /* Restore X,Y,and F */
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArrayRead(Y,&y));
  PetscCall(VecRestoreArray(F,&f));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"F = X + Y\n"));
  PetscCall(VecView(F,0));
  /* Destroy vectors */
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&X1));
  PetscCall(VecDestroy(&Y1));
  PetscCall(VecDestroy(&F1));
  PetscCall(VecDestroy(&X2));
  PetscCall(VecDestroy(&Y2));
  PetscCall(VecDestroy(&F2));

  PetscCall(PetscFinalize());
  return 0;
}
