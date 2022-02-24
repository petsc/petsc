static char help[] = "Tests PetscHeapCreate()\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscHeap      h;
  PetscInt       id,val,cnt,*values;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscHeapCreate(9,&h));
  CHKERRQ(PetscHeapAdd(h,0,100));
  CHKERRQ(PetscHeapAdd(h,1,19));
  CHKERRQ(PetscHeapAdd(h,2,36));
  CHKERRQ(PetscHeapAdd(h,3,17));
  CHKERRQ(PetscHeapAdd(h,4,3));
  CHKERRQ(PetscHeapAdd(h,5,25));
  CHKERRQ(PetscHeapAdd(h,6,1));
  CHKERRQ(PetscHeapAdd(h,8,2));
  CHKERRQ(PetscHeapAdd(h,9,7));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Initial heap:\n"));
  CHKERRQ(PetscHeapView(h,NULL));

  CHKERRQ(PetscHeapPop(h,&id,&val));
  CHKERRQ(PetscHeapStash(h,id,val+10));
  CHKERRQ(PetscHeapPop(h,&id,&val));
  CHKERRQ(PetscHeapStash(h,id,val+10));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Pop two items, increment, and place in stash:\n"));
  CHKERRQ(PetscHeapView(h,NULL));

  CHKERRQ(PetscHeapUnstash(h));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"After unpacking the stash:\n"));
  CHKERRQ(PetscHeapView(h,NULL));

  CHKERRQ(PetscMalloc1(9,&values));
  CHKERRQ(PetscHeapPop(h,&id,&val));
  cnt  = 0;
  while (id >= 0) {
    values[cnt++] = val;
    CHKERRQ(PetscHeapPop(h,&id,&val));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Sorted values:\n"));
  CHKERRQ(PetscIntView(cnt,values,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscFree(values));
  CHKERRQ(PetscHeapDestroy(&h));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
