static char help[] = "Tests PetscHeapCreate()\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscHeap      h;
  PetscInt       id,val,cnt,*values;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscHeapCreate(9,&h));
  PetscCall(PetscHeapAdd(h,0,100));
  PetscCall(PetscHeapAdd(h,1,19));
  PetscCall(PetscHeapAdd(h,2,36));
  PetscCall(PetscHeapAdd(h,3,17));
  PetscCall(PetscHeapAdd(h,4,3));
  PetscCall(PetscHeapAdd(h,5,25));
  PetscCall(PetscHeapAdd(h,6,1));
  PetscCall(PetscHeapAdd(h,8,2));
  PetscCall(PetscHeapAdd(h,9,7));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Initial heap:\n"));
  PetscCall(PetscHeapView(h,NULL));

  PetscCall(PetscHeapPop(h,&id,&val));
  PetscCall(PetscHeapStash(h,id,val+10));
  PetscCall(PetscHeapPop(h,&id,&val));
  PetscCall(PetscHeapStash(h,id,val+10));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Pop two items, increment, and place in stash:\n"));
  PetscCall(PetscHeapView(h,NULL));

  PetscCall(PetscHeapUnstash(h));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"After unpacking the stash:\n"));
  PetscCall(PetscHeapView(h,NULL));

  PetscCall(PetscMalloc1(9,&values));
  PetscCall(PetscHeapPop(h,&id,&val));
  cnt  = 0;
  while (id >= 0) {
    values[cnt++] = val;
    PetscCall(PetscHeapPop(h,&id,&val));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Sorted values:\n"));
  PetscCall(PetscIntView(cnt,values,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscFree(values));
  PetscCall(PetscHeapDestroy(&h));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
