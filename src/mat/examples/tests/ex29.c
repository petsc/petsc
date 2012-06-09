static char help[] = "Tests PetscHeap\n\n";

#include <../src/mat/utils/petscheap.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscHeap h;
  PetscInt id,val;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscHeapCreate(9,&h);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,0,100);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,1,19);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,2,36);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,3,17);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,4,3);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,5,25);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,6,1);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,8,2);CHKERRQ(ierr);
  ierr = PetscHeapAdd(h,9,7);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Initial heap:\n");CHKERRQ(ierr);
  ierr = PetscHeapView(h,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscHeapPop(h,&id,&val);CHKERRQ(ierr);
  ierr = PetscHeapStash(h,id,val+10);CHKERRQ(ierr);
  ierr = PetscHeapPop(h,&id,&val);CHKERRQ(ierr);
  ierr = PetscHeapStash(h,id,val+10);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Pop two items, increment, and place in stash:\n");CHKERRQ(ierr);
  ierr = PetscHeapView(h,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscHeapUnstash(h);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"After unpacking the stash:\n");CHKERRQ(ierr);
  ierr = PetscHeapView(h,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscHeapDestroy(&h);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


