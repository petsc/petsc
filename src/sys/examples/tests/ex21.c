
static char help[] = "Tests PetscTreeProcess()";

#include <petscsys.h>

/*
                          2              6
                    1         4
                    5
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 7,cnt = 0,i,j;
  PetscBool      mask[]     = {PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
  PetscInt       parentId[] = {-1,         2,         0,         -1,         2,         1,         0};
  PetscInt       Nlevels,*Level,*Levelcnt,*Idbylevel,*Column;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = PetscProcessTree(n,mask,parentId,&Nlevels,&Level,&Levelcnt,&Idbylevel,&Column);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (!mask[i]) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," %D ",Level[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNumber of levels %D\n",Nlevels);CHKERRQ(ierr);
  for (i=0; i<Nlevels; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLevel %D ",i);CHKERRQ(ierr);
    for (j=0; j<Levelcnt[i]; j++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%D ",Idbylevel[cnt++]);CHKERRQ(ierr);
    }
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nColumn of each node");CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (!mask[i]) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," %D ",Column[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(Level);CHKERRQ(ierr);
  ierr = PetscFree(Levelcnt);CHKERRQ(ierr);
  ierr = PetscFree(Idbylevel);CHKERRQ(ierr);
  ierr = PetscFree(Column);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
