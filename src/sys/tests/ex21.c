
static char help[] = "Tests PetscTreeProcess()";

#include <petscsys.h>

/*
                          2              6
                    1         4
                    5
*/
int main(int argc,char **argv)
{
  PetscInt       n          = 7,cnt = 0,i,j;
  PetscBool      mask[]     = {PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
  PetscInt       parentId[] = {-1,         2,         0,         -1,         2,         1,         0};
  PetscInt       Nlevels,*Level,*Levelcnt,*Idbylevel,*Column;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscProcessTree(n,mask,parentId,&Nlevels,&Level,&Levelcnt,&Idbylevel,&Column));
  for (i=0; i<n; i++) {
    if (!mask[i]) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " ",Level[i]));
    }
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nNumber of levels %" PetscInt_FMT "\n",Nlevels));
  for (i=0; i<Nlevels; i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nLevel %" PetscInt_FMT " ",i));
    for (j=0; j<Levelcnt[i]; j++) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " ",Idbylevel[cnt++]));
    }
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nColumn of each node"));
  for (i=0; i<n; i++) {
    if (!mask[i]) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " ",Column[i]));
    }
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  CHKERRQ(PetscFree(Level));
  CHKERRQ(PetscFree(Levelcnt));
  CHKERRQ(PetscFree(Idbylevel));
  CHKERRQ(PetscFree(Column));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
