static char help[] = "Test scalability of PetscHSetI hash set.\n\n";

#include <petscsys.h>
#include <petsctime.h>
#include <petsc/private/hashseti.h>

int main(int argc, char **argv)
{
  PetscHSetI     table;
  PetscInt       N = 0, i, j, n;
  PetscBool      flag;
  PetscLogDouble t_add = 0;
  PetscLogDouble t_has = 0;
  PetscLogDouble t_del = 0;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&table);CHKERRQ(ierr);

  /* The following line silences warnings from Clang Static Analyzer */
  ierr = PetscHSetIResize(table,0);CHKERRQ(ierr);

  ierr = PetscTimeSubtract(&t_add);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      ierr  = PetscHSetIQueryAdd(table, key, &flag);CHKERRQ(ierr);
    }
  }
  ierr = PetscTimeAdd(&t_add);CHKERRQ(ierr);

  ierr = PetscHSetIGetSize(table,&n);CHKERRQ(ierr);

  ierr = PetscTimeSubtract(&t_has);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      ierr  = PetscHSetIHas(table, key, &flag);CHKERRQ(ierr);
    }
  }
  ierr = PetscTimeAdd(&t_has);CHKERRQ(ierr);

  ierr = PetscTimeSubtract(&t_del);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      ierr  = PetscHSetIQueryDel(table, key, &flag);CHKERRQ(ierr);
    }
  }
  ierr = PetscTimeAdd(&t_del);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"N = %D - table size: %D, add: %g, has: %g, del: %g\n",N,n,t_add,t_has,t_del);CHKERRQ(ierr);

  ierr = PetscHSetIDestroy(&table);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -N 32

TEST*/
