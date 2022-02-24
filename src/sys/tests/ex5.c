
static char help[] = "Tests retrieving unused PETSc options.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,N,M;
  char           **names,**values;
  PetscBool      set;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-get_an_integer",&M,&set));
  if (set) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Option used: name:-get_an_integer value: %" PetscInt_FMT "\n",M));
  CHKERRQ(PetscOptionsLeftGet(NULL,&N,&names,&values));
  for (i=0; i<N; i++) {
    if (values[i]) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",names[i],values[i]));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s (no value)\n",names[i]));
    }
  }
  CHKERRQ(PetscOptionsLeftRestore(NULL,&N,&names,&values));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: debug
      requires: defined(PETSC_USE_DEBUG)
      args: -unused_petsc_option_1 -unused_petsc_option_2 -get_an_integer 10 -options_left no
      filter: egrep -v \(malloc_dump\|options_left\|nox\|vecscatter_mpi1\|saws_port_auto_select\|saws_port_auto_select_silent\)

   test:
      suffix: opt
      requires: !defined(PETSC_USE_DEBUG)
      args: -checkstack -unused_petsc_option_1 -unused_petsc_option_2 -get_an_integer 10 -options_left no
      filter: egrep -v \(malloc_dump\|options_left\|nox\|vecscatter_mpi1\|saws_port_auto_select\|saws_port_auto_select_silent\)

 TEST*/
