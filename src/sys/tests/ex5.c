
static char help[] = "Tests retrieving unused PETSc options.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,N,M;
  char           **names,**values;
  PetscBool      set;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-get_an_integer",&M,&set);CHKERRQ(ierr);
  if (set) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Option used: name:-get_an_integer value: %D\n",M);CHKERRQ(ierr); }
  ierr = PetscOptionsLeftGet(NULL,&N,&names,&values);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    if (values[i]) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",names[i],values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s (no value)\n",names[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsLeftRestore(NULL,&N,&names,&values);CHKERRQ(ierr);

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
