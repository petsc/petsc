
static char help[] = "Tests retrieving unused PETSc options.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,N,M;
  char           **names,**values;
  PetscBool      set;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-get_an_integer",&M,&set));
  if (set) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Option used: name:-get_an_integer value: %" PetscInt_FMT "\n",M));
  PetscCall(PetscOptionsLeftGet(NULL,&N,&names,&values));
  for (i=0; i<N; i++) {
    if (values[i]) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",names[i],values[i]));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s (no value)\n",names[i]));
    }
  }
  PetscCall(PetscOptionsLeftRestore(NULL,&N,&names,&values));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: debug
      requires: defined(PETSC_USE_DEBUG)
      args: -unused_petsc_option_1 -unused_petsc_option_2 -get_an_integer 10 -options_left no

   test:
      suffix: opt
      requires: !defined(PETSC_USE_DEBUG)
      args: -unused_petsc_option_1 -unused_petsc_option_2 -get_an_integer 10 -options_left no

 TEST*/
