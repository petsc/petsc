static char help[] = "Tests ().\n\n";

#include <petscpf.h>
#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscScalar x = 2, f;
  PF          pf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(PFCreate(PETSC_COMM_WORLD, 1, 1, &pf));
  PetscCall(PFSetType(pf, PFSTRING, (void *)"f = 2*x;"));
  PetscCall(PFSetFromOptions(pf));
  PetscCall(PFApply(pf, 1, &x, &f));
  PetscCheck(f == 4, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in calling string function");
  PetscCall(PFDestroy(&pf));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       requires: defined(PETSC_HAVE_POPEN) defined(PETSC_USE_SHARED_LIBRARIES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
       args: -pf_view

TEST*/
