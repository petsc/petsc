static char help[] = "Test resource recycling and MPI_Comm and keyval creation in mpi or mpiuni, no PETSc\n";

#include <petscsys.h>

#define CHKMPIERR(err) \
  do { \
    if (err) MPI_Abort(MPI_COMM_WORLD, err); \
  } while (0)

int main(int argc, char **argv)
{
  int         err;
  PetscInt    i;
  PetscMPIInt key1, key2, attr1 = 100, attr2 = 200, *attr, iflg;
  MPI_Comm    newcomm;

  err = MPI_Init(&argc, &argv);
  CHKMPIERR(err);

  /* Repeated keyval or comm create/free should not blow up MPI */
  for (i = 0; i < 500; i++) {
    err = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &key1, NULL);
    CHKMPIERR(err);
    err = MPI_Comm_free_keyval(&key1);
    CHKMPIERR(err);
    err = MPI_Comm_dup(MPI_COMM_WORLD, &newcomm);
    CHKMPIERR(err);
    err = MPI_Comm_free(&newcomm);
    CHKMPIERR(err);
  }

  /* The following keyval/attr code exposes a bug in old mpiuni code, where it had wrong newcomm returned in MPI_Comm_dup. */
  err = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &key1, NULL);
  CHKMPIERR(err);
  err = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &key2, NULL);
  CHKMPIERR(err);
  err = MPI_Comm_dup(MPI_COMM_WORLD, &newcomm);
  CHKMPIERR(err);
  if (MPI_COMM_WORLD == newcomm) printf("Error: wrong newcomm returned by MPI_Comm_dup()\n");

  err = MPI_Comm_set_attr(MPI_COMM_WORLD, key1, &attr1);
  CHKMPIERR(err);
  err = MPI_Comm_set_attr(newcomm, key2, &attr2);
  CHKMPIERR(err);
  err = MPI_Comm_get_attr(newcomm, key1, &attr, &iflg);
  CHKMPIERR(err);
  if (iflg) printf("Error: newcomm should not have attribute for keyval %d\n", key1);
  err = MPI_Comm_get_attr(MPI_COMM_WORLD, key1, &attr, &iflg);
  CHKMPIERR(err);
  if (*attr != attr1) printf("Error: expected attribute %d, but got %d\n", attr1, *attr);
  err = MPI_Comm_get_attr(newcomm, key2, &attr, &iflg);
  CHKMPIERR(err);
  if (*attr != attr2) printf("Error: expected attribute %d, but got %d\n", attr2, *attr);

  err = MPI_Comm_delete_attr(MPI_COMM_WORLD, key1);
  CHKMPIERR(err);
  err = MPI_Comm_delete_attr(newcomm, key2);
  CHKMPIERR(err);
  err = MPI_Comm_free_keyval(&key1);
  CHKMPIERR(err);
  err = MPI_Comm_free_keyval(&key2);
  CHKMPIERR(err);
  err = MPI_Comm_free(&newcomm);
  CHKMPIERR(err);

  /* Init/Finalize PETSc multiple times when MPI is initialized */
  for (i = 0; i < 500; i++) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCall(PetscFinalize(); if (err) return err);
  }

  err = MPI_Finalize();
  return err;
}

/*TEST
   # Elemental in debug mode has bugs that it can not be repeatedly init/finalize'd for more than 300 times
   testset:
    output_file: output/empty.out
    test:
      suffix: 1
      requires: !elemental

    test:
      suffix: 2
      requires: elemental !defined(PETSC_USE_DEBUG)
TEST*/
