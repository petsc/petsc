
static char help[] = "Tests ISDuplicate(), ISCopy(), ISShift(), ISEqualUnsorted(), ISEqual().\n\n";

#include <petscis.h>
#include <petscviewer.h>

/*
type = 0 general
type = 1 stride
type = 2 block
*/
static PetscErrorCode CreateIS(MPI_Comm comm, PetscInt type, PetscInt n, PetscInt first, PetscInt step, IS *is)
{
  PetscInt   *idx, i, j;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  first += rank * n * step;
  switch (type) {
  case 0:
    PetscCall(PetscMalloc1(n, &idx));
    for (i = 0, j = first; i < n; i++, j += step) idx[i] = j;
    PetscCall(ISCreateGeneral(comm, n, idx, PETSC_OWN_POINTER, is));
    break;
  case 1:
    PetscCall(ISCreateStride(comm, n, first, step, is));
    break;
  case 2:
    PetscCall(PetscMalloc1(n, &idx));
    for (i = 0, j = first; i < n; i++, j += step) idx[i] = j;
    PetscCall(ISCreateBlock(comm, 1, n, idx, PETSC_OWN_POINTER, is));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  IS        is[128];
  IS        tmp;
  PetscInt  n = 10, first = 0, step = 1, offset = 0;
  PetscInt  i, j = 0, type;
  PetscBool verbose = PETSC_FALSE, flg;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscArrayzero(is, sizeof(is) / sizeof(is[0])));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-first", &first, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-step", &step, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-offset", &offset, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verbose", &verbose, NULL));

  for (type = 0; type < 3; type++) {
    PetscCall(CreateIS(comm, type, n, first + offset, step, &is[j]));
    j++;

    PetscCall(CreateIS(comm, type, n, first + offset, step, &is[j]));
    PetscCall(ISCopy(is[j], is[j]));
    j++;

    PetscCall(CreateIS(comm, type, n, first + offset, step, &tmp));
    PetscCall(ISDuplicate(tmp, &is[j]));
    PetscCall(ISCopy(tmp, is[j]));
    PetscCall(ISDestroy(&tmp));
    j++;

    PetscCall(CreateIS(comm, type, n, first + offset, step, &is[j]));
    PetscCall(ISShift(is[j], 0, is[j]));
    j++;

    PetscCall(CreateIS(comm, type, n, first, step, &is[j]));
    PetscCall(ISShift(is[j], offset, is[j]));
    j++;

    PetscCall(CreateIS(comm, type, n, first + offset, step, &tmp));
    PetscCall(ISDuplicate(tmp, &is[j]));
    PetscCall(ISShift(tmp, 0, is[j]));
    PetscCall(ISDestroy(&tmp));
    j++;

    PetscCall(CreateIS(comm, type, n, first, step, &tmp));
    PetscCall(ISDuplicate(tmp, &is[j]));
    PetscCall(ISShift(tmp, offset, is[j]));
    PetscCall(ISDestroy(&tmp));
    j++;

    PetscCall(CreateIS(comm, type, n, first + 2 * offset, step, &is[j]));
    PetscCall(ISShift(is[j], -offset, is[j]));
    j++;
  }
  PetscAssert(j < (PetscInt)(sizeof(is) / sizeof(is[0])), comm, PETSC_ERR_ARG_OUTOFRANGE, "assertion failed: j < sizeof(is)/sizeof(is[0])");
  PetscCall(ISViewFromOptions(is[0], NULL, "-is0_view"));
  PetscCall(ISViewFromOptions(is[j / 2], NULL, "-is1_view"));
  for (i = 0; i < j; i++) {
    if (!is[i]) continue;
    PetscCall(ISEqualUnsorted(is[i], is[0], &flg));
    PetscCheck(flg, comm, PETSC_ERR_PLIB, "is[%02" PetscInt_FMT "] differs from is[0]", i);
    if (verbose) PetscCall(PetscPrintf(comm, "is[%02" PetscInt_FMT "] identical to is[0]\n", i));
  }
  for (i = 0; i < j; i++) PetscCall(ISDestroy(&is[i]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: 1
      nsize: 3
      args: -n 6 -first {{-2 0 1 3}} -step {{-2 0 1 3}}

    test:
      suffix: 2
      nsize: 2
      args: -n 3 -first 2 -step -1 -is0_view -is1_view -verbose

TEST*/
