static char help[] = "Tests for DMLabel lookup\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;        /* The debugging level */
  PetscInt  pStart, pEnd; /* The label chart */
  PetscInt  numStrata;    /* The number of label strata */
  PetscReal fill;         /* Percentage of label to fill */
  PetscInt  size;         /* The number of set values */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->debug     = 0;
  options->pStart    = 0;
  options->pEnd      = 1000;
  options->numStrata = 5;
  options->fill      = 0.10;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-debug", "The debugging level", "ex6.c", options->debug, &options->debug, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-num_strata", "The number of label values", "ex6.c", options->numStrata, &options->numStrata, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-pend", "The label point limit", "ex6.c", options->pEnd, &options->pEnd, NULL, 0));
  PetscCall(PetscOptionsReal("-fill", "The percentage of label chart to set", "ex6.c", options->fill, &options->fill, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode TestSetup(DMLabel label, AppCtx *user)
{
  PetscRandom r;
  PetscInt    n = (PetscInt)(user->fill * (user->pEnd - user->pStart)), i;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetFromOptions(r)); /* -random_type <> */
  PetscCall(PetscRandomSetInterval(r, user->pStart, user->pEnd));
  PetscCall(PetscRandomSetSeed(r, 123456789L));
  PetscCall(PetscRandomSeed(r));
  user->size = 0;
  for (i = 0; i < n; ++i) {
    PetscReal p;
    PetscInt  val;

    PetscCall(PetscRandomGetValueReal(r, &p));
    PetscCall(DMLabelGetValue(label, (PetscInt)p, &val));
    if (val < 0) {
      ++user->size;
      PetscCall(DMLabelSetValue(label, (PetscInt)p, i % user->numStrata));
    }
  }
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(DMLabelCreateIndex(label, user->pStart, user->pEnd));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Created label with chart [%" PetscInt_FMT ", %" PetscInt_FMT ") and set %" PetscInt_FMT " values\n", user->pStart, user->pEnd, user->size));
  PetscFunctionReturn(0);
}

PetscErrorCode TestLookup(DMLabel label, AppCtx *user)
{
  const PetscInt pStart = user->pStart;
  const PetscInt pEnd   = user->pEnd;
  PetscInt       p, n = 0;

  PetscFunctionBegin;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  val;
    PetscBool has;

    PetscCall(DMLabelGetValue(label, p, &val));
    PetscCall(DMLabelHasPoint(label, p, &has));
    PetscCheck((val < 0 || has) || (val >= 0 || has), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label value %" PetscInt_FMT " does not match contains check %" PetscInt_FMT " for point %" PetscInt_FMT, val, (PetscInt)has, p);
    if (has) ++n;
  }
  PetscCheck(n == user->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of label points detected %" PetscInt_FMT " does not match number set %" PetscInt_FMT, n, user->size);
  /* Also put in timing code */
  PetscFunctionReturn(0);
}

PetscErrorCode TestClear(DMLabel label, AppCtx *user)
{
  PetscInt pStart = user->pStart, pEnd = user->pEnd, p;
  PetscInt defaultValue;

  PetscFunctionBegin;
  PetscCall(DMLabelGetDefaultValue(label, &defaultValue));
  for (p = pStart; p < pEnd; p++) {
    PetscInt  val;
    PetscBool hasPoint;

    PetscCall(DMLabelGetValue(label, p, &val));
    if (val != defaultValue) PetscCall(DMLabelClearValue(label, p, val));
    PetscCall(DMLabelGetValue(label, p, &val));
    PetscCall(DMLabelHasPoint(label, p, &hasPoint));
    PetscCheck(val == defaultValue, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected default value %" PetscInt_FMT " after clearing point %" PetscInt_FMT ", got %" PetscInt_FMT, defaultValue, p, val);
    PetscCheck(!hasPoint, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label contains %" PetscInt_FMT " after clearing", p);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DMLabel label;
  AppCtx  user; /* user-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Test Label", &label));
  PetscCall(TestSetup(label, &user));
  PetscCall(TestLookup(label, &user));
  PetscCall(TestClear(label, &user));
  PetscCall(DMLabelDestroy(&label));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -malloc_dump
  test:
    suffix: 1
    args: -malloc_dump -pend 10000
  test:
    suffix: 2
    args: -malloc_dump -pend 10000 -fill 0.05
  test:
    suffix: 3
    args: -malloc_dump -pend 10000 -fill 0.25

TEST*/
