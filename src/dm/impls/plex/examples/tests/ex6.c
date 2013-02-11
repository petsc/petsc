static char help[] = "Tests for DMLabel lookup\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;        /* The debugging level */
  PetscInt  pStart, pEnd; /* The label chart */
  PetscInt  numStrata;    /* The number of label strata */
  PetscReal fill;         /* Percentage of label to fill */
  PetscInt  size;         /* The number of set values */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug     = 0;
  options->pStart    = 0;
  options->pEnd      = 1000;
  options->numStrata = 5;
  options->fill      = 0.10;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex6.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_strata", "The number of label values", "ex6.c", options->numStrata, &options->numStrata, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pend", "The label point limit", "ex6.c", options->pEnd, &options->pEnd, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-fill", "The percentage of label chart to set", "ex6.c", options->fill, &options->fill, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "TestSetup"
PetscErrorCode TestSetup(DMLabel label, AppCtx *user)
{
  PetscRandom    r;
  PetscInt       n = (PetscInt) (user->fill*(user->pEnd - user->pStart)), i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);/* -random_type <> */
  ierr = PetscRandomSetInterval(r, user->pStart, user->pEnd);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r, 123456789L);CHKERRQ(ierr);
  ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  user->size = 0;
  for(i = 0; i < n; ++i) {
    PetscReal p;
    PetscInt  val;

    ierr = PetscRandomGetValueReal(r, &p);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label, (PetscInt) p, &val);CHKERRQ(ierr);
    if (val < 0) ++user->size;
    ierr = DMLabelSetValue(label, (PetscInt) p, i % user->numStrata);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = DMLabelCreateIndex(label, user->pStart, user->pEnd);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Created label with chart [%d, %d) and set %d values\n", user->pStart, user->pEnd, user->size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "TestLookup"
PetscErrorCode TestLookup(DMLabel label, AppCtx *user)
{
  const PetscInt pStart = user->pStart;
  const PetscInt pEnd   = user->pEnd;
  PetscInt       p, n = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(p = pStart; p < pEnd; ++p) {
    PetscInt  val;
    PetscBool has;

    ierr = DMLabelGetValue(label, p, &val);CHKERRQ(ierr);
    ierr = DMLabelHasPoint(label, p, &has);CHKERRQ(ierr);
    if (((val >= 0) && !has) || ((val < 0) && has)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label value %d does not match contains check %d for point %d", val, (PetscInt) has, p);
    if (has) ++n;
  }
  if (n != user->size) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of label points detected %d does not match number set %d", n, user->size);
  /* Also put in timing code */
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DMLabel        label;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMLabelCreate("Test Label", &label);CHKERRQ(ierr);
  ierr = TestSetup(label, &user);CHKERRQ(ierr);
  ierr = TestLookup(label, &user);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
