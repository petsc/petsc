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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug     = 0;
  options->pStart    = 0;
  options->pEnd      = 1000;
  options->numStrata = 5;
  options->fill      = 0.10;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex6.c", options->debug, &options->debug, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_strata", "The number of label values", "ex6.c", options->numStrata, &options->numStrata, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-pend", "The label point limit", "ex6.c", options->pEnd, &options->pEnd, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-fill", "The percentage of label chart to set", "ex6.c", options->fill, &options->fill, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  for (i = 0; i < n; ++i) {
    PetscReal p;
    PetscInt  val;

    ierr = PetscRandomGetValueReal(r, &p);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label, (PetscInt) p, &val);CHKERRQ(ierr);
    if (val < 0) {
      ++user->size;
      ierr = DMLabelSetValue(label, (PetscInt) p, i % user->numStrata);CHKERRQ(ierr);
    }
  }
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = DMLabelCreateIndex(label, user->pStart, user->pEnd);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Created label with chart [%D, %D) and set %D values\n", user->pStart, user->pEnd, user->size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestLookup(DMLabel label, AppCtx *user)
{
  const PetscInt pStart = user->pStart;
  const PetscInt pEnd   = user->pEnd;
  PetscInt       p, n = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  val;
    PetscBool has;

    ierr = DMLabelGetValue(label, p, &val);CHKERRQ(ierr);
    ierr = DMLabelHasPoint(label, p, &has);CHKERRQ(ierr);
    PetscAssertFalse(((val >= 0) && !has) || ((val < 0) && has),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label value %D does not match contains check %D for point %D", val, (PetscInt) has, p);
    if (has) ++n;
  }
  PetscAssertFalse(n != user->size,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of label points detected %D does not match number set %D", n, user->size);
  /* Also put in timing code */
  PetscFunctionReturn(0);
}

PetscErrorCode TestClear(DMLabel label, AppCtx *user)
{
  PetscInt       pStart = user->pStart, pEnd = user->pEnd, p;
  PetscInt       defaultValue;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetDefaultValue(label,&defaultValue);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt  val;
    PetscBool hasPoint;

    ierr = DMLabelGetValue(label,p,&val);CHKERRQ(ierr);
    if (val != defaultValue) {
      ierr = DMLabelClearValue(label,p,val);CHKERRQ(ierr);
    }
    ierr = DMLabelGetValue(label,p,&val);CHKERRQ(ierr);
    ierr = DMLabelHasPoint(label,p,&hasPoint);CHKERRQ(ierr);
    PetscAssertFalse(val != defaultValue,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected default value %D after clearing point %D, got %D",defaultValue,p,val);
    PetscAssertFalse(hasPoint,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Label contains %D after clearing",p);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DMLabel        label;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Test Label", &label);CHKERRQ(ierr);
  ierr = TestSetup(label, &user);CHKERRQ(ierr);
  ierr = TestLookup(label, &user);CHKERRQ(ierr);
  ierr = TestClear(label,&user);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
