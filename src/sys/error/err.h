#pragma once

#include <petscsys.h>

static inline PetscErrorCode PetscErrorMemoryMessage(PetscErrorCode n)
{
  PetscLogDouble mem, rss;
  PetscErrorCode ierr;
  PetscBool      flg1 = PETSC_FALSE, flg2 = PETSC_FALSE, flg3 = PETSC_FALSE;

  if (n == PETSC_ERR_MEM) {
    ierr = (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
    ierr = (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
    ierr = (*PetscErrorPrintf)("destroying unneeded objects.\n");
  } else {
    ierr = (*PetscErrorPrintf)("Memory leaked due to not properly destroying\n");
    ierr = (*PetscErrorPrintf)("unneeded objects.\n");
  }
  ierr = PetscMallocGetCurrentUsage(&mem);
  ierr = PetscMemoryGetCurrentUsage(&rss);
  ierr = PetscOptionsGetBool(NULL, NULL, "-on_error_malloc_dump", &flg1, NULL);
  ierr = PetscOptionsGetBool(NULL, NULL, "-malloc_view", &flg2, NULL);
  ierr = PetscOptionsHasName(NULL, NULL, "-malloc_view_threshold", &flg3);
  if (flg2 || flg3) ierr = PetscMallocView(stdout);
  else {
    ierr = (*PetscErrorPrintf)("Memory allocated %.0f Memory used by process %.0f\n", mem, rss);
    if (flg1) ierr = PetscMallocDump(stdout);
    else ierr = (*PetscErrorPrintf)("Try running with -on_error_malloc_dump or -malloc_view for info.\n");
  }
  return ierr;
}
