/*
    Macros for preloading a block of code so that startup costs (such as first-touch
    allocation) are not attributed to the measured run. See `PetscPreLoadBegin()`.
*/
#pragma once

#include <petscsys.h>

/* MANSEC = Sys */
/* SUBMANSEC = Log */

/*MC
  PetscPreLoadBegin - Begin a block of code that is timed twice so that startup costs (such as JIT or first-touch allocation) are not attributed to the measured run

  Synopsis:
  #include <petscpreload.h>
  PetscPreLoadBegin(PetscBool flag, const char name[])

  Not Collective; No Fortran Support

  Input Parameters:
+ flag - whether preloading is desired (may be overridden by the command-line option `-preload`)
- name - name to use for the `PetscLogStage` created for the measured run

  Level: intermediate

  Note:
  Use this macro in the form
.vb
    PetscPreLoadBegin(PETSC_TRUE, "Compute");
    // ... code to be timed
    PetscPreLoadEnd();
.ve
  It expands to a loop that runs the enclosed code twice when preloading is enabled and once otherwise.
  Pair with `PetscPreLoadEnd()`. Use `PetscPreLoadStage()` to advance to a new named stage between phases.

.seealso: `PetscPreLoadEnd()`, `PetscPreLoadStage()`, `PetscLogStageRegister()`, `PetscLogStagePush()`
M*/
#define PetscPreLoadBegin(flag, name) \
  do { \
    PetscBool     PetscPreLoading = flag; \
    int           PetscPreLoadMax, PetscPreLoadIt; \
    PetscLogStage _stageNum; \
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-preload", &PetscPreLoading, NULL)); \
    PetscPreLoadMax     = (int)(PetscPreLoading); \
    PetscPreLoadingUsed = PetscPreLoading ? PETSC_TRUE : PetscPreLoadingUsed; \
    PetscCall(PetscLogStageGetId(name, &_stageNum)); \
    for (PetscPreLoadIt = (_stageNum == -1) ? 0 : PetscPreLoadMax; PetscPreLoadIt <= PetscPreLoadMax; PetscPreLoadIt++) { \
      PetscPreLoadingOn = (PetscBool)(PetscPreLoadIt < PetscPreLoadMax); \
      PetscCall(PetscBarrier(NULL)); \
      if (_stageNum == -1) PetscCall(PetscLogStageRegister(name, &_stageNum)); \
      PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(PetscPreLoadIt == PetscPreLoadMax))); \
      PetscCall(PetscLogStagePush(_stageNum))

/*MC
  PetscPreLoadEnd - Close a preload block started with `PetscPreLoadBegin()`

  Synopsis:
  #include <petscpreload.h>
  PetscPreLoadEnd()

  Not Collective; No Fortran Support

  Level: intermediate

.seealso: `PetscPreLoadBegin()`, `PetscPreLoadStage()`
M*/
#define PetscPreLoadEnd() \
  PetscCall(PetscLogStagePop()); \
  } \
  } \
  while (0)

/*MC
  PetscPreLoadStage - Advance to a new named stage inside a `PetscPreLoadBegin()` / `PetscPreLoadEnd()` block

  Synopsis:
  #include <petscpreload.h>
  PetscPreLoadStage(const char name[])

  Not Collective; No Fortran Support

  Input Parameter:
. name - name for the new `PetscLogStage`

  Level: intermediate

  Note:
  Pops the previous stage, registers (or reuses) the new stage, and pushes it. Only valid between
  `PetscPreLoadBegin()` and `PetscPreLoadEnd()`.

.seealso: `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscLogStagePush()`, `PetscLogStagePop()`
M*/
#define PetscPreLoadStage(name) \
  do { \
    PetscCall(PetscLogStagePop()); \
    PetscCall(PetscLogStageGetId(name, &_stageNum)); \
    if (_stageNum == -1) PetscCall(PetscLogStageRegister(name, &_stageNum)); \
    PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(!PetscPreLoadMax || PetscPreLoadIt))); \
    PetscCall(PetscLogStagePush(_stageNum)); \
  } while (0)

PETSC_EXTERN PetscBool PetscPreLoadingUsed; /* true if we are or have done preloading */
PETSC_EXTERN PetscBool PetscPreLoadingOn;   /* true if we are currently in a preloading calculation */
