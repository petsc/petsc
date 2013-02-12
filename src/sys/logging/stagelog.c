#include <petsc-private/petscimpl.h> /*I "petscsys.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscLogGetStageLog"
/*@C
  PetscLogGetStageLog - This function returns the default stage logging object.

  Not collective

  Output Parameter:
. stageLog - The default PetscStageLog

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: PetscStageLogCreate()
@*/
PetscErrorCode PetscLogGetStageLog(PetscStageLog *stageLog)
{
  PetscFunctionBegin;
  PetscValidPointer(stageLog,1);
  if (!petsc_stageLog) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    MPI_Abort(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *stageLog = petsc_stageLog;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscStageLogGetCurrent"
/*@C
  PetscStageLogGetCurrent - This function returns the stage from the top of the stack.

  Not Collective

  Input Parameter:
. stageLog - The PetscStageLog

  Output Parameter:
. stage    - The current stage

  Notes:
  If no stage is currently active, stage is set to -1.

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: PetscStageLogPush(), PetscStageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode  PetscStageLogGetCurrent(PetscStageLog stageLog, int *stage)
{
  PetscBool      empty;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscIntStackEmpty(stageLog->stack, &empty);CHKERRQ(ierr);
  if (empty) {
    *stage = -1;
  } else {
    ierr = PetscIntStackTop(stageLog->stack, stage);CHKERRQ(ierr);
  }
#ifdef PETSC_USE_DEBUG
  if (*stage != stageLog->curStage) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Inconsistency in stage log: stage %d should be %d", *stage, stageLog->curStage);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscStageLogGetEventPerfLog"
/*@C
  PetscStageLogGetEventPerfLog - This function returns the PetscEventPerfLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The PetscStageLog
- stage    - The stage

  Output Parameter:
. eventLog - The PetscEventPerfLog

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: PetscStageLogPush(), PetscStageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode  PetscStageLogGetEventPerfLog(PetscStageLog stageLog, int stage, PetscEventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog,3);
  if ((stage < 0) || (stage >= stageLog->numStages)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  *eventLog = stageLog->stageInfo[stage].eventLog;
  PetscFunctionReturn(0);
}
