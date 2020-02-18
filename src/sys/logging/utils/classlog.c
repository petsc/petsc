
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

/*@C
  PetscClassRegLogCreate - This creates a PetscClassRegLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassRegLog

  Level: developer

.seealso: PetscClassRegLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *classLog)
{
  PetscClassRegLog l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&l);CHKERRQ(ierr);

  l->numClasses = 0;
  l->maxClasses = 100;

  ierr = PetscMalloc1(l->maxClasses, &l->classInfo);CHKERRQ(ierr);

  *classLog = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscClassRegLogDestroy - This destroys a PetscClassRegLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassRegLog

  Level: developer

.seealso: PetscClassRegLogCreate()
@*/
PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog classLog)
{
  int            c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (c = 0; c < classLog->numClasses; c++) {
    ierr = PetscClassRegInfoDestroy(&classLog->classInfo[c]);CHKERRQ(ierr);
  }
  ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
  ierr = PetscFree(classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscClassRegInfoDestroy - This destroys a PetscClassRegInfo object.

  Not collective

  Input Parameter:
. c - The PetscClassRegInfo

  Level: developer

.seealso: PetscStageLogDestroy(), EventLogDestroy()
@*/
PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(c->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscClassPerfLogCreate - This creates a PetscClassPerfLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassPerfLog

  Level: developer

.seealso: PetscClassPerfLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscClassPerfLogCreate(PetscClassPerfLog *classLog)
{
  PetscClassPerfLog l;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&l);CHKERRQ(ierr);

  l->numClasses = 0;
  l->maxClasses = 100;

  ierr = PetscMalloc1(l->maxClasses, &l->classInfo);CHKERRQ(ierr);

  *classLog = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscClassPerfLogDestroy - This destroys a PetscClassPerfLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassPerfLog

  Level: developer

.seealso: PetscClassPerfLogCreate()
@*/
PetscErrorCode PetscClassPerfLogDestroy(PetscClassPerfLog classLog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
  ierr = PetscFree(classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscClassPerfInfoClear - This clears a PetscClassPerfInfo object.

  Not collective

  Input Parameter:
. classInfo - The PetscClassPerfInfo

  Level: developer

.seealso: PetscClassPerfLogCreate()
@*/
PetscErrorCode PetscClassPerfInfoClear(PetscClassPerfInfo *classInfo)
{
  PetscFunctionBegin;
  classInfo->id           = -1;
  classInfo->creations    = 0;
  classInfo->destructions = 0;
  classInfo->mem          = 0.0;
  classInfo->descMem      = 0.0;
  PetscFunctionReturn(0);
}

/*@C
  PetscClassPerfLogEnsureSize - This ensures that a PetscClassPerfLog is at least of a certain size.

  Not collective

  Input Parameters:
+ classLog - The PetscClassPerfLog
- size     - The size

  Level: developer

.seealso: PetscClassPerfLogCreate()
@*/
PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog classLog, int size)
{
  PetscClassPerfInfo *classInfo;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while (size > classLog->maxClasses) {
    ierr = PetscMalloc1(classLog->maxClasses*2, &classInfo);CHKERRQ(ierr);
    ierr = PetscArraycpy(classInfo, classLog->classInfo, classLog->maxClasses);CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);

    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  while (classLog->numClasses < size) {
    ierr = PetscClassPerfInfoClear(&classLog->classInfo[classLog->numClasses++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscClassRegLogRegister - Registers a class for logging operations in an application code.

  Not Collective

  Input Parameters:
+ classLog - The ClassLog
- cname    - The name associated with the class

  Output Parameter:
.  classid   - The classid

  Level: developer

.seealso: PetscClassIdRegister()
@*/
PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog classLog, const char cname[], PetscClassId classid)
{
  PetscClassRegInfo *classInfo;
  char              *str;
  int               c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(cname,2);
  c = classLog->numClasses++;
  if (classLog->numClasses > classLog->maxClasses) {
    ierr = PetscMalloc1(classLog->maxClasses*2, &classInfo);CHKERRQ(ierr);
    ierr = PetscArraycpy(classInfo, classLog->classInfo, classLog->maxClasses);CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);

    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  ierr = PetscStrallocpy(cname, &str);CHKERRQ(ierr);

  classLog->classInfo[c].name    = str;
  classLog->classInfo[c].classid = classid;
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscClassRegLogGetClass - This function returns the class corresponding to a given classid.

  Not Collective

  Input Parameters:
+ classLog - The PetscClassRegLog
- classid  - The cookie

  Output Parameter:
. oclass   - The class id

  Level: developer

.seealso: PetscClassIdRegister(), PetscLogObjCreateDefault(), PetscLogObjDestroyDefault()
@*/
PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog classLog, PetscClassId classid, int *oclass)
{
  int c;

  PetscFunctionBegin;
  PetscValidIntPointer(oclass,3);
  for (c = 0; c < classLog->numClasses; c++) {
    /* Could do bisection here */
    if (classLog->classInfo[c].classid == classid) break;
  }
  if (c >= classLog->numClasses) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid object classid %d\nThis could happen if you compile with PETSC_HAVE_DYNAMIC_LIBRARIES, but link with static libraries.", classid);
  *oclass = c;
  PetscFunctionReturn(0);
}

/*----------------------------------------------- Logging Functions -------------------------------------------------*/
/* Default object create logger */
PetscErrorCode PetscLogObjCreateDefault(PetscObject obj)
{
  PetscStageLog     stageLog;
  PetscClassRegLog  classRegLog;
  PetscClassPerfLog classPerfLog;
  Action            *tmpAction;
  Object            *tmpObjects;
  PetscLogDouble    start, end;
  int               oclass = 0;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetClassRegLog(stageLog, &classRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetClassPerfLog(stageLog, stage, &classPerfLog);CHKERRQ(ierr);
  ierr = PetscClassRegLogGetClass(classRegLog, obj->classid, &oclass);CHKERRQ(ierr);
  classPerfLog->classInfo[oclass].creations++;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscMalloc1(petsc_maxActions*2, &tmpAction);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpAction, petsc_actions, petsc_maxActions);CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }

  petsc_numObjects = obj->id;
  /* Record the creation action */
  if (petsc_logActions) {
    PetscTime(&petsc_actions[petsc_numActions].time);
    petsc_actions[petsc_numActions].time   -= petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = CREATE;
    petsc_actions[petsc_numActions].classid = obj->classid;
    petsc_actions[petsc_numActions].id1     = petsc_numObjects;
    petsc_actions[petsc_numActions].id2     = -1;
    petsc_actions[petsc_numActions].id3     = -1;
    petsc_actions[petsc_numActions].flops   = petsc_TotalFlops;

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Record the object */
  if (petsc_logObjects) {
    petsc_objects[petsc_numObjects].parent = -1;
    petsc_objects[petsc_numObjects].obj    = obj;

    ierr = PetscMemzero(petsc_objects[petsc_numObjects].name, sizeof(petsc_objects[0].name));CHKERRQ(ierr);
    ierr = PetscMemzero(petsc_objects[petsc_numObjects].info, sizeof(petsc_objects[0].info));CHKERRQ(ierr);

    /* Dynamically enlarge logging structures */
    if (petsc_numObjects >= petsc_maxObjects) {
      PetscTime(&start);
      ierr = PetscMalloc1(petsc_maxObjects*2, &tmpObjects);CHKERRQ(ierr);
      ierr = PetscArraycpy(tmpObjects, petsc_objects, petsc_maxObjects);CHKERRQ(ierr);
      ierr = PetscFree(petsc_objects);CHKERRQ(ierr);

      petsc_objects     = tmpObjects;
      petsc_maxObjects *= 2;
      PetscTime(&end);
      petsc_BaseTime += (end - start);
    }
  }
  PetscFunctionReturn(0);
}

/* Default object destroy logger */
PetscErrorCode PetscLogObjDestroyDefault(PetscObject obj)
{
  PetscStageLog     stageLog;
  PetscClassRegLog  classRegLog;
  PetscClassPerfLog classPerfLog;
  Action            *tmpAction;
  PetscLogDouble    start, end;
  int               oclass = 0;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  if (stage != -1) {
    /* That can happen if the log summary is output before some things are destroyed */
    ierr = PetscStageLogGetClassRegLog(stageLog, &classRegLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetClassPerfLog(stageLog, stage, &classPerfLog);CHKERRQ(ierr);
    ierr = PetscClassRegLogGetClass(classRegLog, obj->classid, &oclass);CHKERRQ(ierr);
    classPerfLog->classInfo[oclass].destructions++;
    classPerfLog->classInfo[oclass].mem += obj->mem;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  petsc_numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscMalloc1(petsc_maxActions*2, &tmpAction);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpAction, petsc_actions, petsc_maxActions);CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }
  /* Record the destruction action */
  if (petsc_logActions) {
    PetscTime(&petsc_actions[petsc_numActions].time);
    petsc_actions[petsc_numActions].time   -= petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = DESTROY;
    petsc_actions[petsc_numActions].classid = obj->classid;
    petsc_actions[petsc_numActions].id1     = obj->id;
    petsc_actions[petsc_numActions].id2     = -1;
    petsc_actions[petsc_numActions].id3     = -1;
    petsc_actions[petsc_numActions].flops   = petsc_TotalFlops;

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  if (petsc_logObjects) {
    if (obj->name) {
      ierr = PetscStrncpy(petsc_objects[obj->id].name, obj->name, 64);CHKERRQ(ierr);
    }
    petsc_objects[obj->id].obj = NULL;
    petsc_objects[obj->id].mem = obj->mem;
  }
  PetscFunctionReturn(0);
}
