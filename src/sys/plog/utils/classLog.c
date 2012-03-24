
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc-private/logimpl.h> /*I    "petscsys.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscClassRegLogCreate"
/*@C
  PetscClassRegLogCreate - This creates a PetscClassRegLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassRegLog

  Level: developer

.keywords: log, class, create
.seealso: PetscClassRegLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *classLog)
{
  PetscClassRegLog    l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscClassRegLog, &l);CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(PetscClassRegInfo), &l->classInfo);CHKERRQ(ierr);
  *classLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscClassRegLogDestroy"
/*@C
  PetscClassRegLogDestroy - This destroys a PetscClassRegLog object.

  Not collective

  Input Paramter:
. classLog - The PetscClassRegLog

  Level: developer

.keywords: log, event, destroy
.seealso: PetscClassRegLogCreate()
@*/
PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog classLog)
{
  int            c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(c = 0; c < classLog->numClasses; c++) {
    ierr = PetscClassRegInfoDestroy(&classLog->classInfo[c]);CHKERRQ(ierr);
  }
  ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
  ierr = PetscFree(classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscClassRegInfoDestroy"
/*@C
  PetscClassRegInfoDestroy - This destroys a PetscClassRegInfo object.

  Not collective

  Input Parameter:
. c - The PetscClassRegInfo

  Level: developer

.keywords: log, class, destroy
.seealso: PetscStageLogDestroy(), EventLogDestroy()
@*/
PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(c->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ClassPerfLogCreate"
/*@C
  ClassPerfLogCreate - This creates a PetscClassPerfLog object.

  Not collective

  Input Parameter:
. classLog - The PetscClassPerfLog

  Level: developer

.keywords: log, class, create
.seealso: ClassPerfLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode ClassPerfLogCreate(PetscClassPerfLog *classLog)
{
  PetscClassPerfLog   l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscClassPerfLog, &l);CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(PetscClassPerfInfo), &l->classInfo);CHKERRQ(ierr);
  *classLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ClassPerfLogDestroy"
/*@C
  ClassPerfLogDestroy - This destroys a PetscClassPerfLog object.

  Not collective

  Input Paramter:
. classLog - The PetscClassPerfLog

  Level: developer

.keywords: log, event, destroy
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfLogDestroy(PetscClassPerfLog classLog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
  ierr = PetscFree(classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassPerfInfoClear"
/*@C
  ClassPerfInfoClear - This clears a PetscClassPerfInfo object.

  Not collective

  Input Paramter:
. classInfo - The PetscClassPerfInfo

  Level: developer

.keywords: log, class, destroy
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfInfoClear(PetscClassPerfInfo *classInfo) 
{
  PetscFunctionBegin;
  classInfo->id           = -1;
  classInfo->creations    = 0;
  classInfo->destructions = 0;
  classInfo->mem          = 0.0;
  classInfo->descMem      = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ClassPerfLogEnsureSize"
/*@C
  ClassPerfLogEnsureSize - This ensures that a PetscClassPerfLog is at least of a certain size.

  Not collective

  Input Paramters:
+ classLog - The PetscClassPerfLog
- size     - The size

  Level: developer

.keywords: log, class, size, ensure
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfLogEnsureSize(PetscClassPerfLog classLog, int size) 
{
  PetscClassPerfInfo  *classInfo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while(size > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(PetscClassPerfInfo), &classInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(PetscClassPerfInfo));CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  while(classLog->numClasses < size) {
    ierr = ClassPerfInfoClear(&classLog->classInfo[classLog->numClasses++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscClassRegLogRegister"
/*@C
  PetscClassRegLogRegister - Registers a class for logging operations in an application code.

  Not Collective

  Input Parameters:
+ classLog - The ClassLog
- cname    - The name associated with the class

  Output Parameter:
.  classid   - The classid

  Level: developer

.keywords: log, class, register
.seealso: PetscClassIdRegister()
@*/
PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog classLog, const char cname[], PetscClassId classid)
{
  PetscClassRegInfo   *classInfo;
  char           *str;
  int            c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(cname,2);
  c = classLog->numClasses++;
  if (classLog->numClasses > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(PetscClassRegInfo), &classInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(PetscClassRegInfo));CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "PetscClassRegLogGetClass"
/*@C
  PetscClassRegLogGetClass - This function returns the class corresponding to a given classid.

  Not Collective

  Input Parameters:
+ classLog - The PetscClassRegLog
- cookie   - The cookie
            
  Output Parameter:
. oclass   - The class id

  Level: developer

.keywords: log, class, register
.seealso: PetscClassIdRegister(), PetscLogObjCreateDefault(), PetscLogObjDestroyDefault()
@*/
PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog classLog, PetscClassId classid, int *oclass)
{
  int c;

  PetscFunctionBegin;
  PetscValidIntPointer(oclass,3);
  for(c = 0; c < classLog->numClasses; c++) {
    /* Could do bisection here */
    if (classLog->classInfo[c].classid == classid) break;
  }
  if (c >= classLog->numClasses) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid object classid %d\nThis often happens if you compile with PETSC_USE_DYNAMIC_LIBRARIES, but link with static libraries.", classid);
  }
  *oclass = c;
  PetscFunctionReturn(0);
}

/*----------------------------------------------- Logging Functions -------------------------------------------------*/
/* Default object create logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjCreateDefault"
PetscErrorCode PetscLogObjCreateDefault(PetscObject obj) 
{
  PetscStageLog       stageLog;
  PetscClassRegLog    classRegLog;
  PetscClassPerfLog   classPerfLog;
  Action        *tmpAction;
  Object        *tmpObjects;
  PetscLogDouble start, end;
  int            oclass = 0;
  int            stage;
  PetscErrorCode ierr;

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
    PetscTime(start);
    ierr = PetscMalloc(petsc_maxActions*2 * sizeof(Action), &tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, petsc_actions, petsc_maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);
    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(end);
    petsc_BaseTime += (end - start);
  }

  petsc_numObjects = obj->id;
  /* Record the creation action */
  if (petsc_logActions) {
    PetscTime(petsc_actions[petsc_numActions].time);
    petsc_actions[petsc_numActions].time  -= petsc_BaseTime;
    petsc_actions[petsc_numActions].action = CREATE;
    petsc_actions[petsc_numActions].classid = obj->classid;
    petsc_actions[petsc_numActions].id1    = petsc_numObjects;
    petsc_actions[petsc_numActions].id2    = -1;
    petsc_actions[petsc_numActions].id3    = -1;
    petsc_actions[petsc_numActions].flops  = petsc_TotalFlops;
    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Record the object */
  if (petsc_logObjects) {
    petsc_objects[petsc_numObjects].parent = -1;
    petsc_objects[petsc_numObjects].obj    = obj;
    ierr = PetscMemzero(petsc_objects[petsc_numObjects].name, 64 * sizeof(char));CHKERRQ(ierr);
    ierr = PetscMemzero(petsc_objects[petsc_numObjects].info, 64 * sizeof(char));CHKERRQ(ierr);

  /* Dynamically enlarge logging structures */
    if (petsc_numObjects >= petsc_maxObjects) {
      PetscTime(start);
      ierr = PetscMalloc(petsc_maxObjects*2 * sizeof(Object), &tmpObjects);CHKERRQ(ierr);
      ierr = PetscMemcpy(tmpObjects, petsc_objects, petsc_maxObjects * sizeof(Object));CHKERRQ(ierr);
      ierr = PetscFree(petsc_objects);CHKERRQ(ierr);
      petsc_objects     = tmpObjects;
      petsc_maxObjects *= 2;
      PetscTime(end);
      petsc_BaseTime += (end - start);
    }
  }
  PetscFunctionReturn(0);
}

/* Default object destroy logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjDestroyDefault"
PetscErrorCode PetscLogObjDestroyDefault(PetscObject obj)
{
  PetscStageLog       stageLog;
  PetscClassRegLog    classRegLog;
  PetscClassPerfLog   classPerfLog;
  Action        *tmpAction;
  PetscLogDouble start, end;
  int            oclass = 0;
  int            stage;
  PetscErrorCode ierr;

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
    PetscTime(start);
    ierr = PetscMalloc(petsc_maxActions*2 * sizeof(Action), &tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, petsc_actions, petsc_maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);
    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(end);
    petsc_BaseTime += (end - start);
  }
  /* Record the destruction action */
  if (petsc_logActions) {
    PetscTime(petsc_actions[petsc_numActions].time);
    petsc_actions[petsc_numActions].time  -= petsc_BaseTime;
    petsc_actions[petsc_numActions].action = DESTROY;
    petsc_actions[petsc_numActions].classid = obj->classid;
    petsc_actions[petsc_numActions].id1    = obj->id;
    petsc_actions[petsc_numActions].id2    = -1;
    petsc_actions[petsc_numActions].id3    = -1;
    petsc_actions[petsc_numActions].flops  = petsc_TotalFlops;
    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  if (petsc_logObjects) {
    if (obj->name) {
      ierr = PetscStrncpy(petsc_objects[obj->id].name, obj->name, 64);CHKERRQ(ierr);
    }
    petsc_objects[obj->id].obj      = PETSC_NULL;
    petsc_objects[obj->id].mem      = obj->mem;
  }
  PetscFunctionReturn(0);
}
