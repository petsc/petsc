#define PETSC_DLL

#include "../src/sys/plog/plog.h" /*I    "petscsys.h"   I*/

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */
#undef __FUNCT__  
#define __FUNCT__ "ClassRegLogCreate"
/*@C
  ClassRegLogCreate - This creates a ClassRegLog object.

  Not collective

  Input Parameter:
. classLog - The ClassRegLog

  Level: beginner

.keywords: log, class, create
.seealso: ClassRegLogDestroy(), StageLogCreate()
@*/
PetscErrorCode ClassRegLogCreate(ClassRegLog *classLog)
{
  ClassRegLog l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_ClassRegLog, &l);CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassRegInfo), &l->classInfo);CHKERRQ(ierr);
  *classLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ClassRegLogDestroy"
/*@C
  ClassRegLogDestroy - This destroys a ClassRegLog object.

  Not collective

  Input Paramter:
. classLog - The ClassRegLog

  Level: beginner

.keywords: log, event, destroy
.seealso: ClassRegLogCreate()
@*/
PetscErrorCode ClassRegLogDestroy(ClassRegLog classLog)\
{
  int c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(c = 0; c < classLog->numClasses; c++) {
    ierr = ClassRegInfoDestroy(&classLog->classInfo[c]);CHKERRQ(ierr);
  }
  ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
  ierr = PetscFree(classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ClassRegInfoDestroy"
/*@C
  ClassRegInfoDestroy - This destroys a ClassRegInfo object.

  Not collective

  Input Parameter:
. c - The ClassRegInfo

  Level: beginner

.keywords: log, class, destroy
.seealso: StageLogDestroy(), EventLogDestroy()
@*/
PetscErrorCode ClassRegInfoDestroy(ClassRegInfo *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(c->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ClassPerfLogCreate"
/*@C
  ClassPerfLogCreate - This creates a ClassPerfLog object.

  Not collective

  Input Parameter:
. classLog - The ClassPerfLog

  Level: beginner

.keywords: log, class, create
.seealso: ClassPerfLogDestroy(), StageLogCreate()
@*/
PetscErrorCode ClassPerfLogCreate(ClassPerfLog *classLog)
{
  ClassPerfLog l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_ClassPerfLog, &l);CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassPerfInfo), &l->classInfo);CHKERRQ(ierr);
  *classLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ClassPerfLogDestroy"
/*@C
  ClassPerfLogDestroy - This destroys a ClassPerfLog object.

  Not collective

  Input Paramter:
. classLog - The ClassPerfLog

  Level: beginner

.keywords: log, event, destroy
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfLogDestroy(ClassPerfLog classLog)
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
  ClassPerfInfoClear - This clears a ClassPerfInfo object.

  Not collective

  Input Paramter:
. classInfo - The ClassPerfInfo

  Level: beginner

.keywords: log, class, destroy
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfInfoClear(ClassPerfInfo *classInfo) 
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
  ClassPerfLogEnsureSize - This ensures that a ClassPerfLog is at least of a certain size.

  Not collective

  Input Paramters:
+ classLog - The ClassPerfLog
- size     - The size

  Level: intermediate

.keywords: log, class, size, ensure
.seealso: ClassPerfLogCreate()
@*/
PetscErrorCode ClassPerfLogEnsureSize(ClassPerfLog classLog, int size) 
{
  ClassPerfInfo *classInfo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while(size > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(ClassPerfInfo), &classInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(ClassPerfInfo));CHKERRQ(ierr);
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
#define __FUNCT__ "ClassRegLogRegister"
/*@C
  ClassRegLogRegister - Registers a class for logging operations in an application code.

  Not Collective

  Input Parameters:
+ classLog - The ClassLog
- cname    - The name associated with the class

  Output Parameter:
.  cookie   - The cookie

  Level: developer

.keywords: log, class, register
.seealso: PetscCookieRegister()
@*/
PetscErrorCode ClassRegLogRegister(ClassRegLog classLog, const char cname[], PetscCookie cookie)
{
  ClassRegInfo *classInfo;
  char         *str;
  int           c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(cname,2);
  c = classLog->numClasses++;
  if (classLog->numClasses > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(ClassRegInfo), &classInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(ClassRegInfo));CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);CHKERRQ(ierr);
    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  ierr = PetscStrallocpy(cname, &str);CHKERRQ(ierr);
  classLog->classInfo[c].name     = str;
  classLog->classInfo[c].cookie = cookie;
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassRegLogGetClass"
/*@C
  ClassRegLogGetClass - This function returns the class corresponding to a given cookie.

  Not Collective

  Input Parameters:
+ classLog - The ClassRegLog
- cookie   - The cookie
            
  Output Parameter:
. oclass   - The class id

  Level: developer

.keywords: log, class, register
.seealso: PetscCookieRegister(), PetscLogObjCreateDefault(), PetscLogObjDestroyDefault()
@*/
PetscErrorCode ClassRegLogGetClass(ClassRegLog classLog, PetscCookie cookie, int *oclass)
{
  int c;

  PetscFunctionBegin;
  PetscValidIntPointer(oclass,3);
  for(c = 0; c < classLog->numClasses; c++) {
    /* Could do bisection here */
    if (classLog->classInfo[c].cookie == cookie) break;
  }
  if (c >= classLog->numClasses) {
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid object cookie %d\nThis often happens if you compile with PETSC_USE_DYNAMIC_LIBRARIES, but link with static libraries.", cookie);
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
  StageLog       stageLog;
  ClassRegLog    classRegLog;
  ClassPerfLog   classPerfLog;
  Action        *tmpAction;
  Object        *tmpObjects;
  PetscLogDouble start, end;
  int            oclass = 0;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = StageLogGetClassRegLog(stageLog, &classRegLog);CHKERRQ(ierr);
  ierr = StageLogGetClassPerfLog(stageLog, stage, &classPerfLog);CHKERRQ(ierr);
  ierr = ClassRegLogGetClass(classRegLog, obj->cookie, &oclass);CHKERRQ(ierr);
  classPerfLog->classInfo[oclass].creations++;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(actions);CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }

  numObjects = obj->id;
  /* Record the creation action */
  if (logActions) {
    PetscTime(actions[numActions].time);
    actions[numActions].time  -= BaseTime;
    actions[numActions].action = CREATE;
    actions[numActions].cookie = obj->cookie;
    actions[numActions].id1    = numObjects;
    actions[numActions].id2    = -1;
    actions[numActions].id3    = -1;
    actions[numActions].flops  = _TotalFlops;
    ierr = PetscMallocGetCurrentUsage(&actions[numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&actions[numActions].maxmem);CHKERRQ(ierr);
    numActions++;
  }
  /* Record the object */
  if (logObjects) {
    objects[numObjects].parent = -1;
    objects[numObjects].obj    = obj;
    ierr = PetscMemzero(objects[numObjects].name, 64 * sizeof(char));CHKERRQ(ierr);
    ierr = PetscMemzero(objects[numObjects].info, 64 * sizeof(char));CHKERRQ(ierr);

  /* Dynamically enlarge logging structures */
    if (numObjects >= maxObjects) {
      PetscTime(start);
      ierr = PetscMalloc(maxObjects*2 * sizeof(Object), &tmpObjects);CHKERRQ(ierr);
      ierr = PetscMemcpy(tmpObjects, objects, maxObjects * sizeof(Object));CHKERRQ(ierr);
      ierr = PetscFree(objects);CHKERRQ(ierr);
      objects     = tmpObjects;
      maxObjects *= 2;
      PetscTime(end);
      BaseTime += (end - start);
    }
  }
  PetscFunctionReturn(0);
}

/* Default object destroy logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjDestroyDefault"
PetscErrorCode PetscLogObjDestroyDefault(PetscObject obj)
{
  StageLog       stageLog;
  ClassRegLog    classRegLog;
  ClassPerfLog   classPerfLog;
  Action        *tmpAction;
  PetscLogDouble start, end;
  int            oclass = 0;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  if (stage != -1) {
    /* That can happen if the log summary is output before some things are destroyed */
    ierr = StageLogGetClassRegLog(stageLog, &classRegLog);CHKERRQ(ierr);
    ierr = StageLogGetClassPerfLog(stageLog, stage, &classPerfLog);CHKERRQ(ierr);
    ierr = ClassRegLogGetClass(classRegLog, obj->cookie, &oclass);CHKERRQ(ierr);
    classPerfLog->classInfo[oclass].destructions++;
    classPerfLog->classInfo[oclass].mem += obj->mem;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(actions);CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  /* Record the destruction action */
  if (logActions) {
    PetscTime(actions[numActions].time);
    actions[numActions].time  -= BaseTime;
    actions[numActions].action = DESTROY;
    actions[numActions].cookie = obj->cookie;
    actions[numActions].id1    = obj->id;
    actions[numActions].id2    = -1;
    actions[numActions].id3    = -1;
    actions[numActions].flops  = _TotalFlops;
    ierr = PetscMallocGetCurrentUsage(&actions[numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&actions[numActions].maxmem);CHKERRQ(ierr);
    numActions++;
  }
  if (logObjects) {
    if (obj->name) {
      ierr = PetscStrncpy(objects[obj->id].name, obj->name, 64);CHKERRQ(ierr);
    }
    objects[obj->id].obj      = PETSC_NULL;
    objects[obj->id].mem      = obj->mem;
  }
  PetscFunctionReturn(0);
}
