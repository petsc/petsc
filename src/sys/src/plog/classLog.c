/* $Id: classLog.c,v 1.3 2001/01/27 21:42:08 knepley Exp $ */

#include "petsc.h"        /*I    "petsc.h"   I*/
#include "src/sys/src/plog/ptime.h"
#include "plog.h"

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
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
int ClassRegLogCreate(ClassRegLog *classLog) {
  ClassRegLog l;
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _ClassRegLog, &l);                                                               CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassRegInfo), &l->classInfo);                                CHKERRQ(ierr);
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
int ClassRegLogDestroy(ClassRegLog classLog) {
  int c;
  int ierr;

  PetscFunctionBegin;
  for(c = 0; c < classLog->numClasses; c++) {
    ierr = ClassRegInfoDestroy(&classLog->classInfo[c]);                                                  CHKERRQ(ierr);
  }
  ierr = PetscFree(classLog->classInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(classLog);                                                                             CHKERRQ(ierr);
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
int ClassRegInfoDestroy(ClassRegInfo *c) {
  int ierr;

  PetscFunctionBegin;
  if (c->name != PETSC_NULL) {
    ierr = PetscFree(c->name);                                                                            CHKERRQ(ierr);
  }
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
int ClassPerfLogCreate(ClassPerfLog *classLog) {
  ClassPerfLog l;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _ClassPerfLog, &l);                                                              CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassPerfInfo), &l->classInfo);                               CHKERRQ(ierr);
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
int ClassPerfLogDestroy(ClassPerfLog classLog) {
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(classLog->classInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(classLog);                                                                             CHKERRQ(ierr);
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
int ClassPerfInfoClear(ClassPerfInfo *classInfo) {
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
int ClassPerfLogEnsureSize(ClassPerfLog classLog, int size) {
  ClassPerfInfo *classInfo;
  int            ierr;

  PetscFunctionBegin;
  while(size > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(ClassPerfInfo), &classInfo);                       CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(ClassPerfInfo));     CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);                                                                CHKERRQ(ierr);
    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  while(classLog->numClasses < size) {
    ierr = ClassPerfInfoClear(&classLog->classInfo[classLog->numClasses++]);                              CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassRegLogRegister"
/*@C
  ClassRegLogRegister - Registers a class for logging operations in an application code.
  A prefered cookie is given on input, and the actual cookie is returned on output. If
  the user has no preference, PETSC_DECIDE will cause the cookie to be automatically
  assigned, and unique in this ClassLog.

  Not Collective

  Input Parameters:
+ classLog - The ClassLog
. cname    - The name associated with the class
- cookie   - The prefered cookie (or PETSC_DECIDE), and the actual cookie on output

  Level: developer

.keywords: log, class, register
.seealso: PetscLogClassRegister()
@*/
int ClassRegLogRegister(ClassRegLog classLog, const char cname[], int *cookie) {
  ClassRegInfo *classInfo;
  char         *str;
  int           c;
  int           ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(cname);
  PetscValidIntPointer(cookie);
  c = classLog->numClasses++;
  if (classLog->numClasses > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(ClassRegInfo), &classInfo);                        CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(ClassRegInfo));      CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);                                                                CHKERRQ(ierr);
    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  ierr = PetscStrallocpy(cname, &str);                                                                    CHKERRQ(ierr);
  classLog->classInfo[c].name     = str;
  if (*cookie == PETSC_DECIDE) {
    classLog->classInfo[c].cookie = ++PETSC_LARGEST_COOKIE;
  } else if (*cookie >= 0) {
    classLog->classInfo[c].cookie = *cookie;
    /* Need to check here for montonicity and insert if necessary */
  } else {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Invalid suggested cookie %d", *cookie);
  }
  *cookie = classLog->classInfo[c].cookie;
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
.seealso: PetscLogClassRegister(), PetscLogObjCreateDefault(), PetscLogObjDestroyDefault()
@*/
int ClassRegLogGetClass(ClassRegLog classLog, int cookie, int *oclass) {
  int c;

  PetscFunctionBegin;
  PetscValidIntPointer(oclass);
  for(c = 0; c < classLog->numClasses; c++) {
    /* Could do bisection here */
    if (classLog->classInfo[c].cookie == cookie) break;
  }
  if (c >= classLog->numClasses) SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid object cookie %d", cookie);
  *oclass = c;
  PetscFunctionReturn(0);
}

/*----------------------------------------------- Logging Functions -------------------------------------------------*/
/* Default object create logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjCreateDefault"
int PetscLogObjCreateDefault(PetscObject obj) {
  StageLog       stageLog;
  ClassRegLog    classRegLog;
  ClassPerfLog   classPerfLog;
  Action        *tmpAction;
  Object        *tmpObjects;
  PetscLogDouble start, end;
  int            oclass;
  int            stage;
  int            ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetClassRegLog(stageLog, &classRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetClassPerfLog(stageLog, stage, &classPerfLog);                                         CHKERRQ(ierr);
  ierr = ClassRegLogGetClass(classRegLog, obj->cookie, &oclass);                                          CHKERRQ(ierr);
  classPerfLog->classInfo[oclass].creations++;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);                                        CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));                                  CHKERRQ(ierr);
    ierr = PetscFree(actions);                                                                            CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  /* Record the creation action */
  if (logActions == PETSC_TRUE) {
    PetscTime(actions[numActions].time);
    actions[numActions].time  -= BaseTime;
    actions[numActions].action = CREATE;
    actions[numActions].event  = obj->type;
    actions[numActions].cookie = obj->cookie;
    actions[numActions].id1    = numObjects;
    actions[numActions].id2    = -1;
    actions[numActions].id3    = -1;
    actions[numActions].flops  = _TotalFlops;
    ierr = PetscTrSpace(&actions[numActions].mem, PETSC_NULL, &actions[numActions].maxmem);               CHKERRQ(ierr);
    numActions++;
  }
  /* Record the object */
  if (logObjects == PETSC_TRUE) {
    objects[numObjects].parent = -1;
    objects[numObjects].obj    = obj;
    ierr = PetscMemzero(objects[numObjects].name, 64 * sizeof(char));                                     CHKERRQ(ierr);
    ierr = PetscMemzero(objects[numObjects].info, 64 * sizeof(char));                                     CHKERRQ(ierr);
    numObjects++;
  }
  obj->id = numObjects;
  /* Dynamically enlarge logging structures */
  if (numObjects >= maxObjects) {
    PetscTime(start);
    ierr = PetscMalloc(maxObjects*2 * sizeof(Object), &tmpObjects);                                       CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpObjects, objects, maxObjects * sizeof(Object));                                 CHKERRQ(ierr);
    ierr = PetscFree(objects);                                                                            CHKERRQ(ierr);
    objects     = tmpObjects;
    maxObjects *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  PetscFunctionReturn(0);
}

/* Default object destroy logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjDestroyDefault"
int PetscLogObjDestroyDefault(PetscObject obj) {
  StageLog       stageLog;
  ClassRegLog    classRegLog;
  ClassPerfLog   classPerfLog;
  Action        *tmpAction;
  PetscLogDouble start, end;
  int            oclass, pclass;
  PetscObject    parent;
  PetscTruth     exists;
  int            stage;
  int            ierr;

  PetscFunctionBegin;
  /* Record stage info */
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetClassRegLog(stageLog, &classRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetClassPerfLog(stageLog, stage, &classPerfLog);                                         CHKERRQ(ierr);
  ierr = ClassRegLogGetClass(classRegLog, obj->cookie, &oclass);                                          CHKERRQ(ierr);
  classPerfLog->classInfo[oclass].destructions++;
  classPerfLog->classInfo[oclass].mem += obj->mem;
  /* Credit all ancestors with your memory */
  parent = obj->parent;
  while (parent != PETSC_NULL) {
    ierr = PetscObjectExists(parent, &exists);                                                            CHKERRQ(ierr);
    if (exists == PETSC_FALSE) break;
    ierr = ClassRegLogGetClass(classRegLog, parent->cookie, &pclass);                                     CHKERRQ(ierr);
    classPerfLog->classInfo[pclass].descMem += obj->mem;   
    parent = parent->parent;
  } 
  numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);                                        CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));                                  CHKERRQ(ierr);
    ierr = PetscFree(actions);                                                                            CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  /* Record the destruction action */
  if (logActions == PETSC_TRUE) {
    PetscTime(actions[numActions].time);
    actions[numActions].time  -= BaseTime;
    actions[numActions].action = DESTROY;
    actions[numActions].event  = obj->type;
    actions[numActions].cookie = obj->cookie;
    actions[numActions].id1    = obj->id;
    actions[numActions].id2    = -1;
    actions[numActions].id3    = -1;
    actions[numActions].flops  = _TotalFlops;
    ierr = PetscTrSpace(&actions[numActions].mem, PETSC_NULL, &actions[numActions].maxmem);               CHKERRQ(ierr);
    numActions++;
  }
  if (logObjects == PETSC_TRUE) {
    if (obj->parent != PETSC_NULL) {
      ierr = PetscObjectExists(obj->parent, &exists);                                                     CHKERRQ(ierr);
      if (exists == PETSC_TRUE) {
        objects[obj->id].parent = obj->parent->id;
      } else {
        objects[obj->id].parent = -1;
      }
    } else {
      objects[obj->id].parent   = -1;
    }
    if (obj->name != PETSC_NULL) {
      ierr = PetscStrncpy(objects[obj->id].name, obj->name, 64);                                          CHKERRQ(ierr);
    }
    objects[obj->id].obj      = PETSC_NULL;
    objects[obj->id].mem      = obj->mem;
  }
  PetscFunctionReturn(0);
}
