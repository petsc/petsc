/* $Id: classLog.c,v 1.3 2001/01/27 21:42:08 knepley Exp $ */

#include "petsc.h"        /*I    "petsc.h"   I*/
#include "src/sys/src/plog/ptime.h"
#include "plog.h"

/*------------------------------------------------ General Functions ------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassLogDestroy"
/*
  ClassLogDestroy - This destroys a ClassLog object.

  Not collective

  Input Paramter:
. classLog - The ClassLog

  Level: beginner

.keywords: log, event, destroy
.seealso: ClassLogCreate()
*/
int ClassLogDestroy(ClassLog classLog)
{
  int c;
  int ierr;

  PetscFunctionBegin;
  for(c = 0; c < classLog->numClasses; c++) {
    ierr = ClassInfoDestroy(&classLog->classInfo[c]);                                                     CHKERRQ(ierr);
  }
  ierr = PetscFree(classLog->classInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(classLog);                                                                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ClassLogCopy"
/*
  ClassLogCopy - This copys an ClassLog object.

  Not collective

  Input Parameter:
. classLog - The ClassLog

  Output Parameter:
. newLog   - The copy

  Level: beginner

.keywords: log, class, copy
.seealso: ClassLogCreate(), ClassLogDestroy()
*/
int ClassLogCopy(ClassLog classLog, ClassLog *newLog)
{
  ClassLog l;
  int      c;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _ClassLog, &l);                                                                  CHKERRQ(ierr);
  l->numClasses = classLog->numClasses;
  l->maxClasses = classLog->maxClasses;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassInfo), &l->classInfo);                                   CHKERRQ(ierr);
  for(c = 0; c < classLog->numClasses; c++) {
    ierr = PetscStrallocpy(classLog->classInfo[c].name, &l->classInfo[c].name);                           CHKERRQ(ierr);
    l->classInfo[c].cookie       = classLog->classInfo[c].cookie;
    l->classInfo[c].creations    = 0;
    l->classInfo[c].destructions = 0;
    l->classInfo[c].mem          = 0.0;
    l->classInfo[c].descMem      = 0.0;
  }
  *newLog = l;
  PetscFunctionReturn(0);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassLogRegister"
/*@C
  ClassLogRegister - Registers a class for logging operations in an application code.
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
int ClassLogRegister(ClassLog classLog, const char cname[], int *cookie) {
  ClassInfo *classInfo;
  char      *str;
  int        c;
  int        ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(cname);
  PetscValidIntPointer(cookie);
  c = classLog->numClasses++;
  if (classLog->numClasses > classLog->maxClasses) {
    ierr = PetscMalloc(classLog->maxClasses*2 * sizeof(ClassInfo), &classInfo);                           CHKERRQ(ierr);
    ierr = PetscMemcpy(classInfo, classLog->classInfo, classLog->maxClasses * sizeof(ClassInfo));         CHKERRQ(ierr);
    ierr = PetscFree(classLog->classInfo);                                                                CHKERRQ(ierr);
    classLog->classInfo   = classInfo;
    classLog->maxClasses *= 2;
  }
  ierr = PetscStrallocpy(cname, &str);                                                                    CHKERRQ(ierr);
  classLog->classInfo[c].name         = str;
  classLog->classInfo[c].creations    = 0;
  classLog->classInfo[c].destructions = 0;
  classLog->classInfo[c].mem          = 0.0;
  classLog->classInfo[c].descMem      = 0.0;
  if (*cookie == PETSC_DECIDE) {
    classLog->classInfo[c].cookie     = ++PETSC_LARGEST_COOKIE;
  } else if (*cookie >= 0) {
    classLog->classInfo[c].cookie     = *cookie;
    /* Need to check here for montonicity and insert if necessary */
  } else {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Invalid suggested cookie %d", *cookie);
  }
  *cookie = classLog->classInfo[c].cookie;
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassLogGetClass"
/*@C
  ClassLogGetClass - This function returns the class corresponding to a given cookie.

  Not Collective

  Input Parameters:
+ classLog - The ClassLog
- cookie   - The cookie
            
  Output Parameter:
. oclass   - The class id

  Level: developer

.keywords: log, class, register
.seealso: PetscLogClassRegister(), PetscLogObjCreateDefault(), PetscLogObjDestroyDefault()
@*/
int ClassLogGetClass(ClassLog classLog, int cookie, int *oclass)
{
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

/*----------------------------------------------- Creation Function -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ClassLogCreate"
/*
  ClassLogCreate - This creates a ClassLog object.

  Not collective

  Input Parameter:
. classLog - The ClassLog

  Level: beginner

.keywords: log, class, create
.seealso: ClassLogDestroy(), StageLogCreate()
*/
int ClassLogCreate(ClassLog *classLog)
{
  ClassLog l;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _ClassLog, &l);                                                                  CHKERRQ(ierr);
  l->numClasses = 0;
  l->maxClasses = 100;
  ierr = PetscMalloc(l->maxClasses * sizeof(ClassInfo), &l->classInfo);                                   CHKERRQ(ierr);
  *classLog = l;
  PetscFunctionReturn(0);
}

/*----------------------------------------------- Logging Functions -------------------------------------------------*/
/* Default object create logger */
#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjCreateDefault"
int PetscLogObjCreateDefault(PetscObject obj)
{
  StageLog       stageLog;
  ClassLog       classLog;
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
  ierr = StageLogGetClassLog(stageLog, stage, &classLog);                                                 CHKERRQ(ierr);
  ierr = ClassLogGetClass(classLog, obj->cookie, &oclass);                                                CHKERRQ(ierr);
  classLog->classInfo[oclass].creations++;
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
  if (actions != PETSC_NULL) {
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
  if (objects != PETSC_NULL) {
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
int PetscLogObjDestroyDefault(PetscObject obj)
{
  StageLog       stageLog;
  ClassLog       classLog;
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
  ierr = StageLogGetClassLog(stageLog, stage, &classLog);                                                 CHKERRQ(ierr);
  ierr = ClassLogGetClass(classLog, obj->cookie, &oclass);                                                CHKERRQ(ierr);
  classLog->classInfo[oclass].destructions++;
  classLog->classInfo[oclass].mem += obj->mem;
  /* Credit all ancestors with your memory */
  parent = obj->parent;
  while (parent != PETSC_NULL) {
    ierr = PetscObjectExists(parent, &exists);                                                            CHKERRQ(ierr);
    if (exists == PETSC_FALSE) break;
    ierr = ClassLogGetClass(classLog, parent->cookie, &pclass);                                           CHKERRQ(ierr);
    classLog->classInfo[pclass].descMem += obj->mem;   
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
  if (actions != PETSC_NULL) {
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
  if (objects != PETSC_NULL) {
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
