#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tscreate.c,v 1.7 2000/01/10 03:54:25 knepley Exp $";
#endif

#include "src/ts/tsimpl.h"      /*I "petscts.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "TSPublish_Petsc"
static int TSPublish_Petsc(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  TS   v = (TS) obj;
  int  ierr;
#endif  

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Step",&v->steps,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Time",&v->ptime,1,AMS_DOUBLE,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"CurrentTimeStep",&v->time_step,1,
                               AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "TSCreate"
/*@ 
  TSCreate - This function creates an empty timestepper. The type can then be set with TSSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator

  Output Parameter:
. ts   - The TS

  Level: beginner

.keywords: TS, create
.seealso: TSSetType(), TSSetUp(), TSDestroy(), MeshCreate()
@*/
int TSCreate(MPI_Comm comm, TS *ts) {
  TS  t;
  int ierr;

  PetscFunctionBegin;
  PetscValidPointer(ts);
  *ts = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = TSInitializePackage(PETSC_NULL);                                                                 CHKERRQ(ierr);
#endif

  PetscHeaderCreate(t, _p_TS, struct _TSOps, TS_COOKIE, -1, "TS", comm, TSDestroy, TSView);
  PetscLogObjectCreate(t);
  PetscLogObjectMemory(t, sizeof(struct _p_TS));
  ierr = PetscMemzero(t->ops, sizeof(struct _TSOps));                                                     CHKERRQ(ierr);
  t->bops->publish    = TSPublish_Petsc;
  t->type_name        = PETSC_NULL;
  t->serialize_name   = PETSC_NULL;

  t->ops->applymatrixbc = TSDefaultSystemMatrixBC;
  t->ops->applyrhsbc    = TSDefaultRhsBC;
  t->ops->applysolbc    = TSDefaultSolutionBC;
  t->ops->prestep       = TSDefaultPreStep;
  t->ops->update        = TSDefaultUpdate;
  t->ops->poststep      = TSDefaultPostStep;

  /* General TS description */
  t->problem_type       = TS_LINEAR;
  t->vec_sol            = PETSC_NULL;
  t->vec_sol_always     = PETSC_NULL;
  t->numbermonitors     = 0;
  t->isGTS              = PETSC_FALSE;
  t->isExplicit         = PETSC_NULL;
  t->Iindex             = PETSC_NULL;
  t->sles               = PETSC_NULL;
  t->A                  = PETSC_NULL;
  t->B                  = PETSC_NULL;
  t->snes               = PETSC_NULL;
  t->funP               = PETSC_NULL;
  t->jacP               = PETSC_NULL;
  t->setupcalled        = 0;
  t->data               = PETSC_NULL;
  t->user               = PETSC_NULL;
  t->max_steps          = 5000;
  t->max_time           = 5.0;
  t->time_step          = .1;
  t->time_step_old      = t->time_step;
  t->initial_time_step  = t->time_step;
  t->steps              = 0;
  t->ptime              = 0.0;
  t->linear_its         = 0;
  t->nonlinear_its      = 0;
  t->work               = PETSC_NULL;
  t->nwork              = 0;

  *ts = t;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "TSSerialize"
/*@ 
  TSSerialize - This function stores or recreates a timestepper using a viewer for a binary file.

  Collective on MPI_Comm

  Input Parameters:
+ comm   - The communicator for the ts object
. viewer - The viewer context
- store  - This flag is PETSC_TRUE is data is being written, otherwise it will be read

  Output Parameter:
. ts     - The ts

  Level: beginner

.keywords: TS, serialize
.seealso: PartitionSerialize(), TSSerialize()
@*/
int TSSerialize(MPI_Comm comm, TS *ts, PetscViewer viewer, PetscTruth store)
{
  int      (*serialize)(MPI_Comm, TS *, PetscViewer, PetscTruth);
  int        fd, len;
  char      *name;
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(ts);

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &match);                             CHKERRQ(ierr);
  if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Must be binary viewer");
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);

  if (!TSSerializeRegisterAllCalled) {
    ierr = TSSerializeRegisterAll(PETSC_NULL);                                                            CHKERRQ(ierr);
  }
  if (!TSSerializeList) SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not find table of methods");

  if (store) {
    PetscValidHeaderSpecific(*ts, TS_COOKIE);
    ierr = PetscStrlen((*ts)->class_name, &len);                                                          CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                   1,   PETSC_INT,  0);                              CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*ts)->class_name,     len, PETSC_CHAR, 0);                              CHKERRQ(ierr);
    ierr = PetscStrlen((*ts)->serialize_name, &len);                                                      CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                   1,   PETSC_INT,  0);                              CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*ts)->serialize_name, len, PETSC_CHAR, 0);                              CHKERRQ(ierr);
    ierr = PetscFListFind(comm, TSSerializeList, (*ts)->serialize_name, (void (**)(void)) &serialize);    CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, ts, viewer, store);                                                         CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "TS", &match);                                                               CHKERRQ(ierr);
    ierr = PetscFree(name);                                                                               CHKERRQ(ierr);
    if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Non-ts object");
    /* Dispatch to the correct routine */
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscFListFind(comm, TSSerializeList, name, (void (**)(void)) &serialize);                     CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, ts, viewer, store);                                                         CHKERRQ(ierr);
    ierr = PetscStrfree((*ts)->serialize_name);                                                           CHKERRQ(ierr);
    (*ts)->serialize_name = name;
  }

  PetscFunctionReturn(0);
}
