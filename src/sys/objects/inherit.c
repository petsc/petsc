/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/
#include <petscviewer.h>

#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscObject *PetscObjects;
PETSC_INTERN PetscInt     PetscObjectsCounts;
PETSC_INTERN PetscInt     PetscObjectsMaxCounts;
PETSC_INTERN PetscBool    PetscObjectsLog;
#endif

#if defined(PETSC_USE_LOG)
PetscObject *PetscObjects       = NULL;
PetscInt     PetscObjectsCounts = 0, PetscObjectsMaxCounts = 0;
PetscBool    PetscObjectsLog = PETSC_FALSE;
#endif

PETSC_EXTERN PetscErrorCode PetscObjectCompose_Petsc(PetscObject, const char[], PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectQuery_Petsc(PetscObject, const char[], PetscObject *);
PETSC_EXTERN PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject, const char[], void (*)(void));
PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject, const char[], void (**)(void));

PetscObjectId PetscObjectNewId_Internal(void)
{
  static PetscObjectId idcnt = 1;
  return idcnt++;
}

/*
   PetscHeaderCreate_Private - Creates a base PETSc object header and fills
   in the default values.  Called by the macro PetscHeaderCreate().
*/
PetscErrorCode PetscHeaderCreate_Private(PetscObject h, PetscClassId classid, const char class_name[], const char descr[], const char mansec[], MPI_Comm comm, PetscObjectDestroyFunction destroy, PetscObjectViewFunction view)
{
  void       *get_tmp;
  PetscInt64 *cidx;
  PetscMPIInt flg;

  PetscFunctionBegin;
  h->classid               = classid;
  h->class_name            = (char *)class_name;
  h->description           = (char *)descr;
  h->mansec                = (char *)mansec;
  h->refct                 = 1;
  h->non_cyclic_references = NULL;
  h->id                    = PetscObjectNewId_Internal();
  h->bops->destroy         = destroy;
  h->bops->view            = view;
  h->bops->compose         = PetscObjectCompose_Petsc;
  h->bops->query           = PetscObjectQuery_Petsc;
  h->bops->composefunction = PetscObjectComposeFunction_Petsc;
  h->bops->queryfunction   = PetscObjectQueryFunction_Petsc;

  PetscCall(PetscCommDuplicate(comm, &h->comm, &h->tag));

  /* Increment and store current object creation index */
  PetscCallMPI(MPI_Comm_get_attr(h->comm, Petsc_CreationIdx_keyval, &get_tmp, &flg));
  if (flg) {
    cidx    = (PetscInt64 *)get_tmp;
    h->cidx = (*cidx)++;
    PetscCallMPI(MPI_Comm_set_attr(h->comm, Petsc_CreationIdx_keyval, cidx));
  } else SETERRQ(h->comm, PETSC_ERR_ARG_CORRUPT, "MPI_Comm does not have an object creation index");

#if defined(PETSC_USE_LOG)
  /* Keep a record of object created */
  if (PetscObjectsLog) {
    PetscObject *newPetscObjects;
    PetscInt     newPetscObjectsMaxCounts;

    PetscObjectsCounts++;
    for (PetscInt i = 0; i < PetscObjectsMaxCounts; ++i) {
      if (!PetscObjects[i]) {
        PetscObjects[i] = h;
        PetscFunctionReturn(0);
      }
    }
    /* Need to increase the space for storing PETSc objects */
    if (!PetscObjectsMaxCounts) newPetscObjectsMaxCounts = 100;
    else newPetscObjectsMaxCounts = 2 * PetscObjectsMaxCounts;
    PetscCall(PetscCalloc1(newPetscObjectsMaxCounts, &newPetscObjects));
    PetscCall(PetscArraycpy(newPetscObjects, PetscObjects, PetscObjectsMaxCounts));
    PetscCall(PetscFree(PetscObjects));

    PetscObjects                        = newPetscObjects;
    PetscObjects[PetscObjectsMaxCounts] = h;
    PetscObjectsMaxCounts               = newPetscObjectsMaxCounts;
  }
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscBool      PetscMemoryCollectMaximumUsage;
PETSC_INTERN PetscLogDouble PetscMemoryMaximumUsage;

/*
    PetscHeaderDestroy_Private - Destroys a base PETSc object header. Called by
    the macro PetscHeaderDestroy().
*/
PetscErrorCode PetscHeaderDestroy_Private(PetscObject obj, PetscBool clear_for_reuse)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscLogObjectDestroy(obj));
  PetscCall(PetscComposedQuantitiesDestroy(obj));
  if (PetscMemoryCollectMaximumUsage) {
    PetscLogDouble usage;

    PetscCall(PetscMemoryGetCurrentUsage(&usage));
    if (usage > PetscMemoryMaximumUsage) PetscMemoryMaximumUsage = usage;
  }
  /* first destroy things that could execute arbitrary code */
  if (obj->python_destroy) {
    void *python_context                     = obj->python_context;
    PetscErrorCode (*python_destroy)(void *) = obj->python_destroy;

    obj->python_context = NULL;
    obj->python_destroy = NULL;
    PetscCall((*python_destroy)(python_context));
  }
  PetscCall(PetscObjectDestroyOptionsHandlers(obj));
  PetscCall(PetscObjectListDestroy(&obj->olist));

  /* destroy allocated quantities */
  if (PetscPrintFunctionList) PetscCall(PetscFunctionListPrintNonEmpty(obj->qlist));
  PetscCheck(--(obj->refct) <= 0, obj->comm, PETSC_ERR_PLIB, "Destroying a PetscObject (%s) with reference count %" PetscInt_FMT " >= 1", obj->name ? obj->name : "unnamed", obj->refct);
  PetscCall(PetscFree(obj->name));
  PetscCall(PetscFree(obj->prefix));
  PetscCall(PetscFree(obj->type_name));

  if (clear_for_reuse) {
    /* we will assume that obj->bops->view and destroy are safe to leave as-is */
    obj->bops->compose         = PetscObjectCompose_Petsc;
    obj->bops->query           = PetscObjectQuery_Petsc;
    obj->bops->composefunction = PetscObjectComposeFunction_Petsc;
    obj->bops->queryfunction   = PetscObjectQueryFunction_Petsc;

    /* reset quantities, in order of appearance in _p_PetscObject */
    obj->id       = PetscObjectNewId_Internal();
    obj->refct    = 1;
    obj->tablevel = 0;
    obj->state    = 0;
    /* don't deallocate, zero these out instead */
    PetscCall(PetscFunctionListClear(obj->qlist));
    PetscCall(PetscArrayzero(obj->fortran_func_pointers, obj->num_fortran_func_pointers));
    PetscCall(PetscArrayzero(obj->fortrancallback[PETSC_FORTRAN_CALLBACK_CLASS], obj->num_fortrancallback[PETSC_FORTRAN_CALLBACK_CLASS]));
    PetscCall(PetscArrayzero(obj->fortrancallback[PETSC_FORTRAN_CALLBACK_SUBTYPE], obj->num_fortrancallback[PETSC_FORTRAN_CALLBACK_SUBTYPE]));
    obj->optionsprinted = PETSC_FALSE;
#if PetscDefined(HAVE_SAWS)
    obj->amsmem          = PETSC_FALSE;
    obj->amspublishblock = PETSC_FALSE;
#endif
    obj->options                                  = NULL;
    obj->donotPetscObjectPrintClassNamePrefixType = PETSC_FALSE;
  } else {
    PetscCall(PetscFunctionListDestroy(&obj->qlist));
    PetscCall(PetscFree(obj->fortran_func_pointers));
    PetscCall(PetscFree(obj->fortrancallback[PETSC_FORTRAN_CALLBACK_CLASS]));
    PetscCall(PetscFree(obj->fortrancallback[PETSC_FORTRAN_CALLBACK_SUBTYPE]));
    PetscCall(PetscCommDestroy(&obj->comm));
    obj->classid = PETSCFREEDHEADER;

#if PetscDefined(USE_LOG)
    if (PetscObjectsLog) {
      /* Record object removal from list of all objects */
      for (PetscInt i = 0; i < PetscObjectsMaxCounts; ++i) {
        if (PetscObjects[i] == obj) {
          PetscObjects[i] = NULL;
          --PetscObjectsCounts;
          break;
        }
      }
      if (!PetscObjectsCounts) {
        PetscCall(PetscFree(PetscObjects));
        PetscObjectsMaxCounts = 0;
      }
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*
  PetscHeaderReset_Internal - "Reset" a PetscObject header. This is tantamount to destroying
  the object but does not free all resources. The object retains its:

  - classid
  - bops->view
  - bops->destroy
  - comm
  - tag
  - class_name
  - description
  - mansec
  - cpp

  Note that while subclass information is lost, superclass info remains. Thus this function is
  intended to be used to reuse a PetscObject within the same class to avoid reallocating its
  resources.
*/
PetscErrorCode PetscHeaderReset_Internal(PetscObject obj)
{
  PetscFunctionBegin;
  PetscCall(PetscHeaderDestroy_Private(obj, PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectCopyFortranFunctionPointers - Copy function pointers to another object

   Logically Collective on src

   Input Parameters:
+  src - source object
-  dest - destination object

   Level: developer

   Note:
   Both objects must have the same class.

   This is used to help manage user callback functions that were provided in Fortran
@*/
PetscErrorCode PetscObjectCopyFortranFunctionPointers(PetscObject src, PetscObject dest)
{
  PetscFortranCallbackId cbtype, numcb[PETSC_FORTRAN_CALLBACK_MAXTYPE];

  PetscFunctionBegin;
  PetscValidHeader(src, 1);
  PetscValidHeader(dest, 2);
  PetscCheck(src->classid == dest->classid, src->comm, PETSC_ERR_ARG_INCOMP, "Objects must be of the same class");

  PetscCall(PetscFree(dest->fortran_func_pointers));
  PetscCall(PetscMalloc(src->num_fortran_func_pointers * sizeof(void (*)(void)), &dest->fortran_func_pointers));
  PetscCall(PetscMemcpy(dest->fortran_func_pointers, src->fortran_func_pointers, src->num_fortran_func_pointers * sizeof(void (*)(void))));

  dest->num_fortran_func_pointers = src->num_fortran_func_pointers;

  PetscCall(PetscFortranCallbackGetSizes(src->classid, &numcb[PETSC_FORTRAN_CALLBACK_CLASS], &numcb[PETSC_FORTRAN_CALLBACK_SUBTYPE]));
  for (cbtype = PETSC_FORTRAN_CALLBACK_CLASS; cbtype < PETSC_FORTRAN_CALLBACK_MAXTYPE; cbtype++) {
    PetscCall(PetscFree(dest->fortrancallback[cbtype]));
    PetscCall(PetscCalloc1(numcb[cbtype], &dest->fortrancallback[cbtype]));
    PetscCall(PetscMemcpy(dest->fortrancallback[cbtype], src->fortrancallback[cbtype], src->num_fortrancallback[cbtype] * sizeof(PetscFortranCallback)));
    dest->num_fortrancallback[cbtype] = src->num_fortrancallback[cbtype];
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSetFortranCallback - set fortran callback function pointer and context

   Logically Collective

   Input Parameters:
+  obj - object on which to set callback
.  cbtype - callback type (class or subtype)
.  cid - address of callback Id, updated if not yet initialized (zero)
.  func - Fortran function
-  ctx - Fortran context

   Level: developer

   Note:
   This is used to help manage user callback functions that were provided in Fortran

.seealso: `PetscObjectGetFortranCallback()`
@*/
PetscErrorCode PetscObjectSetFortranCallback(PetscObject obj, PetscFortranCallbackType cbtype, PetscFortranCallbackId *cid, void (*func)(void), void *ctx)
{
  const char *subtype = NULL;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (cbtype == PETSC_FORTRAN_CALLBACK_SUBTYPE) subtype = obj->type_name;
  if (!*cid) PetscCall(PetscFortranCallbackRegister(obj->classid, subtype, cid));
  if (*cid >= PETSC_SMALLEST_FORTRAN_CALLBACK + obj->num_fortrancallback[cbtype]) {
    PetscFortranCallbackId oldnum = obj->num_fortrancallback[cbtype];
    PetscFortranCallbackId newnum = PetscMax(*cid - PETSC_SMALLEST_FORTRAN_CALLBACK + 1, 2 * oldnum);
    PetscFortranCallback  *callback;
    PetscCall(PetscMalloc1(newnum, &callback));
    PetscCall(PetscMemcpy(callback, obj->fortrancallback[cbtype], oldnum * sizeof(*obj->fortrancallback[cbtype])));
    PetscCall(PetscFree(obj->fortrancallback[cbtype]));

    obj->fortrancallback[cbtype]     = callback;
    obj->num_fortrancallback[cbtype] = newnum;
  }
  obj->fortrancallback[cbtype][*cid - PETSC_SMALLEST_FORTRAN_CALLBACK].func = func;
  obj->fortrancallback[cbtype][*cid - PETSC_SMALLEST_FORTRAN_CALLBACK].ctx  = ctx;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectGetFortranCallback - get fortran callback function pointer and context

   Logically Collective

   Input Parameters:
+  obj - object on which to get callback
.  cbtype - callback type
-  cid - address of callback Id

   Output Parameters:
+  func - Fortran function (or NULL if not needed)
-  ctx - Fortran context (or NULL if not needed)

   Level: developer

   Note:
   This is used to help manage user callback functions that were provided in Fortran

.seealso: `PetscObjectSetFortranCallback()`
@*/
PetscErrorCode PetscObjectGetFortranCallback(PetscObject obj, PetscFortranCallbackType cbtype, PetscFortranCallbackId cid, void (**func)(void), void **ctx)
{
  PetscFortranCallback *cb;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCheck(cid >= PETSC_SMALLEST_FORTRAN_CALLBACK, obj->comm, PETSC_ERR_ARG_CORRUPT, "Fortran callback Id invalid");
  PetscCheck(cid < PETSC_SMALLEST_FORTRAN_CALLBACK + obj->num_fortrancallback[cbtype], obj->comm, PETSC_ERR_ARG_CORRUPT, "Fortran callback not set on this object");
  cb = &obj->fortrancallback[cbtype][cid - PETSC_SMALLEST_FORTRAN_CALLBACK];
  if (func) *func = cb->func;
  if (ctx) *ctx = cb->ctx;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_LOG)
/*@C
   PetscObjectsDump - Prints all the currently existing objects.

   On rank 0 of `PETSC_COMM_WORLD` prints the values

   Input Parameters:
+  fd - file pointer
-  all - by default only tries to display objects created explicitly by the user, if all is `PETSC_TRUE` then lists all outstanding objects

   Options Database Key:
.  -objects_dump <all> - print information about all the objects that exist at the end of the programs run

   Level: advanced

@*/
PetscErrorCode PetscObjectsDump(FILE *fd, PetscBool all)
{
  PetscInt i;
  #if defined(PETSC_USE_DEBUG)
  PetscInt j, k = 0;
  #endif
  PetscObject h;

  PetscFunctionBegin;
  if (PetscObjectsCounts) {
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "The following objects were never freed\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "-----------------------------------------\n"));
    for (i = 0; i < PetscObjectsMaxCounts; i++) {
      if ((h = PetscObjects[i])) {
        PetscCall(PetscObjectName(h));
        {
  #if defined(PETSC_USE_DEBUG)
          PetscStack *stack = NULL;
          char       *create, *rclass;

          /* if the PETSc function the user calls is not a create then this object was NOT directly created by them */
          PetscCall(PetscMallocGetStack(h, &stack));
          if (stack) {
            k = stack->currentsize - 2;
            if (!all) {
              k = 0;
              while (!stack->petscroutine[k]) k++;
              PetscCall(PetscStrstr(stack->function[k], "Create", &create));
              if (!create) PetscCall(PetscStrstr(stack->function[k], "Get", &create));
              PetscCall(PetscStrstr(stack->function[k], h->class_name, &rclass));
              if (!create) continue;
              if (!rclass) continue;
            }
          }
  #endif

          PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "[%d] %s %s %s\n", PetscGlobalRank, h->class_name, h->type_name, h->name));

  #if defined(PETSC_USE_DEBUG)
          PetscCall(PetscMallocGetStack(h, &stack));
          if (stack) {
            for (j = k; j >= 0; j--) fprintf(fd, "      [%d]  %s() in %s\n", PetscGlobalRank, stack->function[j], stack->file[j]);
          }
  #endif
        }
      }
    }
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_USE_LOG)

/*@C
   PetscObjectsView - Prints the currently existing objects.

   Logically Collective on viewer

   Input Parameter:
.  viewer - must be an `PETSCVIEWERASCII` viewer

   Level: advanced

@*/
PetscErrorCode PetscObjectsView(PetscViewer viewer)
{
  PetscBool isascii;
  FILE     *fd;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Only supports ASCII viewer");
  PetscCall(PetscViewerASCIIGetPointer(viewer, &fd));
  PetscCall(PetscObjectsDump(fd, PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectsGetObject - Get a pointer to a named object

   Not collective

   Input Parameter:
.  name - the name of an object

   Output Parameters:
+  obj - the object or null if there is no object
-  classname - the name of the class

   Level: advanced

@*/
PetscErrorCode PetscObjectsGetObject(const char *name, PetscObject *obj, char **classname)
{
  PetscInt    i;
  PetscObject h;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 1);
  PetscValidPointer(obj, 2);
  *obj = NULL;
  for (i = 0; i < PetscObjectsMaxCounts; i++) {
    if ((h = PetscObjects[i])) {
      PetscCall(PetscObjectName(h));
      PetscCall(PetscStrcmp(h->name, name, &flg));
      if (flg) {
        *obj = h;
        if (classname) *classname = h->class_name;
        PetscFunctionReturn(0);
      }
    }
  }
  PetscFunctionReturn(0);
}
#endif

/*@
   PetscObjectSetPrintedOptions - indicate to an object that it should behave as if it has already printed the help for its options so it will not display the help message

   Input Parameters:
.  obj  - the `PetscObject`

   Level: developer

   Developer Note:
   This is used, for example to prevent sequential objects that are created from a parallel object; such as the `KSP` created by
   `PCBJACOBI` from all printing the same help messages to the screen

.seealso: `PetscOptionsInsert()`
@*/
PetscErrorCode PetscObjectSetPrintedOptions(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  obj->optionsprinted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectInheritPrintedOptions - If the child object is not on the rank 0 process of the parent object and the child is sequential then the child gets it set.

   Input Parameters:
+  pobj - the parent object
-  obj  - the PetscObject

   Level: developer

   Developer Notes:
   This is used, for example to prevent sequential objects that are created from a parallel object; such as the `KSP` created by
   `PCBJACOBI` from all printing the same help messages to the screen

   This will not handle more complicated situations like with `PCGASM` where children may live on any subset of the parent's processes and overlap

.seealso: `PetscOptionsInsert()`, `PetscObjectSetPrintedOptions()`
@*/
PetscErrorCode PetscObjectInheritPrintedOptions(PetscObject pobj, PetscObject obj)
{
  PetscMPIInt prank, size;

  PetscFunctionBegin;
  PetscValidHeader(pobj, 1);
  PetscValidHeader(obj, 2);
  PetscCallMPI(MPI_Comm_rank(pobj->comm, &prank));
  PetscCallMPI(MPI_Comm_size(obj->comm, &size));
  if (size == 1 && prank > 0) obj->optionsprinted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectAddOptionsHandler - Adds an additional function to check for options when XXXSetFromOptions() is called.

    Not Collective

    Input Parameters:
+   obj - the PETSc object
.   handle - function that checks for options
.   destroy - function to destroy context if provided
-   ctx - optional context for check function

    Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectProcessOptionsHandlers()`, `PetscObjectDestroyOptionsHandlers()`
@*/
PetscErrorCode PetscObjectAddOptionsHandler(PetscObject obj, PetscErrorCode (*handle)(PetscObject, PetscOptionItems *, void *), PetscErrorCode (*destroy)(PetscObject, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCheck(obj->noptionhandler < PETSC_MAX_OPTIONS_HANDLER, obj->comm, PETSC_ERR_ARG_OUTOFRANGE, "To many options handlers added");
  obj->optionhandler[obj->noptionhandler] = handle;
  obj->optiondestroy[obj->noptionhandler] = destroy;
  obj->optionctx[obj->noptionhandler++]   = ctx;
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectProcessOptionsHandlers - Calls all the options handlers attached to an object

    Not Collective

    Input Parameters:
+   obj - the PETSc object
-   PetscOptionsObject - the options context

    Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectAddOptionsHandler()`, `PetscObjectDestroyOptionsHandlers()`
@*/
PetscErrorCode PetscObjectProcessOptionsHandlers(PetscObject obj, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  for (PetscInt i = 0; i < obj->noptionhandler; i++) PetscCall((*obj->optionhandler[i])(obj, PetscOptionsObject, obj->optionctx[i]));
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectDestroyOptionsHandlers - Destroys all the option handlers attached to an object

    Not Collective

    Input Parameter:
.   obj - the PETSc object

    Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectAddOptionsHandler()`, `PetscObjectProcessOptionsHandlers()`
@*/
PetscErrorCode PetscObjectDestroyOptionsHandlers(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  for (PetscInt i = 0; i < obj->noptionhandler; i++) {
    if (obj->optiondestroy[i]) PetscCall((*obj->optiondestroy[i])(obj, obj->optionctx[i]));
  }
  obj->noptionhandler = 0;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectReference - Indicates to any `PetscObject` that it is being
   referenced by another `PetscObject`. This increases the reference
   count for that object by one.

   Logically Collective on obj

   Input Parameter:
.  obj - the PETSc object. This must be cast with (`PetscObject`), for example,
         `PetscObjectReference`((`PetscObject`)mat);

   Level: advanced

.seealso: `PetscObjectCompose()`, `PetscObjectDereference()`
@*/
PetscErrorCode PetscObjectReference(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) PetscFunctionReturn(0);
  PetscValidHeader(obj, 1);
  obj->refct++;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectGetReference - Gets the current reference count for
   any PETSc object.

   Not Collective

   Input Parameter:
.  obj - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectGetReference`((`PetscObject`)mat,&cnt);

   Output Parameter:
.  cnt - the reference count

   Level: advanced

.seealso: `PetscObjectCompose()`, `PetscObjectDereference()`, `PetscObjectReference()`
@*/
PetscErrorCode PetscObjectGetReference(PetscObject obj, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidIntPointer(cnt, 2);
  *cnt = obj->refct;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectDereference - Indicates to any `PetscObject` that it is being
   referenced by one less `PetscObject`. This decreases the reference
   count for that object by one.

   Collective on obj if reference reaches 0 otherwise Logically Collective

   Input Parameter:
.  obj - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectDereference`((`PetscObject`)mat);

   Note:
    `PetscObjectDestroy()` sets the obj pointer to null after the call, this routine does not.

   Level: advanced

.seealso: `PetscObjectCompose()`, `PetscObjectReference()`, `PetscObjectDestroy()`
@*/
PetscErrorCode PetscObjectDereference(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) PetscFunctionReturn(0);
  PetscValidHeader(obj, 1);
  if (obj->bops->destroy) PetscCall((*obj->bops->destroy)(&obj));
  else PetscCheck(--(obj->refct), PETSC_COMM_SELF, PETSC_ERR_SUP, "This PETSc object does not have a generic destroy routine");
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */
/*
     The following routines are the versions private to the PETSc object
     data structures.
*/
PetscErrorCode PetscObjectRemoveReference(PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscObjectListRemoveReference(&obj->olist, name));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectCompose_Petsc(PetscObject obj, const char name[], PetscObject ptr)
{
  PetscFunctionBegin;
  if (ptr) {
    char     *tname;
    PetscBool skipreference;

    PetscCall(PetscObjectListReverseFind(ptr->olist, obj, &tname, &skipreference));
    if (tname) PetscCheck(skipreference, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "An object cannot be composed with an object that was composed with it");
  }
  PetscCall(PetscObjectListAdd(&obj->olist, name, ptr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectQuery_Petsc(PetscObject obj, const char name[], PetscObject *ptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscObjectListFind(obj->olist, name, ptr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject obj, const char name[], void (*ptr)(void))
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscFunctionListAdd(&obj->qlist, name, ptr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject obj, const char name[], void (**ptr)(void))
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscFunctionListFind(obj->qlist, name, ptr));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectCompose - Associates another PETSc object with a given PETSc object.

   Not Collective

   Input Parameters:
+  obj - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
.  name - name associated with the child object
-  ptr - the other PETSc object to associate with the PETSc object; this must also be
         cast with (`PetscObject`)

   Level: advanced

   Notes:
   The second objects reference count is automatically increased by one when it is
   composed.

   Replaces any previous object that had the same name.

   If ptr is null and name has previously been composed using an object, then that
   entry is removed from the obj.

   `PetscObjectCompose()` can be used with any PETSc object (such as
   `Mat`, `Vec`, `KSP`, `SNES`, etc.) or any user-provided object.

   `PetscContainerCreate()` can be used to create an object from a
   user-provided pointer that may then be composed with PETSc objects using `PetscObjectCompose()`

.seealso: `PetscObjectQuery()`, `PetscContainerCreate()`, `PetscObjectComposeFunction()`, `PetscObjectQueryFunction()`, `PetscContainer`,
          `PetscContainerSetPointer()`
@*/
PetscErrorCode PetscObjectCompose(PetscObject obj, const char name[], PetscObject ptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  if (ptr) PetscValidHeader(ptr, 3);
  PetscCheck(obj != ptr, PetscObjectComm((PetscObject)obj), PETSC_ERR_SUP, "Cannot compose object with itself");
  PetscCall((*obj->bops->compose)(obj, name, ptr));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectQuery  - Gets a PETSc object associated with a given object that was composed with `PetscObjectCompose()`

   Not Collective

   Input Parameters:
+  obj - the PETSc object
         Thus must be cast with a (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
.  name - name associated with child object
-  ptr - the other PETSc object associated with the PETSc object, this must be
         cast with (`PetscObject`*)

   Level: advanced

   Note:
   The reference count of neither object is increased in this call

.seealso: `PetscObjectCompose()`, `PetscObjectComposeFunction()`, `PetscObjectQueryFunction()`, `PetscContainer`
          `PetscContainerGetPointer()`
@*/
PetscErrorCode PetscObjectQuery(PetscObject obj, const char name[], PetscObject *ptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ptr, 3);
  PetscCall((*obj->bops->query)(obj, name, ptr));
  PetscFunctionReturn(0);
}

/*MC
   PetscObjectComposeFunction - Associates a function with a given PETSc object.

    Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscObjectComposeFunction(PetscObject obj,const char name[],void (*fptr)(void))

   Logically Collective on obj

   Input Parameters:
+  obj - the PETSc object; this must be cast with a (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
.  name - name associated with the child function
.  fname - name of the function
-  fptr - function pointer

   Level: advanced

   Notes:
   When the first argument of the function is the object within which it has been composed then `PetscTryMethod()` and `PetscUseMethod()`
   can be used to call the function directly with error checking.

   To remove a registered routine, pass in NULL for fptr().

   PetscObjectComposeFunction() can be used with any PETSc object (such as
   `Mat`, `Vec`, `KSP`, `SNES`, etc.) or any user-provided object.

   `PetscCallMethod()` is used to call a function that is stored in the objects obj->ops table.

.seealso: `PetscObjectQueryFunction()`, `PetscContainerCreate()` `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscTryMethod()`, `PetscUseMethod()`,
          `PetscCallMethod()`
M*/

PetscErrorCode PetscObjectComposeFunction_Private(PetscObject obj, const char name[], void (*fptr)(void))
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  PetscCall((*obj->bops->composefunction)(obj, name, fptr));
  PetscFunctionReturn(0);
}

/*MC
   PetscObjectQueryFunction - Gets a function associated with a given object.

    Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscObjectQueryFunction(PetscObject obj,const char name[],void (**fptr)(void))

   Logically Collective on obj

   Input Parameters:
+  obj - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectQueryFunction`((`PetscObject`)ksp,...);
-  name - name associated with the child function

   Output Parameter:
.  fptr - function pointer

   Level: advanced

.seealso: `PetscObjectComposeFunction()`, `PetscFunctionListFind()`, `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Private(PetscObject obj, const char name[], void (**ptr)(void))
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  PetscCall((*obj->bops->queryfunction)(obj, name, ptr));
  PetscFunctionReturn(0);
}

struct _p_PetscContainer {
  PETSCHEADER(int);
  void *ptr;
  PetscErrorCode (*userdestroy)(void *);
};

/*@C
   PetscContainerUserDestroyDefault - Default destroy routine for user-provided data that simply calls `PetscFree()` in the data
   provided with `PetscContainerSetPointer()`

   Logically Collective on the `PetscContainer` containing the user data

   Input Parameter:
.  ctx - pointer to user-provided data

   Level: advanced

.seealso: `PetscContainerDestroy()`, `PetscContainerSetUserDestroy()`
@*/
PetscErrorCode PetscContainerUserDestroyDefault(void *ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/*@C
   PetscContainerGetPointer - Gets the pointer value contained in the container that was provided with `PetscContainerSetPointer()`

   Not Collective

   Input Parameter:
.  obj - the object created with `PetscContainerCreate()`

   Output Parameter:
.  ptr - the pointer value

   Level: advanced

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`,
          `PetscContainerSetPointer()`
@*/
PetscErrorCode PetscContainerGetPointer(PetscContainer obj, void **ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  PetscValidPointer(ptr, 2);
  *ptr = obj->ptr;
  PetscFunctionReturn(0);
}

/*@C
   PetscContainerSetPointer - Sets the pointer value contained in the container.

   Logically Collective on obj

   Input Parameters:
+  obj - the object created with `PetscContainerCreate()`
-  ptr - the pointer value

   Level: advanced

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`, `PetscObjectCompose()`, `PetscObjectQuery()`,
          `PetscContainerGetPointer()`
@*/
PetscErrorCode PetscContainerSetPointer(PetscContainer obj, void *ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  if (ptr) PetscValidPointer(ptr, 2);
  obj->ptr = ptr;
  PetscFunctionReturn(0);
}

/*@C
   PetscContainerDestroy - Destroys a PETSc container object.

   Collective on obj

   Input Parameter:
.  obj - an object that was created with `PetscContainerCreate()`

   Level: advanced

   Note:
   If `PetscContainerSetUserDestroy()` was used to provide a user destroy object for the data provided with `PetscContainerSetPointer()`
   then that function is called to destroy the data.

.seealso: `PetscContainerCreate()`, `PetscContainerSetUserDestroy()`
@*/
PetscErrorCode PetscContainerDestroy(PetscContainer *obj)
{
  PetscFunctionBegin;
  if (!*obj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*obj, PETSC_CONTAINER_CLASSID, 1);
  if (--((PetscObject)(*obj))->refct > 0) {
    *obj = NULL;
    PetscFunctionReturn(0);
  }
  if ((*obj)->userdestroy) PetscCall((*(*obj)->userdestroy)((*obj)->ptr));
  PetscCall(PetscHeaderDestroy(obj));
  PetscFunctionReturn(0);
}

/*@C
   PetscContainerSetUserDestroy - Sets name of the user destroy function for the data provided to the `PetscContainer` with `PetscContainerSetPointer()`

   Logically Collective on obj

   Input Parameters:
+  obj - an object that was created with `PetscContainerCreate()`
-  des - name of the user destroy function

   Note:
   Use `PetscContainerUserDestroyDefault()` if the memory was obtained by calling `PetscMalloc()` or one of its variants for single memory allocation.

   Level: advanced

.seealso: `PetscContainerDestroy()`, `PetscContainerUserDestroyDefault()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc()`, `PetscCalloc1()`
@*/
PetscErrorCode PetscContainerSetUserDestroy(PetscContainer obj, PetscErrorCode (*des)(void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  obj->userdestroy = des;
  PetscFunctionReturn(0);
}

PetscClassId PETSC_CONTAINER_CLASSID;

/*@C
   PetscContainerCreate - Creates a PETSc object that has room to hold
   a single pointer. This allows one to attach any type of data (accessible
   through a pointer) with the `PetscObjectCompose()` function to a `PetscObject`.
   The data item itself is attached by a call to `PetscContainerSetPointer()`.

   Collective

   Input Parameters:
.  comm - MPI communicator that shares the object

   Output Parameters:
.  container - the container created

   Level: advanced

.seealso: `PetscContainerDestroy()`, `PetscContainerSetPointer()`, `PetscContainerGetPointer()`, `PetscObjectCompose()`, `PetscObjectQuery()`,
          `PetscContainerSetUserDestroy()`
@*/
PetscErrorCode PetscContainerCreate(MPI_Comm comm, PetscContainer *container)
{
  PetscFunctionBegin;
  PetscValidPointer(container, 2);
  PetscCall(PetscSysInitializePackage());
  PetscCall(PetscHeaderCreate(*container, PETSC_CONTAINER_CLASSID, "PetscContainer", "Container", "Sys", comm, PetscContainerDestroy, NULL));
  PetscFunctionReturn(0);
}

/*@
   PetscObjectSetFromOptions - Sets generic parameters from user options.

   Collective on obj

   Input Parameter:
.  obj - the `PetscObject`

   Note:
   We have no generic options at present, so this does nothing

   Level: beginner

.seealso: `PetscObjectSetOptionsPrefix()`, `PetscObjectGetOptionsPrefix()`
@*/
PetscErrorCode PetscObjectSetFromOptions(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscFunctionReturn(0);
}

/*@
   PetscObjectSetUp - Sets up the internal data structures for the later use.

   Collective on obj

   Input Parameters:
.  obj - the `PetscObject`

   Note:
   This does nothing at present.

   Level: advanced

.seealso: `PetscObjectDestroy()`
@*/
PetscErrorCode PetscObjectSetUp(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscFunctionReturn(0);
}
