/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/
#include <petscviewer.h>

PETSC_INTERN PetscObject *PetscObjects;
PETSC_INTERN PetscInt     PetscObjectsCounts;
PETSC_INTERN PetscInt     PetscObjectsMaxCounts;
PETSC_INTERN PetscBool    PetscObjectsLog;

PetscObject *PetscObjects       = NULL;
PetscInt     PetscObjectsCounts = 0, PetscObjectsMaxCounts = 0;
PetscBool    PetscObjectsLog = PETSC_FALSE;

PetscObjectId PetscObjectNewId_Internal(void)
{
  static PetscObjectId idcnt = 1;
  return idcnt++;
}

PetscErrorCode PetscHeaderCreate_Function(PetscErrorCode ierr, PetscObject *h, PetscClassId classid, const char class_name[], const char descr[], const char mansec[], MPI_Comm comm, PetscObjectDestroyFn *destroy, PetscObjectViewFn *view)
{
  PetscFunctionBegin;
  if (ierr) PetscFunctionReturn(ierr);
  PetscCall(PetscHeaderCreate_Private(*h, classid, class_name, descr, mansec, comm, destroy, view));
  PetscCall(PetscLogObjectCreate(*h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PetscHeaderCreate_Private - Fills in the default values.
*/
PetscErrorCode PetscHeaderCreate_Private(PetscObject h, PetscClassId classid, const char class_name[], const char descr[], const char mansec[], MPI_Comm comm, PetscObjectDestroyFn *destroy, PetscObjectViewFn *view)
{
  void       *get_tmp;
  PetscInt64 *cidx;
  PetscMPIInt iflg;

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

  PetscCall(PetscCommDuplicate(comm, &h->comm, &h->tag));

  /* Increment and store current object creation index */
  PetscCallMPI(MPI_Comm_get_attr(h->comm, Petsc_CreationIdx_keyval, &get_tmp, &iflg));
  PetscCheck(iflg, h->comm, PETSC_ERR_ARG_CORRUPT, "MPI_Comm does not have an object creation index");
  cidx    = (PetscInt64 *)get_tmp;
  h->cidx = (*cidx)++;

  /* Keep a record of object created */
  if (PetscDefined(USE_LOG) && PetscObjectsLog) {
    PetscObject *newPetscObjects;
    PetscInt     newPetscObjectsMaxCounts;

    PetscObjectsCounts++;
    for (PetscInt i = 0; i < PetscObjectsMaxCounts; ++i) {
      if (!PetscObjects[i]) {
        PetscObjects[i] = h;
        PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscBool      PetscMemoryCollectMaximumUsage;
PETSC_INTERN PetscLogDouble PetscMemoryMaximumUsage;

PetscErrorCode PetscHeaderDestroy_Function(PetscObject *h)
{
  PetscFunctionBegin;
  PetscCall(PetscLogObjectDestroy(*h));
  PetscCall(PetscHeaderDestroy_Private(*h, PETSC_FALSE));
  PetscCall(PetscFree(*h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PetscHeaderDestroy_Private - Destroys a base PETSc object header. Called by
    the macro PetscHeaderDestroy().
*/
PetscErrorCode PetscHeaderDestroy_Private(PetscObject obj, PetscBool clear_for_reuse)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
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

    if (PetscDefined(USE_LOG) && PetscObjectsLog) {
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
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectCopyFortranFunctionPointers - Copy function pointers to another object

  Logically Collective

  Input Parameters:
+ src  - source object
- dest - destination object

  Level: developer

  Note:
  Both objects must have the same class.

  This is used to help manage user callback functions that were provided in Fortran

.seealso: `PetscFortranCallbackRegister()`, `PetscFortranCallbackGetSizes()`
@*/
PetscErrorCode PetscObjectCopyFortranFunctionPointers(PetscObject src, PetscObject dest)
{
  PetscFortranCallbackId cbtype, numcb[PETSC_FORTRAN_CALLBACK_MAXTYPE];

  PetscFunctionBegin;
  PetscValidHeader(src, 1);
  PetscValidHeader(dest, 2);
  PetscCheck(src->classid == dest->classid, src->comm, PETSC_ERR_ARG_INCOMP, "Objects must be of the same class");

  PetscCall(PetscFree(dest->fortran_func_pointers));
  PetscCall(PetscMalloc(src->num_fortran_func_pointers * sizeof(PetscFortranCallbackFn *), &dest->fortran_func_pointers));
  PetscCall(PetscMemcpy(dest->fortran_func_pointers, src->fortran_func_pointers, src->num_fortran_func_pointers * sizeof(PetscFortranCallbackFn *)));

  dest->num_fortran_func_pointers = src->num_fortran_func_pointers;

  PetscCall(PetscFortranCallbackGetSizes(src->classid, &numcb[PETSC_FORTRAN_CALLBACK_CLASS], &numcb[PETSC_FORTRAN_CALLBACK_SUBTYPE]));
  for (cbtype = PETSC_FORTRAN_CALLBACK_CLASS; cbtype < PETSC_FORTRAN_CALLBACK_MAXTYPE; cbtype++) {
    PetscCall(PetscFree(dest->fortrancallback[cbtype]));
    PetscCall(PetscCalloc1(numcb[cbtype], &dest->fortrancallback[cbtype]));
    PetscCall(PetscMemcpy(dest->fortrancallback[cbtype], src->fortrancallback[cbtype], src->num_fortrancallback[cbtype] * sizeof(PetscFortranCallback)));
    dest->num_fortrancallback[cbtype] = src->num_fortrancallback[cbtype];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectSetFortranCallback - set Fortran callback function pointer and context

  Logically Collective, No Fortran Support

  Input Parameters:
+ obj    - object on which to set callback
. cbtype - callback type (class or subtype)
. cid    - address of callback Id, updated if not yet initialized (zero)
. func   - Fortran function
- ctx    - Fortran context

  Level: developer

  Note:
  This is used to help manage user callback functions that were provided in Fortran

.seealso: `PetscObjectGetFortranCallback()`, `PetscFortranCallbackRegister()`, `PetscFortranCallbackGetSizes()`
@*/
PetscErrorCode PetscObjectSetFortranCallback(PetscObject obj, PetscFortranCallbackType cbtype, PetscFortranCallbackId *cid, PetscFortranCallbackFn *func, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectGetFortranCallback - get Fortran callback function pointer and context

  Logically Collective, No Fortran Support

  Input Parameters:
+ obj    - object on which to get callback
. cbtype - callback type
- cid    - address of callback Id

  Output Parameters:
+ func - Fortran function (or `NULL` if not needed)
- ctx  - Fortran context (or `NULL` if not needed)

  Level: developer

  Note:
  This is used to help manage user callback functions that were provided in Fortran

.seealso: `PetscObjectSetFortranCallback()`, `PetscFortranCallbackRegister()`, `PetscFortranCallbackGetSizes()`
@*/
PetscErrorCode PetscObjectGetFortranCallback(PetscObject obj, PetscFortranCallbackType cbtype, PetscFortranCallbackId cid, PetscFortranCallbackFn **func, void **ctx)
{
  PetscFortranCallback *cb;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCheck(cid >= PETSC_SMALLEST_FORTRAN_CALLBACK, obj->comm, PETSC_ERR_ARG_CORRUPT, "Fortran callback Id invalid");
  PetscCheck(cid < PETSC_SMALLEST_FORTRAN_CALLBACK + obj->num_fortrancallback[cbtype], obj->comm, PETSC_ERR_ARG_CORRUPT, "Fortran callback not set on this object");
  cb = &obj->fortrancallback[cbtype][cid - PETSC_SMALLEST_FORTRAN_CALLBACK];
  if (func) *func = cb->func;
  if (ctx) *ctx = cb->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_LOG)
/*@C
  PetscObjectsDump - Prints all the currently existing objects.

  Input Parameters:
+ fd  - file pointer
- all - by default only tries to display objects created explicitly by the user, if all is `PETSC_TRUE` then lists all outstanding objects

  Options Database Key:
. -objects_dump <all> - print information about all the objects that exist at the end of the programs run

  Level: advanced

  Note:
  Only MPI rank 0 of `PETSC_COMM_WORLD` prints the values

.seealso: `PetscObject`
@*/
PetscErrorCode PetscObjectsDump(FILE *fd, PetscBool all)
{
  PetscInt    i, j, k = 0;
  PetscObject h;

  PetscFunctionBegin;
  if (PetscObjectsCounts) {
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "The following objects were never freed\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "-----------------------------------------\n"));
    for (i = 0; i < PetscObjectsMaxCounts; i++) {
      if ((h = PetscObjects[i])) {
        PetscCall(PetscObjectName(h));
        {
          PetscStack *stack  = NULL;
          char       *create = NULL, *rclass = NULL;

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

          PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "[%d] %s %s %s\n", PetscGlobalRank, h->class_name, h->type_name, h->name));

          PetscCall(PetscMallocGetStack(h, &stack));
          if (stack) {
            for (j = k; j >= 0; j--) fprintf(fd, "      [%d]  %s() in %s\n", PetscGlobalRank, stack->function[j], stack->file[j]);
          }
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectsView - Prints the currently existing objects.

  Logically Collective

  Input Parameter:
. viewer - must be an `PETSCVIEWERASCII` viewer

  Level: advanced

.seealso: `PetscObject`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectsGetObject - Get a pointer to a named object

  Not Collective

  Input Parameter:
. name - the name of an object

  Output Parameters:
+ obj       - the object or `NULL` if there is no object, optional, pass in `NULL` if not needed
- classname - the name of the class of the object, optional, pass in `NULL` if not needed

  Level: advanced

.seealso: `PetscObject`
@*/
PetscErrorCode PetscObjectsGetObject(const char name[], PetscObject *obj, const char *classname[])
{
  PetscInt    i;
  PetscObject h;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscAssertPointer(name, 1);
  if (obj) *obj = NULL;
  for (i = 0; i < PetscObjectsMaxCounts; i++) {
    if ((h = PetscObjects[i])) {
      PetscCall(PetscObjectName(h));
      PetscCall(PetscStrcmp(h->name, name, &flg));
      if (flg) {
        if (obj) *obj = h;
        if (classname) *classname = h->class_name;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
PetscErrorCode PetscObjectsView(PetscViewer viewer)
{
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectsGetObject(const char name[], PetscObject *obj, const char *classname[])
{
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
  PetscObjectSetPrintedOptions - indicate to an object that it should behave as if it has already printed the help for its options so it will not display the help message

  Input Parameter:
. obj - the `PetscObject`

  Level: developer

  Developer Notes:
  This is used, for example to prevent sequential objects that are created from a parallel object; such as the `KSP` created by
  `PCBJACOBI` from all printing the same help messages to the screen

.seealso: `PetscOptionsInsert()`, `PetscObject`
@*/
PetscErrorCode PetscObjectSetPrintedOptions(PetscObject obj)
{
  PetscFunctionBegin;
  PetscAssertPointer(obj, 1);
  obj->optionsprinted = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectInheritPrintedOptions - If the child object is not on the MPI rank 0 process of the parent object and the child is sequential then the child gets it set.

  Input Parameters:
+ pobj - the parent object
- obj  - the `PetscObject`

  Level: developer

  Developer Notes:
  This is used, for example to prevent sequential objects that are created from a parallel object; such as the `KSP` created by
  `PCBJACOBI` from all printing the same help messages to the screen

  This will not handle more complicated situations like with `PCGASM` where children may live on any subset of the parent's processes and overlap

.seealso: `PetscOptionsInsert()`, `PetscObjectSetPrintedOptions()`, `PetscObject`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectAddOptionsHandler - Adds an additional function to check for options when `XXXSetFromOptions()` is called.

  Not Collective

  Input Parameters:
+ obj     - the PETSc object
. handle  - function that checks for options
. destroy - function to destroy `ctx` if provided
- ctx     - optional context for check function

  Calling sequence of `handle`:
+ obj                - the PETSc object
. PetscOptionsObject - the `PetscOptionItems` object
- ctx                - optional context for `handle`

  Calling sequence of `destroy`:
+ obj - the PETSc object
- ctx - optional context for `handle`

  Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectProcessOptionsHandlers()`, `PetscObjectDestroyOptionsHandlers()`,
          `PetscObject`
@*/
PetscErrorCode PetscObjectAddOptionsHandler(PetscObject obj, PetscErrorCode (*handle)(PetscObject obj, PetscOptionItems PetscOptionsObject, void *ctx), PetscErrorCode (*destroy)(PetscObject obj, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  for (PetscInt i = 0; i < obj->noptionhandler; i++) {
    PetscBool identical = (PetscBool)(obj->optionhandler[i] == handle && obj->optiondestroy[i] == destroy && obj->optionctx[i] == ctx);
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(obj->noptionhandler < PETSC_MAX_OPTIONS_HANDLER, obj->comm, PETSC_ERR_ARG_OUTOFRANGE, "Too many options handlers added");
  obj->optionhandler[obj->noptionhandler] = handle;
  obj->optiondestroy[obj->noptionhandler] = destroy;
  obj->optionctx[obj->noptionhandler++]   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectProcessOptionsHandlers - Calls all the options handlers attached to an object

  Not Collective

  Input Parameters:
+ obj                - the PETSc object
- PetscOptionsObject - the options context

  Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectAddOptionsHandler()`, `PetscObjectDestroyOptionsHandlers()`,
          `PetscObject`
@*/
PetscErrorCode PetscObjectProcessOptionsHandlers(PetscObject obj, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  for (PetscInt i = 0; i < obj->noptionhandler; i++) PetscCall((*obj->optionhandler[i])(obj, PetscOptionsObject, obj->optionctx[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectDestroyOptionsHandlers - Destroys all the option handlers attached to an object

  Not Collective

  Input Parameter:
. obj - the PETSc object

  Level: developer

.seealso: `KSPSetFromOptions()`, `PCSetFromOptions()`, `SNESSetFromOptions()`, `PetscObjectAddOptionsHandler()`, `PetscObjectProcessOptionsHandlers()`,
          `PetscObject`
@*/
PetscErrorCode PetscObjectDestroyOptionsHandlers(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  for (PetscInt i = 0; i < obj->noptionhandler; i++) {
    if (obj->optiondestroy[i]) PetscCall((*obj->optiondestroy[i])(obj, obj->optionctx[i]));
  }
  obj->noptionhandler = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectReference - Indicates to a `PetscObject` that it is being
  referenced by another `PetscObject`. This increases the reference
  count for that object by one.

  Logically Collective

  Input Parameter:
. obj - the PETSc object. This must be cast with (`PetscObject`), for example, `PetscObjectReference`((`PetscObject`)mat);

  Level: advanced

  Note:
  If `obj` is `NULL` this function returns without doing anything.

.seealso: `PetscObjectCompose()`, `PetscObjectDereference()`, `PetscObject`
@*/
PetscErrorCode PetscObjectReference(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeader(obj, 1);
  obj->refct++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectGetReference - Gets the current reference count for a PETSc object.

  Not Collective

  Input Parameter:
. obj - the PETSc object; this must be cast with (`PetscObject`), for example,
        `PetscObjectGetReference`((`PetscObject`)mat,&cnt); `obj` cannot be `NULL`

  Output Parameter:
. cnt - the reference count

  Level: advanced

.seealso: `PetscObjectCompose()`, `PetscObjectDereference()`, `PetscObjectReference()`, `PetscObject`
@*/
PetscErrorCode PetscObjectGetReference(PetscObject obj, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(cnt, 2);
  *cnt = obj->refct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectDereference - Indicates to any `PetscObject` that it is being
  referenced by one less `PetscObject`. This decreases the reference
  count for that object by one.

  Collective on `obj` if reference reaches 0 otherwise Logically Collective

  Input Parameter:
. obj - the PETSc object; this must be cast with (`PetscObject`), for example,
        `PetscObjectDereference`((`PetscObject`)mat);

  Level: advanced

  Notes:
  `PetscObjectDestroy()` sets the `obj` pointer to `NULL` after the call, this routine does not.

  If `obj` is `NULL` this function returns without doing anything.

.seealso: `PetscObjectCompose()`, `PetscObjectReference()`, `PetscObjectDestroy()`, `PetscObject`
@*/
PetscErrorCode PetscObjectDereference(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeader(obj, 1);
  if (obj->bops->destroy) PetscCall((*obj->bops->destroy)(&obj));
  else PetscCheck(--(obj->refct), PETSC_COMM_SELF, PETSC_ERR_SUP, "This PETSc object does not have a generic destroy routine");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     The following routines are the versions private to the PETSc object
     data structures.
*/
PetscErrorCode PetscObjectRemoveReference(PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(PetscObjectListRemoveReference(&obj->olist, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectCompose - Associates another PETSc object with a given PETSc object.

  Not Collective

  Input Parameters:
+ obj  - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
. name - name associated with the child object
- ptr  - the other PETSc object to associate with the PETSc object; this must also be
         cast with (`PetscObject`)

  Level: advanced

  Notes:
  The second objects reference count is automatically increased by one when it is
  composed.

  Replaces any previous object that had been composed with the same name.

  If `ptr` is `NULL` and `name` has previously been composed using an object, then that
  entry is removed from `obj`.

  `PetscObjectCompose()` can be used with any PETSc object (such as
  `Mat`, `Vec`, `KSP`, `SNES`, etc.) or any user-provided object.

  `PetscContainerCreate()` or `PetscObjectContainerCompose()` can be used to create an object from a
  user-provided pointer that may then be composed with PETSc objects using `PetscObjectCompose()`

  Fortran Note:
  Use
.vb
  call PetscObjectCompose(obj, name, PetscObjectCast(ptr), ierr)
.ve

.seealso: `PetscObjectQuery()`, `PetscContainerCreate()`, `PetscObjectComposeFunction()`, `PetscObjectQueryFunction()`, `PetscContainer`,
          `PetscContainerSetPointer()`, `PetscObject`, `PetscObjectContainerCompose()`
@*/
PetscErrorCode PetscObjectCompose(PetscObject obj, const char name[], PetscObject ptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(name, 2);
  if (ptr) PetscValidHeader(ptr, 3);
  PetscCheck(obj != ptr, PetscObjectComm(obj), PETSC_ERR_SUP, "Cannot compose object with itself");
  if (ptr) {
    const char *tname;
    PetscBool   skipreference;

    PetscCall(PetscObjectListReverseFind(ptr->olist, obj, &tname, &skipreference));
    if (tname) PetscCheck(skipreference, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "An object cannot be composed with an object that was composed with it");
  }
  PetscCall(PetscObjectListAdd(&obj->olist, name, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectQuery - Gets a PETSc object associated with a given object that was composed with `PetscObjectCompose()`

  Not Collective

  Input Parameters:
+ obj  - the PETSc object. It must be cast with a (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
. name - name associated with child object
- ptr  - the other PETSc object associated with the PETSc object, this must be
         cast with (`PetscObject`*)

  Level: advanced

  Note:
  The reference count of neither object is increased in this call

  Fortran Note:
  Use
.vb
  call PetscObjectQuery(PetscObjectCast(obj), name, ptr, ierr)
.ve

.seealso: `PetscObjectCompose()`, `PetscObjectComposeFunction()`, `PetscObjectQueryFunction()`, `PetscContainer`
          `PetscContainerGetPointer()`, `PetscObject`
@*/
PetscErrorCode PetscObjectQuery(PetscObject obj, const char name[], PetscObject *ptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(ptr, 3);
  PetscCall(PetscObjectListFind(obj->olist, name, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscObjectComposeFunction - Associates a function with a given PETSc object.

  Synopsis:
  #include <petscsys.h>
  PetscErrorCode PetscObjectComposeFunction(PetscObject obj, const char name[], PetscErrorCodeFn *fptr)

  Logically Collective

  Input Parameters:
+ obj  - the PETSc object; this must be cast with a (`PetscObject`), for example,
         `PetscObjectCompose`((`PetscObject`)mat,...);
. name - name associated with the child function
- fptr - function pointer

  Level: advanced

  Notes:
  When the first argument of `fptr` is (or is derived from) a `PetscObject` then `PetscTryMethod()` and `PetscUseMethod()`
  can be used to call the function directly with error checking.

  To remove a registered routine, pass in `NULL` for `fptr`.

  `PetscObjectComposeFunction()` can be used with any PETSc object (such as
  `Mat`, `Vec`, `KSP`, `SNES`, etc.) or any user-provided object.

  `PetscUseTypeMethod()` and `PetscTryTypeMethod()` are used to call a function that is stored in the objects `obj->ops` table.

.seealso: `PetscObjectQueryFunction()`, `PetscContainerCreate()` `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscTryMethod()`, `PetscUseMethod()`,
          `PetscUseTypeMethod()`, `PetscTryTypeMethod()`, `PetscObject`
M*/
PetscErrorCode PetscObjectComposeFunction_Private(PetscObject obj, const char name[], PetscErrorCodeFn *fptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PetscFunctionListAdd_Private(&obj->qlist, name, fptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscObjectQueryFunction - Gets a function associated with a given object.

  Synopsis:
  #include <petscsys.h>
  PetscErrorCode PetscObjectQueryFunction(PetscObject obj, const char name[], PetscErrorCodeFn **fptr)

  Logically Collective

  Input Parameters:
+ obj  - the PETSc object; this must be cast with (`PetscObject`), for example,
         `PetscObjectQueryFunction`((`PetscObject`)ksp,...);
- name - name associated with the child function

  Output Parameter:
. fptr - function pointer

  Level: advanced

.seealso: `PetscObjectComposeFunction()`, `PetscFunctionListFind()`, `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObject`
M*/
PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Private(PetscObject obj, const char name[], PetscErrorCodeFn **fptr)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PetscFunctionListFind_Private(obj->qlist, name, fptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectHasFunction - Query if a function is associated with a given object.

  Logically Collective

  Input Parameters:
+ obj  - the PETSc object
- name - name associated with the child function

  Output Parameter:
. has - the boolean value

  Level: advanced

.seealso: `PetscObject`, `PetscObjectComposeFunction()`, `PetscObjectQueryFunction()`
@*/
PetscErrorCode PetscObjectHasFunction(PetscObject obj, const char name[], PetscBool *has)
{
  PetscErrorCodeFn *fptr = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(has, 3);
  PetscCall(PetscObjectQueryFunction(obj, name, &fptr));
  *has = fptr ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct _p_PetscContainer {
  PETSCHEADER(int);
  void              *ctx;
  PetscCtxDestroyFn *ctxdestroy;
  PetscErrorCode (*userdestroy_deprecated)(void *);
};

/*@C
  PetscContainerGetPointer - Gets the pointer value contained in the container that was provided with `PetscContainerSetPointer()`

  Not Collective, No Fortran Support

  Input Parameter:
. obj - the object created with `PetscContainerCreate()`

  Output Parameter:
. ptr - the pointer value

  Level: advanced

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`, `PetscObject`,
          `PetscContainerSetPointer()`, `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerGetPointer(PetscContainer obj, PeCtx ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  PetscAssertPointer(ptr, 2);
  *(void **)ptr = obj->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscContainerSetPointer - Sets the pointer value contained in the container.

  Logically Collective, No Fortran Support

  Input Parameters:
+ obj - the object created with `PetscContainerCreate()`
- ptr - the pointer value

  Level: advanced

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`, `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObject`,
          `PetscContainerGetPointer()`, `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerSetPointer(PetscContainer obj, void *ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  if (ptr) PetscAssertPointer(ptr, 2);
  obj->ctx = ptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscContainerDestroy - Destroys a PETSc container object.

  Collective, No Fortran Support

  Input Parameter:
. obj - an object that was created with `PetscContainerCreate()`

  Level: advanced

  Note:
  If `PetscContainerSetCtxDestroy()` was used to provide a user destroy object for the data provided with `PetscContainerSetPointer()`
  then that function is called to destroy the data.

.seealso: `PetscContainerCreate()`, `PetscContainerSetCtxDestroy()`, `PetscObject`, `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerDestroy(PetscContainer *obj)
{
  PetscFunctionBegin;
  if (!*obj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*obj, PETSC_CONTAINER_CLASSID, 1);
  if (--((PetscObject)*obj)->refct > 0) {
    *obj = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*obj)->ctxdestroy) PetscCall((*(*obj)->ctxdestroy)(&(*obj)->ctx));
  else if ((*obj)->userdestroy_deprecated) PetscCall((*(*obj)->userdestroy_deprecated)((*obj)->ctx));
  PetscCall(PetscHeaderDestroy(obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscContainerSetCtxDestroy - Sets the destroy function for the data provided to the `PetscContainer` with `PetscContainerSetPointer()`

  Logically Collective, No Fortran Support

  Input Parameters:
+ obj - an object that was created with `PetscContainerCreate()`
- des - name of the ctx destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: advanced

  Note:
  Use `PetscCtxDestroyDefault()` if the memory was obtained by calling `PetscMalloc()` or one of its variants for single memory allocation.

.seealso: `PetscContainerDestroy()`, `PetscContainerUserDestroyDefault()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc()`, `PetscCalloc1()`, `PetscObject`,
          `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerSetCtxDestroy(PetscContainer obj, PetscCtxDestroyFn *des)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  obj->ctxdestroy = des;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscContainerSetUserDestroy - Sets the destroy function for the data provided to the `PetscContainer` with `PetscContainerSetPointer()`

  Logically Collective, No Fortran Support

  Input Parameters:
+ obj - an object that was created with `PetscContainerCreate()`
- des - name of the ctx destroy function

  Level: advanced

  Notes:
  Deprecated, use `PetscContainerSetCtxDestroy()`

.seealso: `PetscContainerSetCtxDestroy()`, `PetscContainerDestroy()`, `PetscContainerUserDestroyDefault()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc()`, `PetscCalloc1()`, `PetscObject`,
          `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerSetUserDestroy(PetscContainer obj, PetscErrorCode (*des)(void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj, PETSC_CONTAINER_CLASSID, 1);
  obj->userdestroy_deprecated = des;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscClassId PETSC_CONTAINER_CLASSID;

/*@C
  PetscContainerCreate - Creates a PETSc object that has room to hold a single pointer.

  Collective, No Fortran Support

  Input Parameter:
. comm - MPI communicator that shares the object

  Output Parameter:
. container - the container created

  Level: advanced

  Notes:
  This allows one to attach any type of data (accessible through a pointer) with the
  `PetscObjectCompose()` function to a `PetscObject`. The data item itself is attached by a
  call to `PetscContainerSetPointer()`.

.seealso: `PetscContainerDestroy()`, `PetscContainerSetPointer()`, `PetscContainerGetPointer()`, `PetscObjectCompose()`, `PetscObjectQuery()`,
          `PetscContainerSetCtxDestroy()`, `PetscObject`, `PetscObjectContainerCompose()`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscContainerCreate(MPI_Comm comm, PetscContainer *container)
{
  PetscFunctionBegin;
  PetscAssertPointer(container, 2);
  PetscCall(PetscSysInitializePackage());
  PetscCall(PetscHeaderCreate(*container, PETSC_CONTAINER_CLASSID, "PetscContainer", "Container", "Sys", comm, PetscContainerDestroy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectContainerCompose - Creates a `PetscContainer`, provides all of its values and composes it with a `PetscObject`

  Collective

  Input Parameters:
+ obj     - the `PetscObject`
. name    - the name for the composed container
. pointer - the pointer to the data
- destroy - the routine to destroy the container's data, see `PetscCtxDestroyFn` for its calling sequence; use `PetscCtxDestroyDefault()` if a `PetscFree()` frees the data

  Level: advanced

  Notes:
  This allows one to attach any type of data (accessible through a pointer) with the
  `PetscObjectCompose()` function to a `PetscObject`. The data item itself is attached by a
  call to `PetscContainerSetPointer()`.

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`, `PetscContainerSetPointer()`, `PetscContainerGetPointer()`, `PetscObjectCompose()`, `PetscObjectQuery()`,
          `PetscContainerSetCtxDestroy()`, `PetscObject`, `PetscObjectContainerQuery()`
@*/
PetscErrorCode PetscObjectContainerCompose(PetscObject obj, const char *name, void *pointer, PetscCtxDestroyFn *destroy)
{
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PetscContainerCreate(PetscObjectComm(obj), &container));
  PetscCall(PetscContainerSetPointer(container, pointer));
  if (destroy) PetscCall(PetscContainerSetCtxDestroy(container, destroy));
  PetscCall(PetscObjectCompose(obj, name, (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectContainerQuery - Accesses the pointer in a container composed to a `PetscObject` with `PetscObjectContainerCompose()`

  Collective

  Input Parameters:
+ obj  - the `PetscObject`
- name - the name for the composed container

  Output Parameter:
. pointer - the pointer to the data

  Level: advanced

.seealso: `PetscContainerCreate()`, `PetscContainerDestroy()`, `PetscContainerSetPointer()`, `PetscContainerGetPointer()`, `PetscObjectCompose()`, `PetscObjectQuery()`,
          `PetscContainerSetCtxDestroy()`, `PetscObject`, `PetscObjectContainerCompose()`
@*/
PetscErrorCode PetscObjectContainerQuery(PetscObject obj, const char *name, PeCtx pointer)
{
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery(obj, name, (PetscObject *)&container));
  if (container) PetscCall(PetscContainerGetPointer(container, pointer));
  else *(void **)pointer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectSetFromOptions - Sets generic parameters from user options.

  Collective

  Input Parameter:
. obj - the `PetscObject`

  Level: beginner

  Note:
  We have no generic options at present, so this does nothing.

.seealso: `PetscObjectSetOptionsPrefix()`, `PetscObjectGetOptionsPrefix()`, `PetscObject`
@*/
PetscErrorCode PetscObjectSetFromOptions(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectSetUp - Sets up the internal data structures for later use of the object

  Collective

  Input Parameter:
. obj - the `PetscObject`

  Level: advanced

  Note:
  This does nothing at present.

.seealso: `PetscObjectDestroy()`, `PetscObject`
@*/
PetscErrorCode PetscObjectSetUp(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscObjectIsNull - returns true if the given PETSc object is a null object

  Fortran only

  Synopsis:
  #include <petsc/finclude/petscsys.h>
  PetscBool PetscObjectIsNull(PetscObject obj)

  Logically Collective

  Input Parameter:
. obj  - the PETSc object

  Level: beginner

  Example Usage:
.vb
  if (PetscObjectIsNull(dm)) then
  if (.not. PetscObjectIsNull(dm)) then
.ve

  Note:
  Code such as
.vb
  if (dm == PETSC_NULL_DM) then
.ve
  is not allowed.

.seealso: `PetscObject`, `PETSC_NULL_OBJECT`, `PETSC_NULL_VEC`, `PETSC_NULL_VEC_ARRAY`, `PetscObjectNullify()`
M*/

/*MC
  PetscObjectNullify - sets a PETSc object, such as `Vec`, back to the state it had when it was declared, so it
  can be used in a creation routine, such as `VecCreate()`

  Fortran only

  Synopsis:
  #include <petsc/finclude/petscsys.h>
  PetscObjectNullify(PetscObject obj)

  Logically Collective

  Input Parameter:
. obj  - the PETSc object

  Level: beginner

  Example Usage:
.vb
  Vec x, y

  VecCreate(PETSC_COMM_WORLD, x, ierr)
  ...
  y = x
  ...
  PetscObjectNullify(y)
.ve
  You should not call `VecDestroy()` on `y` because that will destroy `x` since the assignment `y = x` does
  not increase the reference count of `x`

  Note:
  Code such as
.vb
  y = PETSC_NULL_VEC
.ve
  is not allowed.

.seealso: `PetscObject`, `PETSC_NULL_OBJECT`, `PETSC_NULL_VEC`, `PETSC_NULL_VEC_ARRAY`, `PetscObjectIsNull()`
M*/

/*MC
  PetscObjectCast - Casts a `PetscObject` to the base `PetscObject` type in function calls

  Fortran only

  Synopsis:
  use petscsys

  Level: beginner

  Example Usage:
  PetscFE fe
.vb
  PetscCallA(DMAddField(dm, 0, PetscObjectCast(fe),ierr)
.ve

.seealso: `PetscObject`, `PetscObjectSpecificCast()`
M*/

/*MC
  PetscObjectSpecificCast - Casts a `PetscObject` to any specific `PetscObject`

  Fortran only

  Synopsis:
  use petscsys

  Level: beginner

  Example Usage:
  PetscObject obj
  PetscFE     fe
.vb
  PetscCallA(PetscDSGetDiscretization(ds, 0, obj, ierr)
  PetscObjectSpecificCast(fe,obj)
.ve

.seealso: `PetscObject`, `PetscObjectCast()`
M*/

/*MC
  PetscEnumCase - `case()` statement for a PETSc enum variable or value

  Fortran only

  Synopsis:
  #include <petsc/finclude/petscsys.h>
  PetscEnumCase(PetscObject enm)

  Input Parameters:
. enum  - the PETSc enum value or variable

  Level: beginner

  Example Usage:
.vb
  DMPolytopeType cellType
  select PetscEnumCase(cellType)
    PetscEnumCase(DM_POLYTOPE_TRIANGLE)
      write(*,*) 'cell is a triangle'
    PetscEnumCase(DM_POLYTOPE_TETRAHEDRON)
      write(*,*) 'cell is a tetrahedron'
    case default
      write(*,*) 'cell is a something else'
  end select
.ve
  is equivalent to
.vb
  DMPolytopeType cellType
  select case(cellType%v)
    case(DM_POLYTOPE_TRIANGLE%v)
      write(*,*) 'cell is a triangle'
    case(DM_POLYTOPE_TETRAHEDRON%v)
      write(*,*) 'cell is a tetrahedron'
    case default
      write(*,*) 'cell is a something else'
  end select
.ve

.seealso: `PetscObject`
M*/
