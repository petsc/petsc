/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/
#include <petscviewer.h>

static PetscErrorCode DestroyComposedData(void ***composed_star, PetscObjectState **state_star, PetscInt *count_star, void **composed, PetscObjectState **state)
{
  void **tmp_star = *composed_star;

  PetscFunctionBegin;
  for (PetscInt i = 0, imax = *count_star; i < imax; ++i) PetscCall(PetscFree(tmp_star[i]));
  PetscCall(PetscFree2(*composed_star, *state_star));
  PetscCall(PetscFree2(*composed, *state));
  *count_star = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCall(DestroyComposedData((void ***)&obj->intstarcomposeddata, &obj->intstarcomposedstate, &obj->intstar_idmax, (void **)&obj->intcomposeddata, &obj->intcomposedstate));
  PetscCall(DestroyComposedData((void ***)&obj->realstarcomposeddata, &obj->realstarcomposedstate, &obj->realstar_idmax, (void **)&obj->realcomposeddata, &obj->realcomposedstate));
#if PetscDefined(USE_COMPLEX)
  PetscCall(DestroyComposedData((void ***)&obj->scalarstarcomposeddata, &obj->scalarstarcomposedstate, &obj->scalarstar_idmax, (void **)&obj->scalarcomposeddata, &obj->scalarcomposedstate));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectDestroy - Destroys a `PetscObject`, regardless of the type.

  Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`\*), for example,
        `PetscObjectDestroy`((`PetscObject`\*)&mat);

  Level: beginner

.seealso: `PetscObject`
@*/
PetscErrorCode PetscObjectDestroy(PetscObject *obj)
{
  PetscFunctionBegin;
  if (!obj || !*obj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeader(*obj, 1);
  PetscCheck((*obj)->bops->destroy, PETSC_COMM_SELF, PETSC_ERR_PLIB, "This PETSc object of class %s does not have a generic destroy routine", (*obj)->class_name);
  PetscCall((*(*obj)->bops->destroy)(obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectView - Views a `PetscObject` regardless of the type.

  Collective

  Input Parameters:
+ obj    - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
           `PetscObjectView`((`PetscObject`)mat,`viewer`);
- viewer - any PETSc viewer

  Level: intermediate

.seealso: `PetscObject`, `PetscObjectViewFromOptions()`, `PetscViewer`
@*/
PetscErrorCode PetscObjectView(PetscObject obj, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscCheck(obj->bops->view, PETSC_COMM_SELF, PETSC_ERR_SUP, "This PETSc object does not have a generic viewer routine");
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(obj->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall((*obj->bops->view)(obj, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectViewFromOptions - Processes command line options to determine if/how a `PetscObject` is to be viewed.

  Collective

  Input Parameters:
+ obj        - the object
. bobj       - optional other object that provides prefix (if `NULL` then the prefix in `obj` is used)
- optionname - option string that is used to activate viewing

  Options Database Key:
. -optionname_view [viewertype]:... - option name and values. In actual usage this would be something like `-mat_coarse_view`

  Level: developer

  Notes:
  The argument has the following form
.vb
    type:filename:format:filemode
.ve
  where all parts are optional, but you need to include the colon to access the next part. For example, to read from an HDF5 file, use
.vb
    hdf5:sol.h5::read
.ve

.vb
    If no value is provided ascii:stdout is used
       ascii[:[filename][:[format][:append]]]    defaults to stdout - format can be one of ascii_info, ascii_info_detail, or ascii_matlab,
                                                  for example ascii::ascii_info prints just the information about the object not all details
                                                  unless :append is given filename opens in write mode, overwriting what was already there
       binary[:[filename][:[format][:append]]]   defaults to the file binaryoutput
       draw[:drawtype[:filename]]                for example, draw:tikz, draw:tikz:figure.tex  or draw:x
       socket[:port]                             defaults to the standard output port
       saws[:communicatorname]                    publishes object to the Scientific Application Webserver (SAWs)
.ve

  This is not called directly but is called by, for example, `MatViewFromOptions()`

.seealso: `PetscObject`, `PetscObjectView()`, `PetscOptionsCreateViewer()`
@*/
PetscErrorCode PetscObjectViewFromOptions(PetscObject obj, PetscObject bobj, const char optionname[])
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (bobj) PetscValidHeader(bobj, 2);
  if (incall) PetscFunctionReturn(PETSC_SUCCESS);
  incall = PETSC_TRUE;
  prefix = bobj ? bobj->prefix : obj->prefix;
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm(obj), obj->options, prefix, optionname, &viewer, &format, &flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(PetscObjectView(obj, viewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectTypeCompare - Determines whether a PETSc object is of a particular type.

  Not Collective

  Input Parameters:
+ obj       - a PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
              `PetscObjectTypeCompare`((`PetscObject`)mat);
- type_name - string containing a type name

  Output Parameter:
. same - `PETSC_TRUE` if the type of `obj` and `type_name` are the same or both `NULL`, else `PETSC_FALSE`

  Level: intermediate

.seealso: `PetscObject`, `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectBaseTypeCompare()`, `PetscObjectTypeCompareAny()`, `PetscObjectBaseTypeCompareAny()`, `PetscObjectObjectTypeCompare()`
@*/
PetscErrorCode PetscObjectTypeCompare(PetscObject obj, const char type_name[], PetscBool *same)
{
  PetscFunctionBegin;
  PetscAssertPointer(same, 3);
  if (!obj) *same = (PetscBool)!type_name;
  else {
    PetscValidHeader(obj, 1);
    if (!type_name || !obj->type_name) *same = (PetscBool)(!obj->type_name == !type_name);
    else {
      PetscAssertPointer(type_name, 2);
      PetscCall(PetscStrcmp(obj->type_name, type_name, same));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectObjectTypeCompare - Determines whether two PETSc objects are of the same type

  Logically Collective

  Input Parameters:
+ obj1 - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
- obj2 - another PETSc object

  Output Parameter:
. same - `PETSC_TRUE` if they are the same or both unset, else `PETSC_FALSE`

  Level: intermediate

.seealso: `PetscObjectTypeCompare()`, `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectBaseTypeCompare()`, `PetscObjectTypeCompareAny()`, `PetscObjectBaseTypeCompareAny()`

@*/
PetscErrorCode PetscObjectObjectTypeCompare(PetscObject obj1, PetscObject obj2, PetscBool *same)
{
  PetscFunctionBegin;
  PetscValidHeader(obj1, 1);
  PetscValidHeader(obj2, 2);
  PetscAssertPointer(same, 3);
  PetscCall(PetscStrcmp(obj1->type_name, obj2->type_name, same));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectBaseTypeCompare - Determines whether a `PetscObject` is of a given base type. For example the base type of `MATSEQAIJPERM` is `MATSEQAIJ`

  Not Collective

  Input Parameters:
+ obj       - the object
- type_name - string containing a type name

  Output Parameter:
. same - `PETSC_TRUE` if the object is of the same base type identified by `type_name` or both `NULL`, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: `PetscObject`, `PetscObjectTypeCompare()`, `PetscObjectTypeCompareAny()`, `PetscObjectBaseTypeCompareAny()`
@*/
PetscErrorCode PetscObjectBaseTypeCompare(PetscObject obj, const char type_name[], PetscBool *same)
{
  PetscFunctionBegin;
  PetscAssertPointer(same, 3);
  if (!obj) *same = (PetscBool)!type_name;
  else {
    PetscValidHeader(obj, 1);
    if (!type_name || !obj->type_name) *same = (PetscBool)(!obj->type_name == !type_name);
    else {
      PetscAssertPointer(type_name, 2);
      PetscCall(PetscStrbeginswith(obj->type_name, type_name, same));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectTypeCompareAny - Determines whether a PETSc object is of any of a list of types.

  Not Collective

  Input Parameters:
+ obj       - a PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`),
              for example, `PetscObjectTypeCompareAny`((`PetscObject`)mat,...);
- type_name - one or more string arguments containing type names, pass the empty string "" as the last argument

  Output Parameter:
. match - `PETSC_TRUE` if the type of `obj` matches any in the list, else `PETSC_FALSE`

  Level: intermediate

.seealso: `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectTypeCompare()`, `PetscObjectBaseTypeCompare()`
@*/
PetscErrorCode PetscObjectTypeCompareAny(PetscObject obj, PetscBool *match, const char type_name[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  PetscAssertPointer(match, 2);
  *match = PETSC_FALSE;
  if (!obj) PetscFunctionReturn(PETSC_SUCCESS);
  va_start(Argp, type_name);
  while (type_name && type_name[0]) {
    PetscBool found;
    PetscCall(PetscObjectTypeCompare(obj, type_name, &found));
    if (found) {
      *match = PETSC_TRUE;
      break;
    }
    type_name = va_arg(Argp, const char *);
  }
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectBaseTypeCompareAny - Determines whether a PETSc object has the base type of any of a list of types.

  Not Collective

  Input Parameters:
+ obj       - a PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`),
              for example, `PetscObjectBaseTypeCompareAny`((`PetscObject`)mat,...);
- type_name - one or more string arguments containing type names, pass the empty string "" as the last argument

  Output Parameter:
. match - `PETSC_TRUE` if the type of `obj` matches any in the list, else `PETSC_FALSE`

  Level: intermediate

.seealso: `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectTypeCompare()`, `PetscObjectBaseTypeCompare()`, `PetscObjectTypeCompareAny()`
@*/
PetscErrorCode PetscObjectBaseTypeCompareAny(PetscObject obj, PetscBool *match, const char type_name[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  PetscAssertPointer(match, 2);
  *match = PETSC_FALSE;
  va_start(Argp, type_name);
  while (type_name && type_name[0]) {
    PetscBool found;
    PetscCall(PetscObjectBaseTypeCompare(obj, type_name, &found));
    if (found) {
      *match = PETSC_TRUE;
      break;
    }
    type_name = va_arg(Argp, const char *);
  }
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscErrorCode (*func)(void);
} PetscFinalizeFunction;

typedef struct {
  PetscErrorCode (*func)(void *);
  void *ctx;
} PetscFinalizeFunctionWithCtx;

typedef enum {
  PETSC_FINALIZE_EMPTY,
  PETSC_FINALIZE_OBJECT,
  PETSC_FINALIZE_FUNC,
  PETSC_FINALIZE_FUNC_WITH_CTX
} PetscFinalizeType;

static const char *const PetscFinalizeTypes[] = {"PETSC_FINALIZE_EMPTY", "PETSC_FINALIZE_OBJECT", "PETSC_FINALIZE_FUNC", "PETSC_FINALIZE_FUNC_WITH_CTX", PETSC_NULLPTR};

typedef struct {
  union ThunkUnion
  {
    PetscObject                  obj;
    PetscFinalizeFunction        fn;
    PetscFinalizeFunctionWithCtx fnctx;
  } thunk;
  PetscFinalizeType type;
} PetscFinalizerContainer;

#define PETSC_MAX_REGISTERED_FINALIZERS 256
static int                     reg_count = 0;
static PetscFinalizerContainer regfin[PETSC_MAX_REGISTERED_FINALIZERS];

static PetscErrorCode PetscRunRegisteredFinalizers(void)
{
  PetscFunctionBegin;
  while (reg_count) {
    PetscFinalizerContainer top = regfin[--reg_count];

    regfin[reg_count].type = PETSC_FINALIZE_EMPTY;
    PetscCall(PetscArrayzero(&regfin[reg_count].thunk, 1));
    switch (top.type) {
    case PETSC_FINALIZE_OBJECT:
      PetscCall(PetscObjectDestroy(&top.thunk.obj));
      break;
    case PETSC_FINALIZE_FUNC:
      PetscCall((*top.thunk.fn.func)());
      break;
    case PETSC_FINALIZE_FUNC_WITH_CTX:
      PetscCall((*top.thunk.fnctx.func)(top.thunk.fnctx.ctx));
      break;
    case PETSC_FINALIZE_EMPTY:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Finalizer at position %d is empty, yet registration count %d != 0", reg_count, reg_count);
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int PetscFinalizerContainerEqual(const PetscFinalizerContainer *a, const PetscFinalizerContainer *b)
{
  if (a->type != b->type) return 0;
  switch (a->type) {
  case PETSC_FINALIZE_EMPTY:
    break;
  case PETSC_FINALIZE_OBJECT:
    return a->thunk.obj == b->thunk.obj;
  case PETSC_FINALIZE_FUNC:
    return a->thunk.fn.func == b->thunk.fn.func;
  case PETSC_FINALIZE_FUNC_WITH_CTX:
    return a->thunk.fnctx.func == b->thunk.fnctx.func && a->thunk.fnctx.ctx == b->thunk.fnctx.ctx;
  }
  return 1;
}

static PetscErrorCode RegisterFinalizer(PetscFinalizerContainer container)
{
  PetscFunctionBegin;
  PetscAssert(reg_count < (int)PETSC_STATIC_ARRAY_LENGTH(regfin), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "No more room in array, limit %zu, recompile %s with larger value for " PetscStringize(regfin), PETSC_STATIC_ARRAY_LENGTH(regfin), __FILE__);
  PetscAssert(regfin[reg_count].type == PETSC_FINALIZE_EMPTY, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Finalizer type (%s) at position %d is not PETSC_FINALIZE_EMPTY!", PetscFinalizeTypes[regfin[reg_count].type], reg_count);
  if (PetscDefined(USE_DEBUG)) {
    for (int i = 0; i < reg_count; ++i) PetscCheck(!PetscFinalizerContainerEqual(regfin + i, &container), PETSC_COMM_SELF, PETSC_ERR_ORDER, "Finalizer (of type %s) already registered!", PetscFinalizeTypes[container.type]);
  }
  regfin[reg_count++] = container;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectRegisterDestroy - Registers a PETSc object to be destroyed when
  `PetscFinalize()` is called.

  Logically Collective

  Input Parameter:
. obj - a PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectRegisterDestroy`((`PetscObject`)mat);

  Level: developer

  Note:
  This is used by, for example, `PETSC_VIEWER_XXX_()` routines to free the viewer
  when PETSc ends.

.seealso: `PetscObjectRegisterDestroyAll()`
@*/
PetscErrorCode PetscObjectRegisterDestroy(PetscObject obj)
{
  PetscFinalizerContainer container;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  container.thunk.obj = obj;
  container.type      = PETSC_FINALIZE_OBJECT;
  PetscCall(RegisterFinalizer(container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectRegisterDestroyAll - Frees all the PETSc objects that have been registered
  with `PetscObjectRegisterDestroy()`. Called by `PetscFinalize()`

  Logically Collective on the individual `PetscObject`s that are being processed

  Level: developer

.seealso: `PetscObjectRegisterDestroy()`
@*/
PetscErrorCode PetscObjectRegisterDestroyAll(void)
{
  PetscFunctionBegin;
  PetscCall(PetscRunRegisteredFinalizers());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegisterFinalize - Registers a function that is to be called in `PetscFinalize()`

  Not Collective

  Input Parameter:
. f - function to be called

  Level: developer

  Notes:
  This is used by, for example, `DMInitializePackage()` to have `DMFinalizePackage()` called

  Use `PetscObjectRegisterDestroy()` to register the destruction of an object in `PetscFinalize()`

.seealso: `PetscRegisterFinalizeAll()`, `PetscObjectRegisterDestroy()`
@*/
PetscErrorCode PetscRegisterFinalize(PetscErrorCode (*f)(void))
{
  PetscFinalizerContainer container;

  PetscFunctionBegin;
  PetscValidFunction(f, 1);
  container.thunk.fn.func = f;
  container.type          = PETSC_FINALIZE_FUNC;
  PetscCall(RegisterFinalizer(container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegisterFinalizeAll - Runs all the finalize functions set with `PetscRegisterFinalize()`

  Not Collective unless registered functions are collective

  Level: developer

.seealso: `PetscRegisterFinalize()`, `PetscObjectRegisterDestroyAll()`
@*/
PetscErrorCode PetscRegisterFinalizeAll(void)
{
  PetscFunctionBegin;
  PetscCall(PetscRunRegisteredFinalizers());
  PetscFunctionReturn(PETSC_SUCCESS);
}
