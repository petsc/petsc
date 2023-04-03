/*
    The PF mathematical functions interface routines, callable by users.
*/
#include <../src/vec/pf/pfimpl.h> /*I "petscpf.h" I*/

PetscClassId      PF_CLASSID          = 0;
PetscFunctionList PFList              = NULL; /* list of all registered PD functions */
PetscBool         PFRegisterAllCalled = PETSC_FALSE;

/*@C
   PFSet - Sets the C/C++/Fortran functions to be used by the PF function

   Collective

   Input Parameters:
+  pf - the function context
.  apply - function to apply to an array
.  applyvec - function to apply to a Vec
.  view - function that prints information about the `PF`
.  destroy - function to free the private function context
-  ctx - private function context

   Level: beginner

.seealso: `PF`, `PFCreate()`, `PFDestroy()`, `PFSetType()`, `PFApply()`, `PFApplyVec()`
@*/
PetscErrorCode PFSet(PF pf, PetscErrorCode (*apply)(void *, PetscInt, const PetscScalar *, PetscScalar *), PetscErrorCode (*applyvec)(void *, Vec, Vec), PetscErrorCode (*view)(void *, PetscViewer), PetscErrorCode (*destroy)(void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  pf->data          = ctx;
  pf->ops->destroy  = destroy;
  pf->ops->apply    = apply;
  pf->ops->applyvec = applyvec;
  pf->ops->view     = view;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFDestroy - Destroys `PF` context that was created with `PFCreate()`.

   Collective

   Input Parameter:
.  pf - the function context

   Level: beginner

.seealso: `PF`, `PFCreate()`, `PFSet()`, `PFSetType()`
@*/
PetscErrorCode PFDestroy(PF *pf)
{
  PetscFunctionBegin;
  if (!*pf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*pf), PF_CLASSID, 1);
  if (--((PetscObject)(*pf))->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PFViewFromOptions(*pf, NULL, "-pf_view"));
  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*pf));

  if ((*pf)->ops->destroy) PetscCall((*(*pf)->ops->destroy)((*pf)->data));
  PetscCall(PetscHeaderDestroy(pf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFCreate - Creates a mathematical function context.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  dimin - dimension of the space you are mapping from
-  dimout - dimension of the space you are mapping to

   Output Parameter:
.  pf - the function context

   Level: developer

.seealso: `PF`, `PFSet()`, `PFApply()`, `PFDestroy()`, `PFApplyVec()`
@*/
PetscErrorCode PFCreate(MPI_Comm comm, PetscInt dimin, PetscInt dimout, PF *pf)
{
  PF newpf;

  PetscFunctionBegin;
  PetscValidPointer(pf, 4);
  *pf = NULL;
  PetscCall(PFInitializePackage());

  PetscCall(PetscHeaderCreate(newpf, PF_CLASSID, "PF", "Mathematical functions", "Vec", comm, PFDestroy, PFView));
  newpf->data          = NULL;
  newpf->ops->destroy  = NULL;
  newpf->ops->apply    = NULL;
  newpf->ops->applyvec = NULL;
  newpf->ops->view     = NULL;
  newpf->dimin         = dimin;
  newpf->dimout        = dimout;

  *pf = newpf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------------------------------------------------------------------------------*/

/*@
   PFApplyVec - Applies the mathematical function to a vector

   Collective

   Input Parameters:
+  pf - the function context
-  x - input vector (or `NULL` for the vector (0,1, .... N-1)

   Output Parameter:
.  y - output vector

   Level: beginner

.seealso: `PF`, `PFApply()`, `PFCreate()`, `PFDestroy()`, `PFSetType()`, `PFSet()`
@*/
PetscErrorCode PFApplyVec(PF pf, Vec x, Vec y)
{
  PetscInt  i, rstart, rend, n, p;
  PetscBool nox = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  if (x) {
    PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
    PetscCheck(x != y, PETSC_COMM_SELF, PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  } else {
    PetscScalar *xx;
    PetscInt     lsize;

    PetscCall(VecGetLocalSize(y, &lsize));
    lsize = pf->dimin * lsize / pf->dimout;
    PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)y), lsize, PETSC_DETERMINE, &x));
    nox = PETSC_TRUE;
    PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
    PetscCall(VecGetArray(x, &xx));
    for (i = rstart; i < rend; i++) xx[i - rstart] = (PetscScalar)i;
    PetscCall(VecRestoreArray(x, &xx));
  }

  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetLocalSize(y, &p));
  PetscCheck((pf->dimin * (n / pf->dimin)) == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local input vector length %" PetscInt_FMT " not divisible by dimin %" PetscInt_FMT " of function", n, pf->dimin);
  PetscCheck((pf->dimout * (p / pf->dimout)) == p, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local output vector length %" PetscInt_FMT " not divisible by dimout %" PetscInt_FMT " of function", p, pf->dimout);
  PetscCheck((n / pf->dimin) == (p / pf->dimout), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local vector lengths %" PetscInt_FMT " %" PetscInt_FMT " are wrong for dimin and dimout %" PetscInt_FMT " %" PetscInt_FMT " of function", n, p, pf->dimin, pf->dimout);

  if (pf->ops->applyvec) PetscCall((*pf->ops->applyvec)(pf->data, x, y));
  else {
    PetscScalar *xx, *yy;

    PetscCall(VecGetLocalSize(x, &n));
    n = n / pf->dimin;
    PetscCall(VecGetArray(x, &xx));
    PetscCall(VecGetArray(y, &yy));
    PetscCall((*pf->ops->apply)(pf->data, n, xx, yy));
    PetscCall(VecRestoreArray(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
  }
  if (nox) PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PFApply - Applies the mathematical function to an array of values.

   Collective

   Input Parameters:
+  pf - the function context
.  n - number of pointwise function evaluations to perform, each pointwise function evaluation
       is a function of dimin variables and computes dimout variables where dimin and dimout are defined
       in the call to `PFCreate()`
-  x - input array

   Output Parameter:
.  y - output array

   Level: beginner

.seealso: `PF`, `PFApplyVec()`, `PFCreate()`, `PFDestroy()`, `PFSetType()`, `PFSet()`
@*/
PetscErrorCode PFApply(PF pf, PetscInt n, const PetscScalar *x, PetscScalar *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  PetscValidScalarPointer(x, 3);
  PetscValidScalarPointer(y, 4);
  PetscCheck(x != y, PETSC_COMM_SELF, PETSC_ERR_ARG_IDN, "x and y must be different arrays");

  PetscCall((*pf->ops->apply)(pf->data, n, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFViewFromOptions - View a `PF` based on options set in the options database

   Collective

   Input Parameters:
+  A - the `PF` context
.  obj - Optional object that provides the prefix used to search the options database
-  name - command line option

   Level: intermediate

   Note:
  See `PetscObjectViewFromOptions()` for the variety of viewer options available

.seealso: `PF`, `PFView`, `PetscObjectViewFromOptions()`, `PFCreate()`
@*/
PetscErrorCode PFViewFromOptions(PF A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PF_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PFView - Prints information about a mathematical function

   Collective unless `viewer` is `PETSC_VIEWER_STDOUT_SELF`

   Input Parameters:
+  PF - the `PF` context
-  viewer - optional visualization context

   Level: developer

   Note:
   The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization contexts with
   `PetscViewerASCIIOpen()` (output to a specified file).

.seealso: `PF`, `PetscViewerCreate()`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode PFView(PF pf, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pf), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(pf, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)pf, viewer));
    if (pf->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall((*pf->ops->view)(pf->data, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFRegister - Adds a method to the mathematical function package.

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined solver
-  function - routine to create method context

   Sample usage:
.vb
   PFRegister("my_function",MyFunctionSetCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PFSetType(pf,"my_function")
   or at runtime via the option
$     -pf_type my_function

   Level: advanced

   Note:
   `PFRegister()` may be called multiple times to add several user-defined functions

.seealso: `PF`, `PFRegisterAll()`, `PFRegisterDestroy()`, `PFRegister()`
@*/
PetscErrorCode PFRegister(const char sname[], PetscErrorCode (*function)(PF, void *))
{
  PetscFunctionBegin;
  PetscCall(PFInitializePackage());
  PetscCall(PetscFunctionListAdd(&PFList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFGetType - Gets the `PFType` name (as a string) from the `PF`
   context.

   Not Collective

   Input Parameter:
.  pf - the function context

   Output Parameter:
.  type - name of function

   Level: intermediate

.seealso: `PF`, `PFSetType()`
@*/
PetscErrorCode PFGetType(PF pf, PFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)pf)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PFSetType - Builds `PF` for a particular function

   Collective

   Input Parameters:
+  pf - the function context.
.  type - a known method
-  ctx - optional type dependent context

   Options Database Key:
.  -pf_type <type> - Sets PF type

  Level: intermediate

  Note:
  See "petsc/include/petscpf.h" for available methods (for instance, `PFCONSTANT`)

.seealso: `PF`, `PFSet()`, `PFRegister()`, `PFCreate()`, `DMDACreatePF()`
@*/
PetscErrorCode PFSetType(PF pf, PFType type, void *ctx)
{
  PetscBool match;
  PetscErrorCode (*r)(PF, void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)pf, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscTryTypeMethod(pf, destroy);
  pf->data = NULL;

  /* Determine the PFCreateXXX routine for a particular function */
  PetscCall(PetscFunctionListFind(PFList, type, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PF type %s", type);
  pf->ops->destroy  = NULL;
  pf->ops->view     = NULL;
  pf->ops->apply    = NULL;
  pf->ops->applyvec = NULL;

  /* Call the PFCreateXXX routine for this particular function */
  PetscCall((*r)(pf, ctx));

  PetscCall(PetscObjectChangeTypeName((PetscObject)pf, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PFSetFromOptions - Sets `PF` options from the options database.

   Collective

   Input Parameters:
.  pf - the mathematical function context

   Level: intermediate

   Notes:
   To see all options, run your program with the -help option
   or consult the users manual.

.seealso: `PF`
@*/
PetscErrorCode PFSetFromOptions(PF pf)
{
  char      type[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf, PF_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)pf);
  PetscCall(PetscOptionsFList("-pf_type", "Type of function", "PFSetType", PFList, NULL, type, 256, &flg));
  if (flg) PetscCall(PFSetType(pf, type, NULL));
  PetscTryTypeMethod(pf, setfromoptions, PetscOptionsObject);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)pf, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool PFPackageInitialized = PETSC_FALSE;
/*@C
  PFFinalizePackage - This function destroys everything in the PETSc `PF` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: `PF`, `PetscFinalize()`
@*/
PetscErrorCode PFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PFList));
  PFPackageInitialized = PETSC_FALSE;
  PFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PFInitializePackage - This function initializes everything in the `PF` package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to `PFCreate()`
  when using shared or static libraries.

  Level: developer

.seealso: `PF`, `PetscInitialize()`
@*/
PetscErrorCode PFInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("PointFunction", &PF_CLASSID));
  /* Register Constructors */
  PetscCall(PFRegisterAll());
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PF_CLASSID;
    PetscCall(PetscInfoProcessClass("pf", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("pf", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PF_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
