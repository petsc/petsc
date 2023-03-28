#include <petsc/private/petscfvimpl.h> /*I "petscfv.h" I*/
#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscds.h>

PetscClassId PETSCLIMITER_CLASSID = 0;

PetscFunctionList PetscLimiterList              = NULL;
PetscBool         PetscLimiterRegisterAllCalled = PETSC_FALSE;

PetscBool  Limitercite       = PETSC_FALSE;
const char LimiterCitation[] = "@article{BergerAftosmisMurman2005,\n"
                               "  title   = {Analysis of slope limiters on irregular grids},\n"
                               "  journal = {AIAA paper},\n"
                               "  author  = {Marsha Berger and Michael J. Aftosmis and Scott M. Murman},\n"
                               "  volume  = {490},\n"
                               "  year    = {2005}\n}\n";

/*@C
  PetscLimiterRegister - Adds a new `PetscLimiter` implementation

  Not Collective

  Input Parameters:
+ sname - The name of a new user-defined creation routine
- function - The creation routine

  Sample usage:
.vb
    PetscLimiterRegister("my_lim", MyPetscLimiterCreate);
.ve

  Then, your `PetscLimiter` type can be chosen with the procedural interface via
.vb
    PetscLimiterCreate(MPI_Comm, PetscLimiter *);
    PetscLimiterSetType(PetscLimiter, "my_lim");
.ve
   or at runtime via the option
.vb
    -petsclimiter_type my_lim
.ve

  Level: advanced

  Note:
  `PetscLimiterRegister()` may be called multiple times to add several user-defined PetscLimiters

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterRegisterAll()`, `PetscLimiterRegisterDestroy()`
@*/
PetscErrorCode PetscLimiterRegister(const char sname[], PetscErrorCode (*function)(PetscLimiter))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscLimiterList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLimiterSetType - Builds a `PetscLimiter` for a given `PetscLimiterType`

  Collective

  Input Parameters:
+ lim  - The `PetscLimiter` object
- name - The kind of limiter

  Options Database Key:
. -petsclimiter_type <type> - Sets the PetscLimiter type; use -help for a list of available types

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterGetType()`, `PetscLimiterCreate()`
@*/
PetscErrorCode PetscLimiterSetType(PetscLimiter lim, PetscLimiterType name)
{
  PetscErrorCode (*r)(PetscLimiter);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)lim, name, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLimiterRegisterAll());
  PetscCall(PetscFunctionListFind(PetscLimiterList, name, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)lim), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscLimiter type: %s", name);

  if (lim->ops->destroy) {
    PetscUseTypeMethod(lim, destroy);
    lim->ops->destroy = NULL;
  }
  PetscCall((*r)(lim));
  PetscCall(PetscObjectChangeTypeName((PetscObject)lim, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLimiterGetType - Gets the `PetscLimiterType` name (as a string) from the `PetscLimiter`.

  Not Collective

  Input Parameter:
. lim  - The `PetscLimiter`

  Output Parameter:
. name - The `PetscLimiterType`

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterSetType()`, `PetscLimiterCreate()`
@*/
PetscErrorCode PetscLimiterGetType(PetscLimiter lim, PetscLimiterType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidPointer(name, 2);
  PetscCall(PetscLimiterRegisterAll());
  *name = ((PetscObject)lim)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscLimiterViewFromOptions - View a `PetscLimiter` based on values in the options database

   Collective

   Input Parameters:
+  A - the `PetscLimiter` object to view
.  obj - Optional object that provides the options prefix to use
-  name - command line option name

   Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterView()`, `PetscObjectViewFromOptions()`, `PetscLimiterCreate()`
@*/
PetscErrorCode PetscLimiterViewFromOptions(PetscLimiter A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLimiterView - Views a `PetscLimiter`

  Collective

  Input Parameters:
+ lim - the `PetscLimiter` object to view
- v   - the viewer

  Level: beginner

.seealso: `PetscLimiter`, `PetscViewer`, `PetscLimiterDestroy()`, `PetscLimiterViewFromOptions()`
@*/
PetscErrorCode PetscLimiterView(PetscLimiter lim, PetscViewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)lim), &v));
  PetscTryTypeMethod(lim, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLimiterSetFromOptions - sets parameters in a `PetscLimiter` from the options database

  Collective

  Input Parameter:
. lim - the `PetscLimiter` object to set options for

  Level: intermediate

.seealso: `PetscLimiter`, ``PetscLimiterView()`
@*/
PetscErrorCode PetscLimiterSetFromOptions(PetscLimiter lim)
{
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (!((PetscObject)lim)->type_name) defaultType = PETSCLIMITERSIN;
  else defaultType = ((PetscObject)lim)->type_name;
  PetscCall(PetscLimiterRegisterAll());

  PetscObjectOptionsBegin((PetscObject)lim);
  PetscCall(PetscOptionsFList("-petsclimiter_type", "Finite volume slope limiter", "PetscLimiterSetType", PetscLimiterList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(PetscLimiterSetType(lim, name));
  } else if (!((PetscObject)lim)->type_name) {
    PetscCall(PetscLimiterSetType(lim, defaultType));
  }
  PetscTryTypeMethod(lim, setfromoptions);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)lim, PetscOptionsObject));
  PetscOptionsEnd();
  PetscCall(PetscLimiterViewFromOptions(lim, NULL, "-petsclimiter_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLimiterSetUp - Construct data structures for the `PetscLimiter`

  Collective

  Input Parameter:
. lim - the `PetscLimiter` object to setup

  Level: intermediate

.seealso: `PetscLimiter`, ``PetscLimiterView()`, `PetscLimiterDestroy()`
@*/
PetscErrorCode PetscLimiterSetUp(PetscLimiter lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscTryTypeMethod(lim, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLimiterDestroy - Destroys a `PetscLimiter` object

  Collective

  Input Parameter:
. lim - the `PetscLimiter` object to destroy

  Level: beginner

.seealso: `PetscLimiter`, `PetscLimiterView()`
@*/
PetscErrorCode PetscLimiterDestroy(PetscLimiter *lim)
{
  PetscFunctionBegin;
  if (!*lim) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*lim), PETSCLIMITER_CLASSID, 1);

  if (--((PetscObject)(*lim))->refct > 0) {
    *lim = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ((PetscObject)(*lim))->refct = 0;

  PetscTryTypeMethod((*lim), destroy);
  PetscCall(PetscHeaderDestroy(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLimiterCreate - Creates an empty `PetscLimiter` object. The type can then be set with `PetscLimiterSetType()`.

  Collective

  Input Parameter:
. comm - The communicator for the `PetscLimiter` object

  Output Parameter:
. lim - The `PetscLimiter` object

  Level: beginner

.seealso: `PetscLimiter`, PetscLimiterType`, `PetscLimiterSetType()`, `PETSCLIMITERSIN`
@*/
PetscErrorCode PetscLimiterCreate(MPI_Comm comm, PetscLimiter *lim)
{
  PetscLimiter l;

  PetscFunctionBegin;
  PetscValidPointer(lim, 2);
  PetscCall(PetscCitationsRegister(LimiterCitation, &Limitercite));
  *lim = NULL;
  PetscCall(PetscFVInitializePackage());

  PetscCall(PetscHeaderCreate(l, PETSCLIMITER_CLASSID, "PetscLimiter", "Finite Volume Slope Limiter", "PetscLimiter", comm, PetscLimiterDestroy, PetscLimiterView));

  *lim = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLimiterLimit - Limit the flux

  Input Parameters:
+ lim  - The `PetscLimiter`
- flim - The input field

  Output Parameter:
. phi  - The limited field

  Level: beginner

  Note:
  Limiters given in symmetric form following Berger, Aftosmis, and Murman 2005
.vb
 The classical flux-limited formulation is psi(r) where

 r = (u[0] - u[-1]) / (u[1] - u[0])

 The second order TVD region is bounded by

 psi_minmod(r) = min(r,1)      and        psi_superbee(r) = min(2, 2r, max(1,r))

 where all limiters are implicitly clipped to be non-negative. A more convenient slope-limited form is psi(r) =
 phi(r)(r+1)/2 in which the reconstructed interface values are

 u(v) = u[0] + phi(r) (grad u)[0] v

 where v is the vector from centroid to quadrature point. In these variables, the usual limiters become

 phi_minmod(r) = 2 min(1/(1+r),r/(1+r))   phi_superbee(r) = 2 min(2/(1+r), 2r/(1+r), max(1,r)/(1+r))

 For a nicer symmetric formulation, rewrite in terms of

 f = (u[0] - u[-1]) / (u[1] - u[-1])

 where r(f) = f/(1-f). Not that r(1-f) = (1-f)/f = 1/r(f) so the symmetry condition

 phi(r) = phi(1/r)

 becomes

 w(f) = w(1-f).

 The limiters below implement this final form w(f). The reference methods are

 w_minmod(f) = 2 min(f,(1-f))             w_superbee(r) = 4 min((1-f), f)
.ve

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterSetType()`, `PetscLimiterCreate()`
@*/
PetscErrorCode PetscLimiterLimit(PetscLimiter lim, PetscReal flim, PetscReal *phi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidRealPointer(phi, 3);
  PetscUseTypeMethod(lim, limit, flim, phi);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_Sin(PetscLimiter lim)
{
  PetscLimiter_Sin *l = (PetscLimiter_Sin *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Sin_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Sin Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Sin(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_Sin_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_Sin(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscSinReal(PETSC_PI * PetscMax(0, PetscMin(f, 1)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_Sin(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Sin;
  lim->ops->destroy = PetscLimiterDestroy_Sin;
  lim->ops->limit   = PetscLimiterLimit_Sin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERSIN = "sin" - A `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Sin(PetscLimiter lim)
{
  PetscLimiter_Sin *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_Sin(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_Zero(PetscLimiter lim)
{
  PetscLimiter_Zero *l = (PetscLimiter_Zero *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Zero_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Zero Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Zero(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_Zero_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_Zero(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_Zero(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Zero;
  lim->ops->destroy = PetscLimiterDestroy_Zero;
  lim->ops->limit   = PetscLimiterLimit_Zero;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERZERO = "zero" - A simple `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Zero(PetscLimiter lim)
{
  PetscLimiter_Zero *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_Zero(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_None(PetscLimiter lim)
{
  PetscLimiter_None *l = (PetscLimiter_None *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_None_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "None Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_None(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_None_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_None(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_None(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_None;
  lim->ops->destroy = PetscLimiterDestroy_None;
  lim->ops->limit   = PetscLimiterLimit_None;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERNONE = "none" - A trivial `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_None(PetscLimiter lim)
{
  PetscLimiter_None *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_None(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_Minmod(PetscLimiter lim)
{
  PetscLimiter_Minmod *l = (PetscLimiter_Minmod *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Minmod_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Minmod Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Minmod(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_Minmod_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_Minmod(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 2 * PetscMax(0, PetscMin(f, 1 - f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_Minmod(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Minmod;
  lim->ops->destroy = PetscLimiterDestroy_Minmod;
  lim->ops->limit   = PetscLimiterLimit_Minmod;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERMINMOD = "minmod" - A `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Minmod(PetscLimiter lim)
{
  PetscLimiter_Minmod *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_Minmod(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_VanLeer(PetscLimiter lim)
{
  PetscLimiter_VanLeer *l = (PetscLimiter_VanLeer *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_VanLeer_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Van Leer Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_VanLeer(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_VanLeer_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_VanLeer(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 4 * f * (1 - f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_VanLeer(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_VanLeer;
  lim->ops->destroy = PetscLimiterDestroy_VanLeer;
  lim->ops->limit   = PetscLimiterLimit_VanLeer;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERVANLEER = "vanleer" - A `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanLeer(PetscLimiter lim)
{
  PetscLimiter_VanLeer *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_VanLeer(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_VanAlbada(PetscLimiter lim)
{
  PetscLimiter_VanAlbada *l = (PetscLimiter_VanAlbada *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_VanAlbada_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Van Albada Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_VanAlbada(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_VanAlbada_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_VanAlbada(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 2 * f * (1 - f) / (PetscSqr(f) + PetscSqr(1 - f)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_VanAlbada(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_VanAlbada;
  lim->ops->destroy = PetscLimiterDestroy_VanAlbada;
  lim->ops->limit   = PetscLimiterLimit_VanAlbada;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERVANALBADA = "vanalbada" - A PetscLimiter implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanAlbada(PetscLimiter lim)
{
  PetscLimiter_VanAlbada *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_VanAlbada(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_Superbee(PetscLimiter lim)
{
  PetscLimiter_Superbee *l = (PetscLimiter_Superbee *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Superbee_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Superbee Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_Superbee(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_Superbee_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterLimit_Superbee(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 4 * PetscMax(0, PetscMin(f, 1 - f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_Superbee(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Superbee;
  lim->ops->destroy = PetscLimiterDestroy_Superbee;
  lim->ops->limit   = PetscLimiterLimit_Superbee;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERSUPERBEE = "superbee" - A `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Superbee(PetscLimiter lim)
{
  PetscLimiter_Superbee *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_Superbee(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterDestroy_MC(PetscLimiter lim)
{
  PetscLimiter_MC *l = (PetscLimiter_MC *)lim->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_MC_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MC Slope Limiter:\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterView_MC(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscLimiterView_MC_Ascii(lim, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* aka Barth-Jespersen */
static PetscErrorCode PetscLimiterLimit_MC(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMin(1, 4 * PetscMax(0, PetscMin(f, 1 - f)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLimiterInitialize_MC(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_MC;
  lim->ops->destroy = PetscLimiterDestroy_MC;
  lim->ops->limit   = PetscLimiterLimit_MC;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLIMITERMC = "mc" - A `PetscLimiter` implementation

  Level: intermediate

.seealso: `PetscLimiter`, `PetscLimiterType`, `PetscLimiterCreate()`, `PetscLimiterSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_MC(PetscLimiter lim)
{
  PetscLimiter_MC *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscCall(PetscNew(&l));
  lim->data = l;

  PetscCall(PetscLimiterInitialize_MC(lim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscClassId PETSCFV_CLASSID = 0;

PetscFunctionList PetscFVList              = NULL;
PetscBool         PetscFVRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscFVRegister - Adds a new `PetscFV` implementation

  Not Collective

  Input Parameters:
+ sname - The name of a new user-defined creation routine
- function - The creation routine itself

  Sample usage:
.vb
    PetscFVRegister("my_fv", MyPetscFVCreate);
.ve

  Then, your PetscFV type can be chosen with the procedural interface via
.vb
    PetscFVCreate(MPI_Comm, PetscFV *);
    PetscFVSetType(PetscFV, "my_fv");
.ve
   or at runtime via the option
.vb
    -petscfv_type my_fv
.ve

  Level: advanced

  Note:
  `PetscFVRegister()` may be called multiple times to add several user-defined PetscFVs

.seealso: `PetscFV`, `PetscFVType`, `PetscFVRegisterAll()`, `PetscFVRegisterDestroy()`
@*/
PetscErrorCode PetscFVRegister(const char sname[], PetscErrorCode (*function)(PetscFV))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscFVList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVSetType - Builds a particular `PetscFV`

  Collective

  Input Parameters:
+ fvm  - The `PetscFV` object
- name - The type of FVM space

  Options Database Key:
. -petscfv_type <type> - Sets the `PetscFVType`; use -help for a list of available types

  Level: intermediate

.seealso: `PetscFV`, `PetscFVType`, `PetscFVGetType()`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVSetType(PetscFV fvm, PetscFVType name)
{
  PetscErrorCode (*r)(PetscFV);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)fvm, name, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFVRegisterAll());
  PetscCall(PetscFunctionListFind(PetscFVList, name, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)fvm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscFV type: %s", name);

  PetscTryTypeMethod(fvm, destroy);
  fvm->ops->destroy = NULL;

  PetscCall((*r)(fvm));
  PetscCall(PetscObjectChangeTypeName((PetscObject)fvm, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVGetType - Gets the `PetscFVType` (as a string) from a `PetscFV`.

  Not Collective

  Input Parameter:
. fvm  - The `PetscFV`

  Output Parameter:
. name - The `PetscFVType` name

  Level: intermediate

.seealso: `PetscFV`, `PetscFVType`, `PetscFVSetType()`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVGetType(PetscFV fvm, PetscFVType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(name, 2);
  PetscCall(PetscFVRegisterAll());
  *name = ((PetscObject)fvm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscFVViewFromOptions - View a `PetscFV` based on values in the options database

   Collective

   Input Parameters:
+  A - the `PetscFV` object
.  obj - Optional object that provides the options prefix
-  name - command line option name

   Level: intermediate

.seealso: `PetscFV`, `PetscFVView()`, `PetscObjectViewFromOptions()`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVViewFromOptions(PetscFV A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCFV_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVView - Views a `PetscFV`

  Collective

  Input Parameters:
+ fvm - the `PetscFV` object to view
- v   - the viewer

  Level: beginner

.seealso: `PetscFV`, `PetscViewer`, `PetscFVDestroy()`
@*/
PetscErrorCode PetscFVView(PetscFV fvm, PetscViewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)fvm), &v));
  PetscTryTypeMethod(fvm, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetFromOptions - sets parameters in a `PetscFV` from the options database

  Collective

  Input Parameter:
. fvm - the `PetscFV` object to set options for

  Options Database Key:
. -petscfv_compute_gradients <bool> - Determines whether cell gradients are calculated

  Level: intermediate

.seealso: `PetscFV`, `PetscFVView()`
@*/
PetscErrorCode PetscFVSetFromOptions(PetscFV fvm)
{
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (!((PetscObject)fvm)->type_name) defaultType = PETSCFVUPWIND;
  else defaultType = ((PetscObject)fvm)->type_name;
  PetscCall(PetscFVRegisterAll());

  PetscObjectOptionsBegin((PetscObject)fvm);
  PetscCall(PetscOptionsFList("-petscfv_type", "Finite volume discretization", "PetscFVSetType", PetscFVList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(PetscFVSetType(fvm, name));
  } else if (!((PetscObject)fvm)->type_name) {
    PetscCall(PetscFVSetType(fvm, defaultType));
  }
  PetscCall(PetscOptionsBool("-petscfv_compute_gradients", "Compute cell gradients", "PetscFVSetComputeGradients", fvm->computeGradients, &fvm->computeGradients, NULL));
  PetscTryTypeMethod(fvm, setfromoptions);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)fvm, PetscOptionsObject));
  PetscCall(PetscLimiterSetFromOptions(fvm->limiter));
  PetscOptionsEnd();
  PetscCall(PetscFVViewFromOptions(fvm, NULL, "-petscfv_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetUp - Setup the data structures for the `PetscFV` based on the `PetscFVType` provided by `PetscFVSetType()`

  Collective

  Input Parameter:
. fvm - the `PetscFV` object to setup

  Level: intermediate

.seealso: `PetscFV`, `PetscFVView()`, `PetscFVDestroy()`
@*/
PetscErrorCode PetscFVSetUp(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscLimiterSetUp(fvm->limiter));
  PetscTryTypeMethod(fvm, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVDestroy - Destroys a `PetscFV` object

  Collective

  Input Parameter:
. fvm - the `PetscFV` object to destroy

  Level: beginner

.seealso: `PetscFV`, `PetscFVCreate()`, `PetscFVView()`
@*/
PetscErrorCode PetscFVDestroy(PetscFV *fvm)
{
  PetscInt i;

  PetscFunctionBegin;
  if (!*fvm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*fvm), PETSCFV_CLASSID, 1);

  if (--((PetscObject)(*fvm))->refct > 0) {
    *fvm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ((PetscObject)(*fvm))->refct = 0;

  for (i = 0; i < (*fvm)->numComponents; i++) PetscCall(PetscFree((*fvm)->componentNames[i]));
  PetscCall(PetscFree((*fvm)->componentNames));
  PetscCall(PetscLimiterDestroy(&(*fvm)->limiter));
  PetscCall(PetscDualSpaceDestroy(&(*fvm)->dualSpace));
  PetscCall(PetscFree((*fvm)->fluxWork));
  PetscCall(PetscQuadratureDestroy(&(*fvm)->quadrature));
  PetscCall(PetscTabulationDestroy(&(*fvm)->T));

  PetscTryTypeMethod((*fvm), destroy);
  PetscCall(PetscHeaderDestroy(fvm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVCreate - Creates an empty `PetscFV` object. The type can then be set with `PetscFVSetType()`.

  Collective

  Input Parameter:
. comm - The communicator for the `PetscFV` object

  Output Parameter:
. fvm - The `PetscFV` object

  Level: beginner

.seealso: `PetscFVSet`, `PetscFVSetType()`, `PETSCFVUPWIND`, `PetscFVDestroy()`
@*/
PetscErrorCode PetscFVCreate(MPI_Comm comm, PetscFV *fvm)
{
  PetscFV f;

  PetscFunctionBegin;
  PetscValidPointer(fvm, 2);
  *fvm = NULL;
  PetscCall(PetscFVInitializePackage());

  PetscCall(PetscHeaderCreate(f, PETSCFV_CLASSID, "PetscFV", "Finite Volume", "PetscFV", comm, PetscFVDestroy, PetscFVView));
  PetscCall(PetscMemzero(f->ops, sizeof(struct _PetscFVOps)));

  PetscCall(PetscLimiterCreate(comm, &f->limiter));
  f->numComponents    = 1;
  f->dim              = 0;
  f->computeGradients = PETSC_FALSE;
  f->fluxWork         = NULL;
  PetscCall(PetscCalloc1(f->numComponents, &f->componentNames));

  *fvm = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetLimiter - Set the `PetscLimiter` to the `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
- lim - The `PetscLimiter`

  Level: intermediate

.seealso: `PetscFV`, `PetscLimiter`, `PetscFVGetLimiter()`
@*/
PetscErrorCode PetscFVSetLimiter(PetscFV fvm, PetscLimiter lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 2);
  PetscCall(PetscLimiterDestroy(&fvm->limiter));
  PetscCall(PetscObjectReference((PetscObject)lim));
  fvm->limiter = lim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetLimiter - Get the `PetscLimiter` object from the `PetscFV`

  Not Collective

  Input Parameter:
. fvm - the `PetscFV` object

  Output Parameter:
. lim - The `PetscLimiter`

  Level: intermediate

.seealso: `PetscFV`, `PetscLimiter`, `PetscFVSetLimiter()`
@*/
PetscErrorCode PetscFVGetLimiter(PetscFV fvm, PetscLimiter *lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(lim, 2);
  *lim = fvm->limiter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetNumComponents - Set the number of field components in a `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
- comp - The number of components

  Level: intermediate

.seealso: `PetscFV`, `PetscFVGetNumComponents()`
@*/
PetscErrorCode PetscFVSetNumComponents(PetscFV fvm, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (fvm->numComponents != comp) {
    PetscInt i;

    for (i = 0; i < fvm->numComponents; i++) PetscCall(PetscFree(fvm->componentNames[i]));
    PetscCall(PetscFree(fvm->componentNames));
    PetscCall(PetscCalloc1(comp, &fvm->componentNames));
  }
  fvm->numComponents = comp;
  PetscCall(PetscFree(fvm->fluxWork));
  PetscCall(PetscMalloc1(comp, &fvm->fluxWork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetNumComponents - Get the number of field components in a `PetscFV`

  Not Collective

  Input Parameter:
. fvm - the `PetscFV` object

  Output Parameter:
, comp - The number of components

  Level: intermediate

.seealso: `PetscFV`, `PetscFVSetNumComponents()`, `PetscFVSetComponentName()`
@*/
PetscErrorCode PetscFVGetNumComponents(PetscFV fvm, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidIntPointer(comp, 2);
  *comp = fvm->numComponents;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVSetComponentName - Set the name of a component (used in output and viewing) in a `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
. comp - the component number
- name - the component name

  Level: intermediate

.seealso: `PetscFV`, `PetscFVGetComponentName()`
@*/
PetscErrorCode PetscFVSetComponentName(PetscFV fvm, PetscInt comp, const char *name)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(fvm->componentNames[comp]));
  PetscCall(PetscStrallocpy(name, &fvm->componentNames[comp]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVGetComponentName - Get the name of a component (used in output and viewing) in a `PetscFV`

  Logically Collective
  Input Parameters:
+ fvm - the `PetscFV` object
- comp - the component number

  Output Parameter:
. name - the component name

  Level: intermediate

.seealso: `PetscFV`, `PetscFVSetComponentName()`
@*/
PetscErrorCode PetscFVGetComponentName(PetscFV fvm, PetscInt comp, const char **name)
{
  PetscFunctionBegin;
  *name = fvm->componentNames[comp];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetSpatialDimension - Set the spatial dimension of a `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
- dim - The spatial dimension

  Level: intermediate

.seealso: `PetscFV`, ``PetscFVGetSpatialDimension()`
@*/
PetscErrorCode PetscFVSetSpatialDimension(PetscFV fvm, PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->dim = dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetSpatialDimension - Get the spatial dimension of a `PetscFV`

  Not Collective

  Input Parameter:
. fvm - the `PetscFV` object

  Output Parameter:
. dim - The spatial dimension

  Level: intermediate

.seealso: `PetscFV`, `PetscFVSetSpatialDimension()`
@*/
PetscErrorCode PetscFVGetSpatialDimension(PetscFV fvm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  *dim = fvm->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
 PetscFVSetComputeGradients - Toggle computation of cell gradients on a `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
- computeGradients - Flag to compute cell gradients

  Level: intermediate

.seealso: `PetscFV`, `PetscFVGetComputeGradients()`
@*/
PetscErrorCode PetscFVSetComputeGradients(PetscFV fvm, PetscBool computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->computeGradients = computeGradients;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetComputeGradients - Return flag for computation of cell gradients on a `PetscFV`

  Not Collective

  Input Parameter:
. fvm - the `PetscFV` object

  Output Parameter:
. computeGradients - Flag to compute cell gradients

  Level: intermediate

.seealso: `PetscFV`, `PetscFVSetComputeGradients()`
@*/
PetscErrorCode PetscFVGetComputeGradients(PetscFV fvm, PetscBool *computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidBoolPointer(computeGradients, 2);
  *computeGradients = fvm->computeGradients;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetQuadrature - Set the `PetscQuadrature` object for a `PetscFV`

  Logically Collective

  Input Parameters:
+ fvm - the `PetscFV` object
- q - The `PetscQuadrature`

  Level: intermediate

.seealso: `PetscQuadrature`, `PetscFV`, `PetscFVGetQuadrature()`
@*/
PetscErrorCode PetscFVSetQuadrature(PetscFV fvm, PetscQuadrature q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscQuadratureDestroy(&fvm->quadrature));
  PetscCall(PetscObjectReference((PetscObject)q));
  fvm->quadrature = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetQuadrature - Get the `PetscQuadrature` from a `PetscFV`

  Not Collective

  Input Parameter:
. fvm - the `PetscFV` object

  Output Parameter:
. lim - The `PetscQuadrature`

  Level: intermediate

.seealso: `PetscQuadrature`, `PetscFV`, `PetscFVSetQuadrature()`
@*/
PetscErrorCode PetscFVGetQuadrature(PetscFV fvm, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(q, 2);
  if (!fvm->quadrature) {
    /* Create default 1-point quadrature */
    PetscReal *points, *weights;

    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &fvm->quadrature));
    PetscCall(PetscCalloc1(fvm->dim, &points));
    PetscCall(PetscMalloc1(1, &weights));
    weights[0] = 1.0;
    PetscCall(PetscQuadratureSetData(fvm->quadrature, fvm->dim, 1, 1, points, weights));
  }
  *q = fvm->quadrature;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVGetDualSpace - Returns the `PetscDualSpace` used to define the inner product on a `PetscFV`

  Not Collective

  Input Parameter:
. fvm - The `PetscFV` object

  Output Parameter:
. sp - The `PetscDualSpace` object

  Level: intermediate

  Developer Note:
  There is overlap between the methods of `PetscFE` and `PetscFV`, they should probably share a common parent class

.seealso: `PetscDualSpace`, `PetscFV`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVGetDualSpace(PetscFV fvm, PetscDualSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(sp, 2);
  if (!fvm->dualSpace) {
    DM       K;
    PetscInt dim, Nc, c;

    PetscCall(PetscFVGetSpatialDimension(fvm, &dim));
    PetscCall(PetscFVGetNumComponents(fvm, &Nc));
    PetscCall(PetscDualSpaceCreate(PetscObjectComm((PetscObject)fvm), &fvm->dualSpace));
    PetscCall(PetscDualSpaceSetType(fvm->dualSpace, PETSCDUALSPACESIMPLE));
    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, PETSC_FALSE), &K));
    PetscCall(PetscDualSpaceSetNumComponents(fvm->dualSpace, Nc));
    PetscCall(PetscDualSpaceSetDM(fvm->dualSpace, K));
    PetscCall(DMDestroy(&K));
    PetscCall(PetscDualSpaceSimpleSetDimension(fvm->dualSpace, Nc));
    /* Should we be using PetscFVGetQuadrature() here? */
    for (c = 0; c < Nc; ++c) {
      PetscQuadrature qc;
      PetscReal      *points, *weights;

      PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &qc));
      PetscCall(PetscCalloc1(dim, &points));
      PetscCall(PetscCalloc1(Nc, &weights));
      weights[c] = 1.0;
      PetscCall(PetscQuadratureSetData(qc, dim, Nc, 1, points, weights));
      PetscCall(PetscDualSpaceSimpleSetFunctional(fvm->dualSpace, c, qc));
      PetscCall(PetscQuadratureDestroy(&qc));
    }
    PetscCall(PetscDualSpaceSetUp(fvm->dualSpace));
  }
  *sp = fvm->dualSpace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVSetDualSpace - Sets the `PetscDualSpace` used to define the inner product

  Not Collective

  Input Parameters:
+ fvm - The `PetscFV` object
- sp  - The `PetscDualSpace` object

  Level: intermediate

  Note:
  A simple dual space is provided automatically, and the user typically will not need to override it.

.seealso: `PetscDualSpace`, `PetscFV`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVSetDualSpace(PetscFV fvm, PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  PetscCall(PetscDualSpaceDestroy(&fvm->dualSpace));
  fvm->dualSpace = sp;
  PetscCall(PetscObjectReference((PetscObject)fvm->dualSpace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVGetCellTabulation - Returns the tabulation of the basis functions at the quadrature points

  Not Collective

  Input Parameter:
. fvm - The `PetscFV` object

  Output Parameter:
. T - The basis function values and derivatives at quadrature points

  Level: intermediate

  Note:
.vb
  T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
  T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
  T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e
.ve

.seealso: `PetscFV`, `PetscTabulation`, `PetscFEGetCellTabulation()`, `PetscFVCreateTabulation()`, `PetscFVGetQuadrature()`, `PetscQuadratureGetData()`
@*/
PetscErrorCode PetscFVGetCellTabulation(PetscFV fvm, PetscTabulation *T)
{
  PetscInt         npoints;
  const PetscReal *points;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(T, 2);
  PetscCall(PetscQuadratureGetData(fvm->quadrature, NULL, NULL, &npoints, &points, NULL));
  if (!fvm->T) PetscCall(PetscFVCreateTabulation(fvm, 1, npoints, points, 1, &fvm->T));
  *T = fvm->T;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVCreateTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not Collective

  Input Parameters:
+ fvm     - The `PetscFV` object
. nrepl   - The number of replicas
. npoints - The number of tabulation points in a replica
. points  - The tabulation point coordinates
- K       - The order of derivative to tabulate

  Output Parameter:
. T - The basis function values and derivative at tabulation points

  Level: intermediate

  Note:
.vb
  T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
  T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
  T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e
.ve

.seealso: `PetscFV`, `PetscTabulation`, `PetscFECreateTabulation()`, `PetscTabulationDestroy()`, `PetscFEGetCellTabulation()`
@*/
PetscErrorCode PetscFVCreateTabulation(PetscFV fvm, PetscInt nrepl, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation *T)
{
  PetscInt pdim = 1; /* Dimension of approximation space P */
  PetscInt cdim;     /* Spatial dimension */
  PetscInt Nc;       /* Field components */
  PetscInt k, p, d, c, e;

  PetscFunctionBegin;
  if (!npoints || K < 0) {
    *T = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidRealPointer(points, 4);
  PetscValidPointer(T, 6);
  PetscCall(PetscFVGetSpatialDimension(fvm, &cdim));
  PetscCall(PetscFVGetNumComponents(fvm, &Nc));
  PetscCall(PetscMalloc1(1, T));
  (*T)->K    = !cdim ? 0 : K;
  (*T)->Nr   = nrepl;
  (*T)->Np   = npoints;
  (*T)->Nb   = pdim;
  (*T)->Nc   = Nc;
  (*T)->cdim = cdim;
  PetscCall(PetscMalloc1((*T)->K + 1, &(*T)->T));
  for (k = 0; k <= (*T)->K; ++k) PetscCall(PetscMalloc1(nrepl * npoints * pdim * Nc * PetscPowInt(cdim, k), &(*T)->T[k]));
  if (K >= 0) {
    for (p = 0; p < nrepl * npoints; ++p)
      for (d = 0; d < pdim; ++d)
        for (c = 0; c < Nc; ++c) (*T)->T[0][(p * pdim + d) * Nc + c] = 1.0;
  }
  if (K >= 1) {
    for (p = 0; p < nrepl * npoints; ++p)
      for (d = 0; d < pdim; ++d)
        for (c = 0; c < Nc; ++c)
          for (e = 0; e < cdim; ++e) (*T)->T[1][((p * pdim + d) * Nc + c) * cdim + e] = 0.0;
  }
  if (K >= 2) {
    for (p = 0; p < nrepl * npoints; ++p)
      for (d = 0; d < pdim; ++d)
        for (c = 0; c < Nc; ++c)
          for (e = 0; e < cdim * cdim; ++e) (*T)->T[2][((p * pdim + d) * Nc + c) * cdim * cdim + e] = 0.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVComputeGradient - Compute the gradient reconstruction matrix for a given cell

  Input Parameters:
+ fvm      - The `PetscFV` object
. numFaces - The number of cell faces which are not constrained
- dx       - The vector from the cell centroid to the neighboring cell centroid for each face

  Level: advanced

.seealso: `PetscFV`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVComputeGradient(PetscFV fvm, PetscInt numFaces, PetscScalar dx[], PetscScalar grad[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscTryTypeMethod(fvm, computegradient, numFaces, dx, grad);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVIntegrateRHSFunction - Produce the cell residual vector for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ fvm          - The `PetscFV` object for the field being integrated
. prob         - The `PetscDS` specifying the discretizations and continuum functions
. field        - The field being integrated
. Nf           - The number of faces in the chunk
. fgeom        - The face geometry for each face in the chunk
. neighborVol  - The volume for each pair of cells in the chunk
. uL           - The state from the cell on the left
- uR           - The state from the cell on the right

  Output Parameters:
+ fluxL        - the left fluxes for each face
- fluxR        - the right fluxes for each face

  Level: developer

.seealso: `PetscFV`, `PetscDS`, `PetscFVFaceGeom`, `PetscFVCreate()`
@*/
PetscErrorCode PetscFVIntegrateRHSFunction(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol, PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscTryTypeMethod(fvm, integraterhsfunction, prob, field, Nf, fgeom, neighborVol, uL, uR, fluxL, fluxR);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVRefine - Create a "refined" `PetscFV` object that refines the reference cell into smaller copies. This is typically used
  to precondition a higher order method with a lower order method on a refined mesh having the same number of dofs (but more
  sparsity). It is also used to create an interpolation between regularly refined meshes.

  Input Parameter:
. fv - The initial `PetscFV`

  Output Parameter:
. fvRef - The refined `PetscFV`

  Level: advanced

.seealso: `PetscFV`, `PetscFVType`, `PetscFVCreate()`, `PetscFVSetType()`
@*/
PetscErrorCode PetscFVRefine(PetscFV fv, PetscFV *fvRef)
{
  PetscDualSpace  Q, Qref;
  DM              K, Kref;
  PetscQuadrature q, qref;
  DMPolytopeType  ct;
  DMPlexTransform tr;
  PetscReal      *v0;
  PetscReal      *jac, *invjac;
  PetscInt        numComp, numSubelements, s;

  PetscFunctionBegin;
  PetscCall(PetscFVGetDualSpace(fv, &Q));
  PetscCall(PetscFVGetQuadrature(fv, &q));
  PetscCall(PetscDualSpaceGetDM(Q, &K));
  /* Create dual space */
  PetscCall(PetscDualSpaceDuplicate(Q, &Qref));
  PetscCall(DMRefine(K, PetscObjectComm((PetscObject)fv), &Kref));
  PetscCall(PetscDualSpaceSetDM(Qref, Kref));
  PetscCall(DMDestroy(&Kref));
  PetscCall(PetscDualSpaceSetUp(Qref));
  /* Create volume */
  PetscCall(PetscFVCreate(PetscObjectComm((PetscObject)fv), fvRef));
  PetscCall(PetscFVSetDualSpace(*fvRef, Qref));
  PetscCall(PetscFVGetNumComponents(fv, &numComp));
  PetscCall(PetscFVSetNumComponents(*fvRef, numComp));
  PetscCall(PetscFVSetUp(*fvRef));
  /* Create quadrature */
  PetscCall(DMPlexGetCellType(K, 0, &ct));
  PetscCall(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
  PetscCall(DMPlexRefineRegularGetAffineTransforms(tr, ct, &numSubelements, &v0, &jac, &invjac));
  PetscCall(PetscQuadratureExpandComposite(q, numSubelements, v0, jac, &qref));
  PetscCall(PetscDualSpaceSimpleSetDimension(Qref, numSubelements));
  for (s = 0; s < numSubelements; ++s) {
    PetscQuadrature  qs;
    const PetscReal *points, *weights;
    PetscReal       *p, *w;
    PetscInt         dim, Nc, npoints, np;

    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &qs));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, &weights));
    np = npoints / numSubelements;
    PetscCall(PetscMalloc1(np * dim, &p));
    PetscCall(PetscMalloc1(np * Nc, &w));
    PetscCall(PetscArraycpy(p, &points[s * np * dim], np * dim));
    PetscCall(PetscArraycpy(w, &weights[s * np * Nc], np * Nc));
    PetscCall(PetscQuadratureSetData(qs, dim, Nc, np, p, w));
    PetscCall(PetscDualSpaceSimpleSetFunctional(Qref, s, qs));
    PetscCall(PetscQuadratureDestroy(&qs));
  }
  PetscCall(PetscFVSetQuadrature(*fvRef, qref));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(PetscQuadratureDestroy(&qref));
  PetscCall(PetscDualSpaceDestroy(&Qref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVDestroy_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b = (PetscFV_Upwind *)fvm->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVView_Upwind_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc, c;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscFVGetNumComponents(fv, &Nc));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Upwind Finite Volume:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  num components: %" PetscInt_FMT "\n", Nc));
  for (c = 0; c < Nc; c++) {
    if (fv->componentNames[c]) PetscCall(PetscViewerASCIIPrintf(viewer, "    component %" PetscInt_FMT ": %s\n", c, fv->componentNames[c]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVView_Upwind(PetscFV fv, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscFVView_Upwind_Ascii(fv, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVSetUp_Upwind(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode PetscFVIntegrateRHSFunction_Upwind(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol, PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  void (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  void              *rctx;
  PetscScalar       *flux = fvm->fluxWork;
  const PetscScalar *constants;
  PetscInt           dim, numConstants, pdim, totDim, Nc, off, f, d;

  PetscFunctionBegin;
  PetscCall(PetscDSGetTotalComponents(prob, &Nc));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSGetFieldOffset(prob, field, &off));
  PetscCall(PetscDSGetRiemannSolver(prob, field, &riemann));
  PetscCall(PetscDSGetContext(prob, field, &rctx));
  PetscCall(PetscDSGetConstants(prob, &numConstants, &constants));
  PetscCall(PetscFVGetSpatialDimension(fvm, &dim));
  PetscCall(PetscFVGetNumComponents(fvm, &pdim));
  for (f = 0; f < Nf; ++f) {
    (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f * Nc], &uR[f * Nc], numConstants, constants, flux, rctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f * totDim + off + d] = flux[d] / neighborVol[f * 2 + 0];
      fluxR[f * totDim + off + d] = flux[d] / neighborVol[f * 2 + 1];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVInitialize_Upwind(PetscFV fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions       = NULL;
  fvm->ops->setup                = PetscFVSetUp_Upwind;
  fvm->ops->view                 = PetscFVView_Upwind;
  fvm->ops->destroy              = PetscFVDestroy_Upwind;
  fvm->ops->integraterhsfunction = PetscFVIntegrateRHSFunction_Upwind;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCFVUPWIND = "upwind" - A `PetscFV` implementation

  Level: intermediate

.seealso: `PetscFV`, `PetscFVType`, `PetscFVCreate()`, `PetscFVSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscNew(&b));
  fvm->data = b;

  PetscCall(PetscFVInitialize_Upwind(fvm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscblaslapack.h>

static PetscErrorCode PetscFVDestroy_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *ls = (PetscFV_LeastSquares *)fvm->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)fvm, "PetscFVLeastSquaresSetMaxFaces_C", NULL));
  PetscCall(PetscFree4(ls->B, ls->Binv, ls->tau, ls->work));
  PetscCall(PetscFree(ls));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVView_LeastSquares_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc, c;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscFVGetNumComponents(fv, &Nc));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Finite Volume with Least Squares Reconstruction:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  num components: %" PetscInt_FMT "\n", Nc));
  for (c = 0; c < Nc; c++) {
    if (fv->componentNames[c]) PetscCall(PetscViewerASCIIPrintf(viewer, "    component %" PetscInt_FMT ": %s\n", c, fv->componentNames[c]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVView_LeastSquares(PetscFV fv, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscFVView_LeastSquares_Ascii(fv, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVSetUp_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Overwrites A. Can only handle full-rank problems with m>=n */
static PetscErrorCode PetscFVLeastSquaresPseudoInverse_Static(PetscInt m, PetscInt mstride, PetscInt n, PetscScalar *A, PetscScalar *Ainv, PetscScalar *tau, PetscInt worksize, PetscScalar *work)
{
  PetscBool    debug = PETSC_FALSE;
  PetscBLASInt M, N, K, lda, ldb, ldwork, info;
  PetscScalar *R, *Q, *Aback, Alpha;

  PetscFunctionBegin;
  if (debug) {
    PetscCall(PetscMalloc1(m * n, &Aback));
    PetscCall(PetscArraycpy(Aback, A, m * n));
  }

  PetscCall(PetscBLASIntCast(m, &M));
  PetscCall(PetscBLASIntCast(n, &N));
  PetscCall(PetscBLASIntCast(mstride, &lda));
  PetscCall(PetscBLASIntCast(worksize, &ldwork));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&M, &N, A, &lda, tau, work, &ldwork, &info));
  PetscCall(PetscFPTrapPop());
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q = Ainv;
  PetscCall(PetscArraycpy(Q, A, mstride * n));
  K = N; /* full rank */
  PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&M, &N, &K, Q, &lda, tau, work, &ldwork, &info));
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb   = lda;
  BLAStrsm_("Right", "Upper", "ConjugateTranspose", "NotUnitTriangular", &M, &N, &Alpha, R, &lda, Q, &ldb);
  /* Ainv is Q, overwritten with inverse */

  if (debug) { /* Check that pseudo-inverse worked */
    PetscScalar  Beta = 0.0;
    PetscBLASInt ldc;
    K   = N;
    ldc = N;
    BLASgemm_("ConjugateTranspose", "Normal", &N, &K, &M, &Alpha, Ainv, &lda, Aback, &ldb, &Beta, work, &ldc);
    PetscCall(PetscScalarView(n * n, work, PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscFree(Aback));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Overwrites A. Can handle degenerate problems and m<n. */
static PetscErrorCode PetscFVLeastSquaresPseudoInverseSVD_Static(PetscInt m, PetscInt mstride, PetscInt n, PetscScalar *A, PetscScalar *Ainv, PetscScalar *tau, PetscInt worksize, PetscScalar *work)
{
  PetscScalar *Brhs;
  PetscScalar *tmpwork;
  PetscReal    rcond;
#if defined(PETSC_USE_COMPLEX)
  PetscInt   rworkSize;
  PetscReal *rwork;
#endif
  PetscInt     i, j, maxmn;
  PetscBLASInt M, N, lda, ldb, ldwork;
  PetscBLASInt nrhs, irank, info;

  PetscFunctionBegin;
  /* initialize to identity */
  tmpwork = work;
  Brhs    = Ainv;
  maxmn   = PetscMax(m, n);
  for (j = 0; j < maxmn; j++) {
    for (i = 0; i < maxmn; i++) Brhs[i + j * maxmn] = 1.0 * (i == j);
  }

  PetscCall(PetscBLASIntCast(m, &M));
  PetscCall(PetscBLASIntCast(n, &N));
  PetscCall(PetscBLASIntCast(mstride, &lda));
  PetscCall(PetscBLASIntCast(maxmn, &ldb));
  PetscCall(PetscBLASIntCast(worksize, &ldwork));
  rcond = -1;
  nrhs  = M;
#if defined(PETSC_USE_COMPLEX)
  rworkSize = 5 * PetscMin(M, N);
  PetscCall(PetscMalloc1(rworkSize, &rwork));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&M, &N, &nrhs, A, &lda, Brhs, &ldb, (PetscReal *)tau, &rcond, &irank, tmpwork, &ldwork, rwork, &info));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscFree(rwork));
#else
  nrhs = M;
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&M, &N, &nrhs, A, &lda, Brhs, &ldb, (PetscReal *)tau, &rcond, &irank, tmpwork, &ldwork, &info));
  PetscCall(PetscFPTrapPop());
#endif
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "xGELSS error");
  /* The following check should be turned into a diagnostic as soon as someone wants to do this intentionally */
  PetscCheck(irank >= PetscMin(M, N), PETSC_COMM_SELF, PETSC_ERR_USER, "Rank deficient least squares fit, indicates an isolated cell with two colinear points");
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode PetscFVLeastSquaresDebugCell_Static(PetscFV fvm, PetscInt cell, DM dm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  PetscReal       grad[2] = {0, 0};
  const PetscInt *faces;
  PetscInt        numFaces, f;

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, cell, &numFaces));
  PetscCall(DMPlexGetCone(dm, cell, &faces));
  for (f = 0; f < numFaces; ++f) {
    const PetscInt *fcells;
    const CellGeom *cg1;
    const FaceGeom *fg;

    PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
    PetscCall(DMPlexPointLocalRead(dmFace, faces[f], fgeom, &fg));
    for (i = 0; i < 2; ++i) {
      PetscScalar du;

      if (fcells[i] == c) continue;
      PetscCall(DMPlexPointLocalRead(dmCell, fcells[i], cgeom, &cg1));
      du   = cg1->centroid[0] + 3*cg1->centroid[1] - (cg->centroid[0] + 3*cg->centroid[1]);
      grad[0] += fg->grad[!i][0] * du;
      grad[1] += fg->grad[!i][1] * du;
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "cell[%d] grad (%g, %g)\n", cell, grad[0], grad[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  PetscFVComputeGradient - Compute the gradient reconstruction matrix for a given cell

  Input Parameters:
+ fvm      - The `PetscFV` object
. numFaces - The number of cell faces which are not constrained
. dx       - The vector from the cell centroid to the neighboring cell centroid for each face

  Level: developer

.seealso: `PetscFV`, `PetscFVCreate()`
*/
static PetscErrorCode PetscFVComputeGradient_LeastSquares(PetscFV fvm, PetscInt numFaces, const PetscScalar dx[], PetscScalar grad[])
{
  PetscFV_LeastSquares *ls       = (PetscFV_LeastSquares *)fvm->data;
  const PetscBool       useSVD   = PETSC_TRUE;
  const PetscInt        maxFaces = ls->maxFaces;
  PetscInt              dim, f, d;

  PetscFunctionBegin;
  if (numFaces > maxFaces) {
    PetscCheck(maxFaces >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Reconstruction has not been initialized, call PetscFVLeastSquaresSetMaxFaces()");
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input faces %" PetscInt_FMT " > %" PetscInt_FMT " maxfaces", numFaces, maxFaces);
  }
  PetscCall(PetscFVGetSpatialDimension(fvm, &dim));
  for (f = 0; f < numFaces; ++f) {
    for (d = 0; d < dim; ++d) ls->B[d * maxFaces + f] = dx[f * dim + d];
  }
  /* Overwrites B with garbage, returns Binv in row-major format */
  if (useSVD) {
    PetscInt maxmn = PetscMax(numFaces, dim);
    PetscCall(PetscFVLeastSquaresPseudoInverseSVD_Static(numFaces, maxFaces, dim, ls->B, ls->Binv, ls->tau, ls->workSize, ls->work));
    /* Binv shaped in column-major, coldim=maxmn.*/
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) grad[f * dim + d] = ls->Binv[d + maxmn * f];
    }
  } else {
    PetscCall(PetscFVLeastSquaresPseudoInverse_Static(numFaces, maxFaces, dim, ls->B, ls->Binv, ls->tau, ls->workSize, ls->work));
    /* Binv shaped in row-major, rowdim=maxFaces.*/
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) grad[f * dim + d] = ls->Binv[d * maxFaces + f];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode PetscFVIntegrateRHSFunction_LeastSquares(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol, PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  void (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  void              *rctx;
  PetscScalar       *flux = fvm->fluxWork;
  const PetscScalar *constants;
  PetscInt           dim, numConstants, pdim, Nc, totDim, off, f, d;

  PetscFunctionBegin;
  PetscCall(PetscDSGetTotalComponents(prob, &Nc));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSGetFieldOffset(prob, field, &off));
  PetscCall(PetscDSGetRiemannSolver(prob, field, &riemann));
  PetscCall(PetscDSGetContext(prob, field, &rctx));
  PetscCall(PetscDSGetConstants(prob, &numConstants, &constants));
  PetscCall(PetscFVGetSpatialDimension(fvm, &dim));
  PetscCall(PetscFVGetNumComponents(fvm, &pdim));
  for (f = 0; f < Nf; ++f) {
    (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f * Nc], &uR[f * Nc], numConstants, constants, flux, rctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f * totDim + off + d] = flux[d] / neighborVol[f * 2 + 0];
      fluxR[f * totDim + off + d] = flux[d] / neighborVol[f * 2 + 1];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFVLeastSquaresSetMaxFaces_LS(PetscFV fvm, PetscInt maxFaces)
{
  PetscFV_LeastSquares *ls = (PetscFV_LeastSquares *)fvm->data;
  PetscInt              dim, m, n, nrhs, minmn, maxmn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscFVGetSpatialDimension(fvm, &dim));
  PetscCall(PetscFree4(ls->B, ls->Binv, ls->tau, ls->work));
  ls->maxFaces = maxFaces;
  m            = ls->maxFaces;
  n            = dim;
  nrhs         = ls->maxFaces;
  minmn        = PetscMin(m, n);
  maxmn        = PetscMax(m, n);
  ls->workSize = 3 * minmn + PetscMax(2 * minmn, PetscMax(maxmn, nrhs)); /* required by LAPACK */
  PetscCall(PetscMalloc4(m * n, &ls->B, maxmn * maxmn, &ls->Binv, minmn, &ls->tau, ls->workSize, &ls->work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFVInitialize_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions       = NULL;
  fvm->ops->setup                = PetscFVSetUp_LeastSquares;
  fvm->ops->view                 = PetscFVView_LeastSquares;
  fvm->ops->destroy              = PetscFVDestroy_LeastSquares;
  fvm->ops->computegradient      = PetscFVComputeGradient_LeastSquares;
  fvm->ops->integraterhsfunction = PetscFVIntegrateRHSFunction_LeastSquares;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCFVLEASTSQUARES = "leastsquares" - A `PetscFV` implementation

  Level: intermediate

.seealso: `PetscFV`, `PetscFVType`, `PetscFVCreate()`, `PetscFVSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscCall(PetscNew(&ls));
  fvm->data = ls;

  ls->maxFaces = -1;
  ls->workSize = -1;
  ls->B        = NULL;
  ls->Binv     = NULL;
  ls->tau      = NULL;
  ls->work     = NULL;

  PetscCall(PetscFVSetComputeGradients(fvm, PETSC_TRUE));
  PetscCall(PetscFVInitialize_LeastSquares(fvm));
  PetscCall(PetscObjectComposeFunction((PetscObject)fvm, "PetscFVLeastSquaresSetMaxFaces_C", PetscFVLeastSquaresSetMaxFaces_LS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFVLeastSquaresSetMaxFaces - Set the maximum number of cell faces for gradient reconstruction

  Not Collective

  Input parameters:
+ fvm      - The `PetscFV` object
- maxFaces - The maximum number of cell faces

  Level: intermediate

.seealso: `PetscFV`, `PetscFVCreate()`, `PETSCFVLEASTSQUARES`, `PetscFVComputeGradient()`
@*/
PetscErrorCode PetscFVLeastSquaresSetMaxFaces(PetscFV fvm, PetscInt maxFaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscTryMethod(fvm, "PetscFVLeastSquaresSetMaxFaces_C", (PetscFV, PetscInt), (fvm, maxFaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}
