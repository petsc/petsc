#include <petsc/private/petscfvimpl.h> /*I "petscfv.h" I*/
#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscds.h>

PetscClassId PETSCLIMITER_CLASSID = 0;

PetscFunctionList PetscLimiterList              = NULL;
PetscBool         PetscLimiterRegisterAllCalled = PETSC_FALSE;

PetscBool Limitercite = PETSC_FALSE;
const char LimiterCitation[] = "@article{BergerAftosmisMurman2005,\n"
                               "  title   = {Analysis of slope limiters on irregular grids},\n"
                               "  journal = {AIAA paper},\n"
                               "  author  = {Marsha Berger and Michael J. Aftosmis and Scott M. Murman},\n"
                               "  volume  = {490},\n"
                               "  year    = {2005}\n}\n";

/*@C
  PetscLimiterRegister - Adds a new PetscLimiter implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscLimiterRegister() may be called multiple times to add several user-defined PetscLimiters

  Sample usage:
.vb
    PetscLimiterRegister("my_lim", MyPetscLimiterCreate);
.ve

  Then, your PetscLimiter type can be chosen with the procedural interface via
.vb
    PetscLimiterCreate(MPI_Comm, PetscLimiter *);
    PetscLimiterSetType(PetscLimiter, "my_lim");
.ve
   or at runtime via the option
.vb
    -petsclimiter_type my_lim
.ve

  Level: advanced

.seealso: PetscLimiterRegisterAll(), PetscLimiterRegisterDestroy()

@*/
PetscErrorCode PetscLimiterRegister(const char sname[], PetscErrorCode (*function)(PetscLimiter))
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListAdd(&PetscLimiterList, sname, function));
  PetscFunctionReturn(0);
}

/*@C
  PetscLimiterSetType - Builds a particular PetscLimiter

  Collective on lim

  Input Parameters:
+ lim  - The PetscLimiter object
- name - The kind of limiter

  Options Database Key:
. -petsclimiter_type <type> - Sets the PetscLimiter type; use -help for a list of available types

  Level: intermediate

.seealso: PetscLimiterGetType(), PetscLimiterCreate()
@*/
PetscErrorCode PetscLimiterSetType(PetscLimiter lim, PetscLimiterType name)
{
  PetscErrorCode (*r)(PetscLimiter);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) lim, name, &match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(PetscLimiterRegisterAll());
  CHKERRQ(PetscFunctionListFind(PetscLimiterList, name, &r));
  PetscCheck(r,PetscObjectComm((PetscObject) lim), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscLimiter type: %s", name);

  if (lim->ops->destroy) {
    CHKERRQ((*lim->ops->destroy)(lim));
    lim->ops->destroy = NULL;
  }
  CHKERRQ((*r)(lim));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject) lim, name));
  PetscFunctionReturn(0);
}

/*@C
  PetscLimiterGetType - Gets the PetscLimiter type name (as a string) from the object.

  Not Collective

  Input Parameter:
. lim  - The PetscLimiter

  Output Parameter:
. name - The PetscLimiter type name

  Level: intermediate

.seealso: PetscLimiterSetType(), PetscLimiterCreate()
@*/
PetscErrorCode PetscLimiterGetType(PetscLimiter lim, PetscLimiterType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidPointer(name, 2);
  CHKERRQ(PetscLimiterRegisterAll());
  *name = ((PetscObject) lim)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PetscLimiterViewFromOptions - View from Options

   Collective on PetscLimiter

   Input Parameters:
+  A - the PetscLimiter object to view
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscLimiter, PetscLimiterView, PetscObjectViewFromOptions(), PetscLimiterCreate()
@*/
PetscErrorCode  PetscLimiterViewFromOptions(PetscLimiter A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCLIMITER_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
  PetscLimiterView - Views a PetscLimiter

  Collective on lim

  Input Parameters:
+ lim - the PetscLimiter object to view
- v   - the viewer

  Level: beginner

.seealso: PetscLimiterDestroy()
@*/
PetscErrorCode PetscLimiterView(PetscLimiter lim, PetscViewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (!v) CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) lim), &v));
  if (lim->ops->view) CHKERRQ((*lim->ops->view)(lim, v));
  PetscFunctionReturn(0);
}

/*@
  PetscLimiterSetFromOptions - sets parameters in a PetscLimiter from the options database

  Collective on lim

  Input Parameter:
. lim - the PetscLimiter object to set options for

  Level: intermediate

.seealso: PetscLimiterView()
@*/
PetscErrorCode PetscLimiterSetFromOptions(PetscLimiter lim)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (!((PetscObject) lim)->type_name) defaultType = PETSCLIMITERSIN;
  else                                 defaultType = ((PetscObject) lim)->type_name;
  CHKERRQ(PetscLimiterRegisterAll());

  ierr = PetscObjectOptionsBegin((PetscObject) lim);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-petsclimiter_type", "Finite volume slope limiter", "PetscLimiterSetType", PetscLimiterList, defaultType, name, 256, &flg));
  if (flg) {
    CHKERRQ(PetscLimiterSetType(lim, name));
  } else if (!((PetscObject) lim)->type_name) {
    CHKERRQ(PetscLimiterSetType(lim, defaultType));
  }
  if (lim->ops->setfromoptions) CHKERRQ((*lim->ops->setfromoptions)(lim));
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) lim));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscLimiterViewFromOptions(lim, NULL, "-petsclimiter_view"));
  PetscFunctionReturn(0);
}

/*@C
  PetscLimiterSetUp - Construct data structures for the PetscLimiter

  Collective on lim

  Input Parameter:
. lim - the PetscLimiter object to setup

  Level: intermediate

.seealso: PetscLimiterView(), PetscLimiterDestroy()
@*/
PetscErrorCode PetscLimiterSetUp(PetscLimiter lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (lim->ops->setup) CHKERRQ((*lim->ops->setup)(lim));
  PetscFunctionReturn(0);
}

/*@
  PetscLimiterDestroy - Destroys a PetscLimiter object

  Collective on lim

  Input Parameter:
. lim - the PetscLimiter object to destroy

  Level: beginner

.seealso: PetscLimiterView()
@*/
PetscErrorCode PetscLimiterDestroy(PetscLimiter *lim)
{
  PetscFunctionBegin;
  if (!*lim) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*lim), PETSCLIMITER_CLASSID, 1);

  if (--((PetscObject)(*lim))->refct > 0) {*lim = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*lim))->refct = 0;

  if ((*lim)->ops->destroy) CHKERRQ((*(*lim)->ops->destroy)(*lim));
  CHKERRQ(PetscHeaderDestroy(lim));
  PetscFunctionReturn(0);
}

/*@
  PetscLimiterCreate - Creates an empty PetscLimiter object. The type can then be set with PetscLimiterSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscLimiter object

  Output Parameter:
. lim - The PetscLimiter object

  Level: beginner

.seealso: PetscLimiterSetType(), PETSCLIMITERSIN
@*/
PetscErrorCode PetscLimiterCreate(MPI_Comm comm, PetscLimiter *lim)
{
  PetscLimiter   l;

  PetscFunctionBegin;
  PetscValidPointer(lim, 2);
  CHKERRQ(PetscCitationsRegister(LimiterCitation,&Limitercite));
  *lim = NULL;
  CHKERRQ(PetscFVInitializePackage());

  CHKERRQ(PetscHeaderCreate(l, PETSCLIMITER_CLASSID, "PetscLimiter", "Finite Volume Slope Limiter", "PetscLimiter", comm, PetscLimiterDestroy, PetscLimiterView));

  *lim = l;
  PetscFunctionReturn(0);
}

/*@
  PetscLimiterLimit - Limit the flux

  Input Parameters:
+ lim  - The PetscLimiter
- flim - The input field

  Output Parameter:
. phi  - The limited field

Note: Limiters given in symmetric form following Berger, Aftosmis, and Murman 2005
$ The classical flux-limited formulation is psi(r) where
$
$ r = (u[0] - u[-1]) / (u[1] - u[0])
$
$ The second order TVD region is bounded by
$
$ psi_minmod(r) = min(r,1)      and        psi_superbee(r) = min(2, 2r, max(1,r))
$
$ where all limiters are implicitly clipped to be non-negative. A more convenient slope-limited form is psi(r) =
$ phi(r)(r+1)/2 in which the reconstructed interface values are
$
$ u(v) = u[0] + phi(r) (grad u)[0] v
$
$ where v is the vector from centroid to quadrature point. In these variables, the usual limiters become
$
$ phi_minmod(r) = 2 min(1/(1+r),r/(1+r))   phi_superbee(r) = 2 min(2/(1+r), 2r/(1+r), max(1,r)/(1+r))
$
$ For a nicer symmetric formulation, rewrite in terms of
$
$ f = (u[0] - u[-1]) / (u[1] - u[-1])
$
$ where r(f) = f/(1-f). Not that r(1-f) = (1-f)/f = 1/r(f) so the symmetry condition
$
$ phi(r) = phi(1/r)
$
$ becomes
$
$ w(f) = w(1-f).
$
$ The limiters below implement this final form w(f). The reference methods are
$
$ w_minmod(f) = 2 min(f,(1-f))             w_superbee(r) = 4 min((1-f), f)

  Level: beginner

.seealso: PetscLimiterSetType(), PetscLimiterCreate()
@*/
PetscErrorCode PetscLimiterLimit(PetscLimiter lim, PetscReal flim, PetscReal *phi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidPointer(phi, 3);
  CHKERRQ((*lim->ops->limit)(lim, flim, phi));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_Sin(PetscLimiter lim)
{
  PetscLimiter_Sin *l = (PetscLimiter_Sin *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Sin_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Sin Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Sin(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_Sin_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_Sin(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscSinReal(PETSC_PI*PetscMax(0, PetscMin(f, 1)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_Sin(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Sin;
  lim->ops->destroy = PetscLimiterDestroy_Sin;
  lim->ops->limit   = PetscLimiterLimit_Sin;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERSIN = "sin" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Sin(PetscLimiter lim)
{
  PetscLimiter_Sin *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_Sin(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_Zero(PetscLimiter lim)
{
  PetscLimiter_Zero *l = (PetscLimiter_Zero *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Zero_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Zero Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Zero(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_Zero_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_Zero(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 0.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_Zero(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Zero;
  lim->ops->destroy = PetscLimiterDestroy_Zero;
  lim->ops->limit   = PetscLimiterLimit_Zero;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERZERO = "zero" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Zero(PetscLimiter lim)
{
  PetscLimiter_Zero *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_Zero(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_None(PetscLimiter lim)
{
  PetscLimiter_None *l = (PetscLimiter_None *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_None_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "None Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_None(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_None_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_None(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_None(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_None;
  lim->ops->destroy = PetscLimiterDestroy_None;
  lim->ops->limit   = PetscLimiterLimit_None;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERNONE = "none" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_None(PetscLimiter lim)
{
  PetscLimiter_None *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_None(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_Minmod(PetscLimiter lim)
{
  PetscLimiter_Minmod *l = (PetscLimiter_Minmod *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Minmod_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Minmod Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Minmod(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_Minmod_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_Minmod(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 2*PetscMax(0, PetscMin(f, 1-f));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_Minmod(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Minmod;
  lim->ops->destroy = PetscLimiterDestroy_Minmod;
  lim->ops->limit   = PetscLimiterLimit_Minmod;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERMINMOD = "minmod" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Minmod(PetscLimiter lim)
{
  PetscLimiter_Minmod *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_Minmod(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_VanLeer(PetscLimiter lim)
{
  PetscLimiter_VanLeer *l = (PetscLimiter_VanLeer *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_VanLeer_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Van Leer Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_VanLeer(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_VanLeer_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_VanLeer(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 4*f*(1-f));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_VanLeer(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_VanLeer;
  lim->ops->destroy = PetscLimiterDestroy_VanLeer;
  lim->ops->limit   = PetscLimiterLimit_VanLeer;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERVANLEER = "vanleer" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanLeer(PetscLimiter lim)
{
  PetscLimiter_VanLeer *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_VanLeer(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_VanAlbada(PetscLimiter lim)
{
  PetscLimiter_VanAlbada *l = (PetscLimiter_VanAlbada *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_VanAlbada_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Van Albada Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_VanAlbada(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_VanAlbada_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_VanAlbada(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 2*f*(1-f) / (PetscSqr(f) + PetscSqr(1-f)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_VanAlbada(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_VanAlbada;
  lim->ops->destroy = PetscLimiterDestroy_VanAlbada;
  lim->ops->limit   = PetscLimiterLimit_VanAlbada;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERVANALBADA = "vanalbada" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanAlbada(PetscLimiter lim)
{
  PetscLimiter_VanAlbada *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_VanAlbada(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_Superbee(PetscLimiter lim)
{
  PetscLimiter_Superbee *l = (PetscLimiter_Superbee *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Superbee_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Superbee Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_Superbee(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_Superbee_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterLimit_Superbee(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = 4*PetscMax(0, PetscMin(f, 1-f));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_Superbee(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_Superbee;
  lim->ops->destroy = PetscLimiterDestroy_Superbee;
  lim->ops->limit   = PetscLimiterLimit_Superbee;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERSUPERBEE = "superbee" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Superbee(PetscLimiter lim)
{
  PetscLimiter_Superbee *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_Superbee(lim));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterDestroy_MC(PetscLimiter lim)
{
  PetscLimiter_MC *l = (PetscLimiter_MC *) lim->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_MC_Ascii(PetscLimiter lim, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "MC Slope Limiter:\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterView_MC(PetscLimiter lim, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscLimiterView_MC_Ascii(lim, viewer));
  PetscFunctionReturn(0);
}

/* aka Barth-Jespersen */
static PetscErrorCode PetscLimiterLimit_MC(PetscLimiter lim, PetscReal f, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PetscMin(1, 4*PetscMax(0, PetscMin(f, 1-f)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLimiterInitialize_MC(PetscLimiter lim)
{
  PetscFunctionBegin;
  lim->ops->view    = PetscLimiterView_MC;
  lim->ops->destroy = PetscLimiterDestroy_MC;
  lim->ops->limit   = PetscLimiterLimit_MC;
  PetscFunctionReturn(0);
}

/*MC
  PETSCLIMITERMC = "mc" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_MC(PetscLimiter lim)
{
  PetscLimiter_MC *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  CHKERRQ(PetscNewLog(lim, &l));
  lim->data = l;

  CHKERRQ(PetscLimiterInitialize_MC(lim));
  PetscFunctionReturn(0);
}

PetscClassId PETSCFV_CLASSID = 0;

PetscFunctionList PetscFVList              = NULL;
PetscBool         PetscFVRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscFVRegister - Adds a new PetscFV implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscFVRegister() may be called multiple times to add several user-defined PetscFVs

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

.seealso: PetscFVRegisterAll(), PetscFVRegisterDestroy()

@*/
PetscErrorCode PetscFVRegister(const char sname[], PetscErrorCode (*function)(PetscFV))
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListAdd(&PetscFVList, sname, function));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVSetType - Builds a particular PetscFV

  Collective on fvm

  Input Parameters:
+ fvm  - The PetscFV object
- name - The kind of FVM space

  Options Database Key:
. -petscfv_type <type> - Sets the PetscFV type; use -help for a list of available types

  Level: intermediate

.seealso: PetscFVGetType(), PetscFVCreate()
@*/
PetscErrorCode PetscFVSetType(PetscFV fvm, PetscFVType name)
{
  PetscErrorCode (*r)(PetscFV);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) fvm, name, &match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(PetscFVRegisterAll());
  CHKERRQ(PetscFunctionListFind(PetscFVList, name, &r));
  PetscCheck(r,PetscObjectComm((PetscObject) fvm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscFV type: %s", name);

  if (fvm->ops->destroy) {
    CHKERRQ((*fvm->ops->destroy)(fvm));
    fvm->ops->destroy = NULL;
  }
  CHKERRQ((*r)(fvm));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject) fvm, name));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVGetType - Gets the PetscFV type name (as a string) from the object.

  Not Collective

  Input Parameter:
. fvm  - The PetscFV

  Output Parameter:
. name - The PetscFV type name

  Level: intermediate

.seealso: PetscFVSetType(), PetscFVCreate()
@*/
PetscErrorCode PetscFVGetType(PetscFV fvm, PetscFVType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(name, 2);
  CHKERRQ(PetscFVRegisterAll());
  *name = ((PetscObject) fvm)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PetscFVViewFromOptions - View from Options

   Collective on PetscFV

   Input Parameters:
+  A - the PetscFV object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscFV, PetscFVView, PetscObjectViewFromOptions(), PetscFVCreate()
@*/
PetscErrorCode  PetscFVViewFromOptions(PetscFV A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCFV_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVView - Views a PetscFV

  Collective on fvm

  Input Parameters:
+ fvm - the PetscFV object to view
- v   - the viewer

  Level: beginner

.seealso: PetscFVDestroy()
@*/
PetscErrorCode PetscFVView(PetscFV fvm, PetscViewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (!v) CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) fvm), &v));
  if (fvm->ops->view) CHKERRQ((*fvm->ops->view)(fvm, v));
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetFromOptions - sets parameters in a PetscFV from the options database

  Collective on fvm

  Input Parameter:
. fvm - the PetscFV object to set options for

  Options Database Key:
. -petscfv_compute_gradients <bool> - Determines whether cell gradients are calculated

  Level: intermediate

.seealso: PetscFVView()
@*/
PetscErrorCode PetscFVSetFromOptions(PetscFV fvm)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (!((PetscObject) fvm)->type_name) defaultType = PETSCFVUPWIND;
  else                                 defaultType = ((PetscObject) fvm)->type_name;
  CHKERRQ(PetscFVRegisterAll());

  ierr = PetscObjectOptionsBegin((PetscObject) fvm);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-petscfv_type", "Finite volume discretization", "PetscFVSetType", PetscFVList, defaultType, name, 256, &flg));
  if (flg) {
    CHKERRQ(PetscFVSetType(fvm, name));
  } else if (!((PetscObject) fvm)->type_name) {
    CHKERRQ(PetscFVSetType(fvm, defaultType));

  }
  CHKERRQ(PetscOptionsBool("-petscfv_compute_gradients", "Compute cell gradients", "PetscFVSetComputeGradients", fvm->computeGradients, &fvm->computeGradients, NULL));
  if (fvm->ops->setfromoptions) CHKERRQ((*fvm->ops->setfromoptions)(fvm));
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) fvm));
  CHKERRQ(PetscLimiterSetFromOptions(fvm->limiter));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscFVViewFromOptions(fvm, NULL, "-petscfv_view"));
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetUp - Construct data structures for the PetscFV

  Collective on fvm

  Input Parameter:
. fvm - the PetscFV object to setup

  Level: intermediate

.seealso: PetscFVView(), PetscFVDestroy()
@*/
PetscErrorCode PetscFVSetUp(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscLimiterSetUp(fvm->limiter));
  if (fvm->ops->setup) CHKERRQ((*fvm->ops->setup)(fvm));
  PetscFunctionReturn(0);
}

/*@
  PetscFVDestroy - Destroys a PetscFV object

  Collective on fvm

  Input Parameter:
. fvm - the PetscFV object to destroy

  Level: beginner

.seealso: PetscFVView()
@*/
PetscErrorCode PetscFVDestroy(PetscFV *fvm)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!*fvm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*fvm), PETSCFV_CLASSID, 1);

  if (--((PetscObject)(*fvm))->refct > 0) {*fvm = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*fvm))->refct = 0;

  for (i = 0; i < (*fvm)->numComponents; i++) {
    CHKERRQ(PetscFree((*fvm)->componentNames[i]));
  }
  CHKERRQ(PetscFree((*fvm)->componentNames));
  CHKERRQ(PetscLimiterDestroy(&(*fvm)->limiter));
  CHKERRQ(PetscDualSpaceDestroy(&(*fvm)->dualSpace));
  CHKERRQ(PetscFree((*fvm)->fluxWork));
  CHKERRQ(PetscQuadratureDestroy(&(*fvm)->quadrature));
  CHKERRQ(PetscTabulationDestroy(&(*fvm)->T));

  if ((*fvm)->ops->destroy) CHKERRQ((*(*fvm)->ops->destroy)(*fvm));
  CHKERRQ(PetscHeaderDestroy(fvm));
  PetscFunctionReturn(0);
}

/*@
  PetscFVCreate - Creates an empty PetscFV object. The type can then be set with PetscFVSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscFV object

  Output Parameter:
. fvm - The PetscFV object

  Level: beginner

.seealso: PetscFVSetType(), PETSCFVUPWIND
@*/
PetscErrorCode PetscFVCreate(MPI_Comm comm, PetscFV *fvm)
{
  PetscFV        f;

  PetscFunctionBegin;
  PetscValidPointer(fvm, 2);
  *fvm = NULL;
  CHKERRQ(PetscFVInitializePackage());

  CHKERRQ(PetscHeaderCreate(f, PETSCFV_CLASSID, "PetscFV", "Finite Volume", "PetscFV", comm, PetscFVDestroy, PetscFVView));
  CHKERRQ(PetscMemzero(f->ops, sizeof(struct _PetscFVOps)));

  CHKERRQ(PetscLimiterCreate(comm, &f->limiter));
  f->numComponents    = 1;
  f->dim              = 0;
  f->computeGradients = PETSC_FALSE;
  f->fluxWork         = NULL;
  CHKERRQ(PetscCalloc1(f->numComponents,&f->componentNames));

  *fvm = f;
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetLimiter - Set the limiter object

  Logically collective on fvm

  Input Parameters:
+ fvm - the PetscFV object
- lim - The PetscLimiter

  Level: intermediate

.seealso: PetscFVGetLimiter()
@*/
PetscErrorCode PetscFVSetLimiter(PetscFV fvm, PetscLimiter lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 2);
  CHKERRQ(PetscLimiterDestroy(&fvm->limiter));
  CHKERRQ(PetscObjectReference((PetscObject) lim));
  fvm->limiter = lim;
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetLimiter - Get the limiter object

  Not collective

  Input Parameter:
. fvm - the PetscFV object

  Output Parameter:
. lim - The PetscLimiter

  Level: intermediate

.seealso: PetscFVSetLimiter()
@*/
PetscErrorCode PetscFVGetLimiter(PetscFV fvm, PetscLimiter *lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(lim, 2);
  *lim = fvm->limiter;
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetNumComponents - Set the number of field components

  Logically collective on fvm

  Input Parameters:
+ fvm - the PetscFV object
- comp - The number of components

  Level: intermediate

.seealso: PetscFVGetNumComponents()
@*/
PetscErrorCode PetscFVSetNumComponents(PetscFV fvm, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (fvm->numComponents != comp) {
    PetscInt i;

    for (i = 0; i < fvm->numComponents; i++) {
      CHKERRQ(PetscFree(fvm->componentNames[i]));
    }
    CHKERRQ(PetscFree(fvm->componentNames));
    CHKERRQ(PetscCalloc1(comp,&fvm->componentNames));
  }
  fvm->numComponents = comp;
  CHKERRQ(PetscFree(fvm->fluxWork));
  CHKERRQ(PetscMalloc1(comp, &fvm->fluxWork));
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetNumComponents - Get the number of field components

  Not collective

  Input Parameter:
. fvm - the PetscFV object

  Output Parameter:
, comp - The number of components

  Level: intermediate

.seealso: PetscFVSetNumComponents()
@*/
PetscErrorCode PetscFVGetNumComponents(PetscFV fvm, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(comp, 2);
  *comp = fvm->numComponents;
  PetscFunctionReturn(0);
}

/*@C
  PetscFVSetComponentName - Set the name of a component (used in output and viewing)

  Logically collective on fvm
  Input Parameters:
+ fvm - the PetscFV object
. comp - the component number
- name - the component name

  Level: intermediate

.seealso: PetscFVGetComponentName()
@*/
PetscErrorCode PetscFVSetComponentName(PetscFV fvm, PetscInt comp, const char *name)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(fvm->componentNames[comp]));
  CHKERRQ(PetscStrallocpy(name,&fvm->componentNames[comp]));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVGetComponentName - Get the name of a component (used in output and viewing)

  Logically collective on fvm
  Input Parameters:
+ fvm - the PetscFV object
- comp - the component number

  Output Parameter:
. name - the component name

  Level: intermediate

.seealso: PetscFVSetComponentName()
@*/
PetscErrorCode PetscFVGetComponentName(PetscFV fvm, PetscInt comp, const char **name)
{
  PetscFunctionBegin;
  *name = fvm->componentNames[comp];
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetSpatialDimension - Set the spatial dimension

  Logically collective on fvm

  Input Parameters:
+ fvm - the PetscFV object
- dim - The spatial dimension

  Level: intermediate

.seealso: PetscFVGetSpatialDimension()
@*/
PetscErrorCode PetscFVSetSpatialDimension(PetscFV fvm, PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->dim = dim;
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetSpatialDimension - Get the spatial dimension

  Logically collective on fvm

  Input Parameter:
. fvm - the PetscFV object

  Output Parameter:
. dim - The spatial dimension

  Level: intermediate

.seealso: PetscFVSetSpatialDimension()
@*/
PetscErrorCode PetscFVGetSpatialDimension(PetscFV fvm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = fvm->dim;
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetComputeGradients - Toggle computation of cell gradients

  Logically collective on fvm

  Input Parameters:
+ fvm - the PetscFV object
- computeGradients - Flag to compute cell gradients

  Level: intermediate

.seealso: PetscFVGetComputeGradients()
@*/
PetscErrorCode PetscFVSetComputeGradients(PetscFV fvm, PetscBool computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->computeGradients = computeGradients;
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetComputeGradients - Return flag for computation of cell gradients

  Not collective

  Input Parameter:
. fvm - the PetscFV object

  Output Parameter:
. computeGradients - Flag to compute cell gradients

  Level: intermediate

.seealso: PetscFVSetComputeGradients()
@*/
PetscErrorCode PetscFVGetComputeGradients(PetscFV fvm, PetscBool *computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(computeGradients, 2);
  *computeGradients = fvm->computeGradients;
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetQuadrature - Set the quadrature object

  Logically collective on fvm

  Input Parameters:
+ fvm - the PetscFV object
- q - The PetscQuadrature

  Level: intermediate

.seealso: PetscFVGetQuadrature()
@*/
PetscErrorCode PetscFVSetQuadrature(PetscFV fvm, PetscQuadrature q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscQuadratureDestroy(&fvm->quadrature));
  CHKERRQ(PetscObjectReference((PetscObject) q));
  fvm->quadrature = q;
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetQuadrature - Get the quadrature object

  Not collective

  Input Parameter:
. fvm - the PetscFV object

  Output Parameter:
. lim - The PetscQuadrature

  Level: intermediate

.seealso: PetscFVSetQuadrature()
@*/
PetscErrorCode PetscFVGetQuadrature(PetscFV fvm, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(q, 2);
  if (!fvm->quadrature) {
    /* Create default 1-point quadrature */
    PetscReal     *points, *weights;

    CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &fvm->quadrature));
    CHKERRQ(PetscCalloc1(fvm->dim, &points));
    CHKERRQ(PetscMalloc1(1, &weights));
    weights[0] = 1.0;
    CHKERRQ(PetscQuadratureSetData(fvm->quadrature, fvm->dim, 1, 1, points, weights));
  }
  *q = fvm->quadrature;
  PetscFunctionReturn(0);
}

/*@
  PetscFVGetDualSpace - Returns the PetscDualSpace used to define the inner product

  Not collective

  Input Parameter:
. fvm - The PetscFV object

  Output Parameter:
. sp - The PetscDualSpace object

  Note: A simple dual space is provided automatically, and the user typically will not need to override it.

  Level: intermediate

.seealso: PetscFVCreate()
@*/
PetscErrorCode PetscFVGetDualSpace(PetscFV fvm, PetscDualSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(sp, 2);
  if (!fvm->dualSpace) {
    DM              K;
    PetscInt        dim, Nc, c;

    CHKERRQ(PetscFVGetSpatialDimension(fvm, &dim));
    CHKERRQ(PetscFVGetNumComponents(fvm, &Nc));
    CHKERRQ(PetscDualSpaceCreate(PetscObjectComm((PetscObject) fvm), &fvm->dualSpace));
    CHKERRQ(PetscDualSpaceSetType(fvm->dualSpace, PETSCDUALSPACESIMPLE));
    CHKERRQ(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, PETSC_FALSE), &K));
    CHKERRQ(PetscDualSpaceSetNumComponents(fvm->dualSpace, Nc));
    CHKERRQ(PetscDualSpaceSetDM(fvm->dualSpace, K));
    CHKERRQ(DMDestroy(&K));
    CHKERRQ(PetscDualSpaceSimpleSetDimension(fvm->dualSpace, Nc));
    /* Should we be using PetscFVGetQuadrature() here? */
    for (c = 0; c < Nc; ++c) {
      PetscQuadrature qc;
      PetscReal      *points, *weights;

      CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &qc));
      CHKERRQ(PetscCalloc1(dim, &points));
      CHKERRQ(PetscCalloc1(Nc, &weights));
      weights[c] = 1.0;
      CHKERRQ(PetscQuadratureSetData(qc, dim, Nc, 1, points, weights));
      CHKERRQ(PetscDualSpaceSimpleSetFunctional(fvm->dualSpace, c, qc));
      CHKERRQ(PetscQuadratureDestroy(&qc));
    }
    CHKERRQ(PetscDualSpaceSetUp(fvm->dualSpace));
  }
  *sp = fvm->dualSpace;
  PetscFunctionReturn(0);
}

/*@
  PetscFVSetDualSpace - Sets the PetscDualSpace used to define the inner product

  Not collective

  Input Parameters:
+ fvm - The PetscFV object
- sp  - The PetscDualSpace object

  Level: intermediate

  Note: A simple dual space is provided automatically, and the user typically will not need to override it.

.seealso: PetscFVCreate()
@*/
PetscErrorCode PetscFVSetDualSpace(PetscFV fvm, PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  CHKERRQ(PetscDualSpaceDestroy(&fvm->dualSpace));
  fvm->dualSpace = sp;
  CHKERRQ(PetscObjectReference((PetscObject) fvm->dualSpace));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVGetCellTabulation - Returns the tabulation of the basis functions at the quadrature points

  Not collective

  Input Parameter:
. fvm - The PetscFV object

  Output Parameter:
. T - The basis function values and derivatives at quadrature points

  Note:
$ T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFEGetCellTabulation(), PetscFVCreateTabulation(), PetscFVGetQuadrature(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscFVGetCellTabulation(PetscFV fvm, PetscTabulation *T)
{
  PetscInt         npoints;
  const PetscReal *points;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(T, 2);
  CHKERRQ(PetscQuadratureGetData(fvm->quadrature, NULL, NULL, &npoints, &points, NULL));
  if (!fvm->T) CHKERRQ(PetscFVCreateTabulation(fvm, 1, npoints, points, 1, &fvm->T));
  *T = fvm->T;
  PetscFunctionReturn(0);
}

/*@C
  PetscFVCreateTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not collective

  Input Parameters:
+ fvm     - The PetscFV object
. nrepl   - The number of replicas
. npoints - The number of tabulation points in a replica
. points  - The tabulation point coordinates
- K       - The order of derivative to tabulate

  Output Parameter:
. T - The basis function values and derivative at tabulation points

  Note:
$ T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFECreateTabulation(), PetscTabulationDestroy(), PetscFEGetCellTabulation()
@*/
PetscErrorCode PetscFVCreateTabulation(PetscFV fvm, PetscInt nrepl, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation *T)
{
  PetscInt         pdim = 1; /* Dimension of approximation space P */
  PetscInt         cdim;     /* Spatial dimension */
  PetscInt         Nc;       /* Field components */
  PetscInt         k, p, d, c, e;

  PetscFunctionBegin;
  if (!npoints || K < 0) {
    *T = NULL;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(points, 4);
  PetscValidPointer(T, 6);
  CHKERRQ(PetscFVGetSpatialDimension(fvm, &cdim));
  CHKERRQ(PetscFVGetNumComponents(fvm, &Nc));
  CHKERRQ(PetscMalloc1(1, T));
  (*T)->K    = !cdim ? 0 : K;
  (*T)->Nr   = nrepl;
  (*T)->Np   = npoints;
  (*T)->Nb   = pdim;
  (*T)->Nc   = Nc;
  (*T)->cdim = cdim;
  CHKERRQ(PetscMalloc1((*T)->K+1, &(*T)->T));
  for (k = 0; k <= (*T)->K; ++k) {
    CHKERRQ(PetscMalloc1(nrepl*npoints*pdim*Nc*PetscPowInt(cdim, k), &(*T)->T[k]));
  }
  if (K >= 0) {for (p = 0; p < nrepl*npoints; ++p) for (d = 0; d < pdim; ++d) for (c = 0; c < Nc; ++c) (*T)->T[0][(p*pdim + d)*Nc + c] = 1.0;}
  if (K >= 1) {for (p = 0; p < nrepl*npoints; ++p) for (d = 0; d < pdim; ++d) for (c = 0; c < Nc; ++c) for (e = 0; e < cdim; ++e) (*T)->T[1][((p*pdim + d)*Nc + c)*cdim + e] = 0.0;}
  if (K >= 2) {for (p = 0; p < nrepl*npoints; ++p) for (d = 0; d < pdim; ++d) for (c = 0; c < Nc; ++c) for (e = 0; e < cdim*cdim; ++e) (*T)->T[2][((p*pdim + d)*Nc + c)*cdim*cdim + e] = 0.0;}
  PetscFunctionReturn(0);
}

/*@C
  PetscFVComputeGradient - Compute the gradient reconstruction matrix for a given cell

  Input Parameters:
+ fvm      - The PetscFV object
. numFaces - The number of cell faces which are not constrained
- dx       - The vector from the cell centroid to the neighboring cell centroid for each face

  Level: advanced

.seealso: PetscFVCreate()
@*/
PetscErrorCode PetscFVComputeGradient(PetscFV fvm, PetscInt numFaces, PetscScalar dx[], PetscScalar grad[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (fvm->ops->computegradient) CHKERRQ((*fvm->ops->computegradient)(fvm, numFaces, dx, grad));
  PetscFunctionReturn(0);
}

/*@C
  PetscFVIntegrateRHSFunction - Produce the cell residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fvm          - The PetscFV object for the field being integrated
. prob         - The PetscDS specifing the discretizations and continuum functions
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

.seealso: PetscFVCreate()
@*/
PetscErrorCode PetscFVIntegrateRHSFunction(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol,
                                           PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (fvm->ops->integraterhsfunction) CHKERRQ((*fvm->ops->integraterhsfunction)(fvm, prob, field, Nf, fgeom, neighborVol, uL, uR, fluxL, fluxR));
  PetscFunctionReturn(0);
}

/*@
  PetscFVRefine - Create a "refined" PetscFV object that refines the reference cell into smaller copies. This is typically used
  to precondition a higher order method with a lower order method on a refined mesh having the same number of dofs (but more
  sparsity). It is also used to create an interpolation between regularly refined meshes.

  Input Parameter:
. fv - The initial PetscFV

  Output Parameter:
. fvRef - The refined PetscFV

  Level: advanced

.seealso: PetscFVType, PetscFVCreate(), PetscFVSetType()
@*/
PetscErrorCode PetscFVRefine(PetscFV fv, PetscFV *fvRef)
{
  PetscDualSpace    Q, Qref;
  DM                K, Kref;
  PetscQuadrature   q, qref;
  DMPolytopeType    ct;
  DMPlexTransform   tr;
  PetscReal        *v0;
  PetscReal        *jac, *invjac;
  PetscInt          numComp, numSubelements, s;

  PetscFunctionBegin;
  CHKERRQ(PetscFVGetDualSpace(fv, &Q));
  CHKERRQ(PetscFVGetQuadrature(fv, &q));
  CHKERRQ(PetscDualSpaceGetDM(Q, &K));
  /* Create dual space */
  CHKERRQ(PetscDualSpaceDuplicate(Q, &Qref));
  CHKERRQ(DMRefine(K, PetscObjectComm((PetscObject) fv), &Kref));
  CHKERRQ(PetscDualSpaceSetDM(Qref, Kref));
  CHKERRQ(DMDestroy(&Kref));
  CHKERRQ(PetscDualSpaceSetUp(Qref));
  /* Create volume */
  CHKERRQ(PetscFVCreate(PetscObjectComm((PetscObject) fv), fvRef));
  CHKERRQ(PetscFVSetDualSpace(*fvRef, Qref));
  CHKERRQ(PetscFVGetNumComponents(fv,    &numComp));
  CHKERRQ(PetscFVSetNumComponents(*fvRef, numComp));
  CHKERRQ(PetscFVSetUp(*fvRef));
  /* Create quadrature */
  CHKERRQ(DMPlexGetCellType(K, 0, &ct));
  CHKERRQ(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
  CHKERRQ(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
  CHKERRQ(DMPlexRefineRegularGetAffineTransforms(tr, ct, &numSubelements, &v0, &jac, &invjac));
  CHKERRQ(PetscQuadratureExpandComposite(q, numSubelements, v0, jac, &qref));
  CHKERRQ(PetscDualSpaceSimpleSetDimension(Qref, numSubelements));
  for (s = 0; s < numSubelements; ++s) {
    PetscQuadrature  qs;
    const PetscReal *points, *weights;
    PetscReal       *p, *w;
    PetscInt         dim, Nc, npoints, np;

    CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &qs));
    CHKERRQ(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, &weights));
    np   = npoints/numSubelements;
    CHKERRQ(PetscMalloc1(np*dim,&p));
    CHKERRQ(PetscMalloc1(np*Nc,&w));
    CHKERRQ(PetscArraycpy(p, &points[s*np*dim], np*dim));
    CHKERRQ(PetscArraycpy(w, &weights[s*np*Nc], np*Nc));
    CHKERRQ(PetscQuadratureSetData(qs, dim, Nc, np, p, w));
    CHKERRQ(PetscDualSpaceSimpleSetFunctional(Qref, s, qs));
    CHKERRQ(PetscQuadratureDestroy(&qs));
  }
  CHKERRQ(PetscFVSetQuadrature(*fvRef, qref));
  CHKERRQ(DMPlexTransformDestroy(&tr));
  CHKERRQ(PetscQuadratureDestroy(&qref));
  CHKERRQ(PetscDualSpaceDestroy(&Qref));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVDestroy_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b = (PetscFV_Upwind *) fvm->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(b));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVView_Upwind_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc, c;
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Upwind Finite Volume:\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc));
  for (c = 0; c < Nc; c++) {
    if (fv->componentNames[c]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "    component %d: %s\n", c, fv->componentNames[c]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVView_Upwind(PetscFV fv, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscFVView_Upwind_Ascii(fv, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVSetUp_Upwind(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode PetscFVIntegrateRHSFunction_Upwind(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol,
                                                         PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  void             (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  void              *rctx;
  PetscScalar       *flux = fvm->fluxWork;
  const PetscScalar *constants;
  PetscInt           dim, numConstants, pdim, totDim, Nc, off, f, d;

  PetscFunctionBegin;
  CHKERRQ(PetscDSGetTotalComponents(prob, &Nc));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSGetFieldOffset(prob, field, &off));
  CHKERRQ(PetscDSGetRiemannSolver(prob, field, &riemann));
  CHKERRQ(PetscDSGetContext(prob, field, &rctx));
  CHKERRQ(PetscDSGetConstants(prob, &numConstants, &constants));
  CHKERRQ(PetscFVGetSpatialDimension(fvm, &dim));
  CHKERRQ(PetscFVGetNumComponents(fvm, &pdim));
  for (f = 0; f < Nf; ++f) {
    (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f*Nc], &uR[f*Nc], numConstants, constants, flux, rctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f*totDim+off+d] = flux[d] / neighborVol[f*2+0];
      fluxR[f*totDim+off+d] = flux[d] / neighborVol[f*2+1];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVInitialize_Upwind(PetscFV fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions          = NULL;
  fvm->ops->setup                   = PetscFVSetUp_Upwind;
  fvm->ops->view                    = PetscFVView_Upwind;
  fvm->ops->destroy                 = PetscFVDestroy_Upwind;
  fvm->ops->integraterhsfunction    = PetscFVIntegrateRHSFunction_Upwind;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFVUPWIND = "upwind" - A PetscFV object

  Level: intermediate

.seealso: PetscFVType, PetscFVCreate(), PetscFVSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscNewLog(fvm,&b));
  fvm->data = b;

  CHKERRQ(PetscFVInitialize_Upwind(fvm));
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>

static PetscErrorCode PetscFVDestroy_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *ls = (PetscFV_LeastSquares *) fvm->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject) fvm, "PetscFVLeastSquaresSetMaxFaces_C", NULL));
  CHKERRQ(PetscFree4(ls->B, ls->Binv, ls->tau, ls->work));
  CHKERRQ(PetscFree(ls));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVView_LeastSquares_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc, c;
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Finite Volume with Least Squares Reconstruction:\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc));
  for (c = 0; c < Nc; c++) {
    if (fv->componentNames[c]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "    component %d: %s\n", c, fv->componentNames[c]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVView_LeastSquares(PetscFV fv, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscFVView_LeastSquares_Ascii(fv, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVSetUp_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* Overwrites A. Can only handle full-rank problems with m>=n */
static PetscErrorCode PetscFVLeastSquaresPseudoInverse_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscBool      debug = PETSC_FALSE;
  PetscBLASInt   M,N,K,lda,ldb,ldwork,info;
  PetscScalar    *R,*Q,*Aback,Alpha;

  PetscFunctionBegin;
  if (debug) {
    CHKERRQ(PetscMalloc1(m*n,&Aback));
    CHKERRQ(PetscArraycpy(Aback,A,m*n));
  }

  CHKERRQ(PetscBLASIntCast(m,&M));
  CHKERRQ(PetscBLASIntCast(n,&N));
  CHKERRQ(PetscBLASIntCast(mstride,&lda));
  CHKERRQ(PetscBLASIntCast(worksize,&ldwork));
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&M,&N,A,&lda,tau,work,&ldwork,&info));
  CHKERRQ(PetscFPTrapPop());
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q    = Ainv;
  CHKERRQ(PetscArraycpy(Q,A,mstride*n));
  K    = N;                     /* full rank */
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&M,&N,&K,Q,&lda,tau,work,&ldwork,&info));
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb   = lda;
  BLAStrsm_("Right","Upper","ConjugateTranspose","NotUnitTriangular",&M,&N,&Alpha,R,&lda,Q,&ldb);
  /* Ainv is Q, overwritten with inverse */

  if (debug) {                      /* Check that pseudo-inverse worked */
    PetscScalar  Beta = 0.0;
    PetscBLASInt ldc;
    K   = N;
    ldc = N;
    BLASgemm_("ConjugateTranspose","Normal",&N,&K,&M,&Alpha,Ainv,&lda,Aback,&ldb,&Beta,work,&ldc);
    CHKERRQ(PetscScalarView(n*n,work,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(PetscFree(Aback));
  }
  PetscFunctionReturn(0);
}

/* Overwrites A. Can handle degenerate problems and m<n. */
static PetscErrorCode PetscFVLeastSquaresPseudoInverseSVD_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscBool      debug = PETSC_FALSE;
  PetscScalar   *Brhs, *Aback;
  PetscScalar   *tmpwork;
  PetscReal      rcond;
#if defined (PETSC_USE_COMPLEX)
  PetscInt       rworkSize;
  PetscReal     *rwork;
#endif
  PetscInt       i, j, maxmn;
  PetscBLASInt   M, N, lda, ldb, ldwork;
  PetscBLASInt   nrhs, irank, info;

  PetscFunctionBegin;
  if (debug) {
    CHKERRQ(PetscMalloc1(m*n,&Aback));
    CHKERRQ(PetscArraycpy(Aback,A,m*n));
  }

  /* initialize to identity */
  tmpwork = work;
  Brhs = Ainv;
  maxmn = PetscMax(m,n);
  for (j=0; j<maxmn; j++) {
    for (i=0; i<maxmn; i++) Brhs[i + j*maxmn] = 1.0*(i == j);
  }

  CHKERRQ(PetscBLASIntCast(m,&M));
  CHKERRQ(PetscBLASIntCast(n,&N));
  CHKERRQ(PetscBLASIntCast(mstride,&lda));
  CHKERRQ(PetscBLASIntCast(maxmn,&ldb));
  CHKERRQ(PetscBLASIntCast(worksize,&ldwork));
  rcond = -1;
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  nrhs  = M;
#if defined(PETSC_USE_COMPLEX)
  rworkSize = 5 * PetscMin(M,N);
  CHKERRQ(PetscMalloc1(rworkSize,&rwork));
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M,&N,&nrhs,A,&lda,Brhs,&ldb, (PetscReal *) tau,&rcond,&irank,tmpwork,&ldwork,rwork,&info));
  CHKERRQ(PetscFPTrapPop());
  CHKERRQ(PetscFree(rwork));
#else
  nrhs  = M;
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M,&N,&nrhs,A,&lda,Brhs,&ldb, (PetscReal *) tau,&rcond,&irank,tmpwork,&ldwork,&info));
  CHKERRQ(PetscFPTrapPop());
#endif
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"xGELSS error");
  /* The following check should be turned into a diagnostic as soon as someone wants to do this intentionally */
  PetscCheckFalse(irank < PetscMin(M,N),PETSC_COMM_SELF,PETSC_ERR_USER,"Rank deficient least squares fit, indicates an isolated cell with two colinear points");
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode PetscFVLeastSquaresDebugCell_Static(PetscFV fvm, PetscInt cell, DM dm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  PetscReal       grad[2] = {0, 0};
  const PetscInt *faces;
  PetscInt        numFaces, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetConeSize(dm, cell, &numFaces));
  CHKERRQ(DMPlexGetCone(dm, cell, &faces));
  for (f = 0; f < numFaces; ++f) {
    const PetscInt *fcells;
    const CellGeom *cg1;
    const FaceGeom *fg;

    CHKERRQ(DMPlexGetSupport(dm, faces[f], &fcells));
    CHKERRQ(DMPlexPointLocalRead(dmFace, faces[f], fgeom, &fg));
    for (i = 0; i < 2; ++i) {
      PetscScalar du;

      if (fcells[i] == c) continue;
      CHKERRQ(DMPlexPointLocalRead(dmCell, fcells[i], cgeom, &cg1));
      du   = cg1->centroid[0] + 3*cg1->centroid[1] - (cg->centroid[0] + 3*cg->centroid[1]);
      grad[0] += fg->grad[!i][0] * du;
      grad[1] += fg->grad[!i][1] * du;
    }
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "cell[%d] grad (%g, %g)\n", cell, grad[0], grad[1]));
  PetscFunctionReturn(0);
}
#endif

/*
  PetscFVComputeGradient - Compute the gradient reconstruction matrix for a given cell

  Input Parameters:
+ fvm      - The PetscFV object
. numFaces - The number of cell faces which are not constrained
. dx       - The vector from the cell centroid to the neighboring cell centroid for each face

  Level: developer

.seealso: PetscFVCreate()
*/
static PetscErrorCode PetscFVComputeGradient_LeastSquares(PetscFV fvm, PetscInt numFaces, const PetscScalar dx[], PetscScalar grad[])
{
  PetscFV_LeastSquares *ls       = (PetscFV_LeastSquares *) fvm->data;
  const PetscBool       useSVD   = PETSC_TRUE;
  const PetscInt        maxFaces = ls->maxFaces;
  PetscInt              dim, f, d;

  PetscFunctionBegin;
  if (numFaces > maxFaces) {
    PetscCheckFalse(maxFaces < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Reconstruction has not been initialized, call PetscFVLeastSquaresSetMaxFaces()");
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input faces %D > %D maxfaces", numFaces, maxFaces);
  }
  CHKERRQ(PetscFVGetSpatialDimension(fvm, &dim));
  for (f = 0; f < numFaces; ++f) {
    for (d = 0; d < dim; ++d) ls->B[d*maxFaces+f] = dx[f*dim+d];
  }
  /* Overwrites B with garbage, returns Binv in row-major format */
  if (useSVD) {
    PetscInt maxmn = PetscMax(numFaces, dim);
    CHKERRQ(PetscFVLeastSquaresPseudoInverseSVD_Static(numFaces, maxFaces, dim, ls->B, ls->Binv, ls->tau, ls->workSize, ls->work));
    /* Binv shaped in column-major, coldim=maxmn.*/
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) grad[f*dim+d] = ls->Binv[d + maxmn*f];
    }
  } else {
    CHKERRQ(PetscFVLeastSquaresPseudoInverse_Static(numFaces, maxFaces, dim, ls->B, ls->Binv, ls->tau, ls->workSize, ls->work));
    /* Binv shaped in row-major, rowdim=maxFaces.*/
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) grad[f*dim+d] = ls->Binv[d*maxFaces + f];
    }
  }
  PetscFunctionReturn(0);
}

/*
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode PetscFVIntegrateRHSFunction_LeastSquares(PetscFV fvm, PetscDS prob, PetscInt field, PetscInt Nf, PetscFVFaceGeom *fgeom, PetscReal *neighborVol,
                                                               PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
  void             (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  void              *rctx;
  PetscScalar       *flux = fvm->fluxWork;
  const PetscScalar *constants;
  PetscInt           dim, numConstants, pdim, Nc, totDim, off, f, d;

  PetscFunctionBegin;
  CHKERRQ(PetscDSGetTotalComponents(prob, &Nc));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSGetFieldOffset(prob, field, &off));
  CHKERRQ(PetscDSGetRiemannSolver(prob, field, &riemann));
  CHKERRQ(PetscDSGetContext(prob, field, &rctx));
  CHKERRQ(PetscDSGetConstants(prob, &numConstants, &constants));
  CHKERRQ(PetscFVGetSpatialDimension(fvm, &dim));
  CHKERRQ(PetscFVGetNumComponents(fvm, &pdim));
  for (f = 0; f < Nf; ++f) {
    (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f*Nc], &uR[f*Nc], numConstants, constants, flux, rctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f*totDim+off+d] = flux[d] / neighborVol[f*2+0];
      fluxR[f*totDim+off+d] = flux[d] / neighborVol[f*2+1];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFVLeastSquaresSetMaxFaces_LS(PetscFV fvm, PetscInt maxFaces)
{
  PetscFV_LeastSquares *ls = (PetscFV_LeastSquares *) fvm->data;
  PetscInt              dim,m,n,nrhs,minmn,maxmn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscFVGetSpatialDimension(fvm, &dim));
  CHKERRQ(PetscFree4(ls->B, ls->Binv, ls->tau, ls->work));
  ls->maxFaces = maxFaces;
  m       = ls->maxFaces;
  n       = dim;
  nrhs    = ls->maxFaces;
  minmn   = PetscMin(m,n);
  maxmn   = PetscMax(m,n);
  ls->workSize = 3*minmn + PetscMax(2*minmn, PetscMax(maxmn, nrhs)); /* required by LAPACK */
  CHKERRQ(PetscMalloc4(m*n,&ls->B,maxmn*maxmn,&ls->Binv,minmn,&ls->tau,ls->workSize,&ls->work));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFVInitialize_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions          = NULL;
  fvm->ops->setup                   = PetscFVSetUp_LeastSquares;
  fvm->ops->view                    = PetscFVView_LeastSquares;
  fvm->ops->destroy                 = PetscFVDestroy_LeastSquares;
  fvm->ops->computegradient         = PetscFVComputeGradient_LeastSquares;
  fvm->ops->integraterhsfunction    = PetscFVIntegrateRHSFunction_LeastSquares;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFVLEASTSQUARES = "leastsquares" - A PetscFV object

  Level: intermediate

.seealso: PetscFVType, PetscFVCreate(), PetscFVSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscNewLog(fvm, &ls));
  fvm->data = ls;

  ls->maxFaces = -1;
  ls->workSize = -1;
  ls->B        = NULL;
  ls->Binv     = NULL;
  ls->tau      = NULL;
  ls->work     = NULL;

  CHKERRQ(PetscFVSetComputeGradients(fvm, PETSC_TRUE));
  CHKERRQ(PetscFVInitialize_LeastSquares(fvm));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) fvm, "PetscFVLeastSquaresSetMaxFaces_C", PetscFVLeastSquaresSetMaxFaces_LS));
  PetscFunctionReturn(0);
}

/*@
  PetscFVLeastSquaresSetMaxFaces - Set the maximum number of cell faces for gradient reconstruction

  Not collective

  Input parameters:
+ fvm      - The PetscFV object
- maxFaces - The maximum number of cell faces

  Level: intermediate

.seealso: PetscFVCreate(), PETSCFVLEASTSQUARES
@*/
PetscErrorCode PetscFVLeastSquaresSetMaxFaces(PetscFV fvm, PetscInt maxFaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  CHKERRQ(PetscTryMethod(fvm, "PetscFVLeastSquaresSetMaxFaces_C", (PetscFV,PetscInt), (fvm,maxFaces)));
  PetscFunctionReturn(0);
}
