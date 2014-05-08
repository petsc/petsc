#include <petsc-private/petscfvimpl.h> /*I "petscfv.h" I*/
#include <petscdmplex.h>

PetscClassId PETSCLIMITER_CLASSID = 0;

PetscFunctionList PetscLimiterList              = NULL;
PetscBool         PetscLimiterRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterRegister"
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

.keywords: PetscLimiter, register
.seealso: PetscLimiterRegisterAll(), PetscLimiterRegisterDestroy()

@*/
PetscErrorCode PetscLimiterRegister(const char sname[], PetscErrorCode (*function)(PetscLimiter))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscLimiterList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterSetType"
/*@C
  PetscLimiterSetType - Builds a particular PetscLimiter

  Collective on PetscLimiter

  Input Parameters:
+ lim  - The PetscLimiter object
- name - The kind of limiter

  Options Database Key:
. -petsclimiter_type <type> - Sets the PetscLimiter type; use -help for a list of available types

  Level: intermediate

.keywords: PetscLimiter, set, type
.seealso: PetscLimiterGetType(), PetscLimiterCreate()
@*/
PetscErrorCode PetscLimiterSetType(PetscLimiter lim, PetscLimiterType name)
{
  PetscErrorCode (*r)(PetscLimiter);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) lim, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscLimiterRegisterAllCalled) {ierr = PetscLimiterRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscLimiterList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) lim), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscLimiter type: %s", name);

  if (lim->ops->destroy) {
    ierr              = (*lim->ops->destroy)(lim);CHKERRQ(ierr);
    lim->ops->destroy = NULL;
  }
  ierr = (*r)(lim);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) lim, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterGetType"
/*@C
  PetscLimiterGetType - Gets the PetscLimiter type name (as a string) from the object.

  Not Collective

  Input Parameter:
. lim  - The PetscLimiter

  Output Parameter:
. name - The PetscLimiter type name

  Level: intermediate

.keywords: PetscLimiter, get, type, name
.seealso: PetscLimiterSetType(), PetscLimiterCreate()
@*/
PetscErrorCode PetscLimiterGetType(PetscLimiter lim, PetscLimiterType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (!PetscLimiterRegisterAllCalled) {ierr = PetscLimiterRegisterAll();CHKERRQ(ierr);}
  *name = ((PetscObject) lim)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterView"
/*@C
  PetscLimiterView - Views a PetscLimiter

  Collective on PetscLimiter

  Input Parameter:
+ lim - the PetscLimiter object to view
- v   - the viewer

  Level: developer

.seealso PetscLimiterDestroy()
@*/
PetscErrorCode PetscLimiterView(PetscLimiter lim, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) lim), &v);CHKERRQ(ierr);}
  if (lim->ops->view) {ierr = (*lim->ops->view)(lim, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterViewFromOptions"
/*
  PetscLimiterViewFromOptions - Processes command line options to determine if/how a PetscLimiter is to be viewed.

  Collective on PetscLimiter

  Input Parameters:
+ lim    - the PetscLimiter
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscLimiter, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscLimiterViewFromOptions(PetscLimiter lim, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) lim), prefix,                      optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  else        {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) lim), ((PetscObject) lim)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscLimiterView(lim, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterSetFromOptions"
/*@
  PetscLimiterSetFromOptions - sets parameters in a PetscLimiter from the options database

  Collective on PetscLimiter

  Input Parameter:
. lim - the PetscLimiter object to set options for

  Level: developer

.seealso PetscLimiterView()
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
  if (!PetscLimiterRegisterAllCalled) {ierr = PetscLimiterRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) lim);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petsclimiter_type", "Finite volume slope limiter", "PetscLimiterSetType", PetscLimiterList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscLimiterSetType(lim, name);CHKERRQ(ierr);
  } else if (!((PetscObject) lim)->type_name) {
    ierr = PetscLimiterSetType(lim, defaultType);CHKERRQ(ierr);
  }
  if (lim->ops->setfromoptions) {ierr = (*lim->ops->setfromoptions)(lim);CHKERRQ(ierr);}
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) lim);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscLimiterViewFromOptions(lim, NULL, "-petsclimiter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterSetUp"
/*@C
  PetscLimiterSetUp - Construct data structures for the PetscLimiter

  Collective on PetscLimiter

  Input Parameter:
. lim - the PetscLimiter object to setup

  Level: developer

.seealso PetscLimiterView(), PetscLimiterDestroy()
@*/
PetscErrorCode PetscLimiterSetUp(PetscLimiter lim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  if (lim->ops->setup) {ierr = (*lim->ops->setup)(lim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterDestroy"
/*@
  PetscLimiterDestroy - Destroys a PetscLimiter object

  Collective on PetscLimiter

  Input Parameter:
. lim - the PetscLimiter object to destroy

  Level: developer

.seealso PetscLimiterView()
@*/
PetscErrorCode PetscLimiterDestroy(PetscLimiter *lim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*lim) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*lim), PETSCLIMITER_CLASSID, 1);

  if (--((PetscObject)(*lim))->refct > 0) {*lim = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*lim))->refct = 0;

  if ((*lim)->ops->destroy) {ierr = (*(*lim)->ops->destroy)(*lim);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(lim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterCreate"
/*@
  PetscLimiterCreate - Creates an empty PetscLimiter object. The type can then be set with PetscLimiterSetType().

  Collective on MPI_Comm

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(lim, 2);
  *lim = NULL;
  ierr = PetscFVInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(l, _p_PetscLimiter, struct _PetscLimiterOps, PETSCLIMITER_CLASSID, "PetscLimiter", "Finite Volume Slope Limiter", "PetscLimiter", comm, PetscLimiterDestroy, PetscLimiterView);CHKERRQ(ierr);
  ierr = PetscMemzero(l->ops, sizeof(struct _PetscLimiterOps));CHKERRQ(ierr);

  *lim = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit"
/* Limiters given in symmetric form following Berger, Aftosmis, and Murman 2005
 *
 * The classical flux-limited formulation is psi(r) where
 *
 * r = (u[0] - u[-1]) / (u[1] - u[0])
 *
 * The second order TVD region is bounded by
 *
 * psi_minmod(r) = min(r,1)      and        psi_superbee(r) = min(2, 2r, max(1,r))
 *
 * where all limiters are implicitly clipped to be non-negative. A more convenient slope-limited form is psi(r) =
 * phi(r)(r+1)/2 in which the reconstructed interface values are
 *
 * u(v) = u[0] + phi(r) (grad u)[0] v
 *
 * where v is the vector from centroid to quadrature point. In these variables, the usual limiters become
 *
 * phi_minmod(r) = 2 min(1/(1+r),r/(1+r))   phi_superbee(r) = 2 min(2/(1+r), 2r/(1+r), max(1,r)/(1+r))
 *
 * For a nicer symmetric formulation, rewrite in terms of
 *
 * f = (u[0] - u[-1]) / (u[1] - u[-1])
 *
 * where r(f) = f/(1-f). Not that r(1-f) = (1-f)/f = 1/r(f) so the symmetry condition
 *
 * phi(r) = phi(1/r)
 *
 * becomes
 *
 * w(f) = w(1-f).
 *
 * The limiters below implement this final form w(f). The reference methods are
 *
 * w_minmod(f) = 2 min(f,(1-f))             w_superbee(r) = 4 min((1-f), f)
 * */
PetscErrorCode PetscLimiterLimit(PetscLimiter lim, PetscScalar flim, PetscScalar *phi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  PetscValidPointer(phi, 3);
  ierr = (*lim->ops->limit)(lim, flim, phi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterDestroy_Sin"
PetscErrorCode PetscLimiterDestroy_Sin(PetscLimiter fvm)
{
  PetscLimiter_Sin *l = (PetscLimiter_Sin *) fvm->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFree(l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterView_Sin_Ascii"
PetscErrorCode PetscLimiterView_Sin_Ascii(PetscLimiter fv, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Sin Slope Limiter:\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterView_Sin"
PetscErrorCode PetscLimiterView_Sin(PetscLimiter fv, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscLimiterView_Sin_Ascii(fv, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterSetUp_Sin"
PetscErrorCode PetscLimiterSetUp_Sin(PetscLimiter fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_Sin"
PetscErrorCode PetscLimiterLimit_Sin(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = PetscSinReal(PETSC_PI*PetscMax(0, PetscMin(f, 1)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterInitialize_Sin"
PetscErrorCode PetscLimiterInitialize_Sin(PetscLimiter fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions = NULL;
  fvm->ops->setup          = PetscLimiterSetUp_Sin;
  fvm->ops->view           = PetscLimiterView_Sin;
  fvm->ops->destroy        = PetscLimiterDestroy_Sin;
  fvm->ops->limit          = PetscLimiterLimit_Sin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_Zero"
PetscErrorCode PetscLimiterLimit_Zero(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_None"
PetscErrorCode PetscLimiterLimit_None(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = 1.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_Minmod"
PetscErrorCode PetscLimiterLimit_Minmod(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = 2*PetscMax(0, PetscMin(f, 1-f));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_VanLeer"
PetscErrorCode PetscLimiterLimit_VanLeer(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 4*f*(1-f));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_VanAlbada"
PetscErrorCode PetscLimiterLimit_VanAlbada(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = PetscMax(0, 2*f*(1-f) / (PetscSqr(f) + PetscSqr(1-f)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_Superbee"
PetscErrorCode PetscLimiterLimit_Superbee(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = 4*PetscMax(0, PetscMin(f, 1-f));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterLimit_MC"
/* aka Barth-Jespersen */
PetscErrorCode PetscLimiterLimit_MC(PetscLimiter lim, PetscScalar f, PetscScalar *phi)
{
  PetscFunctionBegin;
  *phi = PetscMin(1, 4*PetscMax(0, PetscMin(f, 1-f)));
  PetscFunctionReturn(0);
}

/*MC
  PETSCFVUPWIND = "upwind" - A PetscLimiter object

  Level: intermediate

.seealso: PetscLimiterType, PetscLimiterCreate(), PetscLimiterSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterCreate_Sin"
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Sin(PetscLimiter lim)
{
  PetscLimiter_Sin *l;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 1);
  ierr      = PetscNewLog(lim, &l);CHKERRQ(ierr);
  lim->data = l;

  ierr = PetscLimiterInitialize_Sin(lim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscClassId PETSCFV_CLASSID = 0;

PetscFunctionList PetscFVList              = NULL;
PetscBool         PetscFVRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscFVRegister"
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

.keywords: PetscFV, register
.seealso: PetscFVRegisterAll(), PetscFVRegisterDestroy()

@*/
PetscErrorCode PetscFVRegister(const char sname[], PetscErrorCode (*function)(PetscFV))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscFVList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetType"
/*@C
  PetscFVSetType - Builds a particular PetscFV

  Collective on PetscFV

  Input Parameters:
+ fvm  - The PetscFV object
- name - The kind of FVM space

  Options Database Key:
. -petscfv_type <type> - Sets the PetscFV type; use -help for a list of available types

  Level: intermediate

.keywords: PetscFV, set, type
.seealso: PetscFVGetType(), PetscFVCreate()
@*/
PetscErrorCode PetscFVSetType(PetscFV fvm, PetscFVType name)
{
  PetscErrorCode (*r)(PetscFV);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) fvm, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscFVRegisterAllCalled) {ierr = PetscFVRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscFVList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) fvm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscFV type: %s", name);

  if (fvm->ops->destroy) {
    ierr              = (*fvm->ops->destroy)(fvm);CHKERRQ(ierr);
    fvm->ops->destroy = NULL;
  }
  ierr = (*r)(fvm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) fvm, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVGetType"
/*@C
  PetscFVGetType - Gets the PetscFV type name (as a string) from the object.

  Not Collective

  Input Parameter:
. fvm  - The PetscFV

  Output Parameter:
. name - The PetscFV type name

  Level: intermediate

.keywords: PetscFV, get, type, name
.seealso: PetscFVSetType(), PetscFVCreate()
@*/
PetscErrorCode PetscFVGetType(PetscFV fvm, PetscFVType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (!PetscFVRegisterAllCalled) {ierr = PetscFVRegisterAll();CHKERRQ(ierr);}
  *name = ((PetscObject) fvm)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVView"
/*@C
  PetscFVView - Views a PetscFV

  Collective on PetscFV

  Input Parameter:
+ fvm - the PetscFV object to view
- v   - the viewer

  Level: developer

.seealso PetscFVDestroy()
@*/
PetscErrorCode PetscFVView(PetscFV fvm, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) fvm), &v);CHKERRQ(ierr);}
  if (fvm->ops->view) {ierr = (*fvm->ops->view)(fvm, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVViewFromOptions"
/*
  PetscFVViewFromOptions - Processes command line options to determine if/how a PetscFV is to be viewed.

  Collective on PetscFV

  Input Parameters:
+ fvm    - the PetscFV
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscFV, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscFVViewFromOptions(PetscFV fvm, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) fvm), prefix,                      optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  else        {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) fvm), ((PetscObject) fvm)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscFVView(fvm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetFromOptions"
/*@
  PetscFVSetFromOptions - sets parameters in a PetscFV from the options database

  Collective on PetscFV

  Input Parameter:
. fvm - the PetscFV object to set options for

  Level: developer

.seealso PetscFVView()
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
  if (!PetscFVRegisterAllCalled) {ierr = PetscFVRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) fvm);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscfv_type", "Finite volume discretization", "PetscFVSetType", PetscFVList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFVSetType(fvm, name);CHKERRQ(ierr);
  } else if (!((PetscObject) fvm)->type_name) {
    ierr = PetscFVSetType(fvm, defaultType);CHKERRQ(ierr);
  }
  if (fvm->ops->setfromoptions) {ierr = (*fvm->ops->setfromoptions)(fvm);CHKERRQ(ierr);}
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) fvm);CHKERRQ(ierr);
  ierr = PetscLimiterSetFromOptions(fvm->limiter);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscFVViewFromOptions(fvm, NULL, "-petscfv_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetUp"
/*@C
  PetscFVSetUp - Construct data structures for the PetscFV

  Collective on PetscFV

  Input Parameter:
. fvm - the PetscFV object to setup

  Level: developer

.seealso PetscFVView(), PetscFVDestroy()
@*/
PetscErrorCode PetscFVSetUp(PetscFV fvm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  ierr = PetscLimiterSetUp(fvm->limiter);CHKERRQ(ierr);
  if (fvm->ops->setup) {ierr = (*fvm->ops->setup)(fvm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVDestroy"
/*@
  PetscFVDestroy - Destroys a PetscFV object

  Collective on PetscFV

  Input Parameter:
. fvm - the PetscFV object to destroy

  Level: developer

.seealso PetscFVView()
@*/
PetscErrorCode PetscFVDestroy(PetscFV *fvm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fvm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*fvm), PETSCFV_CLASSID, 1);

  if (--((PetscObject)(*fvm))->refct > 0) {*fvm = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*fvm))->refct = 0;

  ierr = PetscLimiterDestroy(&(*fvm)->limiter);CHKERRQ(ierr);
  ierr = PetscFree((*fvm)->fluxWork);CHKERRQ(ierr);

  if ((*fvm)->ops->destroy) {ierr = (*(*fvm)->ops->destroy)(*fvm);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(fvm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVCreate"
/*@
  PetscFVCreate - Creates an empty PetscFV object. The type can then be set with PetscFVSetType().

  Collective on MPI_Comm

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(fvm, 2);
  *fvm = NULL;
  ierr = PetscFVInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(f, _p_PetscFV, struct _PetscFVOps, PETSCFV_CLASSID, "PetscFV", "Finite Volume", "PetscFV", comm, PetscFVDestroy, PetscFVView);CHKERRQ(ierr);
  ierr = PetscMemzero(f->ops, sizeof(struct _PetscFVOps));CHKERRQ(ierr);

  ierr = PetscLimiterCreate(comm, &f->limiter);CHKERRQ(ierr);
  f->numComponents    = 1;
  f->dim              = 0;
  f->computeGradients = PETSC_FALSE;
  f->fluxWork         = NULL;

  *fvm = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetLimiter"
PetscErrorCode PetscFVSetLimiter(PetscFV fvm, PetscLimiter lim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(lim, PETSCLIMITER_CLASSID, 2);
  ierr = PetscLimiterDestroy(&fvm->limiter);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) lim);CHKERRQ(ierr);
  fvm->limiter = lim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVGetLimiter"
PetscErrorCode PetscFVGetLimiter(PetscFV fvm, PetscLimiter *lim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(lim, 2);
  *lim = fvm->limiter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetNumComponents"
PetscErrorCode PetscFVSetNumComponents(PetscFV fvm, PetscInt comp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->numComponents = comp;
  ierr = PetscFree(fvm->fluxWork);CHKERRQ(ierr);
  ierr = PetscMalloc1(comp, &fvm->fluxWork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVGetNumComponents"
PetscErrorCode PetscFVGetNumComponents(PetscFV fvm, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(comp, 2);
  *comp = fvm->numComponents;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetSpatialDimension"
PetscErrorCode PetscFVSetSpatialDimension(PetscFV fvm, PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVGetSpatialDimension"
PetscErrorCode PetscFVGetSpatialDimension(PetscFV fvm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = fvm->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetComputeGradients"
PetscErrorCode PetscFVSetComputeGradients(PetscFV fvm, PetscBool computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  fvm->computeGradients = computeGradients;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVGetComputeGradients"
PetscErrorCode PetscFVGetComputeGradients(PetscFV fvm, PetscBool *computeGradients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  PetscValidPointer(computeGradients, 2);
  *computeGradients = fvm->computeGradients;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVIntegrateRHSFunction"
/*C
  PetscFVIntegrateRHSFunction - Produce the cell residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fvm          - The PetscFV object for the field being integrated
. Nface        - The number of faces in the chunk
. Nf           - The number of physical fields
. fv           - The PetscFV objects for each field
. field        - The field being integrated
. fgeom        - The face geometry for each face in the chunk
. cgeom        - The cell geometry for each pair of cells in the chunk
. uL           - The state from the cell on the left
. uR           - The state from the cell on the right
. riemann      - Riemann solver
- ctx          - User context passed to Riemann solve

  Output Parameter
+ fluxL        - the left fluxes for each face
- fluxR        - the right fluxes for each face
*/
PetscErrorCode PetscFVIntegrateRHSFunction(PetscFV fvm, PetscInt Nfaces, PetscInt Nf, PetscFV fv[], PetscInt field, PetscCellGeometry fgeom, PetscCellGeometry cgeom, PetscScalar uL[], PetscScalar uR[],
                                           void (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx),
                                           PetscScalar fluxL[], PetscScalar fluxR[], void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  if (fvm->ops->integraterhsfunction) {ierr = (*fvm->ops->integraterhsfunction)(fvm, Nfaces, Nf, fv, field, fgeom, cgeom, uL, uR, riemann, fluxL, fluxR, ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVDestroy_Upwind"
PetscErrorCode PetscFVDestroy_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b = (PetscFV_Upwind *) fvm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVView_Upwind_Ascii"
PetscErrorCode PetscFVView_Upwind_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Upwind Finite Volume:\n");CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVView_Upwind"
PetscErrorCode PetscFVView_Upwind(PetscFV fv, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscFVView_Upwind_Ascii(fv, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetUp_Upwind"
PetscErrorCode PetscFVSetUp_Upwind(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVIntegrateRHSFunction_Upwind"
/*
  fgeom[f]=>v0 is the centroid
  cgeom->vol[f*2+0] contains the left  geom
  cgeom->vol[f*2+1] contains the right geom
*/
PetscErrorCode PetscFVIntegrateRHSFunction_Upwind(PetscFV fvm, PetscInt Nfaces, PetscInt Nf, PetscFV fv[], PetscInt field, PetscCellGeometry fgeom, PetscCellGeometry cgeom,
                                                  PetscScalar xL[], PetscScalar xR[],
                                                  void (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx),
                                                  PetscScalar fluxL[], PetscScalar fluxR[], void *ctx)
{
  PetscScalar   *flux = fvm->fluxWork;
  PetscInt       dim, pdim, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFVGetSpatialDimension(fvm, &dim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  for (f = 0; f < Nfaces; ++f) {
    (*riemann)(&fgeom.v0[f*dim], &fgeom.n[f*dim], &xL[f*pdim], &xR[f*pdim], flux, ctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f*pdim+d] = flux[d] / cgeom.vol[f*2+0];
      fluxR[f*pdim+d] = flux[d] / cgeom.vol[f*2+1];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVInitialize_Upwind"
PetscErrorCode PetscFVInitialize_Upwind(PetscFV fvm)
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

#undef __FUNCT__
#define __FUNCT__ "PetscFVCreate_Upwind"
PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV fvm)
{
  PetscFV_Upwind *b;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  ierr      = PetscNewLog(fvm,&b);CHKERRQ(ierr);
  fvm->data = b;

  ierr = PetscFVInitialize_Upwind(fvm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVDestroy_LeastSquares"
PetscErrorCode PetscFVDestroy_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *b = (PetscFV_LeastSquares *) fvm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVView_LeastSquares_Ascii"
PetscErrorCode PetscFVView_LeastSquares_Ascii(PetscFV fv, PetscViewer viewer)
{
  PetscInt          Nc;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Finite Volume with Least Squares Reconstruction:\n");CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "  num components: %d\n", Nc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVView_LeastSquares"
PetscErrorCode PetscFVView_LeastSquares(PetscFV fv, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscFVView_LeastSquares_Ascii(fv, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVSetUp_LeastSquares"
PetscErrorCode PetscFVSetUp_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVIntegrateRHSFunction_LeastSquares"
/*
  fgeom[f]=>v0 is the centroid
  cgeom->vol[f*2+0] contains the left  geom
  cgeom->vol[f*2+1] contains the right geom
*/
PetscErrorCode PetscFVIntegrateRHSFunction_LeastSquares(PetscFV fvm, PetscInt Nfaces, PetscInt Nf, PetscFV fv[], PetscInt field, PetscCellGeometry fgeom, PetscCellGeometry cgeom,
                                                        PetscScalar xL[], PetscScalar xR[],
                                                        void (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx),
                                                        PetscScalar fluxL[], PetscScalar fluxR[], void *ctx)
{
  PetscScalar   *flux = fvm->fluxWork;
  PetscInt       dim, pdim, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFVGetSpatialDimension(fvm, &dim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  for (f = 0; f < Nfaces; ++f) {
    (*riemann)(&fgeom.v0[f*dim], &fgeom.n[f*dim], &xL[f*pdim], &xR[f*pdim], flux, ctx);
    for (d = 0; d < pdim; ++d) {
      fluxL[f*pdim+d] = flux[d] / cgeom.vol[f*2+0];
      fluxR[f*pdim+d] = flux[d] / cgeom.vol[f*2+1];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVInitialize_LeastSquares"
PetscErrorCode PetscFVInitialize_LeastSquares(PetscFV fvm)
{
  PetscFunctionBegin;
  fvm->ops->setfromoptions          = NULL;
  fvm->ops->setup                   = PetscFVSetUp_LeastSquares;
  fvm->ops->view                    = PetscFVView_LeastSquares;
  fvm->ops->destroy                 = PetscFVDestroy_LeastSquares;
  fvm->ops->integraterhsfunction    = PetscFVIntegrateRHSFunction_LeastSquares;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFVLEASTSQUARES = "leastsquares" - A PetscFV object

  Level: intermediate

.seealso: PetscFVType, PetscFVCreate(), PetscFVSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscFVCreate_LeastSquares"
PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV fvm)
{
  PetscFV_LeastSquares *b;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fvm, PETSCFV_CLASSID, 1);
  ierr      = PetscNewLog(fvm,&b);CHKERRQ(ierr);
  fvm->data = b;

  ierr = PetscFVSetComputeGradients(fvm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFVInitialize_LeastSquares(fvm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
