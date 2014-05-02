#include <petsc-private/petscfvimpl.h> /*I "petscfv.h" I*/
#include <petscdmplex.h>

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
    PetscFVRegister("my_fe", MyPetscFVCreate);
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
. dm  - The PetscFV

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

  f->numComponents = 1;
  f->dim           = 0;
  f->fluxWork      = NULL;

  *fvm = f;
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
