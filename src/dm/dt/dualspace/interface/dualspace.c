#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

PetscClassId PETSCDUALSPACE_CLASSID = 0;

PetscLogEvent PETSCDUALSPACE_SetUp;

PetscFunctionList PetscDualSpaceList              = NULL;
PetscBool         PetscDualSpaceRegisterAllCalled = PETSC_FALSE;

const char *const PetscDualSpaceReferenceCells[] = {"SIMPLEX", "TENSOR", "PetscDualSpaceReferenceCell", "PETSCDUALSPACE_REFCELL_", NULL};

/*
  PetscDualSpaceLatticePointLexicographic_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to at most 'max'.
                                                     Ordering is lexicographic with lowest index as least significant in ordering.
                                                     e.g. for len == 2 and max == 2, this will return, in order, {0,0}, {1,0}, {2,0}, {0,1}, {1,1}, {0,2}.

  Input Parameters:
+ len - The length of the tuple
. max - The maximum sum
- tup - A tuple of length len+1: tup[len] > 0 indicates a stopping condition

  Output Parameter:
. tup - A tuple of len integers whos sum is at most 'max'

  Level: developer

.seealso: PetscDualSpaceTensorPointLexicographic_Internal()
*/
PetscErrorCode PetscDualSpaceLatticePointLexicographic_Internal(PetscInt len, PetscInt max, PetscInt tup[])
{
  PetscFunctionBegin;
  while (len--) {
    max -= tup[len];
    if (!max) {
      tup[len] = 0;
      break;
    }
  }
  tup[++len]++;
  PetscFunctionReturn(0);
}

/*
  PetscDualSpaceTensorPointLexicographic_Internal - Returns all tuples of size 'len' with nonnegative integers that are all less than or equal to 'max'.
                                                    Ordering is lexicographic with lowest index as least significant in ordering.
                                                    e.g. for len == 2 and max == 2, this will return, in order, {0,0}, {1,0}, {2,0}, {0,1}, {1,1}, {2,1}, {0,2}, {1,2}, {2,2}.

  Input Parameters:
+ len - The length of the tuple
. max - The maximum value
- tup - A tuple of length len+1: tup[len] > 0 indicates a stopping condition

  Output Parameter:
. tup - A tuple of len integers whos sum is at most 'max'

  Level: developer

.seealso: PetscDualSpaceLatticePointLexicographic_Internal()
*/
PetscErrorCode PetscDualSpaceTensorPointLexicographic_Internal(PetscInt len, PetscInt max, PetscInt tup[])
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i = 0; i < len; i++) {
    if (tup[i] < max) {
      break;
    } else {
      tup[i] = 0;
    }
  }
  tup[i]++;
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceRegister - Adds a new PetscDualSpace implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscDualSpaceRegister() may be called multiple times to add several user-defined PetscDualSpaces

  Sample usage:
.vb
    PetscDualSpaceRegister("my_space", MyPetscDualSpaceCreate);
.ve

  Then, your PetscDualSpace type can be chosen with the procedural interface via
.vb
    PetscDualSpaceCreate(MPI_Comm, PetscDualSpace *);
    PetscDualSpaceSetType(PetscDualSpace, "my_dual_space");
.ve
   or at runtime via the option
.vb
    -petscdualspace_type my_dual_space
.ve

  Level: advanced

.seealso: PetscDualSpaceRegisterAll(), PetscDualSpaceRegisterDestroy()

@*/
PetscErrorCode PetscDualSpaceRegister(const char sname[], PetscErrorCode (*function)(PetscDualSpace))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscDualSpaceList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceSetType - Builds a particular PetscDualSpace

  Collective on sp

  Input Parameters:
+ sp   - The PetscDualSpace object
- name - The kind of space

  Options Database Key:
. -petscdualspace_type <type> - Sets the PetscDualSpace type; use -help for a list of available types

  Level: intermediate

.seealso: PetscDualSpaceGetType(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceSetType(PetscDualSpace sp, PetscDualSpaceType name)
{
  PetscErrorCode (*r)(PetscDualSpace);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) sp, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscDualSpaceRegisterAllCalled) {ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscDualSpaceList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDualSpace type: %s", name);

  if (sp->ops->destroy) {
    ierr             = (*sp->ops->destroy)(sp);CHKERRQ(ierr);
    sp->ops->destroy = NULL;
  }
  ierr = (*r)(sp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) sp, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceGetType - Gets the PetscDualSpace type name (as a string) from the object.

  Not Collective

  Input Parameter:
. sp  - The PetscDualSpace

  Output Parameter:
. name - The PetscDualSpace type name

  Level: intermediate

.seealso: PetscDualSpaceSetType(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetType(PetscDualSpace sp, PetscDualSpaceType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(name, 2);
  if (!PetscDualSpaceRegisterAllCalled) {
    ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_ASCII(PetscDualSpace sp, PetscViewer v)
{
  PetscViewerFormat format;
  PetscInt          pdim, f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) sp, v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
  if (sp->k) {
    ierr = PetscViewerASCIIPrintf(v, "Dual space for %D-forms %swith %D components, size %D\n", PetscAbsInt(sp->k), sp->k < 0 ? "(stored in dual form) ": "", sp->Nc, pdim);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(v, "Dual space with %D components, size %D\n", sp->Nc, pdim);CHKERRQ(ierr);
  }
  if (sp->ops->view) {ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);}
  ierr = PetscViewerGetFormat(v, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    for (f = 0; f < pdim; ++f) {
      ierr = PetscViewerASCIIPrintf(v, "Dual basis vector %D\n", f);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
      ierr = PetscQuadratureView(sp->functional[f], v);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDualSpaceViewFromOptions - View from Options

   Collective on PetscDualSpace

   Input Parameters:
+  A - the PetscDualSpace object
.  obj - Optional object, proivides prefix
-  name - command line option

   Level: intermediate
.seealso:  PetscDualSpace, PetscDualSpaceView(), PetscObjectViewFromOptions(), PetscDualSpaceCreate()
@*/
PetscErrorCode  PetscDualSpaceViewFromOptions(PetscDualSpace A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCDUALSPACE_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceView - Views a PetscDualSpace

  Collective on sp

  Input Parameters:
+ sp - the PetscDualSpace object to view
- v  - the viewer

  Level: beginner

.seealso PetscDualSpaceDestroy(), PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceView(PetscDualSpace sp, PetscViewer v)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (v) PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscDualSpaceView_ASCII(sp, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetFromOptions - sets parameters in a PetscDualSpace from the options database

  Collective on sp

  Input Parameter:
. sp - the PetscDualSpace object to set options for

  Options Database:
+ -petscdualspace_order <order>      - the approximation order of the space
. -petscdualspace_form_degree <deg>  - the form degree, say 0 for point evaluations, or 2 for area integrals
. -petscdualspace_components <c>     - the number of components, say d for a vector field
. -petscdualspace_refdim <d>         - The spatial dimension of the reference cell
- -petscdualspace_refcell <celltype> - Reference cell type name

  Level: intermediate

.seealso PetscDualSpaceView(), PetscDualSpace, PetscObjectSetFromOptions()
@*/
PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace sp)
{
  PetscDualSpaceReferenceCell refCell = PETSCDUALSPACE_REFCELL_SIMPLEX;
  PetscInt                    refDim  = 0;
  PetscBool                   flg;
  const char                 *defaultType;
  char                        name[256];
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (!((PetscObject) sp)->type_name) {
    defaultType = PETSCDUALSPACELAGRANGE;
  } else {
    defaultType = ((PetscObject) sp)->type_name;
  }
  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscdualspace_type", "Dual space", "PetscDualSpaceSetType", PetscDualSpaceList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscDualSpaceSetType(sp, name);CHKERRQ(ierr);
  } else if (!((PetscObject) sp)->type_name) {
    ierr = PetscDualSpaceSetType(sp, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBoundedInt("-petscdualspace_order", "The approximation order", "PetscDualSpaceSetOrder", sp->order, &sp->order, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscdualspace_form_degree", "The form degree of the dofs", "PetscDualSpaceSetFormDegree", sp->k, &sp->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscdualspace_components", "The number of components", "PetscDualSpaceSetNumComponents", sp->Nc, &sp->Nc, NULL,1);CHKERRQ(ierr);
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(PetscOptionsObject,sp);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBoundedInt("-petscdualspace_refdim", "The spatial dimension of the reference cell", "PetscDualSpaceSetReferenceCell", refDim, &refDim, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-petscdualspace_refcell", "Reference cell", "PetscDualSpaceSetReferenceCell", PetscDualSpaceReferenceCells, (PetscEnum) refCell, (PetscEnum *) &refCell, &flg);CHKERRQ(ierr);
  if (flg) {
    DM K;

    if (!refDim) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_INCOMP, "Reference cell specified without a dimension. Use -petscdualspace_refdim.");
    ierr = PetscDualSpaceCreateReferenceCell(sp, refDim, refCell == PETSCDUALSPACE_REFCELL_SIMPLEX ? PETSC_TRUE : PETSC_FALSE, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(sp, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  sp->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetUp - Construct a basis for the PetscDualSpace

  Collective on sp

  Input Parameter:
. sp - the PetscDualSpace object to setup

  Level: intermediate

.seealso PetscDualSpaceView(), PetscDualSpaceDestroy(), PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (sp->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCDUALSPACE_SetUp, sp, 0, 0, 0);CHKERRQ(ierr);
  sp->setupcalled = PETSC_TRUE;
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(PETSCDUALSPACE_SetUp, sp, 0, 0, 0);CHKERRQ(ierr);
  if (sp->setfromoptionscalled) {ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceClearDMData_Internal(PetscDualSpace sp, DM dm)
{
  PetscInt       pStart = -1, pEnd = -1, depth = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);

  if (sp->pointSpaces) {
    PetscInt i;

    for (i = 0; i < pEnd - pStart; i++) {
      ierr = PetscDualSpaceDestroy(&(sp->pointSpaces[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(sp->pointSpaces);CHKERRQ(ierr);

  if (sp->heightSpaces) {
    PetscInt i;

    for (i = 0; i <= depth; i++) {
      ierr = PetscDualSpaceDestroy(&(sp->heightSpaces[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(sp->heightSpaces);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&(sp->pointSection));CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(sp->intNodes));CHKERRQ(ierr);
  ierr = VecDestroy(&(sp->intDofValues));CHKERRQ(ierr);
  ierr = VecDestroy(&(sp->intNodeValues));CHKERRQ(ierr);
  ierr = MatDestroy(&(sp->intMat));CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(sp->allNodes));CHKERRQ(ierr);
  ierr = VecDestroy(&(sp->allDofValues));CHKERRQ(ierr);
  ierr = VecDestroy(&(sp->allNodeValues));CHKERRQ(ierr);
  ierr = MatDestroy(&(sp->allMat));CHKERRQ(ierr);
  ierr = PetscFree(sp->numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceDestroy - Destroys a PetscDualSpace object

  Collective on sp

  Input Parameter:
. sp - the PetscDualSpace object to destroy

  Level: beginner

.seealso PetscDualSpaceView(), PetscDualSpace(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceDestroy(PetscDualSpace *sp)
{
  PetscInt       dim, f;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCDUALSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;

  ierr = PetscDualSpaceGetDimension(*sp, &dim);CHKERRQ(ierr);
  dm = (*sp)->dm;

  if ((*sp)->ops->destroy) {ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);}
  ierr = PetscDualSpaceClearDMData_Internal(*sp, dm);CHKERRQ(ierr);

  for (f = 0; f < dim; ++f) {
    ierr = PetscQuadratureDestroy(&(*sp)->functional[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*sp)->functional);CHKERRQ(ierr);
  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreate - Creates an empty PetscDualSpace object. The type can then be set with PetscDualSpaceSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscDualSpace object

  Output Parameter:
. sp - The PetscDualSpace object

  Level: beginner

.seealso: PetscDualSpaceSetType(), PETSCDUALSPACELAGRANGE
@*/
PetscErrorCode PetscDualSpaceCreate(MPI_Comm comm, PetscDualSpace *sp)
{
  PetscDualSpace s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sp, 2);
  ierr = PetscCitationsRegister(FECitation,&FEcite);CHKERRQ(ierr);
  *sp  = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(s, PETSCDUALSPACE_CLASSID, "PetscDualSpace", "Dual Space", "PetscDualSpace", comm, PetscDualSpaceDestroy, PetscDualSpaceView);CHKERRQ(ierr);

  s->order       = 0;
  s->Nc          = 1;
  s->k           = 0;
  s->spdim       = -1;
  s->spintdim    = -1;
  s->uniform     = PETSC_TRUE;
  s->setupcalled = PETSC_FALSE;

  *sp = s;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceDuplicate - Creates a duplicate PetscDualSpace object, however it is not setup.

  Collective on sp

  Input Parameter:
. sp - The original PetscDualSpace

  Output Parameter:
. spNew - The duplicate PetscDualSpace

  Level: beginner

.seealso: PetscDualSpaceCreate(), PetscDualSpaceSetType()
@*/
PetscErrorCode PetscDualSpaceDuplicate(PetscDualSpace sp, PetscDualSpace *spNew)
{
  DM             dm;
  PetscDualSpaceType type;
  const char     *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(spNew, 2);
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject)sp), spNew);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sp,     &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *spNew,  name);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetType(sp, &type);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, type);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(*spNew, dm);CHKERRQ(ierr);

  (*spNew)->order   = sp->order;
  (*spNew)->k       = sp->k;
  (*spNew)->Nc      = sp->Nc;
  (*spNew)->uniform = sp->uniform;
  if (sp->ops->duplicate) {ierr = (*sp->ops->duplicate)(sp, *spNew);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetDM - Get the DM representing the reference cell

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. dm - The reference cell

  Level: intermediate

.seealso: PetscDualSpaceSetDM(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace sp, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(dm, 2);
  *dm = sp->dm;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetDM - Get the DM representing the reference cell

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- dm - The reference cell

  Level: intermediate

.seealso: PetscDualSpaceGetDM(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace sp, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (sp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change DM after dualspace is set up");
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  if (sp->dm && sp->dm != dm) {
    ierr = PetscDualSpaceClearDMData_Internal(sp, sp->dm);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&sp->dm);CHKERRQ(ierr);
  sp->dm = dm;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetOrder - Get the order of the dual space

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. order - The order

  Level: intermediate

.seealso: PetscDualSpaceSetOrder(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = sp->order;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetOrder - Set the order of the dual space

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- order - The order

  Level: intermediate

.seealso: PetscDualSpaceGetOrder(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (sp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change order after dualspace is set up");
  sp->order = order;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetNumComponents - Return the number of components for this space

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. Nc - The number of components

  Note: A vector space, for example, will have d components, where d is the spatial dimension

  Level: intermediate

.seealso: PetscDualSpaceSetNumComponents(), PetscDualSpaceGetDimension(), PetscDualSpaceCreate(), PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceGetNumComponents(PetscDualSpace sp, PetscInt *Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(Nc, 2);
  *Nc = sp->Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetNumComponents - Set the number of components for this space

  Input Parameters:
+ sp - The PetscDualSpace
- order - The number of components

  Level: intermediate

.seealso: PetscDualSpaceGetNumComponents(), PetscDualSpaceCreate(), PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceSetNumComponents(PetscDualSpace sp, PetscInt Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (sp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change number of components after dualspace is set up");
  sp->Nc = Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetFunctional - Get the i-th basis functional in the dual space

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- i  - The basis number

  Output Parameter:
. functional - The basis functional

  Level: intermediate

.seealso: PetscDualSpaceGetDimension(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace sp, PetscInt i, PetscQuadrature *functional)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(functional, 3);
  ierr = PetscDualSpaceGetDimension(sp, &dim);CHKERRQ(ierr);
  if ((i < 0) || (i >= dim)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Functional index %d must be in [0, %d)", i, dim);
  *functional = sp->functional[i];
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetDimension - Get the dimension of the dual space, i.e. the number of basis functionals

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. dim - The dimension

  Level: intermediate

.seealso: PetscDualSpaceGetFunctional(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace sp, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  if (sp->spdim < 0) {
    PetscSection section;

    ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
    if (section) {
      ierr = PetscSectionGetStorageSize(section, &(sp->spdim));CHKERRQ(ierr);
    } else sp->spdim = 0;
  }
  *dim = sp->spdim;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetInteriorDimension - Get the interior dimension of the dual space, i.e. the number of basis functionals assigned to the interior of the reference domain

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. dim - The dimension

  Level: intermediate

.seealso: PetscDualSpaceGetFunctional(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetInteriorDimension(PetscDualSpace sp, PetscInt *intdim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(intdim, 2);
  if (sp->spintdim < 0) {
    PetscSection section;

    ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
    if (section) {
      ierr = PetscSectionGetConstrainedStorageSize(section, &(sp->spintdim));CHKERRQ(ierr);
    } else sp->spintdim = 0;
  }
  *intdim = sp->spintdim;
  PetscFunctionReturn(0);
}

/*@
   PetscDualSpaceGetUniform - Whether this dual space is uniform

   Not collective

   Input Parameters:
.  sp - A dual space

   Output Parameters:
.  uniform - PETSC_TRUE if (a) the dual space is the same for each point in a stratum of the reference DMPlex, and
             (b) every symmetry of each point in the reference DMPlex is also a symmetry of the point's dual space.

   Level: advanced

   Note: all of the usual spaces on simplex or tensor-product elements will be uniform, only reference cells
   with non-uniform strata (like trianguar-prisms) or anisotropic hp dual spaces will not be uniform.

.seealso: PetscDualSpaceGetPointSubspace(), PetscDualSpaceGetSymmetries()
@*/
PetscErrorCode PetscDualSpaceGetUniform(PetscDualSpace sp, PetscBool *uniform)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(uniform, 2);
  *uniform = sp->uniform;
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceGetNumDof - Get the number of degrees of freedom for each spatial (topological) dimension

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. numDof - An array of length dim+1 which holds the number of dofs for each dimension

  Level: intermediate

.seealso: PetscDualSpaceGetFunctional(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetNumDof(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  if (!sp->uniform) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "A non-uniform space does not have a fixed number of dofs for each height");
  if (!sp->numDof) {
    DM       dm;
    PetscInt depth, d;
    PetscSection section;

    ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = PetscCalloc1(depth+1,&(sp->numDof));CHKERRQ(ierr);
    ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
    for (d = 0; d <= depth; d++) {
      PetscInt dStart, dEnd;

      ierr = DMPlexGetDepthStratum(dm, d, &dStart, &dEnd);CHKERRQ(ierr);
      if (dEnd <= dStart) continue;
      ierr = PetscSectionGetDof(section, dStart, &(sp->numDof[d]));CHKERRQ(ierr);

    }
  }
  *numDof = sp->numDof;
  if (!*numDof) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_LIB, "Empty numDof[] returned from dual space implementation");
  PetscFunctionReturn(0);
}

/* create the section of the right size and set a permutation for topological ordering */
PetscErrorCode PetscDualSpaceSectionCreate_Internal(PetscDualSpace sp, PetscSection *topSection)
{
  DM             dm;
  PetscInt       pStart, pEnd, cStart, cEnd, c, depth, count, i;
  PetscInt       *seen, *perm;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = sp->dm;
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &section);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd - pStart, &seen);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd - pStart, &perm);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart, count = 0; c < cEnd; c++) {
    PetscInt closureSize = -1, e;
    PetscInt *closure = NULL;

    perm[count++] = c;
    seen[c-pStart] = 1;
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (e = 0; e < closureSize; e++) {
      PetscInt point = closure[2*e];

      if (seen[point-pStart]) continue;
      perm[count++] = point;
      seen[point-pStart] = 1;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  if (count != pEnd - pStart) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad topological ordering");
  for (i = 0; i < pEnd - pStart; i++) if (perm[i] != i) break;
  if (i < pEnd - pStart) {
    IS permIS;

    ierr = ISCreateGeneral(PETSC_COMM_SELF, pEnd - pStart, perm, PETSC_OWN_POINTER, &permIS);CHKERRQ(ierr);
    ierr = ISSetPermutation(permIS);CHKERRQ(ierr);
    ierr = PetscSectionSetPermutation(section, permIS);CHKERRQ(ierr);
    ierr = ISDestroy(&permIS);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(perm);CHKERRQ(ierr);
  }
  ierr = PetscFree(seen);CHKERRQ(ierr);
  *topSection = section;
  PetscFunctionReturn(0);
}

/* mark boundary points and set up */
PetscErrorCode PetscDualSpaceSectionSetUp_Internal(PetscDualSpace sp, PetscSection section)
{
  DM             dm;
  DMLabel        boundary;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = sp->dm;
  ierr = DMLabelCreate(PETSC_COMM_SELF,"boundary",&boundary);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm,1,boundary);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm,boundary);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt bval;

    ierr = DMLabelGetValue(boundary, p, &bval);CHKERRQ(ierr);
    if (bval == 1) {
      PetscInt dof;

      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintDof(section, p, dof);CHKERRQ(ierr);
    }
  }
  ierr = DMLabelDestroy(&boundary);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetSection - Create a PetscSection over the reference cell with the layout from this space

  Collective on sp

  Input Parameters:
. sp      - The PetscDualSpace

  Output Parameter:
. section - The section

  Level: advanced

.seealso: PetscDualSpaceCreate(), DMPLEX
@*/
PetscErrorCode PetscDualSpaceGetSection(PetscDualSpace sp, PetscSection *section)
{
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sp->pointSection) {
    /* mark the boundary */
    ierr = PetscDualSpaceSectionCreate_Internal(sp, &(sp->pointSection));CHKERRQ(ierr);
    ierr = DMPlexGetChart(sp->dm,&pStart,&pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscDualSpace psp;

      ierr = PetscDualSpaceGetPointSubspace(sp, p, &psp);CHKERRQ(ierr);
      if (psp) {
        PetscInt dof;

        ierr = PetscDualSpaceGetInteriorDimension(psp, &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(sp->pointSection,p,dof);CHKERRQ(ierr);
      }
    }
    ierr = PetscDualSpaceSectionSetUp_Internal(sp,sp->pointSection);CHKERRQ(ierr);
  }
  *section = sp->pointSection;
  PetscFunctionReturn(0);
}

/* this assumes that all of the point dual spaces store their interior dofs first, which is true when the point DMs
 * have one cell */
PetscErrorCode PetscDualSpacePushForwardSubspaces_Internal(PetscDualSpace sp, PetscInt sStart, PetscInt sEnd)
{
  PetscReal *sv0, *v0, *J;
  PetscSection section;
  PetscInt     dim, s, k;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim, &v0, dim, &sv0, dim*dim, &J);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFormDegree(sp, &k);CHKERRQ(ierr);
  for (s = sStart; s < sEnd; s++) {
    PetscReal detJ, hdetJ;
    PetscDualSpace ssp;
    PetscInt dof, off, f, sdim;
    PetscInt i, j;
    DM sdm;

    ierr = PetscDualSpaceGetPointSubspace(sp, s, &ssp);CHKERRQ(ierr);
    if (!ssp) continue;
    ierr = PetscSectionGetDof(section, s, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, s, &off);CHKERRQ(ierr);
    /* get the first vertex of the reference cell */
    ierr = PetscDualSpaceGetDM(ssp, &sdm);CHKERRQ(ierr);
    ierr = DMGetDimension(sdm, &sdim);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryAffineFEM(sdm, 0, sv0, NULL, NULL, &hdetJ);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryAffineFEM(dm, s, v0, J, NULL, &detJ);CHKERRQ(ierr);
    /* compactify Jacobian */
    for (i = 0; i < dim; i++) for (j = 0; j < sdim; j++) J[i* sdim + j] = J[i * dim + j];
    for (f = 0; f < dof; f++) {
      PetscQuadrature fn;

      ierr = PetscDualSpaceGetFunctional(ssp, f, &fn);CHKERRQ(ierr);
      ierr = PetscQuadraturePushForward(fn, dim, sv0, v0, J, k, &(sp->functional[off+f]));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(v0, sv0, J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreateReferenceCell - Create a DMPLEX with the appropriate FEM reference cell

  Collective on sp

  Input Parameters:
+ sp      - The PetscDualSpace
. dim     - The spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameter:
. refdm - The reference cell

  Note: This DM is on PETSC_COMM_SELF.

  Level: intermediate

.seealso: PetscDualSpaceCreate(), DMPLEX
@*/
PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace sp, PetscInt dim, PetscBool simplex, DM *refdm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, simplex), refdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApply - Apply a functional from the dual space basis to an input function

  Input Parameters:
+ sp      - The PetscDualSpace object
. f       - The basis functional index
. time    - The time
. cgeom   - A context with geometric information for this cell, we use v0 (the initial vertex) and J (the Jacobian) (or evaluated at the coordinates of the functional)
. numComp - The number of components for the function
. func    - The input function
- ctx     - A context for the function

  Output Parameter:
. value   - numComp output values

  Note: The calling sequence for the callback func is given by:

$ func(PetscInt dim, PetscReal time, const PetscReal x[],
$      PetscInt numComponents, PetscScalar values[], void *ctx)

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApply(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFEGeom *cgeom, PetscInt numComp, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(cgeom, 4);
  PetscValidPointer(value, 8);
  ierr = (*sp->ops->apply)(sp, f, time, cgeom, numComp, func, ctx, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyAll - Apply all functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetAllData()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetAllData()

  Output Parameter:
. spValue   - The values of all dual space functionals

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyAll(PetscDualSpace sp, const PetscScalar *pointEval, PetscScalar *spValue)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = (*sp->ops->applyall)(sp, pointEval, spValue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyInterior - Apply interior functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetInteriorData()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetInteriorData()

  Output Parameter:
. spValue   - The values of interior dual space functionals

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyInterior(PetscDualSpace sp, const PetscScalar *pointEval, PetscScalar *spValue)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = (*sp->ops->applyint)(sp, pointEval, spValue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyDefault - Apply a functional from the dual space basis to an input function by assuming a point evaluation functional.

  Input Parameters:
+ sp    - The PetscDualSpace object
. f     - The basis functional index
. time  - The time
. cgeom - A context with geometric information for this cell, we use v0 (the initial vertex) and J (the Jacobian)
. Nc    - The number of components for the function
. func  - The input function
- ctx   - A context for the function

  Output Parameter:
. value   - The output value

  Note: The calling sequence for the callback func is given by:

$ func(PetscInt dim, PetscReal time, const PetscReal x[],
$      PetscInt numComponents, PetscScalar values[], void *ctx)

and the idea is to evaluate the functional as an integral

$ n(f) = int dx n(x) . f(x)

where both n and f have Nc components.

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyDefault(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFEGeom *cgeom, PetscInt Nc, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  n;
  const PetscReal *points, *weights;
  PetscReal        x[3];
  PetscScalar     *val;
  PetscInt         dim, dE, qNc, c, Nq, q;
  PetscBool        isAffine;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 8);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(n, &dim, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
  if (dim != cgeom->dim) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature spatial dimension %D != cell geometry dimension %D", dim, cgeom->dim);
  if (qNc != Nc) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
  ierr = DMGetWorkArray(dm, Nc, MPIU_SCALAR, &val);CHKERRQ(ierr);
  *value = 0.0;
  isAffine = cgeom->isAffine;
  dE = cgeom->dimEmbed;
  for (q = 0; q < Nq; ++q) {
    if (isAffine) {
      CoordinatesRefToReal(dE, cgeom->dim, cgeom->xi, cgeom->v, cgeom->J, &points[q*dim], x);
      ierr = (*func)(dE, time, x, Nc, val, ctx);CHKERRQ(ierr);
    } else {
      ierr = (*func)(dE, time, &cgeom->v[dE*q], Nc, val, ctx);CHKERRQ(ierr);
    }
    for (c = 0; c < Nc; ++c) {
      *value += val[c]*weights[q*Nc+c];
    }
  }
  ierr = DMRestoreWorkArray(dm, Nc, MPIU_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyAllDefault - Apply all functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetAllData()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetAllData()

  Output Parameter:
. spValue   - The values of all dual space functionals

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyAllDefault(PetscDualSpace sp, const PetscScalar *pointEval, PetscScalar *spValue)
{
  Vec              pointValues, dofValues;
  Mat              allMat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidScalarPointer(pointEval, 2);
  PetscValidScalarPointer(spValue, 3);
  ierr = PetscDualSpaceGetAllData(sp, NULL, &allMat);CHKERRQ(ierr);
  if (!(sp->allNodeValues)) {
    ierr = MatCreateVecs(allMat, &(sp->allNodeValues), NULL);CHKERRQ(ierr);
  }
  pointValues = sp->allNodeValues;
  if (!(sp->allDofValues)) {
    ierr = MatCreateVecs(allMat, NULL, &(sp->allDofValues));CHKERRQ(ierr);
  }
  dofValues = sp->allDofValues;
  ierr = VecPlaceArray(pointValues, pointEval);CHKERRQ(ierr);
  ierr = VecPlaceArray(dofValues, spValue);CHKERRQ(ierr);
  ierr = MatMult(allMat, pointValues, dofValues);CHKERRQ(ierr);
  ierr = VecResetArray(dofValues);CHKERRQ(ierr);
  ierr = VecResetArray(pointValues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyInteriorDefault - Apply interior functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetInteriorData()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetInteriorData()

  Output Parameter:
. spValue   - The values of interior dual space functionals

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyInteriorDefault(PetscDualSpace sp, const PetscScalar *pointEval, PetscScalar *spValue)
{
  Vec              pointValues, dofValues;
  Mat              intMat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidScalarPointer(pointEval, 2);
  PetscValidScalarPointer(spValue, 3);
  ierr = PetscDualSpaceGetInteriorData(sp, NULL, &intMat);CHKERRQ(ierr);
  if (!(sp->intNodeValues)) {
    ierr = MatCreateVecs(intMat, &(sp->intNodeValues), NULL);CHKERRQ(ierr);
  }
  pointValues = sp->intNodeValues;
  if (!(sp->intDofValues)) {
    ierr = MatCreateVecs(intMat, NULL, &(sp->intDofValues));CHKERRQ(ierr);
  }
  dofValues = sp->intDofValues;
  ierr = VecPlaceArray(pointValues, pointEval);CHKERRQ(ierr);
  ierr = VecPlaceArray(dofValues, spValue);CHKERRQ(ierr);
  ierr = MatMult(intMat, pointValues, dofValues);CHKERRQ(ierr);
  ierr = VecResetArray(dofValues);CHKERRQ(ierr);
  ierr = VecResetArray(pointValues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetAllData - Get all quadrature nodes from this space, and the matrix that sends quadrature node values to degree-of-freedom values

  Input Parameter:
. sp - The dualspace

  Output Parameters:
+ allNodes - A PetscQuadrature object containing all evaluation nodes
- allMat - A Mat for the node-to-dof transformation

  Level: advanced

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetAllData(PetscDualSpace sp, PetscQuadrature *allNodes, Mat *allMat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (allNodes) PetscValidPointer(allNodes,2);
  if (allMat) PetscValidPointer(allMat,3);
  if ((!sp->allNodes || !sp->allMat) && sp->ops->createalldata) {
    PetscQuadrature qpoints;
    Mat amat;

    ierr = (*sp->ops->createalldata)(sp,&qpoints,&amat);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&(sp->allNodes));CHKERRQ(ierr);
    ierr = MatDestroy(&(sp->allMat));CHKERRQ(ierr);
    sp->allNodes = qpoints;
    sp->allMat = amat;
  }
  if (allNodes) *allNodes = sp->allNodes;
  if (allMat) *allMat = sp->allMat;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreateAllDataDefault - Create all evaluation nodes and the node-to-dof matrix by examining functionals

  Input Parameter:
. sp - The dualspace

  Output Parameters:
+ allNodes - A PetscQuadrature object containing all evaluation nodes
- allMat - A Mat for the node-to-dof transformation

  Level: advanced

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceCreateAllDataDefault(PetscDualSpace sp, PetscQuadrature *allNodes, Mat *allMat)
{
  PetscInt        spdim;
  PetscInt        numPoints, offset;
  PetscReal       *points;
  PetscInt        f, dim;
  PetscInt        Nc, nrows, ncols;
  PetscInt        maxNumPoints;
  PetscQuadrature q;
  Mat             A;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp,&spdim);CHKERRQ(ierr);
  if (!spdim) {
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,allNodes);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(*allNodes,0,0,0,NULL,NULL);CHKERRQ(ierr);
  }
  nrows = spdim;
  ierr = PetscDualSpaceGetFunctional(sp,0,&q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q,&dim,NULL,&numPoints,NULL,NULL);CHKERRQ(ierr);
  maxNumPoints = numPoints;
  for (f = 1; f < spdim; f++) {
    PetscInt Np;

    ierr = PetscDualSpaceGetFunctional(sp,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&Np,NULL,NULL);CHKERRQ(ierr);
    numPoints += Np;
    maxNumPoints = PetscMax(maxNumPoints,Np);
  }
  ncols = numPoints * Nc;
  ierr = PetscMalloc1(dim*numPoints,&points);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, ncols, maxNumPoints * Nc, NULL, &A);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < spdim; f++) {
    const PetscReal *p, *w;
    PetscInt        Np, i;
    PetscInt        fnc;

    ierr = PetscDualSpaceGetFunctional(sp,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,&fnc,&Np,&p,&w);CHKERRQ(ierr);
    if (fnc != Nc) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "functional component mismatch");
    for (i = 0; i < Np * dim; i++) {
      points[offset* dim + i] = p[i];
    }
    for (i = 0; i < Np * Nc; i++) {
      ierr = MatSetValue(A, f, offset * Nc, w[i], INSERT_VALUES);CHKERRQ(ierr);
    }
    offset += Np;
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,allNodes);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*allNodes,dim,0,numPoints,points,NULL);CHKERRQ(ierr);
  *allMat = A;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetInteriorData - Get all quadrature points necessary to compute the interior degrees of freedom from
  this space, as well as the matrix that computes the degrees of freedom from the quadrature values.  Degrees of
  freedom are interior degrees of freedom if they belong (by PetscDualSpaceGetSection()) to interior points in the
  reference DMPlex: complementary boundary degrees of freedom are marked as constrained in the section returned by
  PetscDualSpaceGetSection()).

  Input Parameter:
. sp - The dualspace

  Output Parameters:
+ intNodes - A PetscQuadrature object containing all evaluation points needed to evaluate interior degrees of freedom
- intMat   - A matrix that computes dual space values from point values: size [spdim0 x (npoints * nc)], where spdim0 is
             the size of the constrained layout (PetscSectionGetConstrainStorageSize()) of the dual space section,
             npoints is the number of points in intNodes and nc is PetscDualSpaceGetNumComponents().

  Level: advanced

.seealso: PetscDualSpaceCreate(), PetscDualSpaceGetDimension(), PetscDualSpaceGetNumComponents(), PetscQuadratureGetData()
@*/
PetscErrorCode PetscDualSpaceGetInteriorData(PetscDualSpace sp, PetscQuadrature *intNodes, Mat *intMat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (intNodes) PetscValidPointer(intNodes,2);
  if (intMat) PetscValidPointer(intMat,3);
  if ((!sp->intNodes || !sp->intMat) && sp->ops->createintdata) {
    PetscQuadrature qpoints;
    Mat imat;

    ierr = (*sp->ops->createintdata)(sp,&qpoints,&imat);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&(sp->intNodes));CHKERRQ(ierr);
    ierr = MatDestroy(&(sp->intMat));CHKERRQ(ierr);
    sp->intNodes = qpoints;
    sp->intMat = imat;
  }
  if (intNodes) *intNodes = sp->intNodes;
  if (intMat) *intMat = sp->intMat;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreateInteriorDataDefault - Create quadrature points by examining interior functionals and create the matrix mapping quadrature point values to interior dual space values

  Input Parameter:
. sp - The dualspace

  Output Parameters:
+ intNodes - A PetscQuadrature object containing all evaluation points needed to evaluate interior degrees of freedom
- intMat    - A matrix that computes dual space values from point values: size [spdim0 x (npoints * nc)], where spdim0 is
              the size of the constrained layout (PetscSectionGetConstrainStorageSize()) of the dual space section,
              npoints is the number of points in allNodes and nc is PetscDualSpaceGetNumComponents().

  Level: advanced

.seealso: PetscDualSpaceCreate(), PetscDualSpaceGetInteriorData()
@*/
PetscErrorCode PetscDualSpaceCreateInteriorDataDefault(PetscDualSpace sp, PetscQuadrature *intNodes, Mat *intMat)
{
  DM              dm;
  PetscInt        spdim0;
  PetscInt        Nc;
  PetscInt        pStart, pEnd, p, f;
  PetscSection    section;
  PetscInt        numPoints, offset, matoffset;
  PetscReal       *points;
  PetscInt        dim;
  PetscInt        *nnz;
  PetscQuadrature q;
  Mat             imat;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCDUALSPACE_CLASSID,1);
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(section, &spdim0);CHKERRQ(ierr);
  if (!spdim0) {
    *intNodes = NULL;
    *intMat = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc1(spdim0, &nnz);CHKERRQ(ierr);
  for (p = pStart, f = 0, numPoints = 0; p < pEnd; p++) {
    PetscInt dof, cdof, off, d;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (!(dof - cdof)) continue;
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    for (d = 0; d < dof; d++, off++, f++) {
      PetscInt Np;

      ierr = PetscDualSpaceGetFunctional(sp,off,&q);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(q,NULL,NULL,&Np,NULL,NULL);CHKERRQ(ierr);
      nnz[f] = Np * Nc;
      numPoints += Np;
    }
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, spdim0, numPoints * Nc, 0, nnz, &imat);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*numPoints,&points);CHKERRQ(ierr);
  for (p = pStart, f = 0, offset = 0, matoffset = 0; p < pEnd; p++) {
    PetscInt dof, cdof, off, d;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (!(dof - cdof)) continue;
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    for (d = 0; d < dof; d++, off++, f++) {
      const PetscReal *p;
      const PetscReal *w;
      PetscInt        Np, i;

      ierr = PetscDualSpaceGetFunctional(sp,off,&q);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(q,NULL,NULL,&Np,&p,&w);CHKERRQ(ierr);
      for (i = 0; i < Np * dim; i++) {
        points[offset + i] = p[i];
      }
      for (i = 0; i < Np * Nc; i++) {
        ierr = MatSetValue(imat, f, matoffset + i, w[i],INSERT_VALUES);CHKERRQ(ierr);
      }
      offset += Np * dim;
      matoffset += Np * Nc;
    }
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,intNodes);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*intNodes,dim,0,numPoints,points,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(imat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(imat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *intMat = imat;
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceApplyFVM - Apply a functional from the dual space basis to an input function by assuming a point evaluation functional at the cell centroid.

  Input Parameters:
+ sp    - The PetscDualSpace object
. f     - The basis functional index
. time  - The time
. cgeom - A context with geometric information for this cell, we currently just use the centroid
. Nc    - The number of components for the function
. func  - The input function
- ctx   - A context for the function

  Output Parameter:
. value - The output value (scalar)

  Note: The calling sequence for the callback func is given by:

$ func(PetscInt dim, PetscReal time, const PetscReal x[],
$      PetscInt numComponents, PetscScalar values[], void *ctx)

and the idea is to evaluate the functional as an integral

$ n(f) = int dx n(x) . f(x)

where both n and f have Nc components.

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyFVM(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFVCellGeom *cgeom, PetscInt Nc, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  n;
  const PetscReal *points, *weights;
  PetscScalar     *val;
  PetscInt         dimEmbed, qNc, c, Nq, q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 8);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(n, NULL, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
  if (qNc != Nc) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
  ierr = DMGetWorkArray(dm, Nc, MPIU_SCALAR, &val);CHKERRQ(ierr);
  *value = 0.;
  for (q = 0; q < Nq; ++q) {
    ierr = (*func)(dimEmbed, time, cgeom->centroid, Nc, val, ctx);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      *value += val[c]*weights[q*Nc+c];
    }
  }
  ierr = DMRestoreWorkArray(dm, Nc, MPIU_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetHeightSubspace - Get the subset of the dual space basis that is supported on a mesh point of a
  given height.  This assumes that the reference cell is symmetric over points of this height.

  If the dual space is not defined on mesh points of the given height (e.g. if the space is discontinuous and
  pointwise values are not defined on the element boundaries), or if the implementation of PetscDualSpace does not
  support extracting subspaces, then NULL is returned.

  This does not increment the reference count on the returned dual space, and the user should not destroy it.

  Not collective

  Input Parameters:
+ sp - the PetscDualSpace object
- height - the height of the mesh point for which the subspace is desired

  Output Parameter:
. subsp - the subspace.  Note that the functionals in the subspace are with respect to the intrinsic geometry of the
  point, which will be of lesser dimension if height > 0.

  Level: advanced

.seealso: PetscSpaceGetHeightSubspace(), PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceGetHeightSubspace(PetscDualSpace sp, PetscInt height, PetscDualSpace *subsp)
{
  PetscInt       depth = -1, cStart, cEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(subsp,3);
  if (!(sp->uniform)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "A non-uniform dual space does not have a single dual space at each height");
  *subsp = NULL;
  dm = sp->dm;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (height < 0 || height > depth) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid height");
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  if (height == 0 && cEnd == cStart + 1) {
    *subsp = sp;
    PetscFunctionReturn(0);
  }
  if (!sp->heightSpaces) {
    PetscInt h;
    ierr = PetscCalloc1(depth+1, &(sp->heightSpaces));CHKERRQ(ierr);

    for (h = 0; h <= depth; h++) {
      if (h == 0 && cEnd == cStart + 1) continue;
      if (sp->ops->createheightsubspace) {ierr = (*sp->ops->createheightsubspace)(sp,height,&(sp->heightSpaces[h]));CHKERRQ(ierr);}
      else if (sp->pointSpaces) {
        PetscInt hStart, hEnd;

        ierr = DMPlexGetHeightStratum(dm,h,&hStart,&hEnd);CHKERRQ(ierr);
        if (hEnd > hStart) {
          const char *name;

          ierr = PetscObjectReference((PetscObject)(sp->pointSpaces[hStart]));CHKERRQ(ierr);
          if (sp->pointSpaces[hStart]) {
            ierr = PetscObjectGetName((PetscObject) sp,                     &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) sp->pointSpaces[hStart], name);CHKERRQ(ierr);
          }
          sp->heightSpaces[h] = sp->pointSpaces[hStart];
        }
      }
    }
  }
  *subsp = sp->heightSpaces[height];
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetPointSubspace - Get the subset of the dual space basis that is supported on a particular mesh point.

  If the dual space is not defined on the mesh point (e.g. if the space is discontinuous and pointwise values are not
  defined on the element boundaries), or if the implementation of PetscDualSpace does not support extracting
  subspaces, then NULL is returned.

  This does not increment the reference count on the returned dual space, and the user should not destroy it.

  Not collective

  Input Parameters:
+ sp - the PetscDualSpace object
- point - the point (in the dual space's DM) for which the subspace is desired

  Output Parameters:
  bdsp - the subspace.  Note that the functionals in the subspace are with respect to the intrinsic geometry of the
  point, which will be of lesser dimension if height > 0.

  Level: advanced

.seealso: PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceGetPointSubspace(PetscDualSpace sp, PetscInt point, PetscDualSpace *bdsp)
{
  PetscInt       pStart = 0, pEnd = 0, cStart, cEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp,3);
  *bdsp = NULL;
  dm = sp->dm;
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (point < pStart || point > pEnd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid point");
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  if (point == cStart && cEnd == cStart + 1) { /* the dual space is only equivalent to the dual space on a cell if the reference mesh has just one cell */
    *bdsp = sp;
    PetscFunctionReturn(0);
  }
  if (!sp->pointSpaces) {
    PetscInt p;
    ierr = PetscCalloc1(pEnd - pStart, &(sp->pointSpaces));CHKERRQ(ierr);

    for (p = 0; p < pEnd - pStart; p++) {
      if (p + pStart == cStart && cEnd == cStart + 1) continue;
      if (sp->ops->createpointsubspace) {ierr = (*sp->ops->createpointsubspace)(sp,p+pStart,&(sp->pointSpaces[p]));CHKERRQ(ierr);}
      else if (sp->heightSpaces || sp->ops->createheightsubspace) {
        PetscInt dim, depth, height;
        DMLabel  label;

        ierr = DMPlexGetDepth(dm,&dim);CHKERRQ(ierr);
        ierr = DMPlexGetDepthLabel(dm,&label);CHKERRQ(ierr);
        ierr = DMLabelGetValue(label,p+pStart,&depth);CHKERRQ(ierr);
        height = dim - depth;
        ierr = PetscDualSpaceGetHeightSubspace(sp, height, &(sp->pointSpaces[p]));CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)sp->pointSpaces[p]);CHKERRQ(ierr);
      }
    }
  }
  *bdsp = sp->pointSpaces[point - pStart];
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceGetSymmetries - Returns a description of the symmetries of this basis

  Not collective

  Input Parameter:
. sp - the PetscDualSpace object

  Output Parameters:
+ perms - Permutations of the interior degrees of freedom, parameterized by the point orientation
- flips - Sign reversal of the interior degrees of freedom, parameterized by the point orientation

  Note: The permutation and flip arrays are organized in the following way
$ perms[p][ornt][dof # on point] = new local dof #
$ flips[p][ornt][dof # on point] = reversal or not

  Level: developer

@*/
PetscErrorCode PetscDualSpaceGetSymmetries(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCDUALSPACE_CLASSID,1);
  if (perms) {PetscValidPointer(perms,2); *perms = NULL;}
  if (flips) {PetscValidPointer(flips,3); *flips = NULL;}
  if (sp->ops->getsymmetries) {ierr = (sp->ops->getsymmetries)(sp,perms,flips);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetFormDegree - Get the form degree k for the k-form the describes the pushforwards/pullbacks of this
  dual space's functionals.

  Input Parameter:
. dsp - The PetscDualSpace

  Output Parameter:
. k   - The *signed* degree k of the k.  If k >= 0, this means that the degrees of freedom are k-forms, and are stored
        in lexicographic order according to the basis of k-forms constructed from the wedge product of 1-forms.  So for example,
        the 1-form basis in 3-D is (dx, dy, dz), and the 2-form basis in 3-D is (dx wedge dy, dx wedge dz, dy wedge dz).
        If k < 0, this means that the degrees transform as k-forms, but are stored as (N-k) forms according to the
        Hodge star map.  So for example if k = -2 and N = 3, this means that the degrees of freedom transform as 2-forms
        but are stored as 1-forms.

  Level: developer

.seealso: PetscDTAltV, PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransform(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceGetFormDegree(PetscDualSpace dsp, PetscInt *k)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(k, 2);
  *k = dsp->k;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetFormDegree - Set the form degree k for the k-form the describes the pushforwards/pullbacks of this
  dual space's functionals.

  Input Parameters:
+ dsp - The PetscDualSpace
- k   - The *signed* degree k of the k.  If k >= 0, this means that the degrees of freedom are k-forms, and are stored
        in lexicographic order according to the basis of k-forms constructed from the wedge product of 1-forms.  So for example,
        the 1-form basis in 3-D is (dx, dy, dz), and the 2-form basis in 3-D is (dx wedge dy, dx wedge dz, dy wedge dz).
        If k < 0, this means that the degrees transform as k-forms, but are stored as (N-k) forms according to the
        Hodge star map.  So for example if k = -2 and N = 3, this means that the degrees of freedom transform as 2-forms
        but are stored as 1-forms.

  Level: developer

.seealso: PetscDTAltV, PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransform(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceSetFormDegree(PetscDualSpace dsp, PetscInt k)
{
  PetscInt dim;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  if (dsp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dsp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change number of components after dualspace is set up");
  dim = dsp->dm->dim;
  if (k < -dim || k > dim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported %D-form on %D-dimensional reference cell", PetscAbsInt(k), dim);
  dsp->k = k;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetDeRahm - Get the k-simplex associated with the functionals in this dual space

  Input Parameter:
. dsp - The PetscDualSpace

  Output Parameter:
. k   - The simplex dimension

  Level: developer

  Note: Currently supported values are
$ 0: These are H_1 methods that only transform coordinates
$ 1: These are Hcurl methods that transform functions using the covariant Piola transform (COVARIANT_PIOLA_TRANSFORM)
$ 2: These are the same as 1
$ 3: These are Hdiv methods that transform functions using the contravariant Piola transform (CONTRAVARIANT_PIOLA_TRANSFORM)

.seealso: PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransform(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceGetDeRahm(PetscDualSpace dsp, PetscInt *k)
{
  PetscInt dim;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(k, 2);
  dim = dsp->dm->dim;
  if (!dsp->k) *k = IDENTITY_TRANSFORM;
  else if (dsp->k == 1) *k = COVARIANT_PIOLA_TRANSFORM;
  else if (dsp->k == -(dim - 1)) *k = CONTRAVARIANT_PIOLA_TRANSFORM;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported transformation");
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceTransform - Transform the function values

  Input Parameters:
+ dsp       - The PetscDualSpace
. trans     - The type of transform
. isInverse - Flag to invert the transform
. fegeom    - The cell geometry
. Nv        - The number of function samples
. Nc        - The number of function components
- vals      - The function values

  Output Parameter:
. vals      - The transformed function values

  Level: intermediate

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpaceTransformGradient(), PetscDualSpaceTransformHessian(), PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceTransform(PetscDualSpace dsp, PetscDualSpaceTransformType trans, PetscBool isInverse, PetscFEGeom *fegeom, PetscInt Nv, PetscInt Nc, PetscScalar vals[])
{
  PetscReal Jstar[9] = {0};
  PetscInt dim, v, c, Nk;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 4);
  PetscValidPointer(vals, 7);
  /* TODO: not handling dimEmbed != dim right now */
  dim = dsp->dm->dim;
  /* No change needed for 0-forms */
  if (!dsp->k) PetscFunctionReturn(0);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(dsp->k), &Nk);CHKERRQ(ierr);
  /* TODO: use fegeom->isAffine */
  ierr = PetscDTAltVPullbackMatrix(dim, dim, isInverse ? fegeom->J : fegeom->invJ, dsp->k, Jstar);CHKERRQ(ierr);
  for (v = 0; v < Nv; ++v) {
    switch (Nk) {
    case 1:
      for (c = 0; c < Nc; c++) vals[v*Nc + c] *= Jstar[0];
      break;
    case 2:
      for (c = 0; c < Nc; c += 2) DMPlex_Mult2DReal_Internal(Jstar, 1, &vals[v*Nc + c], &vals[v*Nc + c]);
      break;
    case 3:
      for (c = 0; c < Nc; c += 3) DMPlex_Mult3DReal_Internal(Jstar, 1, &vals[v*Nc + c], &vals[v*Nc + c]);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported form size %D for transformation", Nk);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceTransformGradient - Transform the function gradient values

  Input Parameters:
+ dsp       - The PetscDualSpace
. trans     - The type of transform
. isInverse - Flag to invert the transform
. fegeom    - The cell geometry
. Nv        - The number of function gradient samples
. Nc        - The number of function components
- vals      - The function gradient values

  Output Parameter:
. vals      - The transformed function gradient values

  Level: intermediate

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpaceTransform(), PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceTransformGradient(PetscDualSpace dsp, PetscDualSpaceTransformType trans, PetscBool isInverse, PetscFEGeom *fegeom, PetscInt Nv, PetscInt Nc, PetscScalar vals[])
{
  const PetscInt dim = dsp->dm->dim, dE = fegeom->dimEmbed;
  PetscInt       v, c, d;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 4);
  PetscValidPointer(vals, 7);
#ifdef PETSC_USE_DEBUG
  if (dE <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid embedding dimension %D", dE);
#endif
  /* Transform gradient */
  if (dim == dE) {
    for (v = 0; v < Nv; ++v) {
      for (c = 0; c < Nc; ++c) {
        switch (dim)
        {
          case 1: vals[(v*Nc+c)*dim] *= fegeom->invJ[0];break;
          case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->invJ, 1, &vals[(v*Nc+c)*dim], &vals[(v*Nc+c)*dim]);break;
          case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->invJ, 1, &vals[(v*Nc+c)*dim], &vals[(v*Nc+c)*dim]);break;
          default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
      }
    }
  } else {
    for (v = 0; v < Nv; ++v) {
      for (c = 0; c < Nc; ++c) {
        DMPlex_MultTransposeReal_Internal(fegeom->invJ, dim, dE, 1, &vals[(v*Nc+c)*dE], &vals[(v*Nc+c)*dE]);
      }
    }
  }
  /* Assume its a vector, otherwise assume its a bunch of scalars */
  if (Nc == 1 || Nc != dim) PetscFunctionReturn(0);
  switch (trans) {
    case IDENTITY_TRANSFORM: break;
    case COVARIANT_PIOLA_TRANSFORM: /* Covariant Piola mapping $\sigma^*(F) = J^{-T} F \circ \phi^{-1)$ */
    if (isInverse) {
      for (v = 0; v < Nv; ++v) {
        for (d = 0; d < dim; ++d) {
          switch (dim)
          {
            case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->J, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->J, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
          }
        }
      }
    } else {
      for (v = 0; v < Nv; ++v) {
        for (d = 0; d < dim; ++d) {
          switch (dim)
          {
            case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->invJ, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->invJ, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
          }
        }
      }
    }
    break;
    case CONTRAVARIANT_PIOLA_TRANSFORM: /* Contravariant Piola mapping $\sigma^*(F) = \frac{1}{|\det J|} J F \circ \phi^{-1}$ */
    if (isInverse) {
      for (v = 0; v < Nv; ++v) {
        for (d = 0; d < dim; ++d) {
          switch (dim)
          {
            case 2: DMPlex_Mult2DReal_Internal(fegeom->invJ, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            case 3: DMPlex_Mult3DReal_Internal(fegeom->invJ, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
          }
          for (c = 0; c < Nc; ++c) vals[(v*Nc+c)*dim+d] *= fegeom->detJ[0];
        }
      }
    } else {
      for (v = 0; v < Nv; ++v) {
        for (d = 0; d < dim; ++d) {
          switch (dim)
          {
            case 2: DMPlex_Mult2DReal_Internal(fegeom->J, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            case 3: DMPlex_Mult3DReal_Internal(fegeom->J, dim, &vals[v*Nc*dim+d], &vals[v*Nc*dim+d]);break;
            default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
          }
          for (c = 0; c < Nc; ++c) vals[(v*Nc+c)*dim+d] /= fegeom->detJ[0];
        }
      }
    }
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceTransformHessian - Transform the function Hessian values

  Input Parameters:
+ dsp       - The PetscDualSpace
. trans     - The type of transform
. isInverse - Flag to invert the transform
. fegeom    - The cell geometry
. Nv        - The number of function Hessian samples
. Nc        - The number of function components
- vals      - The function gradient values

  Output Parameter:
. vals      - The transformed function Hessian values

  Level: intermediate

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpaceTransform(), PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceTransformHessian(PetscDualSpace dsp, PetscDualSpaceTransformType trans, PetscBool isInverse, PetscFEGeom *fegeom, PetscInt Nv, PetscInt Nc, PetscScalar vals[])
{
  const PetscInt dim = dsp->dm->dim, dE = fegeom->dimEmbed;
  PetscInt       v, c;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 4);
  PetscValidPointer(vals, 7);
#ifdef PETSC_USE_DEBUG
  if (dE <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid embedding dimension %D", dE);
#endif
  /* Transform Hessian: J^{-T}_{ik} J^{-T}_{jl} H(f)_{kl} = J^{-T}_{ik} H(f)_{kl} J^{-1}_{lj} */
  if (dim == dE) {
    for (v = 0; v < Nv; ++v) {
      for (c = 0; c < Nc; ++c) {
        switch (dim)
        {
          case 1: vals[(v*Nc+c)*dim*dim] *= PetscSqr(fegeom->invJ[0]);break;
          case 2: DMPlex_PTAP2DReal_Internal(fegeom->invJ, &vals[(v*Nc+c)*dim*dim], &vals[(v*Nc+c)*dim*dim]);break;
          case 3: DMPlex_PTAP3DReal_Internal(fegeom->invJ, &vals[(v*Nc+c)*dim*dim], &vals[(v*Nc+c)*dim*dim]);break;
          default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
      }
    }
  } else {
    for (v = 0; v < Nv; ++v) {
      for (c = 0; c < Nc; ++c) {
        DMPlex_PTAPReal_Internal(fegeom->invJ, dim, dE, &vals[(v*Nc+c)*dE*dE], &vals[(v*Nc+c)*dE*dE]);
      }
    }
  }
  /* Assume its a vector, otherwise assume its a bunch of scalars */
  if (Nc == 1 || Nc != dim) PetscFunctionReturn(0);
  switch (trans) {
    case IDENTITY_TRANSFORM: break;
    case COVARIANT_PIOLA_TRANSFORM: /* Covariant Piola mapping $\sigma^*(F) = J^{-T} F \circ \phi^{-1)$ */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Piola mapping for Hessians not yet supported");
    case CONTRAVARIANT_PIOLA_TRANSFORM: /* Contravariant Piola mapping $\sigma^*(F) = \frac{1}{|\det J|} J F \circ \phi^{-1}$ */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Piola mapping for Hessians not yet supported");
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpacePullback - Transform the given functional so that it operates on real space, rather than the reference element. Operationally, this means that we map the function evaluations depending on continuity requirements of our finite element method.

  Input Parameters:
+ dsp        - The PetscDualSpace
. fegeom     - The geometry for this cell
. Nq         - The number of function samples
. Nc         - The number of function components
- pointEval  - The function values

  Output Parameter:
. pointEval  - The transformed function values

  Level: advanced

  Note: Functions transform in a complementary way (pushforward) to functionals, so that the scalar product is invariant. The type of transform is dependent on the associated k-simplex from the DeRahm complex.

  Note: This only handles tranformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpacePushforward(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePullback(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscInt                    k;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  ierr = PetscDualSpaceGetDeRahm(dsp, &k);CHKERRQ(ierr);
  switch (k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 2:
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", k);
  }
  ierr = PetscDualSpaceTransform(dsp, trans, PETSC_TRUE, fegeom, Nq, Nc, pointEval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpacePushforward - Transform the given function so that it operates on real space, rather than the reference element. Operationally, this means that we map the function evaluations depending on continuity requirements of our finite element method.

  Input Parameters:
+ dsp        - The PetscDualSpace
. fegeom     - The geometry for this cell
. Nq         - The number of function samples
. Nc         - The number of function components
- pointEval  - The function values

  Output Parameter:
. pointEval  - The transformed function values

  Level: advanced

  Note: Functionals transform in a complementary way (pullback) to functions, so that the scalar product is invariant. The type of transform is dependent on the associated k-simplex from the DeRahm complex.

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpacePullback(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePushforward(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscInt                    k;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  ierr = PetscDualSpaceGetDeRahm(dsp, &k);CHKERRQ(ierr);
  switch (k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 2:
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", k);
  }
  ierr = PetscDualSpaceTransform(dsp, trans, PETSC_FALSE, fegeom, Nq, Nc, pointEval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpacePushforwardGradient - Transform the given function gradient so that it operates on real space, rather than the reference element. Operationally, this means that we map the function evaluations depending on continuity requirements of our finite element method.

  Input Parameters:
+ dsp        - The PetscDualSpace
. fegeom     - The geometry for this cell
. Nq         - The number of function gradient samples
. Nc         - The number of function components
- pointEval  - The function gradient values

  Output Parameter:
. pointEval  - The transformed function gradient values

  Level: advanced

  Note: Functionals transform in a complementary way (pullback) to functions, so that the scalar product is invariant. The type of transform is dependent on the associated k-simplex from the DeRahm complex.

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpacePushforward(), PPetscDualSpacePullback(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePushforwardGradient(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscInt                    k;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  ierr = PetscDualSpaceGetDeRahm(dsp, &k);CHKERRQ(ierr);
  switch (k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 2:
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", k);
  }
  ierr = PetscDualSpaceTransformGradient(dsp, trans, PETSC_FALSE, fegeom, Nq, Nc, pointEval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpacePushforwardHessian - Transform the given function Hessian so that it operates on real space, rather than the reference element. Operationally, this means that we map the function evaluations depending on continuity requirements of our finite element method.

  Input Parameters:
+ dsp        - The PetscDualSpace
. fegeom     - The geometry for this cell
. Nq         - The number of function Hessian samples
. Nc         - The number of function components
- pointEval  - The function gradient values

  Output Parameter:
. pointEval  - The transformed function Hessian values

  Level: advanced

  Note: Functionals transform in a complementary way (pullback) to functions, so that the scalar product is invariant. The type of transform is dependent on the associated k-simplex from the DeRahm complex.

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpacePushforward(), PPetscDualSpacePullback(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePushforwardHessian(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscInt                    k;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  ierr = PetscDualSpaceGetDeRahm(dsp, &k);CHKERRQ(ierr);
  switch (k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 2:
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", k);
  }
  ierr = PetscDualSpaceTransformHessian(dsp, trans, PETSC_FALSE, fegeom, Nq, Nc, pointEval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
