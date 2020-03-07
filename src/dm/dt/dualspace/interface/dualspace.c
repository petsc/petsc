#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

PetscClassId PETSCDUALSPACE_CLASSID = 0;

PetscFunctionList PetscDualSpaceList              = NULL;
PetscBool         PetscDualSpaceRegisterAllCalled = PETSC_FALSE;

const char *const PetscDualSpaceReferenceCells[] = {"SIMPLEX", "TENSOR", "PetscDualSpaceReferenceCell", "PETSCDUALSPACE_REFCELL_",0};

/*
  PetscDualSpaceLatticePointLexicographic_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to at most 'max'.
                                                     Ordering is lexicographic with lowest index as least significant in ordering.
                                                     e.g. for len == 2 and max == 2, this will return, in order, {0,0}, {1,0}, {2,0}, {0,1}, {1,1}, {2,0}.

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
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDualSpace type: %s", name);

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
  ierr = PetscViewerASCIIPrintf(v, "Dual space with %D components, size %D\n", sp->Nc, pdim);CHKERRQ(ierr);
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

  Input Parameter:
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
. -petscspace_degree the approximation order of the space

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
  ierr = PetscOptionsBoundedInt("-petscdualspace_degree", "The approximation order", "PetscDualSpaceSetOrder", sp->order, &sp->order, NULL,0);CHKERRQ(ierr);
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
  sp->setupcalled = PETSC_TRUE;
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  if (sp->setfromoptionscalled) {ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCDUALSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;

  ierr = PetscDualSpaceGetDimension(*sp, &dim);CHKERRQ(ierr);
  for (f = 0; f < dim; ++f) {
    ierr = PetscQuadratureDestroy(&(*sp)->functional[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*sp)->functional);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*sp)->allPoints);CHKERRQ(ierr);
  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);

  if ((*sp)->ops->destroy) {ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);}
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

  s->order = 0;
  s->Nc    = 1;
  s->k     = 0;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(spNew, 2);
  ierr = (*sp->ops->duplicate)(sp, spNew);CHKERRQ(ierr);
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
  ierr = DMDestroy(&sp->dm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
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
  if ((i < 0) || (i >= dim)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Functional index %d must be in [0, %d)", i, dim);
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
  *dim = 0;
  if (sp->ops->getdimension) {ierr = (*sp->ops->getdimension)(sp, dim);CHKERRQ(ierr);}
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
  ierr = (*sp->ops->getnumdof)(sp, numDof);CHKERRQ(ierr);
  if (!*numDof) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_LIB, "Empty numDof[] returned from dual space implementation");
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreateSection - Create a PetscSection over the reference cell with the layout from this space

  Collective on sp

  Input Parameters:
+ sp      - The PetscDualSpace

  Output Parameter:
. section - The section

  Level: advanced

.seealso: PetscDualSpaceCreate(), DMPLEX
@*/
PetscErrorCode PetscDualSpaceCreateSection(PetscDualSpace sp, PetscSection *section)
{
  DM             dm;
  PetscInt       pStart, pEnd, depth, h, offset;
  const PetscInt *numDof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)sp),section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section,pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumDof(sp,&numDof);CHKERRQ(ierr);
  for (h = 0; h <= depth; h++) {
    PetscInt hStart, hEnd, p, dof;

    ierr = DMPlexGetHeightStratum(dm,h,&hStart,&hEnd);CHKERRQ(ierr);
    dof = numDof[depth - h];
    for (p = hStart; p < hEnd; p++) {
      ierr = PetscSectionSetDof(*section,p,dof);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  for (h = 0, offset = 0; h <= depth; h++) {
    PetscInt hStart, hEnd, p, dof;

    ierr = DMPlexGetHeightStratum(dm,h,&hStart,&hEnd);CHKERRQ(ierr);
    dof = numDof[depth - h];
    for (p = hStart; p < hEnd; p++) {
      ierr = PetscSectionGetDof(*section,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionSetOffset(*section,p,offset);CHKERRQ(ierr);
      offset += dof;
    }
  }
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

  Level: intermediate

.seealso: PetscDualSpaceCreate(), DMPLEX
@*/
PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace sp, PetscInt dim, PetscBool simplex, DM *refdm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateReferenceCell(PetscObjectComm((PetscObject) sp), dim, simplex, refdm);CHKERRQ(ierr);
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
  PetscDualSpaceApplyAll - Apply all functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetAllPoints()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetAllPoints()

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
  PetscValidPointer(value, 5);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(n, &dim, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
  if (dim != cgeom->dim) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature spatial dimension %D != cell geometry dimension %D", dim, cgeom->dim);
  if (qNc != Nc) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
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
  PetscDualSpaceApplyAllDefault - Apply all functionals from the dual space basis to the result of an evaluation at the points returned by PetscDualSpaceGetAllPoints()

  Input Parameters:
+ sp        - The PetscDualSpace object
- pointEval - Evaluation at the points returned by PetscDualSpaceGetAllPoints()

  Output Parameter:
. spValue   - The values of all dual space functionals

  Level: beginner

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyAllDefault(PetscDualSpace sp, const PetscScalar *pointEval, PetscScalar *spValue)
{
  PetscQuadrature  n;
  const PetscReal *points, *weights;
  PetscInt         qNc, c, Nq, q, f, spdim, Nc;
  PetscInt         offset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidScalarPointer(pointEval, 2);
  PetscValidScalarPointer(spValue, 5);
  ierr = PetscDualSpaceGetDimension(sp, &spdim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < spdim; f++) {
    ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(n, NULL, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
    if (qNc != Nc) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
    spValue[f] = 0.0;
    for (q = 0; q < Nq; ++q) {
      for (c = 0; c < Nc; ++c) {
        spValue[f] += pointEval[offset++]*weights[q*Nc+c];
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetAllPoints - Get all quadrature points from this space

  Input Parameter:
. sp - The dualspace

  Output Parameter:
. allPoints - A PetscQuadrature object containing all evaluation points

  Level: advanced

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetAllPoints(PetscDualSpace sp, PetscQuadrature *allPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(allPoints,2);
  if (!sp->allPoints && sp->ops->createallpoints) {
    ierr = (*sp->ops->createallpoints)(sp,&sp->allPoints);CHKERRQ(ierr);
  }
  *allPoints = sp->allPoints;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreateAllPointsDefault - Create all evaluation points by examining functionals

  Input Parameter:
. sp - The dualspace

  Output Parameter:
. allPoints - A PetscQuadrature object containing all evaluation points

  Level: advanced

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceCreateAllPointsDefault(PetscDualSpace sp, PetscQuadrature *allPoints)
{
  PetscInt        spdim;
  PetscInt        numPoints, offset;
  PetscReal       *points;
  PetscInt        f, dim;
  PetscQuadrature q;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(sp,&spdim);CHKERRQ(ierr);
  if (!spdim) {
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,allPoints);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(*allPoints,0,0,0,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscDualSpaceGetFunctional(sp,0,&q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q,&dim,NULL,&numPoints,NULL,NULL);CHKERRQ(ierr);
  for (f = 1; f < spdim; f++) {
    PetscInt Np;

    ierr = PetscDualSpaceGetFunctional(sp,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&Np,NULL,NULL);CHKERRQ(ierr);
    numPoints += Np;
  }
  ierr = PetscMalloc1(dim*numPoints,&points);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < spdim; f++) {
    const PetscReal *p;
    PetscInt        Np, i;

    ierr = PetscDualSpaceGetFunctional(sp,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&Np,&p,NULL);CHKERRQ(ierr);
    for (i = 0; i < Np * dim; i++) {
      points[offset + i] = p[i];
    }
    offset += Np * dim;
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,allPoints);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*allPoints,dim,0,numPoints,points,NULL);CHKERRQ(ierr);
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
  PetscValidPointer(value, 5);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(n, NULL, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
  if (qNc != Nc) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  *subsp = NULL;
  if (sp->ops->getheightsubspace) {ierr = (*sp->ops->getheightsubspace)(sp, height, subsp);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp,2);
  *bdsp = NULL;
  if (sp->ops->getpointsubspace) {
    ierr = (*sp->ops->getpointsubspace)(sp,point,bdsp);CHKERRQ(ierr);
  } else if (sp->ops->getheightsubspace) {
    DM       dm;
    DMLabel  label;
    PetscInt dim, depth, height;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm,&dim);CHKERRQ(ierr);
    ierr = DMPlexGetDepthLabel(dm,&label);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,point,&depth);CHKERRQ(ierr);
    height = dim - depth;
    ierr = (*sp->ops->getheightsubspace)(sp,height,bdsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDualSpaceGetSymmetries - Returns a description of the symmetries of this basis

  Not collective

  Input Parameter:
. sp - the PetscDualSpace object

  Output Parameters:
+ perms - Permutations of the local degrees of freedom, parameterized by the point orientation
- flips - Sign reversal of the local degrees of freedom, parameterized by the point orientation

  Note: The permutation and flip arrays are organized in the following way
$ perms[p][ornt][dof # on point] = new local dof #
$ flips[p][ornt][dof # on point] = reversal or not

  Level: developer

.seealso: PetscDualSpaceSetSymmetries()
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
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(k, 2);
  *k = dsp->k;
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

.seealso: PetscDualSpaceTransformGradient(), PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceTransform(PetscDualSpace dsp, PetscDualSpaceTransformType trans, PetscBool isInverse, PetscFEGeom *fegeom, PetscInt Nv, PetscInt Nc, PetscScalar vals[])
{
  PetscInt dim, v, c;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 4);
  PetscValidPointer(vals, 7);
  dim = dsp->dm->dim;
  /* Assume its a vector, otherwise assume its a bunch of scalars */
  if (Nc == 1 || Nc != dim) PetscFunctionReturn(0);
  switch (trans) {
    case IDENTITY_TRANSFORM: break;
    case COVARIANT_PIOLA_TRANSFORM: /* Covariant Piola mapping $\sigma^*(F) = J^{-T} F \circ \phi^{-1)$ */
    if (isInverse) {
      for (v = 0; v < Nv; ++v) {
        switch (dim)
        {
          case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->J, 1, &vals[v*Nc], &vals[v*Nc]);break;
          case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->J, 1, &vals[v*Nc], &vals[v*Nc]);break;
          default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
      }
    } else {
      for (v = 0; v < Nv; ++v) {
        switch (dim)
        {
          case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->invJ, 1, &vals[v*Nc], &vals[v*Nc]);break;
          case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->invJ, 1, &vals[v*Nc], &vals[v*Nc]);break;
          default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
      }
    }
    break;
    case CONTRAVARIANT_PIOLA_TRANSFORM: /* Contravariant Piola mapping $\sigma^*(F) = \frac{1}{|\det J|} J F \circ \phi^{-1}$ */
    if (isInverse) {
      for (v = 0; v < Nv; ++v) {
        switch (dim)
        {
          case 2: DMPlex_Mult2DReal_Internal(fegeom->invJ, 1, &vals[v*Nc], &vals[v*Nc]);break;
          case 3: DMPlex_Mult3DReal_Internal(fegeom->invJ, 1, &vals[v*Nc], &vals[v*Nc]);break;
          default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
        for (c = 0; c < Nc; ++c) vals[v*Nc+c] *= fegeom->detJ[0];
      }
    } else {
      for (v = 0; v < Nv; ++v) {
        switch (dim)
        {
          case 2: DMPlex_Mult2DReal_Internal(fegeom->J, 1, &vals[v*Nc], &vals[v*Nc]);break;
          case 3: DMPlex_Mult3DReal_Internal(fegeom->J, 1, &vals[v*Nc], &vals[v*Nc]);break;
          default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
        }
        for (c = 0; c < Nc; ++c) vals[v*Nc+c] /= fegeom->detJ[0];
      }
    }
    break;
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
. vals      - The transformed function values

  Level: intermediate

.seealso: PetscDualSpaceTransform(), PetscDualSpacePullback(), PetscDualSpacePushforward(), PetscDualSpaceTransformType
@*/
PetscErrorCode PetscDualSpaceTransformGradient(PetscDualSpace dsp, PetscDualSpaceTransformType trans, PetscBool isInverse, PetscFEGeom *fegeom, PetscInt Nv, PetscInt Nc, PetscScalar vals[])
{
  PetscInt dim, v, c, d;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 4);
  PetscValidPointer(vals, 7);
  dim = dsp->dm->dim;
  /* Transform gradient */
  for (v = 0; v < Nv; ++v) {
    for (c = 0; c < Nc; ++c) {
      switch (dim)
      {
        case 1: vals[(v*Nc+c)*dim] *= fegeom->invJ[0];break;
        case 2: DMPlex_MultTranspose2DReal_Internal(fegeom->invJ, 1, &vals[(v*Nc+c)*dim], &vals[(v*Nc+c)*dim]);break;
        case 3: DMPlex_MultTranspose3DReal_Internal(fegeom->invJ, 1, &vals[(v*Nc+c)*dim], &vals[(v*Nc+c)*dim]);break;
        default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
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
            default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
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
            default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
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
            default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
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
            default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dim %D for transformation", dim);
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

.seealso: PetscDualSpacePushforward(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePullback(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  switch (dsp->k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    case 2:
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", dsp->k);
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

.seealso: PetscDualSpacePullback(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePushforward(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  switch (dsp->k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    case 2:
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", dsp->k);
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

.seealso: PetscDualSpacePushforward(), PPetscDualSpacePullback(), PetscDualSpaceTransform(), PetscDualSpaceGetDeRahm()
@*/
PetscErrorCode PetscDualSpacePushforwardGradient(PetscDualSpace dsp, PetscFEGeom *fegeom, PetscInt Nq, PetscInt Nc, PetscScalar pointEval[])
{
  PetscDualSpaceTransformType trans;
  PetscErrorCode              ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dsp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(fegeom, 2);
  PetscValidPointer(pointEval, 5);
  /* The dualspace dofs correspond to some simplex in the DeRahm complex, which we label by k.
     This determines their transformation properties. */
  switch (dsp->k)
  {
    case 0: /* H^1 point evaluations */
    trans = IDENTITY_TRANSFORM;break;
    case 1: /* Hcurl preserves tangential edge traces  */
    case 2:
    trans = COVARIANT_PIOLA_TRANSFORM;break;
    case 3: /* Hdiv preserve normal traces */
    trans = CONTRAVARIANT_PIOLA_TRANSFORM;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dsp), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported simplex dim %D for transformation", dsp->k);
  }
  ierr = PetscDualSpaceTransformGradient(dsp, trans, PETSC_FALSE, fegeom, Nq, Nc, pointEval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
