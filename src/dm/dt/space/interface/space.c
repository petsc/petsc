#include <petsc/private/petscfeimpl.h>     /*I  "petscfe.h"  I*/
#include <petscdmshell.h>

PetscClassId PETSCSPACE_CLASSID = 0;

PetscFunctionList PetscSpaceList              = NULL;
PetscBool         PetscSpaceRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscSpaceRegister - Adds a new PetscSpace implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine for the implementation type

  Notes:
  PetscSpaceRegister() may be called multiple times to add several user-defined types of PetscSpaces.  The creation function is called
  when the type is set to 'name'.

  Sample usage:
.vb
    PetscSpaceRegister("my_space", MyPetscSpaceCreate);
.ve

  Then, your PetscSpace type can be chosen with the procedural interface via
.vb
    PetscSpaceCreate(MPI_Comm, PetscSpace *);
    PetscSpaceSetType(PetscSpace, "my_space");
.ve
   or at runtime via the option
.vb
    -petscspace_type my_space
.ve

  Level: advanced

.seealso: PetscSpaceRegisterAll(), PetscSpaceRegisterDestroy()

@*/
PetscErrorCode PetscSpaceRegister(const char sname[], PetscErrorCode (*function)(PetscSpace))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscSpaceList, sname, function));
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceSetType - Builds a particular PetscSpace

  Collective on sp

  Input Parameters:
+ sp   - The PetscSpace object
- name - The kind of space

  Options Database Key:
. -petscspace_type <type> - Sets the PetscSpace type; use -help for a list of available types

  Level: intermediate

.seealso: PetscSpaceGetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceSetType(PetscSpace sp, PetscSpaceType name)
{
  PetscErrorCode (*r)(PetscSpace);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject) sp, name, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscSpaceRegisterAll());
  PetscCall(PetscFunctionListFind(PetscSpaceList, name, &r));
  PetscCheck(r,PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscSpace type: %s", name);

  if (sp->ops->destroy) {
    PetscCall((*sp->ops->destroy)(sp));
    sp->ops->destroy = NULL;
  }
  sp->dim = PETSC_DETERMINE;
  PetscCall((*r)(sp));
  PetscCall(PetscObjectChangeTypeName((PetscObject) sp, name));
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceGetType - Gets the PetscSpace type name (as a string) from the object.

  Not Collective

  Input Parameter:
. sp  - The PetscSpace

  Output Parameter:
. name - The PetscSpace type name

  Level: intermediate

.seealso: PetscSpaceSetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceGetType(PetscSpace sp, PetscSpaceType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(name, 2);
  if (!PetscSpaceRegisterAllCalled) {
    PetscCall(PetscSpaceRegisterAll());
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PetscSpaceViewFromOptions - View from Options

   Collective on PetscSpace

   Input Parameters:
+  A - the PetscSpace object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscSpace, PetscSpaceView, PetscObjectViewFromOptions(), PetscSpaceCreate()
@*/
PetscErrorCode  PetscSpaceViewFromOptions(PetscSpace A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCSPACE_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceView - Views a PetscSpace

  Collective on sp

  Input Parameters:
+ sp - the PetscSpace object to view
- v  - the viewer

  Level: beginner

.seealso PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceView(PetscSpace sp, PetscViewer v)
{
  PetscInt       pdim;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (v) PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v));
  PetscCall(PetscSpaceGetDimension(sp, &pdim));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sp,v));
  PetscCall(PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscViewerASCIIPushTab(v));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(v, "Space in %D variables with %D components, size %D\n", sp->Nv, sp->Nc, pdim));
  if (sp->ops->view) PetscCall((*sp->ops->view)(sp, v));
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetFromOptions - sets parameters in a PetscSpace from the options database

  Collective on sp

  Input Parameter:
. sp - the PetscSpace object to set options for

  Options Database:
+ -petscspace_degree <deg> - the approximation order of the space
. -petscspace_variables <n> - the number of different variables, e.g. x and y
- -petscspace_components <c> - the number of components, say d for a vector field

  Level: intermediate

.seealso PetscSpaceView()
@*/
PetscErrorCode PetscSpaceSetFromOptions(PetscSpace sp)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (!((PetscObject) sp)->type_name) {
    defaultType = PETSCSPACEPOLYNOMIAL;
  } else {
    defaultType = ((PetscObject) sp)->type_name;
  }
  if (!PetscSpaceRegisterAllCalled) PetscCall(PetscSpaceRegisterAll());

  ierr = PetscObjectOptionsBegin((PetscObject) sp);PetscCall(ierr);
  PetscCall(PetscOptionsFList("-petscspace_type", "Linear space", "PetscSpaceSetType", PetscSpaceList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(PetscSpaceSetType(sp, name));
  } else if (!((PetscObject) sp)->type_name) {
    PetscCall(PetscSpaceSetType(sp, defaultType));
  }
  {
    PetscCall(PetscOptionsDeprecated("-petscspace_order","-petscspace_degree","3.11",NULL));
    PetscCall(PetscOptionsBoundedInt("-petscspace_order", "DEPRECATED: The approximation order", "PetscSpaceSetDegree", sp->degree, &sp->degree, NULL,0));
  }
  PetscCall(PetscOptionsBoundedInt("-petscspace_degree", "The (maximally included) polynomial degree", "PetscSpaceSetDegree", sp->degree, &sp->degree, NULL,0));
  PetscCall(PetscOptionsBoundedInt("-petscspace_variables", "The number of different variables, e.g. x and y", "PetscSpaceSetNumVariables", sp->Nv, &sp->Nv, NULL,0));
  PetscCall(PetscOptionsBoundedInt("-petscspace_components", "The number of components", "PetscSpaceSetNumComponents", sp->Nc, &sp->Nc, NULL,0));
  if (sp->ops->setfromoptions) {
    PetscCall((*sp->ops->setfromoptions)(PetscOptionsObject,sp));
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscCall(PetscSpaceViewFromOptions(sp, NULL, "-petscspace_view"));
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceSetUp - Construct data structures for the PetscSpace

  Collective on sp

  Input Parameter:
. sp - the PetscSpace object to setup

  Level: intermediate

.seealso PetscSpaceView(), PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceSetUp(PetscSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (sp->ops->setup) PetscCall((*sp->ops->setup)(sp));
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceDestroy - Destroys a PetscSpace object

  Collective on sp

  Input Parameter:
. sp - the PetscSpace object to destroy

  Level: beginner

.seealso PetscSpaceView()
@*/
PetscErrorCode PetscSpaceDestroy(PetscSpace *sp)
{
  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;
  PetscCall(DMDestroy(&(*sp)->dm));

  PetscCall((*(*sp)->ops->destroy)(*sp));
  PetscCall(PetscHeaderDestroy(sp));
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceCreate - Creates an empty PetscSpace object. The type can then be set with PetscSpaceSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscSpace object

  Output Parameter:
. sp - The PetscSpace object

  Level: beginner

.seealso: PetscSpaceSetType(), PETSCSPACEPOLYNOMIAL
@*/
PetscErrorCode PetscSpaceCreate(MPI_Comm comm, PetscSpace *sp)
{
  PetscSpace     s;

  PetscFunctionBegin;
  PetscValidPointer(sp, 2);
  PetscCall(PetscCitationsRegister(FECitation,&FEcite));
  *sp  = NULL;
  PetscCall(PetscFEInitializePackage());

  PetscCall(PetscHeaderCreate(s, PETSCSPACE_CLASSID, "PetscSpace", "Linear Space", "PetscSpace", comm, PetscSpaceDestroy, PetscSpaceView));

  s->degree    = 0;
  s->maxDegree = PETSC_DETERMINE;
  s->Nc        = 1;
  s->Nv        = 0;
  s->dim       = PETSC_DETERMINE;
  PetscCall(DMShellCreate(comm, &s->dm));
  PetscCall(PetscSpaceSetType(s, PETSCSPACEPOLYNOMIAL));

  *sp = s;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetDimension - Return the dimension of this space, i.e. the number of basis vectors

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. dim - The dimension

  Level: intermediate

.seealso: PetscSpaceGetDegree(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetDimension(PetscSpace sp, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  if (sp->dim == PETSC_DETERMINE) {
    if (sp->ops->getdimension) PetscCall((*sp->ops->getdimension)(sp, &sp->dim));
  }
  *dim = sp->dim;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetDegree - Return the polynomial degrees that characterize this space

  Input Parameter:
. sp - The PetscSpace

  Output Parameters:
+ minDegree - The degree of the largest polynomial space contained in the space
- maxDegree - The degree of the smallest polynomial space containing the space

  Level: intermediate

.seealso: PetscSpaceSetDegree(), PetscSpaceGetDimension(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetDegree(PetscSpace sp, PetscInt *minDegree, PetscInt *maxDegree)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (minDegree) PetscValidIntPointer(minDegree, 2);
  if (maxDegree) PetscValidIntPointer(maxDegree, 3);
  if (minDegree) *minDegree = sp->degree;
  if (maxDegree) *maxDegree = sp->maxDegree;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetDegree - Set the degree of approximation for this space.

  Input Parameters:
+ sp - The PetscSpace
. degree - The degree of the largest polynomial space contained in the space
- maxDegree - The degree of the largest polynomial space containing the space.  One of degree and maxDegree can be PETSC_DETERMINE.

  Level: intermediate

.seealso: PetscSpaceGetDegree(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceSetDegree(PetscSpace sp, PetscInt degree, PetscInt maxDegree)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->degree = degree;
  sp->maxDegree = maxDegree;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetNumComponents - Return the number of components for this space

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. Nc - The number of components

  Note: A vector space, for example, will have d components, where d is the spatial dimension

  Level: intermediate

.seealso: PetscSpaceSetNumComponents(), PetscSpaceGetNumVariables(), PetscSpaceGetDimension(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetNumComponents(PetscSpace sp, PetscInt *Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(Nc, 2);
  *Nc = sp->Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetNumComponents - Set the number of components for this space

  Input Parameters:
+ sp - The PetscSpace
- order - The number of components

  Level: intermediate

.seealso: PetscSpaceGetNumComponents(), PetscSpaceSetNumVariables(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceSetNumComponents(PetscSpace sp, PetscInt Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->Nc = Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetNumVariables - Set the number of variables for this space

  Input Parameters:
+ sp - The PetscSpace
- n - The number of variables, e.g. x, y, z...

  Level: intermediate

.seealso: PetscSpaceGetNumVariables(), PetscSpaceSetNumComponents(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceSetNumVariables(PetscSpace sp, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->Nv = n;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetNumVariables - Return the number of variables for this space

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. Nc - The number of variables, e.g. x, y, z...

  Level: intermediate

.seealso: PetscSpaceSetNumVariables(), PetscSpaceGetNumComponents(), PetscSpaceGetDimension(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetNumVariables(PetscSpace sp, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(n, 2);
  *n = sp->Nv;
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceEvaluate - Evaluate the basis functions and their derivatives (jet) at each point

  Input Parameters:
+ sp      - The PetscSpace
. npoints - The number of evaluation points, in reference coordinates
- points  - The point coordinates

  Output Parameters:
+ B - The function evaluations in a npoints x nfuncs array
. D - The derivative evaluations in a npoints x nfuncs x dim array
- H - The second derivative evaluations in a npoints x nfuncs x dim x dim array

  Note: Above nfuncs is the dimension of the space, and dim is the spatial dimension. The coordinates are given
  on the reference cell, not in real space.

  Level: beginner

.seealso: PetscFECreateTabulation(), PetscFEGetCellTabulation(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceEvaluate(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscFunctionBegin;
  if (!npoints) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (sp->Nv) PetscValidRealPointer(points, 3);
  if (B) PetscValidRealPointer(B, 4);
  if (D) PetscValidRealPointer(D, 5);
  if (H) PetscValidRealPointer(H, 6);
  if (sp->ops->evaluate) PetscCall((*sp->ops->evaluate)(sp, npoints, points, B, D, H));
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetHeightSubspace - Get the subset of the primal space basis that is supported on a mesh point of a given height.

  If the space is not defined on mesh points of the given height (e.g. if the space is discontinuous and
  pointwise values are not defined on the element boundaries), or if the implementation of PetscSpace does not
  support extracting subspaces, then NULL is returned.

  This does not increment the reference count on the returned space, and the user should not destroy it.

  Not collective

  Input Parameters:
+ sp - the PetscSpace object
- height - the height of the mesh point for which the subspace is desired

  Output Parameter:
. subsp - the subspace

  Level: advanced

.seealso: PetscDualSpaceGetHeightSubspace(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetHeightSubspace(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  *subsp = NULL;
  if (sp->ops->getheightsubspace) {
    PetscCall((*sp->ops->getheightsubspace)(sp, height, subsp));
  }
  PetscFunctionReturn(0);
}
