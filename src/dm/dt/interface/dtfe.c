/* Basis Jet Tabulation

We would like to tabulate the nodal basis functions and derivatives at a set of points, usually quadrature points. We
follow here the derviation in http://www.math.ttu.edu/~kirby/papers/fiat-toms-2004.pdf. The nodal basis $\psi_i$ can
be expressed in terms of a prime basis $\phi_i$ which can be stably evaluated. In PETSc, we will use the Legendre basis
as a prime basis.

  \psi_i = \sum_k \alpha_{ki} \phi_k

Our nodal basis is defined in terms of the dual basis $n_j$

  n_j \cdot \psi_i = \delta_{ji}

and we may act on the first equation to obtain

  n_j \cdot \psi_i = \sum_k \alpha_{ki} n_j \cdot \phi_k
       \delta_{ji} = \sum_k \alpha_{ki} V_{jk}
                 I = V \alpha

so the coefficients of the nodal basis in the prime basis are

   \alpha = V^{-1}

We will define the dual basis vectors $n_j$ using a quadrature rule.

Right now, we will just use the polynomial spaces P^k. I know some elements use the space of symmetric polynomials
(I think Nedelec), but we will neglect this for now. Constraints in the space, e.g. Arnold-Winther elements, can
be implemented exactly as in FIAT using functionals $L_j$.

I will have to count the degrees correctly for the Legendre product when we are on simplices.

We will have three objects:
 - Space, P: this just need point evaluation I think
 - Dual Space, P'+K: This looks like a set of functionals that can act on members of P, each n is defined by a Q
 - FEM: This keeps {P, P', Q}
*/
#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h>
#include <petsc/private/dmpleximpl.h> /* For CellRefiner */
#include <petscdmshell.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>

PetscBool FEcite = PETSC_FALSE;
const char FECitation[] = "@article{kirby2004,\n"
                          "  title   = {Algorithm 839: FIAT, a New Paradigm for Computing Finite Element Basis Functions},\n"
                          "  journal = {ACM Transactions on Mathematical Software},\n"
                          "  author  = {Robert C. Kirby},\n"
                          "  volume  = {30},\n"
                          "  number  = {4},\n"
                          "  pages   = {502--516},\n"
                          "  doi     = {10.1145/1039813.1039820},\n"
                          "  year    = {2004}\n}\n";

PetscClassId PETSCSPACE_CLASSID = 0;

PetscFunctionList PetscSpaceList              = NULL;
PetscBool         PetscSpaceRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscSpaceRegister - Adds a new PetscSpace implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscSpaceRegister() may be called multiple times to add several user-defined PetscSpaces

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

.keywords: PetscSpace, register
.seealso: PetscSpaceRegisterAll(), PetscSpaceRegisterDestroy()

@*/
PetscErrorCode PetscSpaceRegister(const char sname[], PetscErrorCode (*function)(PetscSpace))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscSpaceList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceSetType - Builds a particular PetscSpace

  Collective on PetscSpace

  Input Parameters:
+ sp   - The PetscSpace object
- name - The kind of space

  Options Database Key:
. -petscspace_type <type> - Sets the PetscSpace type; use -help for a list of available types

  Level: intermediate

.keywords: PetscSpace, set, type
.seealso: PetscSpaceGetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceSetType(PetscSpace sp, PetscSpaceType name)
{
  PetscErrorCode (*r)(PetscSpace);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) sp, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscSpaceList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscSpace type: %s", name);

  if (sp->ops->destroy) {
    ierr             = (*sp->ops->destroy)(sp);CHKERRQ(ierr);
    sp->ops->destroy = NULL;
  }
  ierr = (*r)(sp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) sp, name);CHKERRQ(ierr);
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

.keywords: PetscSpace, get, type, name
.seealso: PetscSpaceSetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceGetType(PetscSpace sp, PetscSpaceType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(name, 2);
  if (!PetscSpaceRegisterAllCalled) {
    ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceView - Views a PetscSpace

  Collective on PetscSpace

  Input Parameter:
+ sp - the PetscSpace object to view
- v  - the viewer

  Level: developer

.seealso PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceView(PetscSpace sp, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);}
  if (sp->ops->view) {ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetFromOptions - sets parameters in a PetscSpace from the options database

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to set options for

  Options Database:
. -petscspace_order the approximation order of the space

  Level: developer

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
  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscspace_type", "Linear space", "PetscSpaceSetType", PetscSpaceList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscSpaceSetType(sp, name);CHKERRQ(ierr);
  } else if (!((PetscObject) sp)->type_name) {
    ierr = PetscSpaceSetType(sp, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-petscspace_order", "The approximation order", "PetscSpaceSetOrder", sp->order, &sp->order, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_components", "The number of components", "PetscSpaceSetNumComponents", sp->Nc, &sp->Nc, NULL);CHKERRQ(ierr);
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(PetscOptionsObject,sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscSpaceViewFromOptions(sp, NULL, "-petscspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSpaceSetUp - Construct data structures for the PetscSpace

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to setup

  Level: developer

.seealso PetscSpaceView(), PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceSetUp(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceDestroy - Destroys a PetscSpace object

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to destroy

  Level: developer

.seealso PetscSpaceView()
@*/
PetscErrorCode PetscSpaceDestroy(PetscSpace *sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;
  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);

  ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceCreate - Creates an empty PetscSpace object. The type can then be set with PetscSpaceSetType().

  Collective on MPI_Comm

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sp, 2);
  ierr = PetscCitationsRegister(FECitation,&FEcite);CHKERRQ(ierr);
  *sp  = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(s, PETSCSPACE_CLASSID, "PetscSpace", "Linear Space", "PetscSpace", comm, PetscSpaceDestroy, PetscSpaceView);CHKERRQ(ierr);

  s->order = 0;
  s->Nc    = 1;
  ierr = DMShellCreate(comm, &s->dm);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(s, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);

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

.seealso: PetscSpaceGetOrder(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetDimension(PetscSpace sp, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = 0;
  if (sp->ops->getdimension) {ierr = (*sp->ops->getdimension)(sp, dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceGetOrder - Return the order of approximation for this space

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. order - The approximation order

  Level: intermediate

.seealso: PetscSpaceSetOrder(), PetscSpaceGetDimension(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetOrder(PetscSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = sp->order;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetOrder - Set the order of approximation for this space

  Input Parameters:
+ sp - The PetscSpace
- order - The approximation order

  Level: intermediate

.seealso: PetscSpaceGetOrder(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceSetOrder(PetscSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->order = order;
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

.seealso: PetscSpaceSetNumComponents(), PetscSpaceGetDimension(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetNumComponents(PetscSpace sp, PetscInt *Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(Nc, 2);
  *Nc = sp->Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSetNumComponents - Set the number of components for this space

  Input Parameters:
+ sp - The PetscSpace
- order - The number of components

  Level: intermediate

.seealso: PetscSpaceGetNumComponents(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceSetNumComponents(PetscSpace sp, PetscInt Nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->Nc = Nc;
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

  Level: advanced

.seealso: PetscFEGetTabulation(), PetscFEGetDefaultTabulation(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceEvaluate(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!npoints) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  if (sp->ops->evaluate) {ierr = (*sp->ops->evaluate)(sp, npoints, points, B, D, H);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  *subsp = NULL;
  if (sp->ops->getheightsubspace) {
    ierr = (*sp->ops->getheightsubspace)(sp, height, subsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_poly_num_variables", "The number of different variables, e.g. x and y", "PetscSpacePolynomialSetNumVariables", poly->numVariables, &poly->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_sym", "Use only symmetric polynomials", "PetscSpacePolynomialSetSymmetric", poly->symmetric, &poly->symmetric, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_tensor", "Use the tensor product polynomials", "PetscSpacePolynomialSetTensor", poly->tensor, &poly->tensor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (sp->Nc > 1) {
    if (poly->tensor) {ierr = PetscViewerASCIIPrintf(viewer, "Tensor polynomial space in %D variables of degree %D with %D components\n", poly->numVariables, sp->order, sp->Nc);CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %D variables of degree %D with %D components\n", poly->numVariables, sp->order, sp->Nc);CHKERRQ(ierr);}
  } else {
    if (poly->tensor) {ierr = PetscViewerASCIIPrintf(viewer, "Tensor polynomial space in %d variables of degree %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of degree %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceView_Polynomial(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePolynomialView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetUp_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  PetscInt         ndegree = sp->order+1;
  PetscInt         deg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ndegree, &poly->degrees);CHKERRQ(ierr);
  for (deg = 0; deg < ndegree; ++deg) poly->degrees[deg] = deg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(poly->degrees);CHKERRQ(ierr);
  if (poly->subspaces) {
    PetscInt d;

    for (d = 0; d < poly->numVariables; ++d) {
      ierr = PetscSpaceDestroy(&poly->subspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(poly->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(poly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* We treat the space as a tensor product of scalar polynomial spaces, so the dimension is multiplied by Nc */
PetscErrorCode PetscSpaceGetDimension_Polynomial(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         deg  = sp->order;
  PetscInt         n    = poly->numVariables, i;
  PetscReal        D    = 1.0;

  PetscFunctionBegin;
  if (poly->tensor) {
    *dim = 1;
    for (i = 0; i < n; ++i) *dim *= (deg+1);
  } else {
    for (i = 1; i <= n; ++i) {
      D *= ((PetscReal) (deg+i))/i;
    }
    *dim = (PetscInt) (D + 0.5);
  }
  *dim *= sp->Nc;
  PetscFunctionReturn(0);
}

/*
  LatticePoint_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to 'sum'.

  Input Parameters:
+ len - The length of the tuple
. sum - The sum of all entries in the tuple
- ind - The current multi-index of the tuple, initialized to the 0 tuple

  Output Parameter:
+ ind - The multi-index of the tuple, -1 indicates the iteration has terminated
. tup - A tuple of len integers addig to sum

  Level: developer

.seealso:
*/
static PetscErrorCode LatticePoint_Internal(PetscInt len, PetscInt sum, PetscInt ind[], PetscInt tup[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (len == 1) {
    ind[0] = -1;
    tup[0] = sum;
  } else if (sum == 0) {
    for (i = 0; i < len; ++i) {ind[0] = -1; tup[i] = 0;}
  } else {
    tup[0] = sum - ind[0];
    ierr = LatticePoint_Internal(len-1, ind[0], &ind[1], &tup[1]);CHKERRQ(ierr);
    if (ind[1] < 0) {
      if (ind[0] == sum) {ind[0] = -1;}
      else               {ind[1] = 0; ++ind[0];}
    }
  }
  PetscFunctionReturn(0);
}

/*
  LatticePointLexicographic_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to at most 'max'.
                                       Ordering is lexicographic with lowest index as least significant in ordering.
                                       e.g. for len == 2 and max == 2, this will return, in order, {0,0}, {1,0}, {2,0}, {0,1}, {1,1}, {2,0}.

  Input Parameters:
+ len - The length of the tuple
. max - The maximum sum
- tup - A tuple of length len+1: tup[len] > 0 indicates a stopping condition

  Output Parameter:
. tup - A tuple of len integers whos sum is at most 'max'
*/
static PetscErrorCode LatticePointLexicographic_Internal(PetscInt len, PetscInt max, PetscInt tup[])
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
  TensorPoint_Internal - Returns all tuples of size 'len' with nonnegative integers that are less than 'max'.

  Input Parameters:
+ len - The length of the tuple
. max - The max for all entries in the tuple
- ind - The current multi-index of the tuple, initialized to the 0 tuple

  Output Parameter:
+ ind - The multi-index of the tuple, -1 indicates the iteration has terminated
. tup - A tuple of len integers less than max

  Level: developer

.seealso:
*/
static PetscErrorCode TensorPoint_Internal(PetscInt len, PetscInt max, PetscInt ind[], PetscInt tup[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (len == 1) {
    tup[0] = ind[0]++;
    ind[0] = ind[0] >= max ? -1 : ind[0];
  } else if (max == 0) {
    for (i = 0; i < len; ++i) {ind[0] = -1; tup[i] = 0;}
  } else {
    tup[0] = ind[0];
    ierr = TensorPoint_Internal(len-1, max, &ind[1], &tup[1]);CHKERRQ(ierr);
    if (ind[1] < 0) {
      ind[1] = 0;
      if (ind[0] == max-1) {ind[0] = -1;}
      else                 {++ind[0];}
    }
  }
  PetscFunctionReturn(0);
}

/*
  TensorPointLexicographic_Internal - Returns all tuples of size 'len' with nonnegative integers that are all less than or equal to 'max'.
                                      Ordering is lexicographic with lowest index as least significant in ordering.
                                      e.g. for len == 2 and max == 2, this will return, in order, {0,0}, {1,0}, {2,0}, {0,1}, {1,1}, {2,1}, {0,2}, {1,2}, {2,2}.

  Input Parameters:
+ len - The length of the tuple
. max - The maximum value
- tup - A tuple of length len+1: tup[len] > 0 indicates a stopping condition

  Output Parameter:
. tup - A tuple of len integers whos sum is at most 'max'
*/
static PetscErrorCode TensorPointLexicographic_Internal(PetscInt len, PetscInt max, PetscInt tup[])
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

/*
  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]
*/
PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         ndegree = sp->order+1;
  PetscInt        *degrees = poly->degrees;
  PetscInt         dim     = poly->numVariables;
  PetscReal       *lpoints, *tmp, *LB, *LD, *LH;
  PetscInt        *ind, *tup;
  PetscInt         c, pdim, d, der, i, p, deg, o;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  pdim /= Nc;
  ierr = DMGetWorkArray(dm, npoints, PETSC_REAL, &lpoints);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, npoints*ndegree*3, PETSC_REAL, &tmp);CHKERRQ(ierr);
  if (B) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LH);CHKERRQ(ierr);}
  for (d = 0; d < dim; ++d) {
    for (p = 0; p < npoints; ++p) {
      lpoints[p] = points[p*dim+d];
    }
    ierr = PetscDTLegendreEval(npoints, lpoints, ndegree, degrees, tmp, &tmp[1*npoints*ndegree], &tmp[2*npoints*ndegree]);CHKERRQ(ierr);
    /* LB, LD, LH (ndegree * dim x npoints) */
    for (deg = 0; deg < ndegree; ++deg) {
      for (p = 0; p < npoints; ++p) {
        if (B) LB[(deg*dim + d)*npoints + p] = tmp[(0*npoints + p)*ndegree+deg];
        if (D) LD[(deg*dim + d)*npoints + p] = tmp[(1*npoints + p)*ndegree+deg];
        if (H) LH[(deg*dim + d)*npoints + p] = tmp[(2*npoints + p)*ndegree+deg];
      }
    }
  }
  /* Multiply by A (pdim x ndegree * dim) */
  ierr = PetscMalloc2(dim,&ind,dim,&tup);CHKERRQ(ierr);
  if (B) {
    /* B (npoints x pdim x Nc) */
    ierr = PetscMemzero(B, npoints*pdim*Nc*Nc * sizeof(PetscReal));CHKERRQ(ierr);
    if (poly->tensor) {
      i = 0;
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->order+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          B[(p*pdim + i)*Nc*Nc] = 1.0;
          for (d = 0; d < dim; ++d) {
            B[(p*pdim + i)*Nc*Nc] *= LB[(tup[d]*dim + d)*npoints + p];
          }
        }
        ++i;
      }
    } else {
      i = 0;
      for (o = 0; o <= sp->order; ++o) {
        ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            B[(p*pdim + i)*Nc*Nc] = 1.0;
            for (d = 0; d < dim; ++d) {
              B[(p*pdim + i)*Nc*Nc] *= LB[(tup[d]*dim + d)*npoints + p];
            }
          }
          ++i;
        }
      }
    }
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          B[(p*pdim*Nc + i*Nc + c)*Nc + c] = B[(p*pdim + i)*Nc*Nc];
        }
      }
    }
  }
  if (D) {
    /* D (npoints x pdim x Nc x dim) */
    ierr = PetscMemzero(D, npoints*pdim*Nc*Nc*dim * sizeof(PetscReal));CHKERRQ(ierr);
    if (poly->tensor) {
      i = 0;
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->order+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          for (der = 0; der < dim; ++der) {
            D[(p*pdim + i)*Nc*Nc*dim + der] = 1.0;
            for (d = 0; d < dim; ++d) {
              if (d == der) {
                D[(p*pdim + i)*Nc*Nc*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
              } else {
                D[(p*pdim + i)*Nc*Nc*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
              }
            }
          }
        }
        ++i;
      }
    } else {
      i = 0;
      for (o = 0; o <= sp->order; ++o) {
        ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
        while (ind[0] >= 0) {
          ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
          for (p = 0; p < npoints; ++p) {
            for (der = 0; der < dim; ++der) {
              D[(p*pdim + i)*Nc*Nc*dim + der] = 1.0;
              for (d = 0; d < dim; ++d) {
                if (d == der) {
                  D[(p*pdim + i)*Nc*Nc*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
                } else {
                  D[(p*pdim + i)*Nc*Nc*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
                }
              }
            }
          }
          ++i;
        }
      }
    }
    /* Make direct sum basis for multicomponent space */
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (c = 1; c < Nc; ++c) {
          for (d = 0; d < dim; ++d) {
            D[((p*pdim*Nc + i*Nc + c)*Nc + c)*dim + d] = D[(p*pdim + i)*Nc*Nc*dim + d];
          }
        }
      }
    }
  }
  if (H) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to code second derivatives");
  ierr = PetscFree2(ind,tup);CHKERRQ(ierr);
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LH);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*ndegree*3, PETSC_REAL, &tmp);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, npoints, PETSC_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialSetTensor - Set whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variabl is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
+ sp     - the function space object
- tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Level: beginner

.seealso: PetscSpacePolynomialGetTensor(), PetscSpaceSetOrder(), PetscSpacePolynomialSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace sp, PetscBool tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialSetTensor_C",(PetscSpace,PetscBool),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialGetTensor - Get whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variabl is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
. sp     - the function space object

  Output Parameters:
. tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Level: beginner

.seealso: PetscSpacePolynomialSetTensor(), PetscSpaceSetOrder(), PetscSpacePolynomialSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace sp, PetscBool *tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialGetTensor_C",(PetscSpace,PetscBool*),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialSetTensor_Polynomial(PetscSpace sp, PetscBool tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  poly->tensor = tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialGetTensor_Polynomial(PetscSpace sp, PetscBool *tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  *tensor = poly->tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Polynomial(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetNumVariables(sp, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(sp, &tensor);CHKERRQ(ierr);
  if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);}
  if (!poly->subspaces) {ierr = PetscCalloc1(dim, &poly->subspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!poly->subspaces[height-1]) {
      PetscSpace sub;

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetOrder(sub, order);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
      ierr = PetscSpacePolynomialSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpacePolynomialSetTensor(sub, tensor);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      poly->subspaces[height-1] = sub;
    }
    *subsp = poly->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup             = PetscSpaceSetUp_Polynomial;
  sp->ops->view              = PetscSpaceView_Polynomial;
  sp->ops->destroy           = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension      = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate          = PetscSpaceEvaluate_Polynomial;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Polynomial;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Polynomial);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", PetscSpacePolynomialSetTensor_Polynomial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOLYNOMIAL = "poly" - A PetscSpace object that encapsulates a polynomial space, e.g. P1 is the space of
  linear polynomials. The space is replicated for each component.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&poly);CHKERRQ(ierr);
  sp->data = poly;

  poly->numVariables = 0;
  poly->symmetric    = PETSC_FALSE;
  poly->tensor       = PETSC_FALSE;
  poly->degrees      = NULL;
  poly->subspaces    = NULL;

  ierr = PetscSpaceInitialize_Polynomial(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace sp, PetscBool sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->symmetric = sym;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace sp, PetscBool *sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(sym, 2);
  *sym = poly->symmetric;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpacePolynomialSetNumVariables(PetscSpace sp, PetscInt n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->numVariables = n;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpacePolynomialGetNumVariables(PetscSpace sp, PetscInt *n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(n, 2);
  *n = poly->numVariables;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetFromOptions_Point(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace Point options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_point_num_variables", "The number of different variables, e.g. x and y", "PetscSpacePointSetNumVariables", pt->numVariables, &pt->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpacePointView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "Point space in dimension %d:\n", pt->numVariables);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureView(pt->quad, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Point space in dimension %d on %d points\n", pt->numVariables, pt->quad->numPoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceView_Point(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePointView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSetUp_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!pt->quad->points && sp->order >= 0) {
    ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature(pt->numVariables, sp->Nc, PetscMax(sp->order + 1, 1), -1.0, 1.0, &pt->quad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceDestroy_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
  ierr = PetscFree(pt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceGetDimension_Point(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  *dim = pt->quad->numPoints;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceEvaluate_Point(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Point *pt  = (PetscSpace_Point *) sp->data;
  PetscInt          dim = pt->numVariables, pdim = pt->quad->numPoints, d, p, i, c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (npoints != pt->quad->numPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot evaluate Point space on %d points != %d size", npoints, pt->quad->numPoints);
  ierr = PetscMemzero(B, npoints*pdim * sizeof(PetscReal));CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (i = 0; i < pdim; ++i) {
      for (d = 0; d < dim; ++d) {
        if (PetscAbsReal(points[p*dim+d] - pt->quad->points[p*dim+d]) > 1.0e-10) break;
      }
      if (d >= dim) {B[p*pdim+i] = 1.0; break;}
    }
  }
  /* Replicate for other components */
  for (c = 1; c < sp->Nc; ++c) {
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        B[(c*npoints + p)*pdim + i] = B[p*pdim + i];
      }
    }
  }
  if (D) {ierr = PetscMemzero(D, npoints*pdim*dim * sizeof(PetscReal));CHKERRQ(ierr);}
  if (H) {ierr = PetscMemzero(H, npoints*pdim*dim*dim * sizeof(PetscReal));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceInitialize_Point(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_Point;
  sp->ops->setup          = PetscSpaceSetUp_Point;
  sp->ops->view           = PetscSpaceView_Point;
  sp->ops->destroy        = PetscSpaceDestroy_Point;
  sp->ops->getdimension   = PetscSpaceGetDimension_Point;
  sp->ops->evaluate       = PetscSpaceEvaluate_Point;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOINT = "point" - A PetscSpace object that encapsulates functions defined on a set of quadrature points.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Point(PetscSpace sp)
{
  PetscSpace_Point *pt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&pt);CHKERRQ(ierr);
  sp->data = pt;

  pt->numVariables = 0;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &pt->quad);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(pt->quad, 0, 1, 0, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscSpaceInitialize_Point(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePointSetPoints - Sets the evaluation points for the space to coincide with the points of a quadrature rule

  Logically collective

  Input Parameters:
+ sp - The PetscSpace
- q  - The PetscQuadrature defining the points

  Level: intermediate

.keywords: PetscSpacePoint
.seealso: PetscSpaceCreate(), PetscSpaceSetType()
@*/
PetscErrorCode PetscSpacePointSetPoints(PetscSpace sp, PetscQuadrature q)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(q, PETSC_OBJECT_CLASSID, 2);
  ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
  ierr = PetscQuadratureDuplicate(q, &pt->quad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePointGetPoints - Gets the evaluation points for the space as the points of a quadrature rule

  Logically collective

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. q  - The PetscQuadrature defining the points

  Level: intermediate

.keywords: PetscSpacePoint
.seealso: PetscSpaceCreate(), PetscSpaceSetType()
@*/
PetscErrorCode PetscSpacePointGetPoints(PetscSpace sp, PetscQuadrature *q)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(q, 2);
  *q = pt->quad;
  PetscFunctionReturn(0);
}


PetscClassId PETSCDUALSPACE_CLASSID = 0;

PetscFunctionList PetscDualSpaceList              = NULL;
PetscBool         PetscDualSpaceRegisterAllCalled = PETSC_FALSE;

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

.keywords: PetscDualSpace, register
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

  Collective on PetscDualSpace

  Input Parameters:
+ sp   - The PetscDualSpace object
- name - The kind of space

  Options Database Key:
. -petscdualspace_type <type> - Sets the PetscDualSpace type; use -help for a list of available types

  Level: intermediate

.keywords: PetscDualSpace, set, type
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

.keywords: PetscDualSpace, get, type, name
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

/*@
  PetscDualSpaceView - Views a PetscDualSpace

  Collective on PetscDualSpace

  Input Parameter:
+ sp - the PetscDualSpace object to view
- v  - the viewer

  Level: developer

.seealso PetscDualSpaceDestroy()
@*/
PetscErrorCode PetscDualSpaceView(PetscDualSpace sp, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);
  }
  if (sp->ops->view) {
    ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetFromOptions - sets parameters in a PetscDualSpace from the options database

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to set options for

  Options Database:
. -petscspace_order the approximation order of the space

  Level: developer

.seealso PetscDualSpaceView()
@*/
PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace sp)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

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
  ierr = PetscOptionsInt("-petscdualspace_order", "The approximation order", "PetscDualSpaceSetOrder", sp->order, &sp->order, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscdualspace_components", "The number of components", "PetscDualSpaceSetNumComponents", sp->Nc, &sp->Nc, NULL);CHKERRQ(ierr);
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(PetscOptionsObject,sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSetUp - Construct a basis for the PetscDualSpace

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to setup

  Level: developer

.seealso PetscDualSpaceView(), PetscDualSpaceDestroy()
@*/
PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (sp->setupcalled) PetscFunctionReturn(0);
  sp->setupcalled = PETSC_TRUE;
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceDestroy - Destroys a PetscDualSpace object

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to destroy

  Level: developer

.seealso PetscDualSpaceView()
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
  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);

  if ((*sp)->ops->destroy) {ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceCreate - Creates an empty PetscDualSpace object. The type can then be set with PetscDualSpaceSetType().

  Collective on MPI_Comm

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
  s->setupcalled = PETSC_FALSE;

  *sp = s;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceDuplicate - Creates a duplicate PetscDualSpace object, however it is not setup.

  Collective on PetscDualSpace

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
  PetscDualSpaceLagrangeGetTensor - Get the tensor nature of the dual space

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. tensor - Whether the dual space has tensor layout (vs. simplicial)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeSetTensor(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetTensor(PetscDualSpace sp, PetscBool *tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeGetTensor_C",(PetscDualSpace,PetscBool *),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetTensor - Set the tensor nature of the dual space

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- tensor - Whether the dual space has tensor layout (vs. simplicial)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeGetTensor(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetTensor(PetscDualSpace sp, PetscBool tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeSetTensor_C",(PetscDualSpace,PetscBool),(sp,tensor));CHKERRQ(ierr);
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
  PetscDualSpaceCreateReferenceCell - Create a DMPLEX with the appropriate FEM reference cell

  Collective on PetscDualSpace

  Input Parameters:
+ sp      - The PetscDualSpace
. dim     - The spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameter:
. refdm - The reference cell

  Level: advanced

.keywords: PetscDualSpace, reference cell
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
. cgeom   - A context with geometric information for this cell, we use v0 (the initial vertex) and J (the Jacobian)
. numComp - The number of components for the function
. func    - The input function
- ctx     - A context for the function

  Output Parameter:
. value   - numComp output values

  Note: The calling sequence for the callback func is given by:

$ func(PetscInt dim, PetscReal time, const PetscReal x[],
$      PetscInt numComponents, PetscScalar values[], void *ctx)

  Level: developer

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApply(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFECellGeom *cgeom, PetscInt numComp, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
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

  Level: developer

.seealso: PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceApplyDefault(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFECellGeom *cgeom, PetscInt Nc, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  n;
  const PetscReal *points, *weights;
  PetscReal        x[3];
  PetscScalar     *val;
  PetscInt         dim, qNc, c, Nq, q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 5);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &n);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(n, &dim, &qNc, &Nq, &points, &weights);CHKERRQ(ierr);
  if (dim != cgeom->dim) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature spatial dimension %D != cell geometry dimension %D", dim, cgeom->dim);
  if (qNc != Nc) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_SIZ, "The quadrature components %D != function components %D", qNc, Nc);
  ierr = DMGetWorkArray(dm, Nc, PETSC_SCALAR, &val);CHKERRQ(ierr);
  *value = 0.0;
  for (q = 0; q < Nq; ++q) {
    CoordinatesRefToReal(cgeom->dimEmbed, dim, cgeom->v0, cgeom->J, &points[q*dim], x);
    ierr = (*func)(cgeom->dimEmbed, time, x, Nc, val, ctx);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      *value += val[c]*weights[q*Nc+c];
    }
  }
  ierr = DMRestoreWorkArray(dm, Nc, PETSC_SCALAR, &val);CHKERRQ(ierr);
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
. value - The output value

  Note: The calling sequence for the callback func is given by:

$ func(PetscInt dim, PetscReal time, const PetscReal x[],
$      PetscInt numComponents, PetscScalar values[], void *ctx)

and the idea is to evaluate the functional as an integral

$ n(f) = int dx n(x) . f(x)

where both n and f have Nc components.

  Level: developer

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
  ierr = DMGetWorkArray(dm, Nc, PETSC_SCALAR, &val);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) value[c] = 0.0;
  for (q = 0; q < Nq; ++q) {
    ierr = (*func)(dimEmbed, time, cgeom->centroid, Nc, val, ctx);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      value[c] += val[c]*weights[q*Nc+c];
    }
  }
  ierr = DMRestoreWorkArray(dm, Nc, PETSC_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceGetHeightSubspace - Get the subset of the dual space basis that is supported on a mesh point of a given height.

  If the dual space is not defined on mesh points of the given height (e.g. if the space is discontinuous and
  pointwise values are not defined on the element boundaries), or if the implementation of PetscDualSpace does not
  support extracting subspaces, then NULL is returned.

  This does not increment the reference count on the returned dual space, and the user should not destroy it.

  Not collective

  Input Parameters:
+ sp - the PetscDualSpace object
- height - the height of the mesh point for which the subspace is desired

  Output Parameter:
. subsp - the subspace

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
  if (sp->ops->getheightsubspace) {
    ierr = (*sp->ops->getheightsubspace)(sp, height, subsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetTensor_Lagrange(PetscDualSpace sp, PetscBool *tensor)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  *tensor = lag->tensorSpace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetTensor_Lagrange(PetscDualSpace sp, PetscBool tensor)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  lag->tensorSpace = tensor;
  PetscFunctionReturn(0);
}

#define BaryIndex(perEdge,a,b,c) (((b)*(2*perEdge+1-(b)))/2)+(c)

#define CartIndex(perEdge,a,b) (perEdge*(a)+b)

static PetscErrorCode PetscDualSpaceGetSymmetries_Lagrange(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{

  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt           dim, order, p, Nc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = DMGetDimension(sp->dm,&dim);CHKERRQ(ierr);
  if (!dim || !lag->continuous || order < 3) PetscFunctionReturn(0);
  if (dim > 3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Lagrange symmetries not implemented for dim = %D > 3",dim);
  if (!lag->symmetries) { /* store symmetries */
    PetscDualSpace hsp;
    DM             K;
    PetscInt       numPoints = 1, d;
    PetscInt       numFaces;
    PetscInt       ***symmetries;
    const PetscInt ***hsymmetries;

    if (lag->simplexCell) {
      numFaces = 1 + dim;
      for (d = 0; d < dim; d++) numPoints = numPoints * 2 + 1;
    }
    else {
      numPoints = PetscPowInt(3,dim);
      numFaces  = 2 * dim;
    }
    ierr = PetscCalloc1(numPoints,&symmetries);CHKERRQ(ierr);
    if (0 < dim && dim < 3) { /* compute self symmetries */
      PetscInt **cellSymmetries;

      lag->numSelfSym = 2 * numFaces;
      lag->selfSymOff = numFaces;
      ierr = PetscCalloc1(2*numFaces,&cellSymmetries);CHKERRQ(ierr);
      /* we want to be able to index symmetries directly with the orientations, which range from [-numFaces,numFaces) */
      symmetries[0] = &cellSymmetries[numFaces];
      if (dim == 1) {
        PetscInt dofPerEdge = order - 1;

        if (dofPerEdge > 1) {
          PetscInt i, j, *reverse;

          ierr = PetscMalloc1(dofPerEdge*Nc,&reverse);CHKERRQ(ierr);
          for (i = 0; i < dofPerEdge; i++) {
            for (j = 0; j < Nc; j++) {
              reverse[i*Nc + j] = Nc * (dofPerEdge - 1 - i) + j;
            }
          }
          symmetries[0][-2] = reverse;

          /* yes, this is redundant, but it makes it easier to cleanup if I don't have to worry about what not to free */
          ierr = PetscMalloc1(dofPerEdge*Nc,&reverse);CHKERRQ(ierr);
          for (i = 0; i < dofPerEdge; i++) {
            for (j = 0; j < Nc; j++) {
              reverse[i*Nc + j] = Nc * (dofPerEdge - 1 - i) + j;
            }
          }
          symmetries[0][1] = reverse;
        }
      } else {
        PetscInt dofPerEdge = lag->simplexCell ? (order - 2) : (order - 1), s;
        PetscInt dofPerFace;

        if (dofPerEdge > 1) {
          for (s = -numFaces; s < numFaces; s++) {
            PetscInt *sym, i, j, k, l;

            if (!s) continue;
            if (lag->simplexCell) {
              dofPerFace = (dofPerEdge * (dofPerEdge + 1))/2;
              ierr = PetscMalloc1(Nc*dofPerFace,&sym);CHKERRQ(ierr);
              for (j = 0, l = 0; j < dofPerEdge; j++) {
                for (k = 0; k < dofPerEdge - j; k++, l++) {
                  i = dofPerEdge - 1 - j - k;
                  switch (s) {
                  case -3:
                    sym[Nc*l] = BaryIndex(dofPerEdge,i,k,j);
                    break;
                  case -2:
                    sym[Nc*l] = BaryIndex(dofPerEdge,j,i,k);
                    break;
                  case -1:
                    sym[Nc*l] = BaryIndex(dofPerEdge,k,j,i);
                    break;
                  case 1:
                    sym[Nc*l] = BaryIndex(dofPerEdge,k,i,j);
                    break;
                  case 2:
                    sym[Nc*l] = BaryIndex(dofPerEdge,j,k,i);
                    break;
                  }
                }
              }
            } else {
              dofPerFace = dofPerEdge * dofPerEdge;
              ierr = PetscMalloc1(Nc*dofPerFace,&sym);CHKERRQ(ierr);
              for (j = 0, l = 0; j < dofPerEdge; j++) {
                for (k = 0; k < dofPerEdge; k++, l++) {
                  switch (s) {
                  case -4:
                    sym[Nc*l] = CartIndex(dofPerEdge,k,j);
                    break;
                  case -3:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),k);
                    break;
                  case -2:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),(dofPerEdge - 1 - j));
                    break;
                  case -1:
                    sym[Nc*l] = CartIndex(dofPerEdge,j,(dofPerEdge - 1 - k));
                    break;
                  case 1:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),j);
                    break;
                  case 2:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),(dofPerEdge - 1 - k));
                    break;
                  case 3:
                    sym[Nc*l] = CartIndex(dofPerEdge,k,(dofPerEdge - 1 - j));
                    break;
                  }
                }
              }
            }
            for (i = 0; i < dofPerFace; i++) {
              sym[Nc*i] *= Nc;
              for (j = 1; j < Nc; j++) {
                sym[Nc*i+j] = sym[Nc*i] + j;
              }
            }
            symmetries[0][s] = sym;
          }
        }
      }
    }
    ierr = PetscDualSpaceGetHeightSubspace(sp,1,&hsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetSymmetries(hsp,&hsymmetries,NULL);CHKERRQ(ierr);
    if (hsymmetries) {
      PetscBool      *seen;
      const PetscInt *cone;
      PetscInt       KclosureSize, *Kclosure = NULL;

      ierr = PetscDualSpaceGetDM(sp,&K);CHKERRQ(ierr);
      ierr = PetscCalloc1(numPoints,&seen);CHKERRQ(ierr);
      ierr = DMPlexGetCone(K,0,&cone);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(K,0,PETSC_TRUE,&KclosureSize,&Kclosure);CHKERRQ(ierr);
      for (p = 0; p < numFaces; p++) {
        PetscInt closureSize, *closure = NULL, q;

        ierr = DMPlexGetTransitiveClosure(K,cone[p],PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (q = 0; q < closureSize; q++) {
          PetscInt point = closure[2*q], r;

          if(!seen[point]) {
            for (r = 0; r < KclosureSize; r++) {
              if (Kclosure[2 * r] == point) break;
            }
            seen[point] = PETSC_TRUE;
            symmetries[r] = (PetscInt **) hsymmetries[q];
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(K,cone[p],PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(K,0,PETSC_TRUE,&KclosureSize,&Kclosure);CHKERRQ(ierr);
      ierr = PetscFree(seen);CHKERRQ(ierr);
    }
    lag->symmetries = symmetries;
  }
  if (perms) *perms = (const PetscInt ***) lag->symmetries;
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
  if (perms) {
    PetscValidPointer(perms,2);
    *perms = NULL;
  }
  if (flips) {
    PetscValidPointer(flips,3);
    *flips = NULL;
  }
  if (sp->ops->getsymmetries) {
    ierr = (sp->ops->getsymmetries)(sp,perms,flips);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetDimension_SingleCell_Lagrange(PetscDualSpace sp, PetscInt order, PetscInt *dim)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscReal           D   = 1.0;
  PetscInt            n, i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  *dim = -1;                    /* Ensure that the compiler knows *dim is set. */
  ierr = DMGetDimension(sp->dm, &n);CHKERRQ(ierr);
  if (!lag->tensorSpace) {
    for (i = 1; i <= n; ++i) {
      D *= ((PetscReal) (order+i))/i;
    }
    *dim = (PetscInt) (D + 0.5);
  } else {
    *dim = 1;
    for (i = 0; i < n; ++i) *dim *= (order+1);
  }
  *dim *= sp->Nc;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceCreateHeightSubspace_Lagrange(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscBool          continuous, tensor;
  PetscInt           order;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp,2);
  ierr = PetscDualSpaceLagrangeGetContinuity(sp,&continuous);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
  if (height == 0) {
    ierr = PetscObjectReference((PetscObject)sp);CHKERRQ(ierr);
    *bdsp = sp;
  } else if (continuous == PETSC_FALSE || !order) {
    *bdsp = NULL;
  } else {
    DM dm, K;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Asked for dual space at height %d for dimension %d reference element\n",height,dim);}
    ierr = PetscDualSpaceDuplicate(sp,bdsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(*bdsp, dim-height, lag->simplexCell, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeGetTensor(sp,&tensor);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(*bdsp,tensor);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscInt            Nc    = sp->Nc;
  PetscBool           continuous;
  PetscSection        csection;
  Vec                 coordinates;
  PetscReal          *qpoints, *qweights;
  PetscInt            depth, dim, pdimMax, pStart, pEnd, p, *pStratStart, *pStratEnd, coneSize, d, f = 0, c;
  PetscBool           simplex, tensorSpace;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* Classify element type */
  if (!order) lag->continuous = PETSC_FALSE;
  continuous = lag->continuous;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim+1, &lag->numDof);CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,&pStratStart,depth+1,&pStratEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStratStart[d], &pStratEnd[d]);CHKERRQ(ierr);}
  ierr = DMPlexGetConeSize(dm, pStratStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if (depth == 1) {
    if      (coneSize == dim+1)    simplex = PETSC_TRUE;
    else if (coneSize == 1 << dim) simplex = PETSC_FALSE;
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  } else if (depth == dim) {
    if      (coneSize == dim+1)   simplex = PETSC_TRUE;
    else if (coneSize == 2 * dim) simplex = PETSC_FALSE;
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support cell-vertex meshes or interpolated meshes");
  lag->simplexCell = simplex;
  if (dim > 1 && continuous && lag->simplexCell == lag->tensorSpace) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Mismatching simplex/tensor cells and spaces only allowed for discontinuous elements");
  tensorSpace    = lag->tensorSpace;
  lag->height    = 0;
  lag->subspaces = NULL;
  if (continuous && sp->order > 0 && dim > 0) {
    PetscInt i;

    lag->height = dim;
    ierr = PetscMalloc1(dim,&lag->subspaces);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateHeightSubspace_Lagrange(sp,1,&lag->subspaces[0]);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(lag->subspaces[0]);CHKERRQ(ierr);
    for (i = 1; i < dim; i++) {
      ierr = PetscDualSpaceGetHeightSubspace(lag->subspaces[i-1],1,&lag->subspaces[i]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(lag->subspaces[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, sp->order, &pdimMax);CHKERRQ(ierr);
  pdimMax *= (pStratEnd[depth] - pStratStart[depth]);
  ierr = PetscMalloc1(pdimMax, &sp->functional);CHKERRQ(ierr);
  if (!dim) {
    for (c = 0; c < Nc; ++c) {
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
      ierr = PetscCalloc1(Nc, &qweights);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(sp->functional[f], 0, Nc, 1, NULL, qweights);CHKERRQ(ierr);
      qweights[c] = 1.0;
      ++f;
      lag->numDof[0]++;
    }
  } else {
    PetscInt     *tup;
    PetscReal    *v0, *hv0, *J, *invJ, detJ, hdetJ;
    PetscSection section;

    ierr = PetscSectionCreate(PETSC_COMM_SELF,&section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section,pStart,pEnd);CHKERRQ(ierr);
    ierr = PetscCalloc5(dim+1,&tup,dim,&v0,dim,&hv0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt       pointDim, d, nFunc = 0;
      PetscDualSpace hsp;

      ierr = DMPlexComputeCellGeometryFEM(dm, p, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      for (d = 0; d < depth; d++) {if (p >= pStratStart[d] && p < pStratEnd[d]) break;}
      pointDim = (depth == 1 && d == 1) ? dim : d;
      hsp = ((pointDim < dim) && lag->subspaces) ? lag->subspaces[dim - pointDim - 1] : NULL;
      if (hsp) {
        PetscDualSpace_Lag *hlag = (PetscDualSpace_Lag *) hsp->data;
        DM                 hdm;

        ierr = PetscDualSpaceGetDM(hsp,&hdm);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(hdm, 0, NULL, hv0, NULL, NULL, &hdetJ);CHKERRQ(ierr);
        nFunc = lag->numDof[pointDim] = hlag->numDof[pointDim];
      }
      if (pointDim == dim) {
        /* Cells, create for self */
        PetscInt     orderEff = continuous ? (!tensorSpace ? order-1-dim : order-2) : order;
        PetscReal    denom    = continuous ? order : (!tensorSpace ? order+1+dim : order+2);
        PetscReal    numer    = (!simplex || !tensorSpace) ? 2. : (2./dim);
        PetscReal    dx = numer/denom;
        PetscInt     cdim, d, d2;

        if (orderEff < 0) continue;
        ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, orderEff, &cdim);CHKERRQ(ierr);
        ierr = PetscMemzero(tup,(dim+1)*sizeof(PetscInt));CHKERRQ(ierr);
        if (!tensorSpace) {
          while (!tup[dim]) {
            for (c = 0; c < Nc; ++c) {
              ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
              ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
              ierr = PetscCalloc1(Nc,  &qweights);CHKERRQ(ierr);
              ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
              ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, 1, qpoints, qweights);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                qpoints[d] = v0[d];
                for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d2]+1)*dx);
              }
              qweights[c] = 1.0;
              ++f;
            }
            ierr = LatticePointLexicographic_Internal(dim, orderEff, tup);CHKERRQ(ierr);
          }
        } else {
          while (!tup[dim]) {
            for (c = 0; c < Nc; ++c) {
              ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
              ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
              ierr = PetscCalloc1(Nc,  &qweights);CHKERRQ(ierr);
              ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
              ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, 1, qpoints, qweights);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                qpoints[d] = v0[d];
                for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d2]+1)*dx);
              }
              qweights[c] = 1.0;
              ++f;
            }
            ierr = TensorPointLexicographic_Internal(dim, orderEff, tup);CHKERRQ(ierr);
          }
        }
        lag->numDof[dim] = cdim;
      } else { /* transform functionals from subspaces */
        PetscInt q;

        for (q = 0; q < nFunc; q++, f++) {
          PetscQuadrature fn;
          PetscInt        fdim, Nc, c, nPoints, i;
          const PetscReal *points;
          const PetscReal *weights;
          PetscReal       *qpoints;
          PetscReal       *qweights;

          ierr = PetscDualSpaceGetFunctional(hsp, q, &fn);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(fn,&fdim,&Nc,&nPoints,&points,&weights);CHKERRQ(ierr);
          if (fdim != pointDim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected height dual space dim %D, got %D",pointDim,fdim);CHKERRQ(ierr);
          ierr = PetscMalloc1(nPoints * dim, &qpoints);CHKERRQ(ierr);
          ierr = PetscCalloc1(nPoints * Nc,  &qweights);CHKERRQ(ierr);
          for (i = 0; i < nPoints; i++) {
            PetscInt  j, k;
            PetscReal *qp = &qpoints[i * dim];

            for (c = 0; c < Nc; ++c) qweights[i*Nc+c] = weights[i*Nc+c];
            for (j = 0; j < dim; ++j) qp[j] = v0[j];
            for (j = 0; j < dim; ++j) {
              for (k = 0; k < pointDim; k++) qp[j] += J[dim * j + k] * (points[pointDim * i + k] - hv0[k]);
            }
          }
          ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
          ierr = PetscQuadratureSetOrder(sp->functional[f],0);CHKERRQ(ierr);
          ierr = PetscQuadratureSetData(sp->functional[f],dim,Nc,nPoints,qpoints,qweights);CHKERRQ(ierr);
        }
      }
      ierr = PetscSectionSetDof(section,p,lag->numDof[pointDim]);CHKERRQ(ierr);
    }
    ierr = PetscFree5(tup,v0,hv0,J,invJ);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    { /* reorder to closure order */
      PetscInt *key, count;
      PetscQuadrature *reorder = NULL;

      ierr = PetscCalloc1(f,&key);CHKERRQ(ierr);
      ierr = PetscMalloc1(f*sp->Nc,&reorder);CHKERRQ(ierr);

      for (p = pStratStart[depth], count = 0; p < pStratEnd[depth]; p++) {
        PetscInt *closure = NULL, closureSize, c;

        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (c = 0; c < closureSize; c++) {
          PetscInt point = closure[2 * c], dof, off, i;

          ierr = PetscSectionGetDof(section,point,&dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section,point,&off);CHKERRQ(ierr);
          for (i = 0; i < dof; i++) {
            PetscInt fi = i + off;
            if (!key[fi]) {
              key[fi] = 1;
              reorder[count++] = sp->functional[fi];
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
      ierr = PetscFree(sp->functional);CHKERRQ(ierr);
      sp->functional = reorder;
      ierr = PetscFree(key);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  if (pStratEnd[depth] == 1 && f != pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %d not equal to dimension %d", f, pdimMax);
  ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
  if (f > pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %d is greater than dimension %d", f, pdimMax);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (lag->symmetries) {
    PetscInt **selfSyms = lag->symmetries[0];

    if (selfSyms) {
      PetscInt i, **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        ierr = PetscFree(allocated[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(allocated);CHKERRQ(ierr);
    }
    ierr = PetscFree(lag->symmetries);CHKERRQ(ierr);
  }
  for (i = 0; i < lag->height; i++) {
    ierr = PetscDualSpaceDestroy(&lag->subspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(lag->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(lag->numDof);CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDuplicate_Lagrange(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       order, Nc;
  PetscBool      cont, tensor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(*spNew, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(*spNew, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(*spNew, cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(*spNew, tensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetFromOptions_Lagrange(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscBool      continuous, tensor, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &continuous);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscDualSpace Lagrange Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscdualspace_lagrange_continuity", "Flag for continuous element", "PetscDualSpaceLagrangeSetContinuity", continuous, &continuous, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetContinuity(sp, continuous);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-petscdualspace_lagrange_tensor", "Flag for tensor dual space", "PetscDualSpaceLagrangeSetContinuity", tensor, &tensor, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetTensor(sp, tensor);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceGetDimension_Lagrange(PetscDualSpace sp, PetscInt *dim)
{
  DM              K;
  const PetscInt *numDof;
  PetscInt        spatialDim, Nc, size = 0, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumDof(sp, &numDof);CHKERRQ(ierr);
  ierr = DMGetDimension(K, &spatialDim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(K, 0, NULL, &Nc);CHKERRQ(ierr);
  if (Nc == 1) {ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, sp->order, dim);CHKERRQ(ierr); PetscFunctionReturn(0);}
  for (d = 0; d <= spatialDim; ++d) {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetDepthStratum(K, d, &pStart, &pEnd);CHKERRQ(ierr);
    size += (pEnd-pStart)*numDof[d];
  }
  *dim = size;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceGetNumDof_Lagrange(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  *numDof = lag->numDof;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetContinuity_Lagrange(PetscDualSpace sp, PetscBool *continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(continuous, 2);
  *continuous = lag->continuous;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetContinuity_Lagrange(PetscDualSpace sp, PetscBool continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  lag->continuous = continuous;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetContinuity - Retrieves the flag for element continuity

  Not Collective

  Input Parameter:
. sp         - the PetscDualSpace

  Output Parameter:
. continuous - flag for element continuity

  Level: intermediate

.keywords: PetscDualSpace, Lagrange, continuous, discontinuous
.seealso: PetscDualSpaceLagrangeSetContinuity()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetContinuity(PetscDualSpace sp, PetscBool *continuous)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(continuous, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceLagrangeGetContinuity_C", (PetscDualSpace,PetscBool*),(sp,continuous));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetContinuity - Indicate whether the element is continuous

  Logically Collective on PetscDualSpace

  Input Parameters:
+ sp         - the PetscDualSpace
- continuous - flag for element continuity

  Options Database:
. -petscdualspace_lagrange_continuity <bool>

  Level: intermediate

.keywords: PetscDualSpace, Lagrange, continuous, discontinuous
.seealso: PetscDualSpaceLagrangeGetContinuity()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace sp, PetscBool continuous)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveBool(sp, continuous, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceLagrangeSetContinuity_C", (PetscDualSpace,PetscBool),(sp,continuous));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceGetHeightSubspace_Lagrange(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp,2);
  if (height == 0) {
    *bdsp = sp;
  }
  else {
    DM dm;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Asked for dual space at height %d for dimension %d reference element\n",height,dim);}
    if (height <= lag->height) {
      *bdsp = lag->subspaces[height-1];
    }
    else {
      *bdsp = NULL;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceInitialize_Lagrange(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_Lagrange;
  sp->ops->setup             = PetscDualSpaceSetUp_Lagrange;
  sp->ops->view              = NULL;
  sp->ops->destroy           = PetscDualSpaceDestroy_Lagrange;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_Lagrange;
  sp->ops->getdimension      = PetscDualSpaceGetDimension_Lagrange;
  sp->ops->getnumdof         = PetscDualSpaceGetNumDof_Lagrange;
  sp->ops->getheightsubspace = PetscDualSpaceGetHeightSubspace_Lagrange;
  sp->ops->getsymmetries     = PetscDualSpaceGetSymmetries_Lagrange;
  sp->ops->apply             = PetscDualSpaceApplyDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACELAGRANGE = "lagrange" - A PetscDualSpace object that encapsulates a dual space of pointwise evaluation functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&lag);CHKERRQ(ierr);
  sp->data = lag;

  lag->numDof      = NULL;
  lag->simplexCell = PETSC_TRUE;
  lag->tensorSpace = PETSC_FALSE;
  lag->continuous  = PETSC_TRUE;

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", PetscDualSpaceLagrangeGetTensor_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", PetscDualSpaceLagrangeSetTensor_Lagrange);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetUp_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s  = (PetscDualSpace_Simple *) sp->data;
  DM                     dm = sp->dm;
  PetscInt               dim;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim+1, &s->numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDestroy_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscFree(s->numDof);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDuplicate_Simple(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       dim, d, Nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACESIMPLE);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(sp, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceSimpleSetDimension(*spNew, dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    PetscQuadrature q;

    ierr = PetscDualSpaceGetFunctional(sp, d, &q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSimpleSetFunctional(*spNew, d, q);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceGetDimension_Simple(PetscDualSpace sp, PetscInt *dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *dim = s->dim;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSimpleSetDimension_Simple(PetscDualSpace sp, const PetscInt dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  DM                     dm;
  PetscInt               spatialDim, f;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  for (f = 0; f < s->dim; ++f) {ierr = PetscQuadratureDestroy(&sp->functional[f]);CHKERRQ(ierr);}
  ierr = PetscFree(sp->functional);CHKERRQ(ierr);
  s->dim = dim;
  ierr = PetscCalloc1(s->dim, &sp->functional);CHKERRQ(ierr);
  ierr = PetscFree(s->numDof);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spatialDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spatialDim+1, &s->numDof);CHKERRQ(ierr);
  s->numDof[spatialDim] = dim;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceGetNumDof_Simple(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *numDof = s->numDof;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSimpleSetFunctional_Simple(PetscDualSpace sp, PetscInt f, PetscQuadrature q)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  PetscReal             *weights;
  PetscInt               Nc, c, Nq, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if ((f < 0) || (f >= s->dim)) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Basis index %d not in [0, %d)", f, s->dim);
  ierr = PetscQuadratureDuplicate(q, &sp->functional[f]);CHKERRQ(ierr);
  /* Reweight so that it has unit volume: Do we want to do this for Nc > 1? */
  ierr = PetscQuadratureGetData(sp->functional[f], NULL, &Nc, &Nq, NULL, (const PetscReal **) &weights);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) {
    PetscReal vol = 0.0;

    for (p = 0; p < Nq; ++p) vol += weights[p*Nc+c];
    for (p = 0; p < Nq; ++p) weights[p*Nc+c] /= (vol == 0.0 ? 1.0 : vol);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSimpleSetDimension - Set the number of functionals in the dual space basis

  Logically Collective on PetscDualSpace

  Input Parameters:
+ sp  - the PetscDualSpace
- dim - the basis dimension

  Level: intermediate

.keywords: PetscDualSpace, dimension
.seealso: PetscDualSpaceSimpleSetFunctional()
@*/
PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace sp, PetscInt dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveInt(sp, dim, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceSimpleSetDimension_C", (PetscDualSpace,PetscInt),(sp,dim));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSimpleSetFunctional - Set the given basis element for this dual space

  Not Collective

  Input Parameters:
+ sp  - the PetscDualSpace
. f - the basis index
- q - the basis functional

  Level: intermediate

  Note: The quadrature will be reweighted so that it has unit volume.

.keywords: PetscDualSpace, functional
.seealso: PetscDualSpaceSimpleSetDimension()
@*/
PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace sp, PetscInt func, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp, "PetscDualSpaceSimpleSetFunctional_C", (PetscDualSpace,PetscInt,PetscQuadrature),(sp,func,q));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceInitialize_Simple(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_Simple;
  sp->ops->setup             = PetscDualSpaceSetUp_Simple;
  sp->ops->view              = NULL;
  sp->ops->destroy           = PetscDualSpaceDestroy_Simple;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_Simple;
  sp->ops->getdimension      = PetscDualSpaceGetDimension_Simple;
  sp->ops->getnumdof         = PetscDualSpaceGetNumDof_Simple;
  sp->ops->getheightsubspace = NULL;
  sp->ops->getsymmetries     = NULL;
  sp->ops->apply             = PetscDualSpaceApplyDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACESIMPLE = "simple" - A PetscDualSpace object that encapsulates a dual space of arbitrary functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&s);CHKERRQ(ierr);
  sp->data = s;

  s->dim    = 0;
  s->numDof = NULL;

  ierr = PetscDualSpaceInitialize_Simple(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", PetscDualSpaceSimpleSetDimension_Simple);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", PetscDualSpaceSimpleSetFunctional_Simple);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscClassId PETSCFE_CLASSID = 0;

PetscFunctionList PetscFEList              = NULL;
PetscBool         PetscFERegisterAllCalled = PETSC_FALSE;

/*@C
  PetscFERegister - Adds a new PetscFE implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscFERegister() may be called multiple times to add several user-defined PetscFEs

  Sample usage:
.vb
    PetscFERegister("my_fe", MyPetscFECreate);
.ve

  Then, your PetscFE type can be chosen with the procedural interface via
.vb
    PetscFECreate(MPI_Comm, PetscFE *);
    PetscFESetType(PetscFE, "my_fe");
.ve
   or at runtime via the option
.vb
    -petscfe_type my_fe
.ve

  Level: advanced

.keywords: PetscFE, register
.seealso: PetscFERegisterAll(), PetscFERegisterDestroy()

@*/
PetscErrorCode PetscFERegister(const char sname[], PetscErrorCode (*function)(PetscFE))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscFEList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFESetType - Builds a particular PetscFE

  Collective on PetscFE

  Input Parameters:
+ fem  - The PetscFE object
- name - The kind of FEM space

  Options Database Key:
. -petscfe_type <type> - Sets the PetscFE type; use -help for a list of available types

  Level: intermediate

.keywords: PetscFE, set, type
.seealso: PetscFEGetType(), PetscFECreate()
@*/
PetscErrorCode PetscFESetType(PetscFE fem, PetscFEType name)
{
  PetscErrorCode (*r)(PetscFE);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) fem, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscFERegisterAllCalled) {ierr = PetscFERegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscFEList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscFE type: %s", name);

  if (fem->ops->destroy) {
    ierr              = (*fem->ops->destroy)(fem);CHKERRQ(ierr);
    fem->ops->destroy = NULL;
  }
  ierr = (*r)(fem);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) fem, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetType - Gets the PetscFE type name (as a string) from the object.

  Not Collective

  Input Parameter:
. fem  - The PetscFE

  Output Parameter:
. name - The PetscFE type name

  Level: intermediate

.keywords: PetscFE, get, type, name
.seealso: PetscFESetType(), PetscFECreate()
@*/
PetscErrorCode PetscFEGetType(PetscFE fem, PetscFEType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(name, 2);
  if (!PetscFERegisterAllCalled) {
    ierr = PetscFERegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) fem)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEView - Views a PetscFE

  Collective on PetscFE

  Input Parameter:
+ fem - the PetscFE object to view
- v   - the viewer

  Level: developer

.seealso PetscFEDestroy()
@*/
PetscErrorCode PetscFEView(PetscFE fem, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) fem), &v);CHKERRQ(ierr);
  }
  if (fem->ops->view) {
    ierr = (*fem->ops->view)(fem, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscFESetFromOptions - sets parameters in a PetscFE from the options database

  Collective on PetscFE

  Input Parameter:
. fem - the PetscFE object to set options for

  Options Database:
. -petscfe_num_blocks  the number of cell blocks to integrate concurrently
. -petscfe_num_batches the number of cell batches to integrate serially

  Level: developer

.seealso PetscFEView()
@*/
PetscErrorCode PetscFESetFromOptions(PetscFE fem)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (!((PetscObject) fem)->type_name) {
    defaultType = PETSCFEBASIC;
  } else {
    defaultType = ((PetscObject) fem)->type_name;
  }
  if (!PetscFERegisterAllCalled) {ierr = PetscFERegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) fem);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscfe_type", "Finite element space", "PetscFESetType", PetscFEList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFESetType(fem, name);CHKERRQ(ierr);
  } else if (!((PetscObject) fem)->type_name) {
    ierr = PetscFESetType(fem, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-petscfe_num_blocks", "The number of cell blocks to integrate concurrently", "PetscSpaceSetTileSizes", fem->numBlocks, &fem->numBlocks, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscfe_num_batches", "The number of cell batches to integrate serially", "PetscSpaceSetTileSizes", fem->numBatches, &fem->numBatches, NULL);CHKERRQ(ierr);
  if (fem->ops->setfromoptions) {
    ierr = (*fem->ops->setfromoptions)(PetscOptionsObject,fem);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) fem);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscFEViewFromOptions(fem, NULL, "-petscfe_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFESetUp - Construct data structures for the PetscFE

  Collective on PetscFE

  Input Parameter:
. fem - the PetscFE object to setup

  Level: developer

.seealso PetscFEView(), PetscFEDestroy()
@*/
PetscErrorCode PetscFESetUp(PetscFE fem)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->setup) {ierr = (*fem->ops->setup)(fem);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscFEDestroy - Destroys a PetscFE object

  Collective on PetscFE

  Input Parameter:
. fem - the PetscFE object to destroy

  Level: developer

.seealso PetscFEView()
@*/
PetscErrorCode PetscFEDestroy(PetscFE *fem)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fem) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*fem), PETSCFE_CLASSID, 1);

  if (--((PetscObject)(*fem))->refct > 0) {*fem = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*fem))->refct = 0;

  if ((*fem)->subspaces) {
    PetscInt dim, d;

    ierr = PetscDualSpaceGetDimension((*fem)->dualSpace, &dim);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {ierr = PetscFEDestroy(&(*fem)->subspaces[d]);CHKERRQ(ierr);}
  }
  ierr = PetscFree((*fem)->subspaces);CHKERRQ(ierr);
  ierr = PetscFree((*fem)->invV);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->B, &(*fem)->D, NULL /*&(*fem)->H*/);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->Bf, &(*fem)->Df, NULL /*&(*fem)->Hf*/);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->F, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&(*fem)->basisSpace);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&(*fem)->dualSpace);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->faceQuadrature);CHKERRQ(ierr);

  if ((*fem)->ops->destroy) {ierr = (*(*fem)->ops->destroy)(*fem);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFECreate - Creates an empty PetscFE object. The type can then be set with PetscFESetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscFE object

  Output Parameter:
. fem - The PetscFE object

  Level: beginner

.seealso: PetscFESetType(), PETSCFEGALERKIN
@*/
PetscErrorCode PetscFECreate(MPI_Comm comm, PetscFE *fem)
{
  PetscFE        f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(fem, 2);
  ierr = PetscCitationsRegister(FECitation,&FEcite);CHKERRQ(ierr);
  *fem = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(f, PETSCFE_CLASSID, "PetscFE", "Finite Element", "PetscFE", comm, PetscFEDestroy, PetscFEView);CHKERRQ(ierr);

  f->basisSpace    = NULL;
  f->dualSpace     = NULL;
  f->numComponents = 1;
  f->subspaces     = NULL;
  f->invV          = NULL;
  f->B             = NULL;
  f->D             = NULL;
  f->H             = NULL;
  f->Bf            = NULL;
  f->Df            = NULL;
  f->Hf            = NULL;
  ierr = PetscMemzero(&f->quadrature, sizeof(PetscQuadrature));CHKERRQ(ierr);
  ierr = PetscMemzero(&f->faceQuadrature, sizeof(PetscQuadrature));CHKERRQ(ierr);
  f->blockSize     = 0;
  f->numBlocks     = 1;
  f->batchSize     = 0;
  f->numBatches    = 1;

  *fem = f;
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetSpatialDimension - Returns the spatial dimension of the element

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. dim - The spatial dimension

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetSpatialDimension(PetscFE fem, PetscInt *dim)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFESetNumComponents - Sets the number of components in the element

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- comp - The number of field components

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetNumComponents(PetscFE fem, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->numComponents = comp;
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetNumComponents - Returns the number of components in the element

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. comp - The number of field components

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetNumComponents(PetscFE fem, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(comp, 2);
  *comp = fem->numComponents;
  PetscFunctionReturn(0);
}

/*@
  PetscFESetTileSizes - Sets the tile sizes for evaluation

  Not collective

  Input Parameters:
+ fem - The PetscFE object
. blockSize - The number of elements in a block
. numBlocks - The number of blocks in a batch
. batchSize - The number of elements in a batch
- numBatches - The number of batches in a chunk

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetTileSizes(PetscFE fem, PetscInt blockSize, PetscInt numBlocks, PetscInt batchSize, PetscInt numBatches)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->blockSize  = blockSize;
  fem->numBlocks  = numBlocks;
  fem->batchSize  = batchSize;
  fem->numBatches = numBatches;
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetTileSizes - Returns the tile sizes for evaluation

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameters:
+ blockSize - The number of elements in a block
. numBlocks - The number of blocks in a batch
. batchSize - The number of elements in a batch
- numBatches - The number of batches in a chunk

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetTileSizes(PetscFE fem, PetscInt *blockSize, PetscInt *numBlocks, PetscInt *batchSize, PetscInt *numBatches)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (blockSize)  PetscValidPointer(blockSize,  2);
  if (numBlocks)  PetscValidPointer(numBlocks,  3);
  if (batchSize)  PetscValidPointer(batchSize,  4);
  if (numBatches) PetscValidPointer(numBatches, 5);
  if (blockSize)  *blockSize  = fem->blockSize;
  if (numBlocks)  *numBlocks  = fem->numBlocks;
  if (batchSize)  *batchSize  = fem->batchSize;
  if (numBatches) *numBatches = fem->numBatches;
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetBasisSpace - Returns the PetscSpace used for approximation of the solution

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. sp - The PetscSpace object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetBasisSpace(PetscFE fem, PetscSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(sp, 2);
  *sp = fem->basisSpace;
  PetscFunctionReturn(0);
}

/*@
  PetscFESetBasisSpace - Sets the PetscSpace used for approximation of the solution

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- sp - The PetscSpace object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetBasisSpace(PetscFE fem, PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 2);
  ierr = PetscSpaceDestroy(&fem->basisSpace);CHKERRQ(ierr);
  fem->basisSpace = sp;
  ierr = PetscObjectReference((PetscObject) fem->basisSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetDualSpace - Returns the PetscDualSpace used to define the inner product

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. sp - The PetscDualSpace object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetDualSpace(PetscFE fem, PetscDualSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(sp, 2);
  *sp = fem->dualSpace;
  PetscFunctionReturn(0);
}

/*@
  PetscFESetDualSpace - Sets the PetscDualSpace used to define the inner product

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- sp - The PetscDualSpace object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetDualSpace(PetscFE fem, PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  ierr = PetscDualSpaceDestroy(&fem->dualSpace);CHKERRQ(ierr);
  fem->dualSpace = sp;
  ierr = PetscObjectReference((PetscObject) fem->dualSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetQuadrature - Returns the PetscQuadrature used to calculate inner products

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. q - The PetscQuadrature object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(q, 2);
  *q = fem->quadrature;
  PetscFunctionReturn(0);
}

/*@
  PetscFESetQuadrature - Sets the PetscQuadrature used to calculate inner products

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- q - The PetscQuadrature object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscInt       Nc, qNc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscQuadratureGetNumComponents(q, &qNc);CHKERRQ(ierr);
  if ((qNc != 1) && (Nc != qNc)) SETERRQ2(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_SIZ, "FE components %D != Quadrature components %D and non-scalar quadrature", Nc, qNc);
  ierr = PetscFERestoreTabulation(fem, 0, NULL, &fem->B, &fem->D, NULL /*&(*fem)->H*/);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fem->quadrature);CHKERRQ(ierr);
  fem->quadrature = q;
  ierr = PetscObjectReference((PetscObject) q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetFaceQuadrature - Returns the PetscQuadrature used to calculate inner products on faces

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. q - The PetscQuadrature object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetFaceQuadrature(PetscFE fem, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(q, 2);
  *q = fem->faceQuadrature;
  PetscFunctionReturn(0);
}

/*@
  PetscFESetFaceQuadrature - Sets the PetscQuadrature used to calculate inner products on faces

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- q - The PetscQuadrature object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetFaceQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscFERestoreTabulation(fem, 0, NULL, &fem->Bf, &fem->Df, NULL /*&(*fem)->Hf*/);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fem->faceQuadrature);CHKERRQ(ierr);
  fem->faceQuadrature = q;
  ierr = PetscObjectReference((PetscObject) q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetNumDof - Returns the number of dofs (dual basis vectors) associated to mesh points on the reference cell of a given dimension

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameter:
. numDof - Array with the number of dofs per dimension

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFEGetNumDof(PetscFE fem, const PetscInt **numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  ierr = PetscDualSpaceGetNumDof(fem->dualSpace, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetDefaultTabulation - Returns the tabulation of the basis functions at the quadrature points

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameters:
+ B - The basis function values at quadrature points
. D - The basis function derivatives at quadrature points
- H - The basis function second derivatives at quadrature points

  Note:
$ B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFEGetTabulation(), PetscFERestoreTabulation()
@*/
PetscErrorCode PetscFEGetDefaultTabulation(PetscFE fem, PetscReal **B, PetscReal **D, PetscReal **H)
{
  PetscInt         npoints;
  const PetscReal *points;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (B) PetscValidPointer(B, 2);
  if (D) PetscValidPointer(D, 3);
  if (H) PetscValidPointer(H, 4);
  ierr = PetscQuadratureGetData(fem->quadrature, NULL, NULL, &npoints, &points, NULL);CHKERRQ(ierr);
  if (!fem->B) {ierr = PetscFEGetTabulation(fem, npoints, points, &fem->B, &fem->D, NULL/*&fem->H*/);CHKERRQ(ierr);}
  if (B) *B = fem->B;
  if (D) *D = fem->D;
  if (H) *H = fem->H;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetFaceTabulation(PetscFE fem, PetscReal **Bf, PetscReal **Df, PetscReal **Hf)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (Bf) PetscValidPointer(Bf, 2);
  if (Df) PetscValidPointer(Df, 3);
  if (Hf) PetscValidPointer(Hf, 4);
  if (!fem->Bf) {
    PetscFECellGeom  cgeom;
    PetscQuadrature  fq;
    PetscDualSpace   sp;
    DM               dm;
    const PetscInt  *faces;
    PetscInt         dim, numFaces, f, npoints, q;
    const PetscReal *points;
    PetscReal       *facePoints;

    ierr = PetscFEGetDualSpace(fem, &sp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, 0, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, 0, &faces);CHKERRQ(ierr);
    ierr = PetscFEGetFaceQuadrature(fem, &fq);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(fq, NULL, NULL, &npoints, &points, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFaces*npoints*dim, &facePoints);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexComputeCellGeometryFEM(dm, faces[f], NULL, cgeom.v0, cgeom.J, NULL, &cgeom.detJ);CHKERRQ(ierr);
      for (q = 0; q < npoints; ++q) CoordinatesRefToReal(dim, dim-1, cgeom.v0, cgeom.J, &points[q*(dim-1)], &facePoints[(f*npoints+q)*dim]);
    }
    ierr = PetscFEGetTabulation(fem, numFaces*npoints, facePoints, &fem->Bf, &fem->Df, NULL/*&fem->Hf*/);CHKERRQ(ierr);
    ierr = PetscFree(facePoints);CHKERRQ(ierr);
  }
  if (Bf) *Bf = fem->Bf;
  if (Df) *Df = fem->Df;
  if (Hf) *Hf = fem->Hf;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetFaceCentroidTabulation(PetscFE fem, PetscReal **F)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(F, 2);
  if (!fem->F) {
    PetscDualSpace  sp;
    DM              dm;
    const PetscInt *cone;
    PetscReal      *centroids;
    PetscInt        dim, numFaces, f;

    ierr = PetscFEGetDualSpace(fem, &sp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, 0, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, 0, &cone);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFaces*dim, &centroids);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {ierr = DMPlexComputeCellGeometryFVM(dm, cone[f], NULL, &centroids[f*dim], NULL);CHKERRQ(ierr);}
    ierr = PetscFEGetTabulation(fem, numFaces, centroids, &fem->F, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFree(centroids);CHKERRQ(ierr);
  }
  *F = fem->F;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not collective

  Input Parameters:
+ fem     - The PetscFE object
. npoints - The number of tabulation points
- points  - The tabulation point coordinates

  Output Parameters:
+ B - The basis function values at tabulation points
. D - The basis function derivatives at tabulation points
- H - The basis function second derivatives at tabulation points

  Note:
$ B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFERestoreTabulation(), PetscFEGetDefaultTabulation()
@*/
PetscErrorCode PetscFEGetTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         comp; /* Field components */
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!npoints) {
    if (B) *B = NULL;
    if (D) *D = NULL;
    if (H) *H = NULL;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  if (B) {ierr = DMGetWorkArray(dm, npoints*pdim*comp, PETSC_REAL, B);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*comp*dim, PETSC_REAL, D);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*comp*dim*dim, PETSC_REAL, H);CHKERRQ(ierr);}
  ierr = (*fem->ops->gettabulation)(fem, npoints, points, B ? *B : NULL, D ? *D : NULL, H ? *H : NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFERestoreTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  if (B && *B) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, B);CHKERRQ(ierr);}
  if (D && *D) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, D);CHKERRQ(ierr);}
  if (H && *H) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, H);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEDestroy_Basic(PetscFE fem)
{
  PetscFE_Basic *b = (PetscFE_Basic *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEView_Basic_Ascii(PetscFE fe, PetscViewer viewer)
{
  PetscSpace        basis;
  PetscDualSpace    dual;
  PetscQuadrature   q = NULL;
  PetscInt          dim, Nc, Nq;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetBasisSpace(fe, &basis);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &dual);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, &dim, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Basic Finite Element:\n");CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "  dimension:       %d\n", dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  components:      %d\n", Nc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  num quad points: %d\n", Nq);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureView(q, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "  dimension:       %d\n", dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  components:      %d\n", Nc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  num quad points: %d\n", Nq);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscSpaceView(basis, viewer);CHKERRQ(ierr);
  ierr = PetscDualSpaceView(dual, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEView_Basic(PetscFE fe, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscFEView_Basic_Ascii(fe, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Construct the change of basis from prime basis to nodal basis */
PetscErrorCode PetscFESetUp_Basic(PetscFE fem)
{
  PetscScalar   *work, *invVscalar;
  PetscBLASInt  *pivots;
  PetscBLASInt   n, info;
  PetscInt       pdim, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(pdim*pdim,&fem->invV);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(pdim*pdim,&invVscalar);CHKERRQ(ierr);
#else
  invVscalar = fem->invV;
#endif
  for (j = 0; j < pdim; ++j) {
    PetscReal       *Bf;
    PetscQuadrature  f;
    const PetscReal *points, *weights;
    PetscInt         Nc, Nq, q, k, c;

    ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(f, NULL, &Nc, &Nq, &points, &weights);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nc*Nq*pdim,&Bf);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(fem->basisSpace, Nq, points, Bf, NULL, NULL);CHKERRQ(ierr);
    for (k = 0; k < pdim; ++k) {
      /* V_{jk} = n_j(\phi_k) = \int \phi_k(x) n_j(x) dx */
      invVscalar[j*pdim+k] = 0.0;

      for (q = 0; q < Nq; ++q) {
        for (c = 0; c < Nc; ++c) invVscalar[j*pdim+k] += Bf[(q*pdim + k)*Nc + c]*weights[q*Nc + c];
      }
    }
    ierr = PetscFree(Bf);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(pdim,&pivots,pdim,&work);CHKERRQ(ierr);
  n = pdim;
  PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, invVscalar, &n, pivots, &info));
  PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, invVscalar, &n, pivots, work, &n, &info));
#if defined(PETSC_USE_COMPLEX)
  for (j = 0; j < pdim*pdim; j++) fem->invV[j] = PetscRealPart(invVscalar[j]);
  ierr = PetscFree(invVscalar);CHKERRQ(ierr);
#endif
  ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetDimension_Basic(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetTabulation_Basic(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal *B, PetscReal *D, PetscReal *H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         Nc;   /* Field components */
  PetscReal       *tmpB, *tmpD, *tmpH;
  PetscInt         p, d, j, k, c;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (B) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? tmpH : NULL);CHKERRQ(ierr);
  /* Translate to the nodal basis */
  for (p = 0; p < npoints; ++p) {
    if (B) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        const PetscInt i = (p*pdim + j)*Nc;

        for (c = 0; c < Nc; ++c) {
          B[i+c] = 0.0;
          for (k = 0; k < pdim; ++k) {
            B[i+c] += fem->invV[k*pdim+j] * tmpB[(p*pdim + k)*Nc+c];
          }
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (c = 0; c < Nc; ++c) {
          for (d = 0; d < dim; ++d) {
            const PetscInt i = ((p*pdim + j)*Nc + c)*dim + d;

            D[i] = 0.0;
            for (k = 0; k < pdim; ++k) {
              D[i] += fem->invV[k*pdim+j] * tmpD[((p*pdim + k)*Nc + c)*dim + d];
            }
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (c = 0; c < Nc; ++c) {
          for (d = 0; d < dim*dim; ++d) {
            const PetscInt i = ((p*pdim + j)*Nc + c)*dim*dim + d;

            H[i] = 0.0;
            for (k = 0; k < pdim; ++k) {
              H[i] += fem->invV[k*pdim+j] * tmpH[((p*pdim + k)*Nc + c)*dim*dim + d];
            }
          }
        }
      }
    }
  }
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrate_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                      const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal integral[])
{
  const PetscInt     debug = 0;
  PetscPointFunc     obj_func;
  PetscQuadrature    quad;
  PetscScalar       *u, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, e;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetObjective(prob, field, &obj_func);CHKERRQ(ierr);
  if (!obj_func) PetscFunctionReturn(0);
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = cgeom[e].v0;
    const PetscReal *J    = cgeom[e].J;
    const PetscReal *invJ = cgeom[e].invJ;
    const PetscReal  detJ = cgeom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], NULL, u, u_x, NULL);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, numConstants, constants, &integrand);
      integrand *= detJ*quadWeights[q];
      integral[field] += PetscRealPart(integrand);
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", PetscRealPart(integrand), integral[field]);CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscPointFunc     f0_func;
  PetscPointFunc     f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = cgeom[e].v0;
    const PetscReal *J    = cgeom[e].J;
    const PetscReal *invJ = cgeom[e].invJ;
    const PetscReal  detJ = cgeom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    ierr = PetscMemzero(f0, Nq*NcI* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, numConstants, constants, &f0[q*NcI]);
      if (f1_func) {
        ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, numConstants, constants, refSpaceDer);
      }
      TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0_func ? f0 : NULL, f1_func ? f1 : NULL);
    }
    UpdateElementVec(dim, Nq, NbI, NcI, BI, DI, f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEFaceGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscBdPointFunc   f0_func;
  PetscBdPointFunc   f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  if (!f0_func && !f1_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    const PetscReal *v0   = fgeom[e].v0;
    const PetscReal *J    = fgeom[e].J;
    const PetscReal *invJ = fgeom[e].invJ[0];
    const PetscReal  detJ = fgeom[e].detJ;
    const PetscReal *n    = fgeom[e].n;
    const PetscInt   face = fgeom[e].face[0];
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    ierr = PetscMemzero(f0, Nq*NcI* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
     }
     for (q = 0; q < Nq; ++q) {
       if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
       CoordinatesRefToReal(dim, dim-1, v0, J, &quadPoints[q*(dim-1)], x);
       EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
       if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
       if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, n, numConstants, constants, &f0[q*NcI]);
       if (f1_func) {
         ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
         f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, n, numConstants, constants, refSpaceDer);
       }
       TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0_func ? f0 : NULL, f1_func ? f1 : NULL);
     }
     UpdateElementVec(dim, Nq, NbI, NcI, &BI[face*Nq*NbI*NcI], &DI[face*Nq*NbI*NcI*dim], f0, f1, &elemVec[cOffset+fOffset]);
     cOffset    += totDim;
     cOffsetAux += totDimAux;
   }
   PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscPointJac      g0_func;
  PetscPointJac      g1_func;
  PetscPointJac      g2_func;
  PetscPointJac      g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI, *BJ, *DJ;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscDSGetDynamicJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscDSGetJacobianPreconditioner(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscDSGetJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  }
  if (!g0_func && !g1_func && !g2_func && !g3_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = geom[e].v0;
    const PetscReal *J    = geom[e].J;
    const PetscReal *invJ = geom[e].invJ;
    const PetscReal  detJ = geom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *BIq = &BI[q*NbI*NcI], *BJq = &BJ[q*NbJ*NcJ];
      const PetscReal *DIq = &DI[q*NbI*NcI*dim], *DJq = &DJ[q*NbJ*NcJ*dim];
      const PetscReal  w = detJ*quadWeights[q];
      PetscInt f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              for (dp = 0; dp < dim; ++dp) {
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*dim+d)*dim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= w;
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+f; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+g; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += BIq[fidx]*g0[fc*NcJ+gc]*BJq[gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += BIq[fidx]*g1[(fc*NcJ+gc)*dim+d]*DJq[gidx*dim+d];
                elemMat[fOff] += DIq[fidx*dim+d]*g2[(fc*NcJ+gc)*dim+d]*BJq[gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += DIq[fidx*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*DJq[gidx*dim+d2];
                }
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < NbI; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < NbJ; ++g) {
              const PetscInt j = offsetJ + g*NcJ+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscFE fem, PetscDS prob, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEFaceGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscBdPointJac    g0_func;
  PetscBdPointJac    g1_func;
  PetscBdPointJac    g2_func;
  PetscBdPointJac    g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI, *BJ, *DJ;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetBdJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    const PetscReal *v0   = fgeom[e].v0;
    const PetscReal *J    = fgeom[e].J;
    const PetscReal *invJ = fgeom[e].invJ[0];
    const PetscReal  detJ = fgeom[e].detJ;
    const PetscReal *n    = fgeom[e].n;
    const PetscInt   face = fgeom[e].face[0];
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *BIq = &BI[(face*Nq+q)*NbI*NcI], *BJq = &BJ[(face*Nq+q)*NbJ*NcJ];
      const PetscReal *DIq = &DI[(face*Nq+q)*NbI*NcI*dim], *DJq = &DJ[(face*Nq+q)*NbJ*NcJ*dim];
      const PetscReal  w = detJ*quadWeights[q];
      PetscInt f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim-1, v0, J, &quadPoints[q*(dim-1)], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, n, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              for (dp = 0; dp < dim; ++dp) {
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*dim+d)*dim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= w;
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+f; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+g; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += BIq[fidx]*g0[fc*NcJ+gc]*BJq[gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += BIq[fidx]*g1[(fc*NcJ+gc)*dim+d]*DJq[gidx*dim+d];
                elemMat[fOff] += DIq[fidx*dim+d]*g2[(fc*NcJ+gc)*dim+d]*BJq[gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += DIq[fidx*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*DJq[gidx*dim+d2];
                }
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < NbI; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < NbJ; ++g) {
              const PetscInt j = offsetJ + g*NcJ+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEInitialize_Basic(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Basic;
  fem->ops->view                    = PetscFEView_Basic;
  fem->ops->destroy                 = PetscFEDestroy_Basic;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->gettabulation           = PetscFEGetTabulation_Basic;
  fem->ops->integrate               = PetscFEIntegrate_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  fem->ops->integratebdjacobian     = PetscFEIntegrateBdJacobian_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFEBASIC = "basic" - A PetscFE object that integrates with basic tiling and no vectorization

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_Basic(PetscFE fem)
{
  PetscFE_Basic *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem,&b);CHKERRQ(ierr);
  fem->data = b;

  ierr = PetscFEInitialize_Basic(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEDestroy_Nonaffine(PetscFE fem)
{
  PetscFE_Nonaffine *na = (PetscFE_Nonaffine *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(na);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateResidual_Nonaffine(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                                  const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscPointFunc     f0_func;
  PetscPointFunc     f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt          dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    ierr = PetscMemzero(f0, Nq*Nc[field]* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*Nc[field]*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v0   = cgeom[e*Nq+q].v0;
      const PetscReal *J    = cgeom[e*Nq+q].J;
      const PetscReal *invJ = cgeom[e*Nq+q].invJ;
      const PetscReal  detJ = cgeom[e*Nq+q].detJ;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, numConstants, constants, &f0[q*NcI]);
      if (f1_func) {
        ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, numConstants, constants, refSpaceDer);
      }
      TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0, f1);
    }
    UpdateElementVec(dim, Nq, NbI, NcI, BI, DI, f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdResidual_Nonaffine(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEFaceGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt      debug = 0;
  PetscBdPointFunc    f0_func;
  PetscBdPointFunc    f1_func;
  PetscQuadrature     quad;
  PetscScalar        *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal          *x;
  PetscReal         **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt           *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt            dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  if (!f0_func && !f1_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q, face;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
     face = fgeom[e*Nq].face[0];
     ierr = PetscMemzero(f0, Nq*NcI* sizeof(PetscScalar));CHKERRQ(ierr);
     ierr = PetscMemzero(f1, Nq*NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
     for (q = 0; q < Nq; ++q) {
       const PetscReal *v0   = fgeom[e*Nq+q].v0;
       const PetscReal *J    = fgeom[e*Nq+q].J;
       const PetscReal *invJ = fgeom[e*Nq+q].invJ[0];
       const PetscReal  detJ = fgeom[e*Nq+q].detJ;
       const PetscReal *n    = fgeom[e*Nq+q].n;

       if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
       CoordinatesRefToReal(dim, dim-1, v0, J, &quadPoints[q*(dim-1)], x);
       EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
       if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
       if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, n, numConstants, constants, &f0[q*NcI]);
       if (f1_func) {
         ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
         f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, x, n, numConstants, constants, refSpaceDer);
       }
       TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0_func ? f0 : NULL, f1_func ? f1 : NULL);
     }
     UpdateElementVec(dim, Nq, NbI, NcI, &BI[face*Nq*NbI*NcI], &DI[face*Nq*NbI*NcI*dim], f0, f1, &elemVec[cOffset+fOffset]);
     cOffset    += totDim;
     cOffsetAux += totDimAux;
   }
   PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateJacobian_Nonaffine(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *cgeom,
                                                  const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscPointJac      g0_func;
  PetscPointJac      g1_func;
  PetscPointJac      g2_func;
  PetscPointJac      g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI, *BJ, *DJ;
  PetscInt           NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscDSGetDynamicJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscDSGetJacobianPreconditioner(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscDSGetJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  }
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         qNc, Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v0   = cgeom[e*Nq+q].v0;
      const PetscReal *J    = cgeom[e*Nq+q].J;
      const PetscReal *invJ = cgeom[e*Nq+q].invJ;
      const PetscReal  detJ = cgeom[e*Nq+q].detJ;
      const PetscReal *BIq = &BI[q*NbI*NcI], *BJq = &BJ[q*NbJ*NcJ];
      const PetscReal *DIq = &DI[q*NbI*NcI*dim], *DJq = &DJ[q*NbJ*NcJ*dim];
      const PetscReal  w = detJ*quadWeights[q];
      PetscInt         f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, x, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              for (dp = 0; dp < dim; ++dp) {
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*dim+d)*dim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= w;
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+f; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+g; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += BIq[fidx]*g0[fc*NcJ+gc]*BJq[gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += BIq[fidx]*g1[(fc*NcJ+gc)*dim+d]*DJq[gidx*dim+d];
                elemMat[fOff] += DIq[fidx*dim+d]*g2[(fc*NcJ+gc)*dim+d]*BJq[gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += DIq[fidx*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*DJq[gidx*dim+d2];
                }
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < NbI; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < NbJ; ++g) {
              const PetscInt j = offsetJ + g*NcJ+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEInitialize_Nonaffine(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Basic;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_Nonaffine;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->gettabulation           = PetscFEGetTabulation_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Nonaffine;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Nonaffine;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Nonaffine */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Nonaffine;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFENONAFFINE = "nonaffine" - A PetscFE object that integrates with basic tiling and no vectorization for non-affine mappings

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_Nonaffine(PetscFE fem)
{
  PetscFE_Nonaffine *na;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem, &na);CHKERRQ(ierr);
  fem->data = na;

  ierr = PetscFEInitialize_Nonaffine(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_OPENCL

PetscErrorCode PetscFEDestroy_OpenCL(PetscFE fem)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = clReleaseCommandQueue(ocl->queue_id);CHKERRQ(ierr);
  ocl->queue_id = 0;
  ierr = clReleaseContext(ocl->ctx_id);CHKERRQ(ierr);
  ocl->ctx_id = 0;
  ierr = PetscFree(ocl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define STRING_ERROR_CHECK(MSG) do {CHKERRQ(ierr); string_tail += count; if (string_tail == end_of_buffer) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, MSG);} while(0)
enum {LAPLACIAN = 0, ELASTICITY = 1};

/* NOTE: This is now broken for vector problems. Must redo loops to respect vector basis elements */
/* dim     Number of spatial dimensions:          2                   */
/* N_b     Number of basis functions:             generated           */
/* N_{bt}  Number of total basis functions:       N_b * N_{comp}      */
/* N_q     Number of quadrature points:           generated           */
/* N_{bs}  Number of block cells                  LCM(N_b, N_q)       */
/* N_{bst} Number of block cell components        LCM(N_{bt}, N_q)    */
/* N_{bl}  Number of concurrent blocks            generated           */
/* N_t     Number of threads:                     N_{bl} * N_{bs}     */
/* N_{cbc} Number of concurrent basis      cells: N_{bl} * N_q        */
/* N_{cqc} Number of concurrent quadrature cells: N_{bl} * N_b        */
/* N_{sbc} Number of serial     basis      cells: N_{bs} / N_q        */
/* N_{sqc} Number of serial     quadrature cells: N_{bs} / N_b        */
/* N_{cb}  Number of serial cell batches:         input               */
/* N_c     Number of total cells:                 N_{cb}*N_{t}/N_{comp} */
PetscErrorCode PetscFEOpenCLGenerateIntegrationCode(PetscFE fem, char **string_buffer, PetscInt buffer_length, PetscBool useAux, PetscInt N_bl)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;
  PetscQuadrature q;
  char           *string_tail   = *string_buffer;
  char           *end_of_buffer = *string_buffer + buffer_length;
  char            float_str[]   = "float", double_str[]  = "double";
  char           *numeric_str   = &(float_str[0]);
  PetscInt        op            = ocl->op;
  PetscBool       useField      = PETSC_FALSE;
  PetscBool       useFieldDer   = PETSC_TRUE;
  PetscBool       useFieldAux   = useAux;
  PetscBool       useFieldDerAux= PETSC_FALSE;
  PetscBool       useF0         = PETSC_TRUE;
  PetscBool       useF1         = PETSC_TRUE;
  const PetscReal *points, *weights;
  PetscReal      *basis, *basisDer;
  PetscInt        dim, qNc, N_b, N_c, N_q, N_t, p, d, b, c;
  size_t          count;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &N_b);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &N_c);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, &qNc, &N_q, &points, &weights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  N_t  = N_b * N_c * N_q * N_bl;
  /* Enable device extension for double precision */
  if (ocl->realType == PETSC_DOUBLE) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#endif\n",
                              &count);STRING_ERROR_CHECK("Message to short");
    numeric_str  = &(double_str[0]);
  }
  /* Kernel API */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"\n"
"__kernel void integrateElementQuadrature(int N_cb, __global %s *coefficients, __global %s *coefficientsAux, __global %s *jacobianInverses, __global %s *jacobianDeterminants, __global %s *elemVec)\n"
"{\n",
                       &count, numeric_str, numeric_str, numeric_str, numeric_str, numeric_str);STRING_ERROR_CHECK("Message to short");
  /* Quadrature */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Quadrature points\n"
"   - (x1,y1,x2,y2,...) */\n"
"  const %s points[%d] = {\n",
                       &count, numeric_str, N_q*dim);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    for (d = 0; d < dim; ++d) {
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, points[p*dim+d]);STRING_ERROR_CHECK("Message to short");
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Quadrature weights\n"
"   - (v1,v2,...) */\n"
"  const %s weights[%d] = {\n",
                       &count, numeric_str, N_q);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, weights[p]);STRING_ERROR_CHECK("Message to short");
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  /* Basis Functions */
  ierr = PetscFEGetDefaultTabulation(fem, &basis, &basisDer, NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Nodal basis function evaluations\n"
"    - basis component is fastest varying, the basis function, then point */\n"
"  const %s Basis[%d] = {\n",
                       &count, numeric_str, N_q*N_b*N_c);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    for (b = 0; b < N_b; ++b) {
      for (c = 0; c < N_c; ++c) {
        ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, basis[(p*N_b + b)*N_c + c]);STRING_ERROR_CHECK("Message to short");
      }
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"\n"
"  /* Nodal basis function derivative evaluations,\n"
"      - derivative direction is fastest varying, then basis component, then basis function, then point */\n"
"  const %s%d BasisDerivatives[%d] = {\n",
                       &count, numeric_str, dim, N_q*N_b*N_c);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    for (b = 0; b < N_b; ++b) {
      for (c = 0; c < N_c; ++c) {
        ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "(%s%d)(", &count, numeric_str, dim);STRING_ERROR_CHECK("Message to short");
        for (d = 0; d < dim; ++d) {
          if (d > 0) {
            ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, ", %g", &count, basisDer[((p*N_b + b)*dim + d)*N_c + c]);STRING_ERROR_CHECK("Message to short");
          } else {
            ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g", &count, basisDer[((p*N_b + b)*dim + d)*N_c + c]);STRING_ERROR_CHECK("Message to short");
          }
        }
        ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "),\n", &count);STRING_ERROR_CHECK("Message to short");
      }
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  /* Sizes */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  const int dim    = %d;                           // The spatial dimension\n"
"  const int N_bl   = %d;                           // The number of concurrent blocks\n"
"  const int N_b    = %d;                           // The number of basis functions\n"
"  const int N_comp = %d;                           // The number of basis function components\n"
"  const int N_bt   = N_b*N_comp;                    // The total number of scalar basis functions\n"
"  const int N_q    = %d;                           // The number of quadrature points\n"
"  const int N_bst  = N_bt*N_q;                      // The block size, LCM(N_b*N_comp, N_q), Notice that a block is not processed simultaneously\n"
"  const int N_t    = N_bst*N_bl;                    // The number of threads, N_bst * N_bl\n"
"  const int N_bc   = N_t/N_comp;                    // The number of cells per batch (N_b*N_q*N_bl)\n"
"  const int N_sbc  = N_bst / (N_q * N_comp);\n"
"  const int N_sqc  = N_bst / N_bt;\n"
"  /*const int N_c    = N_cb * N_bc;*/\n"
"\n"
"  /* Calculated indices */\n"
"  /*const int tidx    = get_local_id(0) + get_local_size(0)*get_local_id(1);*/\n"
"  const int tidx    = get_local_id(0);\n"
"  const int blidx   = tidx / N_bst;                  // Block number for this thread\n"
"  const int bidx    = tidx %% N_bt;                   // Basis function mapped to this thread\n"
"  const int cidx    = tidx %% N_comp;                 // Basis component mapped to this thread\n"
"  const int qidx    = tidx %% N_q;                    // Quadrature point mapped to this thread\n"
"  const int blbidx  = tidx %% N_q + blidx*N_q;        // Cell mapped to this thread in the basis phase\n"
"  const int blqidx  = tidx %% N_b + blidx*N_b;        // Cell mapped to this thread in the quadrature phase\n"
"  const int gidx    = get_group_id(1)*get_num_groups(0) + get_group_id(0);\n"
"  const int Goffset = gidx*N_cb*N_bc;\n",
                            &count, dim, N_bl, N_b, N_c, N_q);STRING_ERROR_CHECK("Message to short");
  /* Local memory */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"\n"
"  /* Quadrature data */\n"
"  %s                w;                   // $w_q$, Quadrature weight at $x_q$\n"
"  __local %s         phi_i[%d];    //[N_bt*N_q];  // $\\phi_i(x_q)$, Value of the basis function $i$ at $x_q$\n"
"  __local %s%d       phiDer_i[%d]; //[N_bt*N_q];  // $\\frac{\\partial\\phi_i(x_q)}{\\partial x_d}$, Value of the derivative of basis function $i$ in direction $x_d$ at $x_q$\n"
"  /* Geometric data */\n"
"  __local %s        detJ[%d]; //[N_t];           // $|J(x_q)|$, Jacobian determinant at $x_q$\n"
"  __local %s        invJ[%d];//[N_t*dim*dim];   // $J^{-1}(x_q)$, Jacobian inverse at $x_q$\n",
                            &count, numeric_str, numeric_str, N_b*N_c*N_q, numeric_str, dim, N_b*N_c*N_q, numeric_str, N_t,
                            numeric_str, N_t*dim*dim, numeric_str, N_t*N_b*N_c);STRING_ERROR_CHECK("Message to short");
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* FEM data */\n"
"  __local %s        u_i[%d]; //[N_t*N_bt];       // Coefficients $u_i$ of the field $u|_{\\mathcal{T}} = \\sum_i u_i \\phi_i$\n",
                            &count, numeric_str, N_t*N_b*N_c);STRING_ERROR_CHECK("Message to short");
  if (useAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  __local %s        a_i[%d]; //[N_t];            // Coefficients $a_i$ of the auxiliary field $a|_{\\mathcal{T}} = \\sum_i a_i \\phi^R_i$\n",
                            &count, numeric_str, N_t);STRING_ERROR_CHECK("Message to short");
  }
  if (useF0) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Intermediate calculations */\n"
"  __local %s         f_0[%d]; //[N_t*N_sqc];      // $f_0(u(x_q), \\nabla u(x_q)) |J(x_q)| w_q$\n",
                              &count, numeric_str, N_t*N_q);STRING_ERROR_CHECK("Message to short");
  }
  if (useF1) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  __local %s%d       f_1[%d]; //[N_t*N_sqc];      // $f_1(u(x_q), \\nabla u(x_q)) |J(x_q)| w_q$\n",
                              &count, numeric_str, dim, N_t*N_q);STRING_ERROR_CHECK("Message to short");
  }
  /* TODO: If using elasticity, put in mu/lambda coefficients */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Output data */\n"
"  %s                e_i;                 // Coefficient $e_i$ of the residual\n\n",
                            &count, numeric_str);STRING_ERROR_CHECK("Message to short");
  /* One-time loads */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* These should be generated inline */\n"
"  /* Load quadrature weights */\n"
"  w = weights[qidx];\n"
"  /* Load basis tabulation \\phi_i for this cell */\n"
"  if (tidx < N_bt*N_q) {\n"
"    phi_i[tidx]    = Basis[tidx];\n"
"    phiDer_i[tidx] = BasisDerivatives[tidx];\n"
"  }\n\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  /* Batch loads */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  for (int batch = 0; batch < N_cb; ++batch) {\n"
"    /* Load geometry */\n"
"    detJ[tidx] = jacobianDeterminants[Goffset+batch*N_bc+tidx];\n"
"    for (int n = 0; n < dim*dim; ++n) {\n"
"      const int offset = n*N_t;\n"
"      invJ[offset+tidx] = jacobianInverses[(Goffset+batch*N_bc)*dim*dim+offset+tidx];\n"
"    }\n"
"    /* Load coefficients u_i for this cell */\n"
"    for (int n = 0; n < N_bt; ++n) {\n"
"      const int offset = n*N_t;\n"
"      u_i[offset+tidx] = coefficients[(Goffset*N_bt)+batch*N_t*N_b+offset+tidx];\n"
"    }\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  if (useAux) {
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    /* Load coefficients a_i for this cell */\n"
"    /* TODO: This should not be N_t here, it should be N_bc*N_comp_aux */\n"
"    a_i[tidx] = coefficientsAux[Goffset+batch*N_t+tidx];\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  }
  /* Quadrature phase */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    /* Map coefficients to values at quadrature points */\n"
"    for (int c = 0; c < N_sqc; ++c) {\n"
"      const int cell          = c*N_bl*N_b + blqidx;\n"
"      const int fidx          = (cell*N_q + qidx)*N_comp + cidx;\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  if (useField) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      %s  u[%d]; //[N_comp];     // $u(x_q)$, Value of the field at $x_q$\n",
                              &count, numeric_str, N_c);STRING_ERROR_CHECK("Message to short");
  }
  if (useFieldDer) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      %s%d   gradU[%d]; //[N_comp]; // $\\nabla u(x_q)$, Value of the field gradient at $x_q$\n",
                              &count, numeric_str, dim, N_c);STRING_ERROR_CHECK("Message to short");
  }
  if (useFieldAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      %s  a[%d]; //[1];     // $a(x_q)$, Value of the auxiliary fields at $x_q$\n",
                              &count, numeric_str, 1);STRING_ERROR_CHECK("Message to short");
  }
  if (useFieldDerAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      %s%d   gradA[%d]; //[1]; // $\\nabla a(x_q)$, Value of the auxiliary field gradient at $x_q$\n",
                              &count, numeric_str, dim, 1);STRING_ERROR_CHECK("Message to short");
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"\n"
"      for (int comp = 0; comp < N_comp; ++comp) {\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  if (useField) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        u[comp] = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");}
  if (useFieldDer) {
    switch (dim) {
    case 1:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0; gradU[comp].y = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "        gradU[comp].x = 0.0; gradU[comp].y = 0.0; gradU[comp].z = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      }\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  if (useFieldAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      a[0] = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");
  }
  if (useFieldDerAux) {
    switch (dim) {
    case 1:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0; gradA[0].y = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      gradA[0].x = 0.0; gradA[0].y = 0.0; gradA[0].z = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      /* Get field and derivatives at this quadrature point */\n"
"      for (int i = 0; i < N_b; ++i) {\n"
"        for (int comp = 0; comp < N_comp; ++comp) {\n"
"          const int b    = i*N_comp+comp;\n"
"          const int pidx = qidx*N_bt + b;\n"
"          const int uidx = cell*N_bt + b;\n"
"          %s%d   realSpaceDer;\n\n",
                            &count, numeric_str, dim);STRING_ERROR_CHECK("Message to short");
  if (useField) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"          u[comp] += u_i[uidx]*phi_i[pidx];\n", &count);STRING_ERROR_CHECK("Message to short");}
  if (useFieldDer) {
    switch (dim) {
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
"          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
"          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
"          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"          realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
"          gradU[comp].x += u_i[uidx]*realSpaceDer.x;\n"
"          realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
"          gradU[comp].y += u_i[uidx]*realSpaceDer.y;\n"
"          realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
"          gradU[comp].z += u_i[uidx]*realSpaceDer.z;\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"        }\n"
"      }\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  if (useFieldAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"          a[0] += a_i[cell];\n", &count);STRING_ERROR_CHECK("Message to short");
  }
  /* Calculate residual at quadrature points: Should be generated by an weak form egine */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      /* Process values at quadrature points */\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  switch (op) {
  case LAPLACIAN:
    if (useF0) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_0[fidx] = 4.0;\n", &count);STRING_ERROR_CHECK("Message to short");}
    if (useF1) {
      if (useAux) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = a[0]*gradU[cidx];\n", &count);STRING_ERROR_CHECK("Message to short");}
      else        {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = gradU[cidx];\n", &count);STRING_ERROR_CHECK("Message to short");}
    }
    break;
  case ELASTICITY:
    if (useF0) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_0[fidx] = 4.0;\n", &count);STRING_ERROR_CHECK("Message to short");}
    if (useF1) {
    switch (dim) {
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      switch (cidx) {\n"
"      case 0:\n"
"        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[0].y + gradU[1].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y) + mu*(gradU[1].y + gradU[1].y);\n"
"      }\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      switch (cidx) {\n"
"      case 0:\n"
"        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].y + gradU[1].x);\n"
"        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[0].z + gradU[2].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].y + gradU[1].y);\n"
"        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[1].y + gradU[2].y);\n"
"        break;\n"
"      case 2:\n"
"        f_1[fidx].x = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].x + gradU[0].z);\n"
"        f_1[fidx].y = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].y + gradU[1].z);\n"
"        f_1[fidx].z = lambda*(gradU[0].x + gradU[1].y + gradU[2].z) + mu*(gradU[2].y + gradU[2].z);\n"
"      }\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    }}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PDE operator %d is not supported", op);
  }
  if (useF0) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"      f_0[fidx] *= detJ[cell]*w;\n", &count);STRING_ERROR_CHECK("Message to short");}
  if (useF1) {
    switch (dim) {
    case 1:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"      f_1[fidx].x *= detJ[cell]*w;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"      f_1[fidx].x *= detJ[cell]*w; f_1[fidx].y *= detJ[cell]*w; f_1[fidx].z *= detJ[cell]*w;\n", &count);STRING_ERROR_CHECK("Message to short");break;
    }
  }
  /* Thread transpose */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    }\n\n"
"    /* ==== TRANSPOSE THREADS ==== */\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  /* Basis phase */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    /* Map values at quadrature points to coefficients */\n"
"    for (int c = 0; c < N_sbc; ++c) {\n"
"      const int cell = c*N_bl*N_q + blbidx; /* Cell number in batch */\n"
"\n"
"      e_i = 0.0;\n"
"      for (int q = 0; q < N_q; ++q) {\n"
"        const int pidx = q*N_bt + bidx;\n"
"        const int fidx = (cell*N_q + q)*N_comp + cidx;\n"
"        %s%d   realSpaceDer;\n\n",
                       &count, numeric_str, dim);STRING_ERROR_CHECK("Message to short");

  if (useF0) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,"        e_i += phi_i[pidx]*f_0[fidx];\n", &count);STRING_ERROR_CHECK("Message to short");}
  if (useF1) {
    switch (dim) {
    case 2:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y;\n"
"        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
"        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y;\n"
"        e_i           += realSpaceDer.y*f_1[fidx].y;\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"        realSpaceDer.x = invJ[cell*dim*dim+0*dim+0]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+0]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+0]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.x*f_1[fidx].x;\n"
"        realSpaceDer.y = invJ[cell*dim*dim+0*dim+1]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+1]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+1]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.y*f_1[fidx].y;\n"
"        realSpaceDer.z = invJ[cell*dim*dim+0*dim+2]*phiDer_i[pidx].x + invJ[cell*dim*dim+1*dim+2]*phiDer_i[pidx].y + invJ[cell*dim*dim+2*dim+2]*phiDer_i[pidx].z;\n"
"        e_i           += realSpaceDer.z*f_1[fidx].z;\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      }\n"
"      /* Write element vector for N_{cbc} cells at a time */\n"
"      elemVec[(Goffset + batch*N_bc + c*N_bl*N_q)*N_bt + tidx] = e_i;\n"
"    }\n"
"    /* ==== Could do one write per batch ==== */\n"
"  }\n"
"  return;\n"
"}\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEOpenCLGetIntegrationKernel(PetscFE fem, PetscBool useAux, cl_program *ocl_prog, cl_kernel *ocl_kernel)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;
  PetscInt        dim, N_bl;
  PetscBool       flg;
  char           *buffer;
  size_t          len;
  char            errMsg[8192];
  cl_int          ierr2;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc1(8192, &buffer);CHKERRQ(ierr);
  ierr = PetscFEGetTileSizes(fem, NULL, &N_bl, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFEOpenCLGenerateIntegrationCode(fem, &buffer, 8192, useAux, N_bl);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject)fem)->options,((PetscObject)fem)->prefix, "-petscfe_opencl_kernel_print", &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscPrintf(PetscObjectComm((PetscObject) fem), "OpenCL FE Integration Kernel:\n%s\n", buffer);CHKERRQ(ierr);}
  len  = strlen(buffer);
  *ocl_prog = clCreateProgramWithSource(ocl->ctx_id, 1, (const char **) &buffer, &len, &ierr2);CHKERRQ(ierr2);
  ierr = clBuildProgram(*ocl_prog, 0, NULL, NULL, NULL, NULL);
  if (ierr != CL_SUCCESS) {
    ierr = clGetProgramBuildInfo(*ocl_prog, ocl->dev_id, CL_PROGRAM_BUILD_LOG, 8192*sizeof(char), &errMsg, NULL);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Build failed! Log:\n %s", errMsg);
  }
  ierr = PetscFree(buffer);CHKERRQ(ierr);
  *ocl_kernel = clCreateKernel(*ocl_prog, "integrateElementQuadrature", &ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEOpenCLCalculateGrid(PetscFE fem, PetscInt N, PetscInt blockSize, size_t *x, size_t *y, size_t *z)
{
  const PetscInt Nblocks = N/blockSize;

  PetscFunctionBegin;
  if (N % blockSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid block size %d for %d elements", blockSize, N);
  *z = 1;
  for (*x = (size_t) (PetscSqrtReal(Nblocks) + 0.5); *x > 0; --*x) {
    *y = Nblocks / *x;
    if (*x * *y == Nblocks) break;
  }
  if (*x * *y != Nblocks) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Could not find partition for %d with block size %d", N, blockSize);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEOpenCLLogResidual(PetscFE fem, PetscLogDouble time, PetscLogDouble flops)
{
  PetscFE_OpenCL   *ocl = (PetscFE_OpenCL *) fem->data;
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  PetscInt          stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
    /* Log performance info */
  eventLog->eventInfo[ocl->residualEvent].count++;
  eventLog->eventInfo[ocl->residualEvent].time  += time;
  eventLog->eventInfo[ocl->residualEvent].flops += flops;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateResidual_OpenCL(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                               const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  /* Nbc = batchSize */
  PetscFE_OpenCL   *ocl = (PetscFE_OpenCL *) fem->data;
  PetscPointFunc    f0_func;
  PetscPointFunc    f1_func;
  PetscQuadrature   q;
  PetscInt          dim, qNc;
  PetscInt          N_b;    /* The number of basis functions */
  PetscInt          N_comp; /* The number of basis function components */
  PetscInt          N_bt;   /* The total number of scalar basis functions */
  PetscInt          N_q;    /* The number of quadrature points */
  PetscInt          N_bst;  /* The block size, LCM(N_bt, N_q), Notice that a block is not process simultaneously */
  PetscInt          N_t;    /* The number of threads, N_bst * N_bl */
  PetscInt          N_bl;   /* The number of blocks */
  PetscInt          N_bc;   /* The batch size, N_bl*N_q*N_b */
  PetscInt          N_cb;   /* The number of batches */
  PetscInt          numFlops, f0Flops = 0, f1Flops = 0;
  PetscBool         useAux      = probAux ? PETSC_TRUE : PETSC_FALSE;
  PetscBool         useField    = PETSC_FALSE;
  PetscBool         useFieldDer = PETSC_TRUE;
  PetscBool         useF0       = PETSC_TRUE;
  PetscBool         useF1       = PETSC_TRUE;
  /* OpenCL variables */
  cl_program        ocl_prog;
  cl_kernel         ocl_kernel;
  cl_event          ocl_ev;         /* The event for tracking kernel execution */
  cl_ulong          ns_start;       /* Nanoseconds counter on GPU at kernel start */
  cl_ulong          ns_end;         /* Nanoseconds counter on GPU at kernel stop */
  cl_mem            o_jacobianInverses, o_jacobianDeterminants;
  cl_mem            o_coefficients, o_coefficientsAux, o_elemVec;
  float            *f_coeff = NULL, *f_coeffAux = NULL, *f_invJ = NULL, *f_detJ = NULL;
  double           *d_coeff = NULL, *d_coeffAux = NULL, *d_invJ = NULL, *d_detJ = NULL;
  PetscReal        *r_invJ = NULL, *r_detJ = NULL;
  void             *oclCoeff, *oclCoeffAux, *oclInvJ, *oclDetJ;
  size_t            local_work_size[3], global_work_size[3];
  size_t            realSize, x, y, z;
  const PetscReal   *points, *weights;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!Ne) {ierr = PetscFEOpenCLLogResidual(fem, 0.0, 0.0);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, &qNc, &N_q, &points, &weights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  ierr = PetscFEGetDimension(fem, &N_b);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &N_comp);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscFEGetTileSizes(fem, NULL, &N_bl, &N_bc, &N_cb);CHKERRQ(ierr);
  N_bt  = N_b*N_comp;
  N_bst = N_bt*N_q;
  N_t   = N_bst*N_bl;
  if (N_bc*N_comp != N_t) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, N_bc, N_comp);
  /* Calculate layout */
  if (Ne % (N_cb*N_bc)) { /* Remainder cells */
    ierr = PetscFEIntegrateResidual_Basic(fem, prob, field, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscFEOpenCLCalculateGrid(fem, Ne, N_cb*N_bc, &x, &y, &z);CHKERRQ(ierr);
  local_work_size[0]  = N_bc*N_comp;
  local_work_size[1]  = 1;
  local_work_size[2]  = 1;
  global_work_size[0] = x * local_work_size[0];
  global_work_size[1] = y * local_work_size[1];
  global_work_size[2] = z * local_work_size[2];
  ierr = PetscInfo7(fem, "GPU layout grid(%d,%d,%d) block(%d,%d,%d) with %d batches\n", x, y, z, local_work_size[0], local_work_size[1], local_work_size[2], N_cb);CHKERRQ(ierr);
  ierr = PetscInfo2(fem, " N_t: %d, N_cb: %d\n", N_t, N_cb);
  /* Generate code */
  if (probAux) {
    PetscSpace P;
    PetscInt   NfAux, order, f;

    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    for (f = 0; f < NfAux; ++f) {
      PetscFE feAux;

      ierr = PetscDSGetDiscretization(probAux, f, (PetscObject *) &feAux);CHKERRQ(ierr);
      ierr = PetscFEGetBasisSpace(feAux, &P);CHKERRQ(ierr);
      ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
      if (order > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Can only handle P0 coefficient fields");
    }
  }
  ierr = PetscFEOpenCLGetIntegrationKernel(fem, useAux, &ocl_prog, &ocl_kernel);CHKERRQ(ierr);
  /* Create buffers on the device and send data over */
  ierr = PetscDataTypeGetSize(ocl->realType, &realSize);CHKERRQ(ierr);
  if (sizeof(PetscReal) != realSize) {
    switch (ocl->realType) {
    case PETSC_FLOAT:
    {
      PetscInt c, b, d;

      ierr = PetscMalloc4(Ne*N_bt,&f_coeff,Ne,&f_coeffAux,Ne*dim*dim,&f_invJ,Ne,&f_detJ);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        f_detJ[c] = (float) cgeom[c].detJ;
        for (d = 0; d < dim*dim; ++d) {
          f_invJ[c*dim*dim+d] = (float) cgeom[c].invJ[d];
        }
        for (b = 0; b < N_bt; ++b) {
          f_coeff[c*N_bt+b] = (float) coefficients[c*N_bt+b];
        }
      }
      if (coefficientsAux) { /* Assume P0 */
        for (c = 0; c < Ne; ++c) {
          f_coeffAux[c] = (float) coefficientsAux[c];
        }
      }
      oclCoeff      = (void *) f_coeff;
      if (coefficientsAux) {
        oclCoeffAux = (void *) f_coeffAux;
      } else {
        oclCoeffAux = NULL;
      }
      oclInvJ       = (void *) f_invJ;
      oclDetJ       = (void *) f_detJ;
    }
    break;
    case PETSC_DOUBLE:
    {
      PetscInt c, b, d;

      ierr = PetscMalloc4(Ne*N_bt,&d_coeff,Ne,&d_coeffAux,Ne*dim*dim,&d_invJ,Ne,&d_detJ);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        d_detJ[c] = (double) cgeom[c].detJ;
        for (d = 0; d < dim*dim; ++d) {
          d_invJ[c*dim*dim+d] = (double) cgeom[c].invJ[d];
        }
        for (b = 0; b < N_bt; ++b) {
          d_coeff[c*N_bt+b] = (double) coefficients[c*N_bt+b];
        }
      }
      if (coefficientsAux) { /* Assume P0 */
        for (c = 0; c < Ne; ++c) {
          d_coeffAux[c] = (double) coefficientsAux[c];
        }
      }
      oclCoeff      = (void *) d_coeff;
      if (coefficientsAux) {
        oclCoeffAux = (void *) d_coeffAux;
      } else {
        oclCoeffAux = NULL;
      }
      oclInvJ       = (void *) d_invJ;
      oclDetJ       = (void *) d_detJ;
    }
    break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported PETSc type %d", ocl->realType);
    }
  } else {
    PetscInt c, d;

    ierr = PetscMalloc2(Ne*dim*dim,&r_invJ,Ne,&r_detJ);CHKERRQ(ierr);
    for (c = 0; c < Ne; ++c) {
      r_detJ[c] = cgeom[c].detJ;
      for (d = 0; d < dim*dim; ++d) {
        r_invJ[c*dim*dim+d] = cgeom[c].invJ[d];
      }
    }
    oclCoeff    = (void *) coefficients;
    oclCoeffAux = (void *) coefficientsAux;
    oclInvJ     = (void *) r_invJ;
    oclDetJ     = (void *) r_detJ;
  }
  o_coefficients         = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne*N_bt    * realSize, oclCoeff,    &ierr);CHKERRQ(ierr);
  if (coefficientsAux) {
    o_coefficientsAux    = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne         * realSize, oclCoeffAux, &ierr);CHKERRQ(ierr);
  } else {
    o_coefficientsAux    = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY,                        Ne         * realSize, oclCoeffAux, &ierr);CHKERRQ(ierr);
  }
  o_jacobianInverses     = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne*dim*dim * realSize, oclInvJ,     &ierr);CHKERRQ(ierr);
  o_jacobianDeterminants = clCreateBuffer(ocl->ctx_id, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Ne         * realSize, oclDetJ,     &ierr);CHKERRQ(ierr);
  o_elemVec              = clCreateBuffer(ocl->ctx_id, CL_MEM_WRITE_ONLY,                       Ne*N_bt    * realSize, NULL,        &ierr);CHKERRQ(ierr);
  /* Kernel launch */
  ierr = clSetKernelArg(ocl_kernel, 0, sizeof(cl_int), (void*) &N_cb);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 1, sizeof(cl_mem), (void*) &o_coefficients);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 2, sizeof(cl_mem), (void*) &o_coefficientsAux);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 3, sizeof(cl_mem), (void*) &o_jacobianInverses);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 4, sizeof(cl_mem), (void*) &o_jacobianDeterminants);CHKERRQ(ierr);
  ierr = clSetKernelArg(ocl_kernel, 5, sizeof(cl_mem), (void*) &o_elemVec);CHKERRQ(ierr);
  ierr = clEnqueueNDRangeKernel(ocl->queue_id, ocl_kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &ocl_ev);CHKERRQ(ierr);
  /* Read data back from device */
  if (sizeof(PetscReal) != realSize) {
    switch (ocl->realType) {
    case PETSC_FLOAT:
    {
      float   *elem;
      PetscInt c, b;

      ierr = PetscFree4(f_coeff,f_coeffAux,f_invJ,f_detJ);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ne*N_bt, &elem);CHKERRQ(ierr);
      ierr = clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne*N_bt * realSize, elem, 0, NULL, NULL);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        for (b = 0; b < N_bt; ++b) {
          elemVec[c*N_bt+b] = (PetscScalar) elem[c*N_bt+b];
        }
      }
      ierr = PetscFree(elem);CHKERRQ(ierr);
    }
    break;
    case PETSC_DOUBLE:
    {
      double  *elem;
      PetscInt c, b;

      ierr = PetscFree4(d_coeff,d_coeffAux,d_invJ,d_detJ);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ne*N_bt, &elem);CHKERRQ(ierr);
      ierr = clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne*N_bt * realSize, elem, 0, NULL, NULL);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        for (b = 0; b < N_bt; ++b) {
          elemVec[c*N_bt+b] = (PetscScalar) elem[c*N_bt+b];
        }
      }
      ierr = PetscFree(elem);CHKERRQ(ierr);
    }
    break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported PETSc type %d", ocl->realType);
    }
  } else {
    ierr = PetscFree2(r_invJ,r_detJ);CHKERRQ(ierr);
    ierr = clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne*N_bt * realSize, elemVec, 0, NULL, NULL);CHKERRQ(ierr);
  }
  /* Log performance */
  ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ns_start, NULL);CHKERRQ(ierr);
  ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &ns_end,   NULL);CHKERRQ(ierr);
  f0Flops = 0;
  switch (ocl->op) {
  case LAPLACIAN:
    f1Flops = useAux ? dim : 0;break;
  case ELASTICITY:
    f1Flops = 2*dim*dim;break;
  }
  numFlops = Ne*(
    N_q*(
      N_b*N_comp*((useField ? 2 : 0) + (useFieldDer ? 2*dim*(dim + 1) : 0))
      /*+
       N_ba*N_compa*((useFieldAux ? 2 : 0) + (useFieldDerAux ? 2*dim*(dim + 1) : 0))*/
      +
      N_comp*((useF0 ? f0Flops + 2 : 0) + (useF1 ? f1Flops + 2*dim : 0)))
    +
    N_b*((useF0 ? 2 : 0) + (useF1 ? 2*dim*(dim + 1) : 0)));
  ierr = PetscFEOpenCLLogResidual(fem, (ns_end - ns_start)*1.0e-9, numFlops);CHKERRQ(ierr);
  /* Cleanup */
  ierr = clReleaseMemObject(o_coefficients);CHKERRQ(ierr);
  ierr = clReleaseMemObject(o_coefficientsAux);CHKERRQ(ierr);
  ierr = clReleaseMemObject(o_jacobianInverses);CHKERRQ(ierr);
  ierr = clReleaseMemObject(o_jacobianDeterminants);CHKERRQ(ierr);
  ierr = clReleaseMemObject(o_elemVec);CHKERRQ(ierr);
  ierr = clReleaseKernel(ocl_kernel);CHKERRQ(ierr);
  ierr = clReleaseProgram(ocl_prog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEInitialize_OpenCL(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Basic;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_OpenCL;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->gettabulation           = PetscFEGetTabulation_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_OpenCL;
  fem->ops->integratebdresidual     = NULL/* PetscFEIntegrateBdResidual_OpenCL */;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_OpenCL */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFEOPENCL = "opencl" - A PetscFE object that integrates using a vectorized OpenCL implementation

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_OpenCL(PetscFE fem)
{
  PetscFE_OpenCL *ocl;
  cl_uint         num_platforms;
  cl_platform_id  platform_ids[42];
  cl_uint         num_devices;
  cl_device_id    device_ids[42];
  cl_int          ierr2;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem,&ocl);CHKERRQ(ierr);
  fem->data = ocl;

  /* Init Platform */
  ierr = clGetPlatformIDs(42, platform_ids, &num_platforms);CHKERRQ(ierr);
  if (!num_platforms) SETERRQ(PetscObjectComm((PetscObject) fem), PETSC_ERR_SUP, "No OpenCL platform found.");
  ocl->pf_id = platform_ids[0];
  /* Init Device */
  ierr = clGetDeviceIDs(ocl->pf_id, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices);CHKERRQ(ierr);
  if (!num_devices) SETERRQ(PetscObjectComm((PetscObject) fem), PETSC_ERR_SUP, "No OpenCL device found.");
  ocl->dev_id = device_ids[0];
  /* Create context with one command queue */
  ocl->ctx_id   = clCreateContext(0, 1, &(ocl->dev_id), NULL, NULL, &ierr2);CHKERRQ(ierr2);
  ocl->queue_id = clCreateCommandQueue(ocl->ctx_id, ocl->dev_id, CL_QUEUE_PROFILING_ENABLE, &ierr2);CHKERRQ(ierr2);
  /* Types */
  ocl->realType = PETSC_FLOAT;
  /* Register events */
  ierr = PetscLogEventRegister("OpenCL FEResidual", PETSCFE_CLASSID, &ocl->residualEvent);CHKERRQ(ierr);
  /* Equation handling */
  ocl->op = LAPLACIAN;

  ierr = PetscFEInitialize_OpenCL(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEOpenCLSetRealType(PetscFE fem, PetscDataType realType)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ocl->realType = realType;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEOpenCLGetRealType(PetscFE fem, PetscDataType *realType)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(realType, 2);
  *realType = ocl->realType;
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_OPENCL */

PetscErrorCode PetscFEDestroy_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = CellRefinerRestoreAffineTransforms_Internal(cmp->cellRefiner, &cmp->numSubelements, &cmp->v0, &cmp->jac, &cmp->invjac);CHKERRQ(ierr);
  ierr = PetscFree(cmp->embedding);CHKERRQ(ierr);
  ierr = PetscFree(cmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFESetUp_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  DM                 K;
  PetscReal         *subpoint;
  PetscBLASInt      *pivots;
  PetscBLASInt       n, info;
  PetscScalar       *work, *invVscalar;
  PetscInt           dim, pdim, spdim, j, s;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Get affine mapping from reference cell to each subcell */
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &K);CHKERRQ(ierr);
  ierr = DMGetDimension(K, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetCellRefiner_Internal(K, &cmp->cellRefiner);CHKERRQ(ierr);
  ierr = CellRefinerGetAffineTransforms_Internal(cmp->cellRefiner, &cmp->numSubelements, &cmp->v0, &cmp->jac, &cmp->invjac);CHKERRQ(ierr);
  /* Determine dof embedding into subelements */
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &spdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(cmp->numSubelements*spdim,&cmp->embedding);CHKERRQ(ierr);
  ierr = DMGetWorkArray(K, dim, PETSC_REAL, &subpoint);CHKERRQ(ierr);
  for (s = 0; s < cmp->numSubelements; ++s) {
    PetscInt sd = 0;

    for (j = 0; j < pdim; ++j) {
      PetscBool       inside;
      PetscQuadrature f;
      PetscInt        d, e;

      ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);CHKERRQ(ierr);
      /* Apply transform to first point, and check that point is inside subcell */
      for (d = 0; d < dim; ++d) {
        subpoint[d] = -1.0;
        for (e = 0; e < dim; ++e) subpoint[d] += cmp->invjac[(s*dim + d)*dim+e]*(f->points[e] - cmp->v0[s*dim+e]);
      }
      ierr = CellRefinerInCellTest_Internal(cmp->cellRefiner, subpoint, &inside);CHKERRQ(ierr);
      if (inside) {cmp->embedding[s*spdim+sd++] = j;}
    }
    if (sd != spdim) SETERRQ3(PetscObjectComm((PetscObject) fem), PETSC_ERR_PLIB, "Subelement %d has %d dual basis vectors != %d", s, sd, spdim);
  }
  ierr = DMRestoreWorkArray(K, dim, PETSC_REAL, &subpoint);CHKERRQ(ierr);
  /* Construct the change of basis from prime basis to nodal basis for each subelement */
  ierr = PetscMalloc1(cmp->numSubelements*spdim*spdim,&fem->invV);CHKERRQ(ierr);
  ierr = PetscMalloc2(spdim,&pivots,spdim,&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(cmp->numSubelements*spdim*spdim,&invVscalar);CHKERRQ(ierr);
#else
  invVscalar = fem->invV;
#endif
  for (s = 0; s < cmp->numSubelements; ++s) {
    for (j = 0; j < spdim; ++j) {
      PetscReal       *Bf;
      PetscQuadrature  f;
      const PetscReal *points, *weights;
      PetscInt         Nc, Nq, q, k;

      ierr = PetscDualSpaceGetFunctional(fem->dualSpace, cmp->embedding[s*spdim+j], &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, &Nc, &Nq, &points, &weights);CHKERRQ(ierr);
      ierr = PetscMalloc1(f->numPoints*spdim*Nc,&Bf);CHKERRQ(ierr);
      ierr = PetscSpaceEvaluate(fem->basisSpace, Nq, points, Bf, NULL, NULL);CHKERRQ(ierr);
      for (k = 0; k < spdim; ++k) {
        /* n_j \cdot \phi_k */
        invVscalar[(s*spdim + j)*spdim+k] = 0.0;
        for (q = 0; q < Nq; ++q) {
          invVscalar[(s*spdim + j)*spdim+k] += Bf[q*spdim+k]*weights[q];
        }
      }
      ierr = PetscFree(Bf);CHKERRQ(ierr);
    }
    n = spdim;
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, &invVscalar[s*spdim*spdim], &n, pivots, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, &invVscalar[s*spdim*spdim], &n, pivots, work, &n, &info));
  }
#if defined(PETSC_USE_COMPLEX)
  for (s = 0; s <cmp->numSubelements*spdim*spdim; s++) fem->invV[s] = PetscRealPart(invVscalar[s]);
  ierr = PetscFree(invVscalar);CHKERRQ(ierr);
#endif
  ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetTabulation_Composite(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal *B, PetscReal *D, PetscReal *H)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  DM                 dm;
  PetscInt           pdim;  /* Dimension of FE space P */
  PetscInt           spdim; /* Dimension of subelement FE space P */
  PetscInt           dim;   /* Spatial dimension */
  PetscInt           comp;  /* Field components */
  PetscInt          *subpoints;
  PetscReal         *tmpB, *tmpD, *tmpH, *subpoint;
  PetscInt           p, s, d, e, j, k;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &spdim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  /* Divide points into subelements */
  ierr = DMGetWorkArray(dm, npoints, PETSC_INT, &subpoints);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, dim, PETSC_REAL, &subpoint);CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (s = 0; s < cmp->numSubelements; ++s) {
      PetscBool inside;

      /* Apply transform, and check that point is inside cell */
      for (d = 0; d < dim; ++d) {
        subpoint[d] = -1.0;
        for (e = 0; e < dim; ++e) subpoint[d] += cmp->invjac[(s*dim + d)*dim+e]*(points[p*dim+e] - cmp->v0[s*dim+e]);
      }
      ierr = CellRefinerInCellTest_Internal(cmp->cellRefiner, subpoint, &inside);CHKERRQ(ierr);
      if (inside) {subpoints[p] = s; break;}
    }
    if (s >= cmp->numSubelements) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d was not found in any subelement", p);
  }
  ierr = DMRestoreWorkArray(dm, dim, PETSC_REAL, &subpoint);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (B) {ierr = DMGetWorkArray(dm, npoints*spdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*spdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*spdim*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? tmpH : NULL);CHKERRQ(ierr);
  /* Translate to the nodal basis */
  if (B) {ierr = PetscMemzero(B, npoints*pdim*comp * sizeof(PetscReal));CHKERRQ(ierr);}
  if (D) {ierr = PetscMemzero(D, npoints*pdim*comp*dim * sizeof(PetscReal));CHKERRQ(ierr);}
  if (H) {ierr = PetscMemzero(H, npoints*pdim*comp*dim*dim * sizeof(PetscReal));CHKERRQ(ierr);}
  for (p = 0; p < npoints; ++p) {
    const PetscInt s = subpoints[p];

    if (B) {
      /* Multiply by V^{-1} (spdim x spdim) */
      for (j = 0; j < spdim; ++j) {
        const PetscInt i = (p*pdim + cmp->embedding[s*spdim+j])*comp;

        B[i] = 0.0;
        for (k = 0; k < spdim; ++k) {
          B[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpB[p*spdim + k];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (spdim x spdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim + d;

          D[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            D[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpD[(p*spdim + k)*dim + d];
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim*dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim*dim + d;

          H[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            H[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpH[(p*spdim + k)*dim*dim + d];
          }
        }
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, npoints, PETSC_INT, &subpoints);CHKERRQ(ierr);
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*spdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*spdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*spdim*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEInitialize_Composite(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Composite;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_Composite;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->gettabulation           = PetscFEGetTabulation_Composite;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFECOMPOSITE = "composite" - A PetscFE object that represents a composite element

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem, &cmp);CHKERRQ(ierr);
  fem->data = cmp;

  cmp->cellRefiner    = REFINER_NOOP;
  cmp->numSubelements = -1;
  cmp->v0             = NULL;
  cmp->jac            = NULL;

  ierr = PetscFEInitialize_Composite(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFECompositeGetMapping - Returns the mappings from the reference element to each subelement

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameters:
+ blockSize - The number of elements in a block
. numBlocks - The number of blocks in a batch
. batchSize - The number of elements in a batch
- numBatches - The number of batches in a chunk

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFECompositeGetMapping(PetscFE fem, PetscInt *numSubelements, const PetscReal *v0[], const PetscReal *jac[], const PetscReal *invjac[])
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (numSubelements) {PetscValidPointer(numSubelements, 2); *numSubelements = cmp->numSubelements;}
  if (v0)             {PetscValidPointer(v0, 3);             *v0             = cmp->v0;}
  if (jac)            {PetscValidPointer(jac, 4);            *jac            = cmp->jac;}
  if (invjac)         {PetscValidPointer(invjac, 5);         *invjac         = cmp->invjac;}
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetDimension - Get the dimension of the finite element space on a cell

  Not collective

  Input Parameter:
. fe - The PetscFE

  Output Parameter:
. dim - The dimension

  Level: intermediate

.seealso: PetscFECreate(), PetscSpaceGetDimension(), PetscDualSpaceGetDimension()
@*/
PetscErrorCode PetscFEGetDimension(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  if (fem->ops->getdimension) {ierr = (*fem->ops->getdimension)(fem, dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
Purpose: Compute element vector for chunk of elements

Input:
  Sizes:
     Ne:  number of elements
     Nf:  number of fields
     PetscFE
       dim: spatial dimension
       Nb:  number of basis functions
       Nc:  number of field components
       PetscQuadrature
         Nq:  number of quadrature points

  Geometry:
     PetscFECellGeom[Ne] possibly *Nq
       PetscReal v0s[dim]
       PetscReal n[dim]
       PetscReal jacobians[dim*dim]
       PetscReal jacobianInverses[dim*dim]
       PetscReal jacobianDeterminants
  FEM:
     PetscFE
       PetscQuadrature
         PetscReal   quadPoints[Nq*dim]
         PetscReal   quadWeights[Nq]
       PetscReal   basis[Nq*Nb*Nc]
       PetscReal   basisDer[Nq*Nb*Nc*dim]
     PetscScalar coefficients[Ne*Nb*Nc]
     PetscScalar elemVec[Ne*Nb*Nc]

  Problem:
     PetscInt f: the active field
     f0, f1

  Work Space:
     PetscFE
       PetscScalar f0[Nq*dim];
       PetscScalar f1[Nq*dim*dim];
       PetscScalar u[Nc];
       PetscScalar gradU[Nc*dim];
       PetscReal   x[dim];
       PetscScalar realSpaceDer[dim];

Purpose: Compute element vector for N_cb batches of elements

Input:
  Sizes:
     N_cb: Number of serial cell batches

  Geometry:
     PetscReal v0s[Ne*dim]
     PetscReal jacobians[Ne*dim*dim]        possibly *Nq
     PetscReal jacobianInverses[Ne*dim*dim] possibly *Nq
     PetscReal jacobianDeterminants[Ne]     possibly *Nq
  FEM:
     static PetscReal   quadPoints[Nq*dim]
     static PetscReal   quadWeights[Nq]
     static PetscReal   basis[Nq*Nb*Nc]
     static PetscReal   basisDer[Nq*Nb*Nc*dim]
     PetscScalar coefficients[Ne*Nb*Nc]
     PetscScalar elemVec[Ne*Nb*Nc]

ex62.c:
  PetscErrorCode PetscFEIntegrateResidualBatch(PetscInt Ne, PetscInt numFields, PetscInt field, PetscQuadrature quad[], const PetscScalar coefficients[],
                                               const PetscReal v0s[], const PetscReal jacobians[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[],
                                               void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]),
                                               void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]), PetscScalar elemVec[])

ex52.c:
  PetscErrorCode IntegrateLaplacianBatchCPU(PetscInt Ne, PetscInt Nb, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)
  PetscErrorCode IntegrateElasticityBatchCPU(PetscInt Ne, PetscInt Nb, PetscInt Ncomp, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)

ex52_integrateElement.cu
__global__ void integrateElementQuadrature(int N_cb, realType *coefficients, realType *jacobianInverses, realType *jacobianDeterminants, realType *elemVec)

PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt spatial_dim, PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt Nbl, const PetscScalar coefficients[],
                                                     const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                                     PetscLogEvent event, PetscInt debug, PetscInt pde_op)

ex52_integrateElementOpenCL.c:
PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt spatial_dim, PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt N_bl, const PetscScalar coefficients[],
                                                     const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                                     PetscLogEvent event, PetscInt debug, PetscInt pde_op)

__kernel void integrateElementQuadrature(int N_cb, __global float *coefficients, __global float *jacobianInverses, __global float *jacobianDeterminants, __global float *elemVec)
*/

/*@C
  PetscFEIntegrate - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter
. integral     - the integral for this field

  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrate(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal integral[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 2);
  if (fem->ops->integrate) {ierr = (*fem->ops->integrate)(fem, prob, field, Ne, cgeom, coefficients, probAux, coefficientsAux, integral);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateResidual - Produce the element residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t            - The time

  Output Parameter
. elemVec      - the element residual vectors from each element

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)

  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateResidual(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *cgeom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 2);
  if (fem->ops->integrateresidual) {ierr = (*fem->ops->integrateresidual)(fem, prob, field, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateBdResidual - Produce the element residual vector for a chunk of elements by quadrature integration over a boundary

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. fgeom        - The face geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t            - The time

  Output Parameter
. elemVec      - the element residual vectors from each element

  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBdResidual(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEFaceGeom *fgeom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratebdresidual) {ierr = (*fem->ops->integratebdresidual)(fem, prob, field, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateJacobian - Produce the element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. jtype        - The type of matrix pointwise functions that should be used
. fieldI       - The test field being integrated
. fieldJ       - The basis field being integrated
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t            - The time
- u_tShift     - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter
. elemMat      - the element matrices for the Jacobian from each element

  Note:
$ Loop over batch of elements (e):
$   Loop over element matrix entries (f,fc,g,gc --> i,j):
$     Loop over quadrature points (q):
$       Make u_q and gradU_q (loops over fields,Nb,Ncomp)
$         elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
$                      + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateJacobian(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *cgeom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratejacobian) {ierr = (*fem->ops->integratejacobian)(fem, prob, jtype, fieldI, fieldJ, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateBdJacobian - Produce the boundary element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          = The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. fieldI       - The test field being integrated
. fieldJ       - The basis field being integrated
. Ne           - The number of elements in the chunk
. fgeom        - The face geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t            - The time
- u_tShift     - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter
. elemMat              - the element matrices for the Jacobian from each element

  Note:
$ Loop over batch of elements (e):
$   Loop over element matrix entries (f,fc,g,gc --> i,j):
$     Loop over quadrature points (q):
$       Make u_q and gradU_q (loops over fields,Nb,Ncomp)
$         elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
$                      + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
$                      + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
  Level: developer

.seealso: PetscFEIntegrateJacobian(), PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBdJacobian(PetscFE fem, PetscDS prob, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEFaceGeom *fgeom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratebdjacobian) {ierr = (*fem->ops->integratebdjacobian)(fem, prob, fieldI, fieldJ, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetHeightSubspace(PetscFE fe, PetscInt height, PetscFE *subfe)
{
  PetscSpace      P, subP;
  PetscDualSpace  Q, subQ;
  PetscQuadrature subq;
  PetscFEType     fetype;
  PetscInt        dim, Nc;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscValidPointer(subfe, 3);
  if (height == 0) {
    *subfe = fe;
    PetscFunctionReturn(0);
  }
  ierr = PetscFEGetBasisSpace(fe, &P);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fe, &subq);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(Q, &dim);CHKERRQ(ierr);
  if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);}
  if (height != 1) SETERRQ1(PetscObjectComm((PetscObject) fe), PETSC_ERR_SUP, "Height %D not currently supported", height);
  if (!fe->subspaces) {ierr = PetscCalloc1(dim, &fe->subspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!fe->subspaces[height-1]) {
      PetscFE sub;

      ierr = PetscSpaceGetHeightSubspace(P, height, &subP);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetHeightSubspace(Q, height, &subQ);CHKERRQ(ierr);
      ierr = PetscFECreate(PetscObjectComm((PetscObject) fe), &sub);CHKERRQ(ierr);
      ierr = PetscFEGetType(fe, &fetype);CHKERRQ(ierr);
      ierr = PetscFESetType(sub, fetype);CHKERRQ(ierr);
      ierr = PetscFESetBasisSpace(sub, subP);CHKERRQ(ierr);
      ierr = PetscFESetDualSpace(sub, subQ);CHKERRQ(ierr);
      ierr = PetscFESetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscFESetUp(sub);CHKERRQ(ierr);
      ierr = PetscFESetQuadrature(sub, subq);CHKERRQ(ierr);
      fe->subspaces[height-1] = sub;
    }
    *subfe = fe->subspaces[height-1];
  } else {
    *subfe = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscFERefine - Create a "refined" PetscFE object that refines the reference cell into smaller copies. This is typically used
  to precondition a higher order method with a lower order method on a refined mesh having the same number of dofs (but more
  sparsity). It is also used to create an interpolation between regularly refined meshes.

  Collective on PetscFE

  Input Parameter:
. fe - The initial PetscFE

  Output Parameter:
. feRef - The refined PetscFE

  Level: developer

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
@*/
PetscErrorCode PetscFERefine(PetscFE fe, PetscFE *feRef)
{
  PetscSpace       P, Pref;
  PetscDualSpace   Q, Qref;
  DM               K, Kref;
  PetscQuadrature  q, qref;
  const PetscReal *v0, *jac;
  PetscInt         numComp, numSubelements;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetBasisSpace(fe, &P);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(Q, &K);CHKERRQ(ierr);
  /* Create space */
  ierr = PetscObjectReference((PetscObject) P);CHKERRQ(ierr);
  Pref = P;
  /* Create dual space */
  ierr = PetscDualSpaceDuplicate(Q, &Qref);CHKERRQ(ierr);
  ierr = DMRefine(K, PetscObjectComm((PetscObject) fe), &Kref);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Qref, Kref);CHKERRQ(ierr);
  ierr = DMDestroy(&Kref);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Qref);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) fe), feRef);CHKERRQ(ierr);
  ierr = PetscFESetType(*feRef, PETSCFECOMPOSITE);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*feRef, Pref);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*feRef, Qref);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe,    &numComp);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*feRef, numComp);CHKERRQ(ierr);
  ierr = PetscFESetUp(*feRef);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&Pref);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Qref);CHKERRQ(ierr);
  /* Create quadrature */
  ierr = PetscFECompositeGetMapping(*feRef, &numSubelements, &v0, &jac, NULL);CHKERRQ(ierr);
  ierr = PetscQuadratureExpandComposite(q, numSubelements, v0, jac, &qref);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(*feRef, qref);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&qref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFECreateDefault - Create a PetscFE for basic FEM computation

  Collective on DM

  Input Parameters:
+ dm        - The underlying DM for the domain
. dim       - The spatial dimension
. Nc        - The number of components
. isSimplex - Flag for simplex reference cell, otherwise its a tensor product
. prefix    - The options prefix, or NULL
- qorder    - The quadrature order

  Output Parameter:
. fem - The PetscFE object

  Level: beginner

.keywords: PetscFE, finite element
.seealso: PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscFECreateDefault(DM dm, PetscInt dim, PetscInt Nc, PetscBool isSimplex, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order, quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(P, &tensor);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), fem);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(*fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  qorder = qorder >= 0 ? qorder : order;
  ierr = PetscObjectOptionsBegin((PetscObject)*fem);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscfe_default_quadrature_order","Quadrature order is one less than quadture points per edge","PetscFECreateDefault",qorder,&qorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  quadPointsPerEdge = PetscMax(qorder + 1,1);
  if (isSimplex) {
    ierr = PetscDTGaussJacobiQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
