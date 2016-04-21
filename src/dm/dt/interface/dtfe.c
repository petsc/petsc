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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceRegister"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceView"
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
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);
  }
  if (sp->ops->view) {
    ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetFromOptions"
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
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(PetscOptionsObject,sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscSpaceViewFromOptions(sp, NULL, "-petscspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetUp"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceCreate"
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
  ierr = DMShellCreate(comm, &s->dm);CHKERRQ(ierr);

  *sp = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetDimension"
/* Dimension of the space, i.e. number of basis vectors */
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetOrder"
/*@
  PetscSpaceGetOrder - Return the order of approximation for this space

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. order - The approximation order

  Level: intermediate

.seealso: PetscSpaceSetOrder(), PetscSpaceCreate(), PetscSpace
@*/
PetscErrorCode PetscSpaceGetOrder(PetscSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = sp->order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetOrder"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate"
/*@C
  PetscSpaceEvaluate - Evaluate the basis functions and their derivatives (jet) at each point

  Input Parameters:
+ sp      - The PetscSpace
. npoints - The number of evaluation points
- points  - The point coordinates

  Output Parameters:
+ B - The function evaluations in a npoints x nfuncs array
. D - The derivative evaluations in a npoints x nfuncs x dim array
- H - The second derivative evaluations in a npoints x nfuncs x dim x dim array

  Level: advanced

.seealso: PetscFEGetTabulation(), PetscFEGetDefaultTabulation(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceEvaluate(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  if (sp->ops->evaluate) {ierr = (*sp->ops->evaluate)(sp, npoints, points, B, D, H);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetFromOptions_Polynomial"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialView_Ascii"
static PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Poly  *poly = (PetscSpace_Poly *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    if (poly->tensor) {ierr = PetscViewerASCIIPrintf(viewer, "Tensor polynomial space in %d variables of order %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
  } else {
    if (poly->tensor) {ierr = PetscViewerASCIIPrintf(viewer, "Tensor polynomial space in %d variables of order %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d\n", poly->numVariables, sp->order);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceView_Polynomial"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetUp_Polynomial"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy_Polynomial"
PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(poly->degrees);CHKERRQ(ierr);
  ierr = PetscFree(poly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetDimension_Polynomial"
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LatticePoint_Internal"
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

#undef __FUNCT__
#define __FUNCT__ "TensorPoint_Internal"
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate_Polynomial"
PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         ndegree = sp->order+1;
  PetscInt        *degrees = poly->degrees;
  PetscInt         dim     = poly->numVariables;
  PetscReal       *lpoints, *tmp, *LB, *LD, *LH;
  PetscInt        *ind, *tup;
  PetscInt         pdim, d, der, i, p, deg, o;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
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
    /* B (npoints x pdim) */
    if (poly->tensor) {
      i = 0;
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->order+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          B[p*pdim + i] = 1.0;
          for (d = 0; d < dim; ++d) {
            B[p*pdim + i] *= LB[(tup[d]*dim + d)*npoints + p];
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
            B[p*pdim + i] = 1.0;
            for (d = 0; d < dim; ++d) {
              B[p*pdim + i] *= LB[(tup[d]*dim + d)*npoints + p];
            }
          }
          ++i;
        }
      }
    }
  }
  if (D) {
    /* D (npoints x pdim x dim) */
    if (poly->tensor) {
      i = 0;
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = TensorPoint_Internal(dim, sp->order+1, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          for (der = 0; der < dim; ++der) {
            D[(p*pdim + i)*dim + der] = 1.0;
            for (d = 0; d < dim; ++d) {
              if (d == der) {
                D[(p*pdim + i)*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
              } else {
                D[(p*pdim + i)*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
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
              D[(p*pdim + i)*dim + der] = 1.0;
              for (d = 0; d < dim; ++d) {
                if (d == der) {
                  D[(p*pdim + i)*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
                } else {
                  D[(p*pdim + i)*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
                }
              }
            }
          }
          ++i;
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

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceInitialize_Polynomial"
PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup          = PetscSpaceSetUp_Polynomial;
  sp->ops->view           = PetscSpaceView_Polynomial;
  sp->ops->destroy        = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension   = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate       = PetscSpaceEvaluate_Polynomial;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOLYNOMIAL = "poly" - A PetscSpace object that encapsulates a polynomial space, e.g. P1 is the space of linear polynomials.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceCreate_Polynomial"
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

  ierr = PetscSpaceInitialize_Polynomial(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialSetSymmetric"
PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace sp, PetscBool sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->symmetric = sym;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialGetSymmetric"
PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace sp, PetscBool *sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(sym, 2);
  *sym = poly->symmetric;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialSetTensor"
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
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->tensor = tensor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialGetTensor"
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
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  *tensor = poly->tensor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialSetNumVariables"
PetscErrorCode PetscSpacePolynomialSetNumVariables(PetscSpace sp, PetscInt n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->numVariables = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialGetNumVariables"
PetscErrorCode PetscSpacePolynomialGetNumVariables(PetscSpace sp, PetscInt *n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(n, 2);
  *n = poly->numVariables;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetFromOptions_DG"
PetscErrorCode PetscSpaceSetFromOptions_DG(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_DG *dg = (PetscSpace_DG *) sp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace DG options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_dg_num_variables", "The number of different variables, e.g. x and y", "PetscSpaceDGSetNumVariables", dg->numVariables, &dg->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDGView_Ascii"
PetscErrorCode PetscSpaceDGView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_DG    *dg = (PetscSpace_DG *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "DG space in dimension %d:\n", dg->numVariables);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureView(dg->quad, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "DG space in dimension %d on %d points\n", dg->numVariables, dg->quad->numPoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceView_DG"
PetscErrorCode PetscSpaceView_DG(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpaceDGView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetUp_DG"
PetscErrorCode PetscSpaceSetUp_DG(PetscSpace sp)
{
  PetscSpace_DG *dg = (PetscSpace_DG *) sp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dg->quad->points && sp->order) {
    ierr = PetscDTGaussJacobiQuadrature(dg->numVariables, sp->order, -1.0, 1.0, &dg->quad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy_DG"
PetscErrorCode PetscSpaceDestroy_DG(PetscSpace sp)
{
  PetscSpace_DG *dg = (PetscSpace_DG *) sp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureDestroy(&dg->quad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetDimension_DG"
PetscErrorCode PetscSpaceGetDimension_DG(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_DG *dg = (PetscSpace_DG *) sp->data;

  PetscFunctionBegin;
  *dim = dg->quad->numPoints;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate_DG"
PetscErrorCode PetscSpaceEvaluate_DG(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_DG *dg  = (PetscSpace_DG *) sp->data;
  PetscInt       dim = dg->numVariables, d, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (D || H) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Cannot calculate derivatives for a DG space");
  if (npoints != dg->quad->numPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot evaluate DG space on %d points != %d size", npoints, dg->quad->numPoints);
  ierr = PetscMemzero(B, npoints*npoints * sizeof(PetscReal));CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (d = 0; d < dim; ++d) {
      if (PetscAbsReal(points[p*dim+d] - dg->quad->points[p*dim+d]) > 1.0e-10) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot evaluate DG point (%d, %d) %g != %g", p, d, points[p*dim+d], dg->quad->points[p*dim+d]);
    }
    B[p*npoints+p] = 1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceInitialize_DG"
PetscErrorCode PetscSpaceInitialize_DG(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_DG;
  sp->ops->setup          = PetscSpaceSetUp_DG;
  sp->ops->view           = PetscSpaceView_DG;
  sp->ops->destroy        = PetscSpaceDestroy_DG;
  sp->ops->getdimension   = PetscSpaceGetDimension_DG;
  sp->ops->evaluate       = PetscSpaceEvaluate_DG;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEDG = "dg" - A PetscSpace object that encapsulates functions defined on a set of quadrature points.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceCreate_DG"
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_DG(PetscSpace sp)
{
  PetscSpace_DG *dg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&dg);CHKERRQ(ierr);
  sp->data = dg;

  dg->numVariables    = 0;
  dg->quad->dim       = 0;
  dg->quad->numPoints = 0;
  dg->quad->points    = NULL;
  dg->quad->weights   = NULL;

  ierr = PetscSpaceInitialize_DG(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscClassId PETSCDUALSPACE_CLASSID = 0;

PetscFunctionList PetscDualSpaceList              = NULL;
PetscBool         PetscDualSpaceRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceRegister"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceView"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetFromOptions"
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
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(PetscOptionsObject,sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreate"
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
  s->setupcalled = PETSC_FALSE;

  *sp = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDuplicate"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDM"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetDM"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetOrder"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetOrder"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetFunctional"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetNumDof"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreateReferenceCell"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceApply"
/*@C
  PetscDualSpaceApply - Apply a functional from the dual space basis to an input function

  Input Parameters:
+ sp      - The PetscDualSpace object
. f       - The basis functional index
. time    - The time
. geom    - A context with geometric information for this cell, we use v0 (the initial vertex) and J (the Jacobian)
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
PetscErrorCode PetscDualSpaceApply(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFECellGeom *geom, PetscInt numComp, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  quad;
  PetscReal        x[3];
  PetscScalar     *val;
  PetscInt         dim, q, c;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 5);
  dim  = geom->dim;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &quad);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  for (c = 0; c < numComp; ++c) value[c] = 0.0;
  for (q = 0; q < quad->numPoints; ++q) {
    CoordinatesRefToReal(geom->dimEmbed, dim, geom->v0, geom->J, &quad->points[q*dim], x);
    ierr = (*func)(geom->dimEmbed, time, x, numComp, val, ctx);CHKERRQ(ierr);
    for (c = 0; c < numComp; ++c) {
      value[c] += val[c]*quad->weights[q];
    }
  }
  ierr = DMRestoreWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceApplyFVM"
/*@C
  PetscDualSpaceApplyFVM - Apply a functional from the dual space basis to an input function

  Input Parameters:
+ sp      - The PetscDualSpace object
. f       - The basis functional index
. time    - The time
. geom    - A context with geometric information for this cell, we currently just use the centroid
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
PetscErrorCode PetscDualSpaceApplyFVM(PetscDualSpace sp, PetscInt f, PetscReal time, PetscFVCellGeom *geom, PetscInt numComp, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx, PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  quad;
  PetscScalar     *val;
  PetscInt         dimEmbed, q, c;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 5);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &quad);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  for (c = 0; c < numComp; ++c) value[c] = 0.0;
  for (q = 0; q < quad->numPoints; ++q) {
    ierr = (*func)(dimEmbed, time, geom->centroid, numComp, val, ctx);CHKERRQ(ierr);
    for (c = 0; c < numComp; ++c) {
      value[c] += val[c]*quad->weights[q];
    }
  }
  ierr = DMRestoreWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetHeightSubspace"
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

  Output Parameters:
  bdsp - the subspace: must be destroyed by the user

  Level: advanced

.seealso: PetscDualSpace
@*/
PetscErrorCode PetscDualSpaceGetHeightSubspace(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp,2);
  *bdsp = NULL;
  if (sp->ops->getheightsubspace) {
    ierr = (*sp->ops->getheightsubspace)(sp,height,bdsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension_SingleCell_Lagrange"
static PetscErrorCode PetscDualSpaceGetDimension_SingleCell_Lagrange(PetscDualSpace sp, PetscInt order, PetscInt *dim)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscReal           D   = 1.0;
  PetscInt            n, i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  *dim = -1;                    /* Ensure that the compiler knows *dim is set. */
  ierr = DMGetDimension(sp->dm, &n);CHKERRQ(ierr);
  if (lag->simplex || !lag->continuous) {
    for (i = 1; i <= n; ++i) {
      D *= ((PetscReal) (order+i))/i;
    }
    *dim = (PetscInt) (D + 0.5);
  } else {
    *dim = 1;
    for (i = 0; i < n; ++i) *dim *= (order+1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreateHeightSubspace_Lagrange"
static PetscErrorCode PetscDualSpaceCreateHeightSubspace_Lagrange(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscBool          continuous;
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
  }
  else if (continuous == PETSC_FALSE || !order) {
    *bdsp = NULL;
  }
  else {
    DM dm, K;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (height > dim || height < 0) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Asked for dual space at height %d for dimension %d reference element\n",height,dim);}
    ierr = PetscDualSpaceDuplicate(sp,bdsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(*bdsp, dim-height, lag->simplex, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp_Lagrange"
PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscBool           disc  = lag->continuous ? PETSC_FALSE : PETSC_TRUE;
  PetscSection        csection;
  Vec                 coordinates;
  PetscReal          *qpoints, *qweights;
  PetscInt           *closure = NULL, closureSize, c;
  PetscInt            depth, dim, pdimMax, pMax = 0, *pStart, *pEnd, cell, coneSize, d, n, f = 0;
  PetscBool           simplex;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* Classify element type */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim+1, &lag->numDof);CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,&pStart,depth+1,&pEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);}
  ierr = DMPlexGetConeSize(dm, pStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if (depth == 1) {
    if      (coneSize == dim+1)    simplex = PETSC_TRUE;
    else if (coneSize == 1 << dim) simplex = PETSC_FALSE;
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  }
  else if (depth == dim) {
    if      (coneSize == dim+1)   simplex = PETSC_TRUE;
    else if (coneSize == 2 * dim) simplex = PETSC_FALSE;
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  }
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support cell-vertex meshes or interpolated meshes");
  lag->simplex = simplex;
  ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, sp->order, &pdimMax);CHKERRQ(ierr);
  pdimMax *= (pEnd[dim] - pStart[dim]);
  ierr = PetscMalloc1(pdimMax, &sp->functional);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    pMax = PetscMax(pMax,pEnd[d]);
  }
  if (!dim) {
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], PETSC_DETERMINE, 1, qpoints, qweights);CHKERRQ(ierr);
    qpoints[0]  = 0.0;
    qweights[0] = 1.0;
    ++f;
    lag->numDof[0] = 1;
  } else {
    PetscBT seen;

    ierr = PetscBTCreate(pMax, &seen);CHKERRQ(ierr);
    ierr = PetscBTMemzero(pMax, seen);CHKERRQ(ierr);
    for (cell = pStart[depth]; cell < pEnd[depth]; ++cell) {
      ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        const PetscInt p = closure[c];

        if (PetscBTLookup(seen, p)) continue;
        ierr = PetscBTSet(seen, p);CHKERRQ(ierr);
        if ((p >= pStart[0]) && (p < pEnd[0])) {
          /* Vertices */
          const PetscScalar *coords;
          PetscInt           dof, off, d;

          if (order < 1 || disc) continue;
          ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
          ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
          ierr = PetscMalloc1(1, &qweights);CHKERRQ(ierr);
          ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
          ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, qpoints, qweights);CHKERRQ(ierr);
          ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
          ierr = PetscSectionGetDof(csection, p, &dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
          if (dof != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of coordinates %d does not match spatial dimension %d", dof, dim);
          for (d = 0; d < dof; ++d) {qpoints[d] = PetscRealPart(coords[off+d]);}
          qweights[0] = 1.0;
          ++f;
          ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
          lag->numDof[0] = 1;
        } else if ((p >= pStart[1]) && (p < pEnd[1])) {
          /* Edges */
          PetscScalar *coords;
          PetscInt     num = ((dim == 1) && !order) ? 1 : order-1, k;

          if (num < 1 || disc) continue;
          coords = NULL;
          ierr = DMPlexVecGetClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
          if (n != dim*2) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d has %d coordinate values instead of %d", p, n, dim*2);
          for (k = 1; k <= num; ++k) {
            ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
            ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
            ierr = PetscMalloc1(1, &qweights);CHKERRQ(ierr);
            ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
            ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, qpoints, qweights);CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {qpoints[d] = k*PetscRealPart(coords[1*dim+d] - coords[0*dim+d])/order + PetscRealPart(coords[0*dim+d]);}
            qweights[0] = 1.0;
            ++f;
          }
          ierr = DMPlexVecRestoreClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
          lag->numDof[1] = num;
        } else if ((p >= pStart[depth-1]) && (p < pEnd[depth-1])) {
          /* Faces */

          if (disc) continue;
          if ( simplex && (order < 3)) continue;
          if (!simplex && (order < 2)) continue;
          lag->numDof[depth-1] = 0;
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to implement faces");
        } else if ((p >= pStart[depth]) && (p < pEnd[depth])) {
          /* Cells */
          PetscInt     orderEff = lag->continuous && order ? (simplex ? order-3 : order-2) : order;
          PetscReal    denom    = order ? (lag->continuous ? order : (simplex ? order+3 : order+2)) : (simplex ? 3 : 2);
          PetscScalar *coords   = NULL;
          PetscReal    dx = 2.0/denom, *v0, *J, *invJ, detJ;
          PetscInt    *ind, *tup;
          PetscInt     cdim, csize, d, d2, o;

          lag->numDof[depth] = 0;
          if (orderEff < 0) continue;
          ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, orderEff, &cdim);CHKERRQ(ierr);
          ierr = DMPlexVecGetClosure(dm, csection, coordinates, p, &csize, &coords);CHKERRQ(ierr);
          if (csize%dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate size %d is not divisible by spatial dimension %d", csize, dim);

          ierr = PetscCalloc5(dim,&ind,dim,&tup,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
          ierr = DMPlexComputeCellGeometryFEM(dm, p, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
          if (simplex || disc) {
            for (o = 0; o <= orderEff; ++o) {
              ierr = PetscMemzero(ind, dim*sizeof(PetscInt));CHKERRQ(ierr);
              while (ind[0] >= 0) {
                ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
                ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
                ierr = PetscMalloc1(1,   &qweights);CHKERRQ(ierr);
                ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
                ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, qpoints, qweights);CHKERRQ(ierr);
                ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
                for (d = 0; d < dim; ++d) {
                  qpoints[d] = v0[d];
                  for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d2]+1)*dx);
                }
                qweights[0] = 1.0;
                ++f;
              }
            }
          } else {
            while (ind[0] >= 0) {
              ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
              ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
              ierr = PetscMalloc1(1,   &qweights);CHKERRQ(ierr);
              ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
              ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, qpoints, qweights);CHKERRQ(ierr);
              ierr = TensorPoint_Internal(dim, orderEff+1, ind, tup);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                qpoints[d] = v0[d];
                for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d]+1)*dx);
              }
              qweights[0] = 1.0;
              ++f;
            }
          }
          ierr = PetscFree5(ind,tup,v0,J,invJ);CHKERRQ(ierr);
          ierr = DMPlexVecRestoreClosure(dm, csection, coordinates, p, &csize, &coords);CHKERRQ(ierr);
          lag->numDof[depth] = cdim;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, pStart[depth], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscBTDestroy(&seen);CHKERRQ(ierr);
  }
  if (pEnd[dim] == 1 && f != pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %d not equal to dimension %d", f, pdimMax);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  if (f > pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %d is greater than dimension %d", f, pdimMax);
  lag->height    = 0;
  lag->subspaces = NULL;
  if (lag->continuous && sp->order > 0 && dim > 0) {
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy_Lagrange"
PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  for (i = 0; i < lag->height; i++) {
    ierr = PetscDualSpaceDestroy(&lag->subspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(lag->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(lag->numDof);CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDuplicate_Lagrange"
PetscErrorCode PetscDualSpaceDuplicate_Lagrange(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       order;
  PetscBool      cont;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(*spNew, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(*spNew, cont);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetFromOptions_Lagrange"
PetscErrorCode PetscDualSpaceSetFromOptions_Lagrange(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscBool      continuous, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &continuous);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscDualSpace Lagrange Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscdualspace_lagrange_continuity", "Flag for continuous element", "PetscDualSpaceLagrangeSetContinuity", continuous, &continuous, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetContinuity(sp, continuous);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension_Lagrange"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetNumDof_Lagrange"
PetscErrorCode PetscDualSpaceGetNumDof_Lagrange(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  *numDof = lag->numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceLagrangeGetContinuity_Lagrange"
static PetscErrorCode PetscDualSpaceLagrangeGetContinuity_Lagrange(PetscDualSpace sp, PetscBool *continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(continuous, 2);
  *continuous = lag->continuous;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceLagrangeSetContinuity_Lagrange"
static PetscErrorCode PetscDualSpaceLagrangeSetContinuity_Lagrange(PetscDualSpace sp, PetscBool continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  lag->continuous = continuous;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceLagrangeGetContinuity"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceLagrangeSetContinuity"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetHeightSubspace_Lagrange"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceInitialize_Lagrange"
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
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACELAGRANGE = "lagrange" - A PetscDualSpace object that encapsulates a dual space of pointwise evaluation functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreate_Lagrange"
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&lag);CHKERRQ(ierr);
  sp->data = lag;

  lag->numDof     = NULL;
  lag->simplex    = PETSC_TRUE;
  lag->continuous = PETSC_TRUE;

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Lagrange);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp_Simple"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy_Simple"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDuplicate_Simple"
PetscErrorCode PetscDualSpaceDuplicate_Simple(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACESIMPLE);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceSimpleSetDimension(*spNew, dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    PetscQuadrature q;

    ierr = PetscDualSpaceGetFunctional(sp, d, &q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSimpleSetFunctional(*spNew, d, q);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetFromOptions_Simple"
PetscErrorCode PetscDualSpaceSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension_Simple"
PetscErrorCode PetscDualSpaceGetDimension_Simple(PetscDualSpace sp, PetscInt *dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *dim = s->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSimpleSetDimension_Simple"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetNumDof_Simple"
PetscErrorCode PetscDualSpaceGetNumDof_Simple(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *numDof = s->numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSimpleSetFunctional_Simple"
PetscErrorCode PetscDualSpaceSimpleSetFunctional_Simple(PetscDualSpace sp, PetscInt f, PetscQuadrature q)
{
  PetscDualSpace_Simple *s   = (PetscDualSpace_Simple *) sp->data;
  PetscReal              vol = 0.0;
  PetscReal             *weights;
  PetscInt               Nq, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if ((f < 0) || (f >= s->dim)) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Basis index %d not in [0, %d)", f, s->dim);
  ierr = PetscQuadratureDuplicate(q, &sp->functional[f]);CHKERRQ(ierr);
  /* Reweight so that it has unit volume */
  ierr = PetscQuadratureGetData(sp->functional[f], NULL, &Nq, NULL, (const PetscReal **) &weights);CHKERRQ(ierr);
  for (p = 0; p < Nq; ++p) vol += weights[p];
  for (p = 0; p < Nq; ++p) weights[p] /= vol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSimpleSetDimension"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSimpleSetFunctional"
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

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceInitialize_Simple"
PetscErrorCode PetscDualSpaceInitialize_Simple(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscDualSpaceSetFromOptions_Simple;
  sp->ops->setup          = PetscDualSpaceSetUp_Simple;
  sp->ops->view           = NULL;
  sp->ops->destroy        = PetscDualSpaceDestroy_Simple;
  sp->ops->duplicate      = PetscDualSpaceDuplicate_Simple;
  sp->ops->getdimension   = PetscDualSpaceGetDimension_Simple;
  sp->ops->getnumdof      = PetscDualSpaceGetNumDof_Simple;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACESIMPLE = "simple" - A PetscDualSpace object that encapsulates a dual space of arbitrary functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreate_Simple"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFERegister"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEView"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetFromOptions"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetUp"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy"
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

  ierr = PetscFree((*fem)->numDof);CHKERRQ(ierr);
  ierr = PetscFree((*fem)->invV);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->B, &(*fem)->D, NULL /*&(*fem)->H*/);CHKERRQ(ierr);
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->F, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&(*fem)->basisSpace);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&(*fem)->dualSpace);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->quadrature);CHKERRQ(ierr);

  if ((*fem)->ops->destroy) {ierr = (*(*fem)->ops->destroy)(*fem);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate"
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
  f->numDof        = NULL;
  f->invV          = NULL;
  f->B             = NULL;
  f->D             = NULL;
  f->H             = NULL;
  ierr = PetscMemzero(&f->quadrature, sizeof(PetscQuadrature));CHKERRQ(ierr);
  f->blockSize     = 0;
  f->numBlocks     = 1;
  f->batchSize     = 0;
  f->numBatches    = 1;

  *fem = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetSpatialDimension"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetNumComponents"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumComponents"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetTileSizes"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTileSizes"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetBasisSpace"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetBasisSpace"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDualSpace"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetDualSpace"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetQuadrature"
/*@
  PetscFEGetQuadrature - Returns the PetscQuadreture used to calculate inner products

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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetQuadrature"
/*@
  PetscFESetQuadrature - Sets the PetscQuadreture used to calculate inner products

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- q - The PetscQuadrature object

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscFERestoreTabulation(fem, 0, NULL, &fem->B, &fem->D, NULL /*&(*fem)->H*/);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fem->quadrature);CHKERRQ(ierr);
  fem->quadrature = q;
  ierr = PetscObjectReference((PetscObject) q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumDof"
PetscErrorCode PetscFEGetNumDof(PetscFE fem, const PetscInt **numDof)
{
  const PetscInt *numDofDual;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  ierr = PetscDualSpaceGetNumDof(fem->dualSpace, &numDofDual);CHKERRQ(ierr);
  if (!fem->numDof) {
    DM       dm;
    PetscInt dim, d;

    ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscMalloc1(dim+1, &fem->numDof);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) {
      fem->numDof[d] = fem->numComponents*numDofDual[d];
    }
  }
  *numDof = fem->numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDefaultTabulation"
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
  ierr = PetscQuadratureGetData(fem->quadrature, NULL, &npoints, &points, NULL);CHKERRQ(ierr);
  if (!fem->B) {ierr = PetscFEGetTabulation(fem, npoints, points, &fem->B, &fem->D, NULL/*&fem->H*/);CHKERRQ(ierr);}
  if (B) *B = fem->B;
  if (D) *D = fem->D;
  if (H) *H = fem->H;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetFaceTabulation"
PetscErrorCode PetscFEGetFaceTabulation(PetscFE fem, PetscReal **F)
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTabulation"
PetscErrorCode PetscFEGetTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         comp; /* Field components */
  PetscErrorCode   ierr;

  PetscFunctionBegin;
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

#undef __FUNCT__
#define __FUNCT__ "PetscFERestoreTabulation"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy_Basic"
PetscErrorCode PetscFEDestroy_Basic(PetscFE fem)
{
  PetscFE_Basic *b = (PetscFE_Basic *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEView_Basic_Ascii"
PetscErrorCode PetscFEView_Basic_Ascii(PetscFE fe, PetscViewer viewer)
{
  PetscSpace        basis;
  PetscDualSpace    dual;
  PetscQuadrature   q = NULL;
  PetscInt          dim, Nq;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetBasisSpace(fe, &basis);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &dual);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, &dim, &Nq, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Basic Finite Element:\n");CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "  dimension:       %d\n", dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  num quad points: %d\n", Nq);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureView(q, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "  dimension:       %d\n", dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  num quad points: %d\n", Nq);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscSpaceView(basis, viewer);CHKERRQ(ierr);
  ierr = PetscDualSpaceView(dual, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEView_Basic"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetUp_Basic"
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
    PetscReal      *Bf;
    PetscQuadrature f;
    PetscInt        q, k;

    ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);CHKERRQ(ierr);
    ierr = PetscMalloc1(f->numPoints*pdim,&Bf);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(fem->basisSpace, f->numPoints, f->points, Bf, NULL, NULL);CHKERRQ(ierr);
    for (k = 0; k < pdim; ++k) {
      /* n_j \cdot \phi_k */
      invVscalar[j*pdim+k] = 0.0;
      for (q = 0; q < f->numPoints; ++q) {
        invVscalar[j*pdim+k] += Bf[q*pdim+k]*f->weights[q];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDimension_Basic"
PetscErrorCode PetscFEGetDimension_Basic(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTabulation_Basic"
PetscErrorCode PetscFEGetTabulation_Basic(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal *B, PetscReal *D, PetscReal *H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         comp; /* Field components */
  PetscReal       *tmpB, *tmpD, *tmpH;
  PetscInt         p, d, j, k;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (B) {ierr = DMGetWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? tmpH : NULL);CHKERRQ(ierr);
  /* Translate to the nodal basis */
  for (p = 0; p < npoints; ++p) {
    if (B) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        const PetscInt i = (p*pdim + j)*comp;
        PetscInt       c;

        B[i] = 0.0;
        for (k = 0; k < pdim; ++k) {
          B[i] += fem->invV[k*pdim+j] * tmpB[p*pdim + k];
        }
        for (c = 1; c < comp; ++c) {
          B[i+c] = B[i];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + j)*comp + 0)*dim + d;
          PetscInt       c;

          D[i] = 0.0;
          for (k = 0; k < pdim; ++k) {
            D[i] += fem->invV[k*pdim+j] * tmpD[(p*pdim + k)*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            D[((p*pdim + j)*comp + c)*dim + d] = D[i];
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (d = 0; d < dim*dim; ++d) {
          const PetscInt i = ((p*pdim + j)*comp + 0)*dim*dim + d;
          PetscInt       c;

          H[i] = 0.0;
          for (k = 0; k < pdim; ++k) {
            H[i] += fem->invV[k*pdim+j] * tmpH[(p*pdim + k)*dim*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            H[((p*pdim + j)*comp + c)*dim*dim + d] = H[i];
          }
        }
      }
    }
  }
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*pdim*dim*dim, PETSC_REAL, &tmpH);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrate_Basic"
PetscErrorCode PetscFEIntegrate_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                      const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal integral[])
{
  const PetscInt  debug = 0;
  PetscPointFunc  obj_func;
  PetscQuadrature quad;
  PetscScalar    *u, *u_x, *a, *a_x;
  PetscReal      *x;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        dim, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetObjective(prob, field, &obj_func);CHKERRQ(ierr);
  if (!obj_func) PetscFunctionReturn(0);
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, NULL);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = geom[e].v0;
    const PetscReal *J    = geom[e].J;
    const PetscReal *invJ = geom[e].invJ;
    const PetscReal  detJ = geom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
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
      ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, &coefficients[cOffset],       NULL, u, u_x, NULL);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, &integrand);
      integrand *= detJ*quadWeights[q];
      integral[field] += PetscRealPart(integrand);
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", PetscRealPart(integrand), integral[field]);CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateResidual_Basic"
PetscErrorCode PetscFEIntegrateResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscPointFunc  f0_func;
  PetscPointFunc  f1_func;
  PetscQuadrature quad;
  PetscReal     **basisField, **basisFieldDer;
  PetscScalar    *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        dim, Nf, NfAux = 0, Nb, Nc, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = geom[e].v0;
    const PetscReal *J    = geom[e].J;
    const PetscReal *invJ = geom[e].invJ;
    const PetscReal  detJ = geom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    ierr = PetscMemzero(f0, Nq*Nc* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*Nc*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, &f0[q*Nc]);
      if (f1_func) f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, refSpaceDer);
      TransformF(dim, dim, Nc, q, invJ, detJ, quadWeights, refSpaceDer, f0, f1);
    }
    UpdateElementVec(dim, Nq, Nb, Nc, basisField[field], basisFieldDer[field], f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdResidual_Basic"
PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscBdPointFunc f0_func;
  PetscBdPointFunc f1_func;
  PetscQuadrature quad;
  PetscReal     **basisField, **basisFieldDer;
  PetscScalar    *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        dim, Nf, NfAux = 0, bdim, Nb, Nc, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  dim += 1; /* Spatial dimension is one higher than topological dimension */
  bdim = dim-1;
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalBdDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetBdFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetBdTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalBdDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0          = geom[e].v0;
    const PetscReal *n           = geom[e].n;
    const PetscReal *J           = geom[e].J;
    const PetscReal *invJ        = geom[e].invJ;
    const PetscReal  detJ        = geom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    ierr = PetscMemzero(f0, Nq*Nc* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*Nc*bdim * sizeof(PetscScalar));CHKERRQ(ierr);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, bdim, v0, J, &quadPoints[q*bdim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_TRUE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_TRUE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, n, &f0[q*Nc]);
      if (f1_func) f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, n, refSpaceDer);
      TransformF(dim, bdim, Nc, q, invJ, detJ, quadWeights, refSpaceDer, f0, f1);
    }
    UpdateElementVec(bdim, Nq, Nb, Nc, basisField[field], basisFieldDer[field], f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobian_Basic"
PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemMat[])
{
  const PetscInt  debug      = 0;
  PetscPointJac   g0_func;
  PetscPointJac   g1_func;
  PetscPointJac   g2_func;
  PetscPointJac   g3_func;
  PetscFE         fe;
  PetscInt        cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt        cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt        eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt        offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt        offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature quad;
  PetscScalar    *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscReal     **basisField, **basisFieldDer, *basisI, *basisDerI, *basisJ, *basisDerJ;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt        dim, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
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
  ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbI);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, fieldJ, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbJ);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcJ);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  basisI    = basisField[fieldI];
  basisJ    = basisField[fieldJ];
  basisDerI = basisFieldDer[fieldI];
  basisDerJ = basisFieldDer[fieldJ];
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
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscInt f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, g0);
        for (c = 0; c < NcI*NcJ; ++c) {g0[c] *= detJ*quadWeights[q];}
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
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
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= detJ*quadWeights[q];
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+fidx; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+gidx; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g0[fc*NcJ+gc]*basisJ[q*NbJ*NcJ+gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g1[(fc*NcJ+gc)*dim+d]*basisDerJ[(q*NbJ*NcJ+gidx)*dim+d];
                elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*dim+d]*g2[(fc*NcJ+gc)*dim+d]*basisJ[q*NbJ*NcJ+gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*basisDerJ[(q*NbJ*NcJ+gidx)*dim+d2];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdJacobian_Basic"
PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscFE fem, PetscDS prob, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemMat[])
{
  const PetscInt  debug      = 0;
  PetscBdPointJac g0_func;
  PetscBdPointJac g1_func;
  PetscBdPointJac g2_func;
  PetscBdPointJac g3_func;
  PetscFE         fe;
  PetscInt        cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt        cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt        eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt        offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt        offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature quad;
  PetscScalar    *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscReal     **basisField, **basisFieldDer, *basisI, *basisDerI, *basisJ, *basisDerJ;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt        dim, Nf, NfAux = 0, bdim, totDim, totDimAux = 0, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  dim += 1; /* Spatial dimension is one higher than topological dimension */
  bdim = dim-1;
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalBdDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetBdJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetBdTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  ierr = PetscDSGetBdDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbI);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcI);CHKERRQ(ierr);
  ierr = PetscDSGetBdFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetBdDiscretization(prob, fieldJ, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbJ);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcJ);CHKERRQ(ierr);
  ierr = PetscDSGetBdFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalBdDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  basisI    = basisField[fieldI];
  basisJ    = basisField[fieldJ];
  basisDerI = basisFieldDer[fieldI];
  basisDerJ = basisFieldDer[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*bdim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*bdim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*bdim*bdim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = geom[e].v0;
    const PetscReal *n    = geom[e].n;
    const PetscReal *J    = geom[e].J;
    const PetscReal *invJ = geom[e].invJ;
    const PetscReal  detJ = geom[e].detJ;
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscInt f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, bdim, v0, J, &quadPoints[q*bdim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_TRUE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_TRUE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, n, g0);
        for (c = 0; c < NcI*NcJ; ++c) {g0[c] *= detJ*quadWeights[q];}
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, n, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < bdim; ++d) {
              g1[(fc*NcJ+gc)*bdim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*bdim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*bdim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, n, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < bdim; ++d) {
              g2[(fc*NcJ+gc)*bdim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*bdim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*bdim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, n, g3);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < bdim; ++d) {
              for (dp = 0; dp < bdim; ++dp) {
                g3[((fc*NcJ+gc)*bdim+d)*bdim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*bdim+d)*bdim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*bdim+d)*bdim+dp] *= detJ*quadWeights[q];
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+fidx; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+gidx; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g0[fc*NcJ+gc]*basisJ[q*NbJ*NcJ+gidx];
              for (d = 0; d < bdim; ++d) {
                elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g1[(fc*NcJ+gc)*bdim+d]*basisDerJ[(q*NbJ*NcJ+gidx)*bdim+d];
                elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*bdim+d]*g2[(fc*NcJ+gc)*bdim+d]*basisJ[q*NbJ*NcJ+gidx];
                for (d2 = 0; d2 < bdim; ++d2) {
                  elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*bdim+d]*g3[((fc*NcJ+gc)*bdim+d)*bdim+d2]*basisDerJ[(q*NbJ*NcJ+gidx)*bdim+d2];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitialize_Basic"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate_Basic"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy_Nonaffine"
PetscErrorCode PetscFEDestroy_Nonaffine(PetscFE fem)
{
  PetscFE_Nonaffine *na = (PetscFE_Nonaffine *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(na);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateResidual_Nonaffine"
PetscErrorCode PetscFEIntegrateResidual_Nonaffine(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                                  const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscPointFunc  f0_func;
  PetscPointFunc  f1_func;
  PetscQuadrature quad;
  PetscReal     **basisField, **basisFieldDer;
  PetscScalar    *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        dim, Nf, NfAux = 0, Nb, Nc, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    ierr = PetscMemzero(f0, Nq*Nc* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*Nc*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v0   = geom[e*Nq+q].v0;
      const PetscReal *J    = geom[e*Nq+q].J;
      const PetscReal *invJ = geom[e*Nq+q].invJ;
      const PetscReal  detJ = geom[e*Nq+q].detJ;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, &f0[q*Nc]);
      if (f1_func) f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, refSpaceDer);
      TransformF(dim, dim, Nc, q, invJ, detJ, quadWeights, refSpaceDer, f0, f1);
    }
    UpdateElementVec(dim, Nq, Nb, Nc, basisField[field], basisFieldDer[field], f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdResidual_Nonaffine"
PetscErrorCode PetscFEIntegrateBdResidual_Nonaffine(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                                    const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  const PetscInt   debug = 0;
  PetscBdPointFunc f0_func;
  PetscBdPointFunc f1_func;
  PetscQuadrature  quad;
  PetscReal      **basisField, **basisFieldDer;
  PetscScalar     *f0, *f1, *u, *u_t, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal       *x;
  PetscInt        *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt         dim, Nf, NfAux = 0, bdim, Nb, Nc, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  dim += 1; /* Spatial dimension is one higher than topological dimension */
  bdim = dim-1;
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalBdDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentBdDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetBdFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, &u_t, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetBdTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalBdDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentBdDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    ierr = PetscMemzero(f0, Nq*Nc* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*Nc*bdim * sizeof(PetscScalar));CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v0   = geom[e*Nq+q].v0;
      const PetscReal *n    = geom[e*Nq+q].n;
      const PetscReal *J    = geom[e*Nq+q].J;
      const PetscReal *invJ = geom[e*Nq+q].invJ;
      const PetscReal  detJ = geom[e*Nq+q].detJ;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, bdim, v0, J, &quadPoints[q*bdim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_TRUE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_TRUE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, n, &f0[q*Nc]);
      if (f1_func) f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, n, refSpaceDer);
      TransformF(dim, bdim, Nc, q, invJ, detJ, quadWeights, refSpaceDer, f0, f1);
    }
    UpdateElementVec(bdim, Nq, Nb, Nc, basisField[field], basisFieldDer[field], f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobian_Nonaffine"
PetscErrorCode PetscFEIntegrateJacobian_Nonaffine(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                                  const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemMat[])
{
  const PetscInt  debug      = 0;
  PetscPointJac   g0_func;
  PetscPointJac   g1_func;
  PetscPointJac   g2_func;
  PetscPointJac   g3_func;
  PetscFE         fe;
  PetscInt        cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt        cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt        eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt        offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt        offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature quad;
  PetscScalar    *g0, *g1, *g2, *g3, *u, *u_t, *u_x, *a, *a_x, *refSpaceDer;
  PetscReal      *x;
  PetscReal     **basisField, **basisFieldDer, *basisI, *basisDerI, *basisJ, *basisDerJ;
  PetscInt        NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        dim, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscDSGetDynamicJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscDSGetJacobianPreconditioner(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscDSGetJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  }
  ierr = PetscDSGetEvaluationArrays(prob, &u, &u_t, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbI);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, fieldJ, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &NbJ);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &NcJ);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  basisI    = basisField[fieldI];
  basisJ    = basisField[fieldJ];
  basisDerI = basisFieldDer[fieldI];
  basisDerJ = basisFieldDer[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal *quadPoints, *quadWeights;
    PetscInt         Nq, q;

    ierr = PetscQuadratureGetData(quad, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v0   = geom[e*Nq+q].v0;
      const PetscReal *J    = geom[e*Nq+q].J;
      const PetscReal *invJ = geom[e*Nq+q].invJ;
      const PetscReal  detJ = geom[e*Nq+q].detJ;
      PetscInt         f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], x);
      ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, &coefficients[cOffset],       &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, &coefficientsAux[cOffsetAux], NULL,                     a, a_x, NULL);CHKERRQ(ierr);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, g0);
        for (c = 0; c < NcI*NcJ; ++c) {g0[c] *= detJ*quadWeights[q];}
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= detJ*quadWeights[q];
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, 0.0, x, refSpaceDer);
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
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= detJ*quadWeights[q];
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+fidx; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+gidx; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g0[fc*NcJ+gc]*basisJ[q*NbJ*NcJ+gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += basisI[q*NbI*NcI+fidx]*g1[(fc*NcJ+gc)*dim+d]*basisDerJ[(q*NbJ*NcJ+gidx)*dim+d];
                elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*dim+d]*g2[(fc*NcJ+gc)*dim+d]*basisJ[q*NbJ*NcJ+gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += basisDerI[(q*NbI*NcI+fidx)*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*basisDerJ[(q*NbJ*NcJ+gidx)*dim+d2];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitialize_Nonaffine"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate_Nonaffine"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy_OpenCL"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLGenerateIntegrationCode"
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
  PetscReal      *basis, *basisDer;
  PetscInt        dim, N_b, N_c, N_q, N_t, p, d, b, c;
  size_t          count;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &N_b);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &N_c);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &q);CHKERRQ(ierr);
  N_q  = q->numPoints;
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
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, q->points[p*dim+d]);STRING_ERROR_CHECK("Message to short");
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Quadrature weights\n"
"   - (v1,v2,...) */\n"
"  const %s weights[%d] = {\n",
                       &count, numeric_str, N_q);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, q->weights[p]);STRING_ERROR_CHECK("Message to short");
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLGetIntegrationKernel"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLCalculateGrid"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLLogResidual"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateResidual_OpenCL"
PetscErrorCode PetscFEIntegrateResidual_OpenCL(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                               const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  /* Nbc = batchSize */
  PetscFE_OpenCL   *ocl = (PetscFE_OpenCL *) fem->data;
  PetscPointFunc    f0_func;
  PetscPointFunc    f1_func;
  PetscQuadrature   q;
  PetscInt          dim;
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!Ne) {ierr = PetscFEOpenCLLogResidual(fem, 0.0, 0.0);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &q);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &N_b);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &N_comp);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscFEGetTileSizes(fem, NULL, &N_bl, &N_bc, &N_cb);CHKERRQ(ierr);
  N_bt  = N_b*N_comp;
  N_q   = q->numPoints;
  N_bst = N_bt*N_q;
  N_t   = N_bst*N_bl;
  if (N_bc*N_comp != N_t) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, N_bc, N_comp);
  /* Calculate layout */
  if (Ne % (N_cb*N_bc)) { /* Remainder cells */
    ierr = PetscFEIntegrateResidual_Basic(fem, prob, field, Ne, geom, coefficients, coefficients_t, probAux, coefficientsAux, elemVec);CHKERRQ(ierr);
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
        f_detJ[c] = (float) geom[c].detJ;
        for (d = 0; d < dim*dim; ++d) {
          f_invJ[c*dim*dim+d] = (float) geom[c].invJ[d];
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
        d_detJ[c] = (double) geom[c].detJ;
        for (d = 0; d < dim*dim; ++d) {
          d_invJ[c*dim*dim+d] = (double) geom[c].invJ[d];
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
      r_detJ[c] = geom[c].detJ;
      for (d = 0; d < dim*dim; ++d) {
        r_invJ[c*dim*dim+d] = geom[c].invJ[d];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitialize_OpenCL"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate_OpenCL"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLSetRealType"
PetscErrorCode PetscFEOpenCLSetRealType(PetscFE fem, PetscDataType realType)
{
  PetscFE_OpenCL *ocl = (PetscFE_OpenCL *) fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ocl->realType = realType;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEOpenCLGetRealType"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy_Composite"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFESetUp_Composite"
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
      PetscReal      *Bf;
      PetscQuadrature f;
      PetscInt        q, k;

      ierr = PetscDualSpaceGetFunctional(fem->dualSpace, cmp->embedding[s*spdim+j], &f);CHKERRQ(ierr);
      ierr = PetscMalloc1(f->numPoints*spdim,&Bf);CHKERRQ(ierr);
      ierr = PetscSpaceEvaluate(fem->basisSpace, f->numPoints, f->points, Bf, NULL, NULL);CHKERRQ(ierr);
      for (k = 0; k < spdim; ++k) {
        /* n_j \cdot \phi_k */
        invVscalar[(s*spdim + j)*spdim+k] = 0.0;
        for (q = 0; q < f->numPoints; ++q) {
          invVscalar[(s*spdim + j)*spdim+k] += Bf[q*spdim+k]*f->weights[q];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTabulation_Composite"
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
        PetscInt       c;

        B[i] = 0.0;
        for (k = 0; k < spdim; ++k) {
          B[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpB[p*spdim + k];
        }
        for (c = 1; c < comp; ++c) {
          B[i+c] = B[i];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (spdim x spdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim + d;
          PetscInt       c;

          D[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            D[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpD[(p*spdim + k)*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            D[((p*pdim + cmp->embedding[s*spdim+j])*comp + c)*dim + d] = D[i];
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim*dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim*dim + d;
          PetscInt       c;

          H[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            H[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpH[(p*spdim + k)*dim*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            H[((p*pdim + cmp->embedding[s*spdim+j])*comp + c)*dim*dim + d] = H[i];
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitialize_Composite"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate_Composite"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFECompositeGetMapping"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDimension"
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

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrate"
/*@C
  PetscFEIntegrate - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter
. integral     - the integral for this field

  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrate(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal integral[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 2);
  if (fem->ops->integrate) {ierr = (*fem->ops->integrate)(fem, prob, field, Ne, geom, coefficients, probAux, coefficientsAux, integral);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateResidual"
/*@C
  PetscFEIntegrateResidual - Produce the element residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

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
PetscErrorCode PetscFEIntegrateResidual(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 2);
  if (fem->ops->integrateresidual) {ierr = (*fem->ops->integrateresidual)(fem, prob, field, Ne, geom, coefficients, coefficients_t, probAux, coefficientsAux, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdResidual"
/*@C
  PetscFEIntegrateBdResidual - Produce the element residual vector for a chunk of elements by quadrature integration over a boundary

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter
. elemVec      - the element residual vectors from each element

  Level: developer

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBdResidual(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFECellGeom *geom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratebdresidual) {ierr = (*fem->ops->integratebdresidual)(fem, prob, field, Ne, geom, coefficients, coefficients_t, probAux, coefficientsAux, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobian"
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
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

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
*/
PetscErrorCode PetscFEIntegrateJacobian(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemMat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratejacobian) {ierr = (*fem->ops->integratejacobian)(fem, prob, jtype, fieldI, fieldJ, Ne, geom, coefficients, coefficients_t, probAux, coefficientsAux, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdJacobian"
/*C
  PetscFEIntegrateBdJacobian - Produce the boundary element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          = The PetscFE object for the field being integrated
. prob         - The PetscDS specifying the discretizations and continuum functions
. fieldI       - The test field being integrated
. fieldJ       - The basis field being integrated
. Ne           - The number of elements in the chunk
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

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
*/
PetscErrorCode PetscFEIntegrateBdJacobian(PetscFE fem, PetscDS prob, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFECellGeom *geom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar elemMat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratebdjacobian) {ierr = (*fem->ops->integratebdjacobian)(fem, prob, fieldI, fieldJ, Ne, geom, coefficients, coefficients_t, probAux, coefficientsAux, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFERefine"
/*@
  PetscFERefine - Create a "refined" PetscFE object that refines the reference cell into smaller copies. This is typically used
  to precondition a higher order method with a lower order method on a refined mesh having the same number of dofs (but more
  sparsity). It is also used to create an interpolation between regularly refined meshes.

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

#undef __FUNCT__
#define __FUNCT__ "PetscFECreateDefault"
/*@
  PetscFECreateDefault - Create a PetscFE for basic FEM computation

  Collective on DM

  Input Parameters:
+ dm         - The underlying DM for the domain
. dim        - The spatial dimension
. numComp    - The number of components
. isSimplex  - Flag for simplex reference cell, otherwise its a tensor product
. prefix     - The options prefix, or NULL
- qorder     - The quadrature order

  Output Parameter:
. fem - The PetscFE object

  Level: beginner

.keywords: PetscFE, finite element
.seealso: PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscFECreateDefault(DM dm, PetscInt dim, PetscInt numComp, PetscBool isSimplex, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  PetscQuadrature q;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), fem);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(*fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, numComp);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  if (isSimplex) {ierr = PetscDTGaussJacobiQuadrature(dim, PetscMax(qorder > 0 ? qorder : order, 1), -1.0, 1.0, &q);CHKERRQ(ierr);}
  else           {ierr = PetscDTGaussTensorQuadrature(dim, PetscMax(qorder > 0 ? qorder : order, 1), -1.0, 1.0, &q);CHKERRQ(ierr);}
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
