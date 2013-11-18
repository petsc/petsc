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
#include <petsc-private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmshell.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>

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

  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}
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
. dm  - The PetscSpace

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
  PetscValidCharPointer(name, 2);
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
#define __FUNCT__ "PetscSpaceViewFromOptions"
/*
  PetscSpaceViewFromOptions - Processes command line options to determine if/how a PetscSpace is to be viewed.

  Collective on PetscSpace

  Input Parameters:
+ sp   - the PetscSpace
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscSpace, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscSpaceViewFromOptions(PetscSpace sp, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), ((PetscObject) sp)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscSpaceView(sp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
    ierr = (*sp->ops->setfromoptions)(sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) sp);CHKERRQ(ierr);
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
  *sp = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(s, _p_PetscSpace, struct _PetscSpaceOps, PETSCSPACE_CLASSID, "PetscSpace", "Linear Space", "PetscSpace", comm, PetscSpaceDestroy, PetscSpaceView);CHKERRQ(ierr);
  ierr = PetscMemzero(s->ops, sizeof(struct _PetscSpaceOps));CHKERRQ(ierr);

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
PetscErrorCode PetscSpaceSetOrder(PetscSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate"
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
PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_poly_num_variables", "The number of different variables, e.g. x and y", "PetscSpacePolynomialSetNumVariables", poly->numVariables, &poly->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_sym", "Use only symmetric polynomials", "PetscSpacePolynomialSetSymmetric", poly->symmetric, &poly->symmetric, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_tensor", "Use the tensor product polynomials", "PetscSpacePolynomialSetTensor", poly->tensor, &poly->tensor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialView_Ascii"
PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Poly  *poly = (PetscSpace_Poly *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d", poly->numVariables, sp->order);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d", poly->numVariables, sp->order);CHKERRQ(ierr);
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
  ierr = PetscMalloc(ndegree * sizeof(PetscInt), &poly->degrees);CHKERRQ(ierr);
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
      if (ind[0] == max-1) {ind[0] = -1;}
      else                 {ind[1] = 0; ++ind[0];}
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
  ierr = PetscMalloc2(dim,PetscInt,&ind,dim,PetscInt,&tup);CHKERRQ(ierr);
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
  ierr     = PetscNewLog(sp, PetscSpace_Poly, &poly);CHKERRQ(ierr);
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
PetscErrorCode PetscSpaceSetFromOptions_DG(PetscSpace sp)
{
  PetscSpace_DG *dg = (PetscSpace_DG *) sp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_dg_num_variables", "The number of different variables, e.g. x and y", "PetscSpaceDGSetNumVariables", dg->numVariables, &dg->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer, "DG space in dimension %d on %d points\n", dg->numVariables, dg->quad.numPoints);CHKERRQ(ierr);
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
  if (!dg->quad.points && sp->order) {
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
  *dim = dg->quad.numPoints;
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
  if (npoints != dg->quad.numPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot evaluate DG space on %d points != %d size", npoints, dg->quad.numPoints);
  ierr = PetscMemzero(B, npoints*npoints * sizeof(PetscReal));CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (d = 0; d < dim; ++d) {
      if (PetscAbsReal(points[p*dim+d] - dg->quad.points[p*dim+d]) > 1.0e-10) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot evaluate DG point (%d, %d) %g != %g", p, d, points[p*dim+d], dg->quad.points[p*dim+d]);
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
  ierr     = PetscNewLog(sp, PetscSpace_DG, &dg);CHKERRQ(ierr);
  sp->data = dg;

  dg->numVariables   = 0;
  dg->quad.dim       = 0;
  dg->quad.numPoints = 0;
  dg->quad.points    = NULL;
  dg->quad.weights   = NULL;

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
. dm  - The PetscDualSpace

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
  PetscValidCharPointer(name, 2);
  if (!PetscDualSpaceRegisterAllCalled) {
    ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceView"
/*@C
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
#define __FUNCT__ "PetscDualSpaceViewFromOptions"
/*
  PetscDualSpaceViewFromOptions - Processes command line options to determine if/how a PetscDualSpace is to be viewed.

  Collective on PetscDualSpace

  Input Parameters:
+ sp   - the PetscDualSpace
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscDualSpace, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscDualSpaceViewFromOptions(PetscDualSpace sp, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), ((PetscObject) sp)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscDualSpaceView(sp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
    ierr = (*sp->ops->setfromoptions)(sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp"
/*@C
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
  *sp = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(s, _p_PetscDualSpace, struct _PetscDualSpaceOps, PETSCDUALSPACE_CLASSID, "PetscDualSpace", "Dual Space", "PetscDualSpace", comm, PetscDualSpaceDestroy, PetscDualSpaceView);CHKERRQ(ierr);
  ierr = PetscMemzero(s->ops, sizeof(struct _PetscDualSpaceOps));CHKERRQ(ierr);

  s->order = 0;

  *sp = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDM"
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
PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  sp->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetFunctional"
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
/* Dimension of the space, i.e. number of basis vectors */
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
PetscErrorCode PetscDualSpaceGetNumDof(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  *numDof = NULL;
  if (sp->ops->getnumdof) {ierr = (*sp->ops->getnumdof)(sp, numDof);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreateReferenceCell"
PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace sp, PetscInt dim, PetscBool simplex, DM *refdm)
{
  DM             rdm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(PetscObjectComm((PetscObject) sp), &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(rdm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 0:
  {
    PetscInt    numPoints[1]        = {1};
    PetscInt    coneSize[1]         = {0};
    PetscInt    cones[1]            = {0};
    PetscInt    coneOrientations[1] = {0};
    PetscScalar vertexCoords[1]     = {0.0};

    ierr = DMPlexCreateFromDAG(rdm, 0, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  case 1:
  {
    PetscInt    numPoints[2]        = {2, 1};
    PetscInt    coneSize[3]         = {2, 0, 0};
    PetscInt    cones[2]            = {1, 2};
    PetscInt    coneOrientations[2] = {0, 0};
    PetscScalar vertexCoords[2]     = {-1.0,  1.0};

    ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  case 2:
    if (simplex) {
      PetscInt    numPoints[2]        = {3, 1};
      PetscInt    coneSize[4]         = {3, 0, 0, 0};
      PetscInt    cones[3]            = {1, 2, 3};
      PetscInt    coneOrientations[3] = {0, 0, 0};
      PetscScalar vertexCoords[6]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    } else {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
  break;
  case 3:
    if (simplex) {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 3, 2, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  -1.0, -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    } else {
      PetscInt    numPoints[2]        = {8, 1};
      PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {1, 4, 3, 2, 5, 6, 7, 8};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, 1.0, -1.0,  -1.0, 1.0, -1.0,
                                         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0, 1.0,  1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
  break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_WRONG, "Cannot create reference cell for dimension %d", dim);
  }
  ierr = DMPlexInterpolate(rdm, refdm);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(rdm, *refdm);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceApply"
PetscErrorCode PetscDualSpaceApply(PetscDualSpace sp, PetscInt f, PetscCellGeometry geom, PetscInt numComp, void (*func)(const PetscReal [], PetscScalar *), PetscScalar *value)
{
  DM               dm;
  PetscQuadrature  quad;
  const PetscReal *v0 = geom.v0;
  const PetscReal *J  = geom.J;
  PetscReal        x[3];
  PetscScalar     *val;
  PetscInt         dim, q, c, d, d2;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(value, 5);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFunctional(sp, f, &quad);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  for (c = 0; c < numComp; ++c) value[c] = 0.0;
  for (q = 0; q < quad.numPoints; ++q) {
    for (d = 0; d < dim; ++d) {
      x[d] = v0[d];
      for (d2 = 0; d2 < dim; ++d2) {
        x[d] += J[d*dim+d2]*(quad.points[q*dim+d2] + 1.0);
      }
    }
    (*func)(x, val);
    for (c = 0; c < numComp; ++c) {
      value[c] += val[c]*quad.weights[q];
    }
  }
  ierr = DMRestoreWorkArray(dm, numComp, PETSC_SCALAR, &val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp_Lagrange"
PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscSection        csection;
  Vec                 coordinates;
  PetscReal          *qpoints, *qweights;
  PetscInt           *closure = NULL, closureSize, c;
  PetscInt            depth, dim, pdim, *pStart, *pEnd, coneSize, d, n, f = 0;
  PetscBool           simplex;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* Classify element type */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &lag->numDof);CHKERRQ(ierr);
  ierr = PetscMemzero(lag->numDof, (dim+1) * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,PetscInt,&pStart,depth+1,PetscInt,&pEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeSize(dm, pStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if      (coneSize == dim+1)    simplex = PETSC_TRUE;
  else if (coneSize == 1 << dim) simplex = PETSC_FALSE;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  lag->simplex = simplex;
  ierr = PetscDualSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc(pdim * sizeof(PetscQuadrature), &sp->functional);CHKERRQ(ierr);
  if (!dim) {
    sp->functional[f].numPoints = 1;
    ierr = PetscMalloc(sp->functional[f].numPoints * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc(sp->functional[f].numPoints * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
    qpoints[0]  = 0.0;
    qweights[0] = 1.0;
    sp->functional[f].points  = qpoints;
    sp->functional[f].weights = qweights;
    ++f;
    lag->numDof[0] = 1;
  } else {
    ierr = DMPlexGetTransitiveClosure(dm, pStart[depth], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      const PetscInt p = closure[c];

      if ((p >= pStart[0]) && (p < pEnd[0])) {
        /* Vertices */
        const PetscScalar *coords;
        PetscInt           dof, off, d;

        if (order < 1) continue;
        sp->functional[f].numPoints = 1;
        ierr = PetscMalloc(sp->functional[f].numPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
        ierr = PetscMalloc(sp->functional[f].numPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
        ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(csection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
        if (dof != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of coordinates %d does not match spatial dimension %d", dof, dim);
        for (d = 0; d < dof; ++d) {qpoints[d] = PetscRealPart(coords[off+d]);}
        qweights[0] = 1.0;
        sp->functional[f].points  = qpoints;
        sp->functional[f].weights = qweights;
        ++f;
        ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
        lag->numDof[0] = 1;
      } else if ((p >= pStart[1]) && (p < pEnd[1])) {
        /* Edges */
        PetscScalar *coords;
        PetscInt     num = order-1, k;

        if (order < 2) continue;
        coords = NULL;
        ierr = DMPlexVecGetClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
        if (n != dim*2) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d has %d coordinate values instead of %d", p, n, dim*2);
        for (k = 1; k <= num; ++k) {
          sp->functional[f].numPoints = 1;
          ierr = PetscMalloc(sp->functional[f].numPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
          ierr = PetscMalloc(sp->functional[f].numPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {qpoints[d] = k*PetscRealPart(coords[1*dim+d] - coords[0*dim+d])/(num+1) + PetscRealPart(coords[0*dim+d]);}
          qweights[0] = 1.0;
          sp->functional[f].points  = qpoints;
          sp->functional[f].weights = qweights;
          ++f;
        }
        ierr = DMPlexVecRestoreClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
        lag->numDof[1] = num;
      } else if ((p >= pStart[depth-1]) && (p < pEnd[depth-1])) {
        /* Faces */

        if ( simplex && (order < 3)) continue;
        if (!simplex && (order < 2)) continue;
        lag->numDof[depth-1] = 0;
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to implement faces");
      } else if ((p >= pStart[depth]) && (p < pEnd[depth])) {
        /* Cells */
        PetscScalar *coords = NULL;
        PetscInt     csize, v, d;

        if ( simplex && (order > 0) && (order < 3)) continue;
        if (!simplex && (order > 0) && (order < 2)) continue;
        lag->numDof[depth] = 0;
        if (order > 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to implement cells");}

        sp->functional[f].numPoints = 1;
        ierr = PetscMalloc(sp->functional[f].numPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
        ierr = PetscMalloc(sp->functional[f].numPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dm, csection, coordinates, p, &csize, &coords);CHKERRQ(ierr);
        if (csize%dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate size %d is not divisible by spatial dimension %d", csize, dim);
        for (d = 0; d < dim; ++d) {
          const PetscInt numVertices = csize/dim;

          qpoints[d] = 0.0;
          for (v = 0; v < numVertices; ++v) {
            qpoints[d] += PetscRealPart(coords[v*dim+d]);
          }
          qpoints[d] /= numVertices;
        }
        ierr = DMPlexVecRestoreClosure(dm, csection, coordinates, p, &csize, &coords);CHKERRQ(ierr);
        qweights[0] = 1.0;
        sp->functional[f].points  = qpoints;
        sp->functional[f].weights = qweights;
        ++f;
        lag->numDof[depth] = 1;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, pStart[depth], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  if (f != pdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %d not equal to dimension %d", f, pdim);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy_Lagrange"
PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(lag->numDof);CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension_Lagrange"
PetscErrorCode PetscDualSpaceGetDimension_Lagrange(PetscDualSpace sp, PetscInt *dim)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt            deg = sp->order;
  PetscReal           D   = 1.0;
  PetscInt            n, i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(sp->dm, &n);CHKERRQ(ierr);
  if (lag->simplex) {
    for (i = 1; i <= n; ++i) {
      D *= ((PetscReal) (deg+i))/i;
    }
    *dim = (PetscInt) (D + 0.5);
  } else {
    *dim = 1;
    for (i = 0; i < n; ++i) *dim *= (deg+1);
  }
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
#define __FUNCT__ "PetscDualSpaceInitialize_Lagrange"
PetscErrorCode PetscDualSpaceInitialize_Lagrange(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = NULL;
  sp->ops->setup          = PetscDualSpaceSetUp_Lagrange;
  sp->ops->view           = NULL;
  sp->ops->destroy        = PetscDualSpaceDestroy_Lagrange;
  sp->ops->getdimension   = PetscDualSpaceGetDimension_Lagrange;
  sp->ops->getnumdof      = PetscDualSpaceGetNumDof_Lagrange;
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
  ierr     = PetscNewLog(sp, PetscDualSpace_Lag, &lag);CHKERRQ(ierr);
  sp->data = lag;

  lag->numDof  = NULL;
  lag->simplex = PETSC_TRUE;

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
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
. dm  - The PetscFE

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
  PetscValidCharPointer(name, 2);
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
#define __FUNCT__ "PetscFEViewFromOptions"
/*
  PetscFEViewFromOptions - Processes command line options to determine if/how a PetscFE is to be viewed.

  Collective on PetscFE

  Input Parameters:
+ fem    - the PetscFE
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscFE, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscFEViewFromOptions(PetscFE fem, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) fem), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) fem), ((PetscObject) fem)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscFEView(fem, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
    ierr = (*fem->ops->setfromoptions)(fem);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) fem);CHKERRQ(ierr);
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
  ierr = PetscFERestoreTabulation((*fem), 0, NULL, &(*fem)->B, &(*fem)->D, NULL /*&(*fem)->H*/);CHKERRQ(ierr);
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
  *fem = NULL;
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(f, _p_PetscFE, struct _PetscFEOps, PETSCFE_CLASSID, "PetscFE", "Finite Element", "PetscFE", comm, PetscFEDestroy, PetscFEView);CHKERRQ(ierr);
  ierr = PetscMemzero(f->ops, sizeof(struct _PetscFEOps));CHKERRQ(ierr);

  f->basisSpace    = NULL;
  f->dualSpace     = NULL;
  f->numComponents = 1;
  f->numDof        = NULL;
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
#define __FUNCT__ "PetscFEGetDimension"
PetscErrorCode PetscFEGetDimension(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  ierr = PetscSpaceGetDimension(fem->basisSpace, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetSpatialDimension"
PetscErrorCode PetscFEGetSpatialDimension(PetscFE fem, PetscInt *dim)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFESetNumComponents"
PetscErrorCode PetscFESetNumComponents(PetscFE fem, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->numComponents = comp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumComponents"
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
PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscQuadratureDestroy(&fem->quadrature);CHKERRQ(ierr);
  fem->quadrature = q;
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
    ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &fem->numDof);CHKERRQ(ierr);
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
  PetscInt         npoints = fem->quadrature.numPoints;
  const PetscReal *points  = fem->quadrature.points;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (B) PetscValidPointer(B, 2);
  if (D) PetscValidPointer(D, 3);
  if (H) PetscValidPointer(H, 4);
  if (!fem->B) {ierr = PetscFEGetTabulation(fem, npoints, points, &fem->B, &fem->D, NULL/*&fem->H*/);CHKERRQ(ierr);}
  if (B) *B = fem->B;
  if (D) *D = fem->D;
  if (H) *H = fem->H;
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
  PetscReal       *tmpB, *tmpD, *invV;
  PetscInt         p, d, j, k;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);

  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  /* if (nvalues%dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate values %d must be divisible by the spatial dimension %d", nvalues, dim); */

  if (B) {
    ierr = DMGetWorkArray(dm, npoints*pdim*comp, PETSC_REAL, B);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);
  }
  if (D) {
    ierr = DMGetWorkArray(dm, npoints*pdim*comp*dim, PETSC_REAL, D);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);
  }
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*dim*dim, PETSC_REAL, H);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? *H : NULL);CHKERRQ(ierr);

  ierr = DMGetWorkArray(dm, pdim*pdim, PETSC_REAL, &invV);CHKERRQ(ierr);
  for (j = 0; j < pdim; ++j) {
    PetscReal      *Bf;
    PetscQuadrature f;
    PetscInt        q;

    ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);
    ierr = DMGetWorkArray(dm, f.numPoints*pdim, PETSC_REAL, &Bf);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(fem->basisSpace, f.numPoints, f.points, Bf, NULL, NULL);CHKERRQ(ierr);
    for (k = 0; k < pdim; ++k) {
      /* n_j \cdot \phi_k */
      invV[j*pdim+k] = 0.0;
      for (q = 0; q < f.numPoints; ++q) {
        invV[j*pdim+k] += Bf[q*pdim+k]*f.weights[q];
      }
    }
    ierr = DMRestoreWorkArray(dm, f.numPoints*pdim, PETSC_REAL, &Bf);CHKERRQ(ierr);
  }
  {
    PetscReal    *work;
    PetscBLASInt *pivots;
#ifndef PETSC_USE_COMPLEX
    PetscBLASInt  n = pdim, info;
#endif

    ierr = DMGetWorkArray(dm, pdim, PETSC_INT, &pivots);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, pdim, PETSC_REAL, &work);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, invV, &n, pivots, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, invV, &n, pivots, work, &n, &info));
#endif
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_INT, &pivots);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_REAL, &work);CHKERRQ(ierr);
  }
  for (p = 0; p < npoints; ++p) {
    if (B) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        const PetscInt i = (p*pdim + j)*comp;
        PetscInt       c;

        (*B)[i] = 0.0;
        for (k = 0; k < pdim; ++k) {
          (*B)[i] += invV[k*pdim+j] * tmpB[p*pdim + k];
        }
        for (c = 1; c < comp; ++c) {
          (*B)[i+c] = (*B)[i];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + j)*comp + 0)*dim + d;
          PetscInt       c;

          (*D)[i] = 0.0;
          for (k = 0; k < pdim; ++k) {
            (*D)[i] += invV[k*pdim+j] * tmpD[(p*pdim + k)*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            (*D)[((p*pdim + j)*comp + c)*dim + d] = (*D)[i];
          }
        }
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, pdim*pdim, PETSC_REAL, &invV);CHKERRQ(ierr);
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
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
#define __FUNCT__ "PetscFEIntegrateResidual_Basic"
PetscErrorCode PetscFEIntegrateResidual_Basic(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                              PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                              void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]),
                                              void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]),
                                              PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscQuadrature quad;
  PetscScalar    *f0, *f1, *u, *gradU, *a, *gradA = NULL;
  PetscReal      *x, *realSpaceDer;
  PetscInt        dim, numComponents = 0, numComponentsAux = 0, cOffset = 0, cOffsetAux = 0, eOffset = 0, e, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe[0], &dim);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscInt Nc;
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  ierr = PetscFEGetQuadrature(fe[field], &quad);CHKERRQ(ierr);
  ierr = PetscMalloc6(quad.numPoints*dim,PetscScalar,&f0,quad.numPoints*dim*dim,PetscScalar,&f1,numComponents,PetscScalar,&u,numComponents*dim,PetscScalar,&gradU,dim,PetscReal,&x,dim,PetscReal,&realSpaceDer);
  for (f = 0; f < NfAux; ++f) {
    PetscInt Nc;
    ierr = PetscFEGetNumComponents(feAux[f], &Nc);CHKERRQ(ierr);
    numComponentsAux += Nc;
  }
  if (NfAux) {ierr = PetscMalloc2(numComponentsAux,PetscScalar,&a,numComponentsAux*dim,PetscScalar,&gradA);CHKERRQ(ierr);}
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ        = geom.detJ[e];
    const PetscReal *v0          = &geom.v0[e*dim];
    const PetscReal *J           = &geom.J[e*dim*dim];
    const PetscReal *invJ        = &geom.invJ[e*dim*dim];
    const PetscInt   Nq          = quad.numPoints;
    const PetscReal *quadPoints  = quad.points;
    const PetscReal *quadWeights = quad.weights;
    PetscInt         q, f;

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      PetscInt         fOffset = 0,       fOffsetAux = 0;
      PetscInt         dOffset = cOffset, dOffsetAux = cOffsetAux;
      PetscInt         Ncomp, d, d2, f, i;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      ierr = PetscFEGetNumComponents(fe[field], &Ncomp);CHKERRQ(ierr);
      for (d = 0; d < numComponents; ++d)       {u[d]     = 0.0;}
      for (d = 0; d < dim*(numComponents); ++d) {gradU[d] = 0.0;}
      for (d = 0; d < dim; ++d) {
        x[d] = v0[d];
        for (d2 = 0; d2 < dim; ++d2) {
          x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
        }
      }
      for (f = 0; f < Nf; ++f) {
        PetscReal *basis, *basisDer;
        PetscInt   Nb, Ncomp, b, comp;

        ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       d, g;

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, PetscRealPart(u[fOffset+comp]));CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradU[(fOffset+comp)*dim+d]));CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }
      for (d = 0; d < numComponentsAux; ++d)       {a[d]     = 0.0;}
      for (d = 0; d < dim*(numComponentsAux); ++d) {gradA[d] = 0.0;}
      for (f = 0; f < NfAux; ++f) {
        PetscReal *basis, *basisDer;
        PetscInt   Nb, Ncomp, b, comp;

        ierr = PetscFEGetDimension(feAux[f], &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(feAux[f], &Ncomp);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(feAux[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       d, g;

            a[fOffsetAux+comp] += coefficientsAux[dOffsetAux+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradA[(fOffsetAux+comp)*dim+d] += coefficients[dOffsetAux+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    a[%d,%d]: %g\n", f, comp, PetscRealPart(a[fOffsetAux+comp]));CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradA[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradA[(fOffsetAux+comp)*dim+d]));CHKERRQ(ierr);
            }
          }
        }
        fOffsetAux += Ncomp;
        dOffsetAux += Nb*Ncomp;
      }

      f0_func(u, gradU, a, gradA, x, &f0[q*Ncomp]);
      for (i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, a, gradA, x, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c,d;
        for (c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, PetscRealPart(f0[q*Ncomp+c]));CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, PetscRealPart(f1[(q*Ncomp + c)*dim+d]));CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset; cOffsetAux = dOffsetAux;}
    }
    for (f = 0; f < Nf; ++f) {
      PetscInt   Nb, Ncomp, b, comp;

      ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
      if (f == field) {
        PetscReal *basis;
        PetscReal *basisDer;

        ierr = PetscFEGetDefaultTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       q;

            elemVec[eOffset+cidx] = 0.0;
            for (q = 0; q < Nq; ++q) {
              PetscInt d, g;

              elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
              for (d = 0; d < dim; ++d) {
                realSpaceDer[d] = 0.0;
                for (g = 0; g < dim; ++g) {
                  realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
                }
                elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
              }
            }
          }
        }
        if (debug > 1) {
          PetscInt b, comp;

          for (b = 0; b < Nb; ++b) {
            for (comp = 0; comp < Ncomp; ++comp) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, PetscRealPart(elemVec[eOffset+b*Ncomp+comp]));CHKERRQ(ierr);
            }
          }
        }
      }
      eOffset += Nb*Ncomp;
    }
  }
  ierr = PetscFree6(f0,f1,u,gradU,x,realSpaceDer);
  if (NfAux) {ierr = PetscFree2(a,gradA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdResidual_Basic"
PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                                PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                                void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]),
                                                void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]),
                                                PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscQuadrature quad;
  PetscScalar    *f0, *f1, *u, *gradU, *a, *gradA = NULL;
  PetscReal      *x, *realSpaceDer;
  PetscInt        dim, numComponents = 0, numComponentsAux = 0, cOffset = 0, cOffsetAux = 0, eOffset = 0, e, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe[0], &dim);CHKERRQ(ierr);
  dim += 1; /* Spatial dimension is one higher than topological dimension */
  for (f = 0; f < Nf; ++f) {
    PetscInt Nc;
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  ierr = PetscFEGetQuadrature(fe[field], &quad);CHKERRQ(ierr);
  ierr = PetscMalloc6(quad.numPoints*dim,PetscScalar,&f0,quad.numPoints*dim*dim,PetscScalar,&f1,numComponents,PetscScalar,&u,numComponents*dim,PetscScalar,&gradU,dim,PetscReal,&x,dim,PetscReal,&realSpaceDer);
  for (f = 0; f < NfAux; ++f) {
    PetscInt Nc;
    ierr = PetscFEGetNumComponents(feAux[f], &Nc);CHKERRQ(ierr);
    numComponentsAux += Nc;
  }
  if (NfAux) {ierr = PetscMalloc2(numComponentsAux,PetscScalar,&a,numComponentsAux*dim,PetscScalar,&gradA);CHKERRQ(ierr);}
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ        = geom.detJ[e];
    const PetscReal *v0          = &geom.v0[e*dim];
    const PetscReal *n           = &geom.n[e*dim];
    const PetscReal *J           = &geom.J[e*dim*dim];
    const PetscReal *invJ        = &geom.invJ[e*dim*dim];
    const PetscInt   Nq          = quad.numPoints;
    const PetscReal *quadPoints  = quad.points;
    const PetscReal *quadWeights = quad.weights;
    PetscInt         q, f;

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
    }
    for (q = 0; q < Nq; ++q) {
      PetscInt         fOffset = 0,       fOffsetAux = 0;
      PetscInt         dOffset = cOffset, dOffsetAux = cOffsetAux;
      PetscInt         Ncomp, d, d2, f, i;
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}

      ierr = PetscFEGetNumComponents(fe[field], &Ncomp);CHKERRQ(ierr);
      for (d = 0; d < numComponents; ++d)       {u[d]     = 0.0;}
      for (d = 0; d < dim*(numComponents); ++d) {gradU[d] = 0.0;}
      for (d = 0; d < dim; ++d) {
        x[d] = v0[d];
        for (d2 = 0; d2 < dim-1; ++d2) {
          x[d] += J[d*dim+d2]*(quadPoints[q*(dim-1)+d2] + 1.0);
        }
      }
      for (f = 0; f < Nf; ++f) {
        PetscReal *basis, *basisDer;
        PetscInt   Nb, Ncomp, b, comp;

        ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       d, g;

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim-1; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, PetscRealPart(u[fOffset+comp]));CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradU[(fOffset+comp)*dim+d]));CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }
      for (d = 0; d < numComponentsAux; ++d)       {a[d]     = 0.0;}
      for (d = 0; d < dim*(numComponentsAux); ++d) {gradA[d] = 0.0;}
      for (f = 0; f < NfAux; ++f) {
        PetscReal *basis, *basisDer;
        PetscInt   Nb, Ncomp, b, comp;

        ierr = PetscFEGetDimension(feAux[f], &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(feAux[f], &Ncomp);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(feAux[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       d, g;

            a[fOffsetAux+comp] += coefficientsAux[dOffsetAux+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim-1; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradA[(fOffsetAux+comp)*dim+d] += coefficients[dOffsetAux+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    a[%d,%d]: %g\n", f, comp, PetscRealPart(a[fOffsetAux+comp]));CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradA[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradA[(fOffsetAux+comp)*dim+d]));CHKERRQ(ierr);
            }
          }
        }
        fOffsetAux += Ncomp;
        dOffsetAux += Nb*Ncomp;
      }

      f0_func(u, gradU, a, gradA, x, n, &f0[q*Ncomp]);
      for (i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, a, gradA, x, n, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c,d;
        for (c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, PetscRealPart(f0[q*Ncomp+c]));CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, PetscRealPart(f1[(q*Ncomp + c)*dim+d]));CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset; cOffsetAux = dOffsetAux;}
    }
    for (f = 0; f < Nf; ++f) {
      PetscInt   Nb, Ncomp, b, comp;

      ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
      if (f == field) {
        PetscReal *basis;
        PetscReal *basisDer;

        ierr = PetscFEGetDefaultTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       q;

            elemVec[eOffset+cidx] = 0.0;
            for (q = 0; q < Nq; ++q) {
              PetscInt d, g;

              elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
              for (d = 0; d < dim; ++d) {
                realSpaceDer[d] = 0.0;
                for (g = 0; g < dim-1; ++g) {
                  realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
                }
                elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
              }
            }
          }
        }
        if (debug > 1) {
          PetscInt b, comp;

          for (b = 0; b < Nb; ++b) {
            for (comp = 0; comp < Ncomp; ++comp) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, PetscRealPart(elemVec[eOffset+b*Ncomp+comp]));CHKERRQ(ierr);
            }
          }
        }
      }
      eOffset += Nb*Ncomp;
    }
  }
  ierr = PetscFree6(f0,f1,u,gradU,x,realSpaceDer);
  if (NfAux) {ierr = PetscFree2(a,gradA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobian_Basic"
PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt fieldI, PetscInt fieldJ, PetscCellGeometry geom, const PetscScalar coefficients[],
                                              PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                              void (*g0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]),
                                              void (*g1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]),
                                              void (*g2_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]),
                                              void (*g3_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]),
                                              PetscScalar elemMat[])
{
  const PetscInt  debug      = 0;
  PetscInt        cellDof    = 0; /* Total number of dof on a cell */
  PetscInt        cellDofAux = 0; /* Total number of auxiliary dof on a cell */
  PetscInt        cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt        cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt        eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt        offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt        offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature quad;
  PetscScalar    *g0, *g1, *g2, *g3, *u, *gradU, *a, *gradA = NULL;
  PetscReal      *x, *realSpaceDerI, *realSpaceDerJ;
  PetscReal      *basisI, *basisDerI, *basisJ, *basisDerJ;
  PetscInt        NbI = 0, NcI = 0, NbJ = 0, NcJ = 0, numComponents = 0, numComponentsAux = 0;
  PetscInt        dim, f, e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe[fieldI], &dim);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fe[fieldI], &basisI, &basisDerI, NULL);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fe[fieldJ], &basisJ, &basisDerJ, NULL);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    if (f == fieldI) {offsetI = cellDof; NbI = Nb; NcI = Nc;}
    if (f == fieldJ) {offsetJ = cellDof; NbJ = Nb; NcJ = Nc;}
    numComponents += Nc;
    cellDof += Nb*Nc;
  }
  ierr = PetscFEGetQuadrature(fe[fieldI], &quad);CHKERRQ(ierr);
  ierr = PetscMalloc4(NcI*NcJ,PetscScalar,&g0,NcI*NcJ*dim,PetscScalar,&g1,NcI*NcJ*dim,PetscScalar,&g2,NcI*NcJ*dim*dim,PetscScalar,&g3);
  ierr = PetscMalloc5(numComponents,PetscScalar,&u,numComponents*dim,PetscScalar,&gradU,dim,PetscReal,&x,dim,PetscReal,&realSpaceDerI,dim,PetscReal,&realSpaceDerJ);
  for (f = 0; f < NfAux; ++f) {
    PetscInt Nb, Nc;
    ierr = PetscFEGetDimension(feAux[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feAux[f], &Nc);CHKERRQ(ierr);
    numComponentsAux += Nc;
    cellDofAux       += Nb*Nc;
  }
  if (NfAux) {ierr = PetscMalloc2(numComponentsAux,PetscScalar,&a,numComponentsAux*dim,PetscScalar,&gradA);CHKERRQ(ierr);}
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ        = geom.detJ[e];
    const PetscReal *v0          = &geom.v0[e*dim];
    const PetscReal *J           = &geom.J[e*dim*dim];
    const PetscReal *invJ        = &geom.invJ[e*dim*dim];
    const PetscInt   Nq          = quad.numPoints;
    const PetscReal *quadPoints  = quad.points;
    const PetscReal *quadWeights = quad.weights;
    PetscInt         f, g, q;

    for (f = 0; f < NbI; ++f) {
      for (g = 0; g < NbJ; ++g) {
        for (q = 0; q < Nq; ++q) {
          PetscInt    fOffset    = 0;          /* Offset into u[] for field_q (like offsetI) */
          PetscInt    dOffset    = cOffset;    /* Offset into coefficients[] for field_q */
          PetscInt    fOffsetAux = 0;          /* Offset into a[] for field_q (like offsetI) */
          PetscInt    dOffsetAux = cOffsetAux; /* Offset into coefficientsAux[] for field_q */
          PetscInt    field_q, d, d2;
          PetscInt    fc, gc, c;

          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
          for (d = 0; d < numComponents; ++d)     {u[d]     = 0.0;}
          for (d = 0; d < dim*numComponents; ++d) {gradU[d] = 0.0;}
          for (d = 0; d < dim; ++d) {
            x[d] = v0[d];
            for (d2 = 0; d2 < dim; ++d2) {
              x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
            }
          }
          for (field_q = 0; field_q < Nf; ++field_q) {
            PetscReal *basis, *basisDer;
            PetscInt   Nb, Ncomp, b, comp;

            ierr = PetscFEGetDimension(fe[field_q], &Nb);CHKERRQ(ierr);
            ierr = PetscFEGetNumComponents(fe[field_q], &Ncomp);CHKERRQ(ierr);
            ierr = PetscFEGetDefaultTabulation(fe[field_q], &basis, &basisDer, NULL);CHKERRQ(ierr);
            for (b = 0; b < Nb; ++b) {
              for (comp = 0; comp < Ncomp; ++comp) {
                const PetscInt cidx = b*Ncomp+comp;
                PetscInt       d1, d2;

                u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
                for (d1 = 0; d1 < dim; ++d1) {
                  realSpaceDerI[d1] = 0.0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    realSpaceDerI[d1] += invJ[d2*dim+d1]*basisDer[(q*Nb*Ncomp+cidx)*dim+d2];
                  }
                  gradU[(fOffset+comp)*dim+d1] += coefficients[dOffset+cidx]*realSpaceDerI[d1];
                }
              }
            }
            if (debug > 1) {
              for (comp = 0; comp < Ncomp; ++comp) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, PetscRealPart(u[fOffset+comp]));CHKERRQ(ierr);
                for (d = 0; d < dim; ++d) {
                  ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradU[(fOffset+comp)*dim+d]));CHKERRQ(ierr);
                }
              }
            }
            fOffset += Ncomp;
            dOffset += Nb*Ncomp;
          }
          for (d = 0; d < numComponentsAux; ++d)       {a[d]     = 0.0;}
          for (d = 0; d < dim*(numComponentsAux); ++d) {gradA[d] = 0.0;}
          for (field_q = 0; field_q < NfAux; ++field_q) {
            PetscReal *basis, *basisDer;
            PetscInt   Nb, Ncomp, b, comp;

            ierr = PetscFEGetDimension(feAux[field_q], &Nb);CHKERRQ(ierr);
            ierr = PetscFEGetNumComponents(feAux[field_q], &Ncomp);CHKERRQ(ierr);
            ierr = PetscFEGetDefaultTabulation(feAux[field_q], &basis, &basisDer, NULL);CHKERRQ(ierr);
            for (b = 0; b < Nb; ++b) {
              for (comp = 0; comp < Ncomp; ++comp) {
                const PetscInt cidx = b*Ncomp+comp;
                PetscInt       d1, d2;

                a[fOffsetAux+comp] += coefficientsAux[dOffsetAux+cidx]*basis[q*Nb*Ncomp+cidx];
                for (d1 = 0; d1 < dim; ++d1) {
                  realSpaceDerI[d1] = 0.0;
                  for (d2 = 0; d2 < dim; ++d2) {
                    realSpaceDerI[d1] += invJ[d2*dim+d1]*basisDer[(q*Nb*Ncomp+cidx)*dim+d2];
                  }
                  gradA[(fOffsetAux+comp)*dim+d1] += coefficientsAux[dOffsetAux+cidx]*realSpaceDerI[d1];
                }
              }
            }
            if (debug > 1) {
              for (comp = 0; comp < Ncomp; ++comp) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "    a[%d,%d]: %g\n", f, comp, PetscRealPart(a[fOffsetAux+comp]));CHKERRQ(ierr);
                for (d = 0; d < dim; ++d) {
                  ierr = PetscPrintf(PETSC_COMM_SELF, "    gradA[%d,%d]_%c: %g\n", f, comp, 'x'+d, PetscRealPart(gradA[(fOffsetAux+comp)*dim+d]));CHKERRQ(ierr);
                }
              }
            }
            fOffsetAux += Ncomp;
            dOffsetAux += Nb*Ncomp;
          }

          ierr = PetscMemzero(g0, NcI*NcJ         * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g1, NcI*NcJ*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g2, NcI*NcJ*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
          if (g0_func) {
            g0_func(u, gradU, a, gradA, x, g0);
            for (c = 0; c < NcI*NcJ; ++c) {g0[c] *= detJ*quadWeights[q];}
          }
          if (g1_func) {
            g1_func(u, gradU, a, gradA, x, g1);
            for (c = 0; c < NcI*NcJ*dim; ++c) {g1[c] *= detJ*quadWeights[q];}
          }
          if (g2_func) {
            g2_func(u, gradU, a, gradA, x, g2);
            for (c = 0; c < NcI*NcJ*dim; ++c) {g2[c] *= detJ*quadWeights[q];}
          }
          if (g3_func) {
            g3_func(u, gradU, a, gradA, x, g3);
            for (c = 0; c < NcI*NcJ*dim*dim; ++c) {g3[c] *= detJ*quadWeights[q];}
          }

          for (fc = 0; fc < NcI; ++fc) {
            const PetscInt fidx = f*NcI+fc; /* Test function basis index */
            const PetscInt i    = offsetI+fidx; /* Element matrix row */
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+gidx; /* Element matrix column */
              PetscInt       d, d2;

              for (d = 0; d < dim; ++d) {
                realSpaceDerI[d] = 0.0;
                realSpaceDerJ[d] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  realSpaceDerI[d] += invJ[d2*dim+d]*basisDerI[(q*NbI*NcI+fidx)*dim+d2];
                  realSpaceDerJ[d] += invJ[d2*dim+d]*basisDerJ[(q*NbJ*NcJ+gidx)*dim+d2];
                }
              }
              elemMat[eOffset+i*cellDof+j] += basisI[q*NbI*NcI+fidx]*g0[fc*NcJ+gc]*basisJ[q*NbJ*NcJ+gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[eOffset+i*cellDof+j] += basisI[q*NbI*NcI+fidx]*g1[(fc*NcJ+gc)*dim+d]*realSpaceDerJ[d];
                elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g2[(fc*NcJ+gc)*dim+d]*basisJ[q*NbJ*NcJ+gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*realSpaceDerJ[d2];
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
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*cellDof+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += cellDof;
    cOffsetAux += cellDofAux;
    eOffset    += cellDof*cellDof;
  }
  ierr = PetscFree4(g0,g1,g2,g3);
  ierr = PetscFree5(u,gradU,x,realSpaceDerI,realSpaceDerJ);
  if (NfAux) {ierr = PetscFree2(a,gradA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitialize_Basic"
PetscErrorCode PetscFEInitialize_Basic(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = NULL;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
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
  ierr      = PetscNewLog(fem, PetscFE_Basic, &b);CHKERRQ(ierr);
  fem->data = b;

  ierr = PetscFEInitialize_Basic(fem);CHKERRQ(ierr);
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
  N_q  = q.numPoints;
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
"  const int numQuadraturePoints = %d;\n"
"\n"
"  /* Quadrature points\n"
"   - (x1,y1,x2,y2,...) */\n"
"  const %s points[%d] = {\n",
                       &count, N_q, numeric_str, N_q*dim);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    for (d = 0; d < dim; ++d) {
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, q.points[p*dim+d]);STRING_ERROR_CHECK("Message to short");
    }
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  /* Quadrature weights\n"
"   - (v1,v2,...) */\n"
"  const %s weights[%d] = {\n",
                       &count, numeric_str, N_q);STRING_ERROR_CHECK("Message to short");
  for (p = 0; p < N_q; ++p) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "%g,\n", &count, q.weights[p]);STRING_ERROR_CHECK("Message to short");
  }
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "};\n", &count);STRING_ERROR_CHECK("Message to short");
  /* Basis Functions */
  ierr = PetscFEGetDefaultTabulation(fem, &basis, &basisDer, NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  const int numBasisFunctions  = %d;\n"
"  const int numBasisComponents = %d;\n"
"\n"
"  /* Nodal basis function evaluations\n"
"    - basis component is fastest varying, the basis function, then point */\n"
"  const %s Basis[%d] = {\n",
                       &count, N_b, N_c, numeric_str, N_q*N_b*N_c);STRING_ERROR_CHECK("Message to short");
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
"  /* Number of concurrent blocks */\n"
"  const int N_bl = %d;\n"
"\n"
"  const int dim    = %d;\n"
"  const int N_b    = numBasisFunctions;             // The number of basis functions\n"
"  const int N_comp = numBasisComponents;            // The number of basis function components\n"
"  const int N_bt   = N_b*N_comp;                    // The total number of scalar basis functions\n"
"  const int N_q    = numQuadraturePoints;           // The number of quadrature points\n"
"  const int N_bst  = N_bt*N_q;                      // The block size, LCM(N_b*N_comp, N_q), Notice that a block is not processed simultaneously\n"
"  const int N_t    = N_bst*N_bl;                    // The number of threads, N_bst * N_bl\n"
"  const int N_bc   = N_t/N_comp;                    // The number of cells per batch (N_b*N_q*N_bl)\n"
"  const int N_c    = N_cb * N_bc;\n"
"  const int N_sbc  = N_bst / (N_q * N_comp);\n"
"  const int N_sqc  = N_bst / N_bt;\n"
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
"  const int Goffset = gidx*N_c;\n"
"  const int Coffset = gidx*N_c*N_bt;\n"
"  const int Eoffset = gidx*N_c*N_bt;\n",
                       &count, N_bl, dim);STRING_ERROR_CHECK("Message to short");
  if (useAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"  const int Aoffset = gidx*N_c;\n",
                              &count);STRING_ERROR_CHECK("Message to short");
  }
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
"      u_i[offset+tidx] = coefficients[Coffset+batch*N_t*N_b+offset+tidx];\n"
"    }\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  if (useAux) {
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    /* Load coefficients a_i for this cell */\n"
"    a_i[tidx] = coefficientsAux[Aoffset+batch*N_t+tidx];\n",
                            &count);STRING_ERROR_CHECK("Message to short");
  }
  /* Quadrature phase */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
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
  if (useAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      %s  a[%d]; //[1];     // $a(x_q)$, Value of the auxiliary fields at $x_q$\n",
                              &count, numeric_str, 1);STRING_ERROR_CHECK("Message to short");
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
  if (useAux) {
    ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      a[0] = 0.0;\n", &count);STRING_ERROR_CHECK("Message to short");
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
  if (useAux) {
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
      if (useAux) {ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail, "      f_1[fidx] = a[cell]*gradU[cidx];\n", &count);STRING_ERROR_CHECK("Message to short");}
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
"        f_1[fidx].x = 0.5*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = 0.5*(gradU[0].y + gradU[1].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = 0.5*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = 0.5*(gradU[1].y + gradU[1].y);\n"
"      }\n",
                           &count);STRING_ERROR_CHECK("Message to short");break;
    case 3:
      ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"      switch (cidx) {\n"
"      case 0:\n"
"        f_1[fidx].x = 0.5*(gradU[0].x + gradU[0].x);\n"
"        f_1[fidx].y = 0.5*(gradU[0].y + gradU[1].x);\n"
"        f_1[fidx].z = 0.5*(gradU[0].z + gradU[2].x);\n"
"        break;\n"
"      case 1:\n"
"        f_1[fidx].x = 0.5*(gradU[1].x + gradU[0].y);\n"
"        f_1[fidx].y = 0.5*(gradU[1].y + gradU[1].y);\n"
"        f_1[fidx].z = 0.5*(gradU[1].y + gradU[2].y);\n"
"        break;\n"
"      case 2:\n"
"        f_1[fidx].x = 0.5*(gradU[2].x + gradU[0].z);\n"
"        f_1[fidx].y = 0.5*(gradU[2].y + gradU[1].z);\n"
"        f_1[fidx].z = 0.5*(gradU[2].y + gradU[2].z);\n"
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
"    barrier(CLK_GLOBAL_MEM_FENCE);\n\n",
                       &count);STRING_ERROR_CHECK("Message to short");
  /* Basis phase */
  ierr = PetscSNPrintfCount(string_tail, end_of_buffer - string_tail,
"    /* Map values at quadrature points to coefficients */\n"
"    for (int c = 0; c < N_sbc; ++c) {\n"
"      const int cell = c*N_bl*N_q + blbidx;\n"
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
"      elemVec[Eoffset+(batch*N_sbc+c)*N_t+tidx] = e_i;\n"
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
  char           *buffer;
  size_t          len;
  char            errMsg[8192];
  cl_int          ierr2;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc(8192 * sizeof(char), &buffer);CHKERRQ(ierr);
  ierr = PetscFEGetTileSizes(fem, NULL, &N_bl, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFEOpenCLGenerateIntegrationCode(fem, &buffer, 8192, useAux, N_bl);CHKERRQ(ierr);
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
  for (*x = (int) (sqrt(Nblocks) + 0.5); *x > 0; --*x) {
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
PetscErrorCode PetscFEIntegrateResidual_OpenCL(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                               PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                               void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]),
                                               void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]),
                                               PetscScalar elemVec[])
{
  /* Nbc = batchSize */
  PetscFE_OpenCL   *ocl = (PetscFE_OpenCL *) fem->data;
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
  /* OpenCL variables */
  cl_program        ocl_prog;
  cl_kernel         ocl_kernel;
  cl_event          ocl_ev;         /* The event for tracking kernel execution */
  cl_ulong          ns_start;       /* Nanoseconds counter on GPU at kernel start */
  cl_ulong          ns_end;         /* Nanoseconds counter on GPU at kernel stop */
  cl_mem            o_jacobianInverses, o_jacobianDeterminants;
  cl_mem            o_coefficients, o_coefficientsAux, o_elemVec;
  float            *f_coeff, *f_coeffAux, *f_invJ, *f_detJ;
  double           *d_coeff, *d_coeffAux, *d_invJ, *d_detJ;
  void             *oclCoeff, *oclCoeffAux, *oclInvJ, *oclDetJ;
  size_t            local_work_size[3], global_work_size[3];
  size_t            realSize, x, y, z;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!Ne) {ierr = PetscFEOpenCLLogResidual(fem, 0.0, 0.0);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &N_b);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &N_comp);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &q);CHKERRQ(ierr);
  ierr = PetscFEGetTileSizes(fem, NULL, &N_bl, &N_bc, &N_cb);CHKERRQ(ierr);
  N_bt  = N_b*N_comp;
  N_q   = q.numPoints;
  N_bst = N_bt*N_q;
  N_t   = N_bst*N_bl;
  if (N_bc*N_comp != N_t) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of threads %d should be %d * %d", N_t, N_bc, N_comp);
  /* Calculate layout */
  if (Ne % (N_cb*N_bc)) { /* Remainder cells */
    ierr = PetscFEIntegrateResidual_Basic(fem, Ne, Nf, fe, field, geom, coefficients, NfAux, feAux, coefficientsAux, f0_func, f1_func, elemVec);CHKERRQ(ierr);
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
  if (NfAux) {
    PetscSpace P;
    PetscInt   order, f;

    for (f = 0; f < NfAux; ++f) {
      ierr = PetscFEGetBasisSpace(feAux[f], &P);CHKERRQ(ierr);
      ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
      if (order > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Can only handle P0 coefficient fields");
    }
  }
  ierr = PetscFEOpenCLGetIntegrationKernel(fem, coefficientsAux ? PETSC_TRUE : PETSC_FALSE, &ocl_prog, &ocl_kernel);CHKERRQ(ierr);
  /* Create buffers on the device and send data over */
  ierr = PetscDataTypeGetSize(ocl->realType, &realSize);CHKERRQ(ierr);
  if (sizeof(PetscReal) != realSize) {
    switch (ocl->realType) {
    case PETSC_FLOAT:
    {
      PetscInt c, b, d;

      ierr = PetscMalloc4(Ne*N_bt,float,&f_coeff,Ne,float,&f_coeffAux,Ne*dim*dim,float,&f_invJ,Ne,float,&f_detJ);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        f_detJ[c] = (float) geom.detJ[c];
        for (d = 0; d < dim*dim; ++d) {
          f_invJ[c*dim*dim+d] = (float) geom.invJ[c*dim*dim+d];
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

      ierr = PetscMalloc4(Ne*N_bt,double,&d_coeff,Ne,double,&d_coeffAux,Ne*dim*dim,double,&d_invJ,Ne,double,&d_detJ);CHKERRQ(ierr);
      for (c = 0; c < Ne; ++c) {
        d_detJ[c] = (double) geom.detJ[c];
        for (d = 0; d < dim*dim; ++d) {
          d_invJ[c*dim*dim+d] = (double) geom.invJ[c*dim*dim+d];
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
    oclCoeff    = (void *) coefficients;
    oclCoeffAux = (void *) coefficientsAux;
    oclInvJ     = (void *) geom.invJ;
    oclDetJ     = (void *) geom.detJ;
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
      ierr = PetscMalloc(Ne*N_bt * sizeof(float), &elem);CHKERRQ(ierr);
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
      ierr = PetscMalloc(Ne*N_bt * sizeof(double), &elem);CHKERRQ(ierr);
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
    ierr = clEnqueueReadBuffer(ocl->queue_id, o_elemVec, CL_TRUE, 0, Ne*N_bt * realSize, elemVec, 0, NULL, NULL);CHKERRQ(ierr);
  }
  /* Log performance */
  ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ns_start, NULL);CHKERRQ(ierr);
  ierr = clGetEventProfilingInfo(ocl_ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &ns_end,   NULL);CHKERRQ(ierr);
  ierr = PetscFEOpenCLLogResidual(fem, (ns_end - ns_start)*1.0e-9, (((2+(2+2*dim)*dim)*N_comp*N_b+(2+2)*dim*N_comp)*N_q + (2+2*dim)*dim*N_q*N_comp*N_b)*Ne);CHKERRQ(ierr);
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
  fem->ops->setup                   = NULL;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_OpenCL;
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
  ierr      = PetscNewLog(fem, PetscFE_OpenCL, &ocl);CHKERRQ(ierr);
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
     PetscCellGeometry
       PetscReal v0s[Ne*dim]
       PetscReal jacobians[Ne*dim*dim]        possibly *Nq
       PetscReal jacobianInverses[Ne*dim*dim] possibly *Nq
       PetscReal jacobianDeterminants[Ne]     possibly *Nq
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
#define __FUNCT__ "PetscFEIntegrateResidual"
/*C
  PetscFEIntegrateResidual - Produce the element residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. Ne           - The number of elements in the chunk
. Nf           - The number of physical fields
. fe           - The PetscFE objects for each field
. field        - The field being integrated
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. NfAux        - The number of auxiliary physical fields
. feAux        - The PetscFE objects for each auxiliary field
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. f0_func      - f_0 function from the first order FEM model
- f1_func      - f_1 function from the first order FEM model

  Output Parameter
. elemVec      - the element residual vectors from each element

   Calling sequence of f0_func and f1_func:
$    void f0(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)
*/
PetscErrorCode PetscFEIntegrateResidual(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                        PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                        void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]),
                                        void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]),
                                        PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integrateresidual) {ierr = (*fem->ops->integrateresidual)(fem, Ne, Nf, fe, field, geom, coefficients, NfAux, feAux, coefficientsAux, f0_func, f1_func, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateBdResidual"
/*C
  PetscFEIntegrateBdResidual - Produce the element residual vector for a chunk of elements by quadrature integration over a boundary

  Not collective

  Input Parameters:
+ fem          - The PetscFE object for the field being integrated
. Ne           - The number of elements in the chunk
. Nf           - The number of physical fields
. fe           - The PetscFE objects for each field
. field        - The field being integrated
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. NfAux        - The number of auxiliary physical fields
. feAux        - The PetscFE objects for each auxiliary field
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. f0_func      - f_0 function from the first order FEM model
- f1_func      - f_1 function from the first order FEM model

  Output Parameter
. elemVec      - the element residual vectors from each element

   Calling sequence of f0_func and f1_func:
$    void f0(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)
*/
PetscErrorCode PetscFEIntegrateBdResidual(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                          PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                          void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]),
                                          void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]),
                                          PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratebdresidual) {ierr = (*fem->ops->integratebdresidual)(fem, Ne, Nf, fe, field, geom, coefficients, NfAux, feAux, coefficientsAux, f0_func, f1_func, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobianAction"
/*C
  PetscFEIntegrateJacobianAction - Produce the action of the element Jacobian on an element vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          = The PetscFE object for the field being integrated
. Ne           - The number of elements in the chunk
. Nf           - The number of physical fields
. fe           - The PetscFE objects for each field
. field        - The test field being integrated
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. input        - The array of FEM basis coefficients for the elements for the input vector
. g0_func      - g_0 function from the first order FEM model
. g1_func      - g_1 function from the first order FEM model
. g2_func      - g_2 function from the first order FEM model
- g3_func      - g_3 function from the first order FEM model

  Output Parameter
. elemVec      - the element vector for the action from each element

   Calling sequence of g0_func, g1_func, g2_func and g3_func:
$    void g0(PetscScalar u[], const PetscScalar gradU[], PetscScalar a[], const PetscScalar gradA[], PetscScalar x[], PetscScalar g0[])

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
PetscErrorCode PetscFEIntegrateJacobianAction(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[], const PetscScalar input[],
                                              void (**g0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]),
                                              void (**g1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]),
                                              void (**g2_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]),
                                              void (**g3_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]),
                                              PetscScalar elemVec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratejacobianaction) {ierr = (*fem->ops->integratejacobianaction)(fem, Ne, Nf, fe, field, geom, coefficients, input, g0_func, g1_func, g2_func, g3_func, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateJacobian"
/*C
  PetscFEIntegrateJacobian - Produce the element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ fem          = The PetscFE object for the field being integrated
. Ne           - The number of elements in the chunk
. Nf           - The number of physical fields
. fe           - The PetscFE objects for each field
. fieldI       - The test field being integrated
. fieldJ       - The basis field being integrated
. geom         - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. NfAux        - The number of auxiliary physical fields
. feAux        - The PetscFE objects for each auxiliary field
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. g0_func      - g_0 function from the first order FEM model
. g1_func      - g_1 function from the first order FEM model
. g2_func      - g_2 function from the first order FEM model
- g3_func      - g_3 function from the first order FEM model

  Output Parameter
. elemMat              - the element matrices for the Jacobian from each element

   Calling sequence of g0_func, g1_func, g2_func and g3_func:
$    void g0(PetscScalar u[], const PetscScalar gradU[], PetscScalar a[], const PetscScalar gradA[], PetscScalar x[], PetscScalar g0[])

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
PetscErrorCode PetscFEIntegrateJacobian(PetscFE fem, PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt fieldI, PetscInt fieldJ, PetscCellGeometry geom, const PetscScalar coefficients[],
                                        PetscInt NfAux, PetscFE feAux[], const PetscScalar coefficientsAux[],
                                        void (*g0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]),
                                        void (*g1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]),
                                        void (*g2_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]),
                                        void (*g3_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]),
                                        PetscScalar elemMat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->ops->integratejacobian) {ierr = (*fem->ops->integratejacobian)(fem, Ne, Nf, fe, fieldI, fieldJ, geom, coefficients, NfAux, feAux, coefficientsAux, g0_func, g1_func, g2_func, g3_func, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
