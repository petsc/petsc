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
#include <petscdmplex.h>

PetscBool  FEcite       = PETSC_FALSE;
const char FECitation[] = "@article{kirby2004,\n"
                          "  title   = {Algorithm 839: FIAT, a New Paradigm for Computing Finite Element Basis Functions},\n"
                          "  journal = {ACM Transactions on Mathematical Software},\n"
                          "  author  = {Robert C. Kirby},\n"
                          "  volume  = {30},\n"
                          "  number  = {4},\n"
                          "  pages   = {502--516},\n"
                          "  doi     = {10.1145/1039813.1039820},\n"
                          "  year    = {2004}\n}\n";

PetscClassId PETSCFE_CLASSID = 0;

PetscLogEvent PETSCFE_SetUp;

PetscFunctionList PetscFEList              = NULL;
PetscBool         PetscFERegisterAllCalled = PETSC_FALSE;

/*@C
  PetscFERegister - Adds a new `PetscFEType`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - The name of a new user-defined creation routine
- function - The creation routine

  Example Usage:
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

  Note:
  `PetscFERegister()` may be called multiple times to add several user-defined `PetscFE`s

.seealso: `PetscFE`, `PetscFEType`, `PetscFERegisterAll()`, `PetscFERegisterDestroy()`
@*/
PetscErrorCode PetscFERegister(const char sname[], PetscErrorCode (*function)(PetscFE))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscFEList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetType - Builds a particular `PetscFE`

  Collective

  Input Parameters:
+ fem  - The `PetscFE` object
- name - The kind of FEM space

  Options Database Key:
. -petscfe_type <type> - Sets the `PetscFE` type; use -help for a list of available types

  Level: intermediate

.seealso: `PetscFEType`, `PetscFE`, `PetscFEGetType()`, `PetscFECreate()`
@*/
PetscErrorCode PetscFESetType(PetscFE fem, PetscFEType name)
{
  PetscErrorCode (*r)(PetscFE);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)fem, name, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  if (!PetscFERegisterAllCalled) PetscCall(PetscFERegisterAll());
  PetscCall(PetscFunctionListFind(PetscFEList, name, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)fem), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscFE type: %s", name);

  PetscTryTypeMethod(fem, destroy);
  fem->ops->destroy = NULL;

  PetscCall((*r)(fem));
  PetscCall(PetscObjectChangeTypeName((PetscObject)fem, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetType - Gets the `PetscFEType` (as a string) from the `PetscFE` object.

  Not Collective

  Input Parameter:
. fem - The `PetscFE`

  Output Parameter:
. name - The `PetscFEType` name

  Level: intermediate

.seealso: `PetscFEType`, `PetscFE`, `PetscFESetType()`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetType(PetscFE fem, PetscFEType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(name, 2);
  if (!PetscFERegisterAllCalled) PetscCall(PetscFERegisterAll());
  *name = ((PetscObject)fem)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEViewFromOptions - View from a `PetscFE` based on values in the options database

  Collective

  Input Parameters:
+ A    - the `PetscFE` object
. obj  - Optional object that provides the options prefix, pass `NULL` to use the options prefix of `A`
- name - command line option name

  Level: intermediate

.seealso: `PetscFE`, `PetscFEView()`, `PetscObjectViewFromOptions()`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEViewFromOptions(PetscFE A, PeOp PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCFE_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEView - Views a `PetscFE`

  Collective

  Input Parameters:
+ fem    - the `PetscFE` object to view
- viewer - the viewer

  Level: beginner

.seealso: `PetscFE`, `PetscViewer`, `PetscFEDestroy()`, `PetscFEViewFromOptions()`
@*/
PetscErrorCode PetscFEView(PetscFE fem, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)fem), &viewer));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)fem, viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscTryTypeMethod(fem, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetFromOptions - sets parameters in a `PetscFE` from the options database

  Collective

  Input Parameter:
. fem - the `PetscFE` object to set options for

  Options Database Keys:
+ -petscfe_num_blocks  - the number of cell blocks to integrate concurrently
- -petscfe_num_batches - the number of cell batches to integrate serially

  Level: intermediate

.seealso: `PetscFE`, `PetscFEView()`
@*/
PetscErrorCode PetscFESetFromOptions(PetscFE fem)
{
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (!((PetscObject)fem)->type_name) {
    defaultType = PETSCFEBASIC;
  } else {
    defaultType = ((PetscObject)fem)->type_name;
  }
  if (!PetscFERegisterAllCalled) PetscCall(PetscFERegisterAll());

  PetscObjectOptionsBegin((PetscObject)fem);
  PetscCall(PetscOptionsFList("-petscfe_type", "Finite element space", "PetscFESetType", PetscFEList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(PetscFESetType(fem, name));
  } else if (!((PetscObject)fem)->type_name) {
    PetscCall(PetscFESetType(fem, defaultType));
  }
  PetscCall(PetscOptionsBoundedInt("-petscfe_num_blocks", "The number of cell blocks to integrate concurrently", "PetscSpaceSetTileSizes", fem->numBlocks, &fem->numBlocks, NULL, 1));
  PetscCall(PetscOptionsBoundedInt("-petscfe_num_batches", "The number of cell batches to integrate serially", "PetscSpaceSetTileSizes", fem->numBatches, &fem->numBatches, NULL, 1));
  PetscTryTypeMethod(fem, setfromoptions, PetscOptionsObject);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)fem, PetscOptionsObject));
  PetscOptionsEnd();
  PetscCall(PetscFEViewFromOptions(fem, NULL, "-petscfe_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetUp - Construct data structures for the `PetscFE` after the `PetscFEType` has been set

  Collective

  Input Parameter:
. fem - the `PetscFE` object to setup

  Level: intermediate

.seealso: `PetscFE`, `PetscFEView()`, `PetscFEDestroy()`
@*/
PetscErrorCode PetscFESetUp(PetscFE fem)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PETSCFE_SetUp, fem, 0, 0, 0));
  fem->setupcalled = PETSC_TRUE;
  PetscTryTypeMethod(fem, setup);
  PetscCall(PetscLogEventEnd(PETSCFE_SetUp, fem, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEDestroy - Destroys a `PetscFE` object

  Collective

  Input Parameter:
. fem - the `PetscFE` object to destroy

  Level: beginner

.seealso: `PetscFE`, `PetscFEView()`
@*/
PetscErrorCode PetscFEDestroy(PetscFE *fem)
{
  PetscFunctionBegin;
  if (!*fem) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*fem, PETSCFE_CLASSID, 1);

  if (--((PetscObject)*fem)->refct > 0) {
    *fem = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ((PetscObject)*fem)->refct = 0;

  if ((*fem)->subspaces) {
    PetscInt dim, d;

    PetscCall(PetscDualSpaceGetDimension((*fem)->dualSpace, &dim));
    for (d = 0; d < dim; ++d) PetscCall(PetscFEDestroy(&(*fem)->subspaces[d]));
  }
  PetscCall(PetscFree((*fem)->subspaces));
  PetscCall(PetscFree((*fem)->invV));
  PetscCall(PetscTabulationDestroy(&(*fem)->T));
  PetscCall(PetscTabulationDestroy(&(*fem)->Tf));
  PetscCall(PetscTabulationDestroy(&(*fem)->Tc));
  PetscCall(PetscSpaceDestroy(&(*fem)->basisSpace));
  PetscCall(PetscDualSpaceDestroy(&(*fem)->dualSpace));
  PetscCall(PetscQuadratureDestroy(&(*fem)->quadrature));
  PetscCall(PetscQuadratureDestroy(&(*fem)->faceQuadrature));
#ifdef PETSC_HAVE_LIBCEED
  PetscCallCEED(CeedBasisDestroy(&(*fem)->ceedBasis));
  PetscCallCEED(CeedDestroy(&(*fem)->ceed));
#endif

  PetscTryTypeMethod(*fem, destroy);
  PetscCall(PetscHeaderDestroy(fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreate - Creates an empty `PetscFE` object. The type can then be set with `PetscFESetType()`.

  Collective

  Input Parameter:
. comm - The communicator for the `PetscFE` object

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

.seealso: `PetscFE`, `PetscFEType`, `PetscFESetType()`, `PetscFECreateDefault()`, `PETSCFEGALERKIN`
@*/
PetscErrorCode PetscFECreate(MPI_Comm comm, PetscFE *fem)
{
  PetscFE f;

  PetscFunctionBegin;
  PetscAssertPointer(fem, 2);
  PetscCall(PetscCitationsRegister(FECitation, &FEcite));
  PetscCall(PetscFEInitializePackage());

  PetscCall(PetscHeaderCreate(f, PETSCFE_CLASSID, "PetscFE", "Finite Element", "PetscFE", comm, PetscFEDestroy, PetscFEView));

  f->basisSpace    = NULL;
  f->dualSpace     = NULL;
  f->numComponents = 1;
  f->subspaces     = NULL;
  f->invV          = NULL;
  f->T             = NULL;
  f->Tf            = NULL;
  f->Tc            = NULL;
  PetscCall(PetscArrayzero(&f->quadrature, 1));
  PetscCall(PetscArrayzero(&f->faceQuadrature, 1));
  f->blockSize  = 0;
  f->numBlocks  = 1;
  f->batchSize  = 0;
  f->numBatches = 1;

  *fem = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetSpatialDimension - Returns the spatial dimension of the element

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. dim - The spatial dimension

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetSpatialDimension(PetscFE fem, PetscInt *dim)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(dim, 2);
  PetscCall(PetscDualSpaceGetDM(fem->dualSpace, &dm));
  PetscCall(DMGetDimension(dm, dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetNumComponents - Sets the number of field components in the element

  Not Collective

  Input Parameters:
+ fem  - The `PetscFE` object
- comp - The number of field components

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`, `PetscFEGetSpatialDimension()`, `PetscFEGetNumComponents()`
@*/
PetscErrorCode PetscFESetNumComponents(PetscFE fem, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->numComponents = comp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetNumComponents - Returns the number of components in the element

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. comp - The number of field components

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`, `PetscFEGetSpatialDimension()`
@*/
PetscErrorCode PetscFEGetNumComponents(PetscFE fem, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(comp, 2);
  *comp = fem->numComponents;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetTileSizes - Sets the tile sizes for evaluation

  Not Collective

  Input Parameters:
+ fem        - The `PetscFE` object
. blockSize  - The number of elements in a block
. numBlocks  - The number of blocks in a batch
. batchSize  - The number of elements in a batch
- numBatches - The number of batches in a chunk

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`, `PetscFEGetTileSizes()`
@*/
PetscErrorCode PetscFESetTileSizes(PetscFE fem, PetscInt blockSize, PetscInt numBlocks, PetscInt batchSize, PetscInt numBatches)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->blockSize  = blockSize;
  fem->numBlocks  = numBlocks;
  fem->batchSize  = batchSize;
  fem->numBatches = numBatches;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetTileSizes - Returns the tile sizes for evaluation

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameters:
+ blockSize  - The number of elements in a block, pass `NULL` if not needed
. numBlocks  - The number of blocks in a batch, pass `NULL` if not needed
. batchSize  - The number of elements in a batch, pass `NULL` if not needed
- numBatches - The number of batches in a chunk, pass `NULL` if not needed

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`, `PetscFESetTileSizes()`
@*/
PetscErrorCode PetscFEGetTileSizes(PetscFE fem, PeOp PetscInt *blockSize, PeOp PetscInt *numBlocks, PeOp PetscInt *batchSize, PeOp PetscInt *numBatches)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (blockSize) PetscAssertPointer(blockSize, 2);
  if (numBlocks) PetscAssertPointer(numBlocks, 3);
  if (batchSize) PetscAssertPointer(batchSize, 4);
  if (numBatches) PetscAssertPointer(numBatches, 5);
  if (blockSize) *blockSize = fem->blockSize;
  if (numBlocks) *numBlocks = fem->numBlocks;
  if (batchSize) *batchSize = fem->batchSize;
  if (numBatches) *numBatches = fem->numBatches;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetBasisSpace - Returns the `PetscSpace` used for the approximation of the solution for the `PetscFE`

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. sp - The `PetscSpace` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetBasisSpace(PetscFE fem, PetscSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(sp, 2);
  *sp = fem->basisSpace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetBasisSpace - Sets the `PetscSpace` used for the approximation of the solution

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- sp  - The `PetscSpace` object

  Level: intermediate

  Developer Notes:
  There is `PetscFESetBasisSpace()` but the `PetscFESetDualSpace()`, likely the Basis is unneeded in the function name

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscFECreate()`, `PetscFESetDualSpace()`
@*/
PetscErrorCode PetscFESetBasisSpace(PetscFE fem, PetscSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 2);
  PetscCall(PetscSpaceDestroy(&fem->basisSpace));
  fem->basisSpace = sp;
  PetscCall(PetscObjectReference((PetscObject)fem->basisSpace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetDualSpace - Returns the `PetscDualSpace` used to define the inner product for a `PetscFE`

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. sp - The `PetscDualSpace` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetDualSpace(PetscFE fem, PetscDualSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(sp, 2);
  *sp = fem->dualSpace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetDualSpace - Sets the `PetscDualSpace` used to define the inner product

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- sp  - The `PetscDualSpace` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscFECreate()`, `PetscFESetBasisSpace()`
@*/
PetscErrorCode PetscFESetDualSpace(PetscFE fem, PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  PetscCall(PetscDualSpaceDestroy(&fem->dualSpace));
  fem->dualSpace = sp;
  PetscCall(PetscObjectReference((PetscObject)fem->dualSpace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetQuadrature - Returns the `PetscQuadrature` used to calculate inner products

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. q - The `PetscQuadrature` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(q, 2);
  *q = fem->quadrature;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetQuadrature - Sets the `PetscQuadrature` used to calculate inner products

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- q   - The `PetscQuadrature` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`, `PetscFECreate()`, `PetscFEGetFaceQuadrature()`
@*/
PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscInt Nc, qNc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (q == fem->quadrature) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFEGetNumComponents(fem, &Nc));
  PetscCall(PetscQuadratureGetNumComponents(q, &qNc));
  PetscCheck(!(qNc != 1) || !(Nc != qNc), PetscObjectComm((PetscObject)fem), PETSC_ERR_ARG_SIZ, "FE components %" PetscInt_FMT " != Quadrature components %" PetscInt_FMT " and non-scalar quadrature", Nc, qNc);
  PetscCall(PetscTabulationDestroy(&fem->T));
  PetscCall(PetscTabulationDestroy(&fem->Tc));
  PetscCall(PetscObjectReference((PetscObject)q));
  PetscCall(PetscQuadratureDestroy(&fem->quadrature));
  fem->quadrature = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetFaceQuadrature - Returns the `PetscQuadrature` used to calculate inner products on faces

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. q - The `PetscQuadrature` object

  Level: intermediate

  Developer Notes:
  There is a special face quadrature but not edge, likely this API would benefit from a refactorization

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`, `PetscFECreate()`, `PetscFESetQuadrature()`, `PetscFESetFaceQuadrature()`
@*/
PetscErrorCode PetscFEGetFaceQuadrature(PetscFE fem, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(q, 2);
  *q = fem->faceQuadrature;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetFaceQuadrature - Sets the `PetscQuadrature` used to calculate inner products on faces

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- q   - The `PetscQuadrature` object

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`, `PetscFECreate()`, `PetscFESetQuadrature()`
@*/
PetscErrorCode PetscFESetFaceQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscInt Nc, qNc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (q == fem->faceQuadrature) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFEGetNumComponents(fem, &Nc));
  PetscCall(PetscQuadratureGetNumComponents(q, &qNc));
  PetscCheck(!(qNc != 1) || !(Nc != qNc), PetscObjectComm((PetscObject)fem), PETSC_ERR_ARG_SIZ, "FE components %" PetscInt_FMT " != Quadrature components %" PetscInt_FMT " and non-scalar quadrature", Nc, qNc);
  PetscCall(PetscTabulationDestroy(&fem->Tf));
  PetscCall(PetscObjectReference((PetscObject)q));
  PetscCall(PetscQuadratureDestroy(&fem->faceQuadrature));
  fem->faceQuadrature = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECopyQuadrature - Copy both volumetric and surface quadrature to a new `PetscFE`

  Not Collective

  Input Parameters:
+ sfe - The `PetscFE` source for the quadratures
- tfe - The `PetscFE` target for the quadratures

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`, `PetscFECreate()`, `PetscFESetQuadrature()`, `PetscFESetFaceQuadrature()`
@*/
PetscErrorCode PetscFECopyQuadrature(PetscFE sfe, PetscFE tfe)
{
  PetscQuadrature q;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfe, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(tfe, PETSCFE_CLASSID, 2);
  PetscCall(PetscFEGetQuadrature(sfe, &q));
  PetscCall(PetscFESetQuadrature(tfe, q));
  PetscCall(PetscFEGetFaceQuadrature(sfe, &q));
  PetscCall(PetscFESetFaceQuadrature(tfe, q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGetNumDof - Returns the number of dofs (dual basis vectors) associated to mesh points on the reference cell of a given dimension

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. numDof - Array of length `dim` with the number of dofs in each dimension

  Level: intermediate

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscFECreate()`
@*/
PetscErrorCode PetscFEGetNumDof(PetscFE fem, const PetscInt *numDof[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(numDof, 2);
  PetscCall(PetscDualSpaceGetNumDof(fem->dualSpace, numDof));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGetCellTabulation - Returns the tabulation of the basis functions at the quadrature points on the reference cell

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- k   - The highest derivative we need to tabulate, very often 1

  Output Parameter:
. T - The basis function values and derivatives at quadrature points

  Level: intermediate

  Note:
.vb
  T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
  T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
  T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e
.ve

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscTabulation`, `PetscFECreateTabulation()`, `PetscTabulationDestroy()`
@*/
PetscErrorCode PetscFEGetCellTabulation(PetscFE fem, PetscInt k, PetscTabulation *T)
{
  PetscInt         npoints;
  const PetscReal *points;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(T, 3);
  PetscCall(PetscQuadratureGetData(fem->quadrature, NULL, NULL, &npoints, &points, NULL));
  if (!fem->T) PetscCall(PetscFECreateTabulation(fem, 1, npoints, points, k, &fem->T));
  PetscCheck(!fem->T || k <= fem->T->K || (!fem->T->cdim && !fem->T->K), PetscObjectComm((PetscObject)fem), PETSC_ERR_ARG_OUTOFRANGE, "Requested %" PetscInt_FMT " derivatives, but only tabulated %" PetscInt_FMT, k, fem->T->K);
  *T = fem->T;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFEExpandFaceQuadrature(PetscFE fe, PetscQuadrature fq, PetscQuadrature *efq)
{
  DM               dm;
  PetscDualSpace   sp;
  const PetscInt  *faces;
  const PetscReal *points, *weights;
  DMPolytopeType   ct;
  PetscReal       *facePoints, *faceWeights;
  PetscInt         dim, cStart, Nf, Nc, Np, order;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDualSpace(fe, &sp));
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetConeSize(dm, cStart, &Nf));
  PetscCall(DMPlexGetCone(dm, cStart, &faces));
  PetscCall(PetscQuadratureGetData(fq, NULL, &Nc, &Np, &points, &weights));
  PetscCall(PetscMalloc1(Nf * Np * dim, &facePoints));
  PetscCall(PetscMalloc1(Nf * Np * Nc, &faceWeights));
  for (PetscInt f = 0; f < Nf; ++f) {
    const PetscReal xi0[3] = {-1., -1., -1.};
    PetscReal       v0[3], J[9], detJ;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, faces[f], NULL, v0, J, NULL, &detJ));
    for (PetscInt q = 0; q < Np; ++q) {
      CoordinatesRefToReal(dim, dim - 1, xi0, v0, J, &points[q * (dim - 1)], &facePoints[(f * Np + q) * dim]);
      for (PetscInt c = 0; c < Nc; ++c) faceWeights[(f * Np + q) * Nc + c] = weights[q * Nc + c];
    }
  }
  PetscCall(PetscQuadratureCreate(PetscObjectComm((PetscObject)fq), efq));
  PetscCall(PetscQuadratureGetCellType(fq, &ct));
  PetscCall(PetscQuadratureSetCellType(*efq, ct));
  PetscCall(PetscQuadratureGetOrder(fq, &order));
  PetscCall(PetscQuadratureSetOrder(*efq, order));
  PetscCall(PetscQuadratureSetData(*efq, dim, Nc, Nf * Np, facePoints, faceWeights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGetFaceTabulation - Returns the tabulation of the basis functions at the face quadrature points for each face of the reference cell

  Not Collective

  Input Parameters:
+ fem - The `PetscFE` object
- k   - The highest derivative we need to tabulate, very often 1

  Output Parameter:
. Tf - The basis function values and derivatives at face quadrature points

  Level: intermediate

  Note:
.vb
  T->T[0] = Bf[((f*Nq + q)*pdim + i)*Nc + c] is the value at point f,q for basis function i and component c
  T->T[1] = Df[(((f*Nq + q)*pdim + i)*Nc + c)*dim + d] is the derivative value at point f,q for basis function i, component c, in direction d
  T->T[2] = Hf[((((f*Nq + q)*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point f,q for basis function i, component c, in directions d and e
.ve

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscTabulation`, `PetscFEGetCellTabulation()`, `PetscFECreateTabulation()`, `PetscTabulationDestroy()`
@*/
PetscErrorCode PetscFEGetFaceTabulation(PetscFE fem, PetscInt k, PetscTabulation *Tf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(Tf, 3);
  if (!fem->Tf) {
    PetscQuadrature fq;

    PetscCall(PetscFEGetFaceQuadrature(fem, &fq));
    if (fq) {
      PetscQuadrature  efq;
      const PetscReal *facePoints;
      PetscInt         Np, eNp;

      PetscCall(PetscFEExpandFaceQuadrature(fem, fq, &efq));
      PetscCall(PetscQuadratureGetData(fq, NULL, NULL, &Np, NULL, NULL));
      PetscCall(PetscQuadratureGetData(efq, NULL, NULL, &eNp, &facePoints, NULL));
      if (PetscDefined(USE_DEBUG)) {
        PetscDualSpace sp;
        DM             dm;
        PetscInt       cStart, Nf;

        PetscCall(PetscFEGetDualSpace(fem, &sp));
        PetscCall(PetscDualSpaceGetDM(sp, &dm));
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
        PetscCall(DMPlexGetConeSize(dm, cStart, &Nf));
        PetscCheck(Nf == eNp / Np, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of faces %" PetscInt_FMT " != %" PetscInt_FMT " number of quadrature replicas", Nf, eNp / Np);
      }
      PetscCall(PetscFECreateTabulation(fem, eNp / Np, Np, facePoints, k, &fem->Tf));
      PetscCall(PetscQuadratureDestroy(&efq));
    }
  }
  PetscCheck(!fem->Tf || k <= fem->Tf->K, PetscObjectComm((PetscObject)fem), PETSC_ERR_ARG_OUTOFRANGE, "Requested %" PetscInt_FMT " derivatives, but only tabulated %" PetscInt_FMT, k, fem->Tf->K);
  *Tf = fem->Tf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGetFaceCentroidTabulation - Returns the tabulation of the basis functions at the face centroid points

  Not Collective

  Input Parameter:
. fem - The `PetscFE` object

  Output Parameter:
. Tc - The basis function values at face centroid points

  Level: intermediate

  Note:
.vb
  T->T[0] = Bf[(f*pdim + i)*Nc + c] is the value at point f for basis function i and component c
.ve

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscTabulation`, `PetscFEGetFaceTabulation()`, `PetscFEGetCellTabulation()`, `PetscFECreateTabulation()`, `PetscTabulationDestroy()`
@*/
PetscErrorCode PetscFEGetFaceCentroidTabulation(PetscFE fem, PetscTabulation *Tc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(Tc, 2);
  if (!fem->Tc) {
    PetscDualSpace  sp;
    DM              dm;
    const PetscInt *cone;
    PetscReal      *centroids;
    PetscInt        dim, numFaces, f;

    PetscCall(PetscFEGetDualSpace(fem, &sp));
    PetscCall(PetscDualSpaceGetDM(sp, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetConeSize(dm, 0, &numFaces));
    PetscCall(DMPlexGetCone(dm, 0, &cone));
    PetscCall(PetscMalloc1(numFaces * dim, &centroids));
    for (f = 0; f < numFaces; ++f) PetscCall(DMPlexComputeCellGeometryFVM(dm, cone[f], NULL, &centroids[f * dim], NULL));
    PetscCall(PetscFECreateTabulation(fem, 1, numFaces, centroids, 0, &fem->Tc));
    PetscCall(PetscFree(centroids));
  }
  *Tc = fem->Tc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFECreateTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not Collective

  Input Parameters:
+ fem     - The `PetscFE` object
. nrepl   - The number of replicas
. npoints - The number of tabulation points in a replica
. points  - The tabulation point coordinates
- K       - The number of derivatives calculated

  Output Parameter:
. T - The basis function values and derivatives at tabulation points

  Level: intermediate

  Note:
.vb
  T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
  T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
  T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis
  T->function i, component c, in directions d and e
.ve

.seealso: `PetscTabulation`, `PetscFEGetCellTabulation()`, `PetscTabulationDestroy()`
@*/
PetscErrorCode PetscFECreateTabulation(PetscFE fem, PetscInt nrepl, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation *T)
{
  DM             dm;
  PetscDualSpace Q;
  PetscInt       Nb;   /* Dimension of FE space P */
  PetscInt       Nc;   /* Field components */
  PetscInt       cdim; /* Reference coordinate dimension */
  PetscInt       k;

  PetscFunctionBegin;
  if (!npoints || !fem->dualSpace || K < 0) {
    *T = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(points, 4);
  PetscAssertPointer(T, 6);
  PetscCall(PetscFEGetDualSpace(fem, &Q));
  PetscCall(PetscDualSpaceGetDM(Q, &dm));
  PetscCall(DMGetDimension(dm, &cdim));
  PetscCall(PetscDualSpaceGetDimension(Q, &Nb));
  PetscCall(PetscFEGetNumComponents(fem, &Nc));
  PetscCall(PetscMalloc1(1, T));
  (*T)->K    = !cdim ? 0 : K;
  (*T)->Nr   = nrepl;
  (*T)->Np   = npoints;
  (*T)->Nb   = Nb;
  (*T)->Nc   = Nc;
  (*T)->cdim = cdim;
  PetscCall(PetscMalloc1((*T)->K + 1, &(*T)->T));
  for (k = 0; k <= (*T)->K; ++k) PetscCall(PetscCalloc1(nrepl * npoints * Nb * Nc * PetscPowInt(cdim, k), &(*T)->T[k]));
  PetscUseTypeMethod(fem, computetabulation, nrepl * npoints, points, K, *T);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEComputeTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not Collective

  Input Parameters:
+ fem     - The `PetscFE` object
. npoints - The number of tabulation points
. points  - The tabulation point coordinates
. K       - The number of derivatives calculated
- T       - An existing tabulation object with enough allocated space

  Output Parameter:
. T - The basis function values and derivatives at tabulation points

  Level: intermediate

  Note:
.vb
  T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
  T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
  T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e
.ve

.seealso: `PetscTabulation`, `PetscFEGetCellTabulation()`, `PetscTabulationDestroy()`
@*/
PetscErrorCode PetscFEComputeTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  PetscFunctionBeginHot;
  if (!npoints || !fem->dualSpace || K < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(points, 3);
  PetscAssertPointer(T, 5);
  if (PetscDefined(USE_DEBUG)) {
    DM             dm;
    PetscDualSpace Q;
    PetscInt       Nb;   /* Dimension of FE space P */
    PetscInt       Nc;   /* Field components */
    PetscInt       cdim; /* Reference coordinate dimension */

    PetscCall(PetscFEGetDualSpace(fem, &Q));
    PetscCall(PetscDualSpaceGetDM(Q, &dm));
    PetscCall(DMGetDimension(dm, &cdim));
    PetscCall(PetscDualSpaceGetDimension(Q, &Nb));
    PetscCall(PetscFEGetNumComponents(fem, &Nc));
    PetscCheck(T->K == (!cdim ? 0 : K), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation K %" PetscInt_FMT " must match requested K %" PetscInt_FMT, T->K, !cdim ? 0 : K);
    PetscCheck(T->Nb == Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation Nb %" PetscInt_FMT " must match requested Nb %" PetscInt_FMT, T->Nb, Nb);
    PetscCheck(T->Nc == Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation Nc %" PetscInt_FMT " must match requested Nc %" PetscInt_FMT, T->Nc, Nc);
    PetscCheck(T->cdim == cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation cdim %" PetscInt_FMT " must match requested cdim %" PetscInt_FMT, T->cdim, cdim);
  }
  T->Nr = 1;
  T->Np = npoints;
  PetscUseTypeMethod(fem, computetabulation, npoints, points, K, T);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscTabulationDestroy - Frees memory from the associated tabulation.

  Not Collective

  Input Parameter:
. T - The tabulation

  Level: intermediate

.seealso: `PetscTabulation`, `PetscFECreateTabulation()`, `PetscFEGetCellTabulation()`
@*/
PetscErrorCode PetscTabulationDestroy(PetscTabulation *T)
{
  PetscInt k;

  PetscFunctionBegin;
  PetscAssertPointer(T, 1);
  if (!T || !*T) PetscFunctionReturn(PETSC_SUCCESS);
  for (k = 0; k <= (*T)->K; ++k) PetscCall(PetscFree((*T)->T[k]));
  PetscCall(PetscFree((*T)->T));
  PetscCall(PetscFree(*T));
  *T = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFECreatePointTraceDefault_Internal(PetscFE fe, PetscInt refPoint, PetscFE *trFE)
{
  PetscSpace      bsp, bsubsp;
  PetscDualSpace  dsp, dsubsp;
  PetscInt        dim, depth, numComp, i, j, coneSize, order;
  DM              dm;
  DMLabel         label;
  PetscReal      *xi, *v, *J, detJ;
  const char     *name;
  PetscQuadrature origin, fullQuad, subQuad;

  PetscFunctionBegin;
  PetscCall(PetscFEGetBasisSpace(fe, &bsp));
  PetscCall(PetscFEGetDualSpace(fe, &dsp));
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthLabel(dm, &label));
  PetscCall(DMLabelGetValue(label, refPoint, &depth));
  PetscCall(PetscCalloc1(depth, &xi));
  PetscCall(PetscMalloc1(dim, &v));
  PetscCall(PetscMalloc1(dim * dim, &J));
  for (i = 0; i < depth; i++) xi[i] = 0.;
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &origin));
  PetscCall(PetscQuadratureSetData(origin, depth, 0, 1, xi, NULL));
  PetscCall(DMPlexComputeCellGeometryFEM(dm, refPoint, origin, v, J, NULL, &detJ));
  /* CellGeometryFEM computes the expanded Jacobian, we want the true jacobian */
  for (i = 1; i < dim; i++) {
    for (j = 0; j < depth; j++) J[i * depth + j] = J[i * dim + j];
  }
  PetscCall(PetscQuadratureDestroy(&origin));
  PetscCall(PetscDualSpaceGetPointSubspace(dsp, refPoint, &dsubsp));
  PetscCall(PetscSpaceCreateSubspace(bsp, dsubsp, v, J, NULL, NULL, PETSC_OWN_POINTER, &bsubsp));
  PetscCall(PetscSpaceSetUp(bsubsp));
  PetscCall(PetscFECreate(PetscObjectComm((PetscObject)fe), trFE));
  PetscCall(PetscFESetType(*trFE, PETSCFEBASIC));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCall(PetscFESetNumComponents(*trFE, numComp));
  PetscCall(PetscFESetBasisSpace(*trFE, bsubsp));
  PetscCall(PetscFESetDualSpace(*trFE, dsubsp));
  PetscCall(PetscObjectGetName((PetscObject)fe, &name));
  if (name) PetscCall(PetscFESetName(*trFE, name));
  PetscCall(PetscFEGetQuadrature(fe, &fullQuad));
  PetscCall(PetscQuadratureGetOrder(fullQuad, &order));
  PetscCall(DMPlexGetConeSize(dm, refPoint, &coneSize));
  if (coneSize == 2 * depth) PetscCall(PetscDTGaussTensorQuadrature(depth, 1, (order + 2) / 2, -1., 1., &subQuad));
  else PetscCall(PetscDTSimplexQuadrature(depth, order, PETSCDTSIMPLEXQUAD_DEFAULT, &subQuad));
  PetscCall(PetscFESetQuadrature(*trFE, subQuad));
  PetscCall(PetscFESetUp(*trFE));
  PetscCall(PetscQuadratureDestroy(&subQuad));
  PetscCall(PetscSpaceDestroy(&bsubsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscFECreatePointTrace(PetscFE fe, PetscInt refPoint, PetscFE *trFE)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscAssertPointer(trFE, 3);
  if (fe->ops->createpointtrace) PetscUseTypeMethod(fe, createpointtrace, refPoint, trFE);
  else PetscCall(PetscFECreatePointTraceDefault_Internal(fe, refPoint, trFE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFECreateHeightTrace(PetscFE fe, PetscInt height, PetscFE *trFE)
{
  PetscInt       hStart, hEnd;
  PetscDualSpace dsp;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscAssertPointer(trFE, 3);
  *trFE = NULL;
  PetscCall(PetscFEGetDualSpace(fe, &dsp));
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, height, &hStart, &hEnd));
  if (hEnd <= hStart) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFECreatePointTrace(fe, hStart, trFE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetDimension - Get the dimension of the finite element space on a cell

  Not Collective

  Input Parameter:
. fem - The `PetscFE`

  Output Parameter:
. dim - The dimension

  Level: intermediate

.seealso: `PetscFE`, `PetscFECreate()`, `PetscSpaceGetDimension()`, `PetscDualSpaceGetDimension()`
@*/
PetscErrorCode PetscFEGetDimension(PetscFE fem, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscAssertPointer(dim, 2);
  PetscTryTypeMethod(fem, getdimension, dim);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEPushforward - Map the reference element function to real space

  Input Parameters:
+ fe     - The `PetscFE`
. fegeom - The cell geometry
. Nv     - The number of function values
- vals   - The function values

  Output Parameter:
. vals - The transformed function values

  Level: advanced

  Notes:
  This just forwards the call onto `PetscDualSpacePushforward()`.

  It only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: `PetscFE`, `PetscFEGeom`, `PetscDualSpace`, `PetscDualSpacePushforward()`
@*/
PetscErrorCode PetscFEPushforward(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscFunctionBeginHot;
  PetscCall(PetscDualSpacePushforward(fe->dualSpace, fegeom, Nv, fe->numComponents, vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEPushforwardGradient - Map the reference element function gradient to real space

  Input Parameters:
+ fe     - The `PetscFE`
. fegeom - The cell geometry
. Nv     - The number of function gradient values
- vals   - The function gradient values

  Output Parameter:
. vals - The transformed function gradient values

  Level: advanced

  Notes:
  This just forwards the call onto `PetscDualSpacePushforwardGradient()`.

  It only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: `PetscFE`, `PetscFEGeom`, `PetscDualSpace`, `PetscFEPushforward()`, `PetscDualSpacePushforwardGradient()`, `PetscDualSpacePushforward()`
@*/
PetscErrorCode PetscFEPushforwardGradient(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscFunctionBeginHot;
  PetscCall(PetscDualSpacePushforwardGradient(fe->dualSpace, fegeom, Nv, fe->numComponents, vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEPushforwardHessian - Map the reference element function Hessian to real space

  Input Parameters:
+ fe     - The `PetscFE`
. fegeom - The cell geometry
. Nv     - The number of function Hessian values
- vals   - The function Hessian values

  Output Parameter:
. vals - The transformed function Hessian values

  Level: advanced

  Notes:
  This just forwards the call onto `PetscDualSpacePushforwardHessian()`.

  It only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

  Developer Notes:
  It is unclear why all these one line convenience routines are desirable

.seealso: `PetscFE`, `PetscFEGeom`, `PetscDualSpace`, `PetscFEPushforward()`, `PetscDualSpacePushforwardHessian()`, `PetscDualSpacePushforward()`
@*/
PetscErrorCode PetscFEPushforwardHessian(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscFunctionBeginHot;
  PetscCall(PetscDualSpacePushforwardHessian(fe->dualSpace, fegeom, Nv, fe->numComponents, vals));
  PetscFunctionReturn(PETSC_SUCCESS);
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
     PetscFEGeom[Ne] possibly *Nq
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

/*@
  PetscFEIntegrate - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ prob            - The `PetscDS` specifying the discretizations and continuum functions
. field           - The field being integrated
. Ne              - The number of elements in the chunk
. cgeom           - The cell geometry for each cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter:
. integral - the integral for this field

  Level: intermediate

  Developer Notes:
  The function name begins with `PetscFE` and yet the first argument is `PetscDS` and it has no `PetscFE` arguments.

.seealso: `PetscFE`, `PetscDS`, `PetscFEIntegrateResidual()`, `PetscFEIntegrateBd()`
@*/
PetscErrorCode PetscFEIntegrate(PetscDS prob, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  PetscFE fe;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *)&fe));
  if (fe->ops->integrate) PetscCall((*fe->ops->integrate)(prob, field, Ne, cgeom, coefficients, probAux, coefficientsAux, integral));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEIntegrateBd - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ prob            - The `PetscDS` specifying the discretizations and continuum functions
. field           - The field being integrated
. obj_func        - The function to be integrated
. Ne              - The number of elements in the chunk
. geom            - The face geometry for each face in the chunk
. coefficients    - The array of FEM basis coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter:
. integral - the integral for this field

  Level: intermediate

  Developer Notes:
  The function name begins with `PetscFE` and yet the first argument is `PetscDS` and it has no `PetscFE` arguments.

.seealso: `PetscFE`, `PetscDS`, `PetscFEIntegrateResidual()`, `PetscFEIntegrate()`
@*/
PetscErrorCode PetscFEIntegrateBd(PetscDS prob, PetscInt field, void (*obj_func)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt Ne, PetscFEGeom *geom, const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  PetscFE fe;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *)&fe));
  if (fe->ops->integratebd) PetscCall((*fe->ops->integratebd)(prob, field, obj_func, Ne, geom, coefficients, probAux, coefficientsAux, integral));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateResidual - Produce the element residual vector for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ ds              - The `PetscDS` specifying the discretizations and continuum functions
. key             - The (label+value, field) being integrated
. Ne              - The number of elements in the chunk
. cgeom           - The cell geometry for each cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t               - The time

  Output Parameter:
. elemVec - the element residual vectors from each element

  Level: intermediate

  Note:
.vb
  Loop over batch of elements (e):
    Loop over quadrature points (q):
      Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
      Call f_0 and f_1
    Loop over element vector entries (f,fc --> i):
      elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)
.ve

.seealso: `PetscFEIntegrateBdResidual()`
@*/
PetscErrorCode PetscFEIntegrateResidual(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE fe;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetDiscretization(ds, key.field, (PetscObject *)&fe));
  if (fe->ops->integrateresidual) PetscCall((*fe->ops->integrateresidual)(ds, key, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateBdResidual - Produce the element residual vector for a chunk of elements by quadrature integration over a boundary

  Not Collective

  Input Parameters:
+ ds              - The `PetscDS` specifying the discretizations and continuum functions
. wf              - The PetscWeakForm object holding the pointwise functions
. key             - The (label+value, field) being integrated
. Ne              - The number of elements in the chunk
. fgeom           - The face geometry for each cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t               - The time

  Output Parameter:
. elemVec - the element residual vectors from each element

  Level: intermediate

.seealso: `PetscFEIntegrateResidual()`
@*/
PetscErrorCode PetscFEIntegrateBdResidual(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE fe;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetDiscretization(ds, key.field, (PetscObject *)&fe));
  if (fe->ops->integratebdresidual) PetscCall((*fe->ops->integratebdresidual)(ds, wf, key, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateHybridResidual - Produce the element residual vector for a chunk of hybrid element faces by quadrature integration

  Not Collective

  Input Parameters:
+ ds              - The `PetscDS` specifying the discretizations and continuum functions
. dsIn            - The `PetscDS` specifying the discretizations and continuum functions for input
. key             - The (label+value, field) being integrated
. s               - The side of the cell being integrated, 0 for negative and 1 for positive
. Ne              - The number of elements in the chunk
. fgeom           - The face geometry for each cell in the chunk
. cgeom           - The cell geometry for each neighbor cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t               - The time

  Output Parameter:
. elemVec - the element residual vectors from each element

  Level: developer

.seealso: `PetscFEIntegrateResidual()`
@*/
PetscErrorCode PetscFEIntegrateHybridResidual(PetscDS ds, PetscDS dsIn, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE fe;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(dsIn, PETSCDS_CLASSID, 2);
  PetscCall(PetscDSGetDiscretization(ds, key.field, (PetscObject *)&fe));
  if (fe->ops->integratehybridresidual) PetscCall((*fe->ops->integratehybridresidual)(ds, dsIn, key, s, Ne, fgeom, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateJacobian - Produce the element Jacobian for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ rds             - The `PetscDS` specifying the row discretizations and continuum functions
. cds             - The `PetscDS` specifying the column discretizations
. jtype           - The type of matrix pointwise functions that should be used
. key             - The (label+value, fieldI*Nf + fieldJ) being integrated
. Ne              - The number of elements in the chunk
. cgeom           - The cell geometry for each cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. dsAux           - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t               - The time
- u_tshift        - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter:
. elemMat - the element matrices for the Jacobian from each element

  Level: intermediate

  Note:
.vb
  Loop over batch of elements (e):
    Loop over element matrix entries (f,fc,g,gc --> i,j):
      Loop over quadrature points (q):
        Make u_q and gradU_q (loops over fields,Nb,Ncomp)
          elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
                       + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
.ve

.seealso: `PetscFEIntegrateResidual()`
@*/
PetscErrorCode PetscFEIntegrateJacobian(PetscDS rds, PetscDS cds, PetscFEJacobianType jtype, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE  fe;
  PetscInt Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rds, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(cds, PETSCDS_CLASSID, 2);
  PetscCall(PetscDSGetNumFields(rds, &Nf));
  PetscCall(PetscDSGetDiscretization(rds, key.field / Nf, (PetscObject *)&fe));
  if (fe->ops->integratejacobian) PetscCall((*fe->ops->integratejacobian)(rds, cds, jtype, key, Ne, cgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t, u_tshift, elemMat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateBdJacobian - Produce the boundary element Jacobian for a chunk of elements by quadrature integration

  Not Collective

  Input Parameters:
+ ds              - The `PetscDS` specifying the discretizations and continuum functions
. wf              - The PetscWeakForm holding the pointwise functions
. jtype           - The type of matrix pointwise functions that should be used
. key             - The (label+value, fieldI*Nf + fieldJ) being integrated
. Ne              - The number of elements in the chunk
. fgeom           - The face geometry for each cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t               - The time
- u_tshift        - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter:
. elemMat - the element matrices for the Jacobian from each element

  Level: intermediate

  Note:
.vb
  Loop over batch of elements (e):
    Loop over element matrix entries (f,fc,g,gc --> i,j):
      Loop over quadrature points (q):
        Make u_q and gradU_q (loops over fields,Nb,Ncomp)
          elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
                       + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
.ve

.seealso: `PetscFEIntegrateJacobian()`, `PetscFEIntegrateResidual()`
@*/
PetscErrorCode PetscFEIntegrateBdJacobian(PetscDS ds, PetscWeakForm wf, PetscFEJacobianType jtype, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE  fe;
  PetscInt Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *)&fe));
  if (fe->ops->integratebdjacobian) PetscCall((*fe->ops->integratebdjacobian)(ds, wf, jtype, key, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEIntegrateHybridJacobian - Produce the boundary element Jacobian for a chunk of hybrid elements by quadrature integration

  Not Collective

  Input Parameters:
+ ds              - The `PetscDS` specifying the discretizations and continuum functions for the output
. dsIn            - The `PetscDS` specifying the discretizations and continuum functions for the input
. jtype           - The type of matrix pointwise functions that should be used
. key             - The (label+value, fieldI*Nf + fieldJ) being integrated
. s               - The side of the cell being integrated, 0 for negative and 1 for positive
. Ne              - The number of elements in the chunk
. fgeom           - The face geometry for each cell in the chunk
. cgeom           - The cell geometry for each neighbor cell in the chunk
. coefficients    - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t  - The array of FEM basis time derivative coefficients for the elements
. probAux         - The `PetscDS` specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t               - The time
- u_tshift        - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter:
. elemMat - the element matrices for the Jacobian from each element

  Level: developer

  Note:
.vb
  Loop over batch of elements (e):
    Loop over element matrix entries (f,fc,g,gc --> i,j):
      Loop over quadrature points (q):
        Make u_q and gradU_q (loops over fields,Nb,Ncomp)
          elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
                       + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
                       + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
.ve

.seealso: `PetscFEIntegrateJacobian()`, `PetscFEIntegrateResidual()`
@*/
PetscErrorCode PetscFEIntegrateHybridJacobian(PetscDS ds, PetscDS dsIn, PetscFEJacobianType jtype, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE  fe;
  PetscInt Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *)&fe));
  if (fe->ops->integratehybridjacobian) PetscCall((*fe->ops->integratehybridjacobian)(ds, dsIn, jtype, key, s, Ne, fgeom, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFEGetHeightSubspace - Get the subspace of this space for a mesh point of a given height

  Input Parameters:
+ fe     - The finite element space
- height - The height of the `DMPLEX` point

  Output Parameter:
. subfe - The subspace of this `PetscFE` space

  Level: advanced

  Note:
  For example, if we want the subspace of this space for a face, we would choose height = 1.

.seealso: `PetscFECreateDefault()`
@*/
PetscErrorCode PetscFEGetHeightSubspace(PetscFE fe, PetscInt height, PetscFE *subfe)
{
  PetscSpace      P, subP;
  PetscDualSpace  Q, subQ;
  PetscQuadrature subq;
  PetscInt        dim, Nc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscAssertPointer(subfe, 3);
  if (height == 0) {
    *subfe = fe;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscFEGetBasisSpace(fe, &P));
  PetscCall(PetscFEGetDualSpace(fe, &Q));
  PetscCall(PetscFEGetNumComponents(fe, &Nc));
  PetscCall(PetscFEGetFaceQuadrature(fe, &subq));
  PetscCall(PetscDualSpaceGetDimension(Q, &dim));
  PetscCheck(height <= dim && height >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %" PetscInt_FMT " for dimension %" PetscInt_FMT " space", height, dim);
  if (!fe->subspaces) PetscCall(PetscCalloc1(dim, &fe->subspaces));
  if (height <= dim) {
    if (!fe->subspaces[height - 1]) {
      PetscFE     sub = NULL;
      const char *name;

      PetscCall(PetscSpaceGetHeightSubspace(P, height, &subP));
      PetscCall(PetscDualSpaceGetHeightSubspace(Q, height, &subQ));
      if (subQ) {
        PetscCall(PetscObjectReference((PetscObject)subP));
        PetscCall(PetscObjectReference((PetscObject)subQ));
        PetscCall(PetscObjectReference((PetscObject)subq));
        PetscCall(PetscFECreateFromSpaces(subP, subQ, subq, NULL, &sub));
      }
      if (sub) {
        PetscCall(PetscObjectGetName((PetscObject)fe, &name));
        if (name) PetscCall(PetscFESetName(sub, name));
      }
      fe->subspaces[height - 1] = sub;
    }
    *subfe = fe->subspaces[height - 1];
  } else {
    *subfe = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFERefine - Create a "refined" `PetscFE` object that refines the reference cell into
  smaller copies.

  Collective

  Input Parameter:
. fe - The initial `PetscFE`

  Output Parameter:
. feRef - The refined `PetscFE`

  Level: advanced

  Notes:
  This is typically used to generate a preconditioner for a higher order method from a lower order method on a
  refined mesh having the same number of dofs (but more sparsity). It is also used to create an
  interpolation between regularly refined meshes.

.seealso: `PetscFEType`, `PetscFECreate()`, `PetscFESetType()`
@*/
PetscErrorCode PetscFERefine(PetscFE fe, PetscFE *feRef)
{
  PetscSpace       P, Pref;
  PetscDualSpace   Q, Qref;
  DM               K, Kref;
  PetscQuadrature  q, qref;
  const PetscReal *v0, *jac;
  PetscInt         numComp, numSubelements;
  PetscInt         cStart, cEnd, c;
  PetscDualSpace  *cellSpaces;

  PetscFunctionBegin;
  PetscCall(PetscFEGetBasisSpace(fe, &P));
  PetscCall(PetscFEGetDualSpace(fe, &Q));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(PetscDualSpaceGetDM(Q, &K));
  /* Create space */
  PetscCall(PetscObjectReference((PetscObject)P));
  Pref = P;
  /* Create dual space */
  PetscCall(PetscDualSpaceDuplicate(Q, &Qref));
  PetscCall(PetscDualSpaceSetType(Qref, PETSCDUALSPACEREFINED));
  PetscCall(DMRefine(K, PetscObjectComm((PetscObject)fe), &Kref));
  PetscCall(DMGetCoordinatesLocalSetUp(Kref));
  PetscCall(PetscDualSpaceSetDM(Qref, Kref));
  PetscCall(DMPlexGetHeightStratum(Kref, 0, &cStart, &cEnd));
  PetscCall(PetscMalloc1(cEnd - cStart, &cellSpaces));
  /* TODO: fix for non-uniform refinement */
  for (c = 0; c < cEnd - cStart; c++) cellSpaces[c] = Q;
  PetscCall(PetscDualSpaceRefinedSetCellSpaces(Qref, cellSpaces));
  PetscCall(PetscFree(cellSpaces));
  PetscCall(DMDestroy(&Kref));
  PetscCall(PetscDualSpaceSetUp(Qref));
  /* Create element */
  PetscCall(PetscFECreate(PetscObjectComm((PetscObject)fe), feRef));
  PetscCall(PetscFESetType(*feRef, PETSCFECOMPOSITE));
  PetscCall(PetscFESetBasisSpace(*feRef, Pref));
  PetscCall(PetscFESetDualSpace(*feRef, Qref));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCall(PetscFESetNumComponents(*feRef, numComp));
  PetscCall(PetscFESetUp(*feRef));
  PetscCall(PetscSpaceDestroy(&Pref));
  PetscCall(PetscDualSpaceDestroy(&Qref));
  /* Create quadrature */
  PetscCall(PetscFECompositeGetMapping(*feRef, &numSubelements, &v0, &jac, NULL));
  PetscCall(PetscQuadratureExpandComposite(q, numSubelements, v0, jac, &qref));
  PetscCall(PetscFESetQuadrature(*feRef, qref));
  PetscCall(PetscQuadratureDestroy(&qref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFESetDefaultName_Private(PetscFE fe)
{
  PetscSpace     P;
  PetscDualSpace Q;
  DM             K;
  DMPolytopeType ct;
  PetscInt       degree;
  char           name[64];

  PetscFunctionBegin;
  PetscCall(PetscFEGetBasisSpace(fe, &P));
  PetscCall(PetscSpaceGetDegree(P, &degree, NULL));
  PetscCall(PetscFEGetDualSpace(fe, &Q));
  PetscCall(PetscDualSpaceGetDM(Q, &K));
  PetscCall(DMPlexGetCellType(K, 0, &ct));
  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    PetscCall(PetscSNPrintf(name, sizeof(name), "Q%" PetscInt_FMT, degree));
    break;
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(PetscSNPrintf(name, sizeof(name), "P%" PetscInt_FMT, degree));
    break;
  case DM_POLYTOPE_TRI_PRISM:
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
    PetscCall(PetscSNPrintf(name, sizeof(name), "P%" PetscInt_FMT "xQ%" PetscInt_FMT, degree, degree));
    break;
  default:
    PetscCall(PetscSNPrintf(name, sizeof(name), "FE"));
  }
  PetscCall(PetscFESetName(fe, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateFromSpaces - Create a `PetscFE` from the basis and dual spaces

  Collective

  Input Parameters:
+ P  - The basis space
. Q  - The dual space
. q  - The cell quadrature
- fq - The face quadrature

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

  Note:
  The `PetscFE` takes ownership of these spaces by calling destroy on each. They should not be used after this call, and for borrowed references from `PetscFEGetSpace()` and the like, the caller must use `PetscObjectReference` before this call.

.seealso: `PetscFE`, `PetscSpace`, `PetscDualSpace`, `PetscQuadrature`,
          `PetscFECreateLagrangeByCell()`, `PetscFECreateDefault()`, `PetscFECreateByCell()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFECreateFromSpaces(PetscSpace P, PetscDualSpace Q, PetscQuadrature q, PetscQuadrature fq, PetscFE *fem)
{
  PetscInt    Nc;
  PetscInt    p_Ns = -1, p_Nc = -1, q_Ns = -1, q_Nc = -1;
  PetscBool   p_is_uniform_sum = PETSC_FALSE, p_interleave_basis = PETSC_FALSE, p_interleave_components = PETSC_FALSE;
  PetscBool   q_is_uniform_sum = PETSC_FALSE, q_interleave_basis = PETSC_FALSE, q_interleave_components = PETSC_FALSE;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)P, PETSCSPACESUM, &p_is_uniform_sum));
  if (p_is_uniform_sum) {
    PetscSpace subsp_0 = NULL;
    PetscCall(PetscSpaceSumGetNumSubspaces(P, &p_Ns));
    PetscCall(PetscSpaceGetNumComponents(P, &p_Nc));
    PetscCall(PetscSpaceSumGetConcatenate(P, &p_is_uniform_sum));
    PetscCall(PetscSpaceSumGetInterleave(P, &p_interleave_basis, &p_interleave_components));
    for (PetscInt s = 0; s < p_Ns; s++) {
      PetscSpace subsp;

      PetscCall(PetscSpaceSumGetSubspace(P, s, &subsp));
      if (!s) {
        subsp_0 = subsp;
      } else if (subsp != subsp_0) {
        p_is_uniform_sum = PETSC_FALSE;
      }
    }
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)Q, PETSCDUALSPACESUM, &q_is_uniform_sum));
  if (q_is_uniform_sum) {
    PetscDualSpace subsp_0 = NULL;
    PetscCall(PetscDualSpaceSumGetNumSubspaces(Q, &q_Ns));
    PetscCall(PetscDualSpaceGetNumComponents(Q, &q_Nc));
    PetscCall(PetscDualSpaceSumGetConcatenate(Q, &q_is_uniform_sum));
    PetscCall(PetscDualSpaceSumGetInterleave(Q, &q_interleave_basis, &q_interleave_components));
    for (PetscInt s = 0; s < q_Ns; s++) {
      PetscDualSpace subsp;

      PetscCall(PetscDualSpaceSumGetSubspace(Q, s, &subsp));
      if (!s) {
        subsp_0 = subsp;
      } else if (subsp != subsp_0) {
        q_is_uniform_sum = PETSC_FALSE;
      }
    }
  }
  if (p_is_uniform_sum && q_is_uniform_sum && (p_interleave_basis == q_interleave_basis) && (p_interleave_components == q_interleave_components) && (p_Ns == q_Ns) && (p_Nc == q_Nc)) {
    PetscSpace     scalar_space;
    PetscDualSpace scalar_dspace;
    PetscFE        scalar_fe;

    PetscCall(PetscSpaceSumGetSubspace(P, 0, &scalar_space));
    PetscCall(PetscDualSpaceSumGetSubspace(Q, 0, &scalar_dspace));
    PetscCall(PetscObjectReference((PetscObject)scalar_space));
    PetscCall(PetscObjectReference((PetscObject)scalar_dspace));
    PetscCall(PetscObjectReference((PetscObject)q));
    PetscCall(PetscObjectReference((PetscObject)fq));
    PetscCall(PetscFECreateFromSpaces(scalar_space, scalar_dspace, q, fq, &scalar_fe));
    PetscCall(PetscFECreateVector(scalar_fe, p_Ns, p_interleave_basis, p_interleave_components, fem));
    PetscCall(PetscFEDestroy(&scalar_fe));
  } else {
    PetscCall(PetscFECreate(PetscObjectComm((PetscObject)P), fem));
    PetscCall(PetscFESetType(*fem, PETSCFEBASIC));
  }
  PetscCall(PetscSpaceGetNumComponents(P, &Nc));
  PetscCall(PetscFESetNumComponents(*fem, Nc));
  PetscCall(PetscFESetBasisSpace(*fem, P));
  PetscCall(PetscFESetDualSpace(*fem, Q));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)P, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*fem, prefix));
  PetscCall(PetscFESetUp(*fem));
  PetscCall(PetscSpaceDestroy(&P));
  PetscCall(PetscDualSpaceDestroy(&Q));
  PetscCall(PetscFESetQuadrature(*fem, q));
  PetscCall(PetscFESetFaceQuadrature(*fem, fq));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscCall(PetscQuadratureDestroy(&fq));
  PetscCall(PetscFESetDefaultName_Private(*fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFECreate_Internal(MPI_Comm comm, PetscInt dim, PetscInt Nc, DMPolytopeType ct, const char prefix[], PetscInt degree, PetscInt qorder, PetscBool setFromOptions, PetscFE *fem)
{
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscQuadrature q, fq;
  PetscBool       tensor;

  PetscFunctionBegin;
  if (prefix) PetscAssertPointer(prefix, 5);
  PetscAssertPointer(fem, 9);
  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    tensor = PETSC_TRUE;
    break;
  default:
    tensor = PETSC_FALSE;
  }
  /* Create space */
  PetscCall(PetscSpaceCreate(comm, &P));
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)P, prefix));
  PetscCall(PetscSpacePolynomialSetTensor(P, tensor));
  PetscCall(PetscSpaceSetNumComponents(P, Nc));
  PetscCall(PetscSpaceSetNumVariables(P, dim));
  if (degree >= 0) {
    PetscCall(PetscSpaceSetDegree(P, degree, PETSC_DETERMINE));
    if (ct == DM_POLYTOPE_TRI_PRISM || ct == DM_POLYTOPE_TRI_PRISM_TENSOR) {
      PetscSpace Pend, Pside;

      PetscCall(PetscSpaceSetNumComponents(P, 1));
      PetscCall(PetscSpaceCreate(comm, &Pend));
      PetscCall(PetscSpaceSetType(Pend, PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpacePolynomialSetTensor(Pend, PETSC_FALSE));
      PetscCall(PetscSpaceSetNumComponents(Pend, 1));
      PetscCall(PetscSpaceSetNumVariables(Pend, dim - 1));
      PetscCall(PetscSpaceSetDegree(Pend, degree, PETSC_DETERMINE));
      PetscCall(PetscSpaceCreate(comm, &Pside));
      PetscCall(PetscSpaceSetType(Pside, PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpacePolynomialSetTensor(Pside, PETSC_FALSE));
      PetscCall(PetscSpaceSetNumComponents(Pside, 1));
      PetscCall(PetscSpaceSetNumVariables(Pside, 1));
      PetscCall(PetscSpaceSetDegree(Pside, degree, PETSC_DETERMINE));
      PetscCall(PetscSpaceSetType(P, PETSCSPACETENSOR));
      PetscCall(PetscSpaceTensorSetNumSubspaces(P, 2));
      PetscCall(PetscSpaceTensorSetSubspace(P, 0, Pend));
      PetscCall(PetscSpaceTensorSetSubspace(P, 1, Pside));
      PetscCall(PetscSpaceDestroy(&Pend));
      PetscCall(PetscSpaceDestroy(&Pside));

      if (Nc > 1) {
        PetscSpace scalar_P = P;

        PetscCall(PetscSpaceCreate(comm, &P));
        PetscCall(PetscSpaceSetNumVariables(P, dim));
        PetscCall(PetscSpaceSetNumComponents(P, Nc));
        PetscCall(PetscSpaceSetType(P, PETSCSPACESUM));
        PetscCall(PetscSpaceSumSetNumSubspaces(P, Nc));
        PetscCall(PetscSpaceSumSetConcatenate(P, PETSC_TRUE));
        PetscCall(PetscSpaceSumSetInterleave(P, PETSC_TRUE, PETSC_FALSE));
        for (PetscInt i = 0; i < Nc; i++) PetscCall(PetscSpaceSumSetSubspace(P, i, scalar_P));
        PetscCall(PetscSpaceDestroy(&scalar_P));
      }
    }
  }
  if (setFromOptions) PetscCall(PetscSpaceSetFromOptions(P));
  PetscCall(PetscSpaceSetUp(P));
  PetscCall(PetscSpaceGetDegree(P, &degree, NULL));
  PetscCall(PetscSpacePolynomialGetTensor(P, &tensor));
  PetscCall(PetscSpaceGetNumComponents(P, &Nc));
  /* Create dual space */
  PetscCall(PetscDualSpaceCreate(comm, &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Q, prefix));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, ct, &K));
  PetscCall(PetscDualSpaceSetDM(Q, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetNumComponents(Q, Nc));
  PetscCall(PetscDualSpaceSetOrder(Q, degree));
  PetscCall(PetscDualSpaceLagrangeSetTensor(Q, (tensor || (ct == DM_POLYTOPE_TRI_PRISM)) ? PETSC_TRUE : PETSC_FALSE));
  if (setFromOptions) PetscCall(PetscDualSpaceSetFromOptions(Q));
  PetscCall(PetscDualSpaceSetUp(Q));
  /* Create quadrature */
  PetscDTSimplexQuadratureType qtype = PETSCDTSIMPLEXQUAD_DEFAULT;

  qorder = qorder >= 0 ? qorder : degree;
  if (setFromOptions) {
    PetscObjectOptionsBegin((PetscObject)P);
    PetscCall(PetscOptionsBoundedInt("-petscfe_default_quadrature_order", "Quadrature order is one less than quadrature points per edge", "PetscFECreateDefault", qorder, &qorder, NULL, 0));
    PetscCall(PetscOptionsEnum("-petscfe_default_quadrature_type", "Simplex quadrature type", "PetscDTSimplexQuadratureType", PetscDTSimplexQuadratureTypes, (PetscEnum)qtype, (PetscEnum *)&qtype, NULL));
    PetscOptionsEnd();
  }
  PetscCall(PetscDTCreateQuadratureByCell(ct, qorder, qtype, &q, &fq));
  /* Create finite element */
  PetscCall(PetscFECreateFromSpaces(P, Q, q, fq, fem));
  if (setFromOptions) PetscCall(PetscFESetFromOptions(*fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateDefault - Create a `PetscFE` for basic FEM computation

  Collective

  Input Parameters:
+ comm      - The MPI comm
. dim       - The spatial dimension
. Nc        - The number of components
. isSimplex - Flag for simplex reference cell, otherwise its a tensor product
. prefix    - The options prefix, or `NULL`
- qorder    - The quadrature order or `PETSC_DETERMINE` to use `PetscSpace` polynomial degree

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

  Note:
  Each subobject is SetFromOption() during creation, so that the object may be customized from the command line, using the prefix specified above. See the links below for the particular options available.

.seealso: `PetscFECreateLagrange()`, `PetscFECreateByCell()`, `PetscSpaceSetFromOptions()`, `PetscDualSpaceSetFromOptions()`, `PetscFESetFromOptions()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFECreateDefault(MPI_Comm comm, PetscInt dim, PetscInt Nc, PetscBool isSimplex, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  PetscFunctionBegin;
  PetscCall(PetscFECreate_Internal(comm, dim, Nc, DMPolytopeTypeSimpleShape(dim, isSimplex), prefix, PETSC_DECIDE, qorder, PETSC_TRUE, fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateByCell - Create a `PetscFE` for basic FEM computation

  Collective

  Input Parameters:
+ comm   - The MPI comm
. dim    - The spatial dimension
. Nc     - The number of components
. ct     - The celltype of the reference cell
. prefix - The options prefix, or `NULL`
- qorder - The quadrature order or `PETSC_DETERMINE` to use `PetscSpace` polynomial degree

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

  Note:
  Each subobject is SetFromOption() during creation, so that the object may be customized from the command line, using the prefix specified above. See the links below for the particular options available.

.seealso: `PetscFECreateDefault()`, `PetscFECreateLagrange()`, `PetscSpaceSetFromOptions()`, `PetscDualSpaceSetFromOptions()`, `PetscFESetFromOptions()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFECreateByCell(MPI_Comm comm, PetscInt dim, PetscInt Nc, DMPolytopeType ct, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  PetscFunctionBegin;
  PetscCall(PetscFECreate_Internal(comm, dim, Nc, ct, prefix, PETSC_DECIDE, qorder, PETSC_TRUE, fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateLagrange - Create a `PetscFE` for the basic Lagrange space of degree k

  Collective

  Input Parameters:
+ comm      - The MPI comm
. dim       - The spatial dimension
. Nc        - The number of components
. isSimplex - Flag for simplex reference cell, otherwise its a tensor product
. k         - The degree k of the space
- qorder    - The quadrature order or `PETSC_DETERMINE` to use `PetscSpace` polynomial degree

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

  Note:
  For simplices, this element is the space of maximum polynomial degree k, otherwise it is a tensor product of 1D polynomials, each with maximal degree k.

.seealso: `PetscFECreateLagrangeByCell()`, `PetscFECreateDefault()`, `PetscFECreateByCell()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFECreateLagrange(MPI_Comm comm, PetscInt dim, PetscInt Nc, PetscBool isSimplex, PetscInt k, PetscInt qorder, PetscFE *fem)
{
  PetscFunctionBegin;
  PetscCall(PetscFECreate_Internal(comm, dim, Nc, DMPolytopeTypeSimpleShape(dim, isSimplex), NULL, k, qorder, PETSC_FALSE, fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateLagrangeByCell - Create a `PetscFE` for the basic Lagrange space of degree k

  Collective

  Input Parameters:
+ comm   - The MPI comm
. dim    - The spatial dimension
. Nc     - The number of components
. ct     - The celltype of the reference cell
. k      - The degree k of the space
- qorder - The quadrature order or `PETSC_DETERMINE` to use `PetscSpace` polynomial degree

  Output Parameter:
. fem - The `PetscFE` object

  Level: beginner

  Note:
  For simplices, this element is the space of maximum polynomial degree k, otherwise it is a tensor product of 1D polynomials, each with maximal degree k.

.seealso: `PetscFECreateLagrange()`, `PetscFECreateDefault()`, `PetscFECreateByCell()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFECreateLagrangeByCell(MPI_Comm comm, PetscInt dim, PetscInt Nc, DMPolytopeType ct, PetscInt k, PetscInt qorder, PetscFE *fem)
{
  PetscFunctionBegin;
  PetscCall(PetscFECreate_Internal(comm, dim, Nc, ct, NULL, k, qorder, PETSC_FALSE, fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFELimitDegree - Copy a `PetscFE` but limit the degree to be in the given range

  Collective

  Input Parameters:
+ fe        - The `PetscFE`
. minDegree - The minimum degree, or `PETSC_DETERMINE` for no limit
- maxDegree - The maximum degree, or `PETSC_DETERMINE` for no limit

  Output Parameter:
. newfe - The `PetscFE` object

  Level: advanced

  Note:
  This currently only works for Lagrange elements.

.seealso: `PetscFECreateLagrange()`, `PetscFECreateDefault()`, `PetscFECreateByCell()`, `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFELimitDegree(PetscFE fe, PetscInt minDegree, PetscInt maxDegree, PetscFE *newfe)
{
  PetscDualSpace Q;
  PetscBool      islag, issum;
  PetscInt       oldk = 0, k;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDualSpace(fe, &Q));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q, PETSCDUALSPACELAGRANGE, &islag));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q, PETSCDUALSPACESUM, &issum));
  if (islag) {
    PetscCall(PetscDualSpaceGetOrder(Q, &oldk));
  } else if (issum) {
    PetscDualSpace subQ;

    PetscCall(PetscDualSpaceSumGetSubspace(Q, 0, &subQ));
    PetscCall(PetscDualSpaceGetOrder(subQ, &oldk));
  } else {
    PetscCall(PetscObjectReference((PetscObject)fe));
    *newfe = fe;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  k = oldk;
  if (minDegree >= 0) k = PetscMax(k, minDegree);
  if (maxDegree >= 0) k = PetscMin(k, maxDegree);
  if (k != oldk) {
    DM              K;
    PetscSpace      P;
    PetscQuadrature q;
    DMPolytopeType  ct;
    PetscInt        dim, Nc;

    PetscCall(PetscFEGetBasisSpace(fe, &P));
    PetscCall(PetscSpaceGetNumVariables(P, &dim));
    PetscCall(PetscSpaceGetNumComponents(P, &Nc));
    PetscCall(PetscDualSpaceGetDM(Q, &K));
    PetscCall(DMPlexGetCellType(K, 0, &ct));
    PetscCall(PetscFECreateLagrangeByCell(PetscObjectComm((PetscObject)fe), dim, Nc, ct, k, PETSC_DETERMINE, newfe));
    PetscCall(PetscFEGetQuadrature(fe, &q));
    PetscCall(PetscFESetQuadrature(*newfe, q));
  } else {
    PetscCall(PetscObjectReference((PetscObject)fe));
    *newfe = fe;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateBrokenElement - Create a discontinuous version of the input `PetscFE`

  Collective

  Input Parameters:
. cgfe - The continuous `PetscFE` object

  Output Parameter:
. dgfe - The discontinuous `PetscFE` object

  Level: advanced

  Note:
  This only works for Lagrange elements.

.seealso: `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`, `PetscFECreateLagrange()`, `PetscFECreateLagrangeByCell()`, `PetscDualSpaceLagrangeSetContinuity()`
@*/
PetscErrorCode PetscFECreateBrokenElement(PetscFE cgfe, PetscFE *dgfe)
{
  PetscSpace      P;
  PetscDualSpace  Q, dgQ;
  PetscQuadrature q, fq;
  PetscBool       is_lagrange, is_sum;

  PetscFunctionBegin;
  PetscCall(PetscFEGetBasisSpace(cgfe, &P));
  PetscCall(PetscObjectReference((PetscObject)P));
  PetscCall(PetscFEGetDualSpace(cgfe, &Q));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q, PETSCDUALSPACELAGRANGE, &is_lagrange));
  PetscCall(PetscObjectTypeCompare((PetscObject)Q, PETSCDUALSPACESUM, &is_sum));
  PetscCheck(is_lagrange || is_sum, PETSC_COMM_SELF, PETSC_ERR_SUP, "Can only create broken elements of Lagrange elements");
  PetscCall(PetscDualSpaceDuplicate(Q, &dgQ));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(dgQ, PETSC_FALSE));
  PetscCall(PetscDualSpaceSetUp(dgQ));
  PetscCall(PetscFEGetQuadrature(cgfe, &q));
  PetscCall(PetscObjectReference((PetscObject)q));
  PetscCall(PetscFEGetFaceQuadrature(cgfe, &fq));
  PetscCall(PetscObjectReference((PetscObject)fq));
  PetscCall(PetscFECreateFromSpaces(P, dgQ, q, fq, dgfe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFESetName - Names the `PetscFE` and its subobjects

  Not Collective

  Input Parameters:
+ fe   - The `PetscFE`
- name - The name

  Level: intermediate

.seealso: `PetscFECreate()`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`
@*/
PetscErrorCode PetscFESetName(PetscFE fe, const char name[])
{
  PetscSpace     P;
  PetscDualSpace Q;

  PetscFunctionBegin;
  PetscCall(PetscFEGetBasisSpace(fe, &P));
  PetscCall(PetscFEGetDualSpace(fe, &Q));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  PetscCall(PetscObjectSetName((PetscObject)P, name));
  PetscCall(PetscObjectSetName((PetscObject)Q, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFEEvaluateFieldJets_Internal(PetscDS ds, PetscInt Nf, PetscInt r, PetscInt q, PetscTabulation T[], PetscFEGeom *fegeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscInt dOffset = 0, fOffset = 0, f, g;

  for (f = 0; f < Nf; ++f) {
    PetscCheck(r < T[f]->Nr, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Replica number %" PetscInt_FMT " should be in [0, %" PetscInt_FMT ")", r, T[f]->Nr);
    PetscCheck(q < T[f]->Np, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point number %" PetscInt_FMT " should be in [0, %" PetscInt_FMT ")", q, T[f]->Np);
    PetscFE          fe;
    const PetscInt   k       = ds->jetDegree[f];
    const PetscInt   cdim    = T[f]->cdim;
    const PetscInt   dE      = fegeom->dimEmbed;
    const PetscInt   Nq      = T[f]->Np;
    const PetscInt   Nbf     = T[f]->Nb;
    const PetscInt   Ncf     = T[f]->Nc;
    const PetscReal *Bq      = &T[f]->T[0][(r * Nq + q) * Nbf * Ncf];
    const PetscReal *Dq      = &T[f]->T[1][(r * Nq + q) * Nbf * Ncf * cdim];
    const PetscReal *Hq      = k > 1 ? &T[f]->T[2][(r * Nq + q) * Nbf * Ncf * cdim * cdim] : NULL;
    PetscInt         hOffset = 0, b, c, d;

    PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *)&fe));
    for (c = 0; c < Ncf; ++c) u[fOffset + c] = 0.0;
    for (d = 0; d < dE * Ncf; ++d) u_x[fOffset * dE + d] = 0.0;
    for (b = 0; b < Nbf; ++b) {
      for (c = 0; c < Ncf; ++c) {
        const PetscInt cidx = b * Ncf + c;

        u[fOffset + c] += Bq[cidx] * coefficients[dOffset + b];
        for (d = 0; d < cdim; ++d) u_x[(fOffset + c) * dE + d] += Dq[cidx * cdim + d] * coefficients[dOffset + b];
      }
    }
    if (k > 1) {
      for (g = 0; g < Nf; ++g) hOffset += T[g]->Nc * dE;
      for (d = 0; d < dE * dE * Ncf; ++d) u_x[hOffset + fOffset * dE * dE + d] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b * Ncf + c;

          for (d = 0; d < cdim * cdim; ++d) u_x[hOffset + (fOffset + c) * dE * dE + d] += Hq[cidx * cdim * cdim + d] * coefficients[dOffset + b];
        }
      }
      PetscCall(PetscFEPushforwardHessian(fe, fegeom, 1, &u_x[hOffset + fOffset * dE * dE]));
    }
    PetscCall(PetscFEPushforward(fe, fegeom, 1, &u[fOffset]));
    PetscCall(PetscFEPushforwardGradient(fe, fegeom, 1, &u_x[fOffset * dE]));
    if (u_t) {
      for (c = 0; c < Ncf; ++c) u_t[fOffset + c] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b * Ncf + c;

          u_t[fOffset + c] += Bq[cidx] * coefficients_t[dOffset + b];
        }
      }
      PetscCall(PetscFEPushforward(fe, fegeom, 1, &u_t[fOffset]));
    }
    fOffset += Ncf;
    dOffset += Nbf;
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscFEEvaluateFieldJets_Hybrid_Internal(PetscDS ds, PetscInt Nf, PetscInt rc, PetscInt qc, PetscTabulation Tab[], const PetscInt rf[], const PetscInt qf[], PetscTabulation Tabf[], PetscFEGeom *fegeom, PetscFEGeom *fegeomNbr, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscInt dOffset = 0, fOffset = 0, f, g;

  /* f is the field number in the DS, g is the field number in u[] */
  for (f = 0, g = 0; f < Nf; ++f) {
    PetscBool isCohesive;
    PetscInt  Ns, s;

    if (!Tab[f]) continue;
    PetscCall(PetscDSGetCohesive(ds, f, &isCohesive));
    Ns = isCohesive ? 1 : 2;
    {
      PetscTabulation T   = isCohesive ? Tab[f] : Tabf[f];
      PetscFE         fe  = (PetscFE)ds->disc[f];
      const PetscInt  dEt = T->cdim;
      const PetscInt  dE  = fegeom->dimEmbed;
      const PetscInt  Nq  = T->Np;
      const PetscInt  Nbf = T->Nb;
      const PetscInt  Ncf = T->Nc;

      for (s = 0; s < Ns; ++s, ++g) {
        const PetscInt   r  = isCohesive ? rc : rf[s];
        const PetscInt   q  = isCohesive ? qc : qf[s];
        const PetscReal *Bq = &T->T[0][(r * Nq + q) * Nbf * Ncf];
        const PetscReal *Dq = &T->T[1][(r * Nq + q) * Nbf * Ncf * dEt];
        PetscInt         b, c, d;

        PetscCheck(r < T->Nr, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT " Side %" PetscInt_FMT " Replica number %" PetscInt_FMT " should be in [0, %" PetscInt_FMT ")", f, s, r, T->Nr);
        PetscCheck(q < T->Np, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT " Side %" PetscInt_FMT " Point number %" PetscInt_FMT " should be in [0, %" PetscInt_FMT ")", f, s, q, T->Np);
        for (c = 0; c < Ncf; ++c) u[fOffset + c] = 0.0;
        for (d = 0; d < dE * Ncf; ++d) u_x[fOffset * dE + d] = 0.0;
        for (b = 0; b < Nbf; ++b) {
          for (c = 0; c < Ncf; ++c) {
            const PetscInt cidx = b * Ncf + c;

            u[fOffset + c] += Bq[cidx] * coefficients[dOffset + b];
            for (d = 0; d < dEt; ++d) u_x[(fOffset + c) * dE + d] += Dq[cidx * dEt + d] * coefficients[dOffset + b];
          }
        }
        PetscCall(PetscFEPushforward(fe, isCohesive ? fegeom : &fegeomNbr[s], 1, &u[fOffset]));
        PetscCall(PetscFEPushforwardGradient(fe, isCohesive ? fegeom : &fegeomNbr[s], 1, &u_x[fOffset * dE]));
        if (u_t) {
          for (c = 0; c < Ncf; ++c) u_t[fOffset + c] = 0.0;
          for (b = 0; b < Nbf; ++b) {
            for (c = 0; c < Ncf; ++c) {
              const PetscInt cidx = b * Ncf + c;

              u_t[fOffset + c] += Bq[cidx] * coefficients_t[dOffset + b];
            }
          }
          PetscCall(PetscFEPushforward(fe, fegeom, 1, &u_t[fOffset]));
        }
        fOffset += Ncf;
        dOffset += Nbf;
      }
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscFEEvaluateFaceFields_Internal(PetscDS prob, PetscInt field, PetscInt faceLoc, const PetscScalar coefficients[], PetscScalar u[])
{
  PetscFE         fe;
  PetscTabulation Tc;
  PetscInt        b, c;

  if (!prob) return PETSC_SUCCESS;
  PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *)&fe));
  PetscCall(PetscFEGetFaceCentroidTabulation(fe, &Tc));
  {
    const PetscReal *faceBasis = Tc->T[0];
    const PetscInt   Nb        = Tc->Nb;
    const PetscInt   Nc        = Tc->Nc;

    for (c = 0; c < Nc; ++c) u[c] = 0.0;
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) u[c] += coefficients[b] * faceBasis[(faceLoc * Nb + b) * Nc + c];
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscFEUpdateElementVec_Internal(PetscFE fe, PetscTabulation T, PetscInt r, PetscScalar tmpBasis[], PetscScalar tmpBasisDer[], PetscInt e, PetscFEGeom *fegeom, PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  PetscFEGeom      pgeom;
  const PetscInt   dEt      = T->cdim;
  const PetscInt   dE       = fegeom->dimEmbed;
  const PetscInt   Nq       = T->Np;
  const PetscInt   Nb       = T->Nb;
  const PetscInt   Nc       = T->Nc;
  const PetscReal *basis    = &T->T[0][r * Nq * Nb * Nc];
  const PetscReal *basisDer = &T->T[1][r * Nq * Nb * Nc * dEt];
  PetscInt         q, b, c, d;

  for (q = 0; q < Nq; ++q) {
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b * Nc + c;

        tmpBasis[bcidx] = basis[q * Nb * Nc + bcidx];
        for (d = 0; d < dEt; ++d) tmpBasisDer[bcidx * dE + d] = basisDer[q * Nb * Nc * dEt + bcidx * dEt + d];
        for (d = dEt; d < dE; ++d) tmpBasisDer[bcidx * dE + d] = 0.0;
      }
    }
    PetscCall(PetscFEGeomGetCellPoint(fegeom, e, q, &pgeom));
    PetscCall(PetscFEPushforward(fe, &pgeom, Nb, tmpBasis));
    PetscCall(PetscFEPushforwardGradient(fe, &pgeom, Nb, tmpBasisDer));
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b * Nc + c;
        const PetscInt qcidx = q * Nc + c;

        elemVec[b] += tmpBasis[bcidx] * f0[qcidx];
        for (d = 0; d < dE; ++d) elemVec[b] += tmpBasisDer[bcidx * dE + d] * f1[qcidx * dE + d];
      }
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscFEUpdateElementVec_Hybrid_Internal(PetscFE fe, PetscTabulation T, PetscInt r, PetscInt side, PetscScalar tmpBasis[], PetscScalar tmpBasisDer[], PetscFEGeom *fegeom, PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  const PetscInt   dE       = T->cdim;
  const PetscInt   Nq       = T->Np;
  const PetscInt   Nb       = T->Nb;
  const PetscInt   Nc       = T->Nc;
  const PetscReal *basis    = &T->T[0][r * Nq * Nb * Nc];
  const PetscReal *basisDer = &T->T[1][r * Nq * Nb * Nc * dE];

  for (PetscInt q = 0; q < Nq; ++q) {
    for (PetscInt b = 0; b < Nb; ++b) {
      for (PetscInt c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b * Nc + c;

        tmpBasis[bcidx] = basis[q * Nb * Nc + bcidx];
        for (PetscInt d = 0; d < dE; ++d) tmpBasisDer[bcidx * dE + d] = basisDer[q * Nb * Nc * dE + bcidx * dE + d];
      }
    }
    PetscCall(PetscFEPushforward(fe, fegeom, Nb, tmpBasis));
    // TODO This is currently broken since we do not pull the geometry down to the lower dimension
    // PetscCall(PetscFEPushforwardGradient(fe, fegeom, Nb, tmpBasisDer));
    if (side == 2) {
      // Integrating over whole cohesive cell, so insert for both sides
      for (PetscInt s = 0; s < 2; ++s) {
        for (PetscInt b = 0; b < Nb; ++b) {
          for (PetscInt c = 0; c < Nc; ++c) {
            const PetscInt bcidx = b * Nc + c;
            const PetscInt qcidx = (q * 2 + s) * Nc + c;

            elemVec[Nb * s + b] += tmpBasis[bcidx] * f0[qcidx];
            for (PetscInt d = 0; d < dE; ++d) elemVec[Nb * s + b] += tmpBasisDer[bcidx * dE + d] * f1[qcidx * dE + d];
          }
        }
      }
    } else {
      // Integrating over endcaps of cohesive cell, so insert for correct side
      for (PetscInt b = 0; b < Nb; ++b) {
        for (PetscInt c = 0; c < Nc; ++c) {
          const PetscInt bcidx = b * Nc + c;
          const PetscInt qcidx = q * Nc + c;

          elemVec[Nb * side + b] += tmpBasis[bcidx] * f0[qcidx];
          for (PetscInt d = 0; d < dE; ++d) elemVec[Nb * side + b] += tmpBasisDer[bcidx * dE + d] * f1[qcidx * dE + d];
        }
      }
    }
  }
  return PETSC_SUCCESS;
}

#define petsc_elemmat_kernel_g1(_NbI, _NcI, _NbJ, _NcJ, _dE) \
  do { \
    for (PetscInt fc = 0; fc < (_NcI); ++fc) { \
      for (PetscInt gc = 0; gc < (_NcJ); ++gc) { \
        const PetscScalar *G = g1 + (fc * (_NcJ) + gc) * _dE; \
        for (PetscInt f = 0; f < (_NbI); ++f) { \
          const PetscScalar tBIv = tmpBasisI[f * (_NcI) + fc]; \
          for (PetscInt g = 0; g < (_NbJ); ++g) { \
            const PetscScalar *tBDJ = tmpBasisDerJ + (g * (_NcJ) + gc) * (_dE); \
            PetscScalar        s    = 0.0; \
            for (PetscInt df = 0; df < _dE; ++df) s += G[df] * tBDJ[df]; \
            elemMat[(offsetI + f) * totDim + (offsetJ + g)] += s * tBIv; \
          } \
        } \
      } \
    } \
  } while (0)

#define petsc_elemmat_kernel_g2(_NbI, _NcI, _NbJ, _NcJ, _dE) \
  do { \
    for (PetscInt gc = 0; gc < (_NcJ); ++gc) { \
      for (PetscInt fc = 0; fc < (_NcI); ++fc) { \
        const PetscScalar *G = g2 + (fc * (_NcJ) + gc) * _dE; \
        for (PetscInt g = 0; g < (_NbJ); ++g) { \
          const PetscScalar tBJv = tmpBasisJ[g * (_NcJ) + gc]; \
          for (PetscInt f = 0; f < (_NbI); ++f) { \
            const PetscScalar *tBDI = tmpBasisDerI + (f * (_NcI) + fc) * (_dE); \
            PetscScalar        s    = 0.0; \
            for (PetscInt df = 0; df < _dE; ++df) s += tBDI[df] * G[df]; \
            elemMat[(offsetI + f) * totDim + (offsetJ + g)] += s * tBJv; \
          } \
        } \
      } \
    } \
  } while (0)

#define petsc_elemmat_kernel_g3(_NbI, _NcI, _NbJ, _NcJ, _dE) \
  do { \
    for (PetscInt fc = 0; fc < (_NcI); ++fc) { \
      for (PetscInt gc = 0; gc < (_NcJ); ++gc) { \
        const PetscScalar *G = g3 + (fc * (_NcJ) + gc) * (_dE) * (_dE); \
        for (PetscInt f = 0; f < (_NbI); ++f) { \
          const PetscScalar *tBDI = tmpBasisDerI + (f * (_NcI) + fc) * (_dE); \
          for (PetscInt g = 0; g < (_NbJ); ++g) { \
            PetscScalar        s    = 0.0; \
            const PetscScalar *tBDJ = tmpBasisDerJ + (g * (_NcJ) + gc) * (_dE); \
            for (PetscInt df = 0; df < (_dE); ++df) { \
              for (PetscInt dg = 0; dg < (_dE); ++dg) s += tBDI[df] * G[df * (_dE) + dg] * tBDJ[dg]; \
            } \
            elemMat[(offsetI + f) * totDim + (offsetJ + g)] += s; \
          } \
        } \
      } \
    } \
  } while (0)

PetscErrorCode PetscFEUpdateElementMat_Internal(PetscFE feI, PetscFE feJ, PetscInt r, PetscInt q, PetscTabulation TI, PetscScalar tmpBasisI[], PetscScalar tmpBasisDerI[], PetscTabulation TJ, PetscScalar tmpBasisJ[], PetscScalar tmpBasisDerJ[], PetscFEGeom *fegeom, const PetscScalar g0[], const PetscScalar g1[], const PetscScalar g2[], const PetscScalar g3[], PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscScalar elemMat[])
{
  const PetscInt   cdim      = TI->cdim;
  const PetscInt   dE        = fegeom->dimEmbed;
  const PetscInt   NqI       = TI->Np;
  const PetscInt   NbI       = TI->Nb;
  const PetscInt   NcI       = TI->Nc;
  const PetscReal *basisI    = &TI->T[0][(r * NqI + q) * NbI * NcI];
  const PetscReal *basisDerI = &TI->T[1][(r * NqI + q) * NbI * NcI * cdim];
  const PetscInt   NqJ       = TJ->Np;
  const PetscInt   NbJ       = TJ->Nb;
  const PetscInt   NcJ       = TJ->Nc;
  const PetscReal *basisJ    = &TJ->T[0][(r * NqJ + q) * NbJ * NcJ];
  const PetscReal *basisDerJ = &TJ->T[1][(r * NqJ + q) * NbJ * NcJ * cdim];

  for (PetscInt f = 0; f < NbI; ++f) {
    for (PetscInt fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f * NcI + fc; /* Test function basis index */

      tmpBasisI[fidx] = basisI[fidx];
      for (PetscInt df = 0; df < cdim; ++df) tmpBasisDerI[fidx * dE + df] = basisDerI[fidx * cdim + df];
    }
  }
  PetscCall(PetscFEPushforward(feI, fegeom, NbI, tmpBasisI));
  PetscCall(PetscFEPushforwardGradient(feI, fegeom, NbI, tmpBasisDerI));
  if (feI != feJ) {
    for (PetscInt g = 0; g < NbJ; ++g) {
      for (PetscInt gc = 0; gc < NcJ; ++gc) {
        const PetscInt gidx = g * NcJ + gc; /* Trial function basis index */

        tmpBasisJ[gidx] = basisJ[gidx];
        for (PetscInt dg = 0; dg < cdim; ++dg) tmpBasisDerJ[gidx * dE + dg] = basisDerJ[gidx * cdim + dg];
      }
    }
    PetscCall(PetscFEPushforward(feJ, fegeom, NbJ, tmpBasisJ));
    PetscCall(PetscFEPushforwardGradient(feJ, fegeom, NbJ, tmpBasisDerJ));
  } else {
    tmpBasisJ    = tmpBasisI;
    tmpBasisDerJ = tmpBasisDerI;
  }
  if (PetscUnlikely(g0)) {
    for (PetscInt f = 0; f < NbI; ++f) {
      const PetscInt i = offsetI + f; /* Element matrix row */

      for (PetscInt fc = 0; fc < NcI; ++fc) {
        const PetscScalar bI = tmpBasisI[f * NcI + fc]; /* Test function basis value */

        for (PetscInt g = 0; g < NbJ; ++g) {
          const PetscInt j    = offsetJ + g; /* Element matrix column */
          const PetscInt fOff = i * totDim + j;

          for (PetscInt gc = 0; gc < NcJ; ++gc) elemMat[fOff] += bI * g0[fc * NcJ + gc] * tmpBasisJ[g * NcJ + gc];
        }
      }
    }
  }
  if (PetscUnlikely(g1)) {
#if 1
    if (dE == 2) {
      petsc_elemmat_kernel_g1(NbI, NcI, NbJ, NcJ, 2);
    } else if (dE == 3) {
      petsc_elemmat_kernel_g1(NbI, NcI, NbJ, NcJ, 3);
    } else {
      petsc_elemmat_kernel_g1(NbI, NcI, NbJ, NcJ, dE);
    }
#else
    for (PetscInt f = 0; f < NbI; ++f) {
      const PetscInt i = offsetI + f; /* Element matrix row */

      for (PetscInt fc = 0; fc < NcI; ++fc) {
        const PetscScalar bI = tmpBasisI[f * NcI + fc]; /* Test function basis value */

        for (PetscInt g = 0; g < NbJ; ++g) {
          const PetscInt j    = offsetJ + g; /* Element matrix column */
          const PetscInt fOff = i * totDim + j;

          for (PetscInt gc = 0; gc < NcJ; ++gc) {
            const PetscInt gidx = g * NcJ + gc; /* Trial function basis index */

            for (PetscInt df = 0; df < dE; ++df) elemMat[fOff] += bI * g1[(fc * NcJ + gc) * dE + df] * tmpBasisDerJ[gidx * dE + df];
          }
        }
      }
    }
#endif
  }
  if (PetscUnlikely(g2)) {
#if 1
    if (dE == 2) {
      petsc_elemmat_kernel_g2(NbI, NcI, NbJ, NcJ, 2);
    } else if (dE == 3) {
      petsc_elemmat_kernel_g2(NbI, NcI, NbJ, NcJ, 3);
    } else {
      petsc_elemmat_kernel_g2(NbI, NcI, NbJ, NcJ, dE);
    }
#else
    for (PetscInt g = 0; g < NbJ; ++g) {
      const PetscInt j = offsetJ + g; /* Element matrix column */

      for (PetscInt gc = 0; gc < NcJ; ++gc) {
        const PetscScalar bJ = tmpBasisJ[g * NcJ + gc]; /* Trial function basis value */

        for (PetscInt f = 0; f < NbI; ++f) {
          const PetscInt i    = offsetI + f; /* Element matrix row */
          const PetscInt fOff = i * totDim + j;

          for (PetscInt fc = 0; fc < NcI; ++fc) {
            const PetscInt fidx = f * NcI + fc; /* Test function basis index */

            for (PetscInt df = 0; df < dE; ++df) elemMat[fOff] += tmpBasisDerI[fidx * dE + df] * g2[(fc * NcJ + gc) * dE + df] * bJ;
          }
        }
      }
    }
#endif
  }
  if (PetscUnlikely(g3)) {
#if 1
    if (dE == 2) {
      petsc_elemmat_kernel_g3(NbI, NcI, NbJ, NcJ, 2);
    } else if (dE == 3) {
      petsc_elemmat_kernel_g3(NbI, NcI, NbJ, NcJ, 3);
    } else {
      petsc_elemmat_kernel_g3(NbI, NcI, NbJ, NcJ, dE);
    }
#else
    for (PetscInt f = 0; f < NbI; ++f) {
      const PetscInt i = offsetI + f; /* Element matrix row */

      for (PetscInt fc = 0; fc < NcI; ++fc) {
        const PetscInt fidx = f * NcI + fc; /* Test function basis index */

        for (PetscInt g = 0; g < NbJ; ++g) {
          const PetscInt j    = offsetJ + g; /* Element matrix column */
          const PetscInt fOff = i * totDim + j;

          for (PetscInt gc = 0; gc < NcJ; ++gc) {
            const PetscInt gidx = g * NcJ + gc; /* Trial function basis index */

            for (PetscInt df = 0; df < dE; ++df) {
              for (PetscInt dg = 0; dg < dE; ++dg) elemMat[fOff] += tmpBasisDerI[fidx * dE + df] * g3[((fc * NcJ + gc) * dE + df) * dE + dg] * tmpBasisDerJ[gidx * dE + dg];
            }
          }
        }
      }
    }
#endif
  }
  return PETSC_SUCCESS;
}

#undef petsc_elemmat_kernel_g1
#undef petsc_elemmat_kernel_g2
#undef petsc_elemmat_kernel_g3

PetscErrorCode PetscFEUpdateElementMat_Hybrid_Internal(PetscFE feI, PetscBool isHybridI, PetscFE feJ, PetscBool isHybridJ, PetscInt r, PetscInt s, PetscInt t, PetscInt q, PetscTabulation TI, PetscScalar tmpBasisI[], PetscScalar tmpBasisDerI[], PetscTabulation TJ, PetscScalar tmpBasisJ[], PetscScalar tmpBasisDerJ[], PetscFEGeom *fegeom, const PetscScalar g0[], const PetscScalar g1[], const PetscScalar g2[], const PetscScalar g3[], PetscInt eOffset, PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscScalar elemMat[])
{
  const PetscInt   dE        = TI->cdim;
  const PetscInt   NqI       = TI->Np;
  const PetscInt   NbI       = TI->Nb;
  const PetscInt   NcI       = TI->Nc;
  const PetscReal *basisI    = &TI->T[0][(r * NqI + q) * NbI * NcI];
  const PetscReal *basisDerI = &TI->T[1][(r * NqI + q) * NbI * NcI * dE];
  const PetscInt   NqJ       = TJ->Np;
  const PetscInt   NbJ       = TJ->Nb;
  const PetscInt   NcJ       = TJ->Nc;
  const PetscReal *basisJ    = &TJ->T[0][(r * NqJ + q) * NbJ * NcJ];
  const PetscReal *basisDerJ = &TJ->T[1][(r * NqJ + q) * NbJ * NcJ * dE];
  const PetscInt   so        = isHybridI ? 0 : s;
  const PetscInt   to        = isHybridJ ? 0 : t;
  PetscInt         f, fc, g, gc, df, dg;

  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f * NcI + fc; /* Test function basis index */

      tmpBasisI[fidx] = basisI[fidx];
      for (df = 0; df < dE; ++df) tmpBasisDerI[fidx * dE + df] = basisDerI[fidx * dE + df];
    }
  }
  PetscCall(PetscFEPushforward(feI, fegeom, NbI, tmpBasisI));
  PetscCall(PetscFEPushforwardGradient(feI, fegeom, NbI, tmpBasisDerI));
  for (g = 0; g < NbJ; ++g) {
    for (gc = 0; gc < NcJ; ++gc) {
      const PetscInt gidx = g * NcJ + gc; /* Trial function basis index */

      tmpBasisJ[gidx] = basisJ[gidx];
      for (dg = 0; dg < dE; ++dg) tmpBasisDerJ[gidx * dE + dg] = basisDerJ[gidx * dE + dg];
    }
  }
  PetscCall(PetscFEPushforward(feJ, fegeom, NbJ, tmpBasisJ));
  // TODO This is currently broken since we do not pull the geometry down to the lower dimension
  // PetscCall(PetscFEPushforwardGradient(feJ, fegeom, NbJ, tmpBasisDerJ));
  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f * NcI + fc;           /* Test function basis index */
      const PetscInt i    = offsetI + NbI * so + f; /* Element matrix row */
      for (g = 0; g < NbJ; ++g) {
        for (gc = 0; gc < NcJ; ++gc) {
          const PetscInt gidx = g * NcJ + gc;           /* Trial function basis index */
          const PetscInt j    = offsetJ + NbJ * to + g; /* Element matrix column */
          const PetscInt fOff = eOffset + i * totDim + j;

          elemMat[fOff] += tmpBasisI[fidx] * g0[fc * NcJ + gc] * tmpBasisJ[gidx];
          for (df = 0; df < dE; ++df) {
            elemMat[fOff] += tmpBasisI[fidx] * g1[(fc * NcJ + gc) * dE + df] * tmpBasisDerJ[gidx * dE + df];
            elemMat[fOff] += tmpBasisDerI[fidx * dE + df] * g2[(fc * NcJ + gc) * dE + df] * tmpBasisJ[gidx];
            for (dg = 0; dg < dE; ++dg) elemMat[fOff] += tmpBasisDerI[fidx * dE + df] * g3[((fc * NcJ + gc) * dE + df) * dE + dg] * tmpBasisDerJ[gidx * dE + dg];
          }
        }
      }
    }
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscFECreateCellGeometry(PetscFE fe, PetscQuadrature quad, PetscFEGeom *cgeom)
{
  PetscDualSpace  dsp;
  DM              dm;
  PetscQuadrature quadDef;
  PetscInt        dim, cdim, Nq;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDualSpace(fe, &dsp));
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(PetscFEGetQuadrature(fe, &quadDef));
  quad = quad ? quad : quadDef;
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
  PetscCall(PetscMalloc1(Nq * cdim, &cgeom->v));
  PetscCall(PetscMalloc1(Nq * cdim * cdim, &cgeom->J));
  PetscCall(PetscMalloc1(Nq * cdim * cdim, &cgeom->invJ));
  PetscCall(PetscMalloc1(Nq, &cgeom->detJ));
  cgeom->dim       = dim;
  cgeom->dimEmbed  = cdim;
  cgeom->numCells  = 1;
  cgeom->numPoints = Nq;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, 0, quad, cgeom->v, cgeom->J, cgeom->invJ, cgeom->detJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFEDestroyCellGeometry(PetscFE fe, PetscFEGeom *cgeom)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(cgeom->v));
  PetscCall(PetscFree(cgeom->J));
  PetscCall(PetscFree(cgeom->invJ));
  PetscCall(PetscFree(cgeom->detJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
PetscErrorCode PetscFEUpdateElementMat_Internal_SparseIndices(PetscTabulation TI, PetscTabulation TJ, PetscInt dimEmbed, const PetscInt g0[], const PetscInt g1[], const PetscInt g2[], const PetscInt g3[], PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscInt *n_g0, PetscInt **g0_idxs_out, PetscInt *n_g1, PetscInt **g1_idxs_out, PetscInt *n_g2, PetscInt **g2_idxs_out, PetscInt *n_g3, PetscInt **g3_idxs_out)
{
  const PetscInt dE      = dimEmbed;
  const PetscInt NbI     = TI->Nb;
  const PetscInt NcI     = TI->Nc;
  const PetscInt NbJ     = TJ->Nb;
  const PetscInt NcJ     = TJ->Nc;
  PetscBool      has_g0  = g0 ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      has_g1  = g1 ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      has_g2  = g2 ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      has_g3  = g3 ? PETSC_TRUE : PETSC_FALSE;
  PetscInt      *g0_idxs = NULL, *g1_idxs = NULL, *g2_idxs = NULL, *g3_idxs = NULL;
  PetscInt       g0_i, g1_i, g2_i, g3_i;

  PetscFunctionBegin;
  g0_i = g1_i = g2_i = g3_i = 0;
  if (has_g0)
    for (PetscInt i = 0; i < NcI * NcJ; i++)
      if (g0[i]) g0_i += NbI * NbJ;
  if (has_g1)
    for (PetscInt i = 0; i < NcI * NcJ * dE; i++)
      if (g1[i]) g1_i += NbI * NbJ;
  if (has_g2)
    for (PetscInt i = 0; i < NcI * NcJ * dE; i++)
      if (g2[i]) g2_i += NbI * NbJ;
  if (has_g3)
    for (PetscInt i = 0; i < NcI * NcJ * dE * dE; i++)
      if (g3[i]) g3_i += NbI * NbJ;
  if (g0_i == NbI * NbJ * NcI * NcJ) g0_i = 0;
  if (g1_i == NbI * NbJ * NcI * NcJ * dE) g1_i = 0;
  if (g2_i == NbI * NbJ * NcI * NcJ * dE) g2_i = 0;
  if (g3_i == NbI * NbJ * NcI * NcJ * dE * dE) g3_i = 0;
  has_g0 = g0_i ? PETSC_TRUE : PETSC_FALSE;
  has_g1 = g1_i ? PETSC_TRUE : PETSC_FALSE;
  has_g2 = g2_i ? PETSC_TRUE : PETSC_FALSE;
  has_g3 = g3_i ? PETSC_TRUE : PETSC_FALSE;
  if (has_g0) PetscCall(PetscMalloc1(4 * g0_i, &g0_idxs));
  if (has_g1) PetscCall(PetscMalloc1(4 * g1_i, &g1_idxs));
  if (has_g2) PetscCall(PetscMalloc1(4 * g2_i, &g2_idxs));
  if (has_g3) PetscCall(PetscMalloc1(4 * g3_i, &g3_idxs));
  g0_i = g1_i = g2_i = g3_i = 0;

  for (PetscInt f = 0; f < NbI; ++f) {
    const PetscInt i = offsetI + f; /* Element matrix row */
    for (PetscInt fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f * NcI + fc; /* Test function basis index */

      for (PetscInt g = 0; g < NbJ; ++g) {
        const PetscInt j    = offsetJ + g; /* Element matrix column */
        const PetscInt fOff = i * totDim + j;
        for (PetscInt gc = 0; gc < NcJ; ++gc) {
          const PetscInt gidx = g * NcJ + gc; /* Trial function basis index */

          if (has_g0) {
            if (g0[fc * NcJ + gc]) {
              g0_idxs[4 * g0_i + 0] = fidx;
              g0_idxs[4 * g0_i + 1] = fc * NcJ + gc;
              g0_idxs[4 * g0_i + 2] = gidx;
              g0_idxs[4 * g0_i + 3] = fOff;
              g0_i++;
            }
          }

          for (PetscInt df = 0; df < dE; ++df) {
            if (has_g1) {
              if (g1[(fc * NcJ + gc) * dE + df]) {
                g1_idxs[4 * g1_i + 0] = fidx;
                g1_idxs[4 * g1_i + 1] = (fc * NcJ + gc) * dE + df;
                g1_idxs[4 * g1_i + 2] = gidx * dE + df;
                g1_idxs[4 * g1_i + 3] = fOff;
                g1_i++;
              }
            }
            if (has_g2) {
              if (g2[(fc * NcJ + gc) * dE + df]) {
                g2_idxs[4 * g2_i + 0] = fidx * dE + df;
                g2_idxs[4 * g2_i + 1] = (fc * NcJ + gc) * dE + df;
                g2_idxs[4 * g2_i + 2] = gidx;
                g2_idxs[4 * g2_i + 3] = fOff;
                g2_i++;
              }
            }
            if (has_g3) {
              for (PetscInt dg = 0; dg < dE; ++dg) {
                if (g3[((fc * NcJ + gc) * dE + df) * dE + dg]) {
                  g3_idxs[4 * g3_i + 0] = fidx * dE + df;
                  g3_idxs[4 * g3_i + 1] = ((fc * NcJ + gc) * dE + df) * dE + dg;
                  g3_idxs[4 * g3_i + 2] = gidx * dE + dg;
                  g3_idxs[4 * g3_i + 3] = fOff;
                  g3_i++;
                }
              }
            }
          }
        }
      }
    }
  }
  *n_g0 = g0_i;
  *n_g1 = g1_i;
  *n_g2 = g2_i;
  *n_g3 = g3_i;

  *g0_idxs_out = g0_idxs;
  *g1_idxs_out = g1_idxs;
  *g2_idxs_out = g2_idxs;
  *g3_idxs_out = g3_idxs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

//example HOW TO USE
      for (PetscInt i = 0; i < g0_sparse_n; i++) {
        PetscInt bM = g0_sparse_idxs[4 * i + 0];
        PetscInt bN = g0_sparse_idxs[4 * i + 1];
        PetscInt bK = g0_sparse_idxs[4 * i + 2];
        PetscInt bO = g0_sparse_idxs[4 * i + 3];
        elemMat[bO] += tmpBasisI[bM] * g0[bN] * tmpBasisJ[bK];
      }
#endif
