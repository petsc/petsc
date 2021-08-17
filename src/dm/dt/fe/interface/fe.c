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

PetscClassId PETSCFE_CLASSID = 0;

PetscLogEvent PETSCFE_SetUp;

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

  Collective on fem

  Input Parameters:
+ fem  - The PetscFE object
- name - The kind of FEM space

  Options Database Key:
. -petscfe_type <type> - Sets the PetscFE type; use -help for a list of available types

  Level: intermediate

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
   PetscFEViewFromOptions - View from Options

   Collective on PetscFE

   Input Parameters:
+  A - the PetscFE object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscFE(), PetscFEView(), PetscObjectViewFromOptions(), PetscFECreate()
@*/
PetscErrorCode  PetscFEViewFromOptions(PetscFE A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCFE_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEView - Views a PetscFE

  Collective on fem

  Input Parameters:
+ fem - the PetscFE object to view
- viewer   - the viewer

  Level: beginner

.seealso PetscFEDestroy()
@*/
PetscErrorCode PetscFEView(PetscFE fem, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) fem), &viewer);CHKERRQ(ierr);}
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)fem, viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (fem->ops->view) {ierr = (*fem->ops->view)(fem, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscFESetFromOptions - sets parameters in a PetscFE from the options database

  Collective on fem

  Input Parameter:
. fem - the PetscFE object to set options for

  Options Database:
+ -petscfe_num_blocks  - the number of cell blocks to integrate concurrently
- -petscfe_num_batches - the number of cell batches to integrate serially

  Level: intermediate

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
  ierr = PetscOptionsBoundedInt("-petscfe_num_blocks", "The number of cell blocks to integrate concurrently", "PetscSpaceSetTileSizes", fem->numBlocks, &fem->numBlocks, NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscfe_num_batches", "The number of cell batches to integrate serially", "PetscSpaceSetTileSizes", fem->numBatches, &fem->numBatches, NULL,1);CHKERRQ(ierr);
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

  Collective on fem

  Input Parameter:
. fem - the PetscFE object to setup

  Level: intermediate

.seealso PetscFEView(), PetscFEDestroy()
@*/
PetscErrorCode PetscFESetUp(PetscFE fem)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (fem->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCFE_SetUp, fem, 0, 0, 0);CHKERRQ(ierr);
  fem->setupcalled = PETSC_TRUE;
  if (fem->ops->setup) {ierr = (*fem->ops->setup)(fem);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(PETSCFE_SetUp, fem, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFEDestroy - Destroys a PetscFE object

  Collective on fem

  Input Parameter:
. fem - the PetscFE object to destroy

  Level: beginner

.seealso PetscFEView()
@*/
PetscErrorCode PetscFEDestroy(PetscFE *fem)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fem) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*fem), PETSCFE_CLASSID, 1);

  if (--((PetscObject)(*fem))->refct > 0) {*fem = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*fem))->refct = 0;

  if ((*fem)->subspaces) {
    PetscInt dim, d;

    ierr = PetscDualSpaceGetDimension((*fem)->dualSpace, &dim);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {ierr = PetscFEDestroy(&(*fem)->subspaces[d]);CHKERRQ(ierr);}
  }
  ierr = PetscFree((*fem)->subspaces);CHKERRQ(ierr);
  ierr = PetscFree((*fem)->invV);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&(*fem)->T);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&(*fem)->Tf);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&(*fem)->Tc);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&(*fem)->basisSpace);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&(*fem)->dualSpace);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->faceQuadrature);CHKERRQ(ierr);
#ifdef PETSC_HAVE_LIBCEED
  ierr = CeedBasisDestroy(&(*fem)->ceedBasis);CHKERRQ(ierr);
  ierr = CeedDestroy(&(*fem)->ceed);CHKERRQ(ierr);
#endif

  if ((*fem)->ops->destroy) {ierr = (*(*fem)->ops->destroy)(*fem);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFECreate - Creates an empty PetscFE object. The type can then be set with PetscFESetType().

  Collective

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
  f->T             = NULL;
  f->Tf            = NULL;
  f->Tc            = NULL;
  ierr = PetscArrayzero(&f->quadrature, 1);CHKERRQ(ierr);
  ierr = PetscArrayzero(&f->faceQuadrature, 1);CHKERRQ(ierr);
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
  if (q == fem->quadrature) PetscFunctionReturn(0);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscQuadratureGetNumComponents(q, &qNc);CHKERRQ(ierr);
  if ((qNc != 1) && (Nc != qNc)) SETERRQ2(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_SIZ, "FE components %D != Quadrature components %D and non-scalar quadrature", Nc, qNc);
  ierr = PetscTabulationDestroy(&fem->T);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&fem->Tc);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fem->quadrature);CHKERRQ(ierr);
  fem->quadrature = q;
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
  PetscInt       Nc, qNc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscQuadratureGetNumComponents(q, &qNc);CHKERRQ(ierr);
  if ((qNc != 1) && (Nc != qNc)) SETERRQ2(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_SIZ, "FE components %D != Quadrature components %D and non-scalar quadrature", Nc, qNc);
  ierr = PetscTabulationDestroy(&fem->Tf);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fem->faceQuadrature);CHKERRQ(ierr);
  fem->faceQuadrature = q;
  ierr = PetscObjectReference((PetscObject) q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFECopyQuadrature - Copy both volumetric and surface quadrature

  Not collective

  Input Parameters:
+ sfe - The PetscFE source for the quadratures
- tfe - The PetscFE target for the quadratures

  Level: intermediate

.seealso: PetscFECreate(), PetscFESetQuadrature(), PetscFESetFaceQuadrature()
@*/
PetscErrorCode PetscFECopyQuadrature(PetscFE sfe, PetscFE tfe)
{
  PetscQuadrature q;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfe, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(tfe, PETSCFE_CLASSID, 2);
  ierr = PetscFEGetQuadrature(sfe, &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(tfe,  q);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(sfe, &q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(tfe,  q);CHKERRQ(ierr);
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
  PetscFEGetCellTabulation - Returns the tabulation of the basis functions at the quadrature points on the reference cell

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- k   - The highest derivative we need to tabulate, very often 1

  Output Parameter:
. T - The basis function values and derivatives at quadrature points

  Note:
$ T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFECreateTabulation(), PetscTabulationDestroy()
@*/
PetscErrorCode PetscFEGetCellTabulation(PetscFE fem, PetscInt k, PetscTabulation *T)
{
  PetscInt         npoints;
  const PetscReal *points;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(T, 3);
  ierr = PetscQuadratureGetData(fem->quadrature, NULL, NULL, &npoints, &points, NULL);CHKERRQ(ierr);
  if (!fem->T) {ierr = PetscFECreateTabulation(fem, 1, npoints, points, k, &fem->T);CHKERRQ(ierr);}
  if (fem->T && k > fem->T->K) SETERRQ2(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_OUTOFRANGE, "Requested %D derivatives, but only tabulated %D", k, fem->T->K);
  *T = fem->T;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetFaceTabulation - Returns the tabulation of the basis functions at the face quadrature points for each face of the reference cell

  Not collective

  Input Parameters:
+ fem - The PetscFE object
- k   - The highest derivative we need to tabulate, very often 1

  Output Parameters:
. Tf - The basis function values and derivatives at face quadrature points

  Note:
$ T->T[0] = Bf[((f*Nq + q)*pdim + i)*Nc + c] is the value at point f,q for basis function i and component c
$ T->T[1] = Df[(((f*Nq + q)*pdim + i)*Nc + c)*dim + d] is the derivative value at point f,q for basis function i, component c, in direction d
$ T->T[2] = Hf[((((f*Nq + q)*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point f,q for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFEGetCellTabulation(), PetscFECreateTabulation(), PetscTabulationDestroy()
@*/
PetscErrorCode PetscFEGetFaceTabulation(PetscFE fem, PetscInt k, PetscTabulation *Tf)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(Tf, 3);
  if (!fem->Tf) {
    const PetscReal  xi0[3] = {-1., -1., -1.};
    PetscReal        v0[3], J[9], detJ;
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
    if (fq) {
      ierr = PetscQuadratureGetData(fq, NULL, NULL, &npoints, &points, NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(numFaces*npoints*dim, &facePoints);CHKERRQ(ierr);
      for (f = 0; f < numFaces; ++f) {
        ierr = DMPlexComputeCellGeometryFEM(dm, faces[f], NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
        for (q = 0; q < npoints; ++q) CoordinatesRefToReal(dim, dim-1, xi0, v0, J, &points[q*(dim-1)], &facePoints[(f*npoints+q)*dim]);
      }
      ierr = PetscFECreateTabulation(fem, numFaces, npoints, facePoints, k, &fem->Tf);CHKERRQ(ierr);
      ierr = PetscFree(facePoints);CHKERRQ(ierr);
    }
  }
  if (fem->Tf && k > fem->Tf->K) SETERRQ2(PetscObjectComm((PetscObject) fem), PETSC_ERR_ARG_OUTOFRANGE, "Requested %D derivatives, but only tabulated %D", k, fem->Tf->K);
  *Tf = fem->Tf;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetFaceCentroidTabulation - Returns the tabulation of the basis functions at the face centroid points

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameters:
. Tc - The basis function values at face centroid points

  Note:
$ T->T[0] = Bf[(f*pdim + i)*Nc + c] is the value at point f for basis function i and component c

  Level: intermediate

.seealso: PetscFEGetFaceTabulation(), PetscFEGetCellTabulation(), PetscFECreateTabulation(), PetscTabulationDestroy()
@*/
PetscErrorCode PetscFEGetFaceCentroidTabulation(PetscFE fem, PetscTabulation *Tc)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(Tc, 2);
  if (!fem->Tc) {
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
    ierr = PetscFECreateTabulation(fem, 1, numFaces, centroids, 0, &fem->Tc);CHKERRQ(ierr);
    ierr = PetscFree(centroids);CHKERRQ(ierr);
  }
  *Tc = fem->Tc;
  PetscFunctionReturn(0);
}

/*@C
  PetscFECreateTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not collective

  Input Parameters:
+ fem     - The PetscFE object
. nrepl   - The number of replicas
. npoints - The number of tabulation points in a replica
. points  - The tabulation point coordinates
- K       - The number of derivatives calculated

  Output Parameter:
. T - The basis function values and derivatives at tabulation points

  Note:
$ T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFEGetCellTabulation(), PetscTabulationDestroy()
@*/
PetscErrorCode PetscFECreateTabulation(PetscFE fem, PetscInt nrepl, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation *T)
{
  DM               dm;
  PetscDualSpace   Q;
  PetscInt         Nb;   /* Dimension of FE space P */
  PetscInt         Nc;   /* Field components */
  PetscInt         cdim; /* Reference coordinate dimension */
  PetscInt         k;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!npoints || !fem->dualSpace || K < 0) {
    *T = NULL;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 4);
  PetscValidPointer(T, 6);
  ierr = PetscFEGetDualSpace(fem, &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(Q, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &cdim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(Q, &Nb);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  ierr = PetscMalloc1(1, T);CHKERRQ(ierr);
  (*T)->K    = !cdim ? 0 : K;
  (*T)->Nr   = nrepl;
  (*T)->Np   = npoints;
  (*T)->Nb   = Nb;
  (*T)->Nc   = Nc;
  (*T)->cdim = cdim;
  ierr = PetscMalloc1((*T)->K+1, &(*T)->T);CHKERRQ(ierr);
  for (k = 0; k <= (*T)->K; ++k) {
    ierr = PetscMalloc1(nrepl*npoints*Nb*Nc*PetscPowInt(cdim, k), &(*T)->T[k]);CHKERRQ(ierr);
  }
  ierr = (*fem->ops->createtabulation)(fem, nrepl*npoints, points, K, *T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEComputeTabulation - Tabulates the basis functions, and perhaps derivatives, at the points provided.

  Not collective

  Input Parameters:
+ fem     - The PetscFE object
. npoints - The number of tabulation points
. points  - The tabulation point coordinates
. K       - The number of derivatives calculated
- T       - An existing tabulation object with enough allocated space

  Output Parameter:
. T - The basis function values and derivatives at tabulation points

  Note:
$ T->T[0] = B[(p*pdim + i)*Nc + c] is the value at point p for basis function i and component c
$ T->T[1] = D[((p*pdim + i)*Nc + c)*dim + d] is the derivative value at point p for basis function i, component c, in direction d
$ T->T[2] = H[(((p*pdim + i)*Nc + c)*dim + d)*dim + e] is the value at point p for basis function i, component c, in directions d and e

  Level: intermediate

.seealso: PetscFEGetCellTabulation(), PetscTabulationDestroy()
@*/
PetscErrorCode PetscFEComputeTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (!npoints || !fem->dualSpace || K < 0) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 3);
  PetscValidPointer(T, 5);
  if (PetscDefined(USE_DEBUG)) {
    DM               dm;
    PetscDualSpace   Q;
    PetscInt         Nb;   /* Dimension of FE space P */
    PetscInt         Nc;   /* Field components */
    PetscInt         cdim; /* Reference coordinate dimension */

    ierr = PetscFEGetDualSpace(fem, &Q);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDM(Q, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &cdim);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(Q, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
    if (T->K    != (!cdim ? 0 : K)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation K %D must match requested K %D", T->K, !cdim ? 0 : K);
    if (T->Nb   != Nb)              SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation Nb %D must match requested Nb %D", T->Nb, Nb);
    if (T->Nc   != Nc)              SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation Nc %D must match requested Nc %D", T->Nc, Nc);
    if (T->cdim != cdim)            SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Tabulation cdim %D must match requested cdim %D", T->cdim, cdim);
  }
  T->Nr = 1;
  T->Np = npoints;
  ierr = (*fem->ops->createtabulation)(fem, npoints, points, K, T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscTabulationDestroy - Frees memory from the associated tabulation.

  Not collective

  Input Parameter:
. T - The tabulation

  Level: intermediate

.seealso: PetscFECreateTabulation(), PetscFEGetCellTabulation()
@*/
PetscErrorCode PetscTabulationDestroy(PetscTabulation *T)
{
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(T, 1);
  if (!T || !(*T)) PetscFunctionReturn(0);
  for (k = 0; k <= (*T)->K; ++k) {ierr = PetscFree((*T)->T[k]);CHKERRQ(ierr);}
  ierr = PetscFree((*T)->T);CHKERRQ(ierr);
  ierr = PetscFree(*T);CHKERRQ(ierr);
  *T = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFECreatePointTrace(PetscFE fe, PetscInt refPoint, PetscFE *trFE)
{
  PetscSpace     bsp, bsubsp;
  PetscDualSpace dsp, dsubsp;
  PetscInt       dim, depth, numComp, i, j, coneSize, order;
  PetscFEType    type;
  DM             dm;
  DMLabel        label;
  PetscReal      *xi, *v, *J, detJ;
  const char     *name;
  PetscQuadrature origin, fullQuad, subQuad;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe,PETSCFE_CLASSID,1);
  PetscValidPointer(trFE,3);
  ierr = PetscFEGetBasisSpace(fe,&bsp);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe,&dsp);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dsp,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm,&label);CHKERRQ(ierr);
  ierr = DMLabelGetValue(label,refPoint,&depth);CHKERRQ(ierr);
  ierr = PetscCalloc1(depth,&xi);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim,&v);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim,&J);CHKERRQ(ierr);
  for (i = 0; i < depth; i++) xi[i] = 0.;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&origin);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(origin,depth,0,1,xi,NULL);CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(dm,refPoint,origin,v,J,NULL,&detJ);CHKERRQ(ierr);
  /* CellGeometryFEM computes the expanded Jacobian, we want the true jacobian */
  for (i = 1; i < dim; i++) {
    for (j = 0; j < depth; j++) {
      J[i * depth + j] = J[i * dim + j];
    }
  }
  ierr = PetscQuadratureDestroy(&origin);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetPointSubspace(dsp,refPoint,&dsubsp);CHKERRQ(ierr);
  ierr = PetscSpaceCreateSubspace(bsp,dsubsp,v,J,NULL,NULL,PETSC_OWN_POINTER,&bsubsp);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(bsubsp);CHKERRQ(ierr);
  ierr = PetscFECreate(PetscObjectComm((PetscObject)fe),trFE);CHKERRQ(ierr);
  ierr = PetscFEGetType(fe,&type);CHKERRQ(ierr);
  ierr = PetscFESetType(*trFE,type);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe,&numComp);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*trFE,numComp);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*trFE,bsubsp);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*trFE,dsubsp);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) fe, &name);CHKERRQ(ierr);
  if (name) {ierr = PetscFESetName(*trFE, name);CHKERRQ(ierr);}
  ierr = PetscFEGetQuadrature(fe,&fullQuad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(fullQuad,&order);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm,refPoint,&coneSize);CHKERRQ(ierr);
  if (coneSize == 2 * depth) {
    ierr = PetscDTGaussTensorQuadrature(depth,1,(order + 1)/2,-1.,1.,&subQuad);CHKERRQ(ierr);
  } else {
    ierr = PetscDTStroudConicalQuadrature(depth,1,(order + 1)/2,-1.,1.,&subQuad);CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*trFE,subQuad);CHKERRQ(ierr);
  ierr = PetscFESetUp(*trFE);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&subQuad);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&bsubsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFECreateHeightTrace(PetscFE fe, PetscInt height, PetscFE *trFE)
{
  PetscInt       hStart, hEnd;
  PetscDualSpace dsp;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe,PETSCFE_CLASSID,1);
  PetscValidPointer(trFE,3);
  *trFE = NULL;
  ierr = PetscFEGetDualSpace(fe,&dsp);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dsp,&dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,height,&hStart,&hEnd);CHKERRQ(ierr);
  if (hEnd <= hStart) PetscFunctionReturn(0);
  ierr = PetscFECreatePointTrace(fe,hStart,trFE);CHKERRQ(ierr);
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

/*@C
  PetscFEPushforward - Map the reference element function to real space

  Input Parameters:
+ fe     - The PetscFE
. fegeom - The cell geometry
. Nv     - The number of function values
- vals   - The function values

  Output Parameter:
. vals   - The transformed function values

  Level: advanced

  Note: This just forwards the call onto PetscDualSpacePushforward().

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscDualSpacePushforward()
@*/
PetscErrorCode PetscFEPushforward(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDualSpacePushforward(fe->dualSpace, fegeom, Nv, fe->numComponents, vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEPushforwardGradient - Map the reference element function gradient to real space

  Input Parameters:
+ fe     - The PetscFE
. fegeom - The cell geometry
. Nv     - The number of function gradient values
- vals   - The function gradient values

  Output Parameter:
. vals   - The transformed function gradient values

  Level: advanced

  Note: This just forwards the call onto PetscDualSpacePushforwardGradient().

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscFEPushforward(), PetscDualSpacePushforwardGradient(), PetscDualSpacePushforward()
@*/
PetscErrorCode PetscFEPushforwardGradient(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDualSpacePushforwardGradient(fe->dualSpace, fegeom, Nv, fe->numComponents, vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFEPushforwardHessian - Map the reference element function Hessian to real space

  Input Parameters:
+ fe     - The PetscFE
. fegeom - The cell geometry
. Nv     - The number of function Hessian values
- vals   - The function Hessian values

  Output Parameter:
. vals   - The transformed function Hessian values

  Level: advanced

  Note: This just forwards the call onto PetscDualSpacePushforwardHessian().

  Note: This only handles transformations when the embedding dimension of the geometry in fegeom is the same as the reference dimension.

.seealso: PetscFEPushforward(), PetscDualSpacePushforwardHessian(), PetscDualSpacePushforward()
@*/
PetscErrorCode PetscFEPushforwardHessian(PetscFE fe, PetscFEGeom *fegeom, PetscInt Nv, PetscScalar vals[])
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDualSpacePushforwardHessian(fe->dualSpace, fegeom, Nv, fe->numComponents, vals);CHKERRQ(ierr);
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

/*@C
  PetscFEIntegrate - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter:
. integral     - the integral for this field

  Level: intermediate

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrate(PetscDS prob, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integrate) {ierr = (*fe->ops->integrate)(prob, field, Ne, cgeom, coefficients, probAux, coefficientsAux, integral);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateBd - Produce the integral for the given field for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ prob         - The PetscDS specifying the discretizations and continuum functions
. field        - The field being integrated
. obj_func     - The function to be integrated
. Ne           - The number of elements in the chunk
. fgeom        - The face geometry for each face in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
- coefficientsAux - The array of FEM auxiliary basis coefficients for the elements

  Output Parameter:
. integral     - the integral for this field

  Level: intermediate

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBd(PetscDS prob, PetscInt field,
                                  void (*obj_func)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                  PetscInt Ne, PetscFEGeom *geom, const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratebd) {ierr = (*fe->ops->integratebd)(prob, field, obj_func, Ne, geom, coefficients, probAux, coefficientsAux, integral);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateResidual - Produce the element residual vector for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ ds           - The PetscDS specifying the discretizations and continuum functions
. key          - The (label+value, field) being integrated
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t            - The time

  Output Parameter:
. elemVec      - the element residual vectors from each element

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)

  Level: intermediate

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateResidual(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetDiscretization(ds, key.field, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integrateresidual) {ierr = (*fe->ops->integrateresidual)(ds, key, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateBdResidual - Produce the element residual vector for a chunk of elements by quadrature integration over a boundary

  Not collective

  Input Parameters:
+ ds           - The PetscDS specifying the discretizations and continuum functions
. wf           - The PetscWeakForm object holding the pointwise functions
. key          - The (label+value, field) being integrated
. Ne           - The number of elements in the chunk
. fgeom        - The face geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
- t            - The time

  Output Parameter:
. elemVec      - the element residual vectors from each element

  Level: intermediate

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBdResidual(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetDiscretization(ds, key.field, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratebdresidual) {ierr = (*fe->ops->integratebdresidual)(ds, wf, key, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateHybridResidual - Produce the element residual vector for a chunk of hybrid element faces by quadrature integration

  Not collective

  Input Parameters:
+ prob         - The PetscDS specifying the discretizations and continuum functions
. key          - The (label+value, field) being integrated
. s            - The side of the cell being integrated, 0 for negative and 1 for positive
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
PetscErrorCode PetscFEIntegrateHybridResidual(PetscDS prob, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetDiscretization(prob, key.field, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratehybridresidual) {ierr = (*fe->ops->integratehybridresidual)(prob, key, s, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, elemVec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateJacobian - Produce the element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ ds           - The PetscDS specifying the discretizations and continuum functions
. jtype        - The type of matrix pointwise functions that should be used
. key          - The (label+value, fieldI*Nf + fieldJ) being integrated
. s            - The side of the cell being integrated, 0 for negative and 1 for positive
. Ne           - The number of elements in the chunk
. cgeom        - The cell geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t            - The time
- u_tShift     - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter:
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
  Level: intermediate

.seealso: PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateJacobian(PetscDS ds, PetscFEJacobianType jtype, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom,
                                        const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE        fe;
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratejacobian) {ierr = (*fe->ops->integratejacobian)(ds, jtype, key, Ne, cgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateBdJacobian - Produce the boundary element Jacobian for a chunk of elements by quadrature integration

  Not collective

  Input Parameters:
+ ds           - The PetscDS specifying the discretizations and continuum functions
. wf           - The PetscWeakForm holding the pointwise functions
. key          - The (label+value, fieldI*Nf + fieldJ) being integrated
. Ne           - The number of elements in the chunk
. fgeom        - The face geometry for each cell in the chunk
. coefficients - The array of FEM basis coefficients for the elements for the Jacobian evaluation point
. coefficients_t - The array of FEM basis time derivative coefficients for the elements
. probAux      - The PetscDS specifying the auxiliary discretizations
. coefficientsAux - The array of FEM auxiliary basis coefficients for the elements
. t            - The time
- u_tShift     - A multiplier for the dF/du_t term (as opposed to the dF/du term)

  Output Parameter:
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
  Level: intermediate

.seealso: PetscFEIntegrateJacobian(), PetscFEIntegrateResidual()
@*/
PetscErrorCode PetscFEIntegrateBdJacobian(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom,
                                          const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE        fe;
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratebdjacobian) {ierr = (*fe->ops->integratebdjacobian)(ds, wf, key, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscFEIntegrateHybridJacobian - Produce the boundary element Jacobian for a chunk of hybrid elements by quadrature integration

  Not collective

  Input Parameters:
+ ds           - The PetscDS specifying the discretizations and continuum functions
. jtype        - The type of matrix pointwise functions that should be used
. key          - The (label+value, fieldI*Nf + fieldJ) being integrated
. s            - The side of the cell being integrated, 0 for negative and 1 for positive
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
PetscErrorCode PetscFEIntegrateHybridJacobian(PetscDS ds, PetscFEJacobianType jtype, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE        fe;
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *) &fe);CHKERRQ(ierr);
  if (fe->ops->integratehybridjacobian) {ierr = (*fe->ops->integratehybridjacobian)(ds, jtype, key, s, Ne, fgeom, coefficients, coefficients_t, probAux, coefficientsAux, t, u_tshift, elemMat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscFEGetHeightSubspace - Get the subspace of this space for a mesh point of a given height

  Input Parameters:
+ fe     - The finite element space
- height - The height of the Plex point

  Output Parameter:
. subfe  - The subspace of this FE space

  Note: For example, if we want the subspace of this space for a face, we would choose height = 1.

  Level: advanced

.seealso: PetscFECreateDefault()
@*/
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
  if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!fe->subspaces) {ierr = PetscCalloc1(dim, &fe->subspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!fe->subspaces[height-1]) {
      PetscFE     sub = NULL;
      const char *name;

      ierr = PetscSpaceGetHeightSubspace(P, height, &subP);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetHeightSubspace(Q, height, &subQ);CHKERRQ(ierr);
      if (subQ) {
        ierr = PetscFECreate(PetscObjectComm((PetscObject) fe), &sub);CHKERRQ(ierr);
        ierr = PetscObjectGetName((PetscObject) fe,  &name);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
        ierr = PetscFEGetType(fe, &fetype);CHKERRQ(ierr);
        ierr = PetscFESetType(sub, fetype);CHKERRQ(ierr);
        ierr = PetscFESetBasisSpace(sub, subP);CHKERRQ(ierr);
        ierr = PetscFESetDualSpace(sub, subQ);CHKERRQ(ierr);
        ierr = PetscFESetNumComponents(sub, Nc);CHKERRQ(ierr);
        ierr = PetscFESetUp(sub);CHKERRQ(ierr);
        ierr = PetscFESetQuadrature(sub, subq);CHKERRQ(ierr);
      }
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

  Collective on fem

  Input Parameter:
. fe - The initial PetscFE

  Output Parameter:
. feRef - The refined PetscFE

  Level: advanced

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
  PetscInt         cStart, cEnd, c;
  PetscDualSpace  *cellSpaces;
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
  ierr = PetscDualSpaceSetType(Qref, PETSCDUALSPACEREFINED);CHKERRQ(ierr);
  ierr = DMRefine(K, PetscObjectComm((PetscObject) fe), &Kref);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Qref, Kref);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(Kref, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(cEnd - cStart, &cellSpaces);CHKERRQ(ierr);
  /* TODO: fix for non-uniform refinement */
  for (c = 0; c < cEnd - cStart; c++) cellSpaces[c] = Q;
  ierr = PetscDualSpaceRefinedSetCellSpaces(Qref, cellSpaces);CHKERRQ(ierr);
  ierr = PetscFree(cellSpaces);CHKERRQ(ierr);
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

  Collective

  Input Parameters:
+ comm      - The MPI comm
. dim       - The spatial dimension
. Nc        - The number of components
. isSimplex - Flag for simplex reference cell, otherwise its a tensor product
. prefix    - The options prefix, or NULL
- qorder    - The quadrature order or PETSC_DETERMINE to use PetscSpace polynomial degree

  Output Parameter:
. fem - The PetscFE object

  Note:
  Each subobject is SetFromOption() during creation, so that the object may be customized from the command line, using the prefix specified above. See the links below for the particular options available.

  Level: beginner

.seealso: PetscSpaceSetFromOptions(), PetscDualSpaceSetFromOptions(), PetscFESetFromOptions(), PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscFECreateDefault(MPI_Comm comm, PetscInt dim, PetscInt Nc, PetscBool isSimplex, const char prefix[], PetscInt qorder, PetscFE *fem)
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
  ierr = PetscSpaceCreate(comm, &P);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(P, &order, NULL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(P, &tensor);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
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
  ierr = PetscFECreate(comm, fem);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(*fem);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  qorder = qorder >= 0 ? qorder : order;
  ierr = PetscObjectOptionsBegin((PetscObject)*fem);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscfe_default_quadrature_order","Quadrature order is one less than quadrature points per edge","PetscFECreateDefault",qorder,&qorder,NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  quadPointsPerEdge = PetscMax(qorder + 1,1);
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFECreateLagrange - Create a PetscFE for the basic Lagrange space of degree k

  Collective

  Input Parameters:
+ comm      - The MPI comm
. dim       - The spatial dimension
. Nc        - The number of components
. isSimplex - Flag for simplex reference cell, otherwise its a tensor product
. k         - The degree k of the space
- qorder    - The quadrature order or PETSC_DETERMINE to use PetscSpace polynomial degree

  Output Parameter:
. fem       - The PetscFE object

  Level: beginner

  Notes:
  For simplices, this element is the space of maximum polynomial degree k, otherwise it is a tensor product of 1D polynomials, each with maximal degree k.

.seealso: PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscFECreateLagrange(MPI_Comm comm, PetscInt dim, PetscInt Nc, PetscBool isSimplex, PetscInt k, PetscInt qorder, PetscFE *fem)
{
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  char            name[64];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(comm, &P);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, k, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, k);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreate(comm, fem);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name, sizeof(name), "%s%D", isSimplex? "P" : "Q", k);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *fem, name);CHKERRQ(ierr);
  ierr = PetscFESetType(*fem, PETSCFEBASIC);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  qorder = qorder >= 0 ? qorder : k;
  quadPointsPerEdge = PetscMax(qorder + 1,1);
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0, &fq);CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq);CHKERRQ(ierr);
  /* Set finite element name */
  ierr = PetscSNPrintf(name, sizeof(name), "%s%D", isSimplex? "P" : "Q", k);CHKERRQ(ierr);
  ierr = PetscFESetName(*fem, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscFESetName - Names the FE and its subobjects

  Not collective

  Input Parameters:
+ fe   - The PetscFE
- name - The name

  Level: intermediate

.seealso: PetscFECreate(), PetscSpaceCreate(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscFESetName(PetscFE fe, const char name[])
{
  PetscSpace     P;
  PetscDualSpace Q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetBasisSpace(fe, &P);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) P,  name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Q,  name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEEvaluateFieldJets_Internal(PetscDS ds, PetscInt Nf, PetscInt r, PetscInt q, PetscTabulation T[], PetscFEGeom *fegeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscInt       dOffset = 0, fOffset = 0, f, g;
  PetscErrorCode ierr;

  for (f = 0; f < Nf; ++f) {
    PetscFE          fe;
    const PetscInt   k    = ds->jetDegree[f];
    const PetscInt   cdim = T[f]->cdim;
    const PetscInt   Nq   = T[f]->Np;
    const PetscInt   Nbf  = T[f]->Nb;
    const PetscInt   Ncf  = T[f]->Nc;
    const PetscReal *Bq   = &T[f]->T[0][(r*Nq+q)*Nbf*Ncf];
    const PetscReal *Dq   = &T[f]->T[1][(r*Nq+q)*Nbf*Ncf*cdim];
    const PetscReal *Hq   = k > 1 ? &T[f]->T[2][(r*Nq+q)*Nbf*Ncf*cdim*cdim] : NULL;
    PetscInt         hOffset = 0, b, c, d;

    ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
    for (c = 0; c < Ncf; ++c) u[fOffset+c] = 0.0;
    for (d = 0; d < cdim*Ncf; ++d) u_x[fOffset*cdim+d] = 0.0;
    for (b = 0; b < Nbf; ++b) {
      for (c = 0; c < Ncf; ++c) {
        const PetscInt cidx = b*Ncf+c;

        u[fOffset+c] += Bq[cidx]*coefficients[dOffset+b];
        for (d = 0; d < cdim; ++d) u_x[(fOffset+c)*cdim+d] += Dq[cidx*cdim+d]*coefficients[dOffset+b];
      }
    }
    if (k > 1) {
      for (g = 0; g < Nf; ++g) hOffset += T[g]->Nc*cdim;
      for (d = 0; d < cdim*cdim*Ncf; ++d) u_x[hOffset+fOffset*cdim*cdim+d] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b*Ncf+c;

          for (d = 0; d < cdim*cdim; ++d) u_x[hOffset+(fOffset+c)*cdim*cdim+d] += Hq[cidx*cdim*cdim+d]*coefficients[dOffset+b];
        }
      }
      ierr = PetscFEPushforwardHessian(fe, fegeom, 1, &u_x[hOffset+fOffset*cdim*cdim]);CHKERRQ(ierr);
    }
    ierr = PetscFEPushforward(fe, fegeom, 1, &u[fOffset]);CHKERRQ(ierr);
    ierr = PetscFEPushforwardGradient(fe, fegeom, 1, &u_x[fOffset*cdim]);CHKERRQ(ierr);
    if (u_t) {
      for (c = 0; c < Ncf; ++c) u_t[fOffset+c] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b*Ncf+c;

          u_t[fOffset+c] += Bq[cidx]*coefficients_t[dOffset+b];
        }
      }
      ierr = PetscFEPushforward(fe, fegeom, 1, &u_t[fOffset]);CHKERRQ(ierr);
    }
    fOffset += Ncf;
    dOffset += Nbf;
  }
  return 0;
}

PetscErrorCode PetscFEEvaluateFieldJets_Hybrid_Internal(PetscDS ds, PetscInt Nf, PetscInt r, PetscInt q, PetscTabulation T[], PetscFEGeom *fegeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscScalar u[], PetscScalar u_x[], PetscScalar u_t[])
{
  PetscInt       dOffset = 0, fOffset = 0, f, g;
  PetscErrorCode ierr;

  /* f is the field number in the DS, g is the field number in u[] */
  for (f = 0, g = 0; f < Nf; ++f) {
    PetscFE          fe   = (PetscFE) ds->disc[f];
    const PetscInt   cdim = T[f]->cdim;
    const PetscInt   Nq   = T[f]->Np;
    const PetscInt   Nbf  = T[f]->Nb;
    const PetscInt   Ncf  = T[f]->Nc;
    const PetscReal *Bq   = &T[f]->T[0][(r*Nq+q)*Nbf*Ncf];
    const PetscReal *Dq   = &T[f]->T[1][(r*Nq+q)*Nbf*Ncf*cdim];
    PetscBool        isCohesive;
    PetscInt         Ns, s;

    if (!T[f]) continue;
    ierr = PetscDSGetCohesive(ds, f, &isCohesive);CHKERRQ(ierr);
    Ns   = isCohesive ? 1 : 2;
    for (s = 0; s < Ns; ++s, ++g) {
      PetscInt b, c, d;

      for (c = 0; c < Ncf; ++c)      u[fOffset+c] = 0.0;
      for (d = 0; d < cdim*Ncf; ++d) u_x[fOffset*cdim+d] = 0.0;
      for (b = 0; b < Nbf; ++b) {
        for (c = 0; c < Ncf; ++c) {
          const PetscInt cidx = b*Ncf+c;

          u[fOffset+c] += Bq[cidx]*coefficients[dOffset+b];
          for (d = 0; d < cdim; ++d) u_x[(fOffset+c)*cdim+d] += Dq[cidx*cdim+d]*coefficients[dOffset+b];
        }
      }
      ierr = PetscFEPushforward(fe, fegeom, 1, &u[fOffset]);CHKERRQ(ierr);
      ierr = PetscFEPushforwardGradient(fe, fegeom, 1, &u_x[fOffset*cdim]);CHKERRQ(ierr);
      if (u_t) {
        for (c = 0; c < Ncf; ++c) u_t[fOffset+c] = 0.0;
        for (b = 0; b < Nbf; ++b) {
          for (c = 0; c < Ncf; ++c) {
            const PetscInt cidx = b*Ncf+c;

            u_t[fOffset+c] += Bq[cidx]*coefficients_t[dOffset+b];
          }
        }
        ierr = PetscFEPushforward(fe, fegeom, 1, &u_t[fOffset]);CHKERRQ(ierr);
      }
      fOffset += Ncf;
      dOffset += Nbf;
    }
  }
  return 0;
}

PetscErrorCode PetscFEEvaluateFaceFields_Internal(PetscDS prob, PetscInt field, PetscInt faceLoc, const PetscScalar coefficients[], PetscScalar u[])
{
  PetscFE         fe;
  PetscTabulation Tc;
  PetscInt        b, c;
  PetscErrorCode  ierr;

  if (!prob) return 0;
  ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetFaceCentroidTabulation(fe, &Tc);CHKERRQ(ierr);
  {
    const PetscReal *faceBasis = Tc->T[0];
    const PetscInt   Nb        = Tc->Nb;
    const PetscInt   Nc        = Tc->Nc;

    for (c = 0; c < Nc; ++c) {u[c] = 0.0;}
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        u[c] += coefficients[b] * faceBasis[(faceLoc*Nb + b)*Nc + c];
      }
    }
  }
  return 0;
}

PetscErrorCode PetscFEUpdateElementVec_Internal(PetscFE fe, PetscTabulation T, PetscInt r, PetscScalar tmpBasis[], PetscScalar tmpBasisDer[], PetscInt e, PetscFEGeom *fegeom, PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  PetscFEGeom      pgeom;
  const PetscInt   dEt      = T->cdim;
  const PetscInt   dE       = fegeom->dimEmbed;
  const PetscInt   Nq       = T->Np;
  const PetscInt   Nb       = T->Nb;
  const PetscInt   Nc       = T->Nc;
  const PetscReal *basis    = &T->T[0][r*Nq*Nb*Nc];
  const PetscReal *basisDer = &T->T[1][r*Nq*Nb*Nc*dEt];
  PetscInt         q, b, c, d;
  PetscErrorCode   ierr;

  for (b = 0; b < Nb; ++b) elemVec[b] = 0.0;
  for (q = 0; q < Nq; ++q) {
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;

        tmpBasis[bcidx] = basis[q*Nb*Nc+bcidx];
        for (d = 0; d < dEt; ++d) tmpBasisDer[bcidx*dE+d] = basisDer[q*Nb*Nc*dEt+bcidx*dEt+d];
      }
    }
    ierr = PetscFEGeomGetCellPoint(fegeom, e, q, &pgeom);CHKERRQ(ierr);
    ierr = PetscFEPushforward(fe, &pgeom, Nb, tmpBasis);CHKERRQ(ierr);
    ierr = PetscFEPushforwardGradient(fe, &pgeom, Nb, tmpBasisDer);CHKERRQ(ierr);
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;
        const PetscInt qcidx = q*Nc+c;

        elemVec[b] += tmpBasis[bcidx]*f0[qcidx];
        for (d = 0; d < dE; ++d) elemVec[b] += tmpBasisDer[bcidx*dE+d]*f1[qcidx*dE+d];
      }
    }
  }
  return(0);
}

PetscErrorCode PetscFEUpdateElementVec_Hybrid_Internal(PetscFE fe, PetscTabulation T, PetscInt r, PetscInt s, PetscScalar tmpBasis[], PetscScalar tmpBasisDer[], PetscFEGeom *fegeom, PetscScalar f0[], PetscScalar f1[], PetscScalar elemVec[])
{
  const PetscInt   dE       = T->cdim;
  const PetscInt   Nq       = T->Np;
  const PetscInt   Nb       = T->Nb;
  const PetscInt   Nc       = T->Nc;
  const PetscReal *basis    = &T->T[0][r*Nq*Nb*Nc];
  const PetscReal *basisDer = &T->T[1][r*Nq*Nb*Nc*dE];
  PetscInt         q, b, c, d;
  PetscErrorCode   ierr;

  for (b = 0; b < Nb; ++b) elemVec[Nb*s+b] = 0.0;
  for (q = 0; q < Nq; ++q) {
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;

        tmpBasis[bcidx] = basis[q*Nb*Nc+bcidx];
        for (d = 0; d < dE; ++d) tmpBasisDer[bcidx*dE+d] = basisDer[q*Nb*Nc*dE+bcidx*dE+d];
      }
    }
    ierr = PetscFEPushforward(fe, fegeom, Nb, tmpBasis);CHKERRQ(ierr);
    ierr = PetscFEPushforwardGradient(fe, fegeom, Nb, tmpBasisDer);CHKERRQ(ierr);
    for (b = 0; b < Nb; ++b) {
      for (c = 0; c < Nc; ++c) {
        const PetscInt bcidx = b*Nc+c;
        const PetscInt qcidx = q*Nc+c;

        elemVec[Nb*s+b] += tmpBasis[bcidx]*f0[qcidx];
        for (d = 0; d < dE; ++d) elemVec[Nb*s+b] += tmpBasisDer[bcidx*dE+d]*f1[qcidx*dE+d];
      }
    }
  }
  return(0);
}

PetscErrorCode PetscFEUpdateElementMat_Internal(PetscFE feI, PetscFE feJ, PetscInt r, PetscInt q, PetscTabulation TI, PetscScalar tmpBasisI[], PetscScalar tmpBasisDerI[], PetscTabulation TJ, PetscScalar tmpBasisJ[], PetscScalar tmpBasisDerJ[], PetscFEGeom *fegeom, const PetscScalar g0[], const PetscScalar g1[], const PetscScalar g2[], const PetscScalar g3[], PetscInt eOffset, PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscScalar elemMat[])
{
  const PetscInt   dE        = TI->cdim;
  const PetscInt   NqI       = TI->Np;
  const PetscInt   NbI       = TI->Nb;
  const PetscInt   NcI       = TI->Nc;
  const PetscReal *basisI    = &TI->T[0][(r*NqI+q)*NbI*NcI];
  const PetscReal *basisDerI = &TI->T[1][(r*NqI+q)*NbI*NcI*dE];
  const PetscInt   NqJ       = TJ->Np;
  const PetscInt   NbJ       = TJ->Nb;
  const PetscInt   NcJ       = TJ->Nc;
  const PetscReal *basisJ    = &TJ->T[0][(r*NqJ+q)*NbJ*NcJ];
  const PetscReal *basisDerJ = &TJ->T[1][(r*NqJ+q)*NbJ*NcJ*dE];
  PetscInt         f, fc, g, gc, df, dg;
  PetscErrorCode   ierr;

  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc; /* Test function basis index */

      tmpBasisI[fidx] = basisI[fidx];
      for (df = 0; df < dE; ++df) tmpBasisDerI[fidx*dE+df] = basisDerI[fidx*dE+df];
    }
  }
  ierr = PetscFEPushforward(feI, fegeom, NbI, tmpBasisI);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feI, fegeom, NbI, tmpBasisDerI);CHKERRQ(ierr);
  for (g = 0; g < NbJ; ++g) {
    for (gc = 0; gc < NcJ; ++gc) {
      const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */

      tmpBasisJ[gidx] = basisJ[gidx];
      for (dg = 0; dg < dE; ++dg) tmpBasisDerJ[gidx*dE+dg] = basisDerJ[gidx*dE+dg];
    }
  }
  ierr = PetscFEPushforward(feJ, fegeom, NbJ, tmpBasisJ);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feJ, fegeom, NbJ, tmpBasisDerJ);CHKERRQ(ierr);
  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc; /* Test function basis index */
      const PetscInt i    = offsetI+f; /* Element matrix row */
      for (g = 0; g < NbJ; ++g) {
        for (gc = 0; gc < NcJ; ++gc) {
          const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
          const PetscInt j    = offsetJ+g; /* Element matrix column */
          const PetscInt fOff = eOffset+i*totDim+j;

          elemMat[fOff] += tmpBasisI[fidx]*g0[fc*NcJ+gc]*tmpBasisJ[gidx];
          for (df = 0; df < dE; ++df) {
            elemMat[fOff] += tmpBasisI[fidx]*g1[(fc*NcJ+gc)*dE+df]*tmpBasisDerJ[gidx*dE+df];
            elemMat[fOff] += tmpBasisDerI[fidx*dE+df]*g2[(fc*NcJ+gc)*dE+df]*tmpBasisJ[gidx];
            for (dg = 0; dg < dE; ++dg) {
              elemMat[fOff] += tmpBasisDerI[fidx*dE+df]*g3[((fc*NcJ+gc)*dE+df)*dE+dg]*tmpBasisDerJ[gidx*dE+dg];
            }
          }
        }
      }
    }
  }
  return(0);
}

PetscErrorCode PetscFEUpdateElementMat_Hybrid_Internal(PetscFE feI, PetscBool isHybridI, PetscFE feJ, PetscBool isHybridJ, PetscInt r, PetscInt s, PetscInt q, PetscTabulation TI, PetscScalar tmpBasisI[], PetscScalar tmpBasisDerI[], PetscTabulation TJ, PetscScalar tmpBasisJ[], PetscScalar tmpBasisDerJ[], PetscFEGeom *fegeom, const PetscScalar g0[], const PetscScalar g1[], const PetscScalar g2[], const PetscScalar g3[], PetscInt eOffset, PetscInt totDim, PetscInt offsetI, PetscInt offsetJ, PetscScalar elemMat[])
{
  const PetscInt   dE        = TI->cdim;
  const PetscInt   NqI       = TI->Np;
  const PetscInt   NbI       = TI->Nb;
  const PetscInt   NcI       = TI->Nc;
  const PetscReal *basisI    = &TI->T[0][(r*NqI+q)*NbI*NcI];
  const PetscReal *basisDerI = &TI->T[1][(r*NqI+q)*NbI*NcI*dE];
  const PetscInt   NqJ       = TJ->Np;
  const PetscInt   NbJ       = TJ->Nb;
  const PetscInt   NcJ       = TJ->Nc;
  const PetscReal *basisJ    = &TJ->T[0][(r*NqJ+q)*NbJ*NcJ];
  const PetscReal *basisDerJ = &TJ->T[1][(r*NqJ+q)*NbJ*NcJ*dE];
  const PetscInt   so        = isHybridI ? 0 : s;
  const PetscInt   to        = isHybridJ ? 0 : s;
  PetscInt         f, fc, g, gc, df, dg;
  PetscErrorCode   ierr;

  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc; /* Test function basis index */

      tmpBasisI[fidx] = basisI[fidx];
      for (df = 0; df < dE; ++df) tmpBasisDerI[fidx*dE+df] = basisDerI[fidx*dE+df];
    }
  }
  ierr = PetscFEPushforward(feI, fegeom, NbI, tmpBasisI);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feI, fegeom, NbI, tmpBasisDerI);CHKERRQ(ierr);
  for (g = 0; g < NbJ; ++g) {
    for (gc = 0; gc < NcJ; ++gc) {
      const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */

      tmpBasisJ[gidx] = basisJ[gidx];
      for (dg = 0; dg < dE; ++dg) tmpBasisDerJ[gidx*dE+dg] = basisDerJ[gidx*dE+dg];
    }
  }
  ierr = PetscFEPushforward(feJ, fegeom, NbJ, tmpBasisJ);CHKERRQ(ierr);
  ierr = PetscFEPushforwardGradient(feJ, fegeom, NbJ, tmpBasisDerJ);CHKERRQ(ierr);
  for (f = 0; f < NbI; ++f) {
    for (fc = 0; fc < NcI; ++fc) {
      const PetscInt fidx = f*NcI+fc;         /* Test function basis index */
      const PetscInt i    = offsetI+NbI*so+f; /* Element matrix row */
      for (g = 0; g < NbJ; ++g) {
        for (gc = 0; gc < NcJ; ++gc) {
          const PetscInt gidx = g*NcJ+gc;         /* Trial function basis index */
          const PetscInt j    = offsetJ+NbJ*to+g; /* Element matrix column */
          const PetscInt fOff = eOffset+i*totDim+j;

          elemMat[fOff] += tmpBasisI[fidx]*g0[fc*NcJ+gc]*tmpBasisJ[gidx];
          for (df = 0; df < dE; ++df) {
            elemMat[fOff] += tmpBasisI[fidx]*g1[(fc*NcJ+gc)*dE+df]*tmpBasisDerJ[gidx*dE+df];
            elemMat[fOff] += tmpBasisDerI[fidx*dE+df]*g2[(fc*NcJ+gc)*dE+df]*tmpBasisJ[gidx];
            for (dg = 0; dg < dE; ++dg) {
              elemMat[fOff] += tmpBasisDerI[fidx*dE+df]*g3[((fc*NcJ+gc)*dE+df)*dE+dg]*tmpBasisDerJ[gidx*dE+dg];
            }
          }
        }
      }
    }
  }
  return(0);
}

PetscErrorCode PetscFECreateCellGeometry(PetscFE fe, PetscQuadrature quad, PetscFEGeom *cgeom)
{
  PetscDualSpace  dsp;
  DM              dm;
  PetscQuadrature quadDef;
  PetscInt        dim, cdim, Nq;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetDualSpace(fe, &dsp);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dsp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quadDef);CHKERRQ(ierr);
  quad = quad ? quad : quadDef;
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq*cdim,      &cgeom->v);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq*cdim*cdim, &cgeom->J);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq*cdim*cdim, &cgeom->invJ);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nq,           &cgeom->detJ);CHKERRQ(ierr);
  cgeom->dim       = dim;
  cgeom->dimEmbed  = cdim;
  cgeom->numCells  = 1;
  cgeom->numPoints = Nq;
  ierr = DMPlexComputeCellGeometryFEM(dm, 0, quad, cgeom->v, cgeom->J, cgeom->invJ, cgeom->detJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEDestroyCellGeometry(PetscFE fe, PetscFEGeom *cgeom)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cgeom->v);CHKERRQ(ierr);
  ierr = PetscFree(cgeom->J);CHKERRQ(ierr);
  ierr = PetscFree(cgeom->invJ);CHKERRQ(ierr);
  ierr = PetscFree(cgeom->detJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
