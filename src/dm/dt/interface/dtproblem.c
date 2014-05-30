#include <petsc-private/petscproblemimpl.h> /*I "petscproblem.h" I*/

PetscClassId PETSCPROBLEM_CLASSID = 0;

PetscFunctionList PetscProblemList              = NULL;
PetscBool         PetscProblemRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscProblemRegister"
/*@C
  PetscProblemRegister - Adds a new PetscProblem implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscProblemRegister() may be called multiple times to add several user-defined PetscProblems

  Sample usage:
.vb
    PetscProblemRegister("my_prob", MyPetscProblemCreate);
.ve

  Then, your PetscProblem type can be chosen with the procedural interface via
.vb
    PetscProblemCreate(MPI_Comm, PetscProblem *);
    PetscProblemSetType(PetscProblem, "my_prob");
.ve
   or at runtime via the option
.vb
    -petscproblem_type my_prob
.ve

  Level: advanced

.keywords: PetscProblem, register
.seealso: PetscProblemRegisterAll(), PetscProblemRegisterDestroy()

@*/
PetscErrorCode PetscProblemRegister(const char sname[], PetscErrorCode (*function)(PetscProblem))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscProblemList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetType"
/*@C
  PetscProblemSetType - Builds a particular PetscProblem

  Collective on PetscProblem

  Input Parameters:
+ prob - The PetscProblem object
- name - The kind of problem

  Options Database Key:
. -petscproblem_type <type> - Sets the PetscProblem type; use -help for a list of available types

  Level: intermediate

.keywords: PetscProblem, set, type
.seealso: PetscProblemGetType(), PetscProblemCreate()
@*/
PetscErrorCode PetscProblemSetType(PetscProblem prob, PetscProblemType name)
{
  PetscErrorCode (*r)(PetscProblem);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) prob, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscProblemRegisterAllCalled) {ierr = PetscProblemRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscProblemList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscProblem type: %s", name);

  if (prob->ops->destroy) {
    ierr             = (*prob->ops->destroy)(prob);CHKERRQ(ierr);
    prob->ops->destroy = NULL;
  }
  ierr = (*r)(prob);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) prob, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetType"
/*@C
  PetscProblemGetType - Gets the PetscProblem type name (as a string) from the object.

  Not Collective

  Input Parameter:
. prob  - The PetscProblem

  Output Parameter:
. name - The PetscProblem type name

  Level: intermediate

.keywords: PetscProblem, get, type, name
.seealso: PetscProblemSetType(), PetscProblemCreate()
@*/
PetscErrorCode PetscProblemGetType(PetscProblem prob, PetscProblemType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (!PetscProblemRegisterAllCalled) {ierr = PetscProblemRegisterAll();CHKERRQ(ierr);}
  *name = ((PetscObject) prob)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemView"
/*@C
  PetscProblemView - Views a PetscProblem

  Collective on PetscProblem

  Input Parameter:
+ prob - the PetscProblem object to view
- v  - the viewer

  Level: developer

.seealso PetscProblemDestroy()
@*/
PetscErrorCode PetscProblemView(PetscProblem prob, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) prob), &v);CHKERRQ(ierr);}
  if (prob->ops->view) {ierr = (*prob->ops->view)(prob, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemViewFromOptions"
/*
  PetscProblemViewFromOptions - Processes command line options to determine if/how a PetscProblem is to be viewed.

  Collective on PetscProblem

  Input Parameters:
+ prob   - the PetscProblem
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscProblem, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscProblemViewFromOptions(PetscProblem prob, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) prob), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  else        {ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) prob), ((PetscObject) prob)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);}
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscProblemView(prob, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetFromOptions"
/*@
  PetscProblemSetFromOptions - sets parameters in a PetscProblem from the options database

  Collective on PetscProblem

  Input Parameter:
. prob - the PetscProblem object to set options for

  Options Database:

  Level: developer

.seealso PetscProblemView()
@*/
PetscErrorCode PetscProblemSetFromOptions(PetscProblem prob)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (!((PetscObject) prob)->type_name) {
    defaultType = PETSCPROBLEMBASIC;
  } else {
    defaultType = ((PetscObject) prob)->type_name;
  }
  if (!PetscProblemRegisterAllCalled) {ierr = PetscProblemRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) prob);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscproblem_type", "Problem", "PetscProblemSetType", PetscProblemList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscProblemSetType(prob, name);CHKERRQ(ierr);
  } else if (!((PetscObject) prob)->type_name) {
    ierr = PetscProblemSetType(prob, defaultType);CHKERRQ(ierr);
  }
  if (prob->ops->setfromoptions) {ierr = (*prob->ops->setfromoptions)(prob);CHKERRQ(ierr);}
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) prob);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscProblemViewFromOptions(prob, NULL, "-petscproblem_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetUp"
/*@C
  PetscProblemSetUp - Construct data structures for the PetscProblem

  Collective on PetscProblem

  Input Parameter:
. prob - the PetscProblem object to setup

  Level: developer

.seealso PetscProblemView(), PetscProblemDestroy()
@*/
PetscErrorCode PetscProblemSetUp(PetscProblem prob)
{
  const PetscInt Nf = prob->Nf;
  PetscInt       dim, work, NcMax = 0, NqMax = 0, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (prob->setup) PetscFunctionReturn(0);
  /* Calculate sizes */
  ierr = PetscProblemGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  prob->totDim = prob->totDimBd = prob->totComp = 0;
  ierr = PetscMalloc4(Nf,&prob->basis,Nf,&prob->basisDer,Nf,&prob->basisBd,Nf,&prob->basisDerBd);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscFE         fe   = (PetscFE) prob->disc[f];
    PetscFE         feBd = (PetscFE) prob->discBd[f];
    PetscQuadrature q;
    PetscInt        Nq, Nb, Nc;

    /* TODO Dispatch on discretization type*/
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    ierr = PetscFEGetDefaultTabulation(fe, &prob->basis[f], &prob->basisDer[f], NULL);CHKERRQ(ierr);
    NqMax          = PetscMax(NqMax, Nq);
    NcMax          = PetscMax(NcMax, Nc);
    prob->totDim  += Nb*Nc;
    prob->totComp += Nc;
    if (feBd) {
      ierr = PetscFEGetDimension(feBd, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feBd, &Nc);CHKERRQ(ierr);
      ierr = PetscFEGetDefaultTabulation(feBd, &prob->basisBd[f], &prob->basisDerBd[f], NULL);CHKERRQ(ierr);
      prob->totDimBd += Nb*Nc;
    }
  }
  work = PetscMax(prob->totComp*dim, PetscSqr(NcMax*dim));
  /* Allocate works space */
  ierr = PetscMalloc5(prob->totComp,&prob->u,prob->totComp,&prob->u_t,prob->totComp*dim,&prob->u_x,dim,&prob->x,work,&prob->refSpaceDer);CHKERRQ(ierr);
  ierr = PetscMalloc6(NqMax*NcMax,&prob->f0,NqMax*NcMax*dim,&prob->f1,NqMax*NcMax*NcMax,&prob->g0,NqMax*NcMax*NcMax*dim,&prob->g1,NqMax*NcMax*NcMax*dim,&prob->g2,NqMax*NcMax*NcMax*dim*dim,&prob->g3);CHKERRQ(ierr);
  if (prob->ops->setup) {ierr = (*prob->ops->setup)(prob);CHKERRQ(ierr);}
  prob->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemDestroyStructs_Static"
static PetscErrorCode PetscProblemDestroyStructs_Static(PetscProblem prob)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree4(prob->basis,prob->basisDer,prob->basisBd,prob->basisDerBd);CHKERRQ(ierr);
  ierr = PetscFree5(prob->u,prob->u_t,prob->u_x,prob->x,prob->refSpaceDer);CHKERRQ(ierr);
  ierr = PetscFree6(prob->f0,prob->f1,prob->g0,prob->g1,prob->g2,prob->g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemEnlarge_Static"
static PetscErrorCode PetscProblemEnlarge_Static(PetscProblem prob, PetscInt *pNf, PetscObject **pdisc, PointFunc **pf, PointFunc **pg, PetscObject **pdiscBd, BdPointFunc **pfBd, BdPointFunc **pgBd, PetscInt NfNew)
{
  PetscObject   *tmpd, *tmpdbd;
  PointFunc     *tmpf, *tmpg;
  BdPointFunc   *tmpfbd, *tmpgbd;
  PetscInt       Nf = *pNf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  prob->setup = PETSC_FALSE;
  ierr = PetscProblemDestroyStructs_Static(prob);CHKERRQ(ierr);
  if (Nf >= NfNew) PetscFunctionReturn(0);
  ierr = PetscMalloc1(NfNew, &tmpd);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) tmpd[f] = (*pdisc)[f];
  for (f = Nf; f < NfNew; ++f) {ierr = PetscContainerCreate(PetscObjectComm((PetscObject) prob), (PetscContainer *) &tmpd[f]);CHKERRQ(ierr);}
  ierr = PetscFree(*pdisc);CHKERRQ(ierr);
  *pNf   = NfNew;
  *pdisc = tmpd;
  if (pf) {
    ierr = PetscCalloc2(NfNew*2, &tmpf, NfNew*NfNew*4, &tmpg);CHKERRQ(ierr);
    for (f = 0; f < Nf*2; ++f) tmpf[f] = (*pf)[f];
    for (f = 0; f < Nf*Nf*4; ++f) tmpg[f] = (*pg)[f];
    for (f = Nf*2; f < NfNew*2; ++f) tmpf[f] = NULL;
    for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpg[f] = NULL;
    ierr = PetscFree2(*pf, *pg);CHKERRQ(ierr);
    *pf = tmpf;
    *pg = tmpg;
  }
  if (pdiscBd) {
    ierr = PetscMalloc1(NfNew, &tmpdbd);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) tmpdbd[f] = (*pdiscBd)[f];
    for (f = Nf; f < NfNew; ++f) tmpdbd[f] = NULL;
    ierr = PetscFree(*pdiscBd);CHKERRQ(ierr);
    *pdiscBd = tmpdbd;
  }
  if (pfBd) {
    ierr = PetscCalloc2(NfNew*2, &tmpfbd, NfNew*NfNew*4, &tmpgbd);CHKERRQ(ierr);
    for (f = 0; f < Nf*2; ++f) tmpfbd[f] = (*pfBd)[f];
    for (f = 0; f < Nf*Nf*4; ++f) tmpgbd[f] = (*pgBd)[f];
    for (f = Nf*2; f < NfNew*2; ++f) tmpfbd[f] = NULL;
    for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpgbd[f] = NULL;
    ierr = PetscFree2(*pfBd, *pgBd);CHKERRQ(ierr);
    *pfBd = tmpfbd;
    *pgBd = tmpgbd;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemDestroy"
/*@
  PetscProblemDestroy - Destroys a PetscProblem object

  Collective on PetscProblem

  Input Parameter:
. prob - the PetscProblem object to destroy

  Level: developer

.seealso PetscProblemView()
@*/
PetscErrorCode PetscProblemDestroy(PetscProblem *prob)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*prob) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*prob), PETSCPROBLEM_CLASSID, 1);

  if (--((PetscObject)(*prob))->refct > 0) {*prob = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*prob))->refct = 0;
  ierr = PetscProblemDestroyStructs_Static(*prob);CHKERRQ(ierr);
  for (f = 0; f < (*prob)->Nf; ++f) {
    ierr = PetscObjectDereference((*prob)->disc[f]);CHKERRQ(ierr);
    ierr = PetscObjectDereference((*prob)->discBd[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*prob)->disc);CHKERRQ(ierr);
  ierr = PetscFree((*prob)->discBd);CHKERRQ(ierr);
  ierr = PetscFree2((*prob)->f,(*prob)->g);CHKERRQ(ierr);
  ierr = PetscFree2((*prob)->fBd,(*prob)->gBd);CHKERRQ(ierr);
  if ((*prob)->ops->destroy) {ierr = (*(*prob)->ops->destroy)(*prob);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemCreate"
/*@
  PetscProblemCreate - Creates an empty PetscProblem object. The type can then be set with PetscProblemSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscProblem object

  Output Parameter:
. prob - The PetscProblem object

  Level: beginner

.seealso: PetscProblemSetType(), PETSCPROBLEMBASIC
@*/
PetscErrorCode PetscProblemCreate(MPI_Comm comm, PetscProblem *prob)
{
  PetscProblem   p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(prob, 2);
  *prob  = NULL;
  ierr = PetscProblemInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, _p_PetscProblem, struct _PetscProblemOps, PETSCPROBLEM_CLASSID, "PetscProblem", "Problem", "PetscProblem", comm, PetscProblemDestroy, PetscProblemView);CHKERRQ(ierr);
  ierr = PetscMemzero(p->ops, sizeof(struct _PetscProblemOps));CHKERRQ(ierr);

  p->Nf    = 0;
  p->setup = PETSC_FALSE;

  *prob = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetNumFields"
PetscErrorCode PetscProblemGetNumFields(PetscProblem prob, PetscInt *Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(Nf, 2);
  *Nf = prob->Nf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetSpatialDimension"
PetscErrorCode PetscProblemGetSpatialDimension(PetscProblem prob, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = 0;
  if (prob->Nf) {ierr = PetscFEGetSpatialDimension((PetscFE) prob->disc[0], dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetTotalDimension"
PetscErrorCode PetscProblemGetTotalDimension(PetscProblem prob, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(dim, 2);
  *dim = prob->totDim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetTotalBdDimension"
PetscErrorCode PetscProblemGetTotalBdDimension(PetscProblem prob, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(dim, 2);
  *dim = prob->totDimBd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetTotalComponents"
PetscErrorCode PetscProblemGetTotalComponents(PetscProblem prob, PetscInt *Nc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(Nc, 2);
  *Nc = prob->totComp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetDiscretization"
PetscErrorCode PetscProblemGetDiscretization(PetscProblem prob, PetscInt f, PetscObject *disc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(disc, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *disc = prob->disc[f];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetBdDiscretization"
PetscErrorCode PetscProblemGetBdDiscretization(PetscProblem prob, PetscInt f, PetscObject *disc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(disc, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *disc = prob->discBd[f];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetDiscretization"
PetscErrorCode PetscProblemSetDiscretization(PetscProblem prob, PetscInt f, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(disc, 3);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, f+1);CHKERRQ(ierr);
  if (prob->disc[f]) {ierr = PetscObjectDereference(prob->disc[f]);CHKERRQ(ierr);}
  prob->disc[f] = disc;
  ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetBdDiscretization"
PetscErrorCode PetscProblemSetBdDiscretization(PetscProblem prob, PetscInt f, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (disc) PetscValidPointer(disc, 3);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, f+1);CHKERRQ(ierr);
  if (prob->discBd[f]) {ierr = PetscObjectDereference(prob->discBd[f]);CHKERRQ(ierr);}
  prob->discBd[f] = disc;
  ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemAddDiscretization"
PetscErrorCode PetscProblemAddDiscretization(PetscProblem prob, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscProblemSetDiscretization(prob, prob->Nf, disc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemAddBdDiscretization"
PetscErrorCode PetscProblemAddBdDiscretization(PetscProblem prob, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscProblemSetBdDiscretization(prob, prob->Nf, disc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetResidual"
PetscErrorCode PetscProblemGetResidual(PetscProblem prob, PetscInt f,
                                       void (**f0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[]),
                                       void (**f1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f1[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (f0) {PetscValidPointer(f0, 3); *f0 = prob->f[f*2+0];}
  if (f1) {PetscValidPointer(f1, 4); *f1 = prob->f[f*2+1];}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetResidual"
PetscErrorCode PetscProblemSetResidual(PetscProblem prob, PetscInt f,
                                       void (*f0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[]),
                                       void (*f1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f1[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer((const void *) f0, 3);
  PetscValidPointer((const void *) f1, 4);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, f+1);CHKERRQ(ierr);
  prob->f[f*2+0] = f0;
  prob->f[f*2+1] = f1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetJacobian"
PetscErrorCode PetscProblemGetJacobian(PetscProblem prob, PetscInt f, PetscInt g,
                                       void (**g0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g0[]),
                                       void (**g1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g1[]),
                                       void (**g2)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g2[]),
                                       void (**g3)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->g[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->g[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->g[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->g[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetJacobian"
PetscErrorCode PetscProblemSetJacobian(PetscProblem prob, PetscInt f, PetscInt g,
                                       void (*g0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g0[]),
                                       void (*g1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g1[]),
                                       void (*g2)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g2[]),
                                       void (*g3)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (g0) PetscValidPointer((const void *) g0, 4);
  if (g1) PetscValidPointer((const void *) g1, 5);
  if (g2) PetscValidPointer((const void *) g2, 6);
  if (g3) PetscValidPointer((const void *) g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->g[(f*prob->Nf + g)*4+0] = g0;
  prob->g[(f*prob->Nf + g)*4+1] = g1;
  prob->g[(f*prob->Nf + g)*4+2] = g2;
  prob->g[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetBdResidual"
PetscErrorCode PetscProblemGetBdResidual(PetscProblem prob, PetscInt f,
                                         void (**f0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]),
                                         void (**f1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (f0) {PetscValidPointer(f0, 3); *f0 = prob->fBd[f*2+0];}
  if (f1) {PetscValidPointer(f1, 4); *f1 = prob->fBd[f*2+1];}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetBdResidual"
PetscErrorCode PetscProblemSetBdResidual(PetscProblem prob, PetscInt f,
                                         void (*f0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]),
                                         void (*f1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, f+1);CHKERRQ(ierr);
  if (f0) {PetscValidPointer((const void *) f0, 3); prob->fBd[f*2+0] = f0;}
  if (f1) {PetscValidPointer((const void *) f1, 4); prob->fBd[f*2+1] = f1;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetBdJacobian"
PetscErrorCode PetscProblemGetBdJacobian(PetscProblem prob, PetscInt f, PetscInt g,
                                         void (**g0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g0[]),
                                         void (**g1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g1[]),
                                         void (**g2)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g2[]),
                                         void (**g3)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->gBd[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->gBd[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->gBd[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->gBd[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemSetBdJacobian"
PetscErrorCode PetscProblemSetBdJacobian(PetscProblem prob, PetscInt f, PetscInt g,
                                         void (*g0)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g0[]),
                                         void (*g1)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g1[]),
                                         void (*g2)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g2[]),
                                         void (*g3)(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], const PetscReal n[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  if (g0) PetscValidPointer((const void *) g0, 4);
  if (g1) PetscValidPointer((const void *) g1, 5);
  if (g2) PetscValidPointer((const void *) g2, 6);
  if (g3) PetscValidPointer((const void *) g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscProblemEnlarge_Static(prob, &prob->Nf, &prob->disc, &prob->f, &prob->g, &prob->discBd, &prob->fBd, &prob->gBd, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->gBd[(f*prob->Nf + g)*4+0] = g0;
  prob->gBd[(f*prob->Nf + g)*4+1] = g1;
  prob->gBd[(f*prob->Nf + g)*4+2] = g2;
  prob->gBd[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetFieldOffset"
PetscErrorCode PetscProblemGetFieldOffset(PetscProblem prob, PetscInt f, PetscInt *off)
{
  PetscInt       g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(off, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *off = 0;
  for (g = 0; g < f; ++g) {
    PetscFE  fe = (PetscFE) prob->disc[g];
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    *off += Nb*Nc;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetBdFieldOffset"
PetscErrorCode PetscProblemGetBdFieldOffset(PetscProblem prob, PetscInt f, PetscInt *off)
{
  PetscInt       g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  PetscValidPointer(off, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *off = 0;
  for (g = 0; g < f; ++g) {
    PetscFE  fe = (PetscFE) prob->discBd[g];
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    *off += Nb*Nc;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetTabulation"
PetscErrorCode PetscProblemGetTabulation(PetscProblem prob, PetscReal ***basis, PetscReal ***basisDer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  if (basis)    {PetscValidPointer(basis, 2);    *basis    = prob->basis;}
  if (basisDer) {PetscValidPointer(basisDer, 3); *basisDer = prob->basisDer;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetBdTabulation"
PetscErrorCode PetscProblemGetBdTabulation(PetscProblem prob, PetscReal ***basis, PetscReal ***basisDer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  if (basis)    {PetscValidPointer(basis, 2);    *basis    = prob->basisBd;}
  if (basisDer) {PetscValidPointer(basisDer, 3); *basisDer = prob->basisDerBd;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetEvaluationArrays"
PetscErrorCode PetscProblemGetEvaluationArrays(PetscProblem prob, PetscScalar **u, PetscScalar **u_t, PetscScalar **u_x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  if (u)   {PetscValidPointer(u, 2);   *u   = prob->u;}
  if (u_t) {PetscValidPointer(u_t, 3); *u_t = prob->u_t;}
  if (u_x) {PetscValidPointer(u_x, 4); *u_x = prob->u_x;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetWeakFormArrays"
PetscErrorCode PetscProblemGetWeakFormArrays(PetscProblem prob, PetscScalar **f0, PetscScalar **f1, PetscScalar **g0, PetscScalar **g1, PetscScalar **g2, PetscScalar **g3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  if (f0) {PetscValidPointer(f0, 2); *f0 = prob->f0;}
  if (f1) {PetscValidPointer(f1, 3); *f1 = prob->f1;}
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->g0;}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->g1;}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->g2;}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->g3;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemGetRefCoordArrays"
PetscErrorCode PetscProblemGetRefCoordArrays(PetscProblem prob, PetscReal **x, PetscScalar **refSpaceDer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCPROBLEM_CLASSID, 1);
  ierr = PetscProblemSetUp(prob);CHKERRQ(ierr);
  if (x)           {PetscValidPointer(x, 2);           *x           = prob->x;}
  if (refSpaceDer) {PetscValidPointer(refSpaceDer, 3); *refSpaceDer = prob->refSpaceDer;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemDestroy_Basic"
PetscErrorCode PetscProblemDestroy_Basic(PetscProblem prob)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscProblemInitialize_Basic"
PetscErrorCode PetscProblemInitialize_Basic(PetscProblem prob)
{
  PetscFunctionBegin;
  prob->ops->setfromoptions = NULL;
  prob->ops->setup          = NULL;
  prob->ops->view           = NULL;
  prob->ops->destroy        = PetscProblemDestroy_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPROBLEMBASIC = "basic" - The only kind of problem I can think of

  Level: intermediate

.seealso: PetscProblemType, PetscProblemCreate(), PetscProblemSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscProblemCreate_Basic"
PETSC_EXTERN PetscErrorCode PetscProblemCreate_Basic(PetscProblem prob)
{
  PetscProblem_Basic *b;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCSPACE_CLASSID, 1);
  ierr       = PetscNewLog(prob, &b);CHKERRQ(ierr);
  prob->data = b;

  ierr = PetscProblemInitialize_Basic(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
