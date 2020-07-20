#include <petsc/private/petscdsimpl.h> /*I "petscds.h" I*/

PetscClassId PETSCDS_CLASSID = 0;

PetscFunctionList PetscDSList              = NULL;
PetscBool         PetscDSRegisterAllCalled = PETSC_FALSE;

/* A PetscDS (Discrete System) encodes a set of equations posed in a discrete space, which represents a set of
   nonlinear continuum equations. The equations can have multiple fields, each field having a different
   discretization. In addition, different pieces of the domain can have different field combinations and equations.

   The DS provides the user a description of the approximation space on any given cell. It also gives pointwise
   functions representing the equations.

   Each field is associated with a label, marking the cells on which it is supported. Note that a field can be
   supported on the closure of a cell not in the label due to overlap of the boundary of neighboring cells. The DM
   then creates a DS for each set of cells with identical approximation spaces. When assembling, the user asks for
   the space associated with a given cell. DMPlex uses the labels associated with each DS in the default integration loop.
*/

/*@C
  PetscDSRegister - Adds a new PetscDS implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscDSRegister() may be called multiple times to add several user-defined PetscDSs

  Sample usage:
.vb
    PetscDSRegister("my_ds", MyPetscDSCreate);
.ve

  Then, your PetscDS type can be chosen with the procedural interface via
.vb
    PetscDSCreate(MPI_Comm, PetscDS *);
    PetscDSSetType(PetscDS, "my_ds");
.ve
   or at runtime via the option
.vb
    -petscds_type my_ds
.ve

  Level: advanced

   Not available from Fortran

.seealso: PetscDSRegisterAll(), PetscDSRegisterDestroy()

@*/
PetscErrorCode PetscDSRegister(const char sname[], PetscErrorCode (*function)(PetscDS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscDSList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetType - Builds a particular PetscDS

  Collective on prob

  Input Parameters:
+ prob - The PetscDS object
- name - The kind of system

  Options Database Key:
. -petscds_type <type> - Sets the PetscDS type; use -help for a list of available types

  Level: intermediate

   Not available from Fortran

.seealso: PetscDSGetType(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetType(PetscDS prob, PetscDSType name)
{
  PetscErrorCode (*r)(PetscDS);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) prob, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscDSRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscDSList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDS type: %s", name);

  if (prob->ops->destroy) {
    ierr             = (*prob->ops->destroy)(prob);CHKERRQ(ierr);
    prob->ops->destroy = NULL;
  }
  ierr = (*r)(prob);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) prob, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetType - Gets the PetscDS type name (as a string) from the object.

  Not Collective

  Input Parameter:
. prob  - The PetscDS

  Output Parameter:
. name - The PetscDS type name

  Level: intermediate

   Not available from Fortran

.seealso: PetscDSSetType(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetType(PetscDS prob, PetscDSType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(name, 2);
  ierr = PetscDSRegisterAll();CHKERRQ(ierr);
  *name = ((PetscObject) prob)->type_name;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSView_Ascii(PetscDS prob, PetscViewer viewer)
{
  PetscViewerFormat  format;
  const PetscScalar *constants;
  PetscInt           numConstants, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Discrete System with %d fields\n", prob->Nf);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "  cell total dim %D total comp %D\n", prob->totDim, prob->totComp);CHKERRQ(ierr);
  if (prob->isHybrid) {ierr = PetscViewerASCIIPrintf(viewer, "  hybrid cell\n");CHKERRQ(ierr);}
  for (f = 0; f < prob->Nf; ++f) {
    DSBoundary      b;
    PetscObject     obj;
    PetscClassId    id;
    PetscQuadrature q;
    const char     *name;
    PetscInt        Nc, Nq, Nqc;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    ierr = PetscObjectGetName(obj, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Field %s", name ? name : "<unknown>");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {
      ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature((PetscFE) obj, &q);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, " FEM");CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);
      ierr = PetscFVGetQuadrature((PetscFV) obj, &q);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, " FVM");CHKERRQ(ierr);
    }
    else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    if (Nc > 1) {ierr = PetscViewerASCIIPrintf(viewer, "%D components", Nc);CHKERRQ(ierr);}
    else        {ierr = PetscViewerASCIIPrintf(viewer, "%D component ", Nc);CHKERRQ(ierr);}
    if (prob->implicit[f]) {ierr = PetscViewerASCIIPrintf(viewer, " (implicit)");CHKERRQ(ierr);}
    else                   {ierr = PetscViewerASCIIPrintf(viewer, " (explicit)");CHKERRQ(ierr);}
    if (q) {
      ierr = PetscQuadratureGetData(q, NULL, &Nqc, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, " (Nq %D Nqc %D)", Nq, Nqc);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {ierr = PetscFEView((PetscFE) obj, viewer);CHKERRQ(ierr);}
    else if (id == PETSCFV_CLASSID) {ierr = PetscFVView((PetscFV) obj, viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    for (b = prob->boundary; b; b = b->next) {
      PetscInt c, i;

      if (b->field != f) continue;
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Boundary %s (%s) %s\n", b->name, b->labelname, DMBoundaryConditionTypes[b->type]);CHKERRQ(ierr);
      if (!b->numcomps) {
        ierr = PetscViewerASCIIPrintf(viewer, "  all components\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "  components: ");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
        for (c = 0; c < b->numcomps; ++c) {
          if (c > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPrintf(viewer, "%D", b->comps[c]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "  ids: ");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      for (i = 0; i < b->numids; ++i) {
        if (i > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "%D", b->ids[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (numConstants) {
    ierr = PetscViewerASCIIPrintf(viewer, "%D constants\n", numConstants);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (f = 0; f < numConstants; ++f) {ierr = PetscViewerASCIIPrintf(viewer, "%g\n", (double) PetscRealPart(constants[f]));CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDSViewFromOptions - View from Options

   Collective on PetscDS

   Input Parameters:
+  A - the PetscDS object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscDS, PetscDSView, PetscObjectViewFromOptions(), PetscDSCreate()
@*/
PetscErrorCode  PetscDSViewFromOptions(PetscDS A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCDS_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSView - Views a PetscDS

  Collective on prob

  Input Parameter:
+ prob - the PetscDS object to view
- v  - the viewer

  Level: developer

.seealso PetscDSDestroy()
@*/
PetscErrorCode PetscDSView(PetscDS prob, PetscViewer v)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) prob), &v);CHKERRQ(ierr);}
  else    {PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);}
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscDSView_Ascii(prob, v);CHKERRQ(ierr);}
  if (prob->ops->view) {ierr = (*prob->ops->view)(prob, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetFromOptions - sets parameters in a PetscDS from the options database

  Collective on prob

  Input Parameter:
. prob - the PetscDS object to set options for

  Options Database:
+ -petscds_type <type>     : Set the DS type
. -petscds_view <view opt> : View the DS
. -petscds_jac_pre         : Turn formation of a separate Jacobian preconditioner on and off
. -bc_<name> <ids>         : Specify a list of label ids for a boundary condition
- -bc_<name>_comp <comps>  : Specify a list of field components to constrain for a boundary condition

  Level: developer

.seealso PetscDSView()
@*/
PetscErrorCode PetscDSSetFromOptions(PetscDS prob)
{
  DSBoundary     b;
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (!((PetscObject) prob)->type_name) {
    defaultType = PETSCDSBASIC;
  } else {
    defaultType = ((PetscObject) prob)->type_name;
  }
  ierr = PetscDSRegisterAll();CHKERRQ(ierr);

  ierr = PetscObjectOptionsBegin((PetscObject) prob);CHKERRQ(ierr);
  for (b = prob->boundary; b; b = b->next) {
    char       optname[1024];
    PetscInt   ids[1024], len = 1024;
    PetscBool  flg;

    ierr = PetscSNPrintf(optname, sizeof(optname), "-bc_%s", b->name);CHKERRQ(ierr);
    ierr = PetscMemzero(ids, sizeof(ids));CHKERRQ(ierr);
    ierr = PetscOptionsIntArray(optname, "List of boundary IDs", "", ids, &len, &flg);CHKERRQ(ierr);
    if (flg) {
      b->numids = len;
      ierr = PetscFree(b->ids);CHKERRQ(ierr);
      ierr = PetscMalloc1(len, &b->ids);CHKERRQ(ierr);
      ierr = PetscArraycpy(b->ids, ids, len);CHKERRQ(ierr);
    }
    len = 1024;
    ierr = PetscSNPrintf(optname, sizeof(optname), "-bc_%s_comp", b->name);CHKERRQ(ierr);
    ierr = PetscMemzero(ids, sizeof(ids));CHKERRQ(ierr);
    ierr = PetscOptionsIntArray(optname, "List of boundary field components", "", ids, &len, &flg);CHKERRQ(ierr);
    if (flg) {
      b->numcomps = len;
      ierr = PetscFree(b->comps);CHKERRQ(ierr);
      ierr = PetscMalloc1(len, &b->comps);CHKERRQ(ierr);
      ierr = PetscArraycpy(b->comps, ids, len);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsFList("-petscds_type", "Discrete System", "PetscDSSetType", PetscDSList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscDSSetType(prob, name);CHKERRQ(ierr);
  } else if (!((PetscObject) prob)->type_name) {
    ierr = PetscDSSetType(prob, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-petscds_jac_pre", "Discrete System", "PetscDSUseJacobianPreconditioner", prob->useJacPre, &prob->useJacPre, &flg);CHKERRQ(ierr);
  if (prob->ops->setfromoptions) {ierr = (*prob->ops->setfromoptions)(prob);CHKERRQ(ierr);}
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) prob);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (prob->Nf) {ierr = PetscDSViewFromOptions(prob, NULL, "-petscds_view");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetUp - Construct data structures for the PetscDS

  Collective on prob

  Input Parameter:
. prob - the PetscDS object to setup

  Level: developer

.seealso PetscDSView(), PetscDSDestroy()
@*/
PetscErrorCode PetscDSSetUp(PetscDS prob)
{
  const PetscInt Nf = prob->Nf;
  PetscInt       dim, dimEmbed, NbMax = 0, NcMax = 0, NqMax = 0, NsMax = 1, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (prob->setup) PetscFunctionReturn(0);
  /* Calculate sizes */
  ierr = PetscDSGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  ierr = PetscDSGetCoordinateDimension(prob, &dimEmbed);CHKERRQ(ierr);
  prob->totDim = prob->totComp = 0;
  ierr = PetscMalloc2(Nf,&prob->Nc,Nf,&prob->Nb);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf+1,&prob->off,Nf+1,&prob->offDer);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf,&prob->T,Nf,&prob->Tf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject     obj;
    PetscClassId    id;
    PetscQuadrature q = NULL;
    PetscInt        Nq = 0, Nb, Nc;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    if (!obj) {
      /* Empty mesh */
      Nb = Nc = 0;
      prob->T[f] = prob->Tf[f] = NULL;
    } else {
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {
        PetscFE fe = (PetscFE) obj;

        ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
        ierr = PetscFEGetCellTabulation(fe, &prob->T[f]);CHKERRQ(ierr);
        ierr = PetscFEGetFaceTabulation(fe, &prob->Tf[f]);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        ierr = PetscFVGetQuadrature(fv, &q);CHKERRQ(ierr);
        ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
        Nb   = Nc;
        ierr = PetscFVGetCellTabulation(fv, &prob->T[f]);CHKERRQ(ierr);
        /* TODO: should PetscFV also have face tabulation? Otherwise there will be a null pointer in prob->basisFace */
      } else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
    }
    prob->Nc[f]       = Nc;
    prob->Nb[f]       = Nb;
    prob->off[f+1]    = Nc     + prob->off[f];
    prob->offDer[f+1] = Nc*dim + prob->offDer[f];
    if (q) {ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);}
    NqMax          = PetscMax(NqMax, Nq);
    NbMax          = PetscMax(NbMax, Nb);
    NcMax          = PetscMax(NcMax, Nc);
    prob->totDim  += Nb;
    prob->totComp += Nc;
    /* There are two faces for all fields but the cohesive field on a hybrid cell */
    if (prob->isHybrid && (f < Nf-1)) prob->totDim += Nb;
  }
  /* Allocate works space */
  if (prob->isHybrid) NsMax = 2;
  ierr = PetscMalloc3(NsMax*prob->totComp,&prob->u,NsMax*prob->totComp,&prob->u_t,NsMax*prob->totComp*dimEmbed,&prob->u_x);CHKERRQ(ierr);
  ierr = PetscMalloc5(dimEmbed,&prob->x,NbMax*NcMax,&prob->basisReal,NbMax*NcMax*dimEmbed,&prob->basisDerReal,NbMax*NcMax,&prob->testReal,NbMax*NcMax*dimEmbed,&prob->testDerReal);CHKERRQ(ierr);
  ierr = PetscMalloc6(NsMax*NqMax*NcMax,&prob->f0,NsMax*NqMax*NcMax*dimEmbed,&prob->f1,
                      NsMax*NsMax*NqMax*NcMax*NcMax,&prob->g0,NsMax*NsMax*NqMax*NcMax*NcMax*dimEmbed,&prob->g1,
                      NsMax*NsMax*NqMax*NcMax*NcMax*dimEmbed,&prob->g2,NsMax*NsMax*NqMax*NcMax*NcMax*dimEmbed*dimEmbed,&prob->g3);CHKERRQ(ierr);
  if (prob->ops->setup) {ierr = (*prob->ops->setup)(prob);CHKERRQ(ierr);}
  prob->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSDestroyStructs_Static(PetscDS prob)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(prob->Nc,prob->Nb);CHKERRQ(ierr);
  ierr = PetscFree2(prob->off,prob->offDer);CHKERRQ(ierr);
  ierr = PetscFree2(prob->T,prob->Tf);CHKERRQ(ierr);
  ierr = PetscFree3(prob->u,prob->u_t,prob->u_x);CHKERRQ(ierr);
  ierr = PetscFree5(prob->x,prob->basisReal, prob->basisDerReal,prob->testReal,prob->testDerReal);CHKERRQ(ierr);
  ierr = PetscFree6(prob->f0,prob->f1,prob->g0,prob->g1,prob->g2,prob->g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSEnlarge_Static(PetscDS prob, PetscInt NfNew)
{
  PetscObject      *tmpd;
  PetscBool        *tmpi;
  PetscPointFunc   *tmpobj, *tmpf, *tmpup;
  PetscPointJac    *tmpg, *tmpgp, *tmpgt;
  PetscBdPointFunc *tmpfbd;
  PetscBdPointJac  *tmpgbd, *tmpgpbd;
  PetscRiemannFunc *tmpr;
  PetscSimplePointFunc *tmpexactSol,  *tmpexactSol_t;
  void                **tmpexactCtx, **tmpexactCtx_t;
  void            **tmpctx;
  PetscInt          Nf = prob->Nf, f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (Nf >= NfNew) PetscFunctionReturn(0);
  prob->setup = PETSC_FALSE;
  ierr = PetscDSDestroyStructs_Static(prob);CHKERRQ(ierr);
  ierr = PetscMalloc2(NfNew, &tmpd, NfNew, &tmpi);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {tmpd[f] = prob->disc[f]; tmpi[f] = prob->implicit[f];}
  for (f = Nf; f < NfNew; ++f) {tmpd[f] = NULL; tmpi[f] = PETSC_TRUE;}
  ierr = PetscFree2(prob->disc, prob->implicit);CHKERRQ(ierr);
  prob->Nf        = NfNew;
  prob->disc      = tmpd;
  prob->implicit  = tmpi;
  ierr = PetscCalloc7(NfNew, &tmpobj, NfNew*2, &tmpf, NfNew*NfNew*4, &tmpg, NfNew*NfNew*4, &tmpgp, NfNew*NfNew*4, &tmpgt, NfNew, &tmpr, NfNew, &tmpctx);CHKERRQ(ierr);
  ierr = PetscCalloc1(NfNew, &tmpup);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) tmpobj[f] = prob->obj[f];
  for (f = 0; f < Nf*2; ++f) tmpf[f] = prob->f[f];
  for (f = 0; f < Nf*Nf*4; ++f) tmpg[f] = prob->g[f];
  for (f = 0; f < Nf*Nf*4; ++f) tmpgp[f] = prob->gp[f];
  for (f = 0; f < Nf; ++f) tmpr[f] = prob->r[f];
  for (f = 0; f < Nf; ++f) tmpup[f] = prob->update[f];
  for (f = 0; f < Nf; ++f) tmpctx[f] = prob->ctx[f];
  for (f = Nf; f < NfNew; ++f) tmpobj[f] = NULL;
  for (f = Nf*2; f < NfNew*2; ++f) tmpf[f] = NULL;
  for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpg[f] = NULL;
  for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpgp[f] = NULL;
  for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpgt[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpr[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpup[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpctx[f] = NULL;
  ierr = PetscFree7(prob->obj, prob->f, prob->g, prob->gp, prob->gt, prob->r, prob->ctx);CHKERRQ(ierr);
  ierr = PetscFree(prob->update);CHKERRQ(ierr);
  prob->obj = tmpobj;
  prob->f   = tmpf;
  prob->g   = tmpg;
  prob->gp  = tmpgp;
  prob->gt  = tmpgt;
  prob->r   = tmpr;
  prob->update = tmpup;
  prob->ctx = tmpctx;
  ierr = PetscCalloc7(NfNew*2, &tmpfbd, NfNew*NfNew*4, &tmpgbd, NfNew*NfNew*4, &tmpgpbd, NfNew, &tmpexactSol, NfNew, &tmpexactCtx, NfNew, &tmpexactSol_t, NfNew, &tmpexactCtx_t);CHKERRQ(ierr);
  for (f = 0; f < Nf*2; ++f) tmpfbd[f] = prob->fBd[f];
  for (f = 0; f < Nf*Nf*4; ++f) tmpgbd[f] = prob->gBd[f];
  for (f = 0; f < Nf*Nf*4; ++f) tmpgpbd[f] = prob->gpBd[f];
  for (f = 0; f < Nf; ++f) tmpexactSol[f] = prob->exactSol[f];
  for (f = 0; f < Nf; ++f) tmpexactCtx[f] = prob->exactCtx[f];
  for (f = 0; f < Nf; ++f) tmpexactSol_t[f] = prob->exactSol_t[f];
  for (f = 0; f < Nf; ++f) tmpexactCtx_t[f] = prob->exactCtx_t[f];
  for (f = Nf*2; f < NfNew*2; ++f) tmpfbd[f] = NULL;
  for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpgbd[f] = NULL;
  for (f = Nf*Nf*4; f < NfNew*NfNew*4; ++f) tmpgpbd[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactSol[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactCtx[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactSol_t[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactCtx_t[f] = NULL;
  ierr = PetscFree7(prob->fBd, prob->gBd, prob->gpBd, prob->exactSol, prob->exactCtx, prob->exactSol_t, prob->exactCtx_t);CHKERRQ(ierr);
  prob->fBd = tmpfbd;
  prob->gBd = tmpgbd;
  prob->gpBd = tmpgpbd;
  prob->exactSol = tmpexactSol;
  prob->exactCtx = tmpexactCtx;
  prob->exactSol_t = tmpexactSol_t;
  prob->exactCtx_t = tmpexactCtx_t;
  PetscFunctionReturn(0);
}

/*@
  PetscDSDestroy - Destroys a PetscDS object

  Collective on prob

  Input Parameter:
. prob - the PetscDS object to destroy

  Level: developer

.seealso PetscDSView()
@*/
PetscErrorCode PetscDSDestroy(PetscDS *prob)
{
  PetscInt       f;
  DSBoundary     next;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*prob) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*prob), PETSCDS_CLASSID, 1);

  if (--((PetscObject)(*prob))->refct > 0) {*prob = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*prob))->refct = 0;
  if ((*prob)->subprobs) {
    PetscInt dim, d;

    ierr = PetscDSGetSpatialDimension(*prob, &dim);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {ierr = PetscDSDestroy(&(*prob)->subprobs[d]);CHKERRQ(ierr);}
  }
  ierr = PetscFree((*prob)->subprobs);CHKERRQ(ierr);
  ierr = PetscDSDestroyStructs_Static(*prob);CHKERRQ(ierr);
  for (f = 0; f < (*prob)->Nf; ++f) {
    ierr = PetscObjectDereference((*prob)->disc[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree2((*prob)->disc, (*prob)->implicit);CHKERRQ(ierr);
  ierr = PetscFree7((*prob)->obj,(*prob)->f,(*prob)->g,(*prob)->gp,(*prob)->gt,(*prob)->r,(*prob)->ctx);CHKERRQ(ierr);
  ierr = PetscFree((*prob)->update);CHKERRQ(ierr);
  ierr = PetscFree7((*prob)->fBd,(*prob)->gBd,(*prob)->gpBd,(*prob)->exactSol,(*prob)->exactCtx,(*prob)->exactSol_t,(*prob)->exactCtx_t);CHKERRQ(ierr);
  if ((*prob)->ops->destroy) {ierr = (*(*prob)->ops->destroy)(*prob);CHKERRQ(ierr);}
  next = (*prob)->boundary;
  while (next) {
    DSBoundary b = next;

    next = b->next;
    ierr = PetscFree(b->comps);CHKERRQ(ierr);
    ierr = PetscFree(b->ids);CHKERRQ(ierr);
    ierr = PetscFree(b->name);CHKERRQ(ierr);
    ierr = PetscFree(b->labelname);CHKERRQ(ierr);
    ierr = PetscFree(b);CHKERRQ(ierr);
  }
  ierr = PetscFree((*prob)->constants);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDSCreate - Creates an empty PetscDS object. The type can then be set with PetscDSSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscDS object

  Output Parameter:
. prob - The PetscDS object

  Level: beginner

.seealso: PetscDSSetType(), PETSCDSBASIC
@*/
PetscErrorCode PetscDSCreate(MPI_Comm comm, PetscDS *prob)
{
  PetscDS   p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(prob, 2);
  *prob  = NULL;
  ierr = PetscDSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, PETSCDS_CLASSID, "PetscDS", "Discrete System", "PetscDS", comm, PetscDSDestroy, PetscDSView);CHKERRQ(ierr);

  p->Nf           = 0;
  p->setup        = PETSC_FALSE;
  p->numConstants = 0;
  p->constants    = NULL;
  p->dimEmbed     = -1;
  p->useJacPre    = PETSC_TRUE;

  *prob = p;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetNumFields - Returns the number of fields in the DS

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. Nf - The number of fields

  Level: beginner

.seealso: PetscDSGetSpatialDimension(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetNumFields(PetscDS prob, PetscInt *Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(Nf, 2);
  *Nf = prob->Nf;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetSpatialDimension - Returns the spatial dimension of the DS, meaning the topological dimension of the discretizations

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. dim - The spatial dimension

  Level: beginner

.seealso: PetscDSGetCoordinateDimension(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetSpatialDimension(PetscDS prob, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = 0;
  if (prob->Nf) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, 0, &obj);CHKERRQ(ierr);
    if (obj) {
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetSpatialDimension((PetscFE) obj, dim);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetSpatialDimension((PetscFV) obj, dim);CHKERRQ(ierr);}
      else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", 0);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetCoordinateDimension - Returns the coordinate dimension of the DS, meaning the dimension of the space into which the discretiaztions are embedded

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. dimEmbed - The coordinate dimension

  Level: beginner

.seealso: PetscDSSetCoordinateDimension(), PetscDSGetSpatialDimension(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetCoordinateDimension(PetscDS prob, PetscInt *dimEmbed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(dimEmbed, 2);
  if (prob->dimEmbed < 0) SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONGSTATE, "No coordinate dimension set for this DS");
  *dimEmbed = prob->dimEmbed;
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetCoordinateDimension - Set the coordinate dimension of the DS, meaning the dimension of the space into which the discretiaztions are embedded

  Logically collective on prob

  Input Parameters:
+ prob - The PetscDS object
- dimEmbed - The coordinate dimension

  Level: beginner

.seealso: PetscDSGetCoordinateDimension(), PetscDSGetSpatialDimension(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetCoordinateDimension(PetscDS prob, PetscInt dimEmbed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (dimEmbed < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate dimension must be non-negative, not %D", dimEmbed);
  prob->dimEmbed = dimEmbed;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetHybrid - Returns the flag for a hybrid (cohesive) cell

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. isHybrid - The flag

  Level: developer

.seealso: PetscDSSetHybrid(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetHybrid(PetscDS prob, PetscBool *isHybrid)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(isHybrid, 2);
  *isHybrid = prob->isHybrid;
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetHybrid - Set the flag for a hybrid (cohesive) cell

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- isHybrid - The flag

  Level: developer

.seealso: PetscDSGetHybrid(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetHybrid(PetscDS prob, PetscBool isHybrid)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  prob->isHybrid = isHybrid;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetTotalDimension - Returns the total size of the approximation space for this system

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. dim - The total problem dimension

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetTotalDimension(PetscDS prob, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(dim, 2);
  *dim = prob->totDim;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetTotalComponents - Returns the total number of components in this system

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. dim - The total number of components

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetTotalComponents(PetscDS prob, PetscInt *Nc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(Nc, 2);
  *Nc = prob->totComp;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetDiscretization - Returns the discretization object for the given field

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. disc - The discretization object

  Level: beginner

.seealso: PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetDiscretization(PetscDS prob, PetscInt f, PetscObject *disc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(disc, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *disc = prob->disc[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetDiscretization - Sets the discretization object for the given field

  Not collective

  Input Parameters:
+ prob - The PetscDS object
. f - The field number
- disc - The discretization object

  Level: beginner

.seealso: PetscDSGetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetDiscretization(PetscDS prob, PetscInt f, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (disc) PetscValidPointer(disc, 3);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  ierr = PetscObjectDereference(prob->disc[f]);CHKERRQ(ierr);
  prob->disc[f] = disc;
  ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  if (disc) {
    PetscClassId id;

    ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      ierr = PetscDSSetImplicit(prob, f, PETSC_TRUE);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      ierr = PetscDSSetImplicit(prob, f, PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSAddDiscretization - Adds a discretization object

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- disc - The boundary discretization object

  Level: beginner

.seealso: PetscDSGetDiscretization(), PetscDSSetDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSAddDiscretization(PetscDS prob, PetscObject disc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDSSetDiscretization(prob, prob->Nf, disc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetQuadrature - Returns the quadrature, which must agree for all fields in the DS

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. q - The quadrature object

Level: intermediate

.seealso: PetscDSSetImplicit(), PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetQuadrature(PetscDS prob, PetscQuadrature *q)
{
  PetscObject    obj;
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *q = NULL;
  if (!prob->Nf) PetscFunctionReturn(0);
  ierr = PetscDSGetDiscretization(prob, 0, &obj);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
  if      (id == PETSCFE_CLASSID) {ierr = PetscFEGetQuadrature((PetscFE) obj, q);CHKERRQ(ierr);}
  else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetQuadrature((PetscFV) obj, q);CHKERRQ(ierr);}
  else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", 0);
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetImplicit - Returns the flag for implicit solve for this field. This is just a guide for IMEX

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. implicit - The flag indicating what kind of solve to use for this field

  Level: developer

.seealso: PetscDSSetImplicit(), PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetImplicit(PetscDS prob, PetscInt f, PetscBool *implicit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(implicit, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *implicit = prob->implicit[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetImplicit - Set the flag for implicit solve for this field. This is just a guide for IMEX

  Not collective

  Input Parameters:
+ prob - The PetscDS object
. f - The field number
- implicit - The flag indicating what kind of solve to use for this field

  Level: developer

.seealso: PetscDSGetImplicit(), PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetImplicit(PetscDS prob, PetscInt f, PetscBool implicit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  prob->implicit[f] = implicit;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetObjective(PetscDS prob, PetscInt f,
                                   void (**obj)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(obj, 2);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *obj = prob->obj[f];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSSetObjective(PetscDS prob, PetscInt f,
                                   void (*obj)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (obj) PetscValidFunction(obj, 2);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  prob->obj[f] = obj;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetResidual - Get the pointwise residual function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The test field number

  Output Parameters:
+ f0 - integrand for the test function term
- f1 - integrand for the test function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi f_0(u, u_t, \nabla u, x, t) + \nabla\phi \cdot {\vec f}_1(u, u_t, \nabla u, x, t)

The calling sequence for the callbacks f0 and f1 is given by:

$ f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], PetscScalar f0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- f0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetResidual()
@*/
PetscErrorCode PetscDSGetResidual(PetscDS prob, PetscInt f,
                                  void (**f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                  void (**f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (f0) {PetscValidPointer(f0, 3); *f0 = prob->f[f*2+0];}
  if (f1) {PetscValidPointer(f1, 4); *f1 = prob->f[f*2+1];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetResidual - Set the pointwise residual function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. f0 - integrand for the test function term
- f1 - integrand for the test function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi f_0(u, u_t, \nabla u, x, t) + \nabla\phi \cdot {\vec f}_1(u, u_t, \nabla u, x, t)

The calling sequence for the callbacks f0 and f1 is given by:

$ f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], PetscScalar f0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- f0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetResidual()
@*/
PetscErrorCode PetscDSSetResidual(PetscDS prob, PetscInt f,
                                  void (*f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                  void (*f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (f0) PetscValidFunction(f0, 3);
  if (f1) PetscValidFunction(f1, 4);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  prob->f[f*2+0] = f0;
  prob->f[f*2+1] = f1;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSHasJacobian - Signals that Jacobian functions have been set

  Not collective

  Input Parameter:
. prob - The PetscDS

  Output Parameter:
. hasJac - flag that pointwise function for the Jacobian has been set

  Level: intermediate

.seealso: PetscDSGetJacobianPreconditioner(), PetscDSSetJacobianPreconditioner(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSHasJacobian(PetscDS prob, PetscBool *hasJac)
{
  PetscInt f, g, h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  *hasJac = PETSC_FALSE;
  for (f = 0; f < prob->Nf; ++f) {
    for (g = 0; g < prob->Nf; ++g) {
      for (h = 0; h < 4; ++h) {
        if (prob->g[(f*prob->Nf + g)*4+h]) *hasJac = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetJacobian - Get the pointwise Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
- g    - The field number

  Output Parameters:
+ g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal u_tShift, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetJacobian()
@*/
PetscErrorCode PetscDSGetJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                  void (**g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                  void (**g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                  void (**g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                  void (**g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->g[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->g[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->g[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->g[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetJacobian - Set the pointwise Jacobian function for given test and basis fields

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. g    - The field number
. g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSSetJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                  void (*g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                  void (*g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                  void (*g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                  void (*g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscDSEnlarge_Static(prob, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->g[(f*prob->Nf + g)*4+0] = g0;
  prob->g[(f*prob->Nf + g)*4+1] = g1;
  prob->g[(f*prob->Nf + g)*4+2] = g2;
  prob->g[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSUseJacobianPreconditioner - Whether to construct a Jacobian preconditioner

  Not collective

  Input Parameters:
+ prob - The PetscDS
- useJacPre - flag that enables construction of a Jacobian preconditioner

  Level: intermediate

.seealso: PetscDSGetJacobianPreconditioner(), PetscDSSetJacobianPreconditioner(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSUseJacobianPreconditioner(PetscDS prob, PetscBool useJacPre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  prob->useJacPre = useJacPre;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSHasJacobianPreconditioner - Signals that a Jacobian preconditioner matrix has been set

  Not collective

  Input Parameter:
. prob - The PetscDS

  Output Parameter:
. hasJacPre - flag that pointwise function for Jacobian preconditioner matrix has been set

  Level: intermediate

.seealso: PetscDSGetJacobianPreconditioner(), PetscDSSetJacobianPreconditioner(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSHasJacobianPreconditioner(PetscDS prob, PetscBool *hasJacPre)
{
  PetscInt f, g, h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  *hasJacPre = PETSC_FALSE;
  if (!prob->useJacPre) PetscFunctionReturn(0);
  for (f = 0; f < prob->Nf; ++f) {
    for (g = 0; g < prob->Nf; ++g) {
      for (h = 0; h < 4; ++h) {
        if (prob->gp[(f*prob->Nf + g)*4+h]) *hasJacPre = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetJacobianPreconditioner - Get the pointwise Jacobian preconditioner function for given test and basis field. If this is missing, the system matrix is used to build the preconditioner.

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
- g    - The field number

  Output Parameters:
+ g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal u_tShift, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetJacobianPreconditioner(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSGetJacobianPreconditioner(PetscDS prob, PetscInt f, PetscInt g,
                                  void (**g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                  void (**g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                  void (**g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                  void (**g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->gp[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->gp[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->gp[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->gp[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetJacobianPreconditioner - Set the pointwise Jacobian preconditioner function for given test and basis fields. If this is missing, the system matrix is used to build the preconditioner.

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. g    - The field number
. g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetJacobianPreconditioner(), PetscDSSetJacobian()
@*/
PetscErrorCode PetscDSSetJacobianPreconditioner(PetscDS prob, PetscInt f, PetscInt g,
                                  void (*g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                  void (*g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                  void (*g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                  void (*g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscDSEnlarge_Static(prob, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->gp[(f*prob->Nf + g)*4+0] = g0;
  prob->gp[(f*prob->Nf + g)*4+1] = g1;
  prob->gp[(f*prob->Nf + g)*4+2] = g2;
  prob->gp[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSHasDynamicJacobian - Signals that a dynamic Jacobian, dF/du_t, has been set

  Not collective

  Input Parameter:
. prob - The PetscDS

  Output Parameter:
. hasDynJac - flag that pointwise function for dynamic Jacobian has been set

  Level: intermediate

.seealso: PetscDSGetDynamicJacobian(), PetscDSSetDynamicJacobian(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSHasDynamicJacobian(PetscDS prob, PetscBool *hasDynJac)
{
  PetscInt f, g, h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  *hasDynJac = PETSC_FALSE;
  for (f = 0; f < prob->Nf; ++f) {
    for (g = 0; g < prob->Nf; ++g) {
      for (h = 0; h < 4; ++h) {
        if (prob->gt[(f*prob->Nf + g)*4+h]) *hasDynJac = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetDynamicJacobian - Get the pointwise dynamic Jacobian, dF/du_t, function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
- g    - The field number

  Output Parameters:
+ g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal u_tShift, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetJacobian()
@*/
PetscErrorCode PetscDSGetDynamicJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                         void (**g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                         void (**g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                         void (**g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                         void (**g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->gt[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->gt[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->gt[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->gt[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetDynamicJacobian - Set the pointwise dynamic Jacobian, dF/du_t, function for given test and basis fields

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. g    - The field number
. g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSSetDynamicJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                         void (*g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                         void (*g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                         void (*g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                         void (*g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscDSEnlarge_Static(prob, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->gt[(f*prob->Nf + g)*4+0] = g0;
  prob->gt[(f*prob->Nf + g)*4+1] = g1;
  prob->gt[(f*prob->Nf + g)*4+2] = g2;
  prob->gt[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetRiemannSolver - Returns the Riemann solver for the given field

  Not collective

  Input Arguments:
+ prob - The PetscDS object
- f    - The field number

  Output Argument:
. r    - Riemann solver

  Calling sequence for r:

$ r(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx)

+ dim  - The spatial dimension
. Nf   - The number of fields
. x    - The coordinates at a point on the interface
. n    - The normal vector to the interface
. uL   - The state vector to the left of the interface
. uR   - The state vector to the right of the interface
. flux - output array of flux through the interface
. numConstants - number of constant parameters
. constants - constant parameters
- ctx  - optional user context

  Level: intermediate

.seealso: PetscDSSetRiemannSolver()
@*/
PetscErrorCode PetscDSGetRiemannSolver(PetscDS prob, PetscInt f,
                                       void (**r)(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  PetscValidPointer(r, 3);
  *r = prob->r[f];
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetRiemannSolver - Sets the Riemann solver for the given field

  Not collective

  Input Arguments:
+ prob - The PetscDS object
. f    - The field number
- r    - Riemann solver

  Calling sequence for r:

$ r(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx)

+ dim  - The spatial dimension
. Nf   - The number of fields
. x    - The coordinates at a point on the interface
. n    - The normal vector to the interface
. uL   - The state vector to the left of the interface
. uR   - The state vector to the right of the interface
. flux - output array of flux through the interface
. numConstants - number of constant parameters
. constants - constant parameters
- ctx  - optional user context

  Level: intermediate

.seealso: PetscDSGetRiemannSolver()
@*/
PetscErrorCode PetscDSSetRiemannSolver(PetscDS prob, PetscInt f,
                                       void (*r)(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (r) PetscValidFunction(r, 3);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  prob->r[f] = r;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetUpdate - Get the pointwise update function for a given field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The field number

  Output Parameters:
. update - update function

  Note: The calling sequence for the callback update is given by:

$ update(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$        PetscReal t, const PetscReal x[], PetscScalar uNew[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
- uNew - new value for field at the current point

  Level: intermediate

.seealso: PetscDSSetUpdate(), PetscDSSetResidual()
@*/
PetscErrorCode PetscDSGetUpdate(PetscDS prob, PetscInt f,
                                  void (**update)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uNew[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (update) {PetscValidPointer(update, 3); *update = prob->update[f];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetUpdate - Set the pointwise update function for a given field

  Not collective

  Input Parameters:
+ prob   - The PetscDS
. f      - The field number
- update - update function

  Note: The calling sequence for the callback update is given by:

$ update(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$        PetscReal t, const PetscReal x[], PetscScalar uNew[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
- uNew - new field values at the current point

  Level: intermediate

.seealso: PetscDSGetResidual()
@*/
PetscErrorCode PetscDSSetUpdate(PetscDS prob, PetscInt f,
                                void (*update)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uNew[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (update) PetscValidFunction(update, 3);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  prob->update[f] = update;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetContext(PetscDS prob, PetscInt f, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  PetscValidPointer(ctx, 3);
  *ctx = prob->ctx[f];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSSetContext(PetscDS prob, PetscInt f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  prob->ctx[f] = ctx;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdResidual - Get the pointwise boundary residual function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The test field number

  Output Parameters:
+ f0 - boundary integrand for the test function term
- f1 - boundary integrand for the test function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec f}_0(u, u_t, \nabla u, x, t) \cdot \hat n + \nabla\phi \cdot {\overleftrightarrow f}_1(u, u_t, \nabla u, x, t) \cdot \hat n

The calling sequence for the callbacks f0 and f1 is given by:

$ f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. n - unit normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- f0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetBdResidual()
@*/
PetscErrorCode PetscDSGetBdResidual(PetscDS prob, PetscInt f,
                                    void (**f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                    void (**f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (f0) {PetscValidPointer(f0, 3); *f0 = prob->fBd[f*2+0];}
  if (f1) {PetscValidPointer(f1, 4); *f1 = prob->fBd[f*2+1];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdResidual - Get the pointwise boundary residual function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. f0 - boundary integrand for the test function term
- f1 - boundary integrand for the test function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec f}_0(u, u_t, \nabla u, x, t) \cdot \hat n + \nabla\phi \cdot {\overleftrightarrow f}_1(u, u_t, \nabla u, x, t) \cdot \hat n

The calling sequence for the callbacks f0 and f1 is given by:

$ f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. n - unit normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- f0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetBdResidual()
@*/
PetscErrorCode PetscDSSetBdResidual(PetscDS prob, PetscInt f,
                                    void (*f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                    void (*f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  if (f0) {PetscValidFunction(f0, 3); prob->fBd[f*2+0] = f0;}
  if (f1) {PetscValidFunction(f1, 4); prob->fBd[f*2+1] = f1;}
  PetscFunctionReturn(0);
}

/*@
  PetscDSHasBdJacobian - Signals that boundary Jacobian functions have been set

  Not collective

  Input Parameter:
. prob - The PetscDS

  Output Parameter:
. hasBdJac - flag that pointwise function for the boundary Jacobian has been set

  Level: intermediate

.seealso: PetscDSHasJacobian(), PetscDSSetBdJacobian(), PetscDSGetBdJacobian()
@*/
PetscErrorCode PetscDSHasBdJacobian(PetscDS prob, PetscBool *hasBdJac)
{
  PetscInt f, g, h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  *hasBdJac = PETSC_FALSE;
  for (f = 0; f < prob->Nf; ++f) {
    for (g = 0; g < prob->Nf; ++g) {
      for (h = 0; h < 4; ++h) {
        if (prob->gBd[(f*prob->Nf + g)*4+h]) *hasBdJac = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdJacobian - Get the pointwise boundary Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
- g    - The field number

  Output Parameters:
+ g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec g}_0(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \cdot \hat n \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \hat n \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. n - normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSSetBdJacobian()
@*/
PetscErrorCode PetscDSGetBdJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                    void (**g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                    void (**g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                    void (**g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                    void (**g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->gBd[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->gBd[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->gBd[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->gBd[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdJacobian - Set the pointwise boundary Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. g    - The field number
. g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec g}_0(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \cdot \hat n \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \hat n \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. n - normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  Level: intermediate

.seealso: PetscDSGetBdJacobian()
@*/
PetscErrorCode PetscDSSetBdJacobian(PetscDS prob, PetscInt f, PetscInt g,
                                    void (*g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                    void (*g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                    void (*g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                    void (*g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscDSEnlarge_Static(prob, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->gBd[(f*prob->Nf + g)*4+0] = g0;
  prob->gBd[(f*prob->Nf + g)*4+1] = g1;
  prob->gBd[(f*prob->Nf + g)*4+2] = g2;
  prob->gBd[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

/*@
  PetscDSHasBdJacobianPreconditioner - Signals that boundary Jacobian preconditioner functions have been set

  Not collective

  Input Parameter:
. prob - The PetscDS

  Output Parameter:
. hasBdJac - flag that pointwise function for the boundary Jacobian preconditioner has been set

  Level: intermediate

.seealso: PetscDSHasJacobian(), PetscDSSetBdJacobian(), PetscDSGetBdJacobian()
@*/
PetscErrorCode PetscDSHasBdJacobianPreconditioner(PetscDS prob, PetscBool *hasBdJacPre)
{
  PetscInt f, g, h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  *hasBdJacPre = PETSC_FALSE;
  for (f = 0; f < prob->Nf; ++f) {
    for (g = 0; g < prob->Nf; ++g) {
      for (h = 0; h < 4; ++h) {
        if (prob->gpBd[(f*prob->Nf + g)*4+h]) *hasBdJacPre = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdJacobianPreconditioner - Get the pointwise boundary Jacobian preconditioner function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
- g    - The field number

  Output Parameters:
+ g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec g}_0(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \cdot \hat n \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \hat n \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. NfAux - the number of auxiliary fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. n - normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  This is not yet available in Fortran.

  Level: intermediate

.seealso: PetscDSSetBdJacobianPreconditioner()
@*/
PetscErrorCode PetscDSGetBdJacobianPreconditioner(PetscDS prob, PetscInt f, PetscInt g,
                                                  void (**g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                                  void (**g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                                  void (**g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                                  void (**g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if ((g < 0) || (g >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, prob->Nf);
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->gpBd[(f*prob->Nf + g)*4+0];}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->gpBd[(f*prob->Nf + g)*4+1];}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->gpBd[(f*prob->Nf + g)*4+2];}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->gpBd[(f*prob->Nf + g)*4+3];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdJacobianPreconditioner - Set the pointwise boundary Jacobian preconditioner function for given test and basis field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. g    - The field number
. g0 - integrand for the test and basis function term
. g1 - integrand for the test function and basis function gradient term
. g2 - integrand for the test function gradient and basis function term
- g3 - integrand for the test function gradient and basis function gradient term

  Note: We are using a first order FEM model for the weak form:

  \int_\Gamma \phi {\vec g}_0(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \cdot \hat n \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \hat n \cdot \nabla \psi

The calling sequence for the callbacks g0, g1, g2 and g3 is given by:

$ g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])

+ dim - the spatial dimension
. Nf - the number of fields
. NfAux - the number of auxiliary fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. u_tShift - the multiplier a for dF/dU_t
. x - coordinates of the current point
. n - normal at the current point
. numConstants - number of constant parameters
. constants - constant parameters
- g0 - output values at the current point

  This is not yet available in Fortran.

  Level: intermediate

.seealso: PetscDSGetBdJacobianPreconditioner()
@*/
PetscErrorCode PetscDSSetBdJacobianPreconditioner(PetscDS prob, PetscInt f, PetscInt g,
                                                  void (*g0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]),
                                                  void (*g1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]),
                                                  void (*g2)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]),
                                                  void (*g3)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  if (g < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscDSEnlarge_Static(prob, PetscMax(f, g)+1);CHKERRQ(ierr);
  prob->gpBd[(f*prob->Nf + g)*4+0] = g0;
  prob->gpBd[(f*prob->Nf + g)*4+1] = g1;
  prob->gpBd[(f*prob->Nf + g)*4+2] = g2;
  prob->gpBd[(f*prob->Nf + g)*4+3] = g3;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetExactSolution - Get the pointwise exact solution function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The test field number

  Output Parameter:
+ exactSol - exact solution for the test field
- exactCtx - exact solution context

  Note: The calling sequence for the solution functions is given by:

$ sol(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)

+ dim - the spatial dimension
. t - current time
. x - coordinates of the current point
. Nc - the number of field components
. u - the solution field evaluated at the current point
- ctx - a user context

  Level: intermediate

.seealso: PetscDSSetExactSolution(), PetscDSGetExactSolutionTimeDerivative()
@*/
PetscErrorCode PetscDSGetExactSolution(PetscDS prob, PetscInt f, PetscErrorCode (**sol)(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (sol) {PetscValidPointer(sol, 3); *sol = prob->exactSol[f];}
  if (ctx) {PetscValidPointer(ctx, 4); *ctx = prob->exactCtx[f];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetExactSolution - Set the pointwise exact solution function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. sol  - solution function for the test fields
- ctx  - solution context or NULL

  Note: The calling sequence for solution functions is given by:

$ sol(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)

+ dim - the spatial dimension
. t - current time
. x - coordinates of the current point
. Nc - the number of field components
. u - the solution field evaluated at the current point
- ctx - a user context

  Level: intermediate

.seealso: PetscDSGetExactSolution()
@*/
PetscErrorCode PetscDSSetExactSolution(PetscDS prob, PetscInt f, PetscErrorCode (*sol)(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  if (sol) {PetscValidFunction(sol, 3); prob->exactSol[f] = sol;}
  if (ctx) {PetscValidFunction(ctx, 4); prob->exactCtx[f] = ctx;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetExactSolutionTimeDerivative - Get the pointwise time derivative of the exact solution function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The test field number

  Output Parameter:
+ exactSol - time derivative of the exact solution for the test field
- exactCtx - time derivative of the exact solution context

  Note: The calling sequence for the solution functions is given by:

$ sol(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)

+ dim - the spatial dimension
. t - current time
. x - coordinates of the current point
. Nc - the number of field components
. u - the solution field evaluated at the current point
- ctx - a user context

  Level: intermediate

.seealso: PetscDSSetExactSolutionTimeDerivative(), PetscDSGetExactSolution()
@*/
PetscErrorCode PetscDSGetExactSolutionTimeDerivative(PetscDS prob, PetscInt f, PetscErrorCode (**sol)(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  if (sol) {PetscValidPointer(sol, 3); *sol = prob->exactSol_t[f];}
  if (ctx) {PetscValidPointer(ctx, 4); *ctx = prob->exactCtx_t[f];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetExactSolutionTimeDerivative - Set the pointwise time derivative of the exact solution function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
. f    - The test field number
. sol  - time derivative of the solution function for the test fields
- ctx  - time derivative of the solution context or NULL

  Note: The calling sequence for solution functions is given by:

$ sol(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)

+ dim - the spatial dimension
. t - current time
. x - coordinates of the current point
. Nc - the number of field components
. u - the solution field evaluated at the current point
- ctx - a user context

  Level: intermediate

.seealso: PetscDSGetExactSolutionTimeDerivative(), PetscDSSetExactSolution()
@*/
PetscErrorCode PetscDSSetExactSolutionTimeDerivative(PetscDS prob, PetscInt f, PetscErrorCode (*sol)(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(prob, f+1);CHKERRQ(ierr);
  if (sol) {PetscValidFunction(sol, 3); prob->exactSol_t[f] = sol;}
  if (ctx) {PetscValidFunction(ctx, 4); prob->exactCtx_t[f] = ctx;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetConstants - Returns the array of constants passed to point functions

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameters:
+ numConstants - The number of constants
- constants    - The array of constants, NULL if there are none

  Level: intermediate

.seealso: PetscDSSetConstants(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetConstants(PetscDS prob, PetscInt *numConstants, const PetscScalar *constants[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (numConstants) {PetscValidPointer(numConstants, 2); *numConstants = prob->numConstants;}
  if (constants)    {PetscValidPointer(constants, 3);    *constants    = prob->constants;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetConstants - Set the array of constants passed to point functions

  Not collective

  Input Parameters:
+ prob         - The PetscDS object
. numConstants - The number of constants
- constants    - The array of constants, NULL if there are none

  Level: intermediate

.seealso: PetscDSGetConstants(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetConstants(PetscDS prob, PetscInt numConstants, PetscScalar constants[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (numConstants != prob->numConstants) {
    ierr = PetscFree(prob->constants);CHKERRQ(ierr);
    prob->numConstants = numConstants;
    if (prob->numConstants) {
      ierr = PetscMalloc1(prob->numConstants, &prob->constants);CHKERRQ(ierr);
    } else {
      prob->constants = NULL;
    }
  }
  if (prob->numConstants) {
    PetscValidPointer(constants, 3);
    ierr = PetscArraycpy(prob->constants, constants, prob->numConstants);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetFieldIndex - Returns the index of the given field

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- disc - The discretization object

  Output Parameter:
. f - The field number

  Level: beginner

.seealso: PetscGetDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetFieldIndex(PetscDS prob, PetscObject disc, PetscInt *f)
{
  PetscInt g;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(f, 3);
  *f = -1;
  for (g = 0; g < prob->Nf; ++g) {if (disc == prob->disc[g]) break;}
  if (g == prob->Nf) SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Field not found in PetscDS.");
  *f = g;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetFieldSize - Returns the size of the given field in the full space basis

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. size - The size

  Level: beginner

.seealso: PetscDSGetFieldOffset(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetFieldSize(PetscDS prob, PetscInt f, PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(size, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  *size = prob->Nb[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetFieldOffset - Returns the offset of the given field in the full space basis

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. off - The offset

  Level: beginner

.seealso: PetscDSGetFieldSize(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetFieldOffset(PetscDS prob, PetscInt f, PetscInt *off)
{
  PetscInt       size, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(off, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *off = 0;
  for (g = 0; g < f; ++g) {
    ierr = PetscDSGetFieldSize(prob, g, &size);CHKERRQ(ierr);
    *off += size;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetDimensions - Returns the size of the approximation space for each field on an evaluation point

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. dimensions - The number of dimensions

  Level: beginner

.seealso: PetscDSGetComponentOffsets(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetDimensions(PetscDS prob, PetscInt *dimensions[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(dimensions, 2);
  *dimensions = prob->Nb;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponents - Returns the number of components for each field on an evaluation point

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. components - The number of components

  Level: beginner

.seealso: PetscDSGetComponentOffsets(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponents(PetscDS prob, PetscInt *components[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  PetscValidPointer(components, 2);
  *components = prob->Nc;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponentOffset - Returns the offset of the given field on an evaluation point

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. off - The offset

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponentOffset(PetscDS prob, PetscInt f, PetscInt *off)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(off, 3);
  if ((f < 0) || (f >= prob->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *off = prob->off[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponentOffsets - Returns the offset of each field on an evaluation point

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. offsets - The offsets

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponentOffsets(PetscDS prob, PetscInt *offsets[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 2);
  *offsets = prob->off;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponentDerivativeOffsets - Returns the offset of each field derivative on an evaluation point

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. offsets - The offsets

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponentDerivativeOffsets(PetscDS prob, PetscInt *offsets[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 2);
  *offsets = prob->offDer;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetTabulation - Return the basis tabulation at quadrature points for the volume discretization

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. T - The basis function and derivatives tabulation at quadrature points for each field

  Level: intermediate

.seealso: PetscDSCreate()
@*/
PetscErrorCode PetscDSGetTabulation(PetscDS prob, PetscTabulation *T[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(T, 2);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  *T = prob->T;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetFaceTabulation - Return the basis tabulation at quadrature points on the faces

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. Tf - The basis function and derviative tabulation on each local face at quadrature points for each and field

  Level: intermediate

.seealso: PetscDSGetTabulation(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetFaceTabulation(PetscDS prob, PetscTabulation *Tf[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(Tf, 2);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  *Tf = prob->Tf;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetEvaluationArrays(PetscDS prob, PetscScalar **u, PetscScalar **u_t, PetscScalar **u_x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  if (u)   {PetscValidPointer(u, 2);   *u   = prob->u;}
  if (u_t) {PetscValidPointer(u_t, 3); *u_t = prob->u_t;}
  if (u_x) {PetscValidPointer(u_x, 4); *u_x = prob->u_x;}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetWeakFormArrays(PetscDS prob, PetscScalar **f0, PetscScalar **f1, PetscScalar **g0, PetscScalar **g1, PetscScalar **g2, PetscScalar **g3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  if (f0) {PetscValidPointer(f0, 2); *f0 = prob->f0;}
  if (f1) {PetscValidPointer(f1, 3); *f1 = prob->f1;}
  if (g0) {PetscValidPointer(g0, 4); *g0 = prob->g0;}
  if (g1) {PetscValidPointer(g1, 5); *g1 = prob->g1;}
  if (g2) {PetscValidPointer(g2, 6); *g2 = prob->g2;}
  if (g3) {PetscValidPointer(g3, 7); *g3 = prob->g3;}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetWorkspace(PetscDS prob, PetscReal **x, PetscScalar **basisReal, PetscScalar **basisDerReal, PetscScalar **testReal, PetscScalar **testDerReal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  if (x)            {PetscValidPointer(x, 2);            *x            = prob->x;}
  if (basisReal)    {PetscValidPointer(basisReal, 3);    *basisReal    = prob->basisReal;}
  if (basisDerReal) {PetscValidPointer(basisDerReal, 4); *basisDerReal = prob->basisDerReal;}
  if (testReal)     {PetscValidPointer(testReal, 5);     *testReal     = prob->testReal;}
  if (testDerReal)  {PetscValidPointer(testDerReal, 6);  *testDerReal  = prob->testDerReal;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSAddBoundary - Add a boundary condition to the model

  Collective on ds

  Input Parameters:
+ ds          - The PetscDS object
. type        - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components (0 will constrain all fields)
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: PetscDSGetBoundary()
@*/
PetscErrorCode PetscDSAddBoundary(PetscDS ds, DMBoundaryConditionType type, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(void), PetscInt numids, const PetscInt *ids, void *ctx)
{
  DSBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ds, type, 2);
  PetscValidLogicalCollectiveInt(ds, field, 5);
  PetscValidLogicalCollectiveInt(ds, numcomps, 6);
  PetscValidLogicalCollectiveInt(ds, numids, 9);
  ierr = PetscNew(&b);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(labelname, (char **) &b->labelname);CHKERRQ(ierr);
  ierr = PetscMalloc1(numcomps, &b->comps);CHKERRQ(ierr);
  if (numcomps) {ierr = PetscArraycpy(b->comps, comps, numcomps);CHKERRQ(ierr);}
  ierr = PetscMalloc1(numids, &b->ids);CHKERRQ(ierr);
  if (numids) {ierr = PetscArraycpy(b->ids, ids, numids);CHKERRQ(ierr);}
  b->type            = type;
  b->field           = field;
  b->numcomps        = numcomps;
  b->func            = bcFunc;
  b->numids          = numids;
  b->ctx             = ctx;
  b->next            = ds->boundary;
  ds->boundary       = b;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSUpdateBoundary - Change a boundary condition for the model

  Input Parameters:
+ ds          - The PetscDS object
. bd          - The boundary condition number
. type        - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Note: The boundary condition number is the order in which it was registered. The user can get the number of boundary conditions from PetscDSGetNumBoundary().

  Level: developer

.seealso: PetscDSAddBoundary(), PetscDSGetBoundary(), PetscDSGetNumBoundary()
@*/
PetscErrorCode PetscDSUpdateBoundary(PetscDS ds, PetscInt bd, DMBoundaryConditionType type, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(void), PetscInt numids, const PetscInt *ids, void *ctx)
{
  DSBoundary     b = ds->boundary;
  PetscInt       n = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  while (b) {
    if (n == bd) break;
    b = b->next;
    ++n;
  }
  if (!b) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (name) {
    ierr = PetscFree(b->name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  }
  if (labelname) {
    ierr = PetscFree(b->labelname);CHKERRQ(ierr);
    ierr = PetscStrallocpy(labelname, (char **) &b->labelname);CHKERRQ(ierr);
  }
  if (numcomps >= 0 && numcomps != b->numcomps) {
    b->numcomps = numcomps;
    ierr = PetscFree(b->comps);CHKERRQ(ierr);
    ierr = PetscMalloc1(numcomps, &b->comps);CHKERRQ(ierr);
    if (numcomps) {ierr = PetscArraycpy(b->comps, comps, numcomps);CHKERRQ(ierr);}
  }
  if (numids >= 0 && numids != b->numids) {
    b->numids = numids;
    ierr = PetscFree(b->ids);CHKERRQ(ierr);
    ierr = PetscMalloc1(numids, &b->ids);CHKERRQ(ierr);
    if (numids) {ierr = PetscArraycpy(b->ids, ids, numids);CHKERRQ(ierr);}
  }
  b->type = type;
  if (field >= 0) {b->field  = field;}
  if (bcFunc)     {b->func   = bcFunc;}
  if (ctx)        {b->ctx    = ctx;}
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetNumBoundary - Get the number of registered BC

  Input Parameters:
. ds - The PetscDS object

  Output Parameters:
. numBd - The number of BC

  Level: intermediate

.seealso: PetscDSAddBoundary(), PetscDSGetBoundary()
@*/
PetscErrorCode PetscDSGetNumBoundary(PetscDS ds, PetscInt *numBd)
{
  DSBoundary b = ds->boundary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(numBd, 2);
  *numBd = 0;
  while (b) {++(*numBd); b = b->next;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBoundary - Gets a boundary condition to the model

  Input Parameters:
+ ds          - The PetscDS object
- bd          - The BC number

  Output Parameters:
+ type        - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: PetscDSAddBoundary()
@*/
PetscErrorCode PetscDSGetBoundary(PetscDS ds, PetscInt bd, DMBoundaryConditionType *type, const char **name, const char **labelname, PetscInt *field, PetscInt *numcomps, const PetscInt **comps, void (**func)(void), PetscInt *numids, const PetscInt **ids, void **ctx)
{
  DSBoundary b    = ds->boundary;
  PetscInt   n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  while (b) {
    if (n == bd) break;
    b = b->next;
    ++n;
  }
  if (!b) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (type) {
    PetscValidPointer(type, 3);
    *type = b->type;
  }
  if (name) {
    PetscValidPointer(name, 4);
    *name = b->name;
  }
  if (labelname) {
    PetscValidPointer(labelname, 5);
    *labelname = b->labelname;
  }
  if (field) {
    PetscValidPointer(field, 6);
    *field = b->field;
  }
  if (numcomps) {
    PetscValidPointer(numcomps, 7);
    *numcomps = b->numcomps;
  }
  if (comps) {
    PetscValidPointer(comps, 8);
    *comps = b->comps;
  }
  if (func) {
    PetscValidPointer(func, 9);
    *func = b->func;
  }
  if (numids) {
    PetscValidPointer(numids, 10);
    *numids = b->numids;
  }
  if (ids) {
    PetscValidPointer(ids, 11);
    *ids = b->ids;
  }
  if (ctx) {
    PetscValidPointer(ctx, 12);
    *ctx = b->ctx;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSCopyBoundary - Copy all boundary condition objects to the new problem

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSCopyEquations(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSCopyBoundary(PetscDS probA, PetscDS probB)
{
  DSBoundary     b, next, *lastnext;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(probA, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(probB, PETSCDS_CLASSID, 2);
  if (probA == probB) PetscFunctionReturn(0);
  next = probB->boundary;
  while (next) {
    DSBoundary b = next;

    next = b->next;
    ierr = PetscFree(b->comps);CHKERRQ(ierr);
    ierr = PetscFree(b->ids);CHKERRQ(ierr);
    ierr = PetscFree(b->name);CHKERRQ(ierr);
    ierr = PetscFree(b->labelname);CHKERRQ(ierr);
    ierr = PetscFree(b);CHKERRQ(ierr);
  }
  lastnext = &(probB->boundary);
  for (b = probA->boundary; b; b = b->next) {
    DSBoundary bNew;

    ierr = PetscNew(&bNew);CHKERRQ(ierr);
    bNew->numcomps = b->numcomps;
    ierr = PetscMalloc1(bNew->numcomps, &bNew->comps);CHKERRQ(ierr);
    ierr = PetscArraycpy(bNew->comps, b->comps, bNew->numcomps);CHKERRQ(ierr);
    bNew->numids = b->numids;
    ierr = PetscMalloc1(bNew->numids, &bNew->ids);CHKERRQ(ierr);
    ierr = PetscArraycpy(bNew->ids, b->ids, bNew->numids);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->labelname,(char **) &(bNew->labelname));CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->name,(char **) &(bNew->name));CHKERRQ(ierr);
    bNew->ctx   = b->ctx;
    bNew->type  = b->type;
    bNew->field = b->field;
    bNew->func  = b->func;

    *lastnext = bNew;
    lastnext = &(bNew->next);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSelectEquations - Copy pointwise function pointers to the new problem with different field layout

  Not collective

  Input Parameter:
+ prob - The PetscDS object
. numFields - Number of new fields
- fields - Old field number for each new field

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSCopyBoundary(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSelectEquations(PetscDS prob, PetscInt numFields, const PetscInt fields[], PetscDS newprob)
{
  PetscInt       Nf, Nfn, fn, gn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (fields) PetscValidPointer(fields, 3);
  PetscValidHeaderSpecific(newprob, PETSCDS_CLASSID, 4);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(newprob, &Nfn);CHKERRQ(ierr);
  if (numFields > Nfn) SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_SIZ, "Number of fields %D to transfer must not be greater then the total number of fields %D", numFields, Nfn);
  for (fn = 0; fn < numFields; ++fn) {
    const PetscInt   f = fields ? fields[fn] : fn;
    PetscPointFunc   obj;
    PetscPointFunc   f0, f1;
    PetscBdPointFunc f0Bd, f1Bd;
    PetscRiemannFunc r;

    if (f >= Nf) continue;
    ierr = PetscDSGetObjective(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscDSGetResidual(prob, f, &f0, &f1);CHKERRQ(ierr);
    ierr = PetscDSGetBdResidual(prob, f, &f0Bd, &f1Bd);CHKERRQ(ierr);
    ierr = PetscDSGetRiemannSolver(prob, f, &r);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(newprob, fn, obj);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(newprob, fn, f0, f1);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(newprob, fn, f0Bd, f1Bd);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(newprob, fn, r);CHKERRQ(ierr);
    for (gn = 0; gn < numFields; ++gn) {
      const PetscInt  g = fields ? fields[gn] : gn;
      PetscPointJac   g0, g1, g2, g3;
      PetscPointJac   g0p, g1p, g2p, g3p;
      PetscBdPointJac g0Bd, g1Bd, g2Bd, g3Bd;

      if (g >= Nf) continue;
      ierr = PetscDSGetJacobian(prob, f, g, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
      ierr = PetscDSGetJacobianPreconditioner(prob, f, g, &g0p, &g1p, &g2p, &g3p);CHKERRQ(ierr);
      ierr = PetscDSGetBdJacobian(prob, f, g, &g0Bd, &g1Bd, &g2Bd, &g3Bd);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(newprob, fn, gn, g0, g1, g2, g3);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, fn, gn, g0p, g1p, g2p, g3p);CHKERRQ(ierr);
      ierr = PetscDSSetBdJacobian(newprob, fn, gn, g0Bd, g1Bd, g2Bd, g3Bd);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSCopyEquations - Copy all pointwise function pointers to the new problem

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSCopyBoundary(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSCopyEquations(PetscDS prob, PetscDS newprob)
{
  PetscInt       Nf, Ng;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(newprob, PETSCDS_CLASSID, 2);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(newprob, &Ng);CHKERRQ(ierr);
  if (Nf != Ng) SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_SIZ, "Number of fields must match %D != %D", Nf, Ng);
  ierr = PetscDSSelectEquations(prob, Nf, NULL, newprob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*@
  PetscDSCopyConstants - Copy all constants to the new problem

  Not collective

  Input Parameter:
. prob - The PetscDS object

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSCopyBoundary(), PetscDSCopyEquations(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSCopyConstants(PetscDS prob, PetscDS newprob)
{
  PetscInt           Nc;
  const PetscScalar *constants;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(newprob, PETSCDS_CLASSID, 2);
  ierr = PetscDSGetConstants(prob, &Nc, &constants);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(newprob, Nc, (PetscScalar *) constants);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetHeightSubspace(PetscDS prob, PetscInt height, PetscDS *subprob)
{
  PetscInt       dim, Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(subprob, 3);
  if (height == 0) {*subprob = prob; PetscFunctionReturn(0);}
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  if (height > dim) SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_OUTOFRANGE, "DS can only handle height in [0, %D], not %D", dim, height);
  if (!prob->subprobs) {ierr = PetscCalloc1(dim, &prob->subprobs);CHKERRQ(ierr);}
  if (!prob->subprobs[height-1]) {
    PetscInt cdim;

    ierr = PetscDSCreate(PetscObjectComm((PetscObject) prob), &prob->subprobs[height-1]);CHKERRQ(ierr);
    ierr = PetscDSGetCoordinateDimension(prob, &cdim);CHKERRQ(ierr);
    ierr = PetscDSSetCoordinateDimension(prob->subprobs[height-1], cdim);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE      subfe;
      PetscObject  obj;
      PetscClassId id;

      ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {ierr = PetscFEGetHeightSubspace((PetscFE) obj, height, &subfe);CHKERRQ(ierr);}
      else SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported discretization type for field %d", f);
      ierr = PetscDSSetDiscretization(prob->subprobs[height-1], f, (PetscObject) subfe);CHKERRQ(ierr);
    }
  }
  *subprob = prob->subprobs[height-1];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetDiscType_Internal(PetscDS ds, PetscInt f, PetscDiscType *disctype)
{
  PetscObject    obj;
  PetscClassId   id;
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(disctype, 3);
  *disctype = PETSC_DISC_NONE;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  if (f >= Nf) SETERRQ2(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_SIZ, "Field %D must be in [0, %D)", f, Nf);
  ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
  if (obj) {
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) *disctype = PETSC_DISC_FE;
    else                       *disctype = PETSC_DISC_FV;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSDestroy_Basic(PetscDS prob)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(prob->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSInitialize_Basic(PetscDS prob)
{
  PetscFunctionBegin;
  prob->ops->setfromoptions = NULL;
  prob->ops->setup          = NULL;
  prob->ops->view           = NULL;
  prob->ops->destroy        = PetscDSDestroy_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDSBASIC = "basic" - A discrete system with pointwise residual and boundary residual functions

  Level: intermediate

.seealso: PetscDSType, PetscDSCreate(), PetscDSSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDSCreate_Basic(PetscDS prob)
{
  PetscDS_Basic *b;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  ierr       = PetscNewLog(prob, &b);CHKERRQ(ierr);
  prob->data = b;

  ierr = PetscDSInitialize_Basic(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
