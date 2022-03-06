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
  PetscCheckFalse(!r,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDS type: %s", name);

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

static PetscErrorCode PetscDSView_Ascii(PetscDS ds, PetscViewer viewer)
{
  PetscViewerFormat  format;
  const PetscScalar *constants;
  PetscInt           Nf, numConstants, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Discrete System with %d fields\n", Nf);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "  cell total dim %D total comp %D\n", ds->totDim, ds->totComp);CHKERRQ(ierr);
  if (ds->isCohesive) {ierr = PetscViewerASCIIPrintf(viewer, "  cohesive cell\n");CHKERRQ(ierr);}
  for (f = 0; f < Nf; ++f) {
    DSBoundary      b;
    PetscObject     obj;
    PetscClassId    id;
    PetscQuadrature q;
    const char     *name;
    PetscInt        Nc, Nq, Nqc;

    ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
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
    else SETERRQ(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    if (Nc > 1) {ierr = PetscViewerASCIIPrintf(viewer, " %D components", Nc);CHKERRQ(ierr);}
    else        {ierr = PetscViewerASCIIPrintf(viewer, " %D component ", Nc);CHKERRQ(ierr);}
    if (ds->implicit[f]) {ierr = PetscViewerASCIIPrintf(viewer, " (implicit)");CHKERRQ(ierr);}
    else                 {ierr = PetscViewerASCIIPrintf(viewer, " (explicit)");CHKERRQ(ierr);}
    if (q) {
      ierr = PetscQuadratureGetData(q, NULL, &Nqc, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, " (Nq %D Nqc %D)", Nq, Nqc);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, " %D-jet", ds->jetDegree[f]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {ierr = PetscFEView((PetscFE) obj, viewer);CHKERRQ(ierr);}
    else if (id == PETSCFV_CLASSID) {ierr = PetscFVView((PetscFV) obj, viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    for (b = ds->boundary; b; b = b->next) {
      char     *name;
      PetscInt  c, i;

      if (b->field != f) continue;
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Boundary %s (%s) %s\n", b->name, b->lname, DMBoundaryConditionTypes[b->type]);CHKERRQ(ierr);
      if (!b->Nc) {
        ierr = PetscViewerASCIIPrintf(viewer, "  all components\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "  components: ");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
        for (c = 0; c < b->Nc; ++c) {
          if (c > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPrintf(viewer, "%D", b->comps[c]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "  values: ");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      for (i = 0; i < b->Nv; ++i) {
        if (i > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "%D", b->values[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      if (b->func) {
        ierr = PetscDLAddr(b->func, &name);CHKERRQ(ierr);
        if (name) {ierr = PetscViewerASCIIPrintf(viewer, "  func: %s\n", name);CHKERRQ(ierr);}
        else      {ierr = PetscViewerASCIIPrintf(viewer, "  func: %p\n", b->func);CHKERRQ(ierr);}
        ierr = PetscFree(name);CHKERRQ(ierr);
      }
      if (b->func_t) {
        ierr = PetscDLAddr(b->func_t, &name);CHKERRQ(ierr);
        if (name) {ierr = PetscViewerASCIIPrintf(viewer, "  func_t: %s\n", name);CHKERRQ(ierr);}
        else      {ierr = PetscViewerASCIIPrintf(viewer, "  func_t: %p\n", b->func_t);CHKERRQ(ierr);}
        ierr = PetscFree(name);CHKERRQ(ierr);
      }
      ierr = PetscWeakFormView(b->wf, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (numConstants) {
    ierr = PetscViewerASCIIPrintf(viewer, "%D constants\n", numConstants);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (f = 0; f < numConstants; ++f) {ierr = PetscViewerASCIIPrintf(viewer, "%g\n", (double) PetscRealPart(constants[f]));CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscWeakFormView(ds->wf, viewer);CHKERRQ(ierr);
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

  Input Parameters:
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
      b->Nv = len;
      ierr = PetscFree(b->values);CHKERRQ(ierr);
      ierr = PetscMalloc1(len, &b->values);CHKERRQ(ierr);
      ierr = PetscArraycpy(b->values, ids, len);CHKERRQ(ierr);
      ierr = PetscWeakFormRewriteKeys(b->wf, b->label, len, b->values);CHKERRQ(ierr);
    }
    len = 1024;
    ierr = PetscSNPrintf(optname, sizeof(optname), "-bc_%s_comp", b->name);CHKERRQ(ierr);
    ierr = PetscMemzero(ids, sizeof(ids));CHKERRQ(ierr);
    ierr = PetscOptionsIntArray(optname, "List of boundary field components", "", ids, &len, &flg);CHKERRQ(ierr);
    if (flg) {
      b->Nc = len;
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
  const PetscInt Nf   = prob->Nf;
  PetscBool      hasH = PETSC_FALSE;
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
  ierr = PetscCalloc6(Nf+1,&prob->offCohesive[0],Nf+1,&prob->offCohesive[1],Nf+1,&prob->offCohesive[2],Nf+1,&prob->offDerCohesive[0],Nf+1,&prob->offDerCohesive[1],Nf+1,&prob->offDerCohesive[2]);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf,&prob->T,Nf,&prob->Tf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject     obj;
    PetscClassId    id;
    PetscQuadrature q = NULL;
    PetscInt        Nq = 0, Nb, Nc;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    if (prob->jetDegree[f] > 1) hasH = PETSC_TRUE;
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
        ierr = PetscFEGetCellTabulation(fe, prob->jetDegree[f], &prob->T[f]);CHKERRQ(ierr);
        ierr = PetscFEGetFaceTabulation(fe, prob->jetDegree[f], &prob->Tf[f]);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        ierr = PetscFVGetQuadrature(fv, &q);CHKERRQ(ierr);
        ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
        Nb   = Nc;
        ierr = PetscFVGetCellTabulation(fv, &prob->T[f]);CHKERRQ(ierr);
        /* TODO: should PetscFV also have face tabulation? Otherwise there will be a null pointer in prob->basisFace */
      } else SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
    }
    prob->Nc[f]       = Nc;
    prob->Nb[f]       = Nb;
    prob->off[f+1]    = Nc     + prob->off[f];
    prob->offDer[f+1] = Nc*dim + prob->offDer[f];
    prob->offCohesive[0][f+1]    = (prob->cohesive[f] ? Nc : Nc*2)          + prob->offCohesive[0][f];
    prob->offDerCohesive[0][f+1] = (prob->cohesive[f] ? Nc : Nc*2)*dimEmbed + prob->offDerCohesive[0][f];
    prob->offCohesive[1][f]      = (prob->cohesive[f] ? 0 : Nc)             + prob->offCohesive[0][f];
    prob->offDerCohesive[1][f]   = (prob->cohesive[f] ? 0 : Nc)*dimEmbed    + prob->offDerCohesive[0][f];
    prob->offCohesive[2][f+1]    = (prob->cohesive[f] ? Nc : Nc*2)          + prob->offCohesive[2][f];
    prob->offDerCohesive[2][f+1] = (prob->cohesive[f] ? Nc : Nc*2)*dimEmbed + prob->offDerCohesive[2][f];
    if (q) {ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);}
    NqMax          = PetscMax(NqMax, Nq);
    NbMax          = PetscMax(NbMax, Nb);
    NcMax          = PetscMax(NcMax, Nc);
    prob->totDim  += Nb;
    prob->totComp += Nc;
    /* There are two faces for all fields on a cohesive cell, except for cohesive fields */
    if (prob->isCohesive && !prob->cohesive[f]) prob->totDim += Nb;
  }
  prob->offCohesive[1][Nf]    = prob->offCohesive[0][Nf];
  prob->offDerCohesive[1][Nf] = prob->offDerCohesive[0][Nf];
  /* Allocate works space */
  NsMax = 2; /* A non-cohesive discretizations can be used on a cohesive cell, so we need this extra workspace for all DS */
  ierr = PetscMalloc3(NsMax*prob->totComp,&prob->u,NsMax*prob->totComp,&prob->u_t,NsMax*prob->totComp*dimEmbed + (hasH ? NsMax*prob->totComp*dimEmbed*dimEmbed : 0),&prob->u_x);CHKERRQ(ierr);
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
  ierr = PetscFree6(prob->offCohesive[0],prob->offCohesive[1],prob->offCohesive[2],prob->offDerCohesive[0],prob->offDerCohesive[1],prob->offDerCohesive[2]);CHKERRQ(ierr);
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
  PetscInt         *tmpk;
  PetscBool        *tmpc;
  PetscPointFunc   *tmpup;
  PetscSimplePointFunc *tmpexactSol,  *tmpexactSol_t;
  void                **tmpexactCtx, **tmpexactCtx_t;
  void            **tmpctx;
  PetscInt          Nf = prob->Nf, f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (Nf >= NfNew) PetscFunctionReturn(0);
  prob->setup = PETSC_FALSE;
  ierr = PetscDSDestroyStructs_Static(prob);CHKERRQ(ierr);
  ierr = PetscMalloc4(NfNew, &tmpd, NfNew, &tmpi, NfNew, &tmpc, NfNew, &tmpk);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {tmpd[f] = prob->disc[f]; tmpi[f] = prob->implicit[f]; tmpc[f] = prob->cohesive[f]; tmpk[f] = prob->jetDegree[f];}
  for (f = Nf; f < NfNew; ++f) {tmpd[f] = NULL; tmpi[f] = PETSC_TRUE, tmpc[f] = PETSC_FALSE; tmpk[f] = 1;}
  ierr = PetscFree4(prob->disc, prob->implicit, prob->cohesive, prob->jetDegree);CHKERRQ(ierr);
  ierr = PetscWeakFormSetNumFields(prob->wf, NfNew);CHKERRQ(ierr);
  prob->Nf        = NfNew;
  prob->disc      = tmpd;
  prob->implicit  = tmpi;
  prob->cohesive  = tmpc;
  prob->jetDegree = tmpk;
  ierr = PetscCalloc2(NfNew, &tmpup, NfNew, &tmpctx);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) tmpup[f] = prob->update[f];
  for (f = 0; f < Nf; ++f) tmpctx[f] = prob->ctx[f];
  for (f = Nf; f < NfNew; ++f) tmpup[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpctx[f] = NULL;
  ierr = PetscFree2(prob->update, prob->ctx);CHKERRQ(ierr);
  prob->update = tmpup;
  prob->ctx = tmpctx;
  ierr = PetscCalloc4(NfNew, &tmpexactSol, NfNew, &tmpexactCtx, NfNew, &tmpexactSol_t, NfNew, &tmpexactCtx_t);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) tmpexactSol[f] = prob->exactSol[f];
  for (f = 0; f < Nf; ++f) tmpexactCtx[f] = prob->exactCtx[f];
  for (f = 0; f < Nf; ++f) tmpexactSol_t[f] = prob->exactSol_t[f];
  for (f = 0; f < Nf; ++f) tmpexactCtx_t[f] = prob->exactCtx_t[f];
  for (f = Nf; f < NfNew; ++f) tmpexactSol[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactCtx[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactSol_t[f] = NULL;
  for (f = Nf; f < NfNew; ++f) tmpexactCtx_t[f] = NULL;
  ierr = PetscFree4(prob->exactSol, prob->exactCtx, prob->exactSol_t, prob->exactCtx_t);CHKERRQ(ierr);
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
PetscErrorCode PetscDSDestroy(PetscDS *ds)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ds) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ds), PETSCDS_CLASSID, 1);

  if (--((PetscObject)(*ds))->refct > 0) {*ds = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*ds))->refct = 0;
  if ((*ds)->subprobs) {
    PetscInt dim, d;

    ierr = PetscDSGetSpatialDimension(*ds, &dim);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {ierr = PetscDSDestroy(&(*ds)->subprobs[d]);CHKERRQ(ierr);}
  }
  ierr = PetscFree((*ds)->subprobs);CHKERRQ(ierr);
  ierr = PetscDSDestroyStructs_Static(*ds);CHKERRQ(ierr);
  for (f = 0; f < (*ds)->Nf; ++f) {
    ierr = PetscObjectDereference((*ds)->disc[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree4((*ds)->disc, (*ds)->implicit, (*ds)->cohesive, (*ds)->jetDegree);CHKERRQ(ierr);
  ierr = PetscWeakFormDestroy(&(*ds)->wf);CHKERRQ(ierr);
  ierr = PetscFree2((*ds)->update,(*ds)->ctx);CHKERRQ(ierr);
  ierr = PetscFree4((*ds)->exactSol,(*ds)->exactCtx,(*ds)->exactSol_t,(*ds)->exactCtx_t);CHKERRQ(ierr);
  if ((*ds)->ops->destroy) {ierr = (*(*ds)->ops->destroy)(*ds);CHKERRQ(ierr);}
  ierr = PetscDSDestroyBoundary(*ds);CHKERRQ(ierr);
  ierr = PetscFree((*ds)->constants);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDSCreate - Creates an empty PetscDS object. The type can then be set with PetscDSSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscDS object

  Output Parameter:
. ds   - The PetscDS object

  Level: beginner

.seealso: PetscDSSetType(), PETSCDSBASIC
@*/
PetscErrorCode PetscDSCreate(MPI_Comm comm, PetscDS *ds)
{
  PetscDS        p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ds, 2);
  *ds  = NULL;
  ierr = PetscDSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, PETSCDS_CLASSID, "PetscDS", "Discrete System", "PetscDS", comm, PetscDSDestroy, PetscDSView);CHKERRQ(ierr);

  p->Nf           = 0;
  p->setup        = PETSC_FALSE;
  p->numConstants = 0;
  p->constants    = NULL;
  p->dimEmbed     = -1;
  p->useJacPre    = PETSC_TRUE;
  ierr = PetscWeakFormCreate(comm, &p->wf);CHKERRQ(ierr);

  *ds = p;
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
      else SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", 0);
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
  PetscCheckFalse(prob->dimEmbed < 0,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONGSTATE, "No coordinate dimension set for this DS");
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
  PetscCheckFalse(dimEmbed < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate dimension must be non-negative, not %D", dimEmbed);
  prob->dimEmbed = dimEmbed;
  PetscFunctionReturn(0);
}

/*@
  PetscDSIsCohesive - Returns the flag indicating that this DS is for a cohesive cell

  Not collective

  Input Parameter:
. ds - The PetscDS object

  Output Parameter:
. isCohesive - The flag

  Level: developer

.seealso: PetscDSGetNumCohesive(), PetscDSGetCohesive(), PetscDSSetCohesive(), PetscDSCreate()
@*/
PetscErrorCode PetscDSIsCohesive(PetscDS ds, PetscBool *isCohesive)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(isCohesive, 2);
  *isCohesive = ds->isCohesive;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetNumCohesive - Returns the numer of cohesive fields, meaning those defined on the interior of a cohesive cell

  Not collective

  Input Parameter:
. ds - The PetscDS object

  Output Parameter:
. numCohesive - The number of cohesive fields

  Level: developer

.seealso: PetscDSSetCohesive(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetNumCohesive(PetscDS ds, PetscInt *numCohesive)
{
  PetscInt f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(numCohesive, 2);
  *numCohesive = 0;
  for (f = 0;  f < ds->Nf; ++f) *numCohesive += ds->cohesive[f] ? 1 : 0;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetCohesive - Returns the flag indicating that a field is cohesive, meaning it is defined on the interior of a cohesive cell

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- f  - The field index

  Output Parameter:
. isCohesive - The flag

  Level: developer

.seealso: PetscDSSetCohesive(), PetscDSIsCohesive(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetCohesive(PetscDS ds, PetscInt f, PetscBool *isCohesive)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(isCohesive, 3);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  *isCohesive = ds->cohesive[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetCohesive - Set the flag indicating that a field is cohesive, meaning it is defined on the interior of a cohesive cell

  Not collective

  Input Parameters:
+ ds - The PetscDS object
. f  - The field index
- isCohesive - The flag for a cohesive field

  Level: developer

.seealso: PetscDSGetCohesive(), PetscDSIsCohesive(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetCohesive(PetscDS ds, PetscInt f, PetscBool isCohesive)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ds->cohesive[f] = isCohesive;
  ds->isCohesive = PETSC_FALSE;
  for (i = 0; i < ds->Nf; ++i) ds->isCohesive = ds->isCohesive || ds->cohesive[f] ? PETSC_TRUE : PETSC_FALSE;
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
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(disc, 3);
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
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
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
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
    ierr = PetscDSSetJetDegree(prob, f, 1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetWeakForm - Returns the weak form object

  Not collective

  Input Parameter:
. ds - The PetscDS object

  Output Parameter:
. wf - The weak form object

  Level: beginner

.seealso: PetscDSSetWeakForm(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetWeakForm(PetscDS ds, PetscWeakForm *wf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(wf, 2);
  *wf = ds->wf;
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetWeakForm - Sets the weak form object

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- wf - The weak form object

  Level: beginner

.seealso: PetscDSGetWeakForm(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetWeakForm(PetscDS ds, PetscWeakForm wf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 2);
  ierr = PetscObjectDereference((PetscObject) ds->wf);CHKERRQ(ierr);
  ds->wf = wf;
  ierr = PetscObjectReference((PetscObject) wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetNumFields(wf, ds->Nf);CHKERRQ(ierr);
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
  else SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", 0);
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  prob->implicit[f] = implicit;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetJetDegree - Returns the highest derivative for this field equation, or the k-jet that the discretization needs to tabulate.

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- f  - The field number

  Output Parameter:
. k  - The highest derivative we need to tabulate

  Level: developer

.seealso: PetscDSSetJetDegree(), PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetJetDegree(PetscDS ds, PetscInt f, PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(k, 3);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  *k = ds->jetDegree[f];
  PetscFunctionReturn(0);
}

/*@
  PetscDSSetJetDegree - Set the highest derivative for this field equation, or the k-jet that the discretization needs to tabulate.

  Not collective

  Input Parameters:
+ ds - The PetscDS object
. f  - The field number
- k  - The highest derivative we need to tabulate

  Level: developer

.seealso: PetscDSGetJetDegree(), PetscDSSetDiscretization(), PetscDSAddDiscretization(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSetJetDegree(PetscDS ds, PetscInt f, PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ds->jetDegree[f] = k;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetObjective(PetscDS ds, PetscInt f,
                                   void (**obj)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[]))
{
  PetscPointFunc *tmp;
  PetscInt        n;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(obj, 3);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ierr = PetscWeakFormGetObjective(ds->wf, NULL, 0, f, 0, &n, &tmp);CHKERRQ(ierr);
  *obj = tmp ? tmp[0] : NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSSetObjective(PetscDS ds, PetscInt f,
                                   void (*obj)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (obj) PetscValidFunction(obj, 3);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscWeakFormSetIndexObjective(ds->wf, NULL, 0, f, 0, 0, obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetResidual - Get the pointwise residual function for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
- f  - The test field number

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
PetscErrorCode PetscDSGetResidual(PetscDS ds, PetscInt f,
                                  void (**f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                  void (**f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscPointFunc *tmp0, *tmp1;
  PetscInt        n0, n1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ierr = PetscWeakFormGetResidual(ds->wf, NULL, 0, f, 0, &n0, &tmp0, &n1, &tmp1);CHKERRQ(ierr);
  *f0  = tmp0 ? tmp0[0] : NULL;
  *f1  = tmp1 ? tmp1[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetResidual - Set the pointwise residual function for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
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
PetscErrorCode PetscDSSetResidual(PetscDS ds, PetscInt f,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (f0) PetscValidFunction(f0, 3);
  if (f1) PetscValidFunction(f1, 4);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscWeakFormSetIndexResidual(ds->wf, NULL, 0, f, 0, 0, f0, 0, f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetRHSResidual - Get the pointwise RHS residual function for explicit timestepping for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
- f  - The test field number

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

.seealso: PetscDSSetRHSResidual()
@*/
PetscErrorCode PetscDSGetRHSResidual(PetscDS ds, PetscInt f,
                                     void (**f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                     void (**f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscPointFunc *tmp0, *tmp1;
  PetscInt        n0, n1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ierr = PetscWeakFormGetResidual(ds->wf, NULL, 0, f, 100, &n0, &tmp0, &n1, &tmp1);CHKERRQ(ierr);
  *f0  = tmp0 ? tmp0[0] : NULL;
  *f1  = tmp1 ? tmp1[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetRHSResidual - Set the pointwise residual function for explicit timestepping for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
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
PetscErrorCode PetscDSSetRHSResidual(PetscDS ds, PetscInt f,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (f0) PetscValidFunction(f0, 3);
  if (f1) PetscValidFunction(f1, 4);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscWeakFormSetIndexResidual(ds->wf, NULL, 0, f, 100, 0, f0, 0, f1);CHKERRQ(ierr);
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
PetscErrorCode PetscDSHasJacobian(PetscDS ds, PetscBool *hasJac)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscWeakFormHasJacobian(ds->wf, hasJac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetJacobian - Get the pointwise Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
- g  - The field number

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
PetscErrorCode PetscDSGetJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscPointJac *tmp0, *tmp1, *tmp2, *tmp3;
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscCheckFalse((g < 0) || (g >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, ds->Nf);
  ierr = PetscWeakFormGetJacobian(ds->wf, NULL, 0, f, g, 0, &n0, &tmp0, &n1, &tmp1, &n2, &tmp2, &n3, &tmp3);CHKERRQ(ierr);
  *g0  = tmp0 ? tmp0[0] : NULL;
  *g1  = tmp1 ? tmp1[0] : NULL;
  *g2  = tmp2 ? tmp2[0] : NULL;
  *g3  = tmp3 ? tmp3[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetJacobian - Set the pointwise Jacobian function for given test and basis fields

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
. g  - The field number
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
PetscErrorCode PetscDSSetJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  PetscCheckFalse(g < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscWeakFormSetIndexJacobian(ds->wf, NULL, 0, f, g, 0, 0, g0, 0, g1, 0, g2, 0, g3);CHKERRQ(ierr);
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
PetscErrorCode PetscDSHasJacobianPreconditioner(PetscDS ds, PetscBool *hasJacPre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  *hasJacPre = PETSC_FALSE;
  if (!ds->useJacPre) PetscFunctionReturn(0);
  ierr = PetscWeakFormHasJacobianPreconditioner(ds->wf, hasJacPre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetJacobianPreconditioner - Get the pointwise Jacobian preconditioner function for given test and basis field. If this is missing, the system matrix is used to build the preconditioner.

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
- g  - The field number

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
PetscErrorCode PetscDSGetJacobianPreconditioner(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscPointJac *tmp0, *tmp1, *tmp2, *tmp3;
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscCheckFalse((g < 0) || (g >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, ds->Nf);
  ierr = PetscWeakFormGetJacobianPreconditioner(ds->wf, NULL, 0, f, g, 0, &n0, &tmp0, &n1, &tmp1, &n2, &tmp2, &n3, &tmp3);CHKERRQ(ierr);
  *g0  = tmp0 ? tmp0[0] : NULL;
  *g1  = tmp1 ? tmp1[0] : NULL;
  *g2  = tmp2 ? tmp2[0] : NULL;
  *g3  = tmp3 ? tmp3[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetJacobianPreconditioner - Set the pointwise Jacobian preconditioner function for given test and basis fields. If this is missing, the system matrix is used to build the preconditioner.

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
. g  - The field number
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
PetscErrorCode PetscDSSetJacobianPreconditioner(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  PetscCheckFalse(g < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscWeakFormSetIndexJacobianPreconditioner(ds->wf, NULL, 0, f, g, 0, 0, g0, 0, g1, 0, g2, 0, g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSHasDynamicJacobian - Signals that a dynamic Jacobian, dF/du_t, has been set

  Not collective

  Input Parameter:
. ds - The PetscDS

  Output Parameter:
. hasDynJac - flag that pointwise function for dynamic Jacobian has been set

  Level: intermediate

.seealso: PetscDSGetDynamicJacobian(), PetscDSSetDynamicJacobian(), PetscDSGetJacobian()
@*/
PetscErrorCode PetscDSHasDynamicJacobian(PetscDS ds, PetscBool *hasDynJac)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscWeakFormHasDynamicJacobian(ds->wf, hasDynJac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetDynamicJacobian - Get the pointwise dynamic Jacobian, dF/du_t, function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
- g  - The field number

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
PetscErrorCode PetscDSGetDynamicJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscPointJac *tmp0, *tmp1, *tmp2, *tmp3;
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscCheckFalse((g < 0) || (g >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, ds->Nf);
  ierr = PetscWeakFormGetDynamicJacobian(ds->wf, NULL, 0, f, g, 0, &n0, &tmp0, &n1, &tmp1, &n2, &tmp2, &n3, &tmp3);CHKERRQ(ierr);
  *g0  = tmp0 ? tmp0[0] : NULL;
  *g1  = tmp1 ? tmp1[0] : NULL;
  *g2  = tmp2 ? tmp2[0] : NULL;
  *g3  = tmp3 ? tmp3[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetDynamicJacobian - Set the pointwise dynamic Jacobian, dF/du_t, function for given test and basis fields

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
. g  - The field number
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
PetscErrorCode PetscDSSetDynamicJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  PetscCheckFalse(g < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscWeakFormSetIndexDynamicJacobian(ds->wf, NULL, 0, f, g, 0, 0, g0, 0, g1, 0, g2, 0, g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetRiemannSolver - Returns the Riemann solver for the given field

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- f  - The field number

  Output Parameter:
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
PetscErrorCode PetscDSGetRiemannSolver(PetscDS ds, PetscInt f,
                                       void (**r)(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx))
{
  PetscRiemannFunc *tmp;
  PetscInt          n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(r, 3);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ierr = PetscWeakFormGetRiemannSolver(ds->wf, NULL, 0, f, 0, &n, &tmp);CHKERRQ(ierr);
  *r   = tmp ? tmp[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetRiemannSolver - Sets the Riemann solver for the given field

  Not collective

  Input Parameters:
+ ds - The PetscDS object
. f  - The field number
- r  - Riemann solver

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
PetscErrorCode PetscDSSetRiemannSolver(PetscDS ds, PetscInt f,
                                       void (*r)(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (r) PetscValidFunction(r, 3);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscWeakFormSetIndexRiemannSolver(ds->wf, NULL, 0, f, 0, 0, r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetUpdate - Get the pointwise update function for a given field

  Not collective

  Input Parameters:
+ ds - The PetscDS
- f  - The field number

  Output Parameter:
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
PetscErrorCode PetscDSGetUpdate(PetscDS ds, PetscInt f,
                                  void (**update)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uNew[]))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  if (update) {PetscValidPointer(update, 3); *update = ds->update[f];}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetUpdate - Set the pointwise update function for a given field

  Not collective

  Input Parameters:
+ ds     - The PetscDS
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
PetscErrorCode PetscDSSetUpdate(PetscDS ds, PetscInt f,
                                void (*update)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uNew[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (update) PetscValidFunction(update, 3);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(ds, f+1);CHKERRQ(ierr);
  ds->update[f] = update;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSGetContext(PetscDS ds, PetscInt f, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscValidPointer(ctx, 3);
  *(void**)ctx = ds->ctx[f];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDSSetContext(PetscDS ds, PetscInt f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscDSEnlarge_Static(ds, f+1);CHKERRQ(ierr);
  ds->ctx[f] = ctx;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdResidual - Get the pointwise boundary residual function for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
- f  - The test field number

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
PetscErrorCode PetscDSGetBdResidual(PetscDS ds, PetscInt f,
                                    void (**f0)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]),
                                    void (**f1)(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]))
{
  PetscBdPointFunc *tmp0, *tmp1;
  PetscInt          n0, n1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  ierr = PetscWeakFormGetBdResidual(ds->wf, NULL, 0, f, 0, &n0, &tmp0, &n1, &tmp1);CHKERRQ(ierr);
  *f0  = tmp0 ? tmp0[0] : NULL;
  *f1  = tmp1 ? tmp1[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdResidual - Get the pointwise boundary residual function for a given test field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
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
PetscErrorCode PetscDSSetBdResidual(PetscDS ds, PetscInt f,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = PetscWeakFormSetIndexBdResidual(ds->wf, NULL, 0, f, 0, 0, f0, 0, f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDSHasBdJacobian - Signals that boundary Jacobian functions have been set

  Not collective

  Input Parameter:
. ds - The PetscDS

  Output Parameter:
. hasBdJac - flag that pointwise function for the boundary Jacobian has been set

  Level: intermediate

.seealso: PetscDSHasJacobian(), PetscDSSetBdJacobian(), PetscDSGetBdJacobian()
@*/
PetscErrorCode PetscDSHasBdJacobian(PetscDS ds, PetscBool *hasBdJac)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidBoolPointer(hasBdJac, 2);
  ierr = PetscWeakFormHasBdJacobian(ds->wf, hasBdJac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdJacobian - Get the pointwise boundary Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
- g  - The field number

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
PetscErrorCode PetscDSGetBdJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscBdPointJac *tmp0, *tmp1, *tmp2, *tmp3;
  PetscInt         n0, n1, n2, n3;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscCheckFalse((g < 0) || (g >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, ds->Nf);
  ierr = PetscWeakFormGetBdJacobian(ds->wf, NULL, 0, f, g, 0, &n0, &tmp0, &n1, &tmp1, &n2, &tmp2, &n3, &tmp3);CHKERRQ(ierr);
  *g0  = tmp0 ? tmp0[0] : NULL;
  *g1  = tmp1 ? tmp1[0] : NULL;
  *g2  = tmp2 ? tmp2[0] : NULL;
  *g3  = tmp3 ? tmp3[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdJacobian - Set the pointwise boundary Jacobian function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
. g  - The field number
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
PetscErrorCode PetscDSSetBdJacobian(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  PetscCheckFalse(g < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscWeakFormSetIndexBdJacobian(ds->wf, NULL, 0, f, g, 0, 0, g0, 0, g1, 0, g2, 0, g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDSHasBdJacobianPreconditioner - Signals that boundary Jacobian preconditioner functions have been set

  Not collective

  Input Parameter:
. ds - The PetscDS

  Output Parameter:
. hasBdJac - flag that pointwise function for the boundary Jacobian preconditioner has been set

  Level: intermediate

.seealso: PetscDSHasJacobian(), PetscDSSetBdJacobian(), PetscDSGetBdJacobian()
@*/
PetscErrorCode PetscDSHasBdJacobianPreconditioner(PetscDS ds, PetscBool *hasBdJacPre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidBoolPointer(hasBdJacPre, 2);
  ierr = PetscWeakFormHasBdJacobianPreconditioner(ds->wf, hasBdJacPre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetBdJacobianPreconditioner - Get the pointwise boundary Jacobian preconditioner function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
- g  - The field number

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
PetscErrorCode PetscDSGetBdJacobianPreconditioner(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscBdPointJac *tmp0, *tmp1, *tmp2, *tmp3;
  PetscInt         n0, n1, n2, n3;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  PetscCheckFalse((g < 0) || (g >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", g, ds->Nf);
  ierr = PetscWeakFormGetBdJacobianPreconditioner(ds->wf, NULL, 0, f, g, 0, &n0, &tmp0, &n1, &tmp1, &n2, &tmp2, &n3, &tmp3);CHKERRQ(ierr);
  *g0  = tmp0 ? tmp0[0] : NULL;
  *g1  = tmp1 ? tmp1[0] : NULL;
  *g2  = tmp2 ? tmp2[0] : NULL;
  *g3  = tmp3 ? tmp3[0] : NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSSetBdJacobianPreconditioner - Set the pointwise boundary Jacobian preconditioner function for given test and basis field

  Not collective

  Input Parameters:
+ ds - The PetscDS
. f  - The test field number
. g  - The field number
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
PetscErrorCode PetscDSSetBdJacobianPreconditioner(PetscDS ds, PetscInt f, PetscInt g,
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
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  if (g0) PetscValidFunction(g0, 4);
  if (g1) PetscValidFunction(g1, 5);
  if (g2) PetscValidFunction(g2, 6);
  if (g3) PetscValidFunction(g3, 7);
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  PetscCheckFalse(g < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", g);
  ierr = PetscWeakFormSetIndexBdJacobianPreconditioner(ds->wf, NULL, 0, f, g, 0, 0, g0, 0, g1, 0, g2, 0, g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDSGetExactSolution - Get the pointwise exact solution function for a given test field

  Not collective

  Input Parameters:
+ prob - The PetscDS
- f    - The test field number

  Output Parameters:
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
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
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
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

  Output Parameters:
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
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
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
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
  PetscCheckFalse(g == prob->Nf,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Field not found in PetscDS.");
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
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
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  *off = 0;
  for (g = 0; g < f; ++g) {
    ierr = PetscDSGetFieldSize(prob, g, &size);CHKERRQ(ierr);
    *off += size;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetFieldOffsetCohesive - Returns the offset of the given field in the full space basis on a cohesive cell

  Not collective

  Input Parameters:
+ prob - The PetscDS object
- f - The field number

  Output Parameter:
. off - The offset

  Level: beginner

.seealso: PetscDSGetFieldSize(), PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetFieldOffsetCohesive(PetscDS ds, PetscInt f, PetscInt *off)
{
  PetscInt       size, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(off, 3);
  PetscCheckFalse((f < 0) || (f >= ds->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, ds->Nf);
  *off = 0;
  for (g = 0; g < f; ++g) {
    PetscBool cohesive;

    ierr = PetscDSGetCohesive(ds, g, &cohesive);CHKERRQ(ierr);
    ierr = PetscDSGetFieldSize(ds, g, &size);CHKERRQ(ierr);
    *off += cohesive ? size : size*2;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(off, 3);
  PetscCheckFalse((f < 0) || (f >= prob->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, prob->Nf);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 2);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 2);
  ierr = PetscDSSetUp(prob);CHKERRQ(ierr);
  *offsets = prob->offDer;
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponentOffsetsCohesive - Returns the offset of each field on an evaluation point

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- s  - The cohesive side, 0 for negative, 1 for positive, 2 for cohesive

  Output Parameter:
. offsets - The offsets

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponentOffsetsCohesive(PetscDS ds, PetscInt s, PetscInt *offsets[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 3);
  PetscCheckFalse(!ds->isCohesive,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cohesive offsets are only valid for a cohesive DS");
  PetscCheckFalse((s < 0) || (s > 2),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cohesive side %D is not in [0, 2]", s);
  ierr = PetscDSSetUp(ds);CHKERRQ(ierr);
  *offsets = ds->offCohesive[s];
  PetscFunctionReturn(0);
}

/*@
  PetscDSGetComponentDerivativeOffsetsCohesive - Returns the offset of each field derivative on an evaluation point

  Not collective

  Input Parameters:
+ ds - The PetscDS object
- s  - The cohesive side, 0 for negative, 1 for positive, 2 for cohesive

  Output Parameter:
. offsets - The offsets

  Level: beginner

.seealso: PetscDSGetNumFields(), PetscDSCreate()
@*/
PetscErrorCode PetscDSGetComponentDerivativeOffsetsCohesive(PetscDS ds, PetscInt s, PetscInt *offsets[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidPointer(offsets, 3);
  PetscCheckFalse(!ds->isCohesive,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cohesive offsets are only valid for a cohesive DS");
  PetscCheckFalse((s < 0) || (s > 2),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cohesive side %D is not in [0, 2]", s);
  ierr = PetscDSSetUp(ds);CHKERRQ(ierr);
  *offsets = ds->offDerCohesive[s];
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
. Tf - The basis function and derivative tabulation on each local face at quadrature points for each and field

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
  PetscDSAddBoundary - Add a boundary condition to the model. The pointwise functions are used to provide boundary values for essential boundary conditions. In FEM, they are acting upon by dual basis functionals to generate FEM coefficients which are fixed. Natural boundary conditions signal to PETSc that boundary integrals should be performaed, using the kernels from PetscDSSetBdResidual().

  Collective on ds

  Input Parameters:
+ ds       - The PetscDS object
. type     - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name     - The BC name
. label    - The label defining constrained points
. Nv       - The number of DMLabel values for constrained points
. values   - An array of label values for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components (0 will constrain all fields)
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time derivative of the boundary values, or NULL
- ctx      - An optional user context for bcFunc

  Output Parameters:
- bd       - The boundary number

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Note:
  Both bcFunc abd bcFunc_t will depend on the boundary condition type. If the type if DM_BC_ESSENTIAL, Then the calling sequence is:

$ bcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[])

  If the type is DM_BC_ESSENTIAL_FIELD or other _FIELD value, then the calling sequence is:

$ bcFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$        PetscReal time, const PetscReal x[], PetscScalar bcval[])

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
- bcval - output values at the current point

  Level: developer

.seealso: PetscDSAddBoundaryByName(), PetscDSGetBoundary(), PetscDSSetResidual(), PetscDSSetBdResidual()
@*/
PetscErrorCode PetscDSAddBoundary(PetscDS ds, DMBoundaryConditionType type, const char name[], DMLabel label, PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx, PetscInt *bd)
{
  DSBoundary     head = ds->boundary, b;
  PetscInt       n    = 0;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ds, type, 2);
  PetscValidCharPointer(name, 3);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  PetscValidLogicalCollectiveInt(ds, Nv, 5);
  PetscValidLogicalCollectiveInt(ds, field, 7);
  PetscValidLogicalCollectiveInt(ds, Nc, 8);
  if (Nc > 0) {
    PetscInt *fcomps;
    PetscInt  c;

    ierr = PetscDSGetComponents(ds, &fcomps);CHKERRQ(ierr);
    PetscCheckFalse(Nc > fcomps[field],PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_OUTOFRANGE, "Number of constrained components %D > %D components for field %D", Nc, fcomps[field], field);
    for (c = 0; c < Nc; ++c) {
      PetscCheckFalse(comps[c] < 0 || comps[c] >= fcomps[field],PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_OUTOFRANGE, "Constrained component[%D] %D not in [0, %D) components for field %D", c, comps[c], fcomps[field], field);
    }
  }
  ierr = PetscNew(&b);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  ierr = PetscWeakFormCreate(PETSC_COMM_SELF, &b->wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetNumFields(b->wf, ds->Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nv, &b->values);CHKERRQ(ierr);
  if (Nv) {ierr = PetscArraycpy(b->values, values, Nv);CHKERRQ(ierr);}
  ierr = PetscMalloc1(Nc, &b->comps);CHKERRQ(ierr);
  if (Nc) {ierr = PetscArraycpy(b->comps, comps, Nc);CHKERRQ(ierr);}
  ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(lname, (char **) &b->lname);CHKERRQ(ierr);
  b->type   = type;
  b->label  = label;
  b->Nv     = Nv;
  b->field  = field;
  b->Nc     = Nc;
  b->func   = bcFunc;
  b->func_t = bcFunc_t;
  b->ctx    = ctx;
  b->next   = NULL;
  /* Append to linked list so that we can preserve the order */
  if (!head) ds->boundary = b;
  while (head) {
    if (!head->next) {
      head->next = b;
      head       = b;
    }
    head = head->next;
    ++n;
  }
  if (bd) {PetscValidIntPointer(bd, 13); *bd = n;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSAddBoundaryByName - Add a boundary condition to the model. The pointwise functions are used to provide boundary values for essential boundary conditions. In FEM, they are acting upon by dual basis functionals to generate FEM coefficients which are fixed. Natural boundary conditions signal to PETSc that boundary integrals should be performaed, using the kernels from PetscDSSetBdResidual().

  Collective on ds

  Input Parameters:
+ ds       - The PetscDS object
. type     - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name     - The BC name
. lname    - The naem of the label defining constrained points
. Nv       - The number of DMLabel values for constrained points
. values   - An array of label values for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components (0 will constrain all fields)
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time derivative of the boundary values, or NULL
- ctx      - An optional user context for bcFunc

  Output Parameters:
- bd       - The boundary number

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Note:
  This function should only be used with DMForest currently, since labels cannot be defined before the underlygin Plex is built.

  Both bcFunc abd bcFunc_t will depend on the boundary condition type. If the type if DM_BC_ESSENTIAL, Then the calling sequence is:

$ bcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[])

  If the type is DM_BC_ESSENTIAL_FIELD or other _FIELD value, then the calling sequence is:

$ bcFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$        PetscReal time, const PetscReal x[], PetscScalar bcval[])

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
- bcval - output values at the current point

  Level: developer

.seealso: PetscDSAddBoundary(), PetscDSGetBoundary(), PetscDSSetResidual(), PetscDSSetBdResidual()
@*/
PetscErrorCode PetscDSAddBoundaryByName(PetscDS ds, DMBoundaryConditionType type, const char name[], const char lname[], PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx, PetscInt *bd)
{
  DSBoundary     head = ds->boundary, b;
  PetscInt       n    = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ds, type, 2);
  PetscValidCharPointer(name, 3);
  PetscValidCharPointer(lname, 4);
  PetscValidLogicalCollectiveInt(ds, Nv, 5);
  PetscValidLogicalCollectiveInt(ds, field, 7);
  PetscValidLogicalCollectiveInt(ds, Nc, 8);
  ierr = PetscNew(&b);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  ierr = PetscWeakFormCreate(PETSC_COMM_SELF, &b->wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetNumFields(b->wf, ds->Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nv, &b->values);CHKERRQ(ierr);
  if (Nv) {ierr = PetscArraycpy(b->values, values, Nv);CHKERRQ(ierr);}
  ierr = PetscMalloc1(Nc, &b->comps);CHKERRQ(ierr);
  if (Nc) {ierr = PetscArraycpy(b->comps, comps, Nc);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(lname, (char **) &b->lname);CHKERRQ(ierr);
  b->type   = type;
  b->label  = NULL;
  b->Nv     = Nv;
  b->field  = field;
  b->Nc     = Nc;
  b->func   = bcFunc;
  b->func_t = bcFunc_t;
  b->ctx    = ctx;
  b->next   = NULL;
  /* Append to linked list so that we can preserve the order */
  if (!head) ds->boundary = b;
  while (head) {
    if (!head->next) {
      head->next = b;
      head       = b;
    }
    head = head->next;
    ++n;
  }
  if (bd) {PetscValidIntPointer(bd, 13); *bd = n;}
  PetscFunctionReturn(0);
}

/*@C
  PetscDSUpdateBoundary - Change a boundary condition for the model. The pointwise functions are used to provide boundary values for essential boundary conditions. In FEM, they are acting upon by dual basis functionals to generate FEM coefficients which are fixed. Natural boundary conditions signal to PETSc that boundary integrals should be performaed, using the kernels from PetscDSSetBdResidual().

  Input Parameters:
+ ds       - The PetscDS object
. bd       - The boundary condition number
. type     - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name     - The BC name
. label    - The label defining constrained points
. Nv       - The number of DMLabel ids for constrained points
. values   - An array of ids for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time derivative of the boundary values, or NULL
- ctx      - An optional user context for bcFunc

  Note:
  The boundary condition number is the order in which it was registered. The user can get the number of boundary conditions from PetscDSGetNumBoundary(). See PetscDSAddBoundary() for a description of the calling sequences for the callbacks.

  Level: developer

.seealso: PetscDSAddBoundary(), PetscDSGetBoundary(), PetscDSGetNumBoundary()
@*/
PetscErrorCode PetscDSUpdateBoundary(PetscDS ds, PetscInt bd, DMBoundaryConditionType type, const char name[], DMLabel label, PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx)
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
  PetscCheckFalse(!b,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (name) {
    ierr = PetscFree(b->name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  }
  b->type = type;
  if (label) {
    const char *name;

    b->label = label;
    ierr = PetscFree(b->lname);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &b->lname);CHKERRQ(ierr);
  }
  if (Nv >= 0) {
    b->Nv = Nv;
    ierr = PetscFree(b->values);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nv, &b->values);CHKERRQ(ierr);
    if (Nv) {ierr = PetscArraycpy(b->values, values, Nv);CHKERRQ(ierr);}
  }
  if (field >= 0) b->field = field;
  if (Nc >= 0) {
    b->Nc = Nc;
    ierr = PetscFree(b->comps);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nc, &b->comps);CHKERRQ(ierr);
    if (Nc) {ierr = PetscArraycpy(b->comps, comps, Nc);CHKERRQ(ierr);}
  }
  if (bcFunc)   b->func   = bcFunc;
  if (bcFunc_t) b->func_t = bcFunc_t;
  if (ctx)      b->ctx    = ctx;
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
+ wf       - The PetscWeakForm holding the pointwise functions
. type     - The type of condition, e.g. DM_BC_ESSENTIAL/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name     - The BC name
. label    - The label defining constrained points
. Nv       - The number of DMLabel ids for constrained points
. values   - An array of ids for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time derivative of the boundary values
- ctx      - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: PetscDSAddBoundary()
@*/
PetscErrorCode PetscDSGetBoundary(PetscDS ds, PetscInt bd, PetscWeakForm *wf, DMBoundaryConditionType *type, const char *name[], DMLabel *label, PetscInt *Nv, const PetscInt *values[], PetscInt *field, PetscInt *Nc, const PetscInt *comps[], void (**func)(void), void (**func_t)(void), void **ctx)
{
  DSBoundary b = ds->boundary;
  PetscInt   n = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  while (b) {
    if (n == bd) break;
    b = b->next;
    ++n;
  }
  PetscCheckFalse(!b,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (wf) {
    PetscValidPointer(wf, 3);
    *wf = b->wf;
  }
  if (type) {
    PetscValidPointer(type, 4);
    *type = b->type;
  }
  if (name) {
    PetscValidPointer(name, 5);
    *name = b->name;
  }
  if (label) {
    PetscValidPointer(label, 6);
    *label = b->label;
  }
  if (Nv) {
    PetscValidIntPointer(Nv, 7);
    *Nv = b->Nv;
  }
  if (values) {
    PetscValidPointer(values, 8);
    *values = b->values;
  }
  if (field) {
    PetscValidIntPointer(field, 9);
    *field = b->field;
  }
  if (Nc) {
    PetscValidIntPointer(Nc, 10);
    *Nc = b->Nc;
  }
  if (comps) {
    PetscValidPointer(comps, 11);
    *comps = b->comps;
  }
  if (func) {
    PetscValidPointer(func, 12);
    *func = b->func;
  }
  if (func_t) {
    PetscValidPointer(func_t, 13);
    *func_t = b->func_t;
  }
  if (ctx) {
    PetscValidPointer(ctx, 14);
    *ctx = b->ctx;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSBoundaryDuplicate_Internal(DSBoundary b, DSBoundary *bNew)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(bNew);CHKERRQ(ierr);
  ierr = PetscWeakFormCreate(PETSC_COMM_SELF, &(*bNew)->wf);CHKERRQ(ierr);
  ierr = PetscWeakFormCopy(b->wf, (*bNew)->wf);CHKERRQ(ierr);
  ierr = PetscStrallocpy(b->name,(char **) &((*bNew)->name));CHKERRQ(ierr);
  ierr = PetscStrallocpy(b->lname,(char **) &((*bNew)->lname));CHKERRQ(ierr);
  (*bNew)->type   = b->type;
  (*bNew)->label  = b->label;
  (*bNew)->Nv     = b->Nv;
  ierr = PetscMalloc1(b->Nv, &(*bNew)->values);CHKERRQ(ierr);
  ierr = PetscArraycpy((*bNew)->values, b->values, b->Nv);CHKERRQ(ierr);
  (*bNew)->field  = b->field;
  (*bNew)->Nc     = b->Nc;
  ierr = PetscMalloc1(b->Nc, &(*bNew)->comps);CHKERRQ(ierr);
  ierr = PetscArraycpy((*bNew)->comps, b->comps, b->Nc);CHKERRQ(ierr);
  (*bNew)->func   = b->func;
  (*bNew)->func_t = b->func_t;
  (*bNew)->ctx    = b->ctx;
  PetscFunctionReturn(0);
}

/*@
  PetscDSCopyBoundary - Copy all boundary condition objects to the new problem

  Not collective

  Input Parameters:
+ ds        - The source PetscDS object
. numFields - The number of selected fields, or PETSC_DEFAULT for all fields
- fields    - The selected fields, or NULL for all fields

  Output Parameter:
. newds     - The target PetscDS, now with a copy of the boundary conditions

  Level: intermediate

.seealso: PetscDSCopyEquations(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSCopyBoundary(PetscDS ds, PetscInt numFields, const PetscInt fields[], PetscDS newds)
{
  DSBoundary     b, *lastnext;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,    PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(newds, PETSCDS_CLASSID, 4);
  if (ds == newds) PetscFunctionReturn(0);
  ierr = PetscDSDestroyBoundary(newds);CHKERRQ(ierr);
  lastnext = &(newds->boundary);
  for (b = ds->boundary; b; b = b->next) {
    DSBoundary bNew;
    PetscInt   fieldNew = -1;

    if (numFields > 0 && fields) {
      PetscInt f;

      for (f = 0; f < numFields; ++f) if (b->field == fields[f]) break;
      if (f == numFields) continue;
      fieldNew = f;
    }
    ierr = DSBoundaryDuplicate_Internal(b, &bNew);CHKERRQ(ierr);
    bNew->field = fieldNew < 0 ? b->field : fieldNew;
    *lastnext = bNew;
    lastnext  = &(bNew->next);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSDestroyBoundary - Remove all DMBoundary objects from the PetscDS

  Not collective

  Input Parameter:
. ds - The PetscDS object

  Level: intermediate

.seealso: PetscDSCopyBoundary(), PetscDSCopyEquations()
@*/
PetscErrorCode PetscDSDestroyBoundary(PetscDS ds)
{
  DSBoundary     next = ds->boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (next) {
    DSBoundary b = next;

    next = b->next;
    ierr = PetscWeakFormDestroy(&b->wf);CHKERRQ(ierr);
    ierr = PetscFree(b->name);CHKERRQ(ierr);
    ierr = PetscFree(b->lname);CHKERRQ(ierr);
    ierr = PetscFree(b->values);CHKERRQ(ierr);
    ierr = PetscFree(b->comps);CHKERRQ(ierr);
    ierr = PetscFree(b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSSelectDiscretizations - Copy discretizations to the new problem with different field layout

  Not collective

  Input Parameters:
+ prob - The PetscDS object
. numFields - Number of new fields
- fields - Old field number for each new field

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSSelectEquations(), PetscDSCopyBoundary(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSSelectDiscretizations(PetscDS prob, PetscInt numFields, const PetscInt fields[], PetscDS newprob)
{
  PetscInt       Nf, Nfn, fn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  if (fields) PetscValidPointer(fields, 3);
  PetscValidHeaderSpecific(newprob, PETSCDS_CLASSID, 4);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(newprob, &Nfn);CHKERRQ(ierr);
  numFields = numFields < 0 ? Nf : numFields;
  for (fn = 0; fn < numFields; ++fn) {
    const PetscInt f = fields ? fields[fn] : fn;
    PetscObject    disc;

    if (f >= Nf) continue;
    ierr = PetscDSGetDiscretization(prob, f, &disc);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(newprob, fn, disc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDSSelectEquations - Copy pointwise function pointers to the new problem with different field layout

  Not collective

  Input Parameters:
+ prob - The PetscDS object
. numFields - Number of new fields
- fields - Old field number for each new field

  Output Parameter:
. newprob - The PetscDS copy

  Level: intermediate

.seealso: PetscDSSelectDiscretizations(), PetscDSCopyBoundary(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
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
  PetscCheckFalse(numFields > Nfn,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_SIZ, "Number of fields %D to transfer must not be greater then the total number of fields %D", numFields, Nfn);
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
      ierr = PetscDSSetJacobianPreconditioner(newprob, fn, gn, g0p, g1p, g2p, g3p);CHKERRQ(ierr);
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
  PetscWeakForm  wf, newwf;
  PetscInt       Nf, Ng;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(prob, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(newprob, PETSCDS_CLASSID, 2);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(newprob, &Ng);CHKERRQ(ierr);
  PetscCheckFalse(Nf != Ng,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_SIZ, "Number of fields must match %D != %D", Nf, Ng);
  ierr = PetscDSGetWeakForm(prob, &wf);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(newprob, &newwf);CHKERRQ(ierr);
  ierr = PetscWeakFormCopy(wf, newwf);CHKERRQ(ierr);
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

/*@
  PetscDSCopyExactSolutions - Copy all exact solutions to the new problem

  Not collective

  Input Parameter:
. ds - The PetscDS object

  Output Parameter:
. newds - The PetscDS copy

  Level: intermediate

.seealso: PetscDSCopyBoundary(), PetscDSCopyEquations(), PetscDSSetResidual(), PetscDSSetJacobian(), PetscDSSetRiemannSolver(), PetscDSSetBdResidual(), PetscDSSetBdJacobian(), PetscDSCreate()
@*/
PetscErrorCode PetscDSCopyExactSolutions(PetscDS ds, PetscDS newds)
{
  PetscSimplePointFunc sol;
  void                *ctx;
  PetscInt             Nf, f;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  PetscValidHeaderSpecific(newds, PETSCDS_CLASSID, 2);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscDSGetExactSolution(ds,    f, &sol, &ctx);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(newds, f,  sol,  ctx);CHKERRQ(ierr);
    ierr = PetscDSGetExactSolutionTimeDerivative(ds,    f, &sol, &ctx);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(newds, f,  sol,  ctx);CHKERRQ(ierr);
  }
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
  PetscCheckFalse(height > dim,PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_OUTOFRANGE, "DS can only handle height in [0, %D], not %D", dim, height);
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
      else SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported discretization type for field %d", f);
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
  PetscCheckFalse(f >= Nf,PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_SIZ, "Field %D must be in [0, %D)", f, Nf);
  ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
  if (obj) {
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) *disctype = PETSC_DISC_FE;
    else                       *disctype = PETSC_DISC_FV;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSDestroy_Basic(PetscDS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDSInitialize_Basic(PetscDS ds)
{
  PetscFunctionBegin;
  ds->ops->setfromoptions = NULL;
  ds->ops->setup          = NULL;
  ds->ops->view           = NULL;
  ds->ops->destroy        = PetscDSDestroy_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDSBASIC = "basic" - A discrete system with pointwise residual and boundary residual functions

  Level: intermediate

.seealso: PetscDSType, PetscDSCreate(), PetscDSSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDSCreate_Basic(PetscDS ds)
{
  PetscDS_Basic *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 1);
  ierr = PetscNewLog(ds, &b);CHKERRQ(ierr);
  ds->data = b;

  ierr = PetscDSInitialize_Basic(ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
