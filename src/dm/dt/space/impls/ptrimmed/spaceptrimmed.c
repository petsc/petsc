#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Ptrimmed(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_ptrimmed_form_degree", "form degree of trimmed space", "PetscSpacePTrimmedSetFormDegree", pt->formDegree, &(pt->formDegree), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             f, tdegree;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  f = pt->formDegree;
  tdegree = f == 0 ? sp->degree : sp->degree + 1;
  ierr = PetscViewerASCIIPrintf(v, "Trimmed polynomials %D%s-forms of degree %D (P-%D/\\%D)\n", PetscAbsInt(f), f < 0 ? "*" : "", sp->degree, tdegree, PetscAbsInt(f));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Ptrimmed(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePTrimmedView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", NULL);CHKERRQ(ierr);
  if (pt->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&pt->subspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(pt->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(pt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             Nf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (pt->setupCalled) PetscFunctionReturn(0);
  if (pt->formDegree < -sp->Nv || pt->formDegree > sp->Nv) SETERRQ3(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Form degree %D not in valid range [%D,%D]", pt->formDegree, sp->Nv, sp->Nv);
  ierr = PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
  if (sp->Nc == PETSC_DETERMINE) {
    sp->Nc = Nf;
  }
  if (sp->Nc % Nf) SETERRQ2(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "Number of components %D is not a multiple of form dimension %D", sp->Nc, Nf);
  if (sp->Nc != Nf) {
    PetscSpace  subsp;
    PetscInt    nCopies = sp->Nc / Nf;
    PetscInt    Nv, deg, maxDeg;
    PetscInt    formDegree = pt->formDegree;
    const char *prefix;
    const char *name;
    char        subname[PETSC_MAX_PATH_LEN];

    ierr = PetscSpaceSetType(sp, PETSCSPACESUM);CHKERRQ(ierr);
    ierr = PetscSpaceSumSetConcatenate(sp, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSpaceSumSetNumSubspaces(sp, nCopies);CHKERRQ(ierr);
    ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &subsp);CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject)sp, &prefix);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)subsp, prefix);CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)subsp, "sumcomp_");CHKERRQ(ierr);
    if (((PetscObject)sp)->name) {
      ierr = PetscObjectGetName((PetscObject)sp, &name);CHKERRQ(ierr);
      ierr = PetscSNPrintf(subname, PETSC_MAX_PATH_LEN-1, "%s sum component", name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)subsp, subname);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectSetName((PetscObject)subsp, "sum component");CHKERRQ(ierr);
    }
    ierr = PetscSpaceSetType(subsp, PETSCSPACEPTRIMMED);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumVariables(subsp, Nv);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(subsp, Nf);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(sp, &deg, &maxDeg);CHKERRQ(ierr);
    ierr = PetscSpaceSetDegree(subsp, deg, maxDeg);CHKERRQ(ierr);
    ierr = PetscSpacePTrimmedSetFormDegree(subsp, formDegree);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(subsp);CHKERRQ(ierr);
    for (PetscInt i = 0; i < nCopies; i++) {
      ierr = PetscSpaceSumSetSubspace(sp, i, subsp);CHKERRQ(ierr);
    }
    ierr = PetscSpaceDestroy(&subsp);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (sp->degree == PETSC_DEFAULT) {
    sp->degree = 0;
  } else if (sp->degree < 0) {
    SETERRQ1(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Invalid negative degree %D", sp->degree);
  }
  sp->maxDegree = (pt->formDegree == 0 || PetscAbsInt(pt->formDegree) == sp->Nv) ? sp->degree : sp->degree + 1;
  if (pt->formDegree == 0 || PetscAbsInt(pt->formDegree) == sp->Nv) {
    // Convert to regular polynomial space
    ierr = PetscSpaceSetType(sp, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  pt->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Ptrimmed(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             f;
  PetscInt             Nf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  f = pt->formDegree;
  // For PetscSpace, degree refers to the largest complete polynomial degree contained in the space which
  // is equal to the index of a P trimmed space only for 0-forms: otherwise, the index is degree + 1
  ierr = PetscDTPTrimmedSize(sp->Nv, f == 0 ? sp->degree : sp->degree + 1, pt->formDegree, dim);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
  *dim *= (sp->Nc / Nf);
  PetscFunctionReturn(0);
}

/*
  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]
*/
static PetscErrorCode PetscSpaceEvaluate_Ptrimmed(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Ptrimmed *pt    = (PetscSpace_Ptrimmed *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         jet, degree, Nf, Ncopies, Njet;
  PetscInt         Nc      = sp->Nc;
  PetscInt         f;
  PetscInt         dim     = sp->Nv;
  PetscReal       *eval;
  PetscInt         Nb;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!pt->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(sp, npoints, points, B, D, H);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (H) {
    jet = 2;
  } else if (D) {
    jet = 1;
  } else {
    jet = 0;
  }
  f = pt->formDegree;
  degree = f == 0 ? sp->degree : sp->degree + 1;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(f), &Nf);CHKERRQ(ierr);
  Ncopies = Nc / Nf;
  if (Ncopies != 1) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_PLIB, "Multicopy spaces should have been converted to PETSCSPACESUM");
  ierr = PetscDTBinomialInt(dim + jet, dim, &Njet);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedSize(dim, degree, f, &Nb);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedEvalJet(dim, npoints, points, degree, f, jet, eval);CHKERRQ(ierr);
  if (B) {
    PetscInt p_strl = Nf*Nb;
    PetscInt b_strl = Nf;
    PetscInt v_strl = 1;

    PetscInt b_strr = Nf*Njet*npoints;
    PetscInt v_strr = Njet*npoints;
    PetscInt p_strr = 1;

    for (PetscInt v = 0; v < Nf; v++) {
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt p = 0; p < npoints; p++) {
          B[p*p_strl + b*b_strl + v*v_strl] = eval[b*b_strr + v*v_strr + p*p_strr];
        }
      }
    }
  }
  if (D) {
    PetscInt p_strl = dim*Nf*Nb;
    PetscInt b_strl = dim*Nf;
    PetscInt v_strl = dim;
    PetscInt d_strl = 1;

    PetscInt b_strr = Nf*Njet*npoints;
    PetscInt v_strr = Njet*npoints;
    PetscInt d_strr = npoints;
    PetscInt p_strr = 1;

    for (PetscInt v = 0; v < Nf; v++) {
      for (PetscInt d = 0; d < dim; d++) {
        for (PetscInt b = 0; b < Nb; b++) {
          for (PetscInt p = 0; p < npoints; p++) {
            D[p*p_strl + b*b_strl + v*v_strl + d*d_strl] = eval[b*b_strr + v*v_strr + (1+d)*d_strr + p*p_strr];
          }
        }
      }
    }
  }
  if (H) {
    PetscInt p_strl  = dim*dim*Nf*Nb;
    PetscInt b_strl  = dim*dim*Nf;
    PetscInt v_strl  = dim*dim;
    PetscInt d1_strl = dim;
    PetscInt d2_strl = 1;

    PetscInt b_strr = Nf*Njet*npoints;
    PetscInt v_strr = Njet*npoints;
    PetscInt j_strr = npoints;
    PetscInt p_strr = 1;

    PetscInt *derivs;
    ierr = PetscCalloc1(dim, &derivs);CHKERRQ(ierr);
    for (PetscInt d1 = 0; d1 < dim; d1++) {
      for (PetscInt d2 = 0; d2 < dim; d2++) {
        PetscInt j;
        derivs[d1]++;
        derivs[d2]++;
        ierr = PetscDTGradedOrderToIndex(dim, derivs, &j);CHKERRQ(ierr);
        derivs[d1]--;
        derivs[d2]--;
        for (PetscInt v = 0; v < Nf; v++) {
          for (PetscInt b = 0; b < Nb; b++) {
            for (PetscInt p = 0; p < npoints; p++) {
              H[p*p_strl + b*b_strl + v*v_strl + d1*d1_strl + d2*d2_strl] = eval[b*b_strr + v*v_strr + j*j_strr + p*p_strr];
            }
          }
        }
      }
    }
    ierr = PetscFree(derivs);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePTrimmedSetFormDegree - Set the form degree of the trimmed polynomials.

  Input Parameters:
+ sp         - the function space object
- formDegree - the form degree

  Options Database:
. -petscspace_ptrimmed_form_degree <int> - The trimmed polynomial form degree

  Level: intermediate

.seealso: PetscDTAltV, PetscDTPTrimmedEvalJet(), PetscSpacePTrimmedGetFormDegree()
@*/
PetscErrorCode PetscSpacePTrimmedSetFormDegree(PetscSpace sp, PetscInt formDegree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpacePTrimmedSetFormDegree_C",(PetscSpace,PetscInt),(sp,formDegree));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePTrimmedGetFormDegree - Get the form degree of the trimmed polynomials.

  Input Parameters:
. sp     - the function space object

  Output Parameters:
. formDegee - the form degree

  Level: intermediate

.seealso: PetscDTAltV, PetscDTPTrimmedEvalJet(), PetscSpacePTrimmedSetFormDegree()
@*/
PetscErrorCode PetscSpacePTrimmedGetFormDegree(PetscSpace sp, PetscInt *formDegree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(formDegree, 2);
  ierr = PetscTryMethod(sp,"PetscSpacePTrimmedGetFormDegree_C",(PetscSpace,PetscInt*),(sp,formDegree));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedSetFormDegree_Ptrimmed(PetscSpace sp, PetscInt formDegree)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  pt->formDegree = formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedGetFormDegree_Ptrimmed(PetscSpace sp, PetscInt *formDegree)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(formDegree, 2);
  *formDegree = pt->formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Ptrimmed(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt         dim;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!pt->subspaces) {ierr = PetscCalloc1(dim, &(pt->subspaces));CHKERRQ(ierr);}
  if ((dim - height) <= PetscAbsInt(pt->formDegree)) {
    if (!pt->subspaces[height-1]) {
      PetscInt Nc, degree, Nf, Ncopies, Nfsub;
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
      ierr = PetscDTBinomialInt(dim, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
      ierr = PetscDTBinomialInt((dim-height), PetscAbsInt(pt->formDegree), &Nfsub);CHKERRQ(ierr);
      Ncopies = Nf / Nc;
      ierr = PetscSpaceGetDegree(sp, &degree, NULL);CHKERRQ(ierr);

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACEPTRIMMED);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nfsub * Ncopies);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(sub, degree, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpacePTrimmedSetFormDegree(sub, pt->formDegree);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      pt->subspaces[height-1] = sub;
    }
    *subsp = pt->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Ptrimmed(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", PetscSpacePTrimmedGetFormDegree_Ptrimmed);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", PetscSpacePTrimmedSetFormDegree_Ptrimmed);CHKERRQ(ierr);
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Ptrimmed;
  sp->ops->setup             = PetscSpaceSetUp_Ptrimmed;
  sp->ops->view              = PetscSpaceView_Ptrimmed;
  sp->ops->destroy           = PetscSpaceDestroy_Ptrimmed;
  sp->ops->getdimension      = PetscSpaceGetDimension_Ptrimmed;
  sp->ops->evaluate          = PetscSpaceEvaluate_Ptrimmed;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Ptrimmed;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPTRIMMED = "ptrimmed" - A PetscSpace object that encapsulates a trimmed polynomial space.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType(), PetscDTPTrimmedEvalJet()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&pt);CHKERRQ(ierr);
  sp->data = pt;

  pt->subspaces = NULL;
  sp->Nc        = PETSC_DETERMINE;

  ierr = PetscSpaceInitialize_Ptrimmed(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

