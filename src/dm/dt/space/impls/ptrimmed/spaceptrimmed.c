#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Ptrimmed(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options"));
  PetscCall(PetscOptionsInt("-petscspace_ptrimmed_form_degree", "form degree of trimmed space", "PetscSpacePTrimmedSetFormDegree", pt->formDegree, &(pt->formDegree), NULL));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             f, tdegree;

  PetscFunctionBegin;
  f = pt->formDegree;
  tdegree = f == 0 ? sp->degree : sp->degree + 1;
  PetscCall(PetscViewerASCIIPrintf(v, "Trimmed polynomials %D%s-forms of degree %D (P-%D/\\%D)\n", PetscAbsInt(f), f < 0 ? "*" : "", sp->degree, tdegree, PetscAbsInt(f)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Ptrimmed(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscSpacePTrimmedView_Ascii(sp, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", NULL));
  if (pt->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      PetscCall(PetscSpaceDestroy(&pt->subspaces[d]));
    }
  }
  PetscCall(PetscFree(pt->subspaces));
  PetscCall(PetscFree(pt));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             Nf;

  PetscFunctionBegin;
  if (pt->setupCalled) PetscFunctionReturn(0);
  PetscCheckFalse(pt->formDegree < -sp->Nv || pt->formDegree > sp->Nv,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Form degree %D not in valid range [%D,%D]", pt->formDegree, sp->Nv, sp->Nv);
  PetscCall(PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf));
  if (sp->Nc == PETSC_DETERMINE) {
    sp->Nc = Nf;
  }
  PetscCheckFalse(sp->Nc % Nf,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "Number of components %D is not a multiple of form dimension %D", sp->Nc, Nf);
  if (sp->Nc != Nf) {
    PetscSpace  subsp;
    PetscInt    nCopies = sp->Nc / Nf;
    PetscInt    Nv, deg, maxDeg;
    PetscInt    formDegree = pt->formDegree;
    const char *prefix;
    const char *name;
    char        subname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSpaceSetType(sp, PETSCSPACESUM));
    PetscCall(PetscSpaceSumSetConcatenate(sp, PETSC_TRUE));
    PetscCall(PetscSpaceSumSetNumSubspaces(sp, nCopies));
    PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &subsp));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sp, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)subsp, prefix));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)subsp, "sumcomp_"));
    if (((PetscObject)sp)->name) {
      PetscCall(PetscObjectGetName((PetscObject)sp, &name));
      PetscCall(PetscSNPrintf(subname, PETSC_MAX_PATH_LEN-1, "%s sum component", name));
      PetscCall(PetscObjectSetName((PetscObject)subsp, subname));
    } else {
      PetscCall(PetscObjectSetName((PetscObject)subsp, "sum component"));
    }
    PetscCall(PetscSpaceSetType(subsp, PETSCSPACEPTRIMMED));
    PetscCall(PetscSpaceGetNumVariables(sp, &Nv));
    PetscCall(PetscSpaceSetNumVariables(subsp, Nv));
    PetscCall(PetscSpaceSetNumComponents(subsp, Nf));
    PetscCall(PetscSpaceGetDegree(sp, &deg, &maxDeg));
    PetscCall(PetscSpaceSetDegree(subsp, deg, maxDeg));
    PetscCall(PetscSpacePTrimmedSetFormDegree(subsp, formDegree));
    PetscCall(PetscSpaceSetUp(subsp));
    for (PetscInt i = 0; i < nCopies; i++) {
      PetscCall(PetscSpaceSumSetSubspace(sp, i, subsp));
    }
    PetscCall(PetscSpaceDestroy(&subsp));
    PetscCall(PetscSpaceSetUp(sp));
    PetscFunctionReturn(0);
  }
  if (sp->degree == PETSC_DEFAULT) {
    sp->degree = 0;
  } else if (sp->degree < 0) {
    SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Invalid negative degree %D", sp->degree);
  }
  sp->maxDegree = (pt->formDegree == 0 || PetscAbsInt(pt->formDegree) == sp->Nv) ? sp->degree : sp->degree + 1;
  if (pt->formDegree == 0 || PetscAbsInt(pt->formDegree) == sp->Nv) {
    // Convert to regular polynomial space
    PetscCall(PetscSpaceSetType(sp, PETSCSPACEPOLYNOMIAL));
    PetscCall(PetscSpaceSetUp(sp));
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

  PetscFunctionBegin;
  f = pt->formDegree;
  // For PetscSpace, degree refers to the largest complete polynomial degree contained in the space which
  // is equal to the index of a P trimmed space only for 0-forms: otherwise, the index is degree + 1
  PetscCall(PetscDTPTrimmedSize(sp->Nv, f == 0 ? sp->degree : sp->degree + 1, pt->formDegree, dim));
  PetscCall(PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf));
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

  PetscFunctionBegin;
  if (!pt->setupCalled) {
    PetscCall(PetscSpaceSetUp(sp));
    PetscCall(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
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
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(f), &Nf));
  Ncopies = Nc / Nf;
  PetscCheckFalse(Ncopies != 1,PetscObjectComm((PetscObject) sp), PETSC_ERR_PLIB, "Multicopy spaces should have been converted to PETSCSPACESUM");
  PetscCall(PetscDTBinomialInt(dim + jet, dim, &Njet));
  PetscCall(PetscDTPTrimmedSize(dim, degree, f, &Nb));
  PetscCall(DMGetWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval));
  PetscCall(PetscDTPTrimmedEvalJet(dim, npoints, points, degree, f, jet, eval));
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
    PetscCall(PetscCalloc1(dim, &derivs));
    for (PetscInt d1 = 0; d1 < dim; d1++) {
      for (PetscInt d2 = 0; d2 < dim; d2++) {
        PetscInt j;
        derivs[d1]++;
        derivs[d2]++;
        PetscCall(PetscDTGradedOrderToIndex(dim, derivs, &j));
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
    PetscCall(PetscFree(derivs));
  }
  PetscCall(DMRestoreWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscSpacePTrimmedSetFormDegree_C",(PetscSpace,PetscInt),(sp,formDegree));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(formDegree, 2);
  PetscTryMethod(sp,"PetscSpacePTrimmedGetFormDegree_C",(PetscSpace,PetscInt*),(sp,formDegree));
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
  PetscValidIntPointer(formDegree, 2);
  *formDegree = pt->formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Ptrimmed(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt         dim;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetNumVariables(sp, &dim));
  PetscCheckFalse(height > dim || height < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!pt->subspaces) PetscCall(PetscCalloc1(dim, &(pt->subspaces)));
  if ((dim - height) <= PetscAbsInt(pt->formDegree)) {
    if (!pt->subspaces[height-1]) {
      PetscInt Nc, degree, Nf, Ncopies, Nfsub;
      PetscSpace  sub;
      const char *name;

      PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
      PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(pt->formDegree), &Nf));
      PetscCall(PetscDTBinomialInt((dim-height), PetscAbsInt(pt->formDegree), &Nfsub));
      Ncopies = Nf / Nc;
      PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));

      PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub));
      PetscCall(PetscObjectGetName((PetscObject) sp,  &name));
      PetscCall(PetscObjectSetName((PetscObject) sub,  name));
      PetscCall(PetscSpaceSetType(sub, PETSCSPACEPTRIMMED));
      PetscCall(PetscSpaceSetNumComponents(sub, Nfsub * Ncopies));
      PetscCall(PetscSpaceSetDegree(sub, degree, PETSC_DETERMINE));
      PetscCall(PetscSpaceSetNumVariables(sub, dim-height));
      PetscCall(PetscSpacePTrimmedSetFormDegree(sub, pt->formDegree));
      PetscCall(PetscSpaceSetUp(sub));
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
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", PetscSpacePTrimmedGetFormDegree_Ptrimmed));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", PetscSpacePTrimmedSetFormDegree_Ptrimmed));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscCall(PetscNewLog(sp,&pt));
  sp->data = pt;

  pt->subspaces = NULL;
  sp->Nc        = PETSC_DETERMINE;

  PetscCall(PetscSpaceInitialize_Ptrimmed(sp));
  PetscFunctionReturn(0);
}
