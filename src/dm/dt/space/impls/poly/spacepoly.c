#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"PetscSpace polynomial options");
  PetscCall(PetscOptionsBool("-petscspace_poly_tensor", "Use the tensor product polynomials", "PetscSpacePolynomialSetTensor", poly->tensor, &poly->tensor, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(v, "%s space of degree %D\n", poly->tensor ? "Tensor polynomial" : "Polynomial", sp->degree));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Polynomial(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscSpacePolynomialView_Ascii(sp, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", NULL));
  if (poly->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      PetscCall(PetscSpaceDestroy(&poly->subspaces[d]));
    }
  }
  PetscCall(PetscFree(poly->subspaces));
  PetscCall(PetscFree(poly));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  if (poly->setupCalled) PetscFunctionReturn(0);
  if (sp->Nv <= 1) {
    poly->tensor = PETSC_FALSE;
  }
  if (sp->Nc != 1) {
    PetscInt    Nc = sp->Nc;
    PetscBool   tensor = poly->tensor;
    PetscInt    Nv = sp->Nv;
    PetscInt    degree = sp->degree;
    const char *prefix;
    const char *name;
    char        subname[PETSC_MAX_PATH_LEN];
    PetscSpace  subsp;

    PetscCall(PetscSpaceSetType(sp, PETSCSPACESUM));
    PetscCall(PetscSpaceSumSetNumSubspaces(sp, Nc));
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
    PetscCall(PetscSpaceSetType(subsp, PETSCSPACEPOLYNOMIAL));
    PetscCall(PetscSpaceSetDegree(subsp, degree, PETSC_DETERMINE));
    PetscCall(PetscSpaceSetNumComponents(subsp, 1));
    PetscCall(PetscSpaceSetNumVariables(subsp, Nv));
    PetscCall(PetscSpacePolynomialSetTensor(subsp, tensor));
    PetscCall(PetscSpaceSetUp(subsp));
    for (PetscInt i = 0; i < Nc; i++) {
      PetscCall(PetscSpaceSumSetSubspace(sp, i, subsp));
    }
    PetscCall(PetscSpaceDestroy(&subsp));
    PetscCall(PetscSpaceSetUp(sp));
    PetscFunctionReturn(0);
  }
  if (poly->tensor) {
    sp->maxDegree = PETSC_DETERMINE;
    PetscCall(PetscSpaceSetType(sp, PETSCSPACETENSOR));
    PetscCall(PetscSpaceSetUp(sp));
    PetscFunctionReturn(0);
  }
  PetscCheck(sp->degree >= 0,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Negative degree %D invalid", sp->degree);
  sp->maxDegree = sp->degree;
  poly->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Polynomial(PetscSpace sp, PetscInt *dim)
{
  PetscInt         deg  = sp->degree;
  PetscInt         n    = sp->Nv;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(n + deg, n, dim));
  *dim *= sp->Nc;
  PetscFunctionReturn(0);
}

static PetscErrorCode CoordinateBasis(PetscInt dim, PetscInt npoints, const PetscReal points[], PetscInt jet, PetscInt Njet, PetscReal pScalar[])
{
  PetscFunctionBegin;
  PetscCall(PetscArrayzero(pScalar, (1 + dim) * Njet * npoints));
  for (PetscInt b = 0; b < 1 + dim; b++) {
    for (PetscInt j = 0; j < PetscMin(1 + dim, Njet); j++) {
      if (j == 0) {
        if (b == 0) {
          for (PetscInt pt = 0; pt < npoints; pt++) {
            pScalar[b * Njet * npoints + j * npoints + pt] = 1.;
          }
        } else {
          for (PetscInt pt = 0; pt < npoints; pt++) {
            pScalar[b * Njet * npoints + j * npoints + pt] = points[pt * dim + (b-1)];
          }
        }
      } else if (j == b) {
        for (PetscInt pt = 0; pt < npoints; pt++) {
          pScalar[b * Njet * npoints + j * npoints + pt] = 1.;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         dim     = sp->Nv;
  PetscInt         Nb, jet, Njet;
  PetscReal       *pScalar;

  PetscFunctionBegin;
  if (!poly->setupCalled) {
    PetscCall(PetscSpaceSetUp(sp));
    PetscCall(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
    PetscFunctionReturn(0);
  }
  PetscCheckFalse(poly->tensor || sp->Nc != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "tensor and multicomponent spaces should have been converted");
  PetscCall(PetscDTBinomialInt(dim + sp->degree, dim, &Nb));
  if (H) {
    jet = 2;
  } else if (D) {
    jet = 1;
  } else {
    jet = 0;
  }
  PetscCall(PetscDTBinomialInt(dim + jet, dim, &Njet));
  PetscCall(DMGetWorkArray(dm, Nb * Njet * npoints, MPIU_REAL, &pScalar));
  // Why are we handling the case degree == 1 specially?  Because we don't want numerical noise when we evaluate hat
  // functions at the vertices of a simplex, which happens when we invert the Vandermonde matrix of the PKD basis.
  // We don't make any promise about which basis is used.
  if (sp->degree == 1) {
    PetscCall(CoordinateBasis(dim, npoints, points, jet, Njet, pScalar));
  } else {
    PetscCall(PetscDTPKDEvalJet(dim, npoints, points, sp->degree, jet, pScalar));
  }
  if (B) {
    PetscInt p_strl = Nb;
    PetscInt b_strl = 1;

    PetscInt b_strr = Njet*npoints;
    PetscInt p_strr = 1;

    PetscCall(PetscArrayzero(B, npoints * Nb));
    for (PetscInt b = 0; b < Nb; b++) {
      for (PetscInt p = 0; p < npoints; p++) {
        B[p*p_strl + b*b_strl] = pScalar[b*b_strr + p*p_strr];
      }
    }
  }
  if (D) {
    PetscInt p_strl = dim*Nb;
    PetscInt b_strl = dim;
    PetscInt d_strl = 1;

    PetscInt b_strr = Njet*npoints;
    PetscInt d_strr = npoints;
    PetscInt p_strr = 1;

    PetscCall(PetscArrayzero(D, npoints * Nb * dim));
    for (PetscInt d = 0; d < dim; d++) {
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt p = 0; p < npoints; p++) {
          D[p*p_strl + b*b_strl + d*d_strl] = pScalar[b*b_strr + (1+d)*d_strr + p*p_strr];
        }
      }
    }
  }
  if (H) {
    PetscInt p_strl  = dim*dim*Nb;
    PetscInt b_strl  = dim*dim;
    PetscInt d1_strl = dim;
    PetscInt d2_strl = 1;

    PetscInt b_strr = Njet*npoints;
    PetscInt j_strr = npoints;
    PetscInt p_strr = 1;

    PetscInt *derivs;
    PetscCall(PetscCalloc1(dim, &derivs));
    PetscCall(PetscArrayzero(H, npoints * Nb * dim * dim));
    for (PetscInt d1 = 0; d1 < dim; d1++) {
      for (PetscInt d2 = 0; d2 < dim; d2++) {
        PetscInt j;
        derivs[d1]++;
        derivs[d2]++;
        PetscCall(PetscDTGradedOrderToIndex(dim, derivs, &j));
        derivs[d1]--;
        derivs[d2]--;
        for (PetscInt b = 0; b < Nb; b++) {
          for (PetscInt p = 0; p < npoints; p++) {
            H[p*p_strl + b*b_strl + d1*d1_strl + d2*d2_strl] = pScalar[b*b_strr + j*j_strr + p*p_strr];
          }
        }
      }
    }
    PetscCall(PetscFree(derivs));
  }
  PetscCall(DMRestoreWorkArray(dm, Nb * Njet * npoints, MPIU_REAL, &pScalar));
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialSetTensor - Set whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variable is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
+ sp     - the function space object
- tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Options Database:
. -petscspace_poly_tensor <bool> - Whether to use tensor product polynomials in higher dimension

  Level: intermediate

.seealso: PetscSpacePolynomialGetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace sp, PetscBool tensor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscSpacePolynomialSetTensor_C",(PetscSpace,PetscBool),(sp,tensor));
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

  Level: intermediate

.seealso: PetscSpacePolynomialSetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace sp, PetscBool *tensor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidBoolPointer(tensor, 2);
  PetscTryMethod(sp,"PetscSpacePolynomialGetTensor_C",(PetscSpace,PetscBool*),(sp,tensor));
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
  PetscValidBoolPointer(tensor, 2);
  *tensor = poly->tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Polynomial(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscSpaceGetNumVariables(sp, &dim));
  PetscCall(PetscSpaceGetDegree(sp, &order, NULL));
  PetscCall(PetscSpacePolynomialGetTensor(sp, &tensor));
  PetscCheckFalse(height > dim || height < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!poly->subspaces) PetscCall(PetscCalloc1(dim, &poly->subspaces));
  if (height <= dim) {
    if (!poly->subspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub));
      PetscCall(PetscObjectGetName((PetscObject) sp,  &name));
      PetscCall(PetscObjectSetName((PetscObject) sub,  name));
      PetscCall(PetscSpaceSetType(sub, PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpaceSetNumComponents(sub, Nc));
      PetscCall(PetscSpaceSetDegree(sub, order, PETSC_DETERMINE));
      PetscCall(PetscSpaceSetNumVariables(sub, dim-height));
      PetscCall(PetscSpacePolynomialSetTensor(sub, tensor));
      PetscCall(PetscSpaceSetUp(sub));
      poly->subspaces[height-1] = sub;
    }
    *subsp = poly->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup             = PetscSpaceSetUp_Polynomial;
  sp->ops->view              = PetscSpaceView_Polynomial;
  sp->ops->destroy           = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension      = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate          = PetscSpaceEvaluate_Polynomial;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Polynomial;
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Polynomial));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", PetscSpacePolynomialSetTensor_Polynomial));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscCall(PetscNewLog(sp,&poly));
  sp->data = poly;

  poly->tensor    = PETSC_FALSE;
  poly->subspaces = NULL;

  PetscCall(PetscSpaceInitialize_Polynomial(sp));
  PetscFunctionReturn(0);
}
