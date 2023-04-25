#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@
  PetscSpaceSumGetNumSubspaces - Get the number of spaces in the sum space

  Input Parameter:
. sp  - the function space object

  Output Parameter:
. numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually getting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumSetNumSubspaces()`, `PetscSpaceSetDegree()`, `PetscSpaceSetNumVariables()`
@*/
PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace sp, PetscInt *numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(numSumSpaces, 2);
  PetscTryMethod(sp, "PetscSpaceSumGetNumSubspaces_C", (PetscSpace, PetscInt *), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSpaceSumSetNumSubspaces - Set the number of spaces in the sum space

  Input Parameters:
+ sp  - the function space object
- numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually setting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumGetNumSubspaces()`, `PetscSpaceSetDegree()`, `PetscSpaceSetNumVariables()`
@*/
PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace sp, PetscInt numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscSpaceSumSetNumSubspaces_C", (PetscSpace, PetscInt), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
 PetscSpaceSumGetConcatenate - Get the concatenate flag for this space.
 A concatenated sum space will have number of components equal to the sum of the number of components of all subspaces. A non-concatenated,
 or direct sum space will have the same number of components as its subspaces.

 Input Parameter:
. sp - the function space object

 Output Parameter:
. concatenate - flag indicating whether subspaces are concatenated.

  Level: intermediate

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumSetConcatenate()`
@*/
PetscErrorCode PetscSpaceSumGetConcatenate(PetscSpace sp, PetscBool *concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscSpaceSumGetConcatenate_C", (PetscSpace, PetscBool *), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSpaceSumSetConcatenate - Sets the concatenate flag for this space.
  A concatenated sum space will have number of components equal to the sum of the number of components of all subspaces. A non-concatenated,
  or direct sum space will have the same number of components as its subspaces .

 Input Parameters:
+ sp - the function space object
- concatenate - are subspaces concatenated components (true) or direct summands (false)

  Level: intermediate

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumGetConcatenate()`
@*/
PetscErrorCode PetscSpaceSumSetConcatenate(PetscSpace sp, PetscBool concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscSpaceSumSetConcatenate_C", (PetscSpace, PetscBool), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSpaceSumGetSubspace - Get a space in the sum space

  Input Parameters:
+ sp - the function space object
- s  - The space number

  Output Parameter:
. subsp - the `PetscSpace`

  Level: intermediate

  Note:
  The name GetSubspace is slightly misleading because it is actually getting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumSetSubspace()`, `PetscSpaceSetDegree()`, `PetscSpaceSetNumVariables()`
@*/
PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace sp, PetscInt s, PetscSpace *subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  PetscTryMethod(sp, "PetscSpaceSumGetSubspace_C", (PetscSpace, PetscInt, PetscSpace *), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSpaceSumSetSubspace - Set a space in the sum space

  Input Parameters:
+ sp    - the function space object
. s     - The space number
- subsp - the number of spaces

  Level: intermediate

  Note:
  The name SetSubspace is slightly misleading because it is actually setting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCSPACESUM`, `PetscSpace`, `PetscSpaceSumGetSubspace()`, `PetscSpaceSetDegree()`, `PetscSpaceSetNumVariables()`
@*/
PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace sp, PetscInt s, PetscSpace subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCSPACE_CLASSID, 3);
  PetscTryMethod(sp, "PetscSpaceSumSetSubspace_C", (PetscSpace, PetscInt, PetscSpace), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumGetNumSubspaces_Sum(PetscSpace space, PetscInt *numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)space->data;

  PetscFunctionBegin;
  *numSumSpaces = sum->numSumSpaces;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumSetNumSubspaces_Sum(PetscSpace space, PetscInt numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)space->data;
  PetscInt        Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change number of subspaces after setup called");
  if (numSumSpaces == Ns) PetscFunctionReturn(PETSC_SUCCESS);
  if (Ns >= 0) {
    PetscInt s;
    for (s = 0; s < Ns; ++s) PetscCall(PetscSpaceDestroy(&sum->sumspaces[s]));
    PetscCall(PetscFree(sum->sumspaces));
  }

  Ns = sum->numSumSpaces = numSumSpaces;
  PetscCall(PetscCalloc1(Ns, &sum->sumspaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumGetConcatenate_Sum(PetscSpace sp, PetscBool *concatenate)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;

  PetscFunctionBegin;
  *concatenate = sum->concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumSetConcatenate_Sum(PetscSpace sp, PetscBool concatenate)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change space concatenation after setup called.");

  sum->concatenate = concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space, PetscInt s, PetscSpace *subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)space->data;
  PetscInt        Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumSetSubspace_Sum(PetscSpace space, PetscInt s, PetscSpace subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)space->data;
  PetscInt        Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change subspace after setup called");
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  PetscCall(PetscObjectReference((PetscObject)subspace));
  PetscCall(PetscSpaceDestroy(&sum->sumspaces[s]));
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscSpace sp, PetscOptionItems *PetscOptionsObject)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;
  PetscInt        Ns, Nc, Nv, deg, i;
  PetscBool       concatenate = PETSC_TRUE;
  const char     *prefix;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetNumVariables(sp, &Nv));
  if (!Nv) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscSpaceSumGetNumSubspaces(sp, &Ns));
  PetscCall(PetscSpaceGetDegree(sp, &deg, NULL));
  Ns = (Ns == PETSC_DEFAULT) ? 1 : Ns;

  PetscOptionsHeadBegin(PetscOptionsObject, "PetscSpace sum options");
  PetscCall(PetscOptionsBoundedInt("-petscspace_sum_spaces", "The number of subspaces", "PetscSpaceSumSetNumSubspaces", Ns, &Ns, NULL, 0));
  PetscCall(PetscOptionsBool("-petscspace_sum_concatenate", "Subspaces are concatenated components of the final space", "PetscSpaceSumSetFromOptions", concatenate, &concatenate, NULL));
  PetscOptionsHeadEnd();

  PetscCheck(Ns >= 0 && (Nv <= 0 || Ns != 0), PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have a sum space of %" PetscInt_FMT " spaces", Ns);
  if (Ns != sum->numSumSpaces) PetscCall(PetscSpaceSumSetNumSubspaces(sp, Ns));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sp, &prefix));
  for (i = 0; i < Ns; ++i) {
    PetscInt   sNv;
    PetscSpace subspace;

    PetscCall(PetscSpaceSumGetSubspace(sp, i, &subspace));
    if (!subspace) {
      char subspacePrefix[256];

      PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &subspace));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)subspace, prefix));
      PetscCall(PetscSNPrintf(subspacePrefix, 256, "sumcomp_%" PetscInt_FMT "_", i));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)subspace, subspacePrefix));
    } else PetscCall(PetscObjectReference((PetscObject)subspace));
    PetscCall(PetscSpaceSetFromOptions(subspace));
    PetscCall(PetscSpaceGetNumVariables(subspace, &sNv));
    PetscCheck(sNv, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace %" PetscInt_FMT " has not been set properly, number of variables is 0.", i);
    PetscCall(PetscSpaceSumSetSubspace(sp, i, subspace));
    PetscCall(PetscSpaceDestroy(&subspace));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSetUp_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum         = (PetscSpace_Sum *)sp->data;
  PetscBool       concatenate = PETSC_TRUE;
  PetscBool       uniform;
  PetscInt        Nv, Ns, Nc, i, sum_Nc = 0, deg = PETSC_MAX_INT, maxDeg = PETSC_MIN_INT;
  PetscInt        minNc, maxNc;

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscSpaceGetNumVariables(sp, &Nv));
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscSpaceSumGetNumSubspaces(sp, &Ns));
  if (Ns == PETSC_DEFAULT) {
    Ns = 1;
    PetscCall(PetscSpaceSumSetNumSubspaces(sp, Ns));
  }
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have %" PetscInt_FMT " subspaces", Ns);
  uniform = PETSC_TRUE;
  if (Ns) {
    PetscSpace s0;

    PetscCall(PetscSpaceSumGetSubspace(sp, 0, &s0));
    for (PetscInt i = 1; i < Ns; i++) {
      PetscSpace si;

      PetscCall(PetscSpaceSumGetSubspace(sp, i, &si));
      if (si != s0) {
        uniform = PETSC_FALSE;
        break;
      }
    }
  }

  minNc = Nc;
  maxNc = Nc;
  for (i = 0; i < Ns; ++i) {
    PetscInt   sNv, sNc, iDeg, iMaxDeg;
    PetscSpace si;

    PetscCall(PetscSpaceSumGetSubspace(sp, i, &si));
    PetscCall(PetscSpaceSetUp(si));
    PetscCall(PetscSpaceGetNumVariables(si, &sNv));
    PetscCheck(sNv == Nv, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace %" PetscInt_FMT " has %" PetscInt_FMT " variables, space has %" PetscInt_FMT ".", i, sNv, Nv);
    PetscCall(PetscSpaceGetNumComponents(si, &sNc));
    if (i == 0 && sNc == Nc) concatenate = PETSC_FALSE;
    minNc = PetscMin(minNc, sNc);
    maxNc = PetscMax(maxNc, sNc);
    sum_Nc += sNc;
    PetscCall(PetscSpaceSumGetSubspace(sp, i, &si));
    PetscCall(PetscSpaceGetDegree(si, &iDeg, &iMaxDeg));
    deg    = PetscMin(deg, iDeg);
    maxDeg = PetscMax(maxDeg, iMaxDeg);
  }

  if (concatenate) PetscCheck(sum_Nc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Total number of subspace components (%" PetscInt_FMT ") does not match number of target space components (%" PetscInt_FMT ").", sum_Nc, Nc);
  else PetscCheck(minNc == Nc && maxNc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Subspaces must have same number of components as the target space.");

  sp->degree       = deg;
  sp->maxDegree    = maxDeg;
  sum->concatenate = concatenate;
  sum->uniform     = uniform;
  sum->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Sum *sum         = (PetscSpace_Sum *)sp->data;
  PetscBool       concatenate = sum->concatenate;
  PetscInt        i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (concatenate) PetscCall(PetscViewerASCIIPrintf(v, "Sum space of %" PetscInt_FMT " concatenated subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  else PetscCall(PetscViewerASCIIPrintf(v, "Sum space of %" PetscInt_FMT " subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  for (i = 0; i < (sum->uniform ? (Ns > 0 ? 1 : 0) : Ns); ++i) {
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscSpaceView(sum->sumspaces[i], v));
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscSpaceSumView_Ascii(sp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;
  PetscInt        i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  for (i = 0; i < Ns; ++i) PetscCall(PetscSpaceDestroy(&sum->sumspaces[i]));
  PetscCall(PetscFree(sum->sumspaces));
  if (sum->heightsubspaces) {
    PetscInt d;

    /* sp->Nv is the spatial dimension, so it is equal to the number
     * of subspaces on higher co-dimension points */
    for (d = 0; d < sp->Nv; ++d) PetscCall(PetscSpaceDestroy(&sum->heightsubspaces[d]));
  }
  PetscCall(PetscFree(sum->heightsubspaces));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetConcatenate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetConcatenate_C", NULL));
  PetscCall(PetscFree(sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;
  PetscInt        i, d = 0, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    PetscCall(PetscSpaceSetUp(sp));
    PetscCall(PetscSpaceGetDimension(sp, dim));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  for (i = 0; i < Ns; ++i) {
    PetscInt id;

    PetscCall(PetscSpaceGetDimension(sum->sumspaces[i], &id));
    d += id;
  }

  *dim = d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Sum *sum         = (PetscSpace_Sum *)sp->data;
  PetscBool       concatenate = sum->concatenate;
  DM              dm          = sp->dm;
  PetscInt        Nc = sp->Nc, Nv = sp->Nv, Ns = sum->numSumSpaces;
  PetscInt        i, s, offset, ncoffset, pdimfull, numelB, numelD, numelH;
  PetscReal      *sB = NULL, *sD = NULL, *sH = NULL;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    PetscCall(PetscSpaceSetUp(sp));
    PetscCall(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSpaceGetDimension(sp, &pdimfull));
  numelB = npoints * pdimfull * Nc;
  numelD = numelB * Nv;
  numelH = numelD * Nv;
  if (B || D || H) PetscCall(DMGetWorkArray(dm, numelB, MPIU_REAL, &sB));
  if (D || H) PetscCall(DMGetWorkArray(dm, numelD, MPIU_REAL, &sD));
  if (H) PetscCall(DMGetWorkArray(dm, numelH, MPIU_REAL, &sH));
  if (B)
    for (i = 0; i < numelB; ++i) B[i] = 0.;
  if (D)
    for (i = 0; i < numelD; ++i) D[i] = 0.;
  if (H)
    for (i = 0; i < numelH; ++i) H[i] = 0.;

  for (s = 0, offset = 0, ncoffset = 0; s < Ns; ++s) {
    PetscInt sNv, spdim, sNc, p;

    PetscCall(PetscSpaceGetNumVariables(sum->sumspaces[s], &sNv));
    PetscCall(PetscSpaceGetNumComponents(sum->sumspaces[s], &sNc));
    PetscCall(PetscSpaceGetDimension(sum->sumspaces[s], &spdim));
    PetscCheck(offset + spdim <= pdimfull, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Subspace dimensions exceed target space dimension.");
    if (s == 0 || !sum->uniform) PetscCall(PetscSpaceEvaluate(sum->sumspaces[s], npoints, points, sB, sD, sH));
    if (B || D || H) {
      for (p = 0; p < npoints; ++p) {
        PetscInt j;

        for (j = 0; j < spdim; ++j) {
          PetscInt c;

          for (c = 0; c < sNc; ++c) {
            PetscInt compoffset, BInd, sBInd;

            compoffset = concatenate ? c + ncoffset : c;
            BInd       = (p * pdimfull + j + offset) * Nc + compoffset;
            sBInd      = (p * spdim + j) * sNc + c;
            if (B) B[BInd] = sB[sBInd];
            if (D || H) {
              PetscInt v;

              for (v = 0; v < Nv; ++v) {
                PetscInt DInd, sDInd;

                DInd  = BInd * Nv + v;
                sDInd = sBInd * Nv + v;
                if (D) D[DInd] = sD[sDInd];
                if (H) {
                  PetscInt v2;

                  for (v2 = 0; v2 < Nv; ++v2) {
                    PetscInt HInd, sHInd;

                    HInd    = DInd * Nv + v2;
                    sHInd   = sDInd * Nv + v2;
                    H[HInd] = sH[sHInd];
                  }
                }
              }
            }
          }
        }
      }
    }
    offset += spdim;
    ncoffset += sNc;
  }

  if (H) PetscCall(DMRestoreWorkArray(dm, numelH, MPIU_REAL, &sH));
  if (D || H) PetscCall(DMRestoreWorkArray(dm, numelD, MPIU_REAL, &sD));
  if (B || D || H) PetscCall(DMRestoreWorkArray(dm, numelB, MPIU_REAL, &sB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Sum(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *)sp->data;
  PetscInt        Nc, dim, order;
  PetscBool       tensor;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscSpaceGetNumVariables(sp, &dim));
  PetscCall(PetscSpaceGetDegree(sp, &order, NULL));
  PetscCall(PetscSpacePolynomialGetTensor(sp, &tensor));
  PetscCheck(height <= dim && height >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %" PetscInt_FMT " for dimension %" PetscInt_FMT " space", height, dim);
  if (!sum->heightsubspaces) PetscCall(PetscCalloc1(dim, &sum->heightsubspaces));
  if (height <= dim) {
    if (!sum->heightsubspaces[height - 1]) {
      PetscSpace  sub;
      const char *name;

      PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)sp), &sub));
      PetscCall(PetscObjectGetName((PetscObject)sp, &name));
      PetscCall(PetscObjectSetName((PetscObject)sub, name));
      PetscCall(PetscSpaceSetType(sub, PETSCSPACESUM));
      PetscCall(PetscSpaceSumSetNumSubspaces(sub, sum->numSumSpaces));
      PetscCall(PetscSpaceSumSetConcatenate(sub, sum->concatenate));
      PetscCall(PetscSpaceSetNumComponents(sub, Nc));
      PetscCall(PetscSpaceSetNumVariables(sub, dim - height));
      for (PetscInt i = 0; i < sum->numSumSpaces; i++) {
        PetscSpace subh;

        PetscCall(PetscSpaceGetHeightSubspace(sum->sumspaces[i], height, &subh));
        PetscCall(PetscSpaceSumSetSubspace(sub, i, subh));
      }
      PetscCall(PetscSpaceSetUp(sub));
      sum->heightsubspaces[height - 1] = sub;
    }
    *subsp = sum->heightsubspaces[height - 1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceInitialize_Sum(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Sum;
  sp->ops->setup             = PetscSpaceSetUp_Sum;
  sp->ops->view              = PetscSpaceView_Sum;
  sp->ops->destroy           = PetscSpaceDestroy_Sum;
  sp->ops->getdimension      = PetscSpaceGetDimension_Sum;
  sp->ops->evaluate          = PetscSpaceEvaluate_Sum;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Sum;

  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetNumSubspaces_C", PetscSpaceSumGetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetNumSubspaces_C", PetscSpaceSumSetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetSubspace_C", PetscSpaceSumGetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetSubspace_C", PetscSpaceSumSetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumGetConcatenate_C", PetscSpaceSumGetConcatenate_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpaceSumSetConcatenate_C", PetscSpaceSumSetConcatenate_Sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCSPACESUM = "sum" - A `PetscSpace` object that encapsulates a sum of subspaces.

  Level: intermediate

  Note:
   That sum can either be direct or a concatenation. For example if A and B are spaces each with 2 components,
  the direct sum of A and B will also have 2 components while the concatenated sum will have 4 components. In both cases A and B must be defined over the
  same number of variables.

.seealso: `PetscSpace`, `PetscSpaceType`, `PetscSpaceCreate()`, `PetscSpaceSetType()`, `PetscSpaceSumGetNumSubspaces()`, `PetscSpaceSumSetNumSubspaces()`,
          `PetscSpaceSumGetConcatenate()`, `PetscSpaceSumSetConcatenate()`
M*/
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscCall(PetscNew(&sum));
  sum->numSumSpaces = PETSC_DEFAULT;
  sp->data          = sum;
  PetscCall(PetscSpaceInitialize_Sum(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces, const PetscSpace subspaces[], PetscBool concatenate, PetscSpace *sumSpace)
{
  PetscInt i, Nv, Nc = 0;

  PetscFunctionBegin;
  if (sumSpace) PetscCall(PetscSpaceDestroy(sumSpace));
  PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]), sumSpace));
  PetscCall(PetscSpaceSetType(*sumSpace, PETSCSPACESUM));
  PetscCall(PetscSpaceSumSetNumSubspaces(*sumSpace, numSubspaces));
  PetscCall(PetscSpaceSumSetConcatenate(*sumSpace, concatenate));
  for (i = 0; i < numSubspaces; ++i) {
    PetscInt sNc;

    PetscCall(PetscSpaceSumSetSubspace(*sumSpace, i, subspaces[i]));
    PetscCall(PetscSpaceGetNumComponents(subspaces[i], &sNc));
    if (concatenate) Nc += sNc;
    else Nc = sNc;
  }
  PetscCall(PetscSpaceGetNumVariables(subspaces[0], &Nv));
  PetscCall(PetscSpaceSetNumComponents(*sumSpace, Nc));
  PetscCall(PetscSpaceSetNumVariables(*sumSpace, Nv));
  PetscCall(PetscSpaceSetUp(*sumSpace));

  PetscFunctionReturn(PETSC_SUCCESS);
}
