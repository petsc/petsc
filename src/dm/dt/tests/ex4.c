static char help[] = "Tests dual space symmetry.\n\n";

#include <petscfe.h>
#include <petscdmplex.h>

static PetscErrorCode CheckSymmetry(PetscInt dim, PetscInt order, PetscBool tensor)
{
  DM                   dm;
  PetscDualSpace       sp;
  PetscInt             nFunc, *ids, *idsCopy, *idsCopy2, i, closureSize, *closure = NULL, offset, depth;
  DMLabel              depthLabel;
  PetscBool            printed = PETSC_FALSE;
  PetscScalar         *vals, *valsCopy, *valsCopy2;
  const PetscInt      *numDofs;
  const PetscInt    ***perms = NULL;
  const PetscScalar ***flips = NULL;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceCreate(PETSC_COMM_SELF, &sp));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, tensor ? PETSC_FALSE : PETSC_TRUE), &dm));
  PetscCall(PetscDualSpaceSetType(sp, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetDM(sp, dm));
  PetscCall(PetscDualSpaceSetOrder(sp, order));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(sp, PETSC_TRUE));
  PetscCall(PetscDualSpaceLagrangeSetTensor(sp, tensor));
  PetscCall(PetscDualSpaceSetFromOptions(sp));
  PetscCall(PetscDualSpaceSetUp(sp));
  PetscCall(PetscDualSpaceGetDimension(sp, &nFunc));
  PetscCall(PetscDualSpaceGetSymmetries(sp, &perms, &flips));
  if (!perms && !flips) {
    PetscCall(PetscDualSpaceDestroy(&sp));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc6(nFunc, &ids, nFunc, &idsCopy, nFunc, &idsCopy2, nFunc * dim, &vals, nFunc * dim, &valsCopy, nFunc * dim, &valsCopy2));
  for (i = 0; i < nFunc; i++) ids[i] = idsCopy2[i] = i;
  for (i = 0; i < nFunc; i++) {
    PetscQuadrature  q;
    PetscInt         numPoints, Nc, j;
    const PetscReal *points;
    const PetscReal *weights;

    PetscCall(PetscDualSpaceGetFunctional(sp, i, &q));
    PetscCall(PetscQuadratureGetData(q, NULL, &Nc, &numPoints, &points, &weights));
    PetscCheck(Nc == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support scalar quadrature, not %" PetscInt_FMT " components", Nc);
    for (j = 0; j < dim; j++) vals[dim * i + j] = valsCopy2[dim * i + j] = (PetscScalar)points[j];
  }
  PetscCall(PetscDualSpaceGetNumDof(sp, &numDofs));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  for (i = 0, offset = 0; i < closureSize; i++, offset += numDofs[depth]) {
    PetscInt            point      = closure[2 * i], numFaces, j;
    const PetscInt    **pointPerms = perms ? perms[i] : NULL;
    const PetscScalar **pointFlips = flips ? flips[i] : NULL;
    PetscBool           anyPrinted = PETSC_FALSE;

    if (!pointPerms && !pointFlips) continue;
    PetscCall(DMLabelGetValue(depthLabel, point, &depth));
    {
      DMPolytopeType ct;
      /* The number of arrangements is no longer based on the number of faces */
      PetscCall(DMPlexGetCellType(dm, point, &ct));
      numFaces = DMPolytopeTypeGetNumArrangments(ct) / 2;
    }
    for (j = -numFaces; j < numFaces; j++) {
      PetscInt           k, l;
      const PetscInt    *perm = pointPerms ? pointPerms[j] : NULL;
      const PetscScalar *flip = pointFlips ? pointFlips[j] : NULL;

      for (k = 0; k < numDofs[depth]; k++) {
        PetscInt kLocal = perm ? perm[k] : k;

        idsCopy[kLocal] = ids[offset + k];
        for (l = 0; l < dim; l++) valsCopy[kLocal * dim + l] = vals[(offset + k) * dim + l] * (flip ? flip[kLocal] : 1.);
      }
      if (!printed && numDofs[depth] > 1) {
        IS   is;
        Vec  vec;
        char name[256];

        anyPrinted = PETSC_TRUE;
        PetscCall(PetscSNPrintf(name, 256, "%" PetscInt_FMT "D, %s, Order %" PetscInt_FMT ", Point %" PetscInt_FMT " Symmetry %" PetscInt_FMT, dim, tensor ? "Tensor" : "Simplex", order, point, j));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numDofs[depth], idsCopy, PETSC_USE_POINTER, &is));
        PetscCall(PetscObjectSetName((PetscObject)is, name));
        PetscCall(ISView(is, PETSC_VIEWER_STDOUT_SELF));
        PetscCall(ISDestroy(&is));
        PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, numDofs[depth] * dim, valsCopy, &vec));
        PetscCall(PetscObjectSetName((PetscObject)vec, name));
        PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_SELF));
        PetscCall(VecDestroy(&vec));
      }
      for (k = 0; k < numDofs[depth]; k++) {
        PetscInt kLocal = perm ? perm[k] : k;

        idsCopy2[offset + k] = idsCopy[kLocal];
        for (l = 0; l < dim; l++) valsCopy2[(offset + k) * dim + l] = valsCopy[kLocal * dim + l] * (flip ? PetscConj(flip[kLocal]) : 1.);
      }
      for (k = 0; k < nFunc; k++) {
        PetscCheck(idsCopy2[k] == ids[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Symmetry failure: %" PetscInt_FMT "D, %s, point %" PetscInt_FMT ", symmetry %" PetscInt_FMT ", order %" PetscInt_FMT ", functional %" PetscInt_FMT ": (%" PetscInt_FMT " != %" PetscInt_FMT ")", dim, tensor ? "Tensor" : "Simplex", point, j, order, k, ids[k], k);
        for (l = 0; l < dim; l++) {
          PetscCheck(valsCopy2[dim * k + l] == vals[dim * k + l], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Symmetry failure: %" PetscInt_FMT "D, %s, point %" PetscInt_FMT ", symmetry %" PetscInt_FMT ", order %" PetscInt_FMT ", functional %" PetscInt_FMT ", component %" PetscInt_FMT ": (%g != %g)", dim, tensor ? "Tensor" : "Simplex", point, j, order, k, l, (double)PetscAbsScalar(valsCopy2[dim * k + l]), (double)PetscAbsScalar(vals[dim * k + l]));
        }
      }
    }
    if (anyPrinted && !printed) printed = PETSC_TRUE;
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure));
  PetscCall(PetscFree6(ids, idsCopy, idsCopy2, vals, valsCopy, valsCopy2));
  PetscCall(PetscDualSpaceDestroy(&sp));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt dim, order, tensor;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  for (tensor = 0; tensor < 2; tensor++) {
    for (dim = 1; dim <= 3; dim++) {
      if (dim == 1 && tensor) continue;
      for (order = 0; order <= (tensor ? 5 : 6); order++) PetscCall(CheckSymmetry(dim, order, tensor ? PETSC_TRUE : PETSC_FALSE));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
TEST*/
