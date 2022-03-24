static char help[] = "Tests dual space symmetry.\n\n";

#include <petscfe.h>
#include <petscdmplex.h>

static PetscErrorCode CheckSymmetry(PetscInt dim, PetscInt order, PetscBool tensor)
{
  DM                dm;
  PetscDualSpace    sp;
  PetscInt          nFunc, *ids, *idsCopy, *idsCopy2, i, closureSize, *closure = NULL, offset, depth;
  DMLabel           depthLabel;
  PetscBool         printed = PETSC_FALSE;
  PetscScalar       *vals, *valsCopy, *valsCopy2;
  const PetscInt    *numDofs;
  const PetscInt    ***perms = NULL;
  const PetscScalar ***flips = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscDualSpaceCreate(PETSC_COMM_SELF,&sp));
  CHKERRQ(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, tensor ? PETSC_FALSE : PETSC_TRUE), &dm));
  CHKERRQ(PetscDualSpaceSetType(sp,PETSCDUALSPACELAGRANGE));
  CHKERRQ(PetscDualSpaceSetDM(sp,dm));
  CHKERRQ(PetscDualSpaceSetOrder(sp,order));
  CHKERRQ(PetscDualSpaceLagrangeSetContinuity(sp,PETSC_TRUE));
  CHKERRQ(PetscDualSpaceLagrangeSetTensor(sp,tensor));
  CHKERRQ(PetscDualSpaceSetFromOptions(sp));
  CHKERRQ(PetscDualSpaceSetUp(sp));
  CHKERRQ(PetscDualSpaceGetDimension(sp,&nFunc));
  CHKERRQ(PetscDualSpaceGetSymmetries(sp,&perms,&flips));
  if (!perms && !flips) {
    CHKERRQ(PetscDualSpaceDestroy(&sp));
    CHKERRQ(DMDestroy(&dm));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscMalloc6(nFunc,&ids,nFunc,&idsCopy,nFunc,&idsCopy2,nFunc*dim,&vals,nFunc*dim,&valsCopy,nFunc*dim,&valsCopy2));
  for (i = 0; i < nFunc; i++) ids[i] = idsCopy2[i] = i;
  for (i = 0; i < nFunc; i++) {
    PetscQuadrature q;
    PetscInt        numPoints, Nc, j;
    const PetscReal *points;
    const PetscReal *weights;

    CHKERRQ(PetscDualSpaceGetFunctional(sp,i,&q));
    CHKERRQ(PetscQuadratureGetData(q,NULL,&Nc,&numPoints,&points,&weights));
    PetscCheckFalse(Nc != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support scalar quadrature, not %D components",Nc);
    for (j = 0; j < dim; j++) vals[dim * i + j] = valsCopy2[dim * i + j] = (PetscScalar) points[j];
  }
  CHKERRQ(PetscDualSpaceGetNumDof(sp,&numDofs));
  CHKERRQ(DMPlexGetDepth(dm,&depth));
  CHKERRQ(DMPlexGetTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure));
  CHKERRQ(DMPlexGetDepthLabel(dm,&depthLabel));
  for (i = 0, offset = 0; i < closureSize; i++, offset += numDofs[depth]) {
    PetscInt          point = closure[2 * i], numFaces, j;
    const PetscInt    **pointPerms = perms ? perms[i] : NULL;
    const PetscScalar **pointFlips = flips ? flips[i] : NULL;
    PetscBool         anyPrinted = PETSC_FALSE;

    if (!pointPerms && !pointFlips) continue;
    CHKERRQ(DMLabelGetValue(depthLabel,point,&depth));
    {
      DMPolytopeType ct;
      /* The number of arrangements is no longer based on the number of faces */
      CHKERRQ(DMPlexGetCellType(dm, point, &ct));
      numFaces = DMPolytopeTypeGetNumArrangments(ct) / 2;
    }
    for (j = -numFaces; j < numFaces; j++) {
      PetscInt          k, l;
      const PetscInt    *perm = pointPerms ? pointPerms[j] : NULL;
      const PetscScalar *flip = pointFlips ? pointFlips[j] : NULL;

      for (k = 0; k < numDofs[depth]; k++) {
        PetscInt kLocal = perm ? perm[k] : k;

        idsCopy[kLocal] = ids[offset + k];
        for (l = 0; l < dim; l++) {
          valsCopy[kLocal * dim + l] = vals[(offset + k) * dim + l] * (flip ? flip[kLocal] : 1.);
        }
      }
      if (!printed && numDofs[depth] > 1) {
        IS   is;
        Vec  vec;
        char name[256];

        anyPrinted = PETSC_TRUE;
        CHKERRQ(PetscSNPrintf(name,256,"%DD, %s, Order %D, Point %D Symmetry %D",dim,tensor ? "Tensor" : "Simplex", order, point,j));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,numDofs[depth],idsCopy,PETSC_USE_POINTER,&is));
        CHKERRQ(PetscObjectSetName((PetscObject)is,name));
        CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_SELF));
        CHKERRQ(ISDestroy(&is));
        CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,dim,numDofs[depth]*dim,valsCopy,&vec));
        CHKERRQ(PetscObjectSetName((PetscObject)vec,name));
        CHKERRQ(VecView(vec,PETSC_VIEWER_STDOUT_SELF));
        CHKERRQ(VecDestroy(&vec));
      }
      for (k = 0; k < numDofs[depth]; k++) {
        PetscInt kLocal = perm ? perm[k] : k;

        idsCopy2[offset + k] = idsCopy[kLocal];
        for (l = 0; l < dim; l++) {
          valsCopy2[(offset + k) * dim + l] = valsCopy[kLocal * dim + l] * (flip ? PetscConj(flip[kLocal]) : 1.);
        }
      }
      for (k = 0; k < nFunc; k++) {
        PetscCheckFalse(idsCopy2[k] != ids[k],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Symmetry failure: %DD, %s, point %D, symmetry %D, order %D, functional %D: (%D != %D)",dim, tensor ? "Tensor" : "Simplex",point,j,order,k,ids[k],k);
        for (l = 0; l < dim; l++) {
          PetscCheckFalse(valsCopy2[dim * k + l] != vals[dim * k + l],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Symmetry failure: %DD, %s, point %D, symmetry %D, order %D, functional %D, component %D: (%D != %D)",dim, tensor ? "Tensor" : "Simplex",point,j,order,k,l);
        }
      }
    }
    if (anyPrinted && !printed) printed = PETSC_TRUE;
  }
  CHKERRQ(DMPlexRestoreTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure));
  CHKERRQ(PetscFree6(ids,idsCopy,idsCopy2,vals,valsCopy,valsCopy2));
  CHKERRQ(PetscDualSpaceDestroy(&sp));
  CHKERRQ(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       dim, order, tensor;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  for (tensor = 0; tensor < 2; tensor++) {
    for (dim = 1; dim <= 3; dim++) {
      if (dim == 1 && tensor) continue;
      for (order = 0; order <= (tensor ? 5 : 6); order++) {
        CHKERRQ(CheckSymmetry(dim,order,tensor ? PETSC_TRUE : PETSC_FALSE));
      }
    }
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
TEST*/
