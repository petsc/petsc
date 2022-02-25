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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF,&sp);CHKERRQ(ierr);
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, tensor ? PETSC_FALSE : PETSC_TRUE), &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(sp,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(sp,dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(sp,order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(sp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(sp,tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(sp);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(sp);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp,&nFunc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSymmetries(sp,&perms,&flips);CHKERRQ(ierr);
  if (!perms && !flips) {
    ierr = PetscDualSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc6(nFunc,&ids,nFunc,&idsCopy,nFunc,&idsCopy2,nFunc*dim,&vals,nFunc*dim,&valsCopy,nFunc*dim,&valsCopy2);CHKERRQ(ierr);
  for (i = 0; i < nFunc; i++) ids[i] = idsCopy2[i] = i;
  for (i = 0; i < nFunc; i++) {
    PetscQuadrature q;
    PetscInt        numPoints, Nc, j;
    const PetscReal *points;
    const PetscReal *weights;

    ierr = PetscDualSpaceGetFunctional(sp,i,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,&Nc,&numPoints,&points,&weights);CHKERRQ(ierr);
    PetscCheckFalse(Nc != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support scalar quadrature, not %D components",Nc);
    for (j = 0; j < dim; j++) vals[dim * i + j] = valsCopy2[dim * i + j] = (PetscScalar) points[j];
  }
  ierr = PetscDualSpaceGetNumDof(sp,&numDofs);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
  for (i = 0, offset = 0; i < closureSize; i++, offset += numDofs[depth]) {
    PetscInt          point = closure[2 * i], numFaces, j;
    const PetscInt    **pointPerms = perms ? perms[i] : NULL;
    const PetscScalar **pointFlips = flips ? flips[i] : NULL;
    PetscBool         anyPrinted = PETSC_FALSE;

    if (!pointPerms && !pointFlips) continue;
    ierr = DMLabelGetValue(depthLabel,point,&depth);CHKERRQ(ierr);
    {
      DMPolytopeType ct;
      /* The number of arrangements is no longer based on the number of faces */
      ierr = DMPlexGetCellType(dm, point, &ct);CHKERRQ(ierr);
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
        ierr = PetscSNPrintf(name,256,"%DD, %s, Order %D, Point %D Symmetry %D",dim,tensor ? "Tensor" : "Simplex", order, point,j);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,numDofs[depth],idsCopy,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)is,name);CHKERRQ(ierr);
        ierr = ISView(is,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dim,numDofs[depth]*dim,valsCopy,&vec);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)vec,name);CHKERRQ(ierr);
        ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&vec);CHKERRQ(ierr);
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
  ierr = DMPlexRestoreTransitiveClosure(dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  ierr = PetscFree6(ids,idsCopy,idsCopy2,vals,valsCopy,valsCopy2);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&sp);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       dim, order, tensor;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  for (tensor = 0; tensor < 2; tensor++) {
    for (dim = 1; dim <= 3; dim++) {
      if (dim == 1 && tensor) continue;
      for (order = 0; order <= (tensor ? 5 : 6); order++) {
        ierr = CheckSymmetry(dim,order,tensor ? PETSC_TRUE : PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
TEST*/
