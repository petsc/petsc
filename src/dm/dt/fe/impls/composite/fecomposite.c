#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h> /*I "petscdt.h" I*/
#include <petscblaslapack.h>
#include <petscdmplextransform.h>

static PetscErrorCode PetscFEDestroy_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(cmp->embedding));
  PetscCall(PetscFree(cmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESetUp_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  DM                 K;
  DMPolytopeType     ct;
  DMPlexTransform    tr;
  PetscReal         *subpoint;
  PetscBLASInt      *pivots;
  PetscBLASInt       n, info;
  PetscScalar       *work, *invVscalar;
  PetscInt           dim, pdim, spdim, j, s;
  PetscSection       section;

  PetscFunctionBegin;
  /* Get affine mapping from reference cell to each subcell */
  PetscCall(PetscDualSpaceGetDM(fem->dualSpace, &K));
  PetscCall(DMGetDimension(K, &dim));
  PetscCall(DMPlexGetCellType(K, 0, &ct));
  PetscCall(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
  PetscCall(DMPlexRefineRegularGetAffineTransforms(tr, ct, &cmp->numSubelements, &cmp->v0, &cmp->jac, &cmp->invjac));
  PetscCall(DMPlexTransformDestroy(&tr));
  /* Determine dof embedding into subelements */
  PetscCall(PetscDualSpaceGetDimension(fem->dualSpace, &pdim));
  PetscCall(PetscSpaceGetDimension(fem->basisSpace, &spdim));
  PetscCall(PetscMalloc1(cmp->numSubelements*spdim,&cmp->embedding));
  PetscCall(DMGetWorkArray(K, dim, MPIU_REAL, &subpoint));
  PetscCall(PetscDualSpaceGetSection(fem->dualSpace, &section));
  for (s = 0; s < cmp->numSubelements; ++s) {
    PetscInt sd = 0;
    PetscInt closureSize;
    PetscInt *closure = NULL;

    PetscCall(DMPlexGetTransitiveClosure(K, s, PETSC_TRUE, &closureSize, &closure));
    for (j = 0; j < closureSize; j++) {
      PetscInt point = closure[2*j];
      PetscInt dof, off, k;

      PetscCall(PetscSectionGetDof(section, point, &dof));
      PetscCall(PetscSectionGetOffset(section, point, &off));
      for (k = 0; k < dof; k++) cmp->embedding[s*spdim+sd++] = off + k;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(K, s, PETSC_TRUE, &closureSize, &closure));
    PetscCheckFalse(sd != spdim,PetscObjectComm((PetscObject) fem), PETSC_ERR_PLIB, "Subelement %d has %d dual basis vectors != %d", s, sd, spdim);
  }
  PetscCall(DMRestoreWorkArray(K, dim, MPIU_REAL, &subpoint));
  /* Construct the change of basis from prime basis to nodal basis for each subelement */
  PetscCall(PetscMalloc1(cmp->numSubelements*spdim*spdim,&fem->invV));
  PetscCall(PetscMalloc2(spdim,&pivots,spdim,&work));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(cmp->numSubelements*spdim*spdim,&invVscalar));
#else
  invVscalar = fem->invV;
#endif
  for (s = 0; s < cmp->numSubelements; ++s) {
    for (j = 0; j < spdim; ++j) {
      PetscReal       *Bf;
      PetscQuadrature  f;
      const PetscReal *points, *weights;
      PetscInt         Nc, Nq, q, k;

      PetscCall(PetscDualSpaceGetFunctional(fem->dualSpace, cmp->embedding[s*spdim+j], &f));
      PetscCall(PetscQuadratureGetData(f, NULL, &Nc, &Nq, &points, &weights));
      PetscCall(PetscMalloc1(f->numPoints*spdim*Nc,&Bf));
      PetscCall(PetscSpaceEvaluate(fem->basisSpace, Nq, points, Bf, NULL, NULL));
      for (k = 0; k < spdim; ++k) {
        /* n_j \cdot \phi_k */
        invVscalar[(s*spdim + j)*spdim+k] = 0.0;
        for (q = 0; q < Nq; ++q) {
          invVscalar[(s*spdim + j)*spdim+k] += Bf[q*spdim+k]*weights[q];
        }
      }
      PetscCall(PetscFree(Bf));
    }
    n = spdim;
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, &invVscalar[s*spdim*spdim], &n, pivots, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, &invVscalar[s*spdim*spdim], &n, pivots, work, &n, &info));
  }
#if defined(PETSC_USE_COMPLEX)
  for (s = 0; s <cmp->numSubelements*spdim*spdim; s++) fem->invV[s] = PetscRealPart(invVscalar[s]);
  PetscCall(PetscFree(invVscalar));
#endif
  PetscCall(PetscFree2(pivots,work));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFECreateTabulation_Composite(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  DM                 dm;
  DMPolytopeType     ct;
  PetscInt           pdim;  /* Dimension of FE space P */
  PetscInt           spdim; /* Dimension of subelement FE space P */
  PetscInt           dim;   /* Spatial dimension */
  PetscInt           comp;  /* Field components */
  PetscInt          *subpoints;
  PetscReal         *B = K >= 0 ? T->T[0] : NULL;
  PetscReal         *D = K >= 1 ? T->T[1] : NULL;
  PetscReal         *H = K >= 2 ? T->T[2] : NULL;
  PetscReal         *tmpB = NULL, *tmpD = NULL, *tmpH = NULL, *subpoint;
  PetscInt           p, s, d, e, j, k;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(fem->dualSpace, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetCellType(dm, 0, &ct));
  PetscCall(PetscSpaceGetDimension(fem->basisSpace, &spdim));
  PetscCall(PetscDualSpaceGetDimension(fem->dualSpace, &pdim));
  PetscCall(PetscFEGetNumComponents(fem, &comp));
  /* Divide points into subelements */
  PetscCall(DMGetWorkArray(dm, npoints, MPIU_INT, &subpoints));
  PetscCall(DMGetWorkArray(dm, dim, MPIU_REAL, &subpoint));
  for (p = 0; p < npoints; ++p) {
    for (s = 0; s < cmp->numSubelements; ++s) {
      PetscBool inside;

      /* Apply transform, and check that point is inside cell */
      for (d = 0; d < dim; ++d) {
        subpoint[d] = -1.0;
        for (e = 0; e < dim; ++e) subpoint[d] += cmp->invjac[(s*dim + d)*dim+e]*(points[p*dim+e] - cmp->v0[s*dim+e]);
      }
      PetscCall(DMPolytopeInCellTest(ct, subpoint, &inside));
      if (inside) {subpoints[p] = s; break;}
    }
    PetscCheckFalse(s >= cmp->numSubelements,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d was not found in any subelement", p);
  }
  PetscCall(DMRestoreWorkArray(dm, dim, MPIU_REAL, &subpoint));
  /* Evaluate the prime basis functions at all points */
  if (K >= 0) PetscCall(DMGetWorkArray(dm, npoints*spdim, MPIU_REAL, &tmpB));
  if (K >= 1) PetscCall(DMGetWorkArray(dm, npoints*spdim*dim, MPIU_REAL, &tmpD));
  if (K >= 2) PetscCall(DMGetWorkArray(dm, npoints*spdim*dim*dim, MPIU_REAL, &tmpH));
  PetscCall(PetscSpaceEvaluate(fem->basisSpace, npoints, points, tmpB, tmpD, tmpH));
  /* Translate to the nodal basis */
  if (K >= 0) PetscCall(PetscArrayzero(B, npoints*pdim*comp));
  if (K >= 1) PetscCall(PetscArrayzero(D, npoints*pdim*comp*dim));
  if (K >= 2) PetscCall(PetscArrayzero(H, npoints*pdim*comp*dim*dim));
  for (p = 0; p < npoints; ++p) {
    const PetscInt s = subpoints[p];

    if (B) {
      /* Multiply by V^{-1} (spdim x spdim) */
      for (j = 0; j < spdim; ++j) {
        const PetscInt i = (p*pdim + cmp->embedding[s*spdim+j])*comp;

        B[i] = 0.0;
        for (k = 0; k < spdim; ++k) {
          B[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpB[p*spdim + k];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (spdim x spdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim + d;

          D[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            D[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpD[(p*spdim + k)*dim + d];
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < spdim; ++j) {
        for (d = 0; d < dim*dim; ++d) {
          const PetscInt i = ((p*pdim + cmp->embedding[s*spdim+j])*comp + 0)*dim*dim + d;

          H[i] = 0.0;
          for (k = 0; k < spdim; ++k) {
            H[i] += fem->invV[(s*spdim + k)*spdim+j] * tmpH[(p*spdim + k)*dim*dim + d];
          }
        }
      }
    }
  }
  PetscCall(DMRestoreWorkArray(dm, npoints, MPIU_INT, &subpoints));
  if (K >= 0) PetscCall(DMRestoreWorkArray(dm, npoints*spdim, MPIU_REAL, &tmpB));
  if (K >= 1) PetscCall(DMRestoreWorkArray(dm, npoints*spdim*dim, MPIU_REAL, &tmpD));
  if (K >= 2) PetscCall(DMRestoreWorkArray(dm, npoints*spdim*dim*dim, MPIU_REAL, &tmpH));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEInitialize_Composite(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Composite;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_Composite;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->createtabulation        = PetscFECreateTabulation_Composite;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFECOMPOSITE = "composite" - A PetscFE object that represents a composite element

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/
PETSC_EXTERN PetscErrorCode PetscFECreate_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscCall(PetscNewLog(fem, &cmp));
  fem->data = cmp;

  cmp->numSubelements = -1;
  cmp->v0             = NULL;
  cmp->jac            = NULL;

  PetscCall(PetscFEInitialize_Composite(fem));
  PetscFunctionReturn(0);
}

/*@C
  PetscFECompositeGetMapping - Returns the mappings from the reference element to each subelement

  Not collective

  Input Parameter:
. fem - The PetscFE object

  Output Parameters:
+ blockSize - The number of elements in a block
. numBlocks - The number of blocks in a batch
. batchSize - The number of elements in a batch
- numBatches - The number of batches in a chunk

  Level: intermediate

.seealso: PetscFECreate()
@*/
PetscErrorCode PetscFECompositeGetMapping(PetscFE fem, PetscInt *numSubelements, const PetscReal *v0[], const PetscReal *jac[], const PetscReal *invjac[])
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (numSubelements) {PetscValidIntPointer(numSubelements, 2); *numSubelements = cmp->numSubelements;}
  if (v0)             {PetscValidPointer(v0, 3);             *v0             = cmp->v0;}
  if (jac)            {PetscValidPointer(jac, 4);            *jac            = cmp->jac;}
  if (invjac)         {PetscValidPointer(invjac, 5);         *invjac         = cmp->invjac;}
  PetscFunctionReturn(0);
}
