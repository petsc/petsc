#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h> /*I "petscdt.h" I*/
#include <petscblaslapack.h>
#include <petscdmplextransform.h>

static PetscErrorCode PetscFEDestroy_Composite(PetscFE fem)
{
  PetscFE_Composite *cmp = (PetscFE_Composite *) fem->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cmp->embedding);CHKERRQ(ierr);
  ierr = PetscFree(cmp);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Get affine mapping from reference cell to each subcell */
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &K);CHKERRQ(ierr);
  ierr = DMGetDimension(K, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(K, 0, &ct);CHKERRQ(ierr);
  ierr = DMPlexTransformCreate(PETSC_COMM_SELF, &tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR);CHKERRQ(ierr);
  ierr = DMPlexRefineRegularGetAffineTransforms(tr, ct, &cmp->numSubelements, &cmp->v0, &cmp->jac, &cmp->invjac);CHKERRQ(ierr);
  ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  /* Determine dof embedding into subelements */
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &spdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(cmp->numSubelements*spdim,&cmp->embedding);CHKERRQ(ierr);
  ierr = DMGetWorkArray(K, dim, MPIU_REAL, &subpoint);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSection(fem->dualSpace, &section);CHKERRQ(ierr);
  for (s = 0; s < cmp->numSubelements; ++s) {
    PetscInt sd = 0;
    PetscInt closureSize;
    PetscInt *closure = NULL;

    ierr = DMPlexGetTransitiveClosure(K, s, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (j = 0; j < closureSize; j++) {
      PetscInt point = closure[2*j];
      PetscInt dof, off, k;

      ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
      for (k = 0; k < dof; k++) cmp->embedding[s*spdim+sd++] = off + k;
    }
    ierr = DMPlexRestoreTransitiveClosure(K, s, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    PetscCheckFalse(sd != spdim,PetscObjectComm((PetscObject) fem), PETSC_ERR_PLIB, "Subelement %d has %d dual basis vectors != %d", s, sd, spdim);
  }
  ierr = DMRestoreWorkArray(K, dim, MPIU_REAL, &subpoint);CHKERRQ(ierr);
  /* Construct the change of basis from prime basis to nodal basis for each subelement */
  ierr = PetscMalloc1(cmp->numSubelements*spdim*spdim,&fem->invV);CHKERRQ(ierr);
  ierr = PetscMalloc2(spdim,&pivots,spdim,&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(cmp->numSubelements*spdim*spdim,&invVscalar);CHKERRQ(ierr);
#else
  invVscalar = fem->invV;
#endif
  for (s = 0; s < cmp->numSubelements; ++s) {
    for (j = 0; j < spdim; ++j) {
      PetscReal       *Bf;
      PetscQuadrature  f;
      const PetscReal *points, *weights;
      PetscInt         Nc, Nq, q, k;

      ierr = PetscDualSpaceGetFunctional(fem->dualSpace, cmp->embedding[s*spdim+j], &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, &Nc, &Nq, &points, &weights);CHKERRQ(ierr);
      ierr = PetscMalloc1(f->numPoints*spdim*Nc,&Bf);CHKERRQ(ierr);
      ierr = PetscSpaceEvaluate(fem->basisSpace, Nq, points, Bf, NULL, NULL);CHKERRQ(ierr);
      for (k = 0; k < spdim; ++k) {
        /* n_j \cdot \phi_k */
        invVscalar[(s*spdim + j)*spdim+k] = 0.0;
        for (q = 0; q < Nq; ++q) {
          invVscalar[(s*spdim + j)*spdim+k] += Bf[q*spdim+k]*weights[q];
        }
      }
      ierr = PetscFree(Bf);CHKERRQ(ierr);
    }
    n = spdim;
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, &invVscalar[s*spdim*spdim], &n, pivots, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, &invVscalar[s*spdim*spdim], &n, pivots, work, &n, &info));
  }
#if defined(PETSC_USE_COMPLEX)
  for (s = 0; s <cmp->numSubelements*spdim*spdim; s++) fem->invV[s] = PetscRealPart(invVscalar[s]);
  ierr = PetscFree(invVscalar);CHKERRQ(ierr);
#endif
  ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, 0, &ct);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &spdim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  /* Divide points into subelements */
  ierr = DMGetWorkArray(dm, npoints, MPIU_INT, &subpoints);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, dim, MPIU_REAL, &subpoint);CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (s = 0; s < cmp->numSubelements; ++s) {
      PetscBool inside;

      /* Apply transform, and check that point is inside cell */
      for (d = 0; d < dim; ++d) {
        subpoint[d] = -1.0;
        for (e = 0; e < dim; ++e) subpoint[d] += cmp->invjac[(s*dim + d)*dim+e]*(points[p*dim+e] - cmp->v0[s*dim+e]);
      }
      ierr = DMPolytopeInCellTest(ct, subpoint, &inside);CHKERRQ(ierr);
      if (inside) {subpoints[p] = s; break;}
    }
    PetscCheckFalse(s >= cmp->numSubelements,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d was not found in any subelement", p);
  }
  ierr = DMRestoreWorkArray(dm, dim, MPIU_REAL, &subpoint);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (K >= 0) {ierr = DMGetWorkArray(dm, npoints*spdim, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (K >= 1) {ierr = DMGetWorkArray(dm, npoints*spdim*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (K >= 2) {ierr = DMGetWorkArray(dm, npoints*spdim*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, tmpB, tmpD, tmpH);CHKERRQ(ierr);
  /* Translate to the nodal basis */
  if (K >= 0) {ierr = PetscArrayzero(B, npoints*pdim*comp);CHKERRQ(ierr);}
  if (K >= 1) {ierr = PetscArrayzero(D, npoints*pdim*comp*dim);CHKERRQ(ierr);}
  if (K >= 2) {ierr = PetscArrayzero(H, npoints*pdim*comp*dim*dim);CHKERRQ(ierr);}
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
  ierr = DMRestoreWorkArray(dm, npoints, MPIU_INT, &subpoints);CHKERRQ(ierr);
  if (K >= 0) {ierr = DMRestoreWorkArray(dm, npoints*spdim, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (K >= 1) {ierr = DMRestoreWorkArray(dm, npoints*spdim*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (K >= 2) {ierr = DMRestoreWorkArray(dm, npoints*spdim*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem, &cmp);CHKERRQ(ierr);
  fem->data = cmp;

  cmp->numSubelements = -1;
  cmp->v0             = NULL;
  cmp->jac            = NULL;

  ierr = PetscFEInitialize_Composite(fem);CHKERRQ(ierr);
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
  if (numSubelements) {PetscValidPointer(numSubelements, 2); *numSubelements = cmp->numSubelements;}
  if (v0)             {PetscValidPointer(v0, 3);             *v0             = cmp->v0;}
  if (jac)            {PetscValidPointer(jac, 4);            *jac            = cmp->jac;}
  if (invjac)         {PetscValidPointer(invjac, 5);         *invjac         = cmp->invjac;}
  PetscFunctionReturn(0);
}
