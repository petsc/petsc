#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/*@C
     DMCreateNeumannOverlap - Generates an IS, an unassembled (Neumann) Mat, a setup function, and the corresponding context to be used by PCHPDDM.

   Input Parameter:
.     dm - preconditioner context

   Output Parameters:
+     ovl - index set of overlapping subdomains
.     J - unassembled (Neumann) local matrix
.     setup - function for generating the matrix
-     setup_ctx - context for setup

   Options Database Keys:
+   -dm_plex_view_neumann_original - view the DM without overlap
-   -dm_plex_view_neumann_overlap - view the DM with overlap as needed by PCHPDDM

   Level: advanced

.seealso:  DMCreate(), DM, MATIS, PCHPDDM, PCHPDDMSetAuxiliaryMat()
@*/
PetscErrorCode DMCreateNeumannOverlap_Plex(DM dm, IS *ovl, Mat *J, PetscErrorCode (**setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void **setup_ctx)
{
  DM                     odm;
  Mat                    pJ;
  PetscSF                sf = NULL;
  PetscSection           sec, osec;
  ISLocalToGlobalMapping l2g;
  const PetscInt         *idxs;
  PetscInt               n, mh;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  *setup     = NULL;
  *setup_ctx = NULL;
  *ovl       = NULL;
  *J         = NULL;

  /* Overlapped mesh
     overlap is a little more generous, since it is not computed starting from the owned (Dirichlet) points, but from the locally owned cells */
  ierr = DMPlexDistributeOverlap(dm, 1, &sf, &odm);CHKERRQ(ierr);
  if (!odm) {
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* share discretization */
  ierr = DMGetLocalSection(dm, &sec);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sf, sec, NULL, osec);CHKERRQ(ierr);
  /* what to do here? using both is fine? */
  ierr = DMSetLocalSection(odm, osec);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, odm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(dm, &mh);CHKERRQ(ierr);
  ierr = DMPlexSetMaxProjectionHeight(odm, mh);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&osec);CHKERRQ(ierr);

  /* material parameters */
  {
    Vec A;

    ierr = DMGetAuxiliaryVec(dm, NULL, 0, 0, &A);CHKERRQ(ierr);
    if (A) {
      DM dmAux, ocdm, odmAux;
      Vec oA;

      ierr = VecGetDM(A, &dmAux);CHKERRQ(ierr);
      ierr = DMClone(odm, &odmAux);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(odm, &ocdm);CHKERRQ(ierr);
      ierr = DMSetCoordinateDM(odmAux, ocdm);CHKERRQ(ierr);
      ierr = DMCopyDisc(dmAux, odmAux);CHKERRQ(ierr);

      ierr = DMGetLocalSection(dmAux, &sec);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec);CHKERRQ(ierr);
      ierr = VecCreate(PetscObjectComm((PetscObject)A), &oA);CHKERRQ(ierr);
      ierr = VecSetDM(oA, odmAux);CHKERRQ(ierr);
      /* TODO: what if these values changes? */
      ierr = DMPlexDistributeField(dmAux, sf, sec, A, osec, oA);CHKERRQ(ierr);
      ierr = DMSetLocalSection(odmAux, osec);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&osec);CHKERRQ(ierr);
      ierr = DMSetAuxiliaryVec(odm, NULL, 0, 0, oA);CHKERRQ(ierr);
      ierr = VecDestroy(&oA);CHKERRQ(ierr);
      ierr = DMDestroy(&odmAux);CHKERRQ(ierr);
    }
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dm, NULL, "-dm_plex_view_neumann_original");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)odm, "OVL");CHKERRQ(ierr);
  ierr = DMViewFromOptions(odm, NULL, "-dm_plex_view_neumann_overlap");CHKERRQ(ierr);

  /* MATIS for the overlap region
     the HPDDM interface wants local matrices,
     we attach the global MATIS to the overlap IS,
     since we need it to do assembly */
  ierr = DMSetMatType(odm, MATIS);CHKERRQ(ierr);
  ierr = DMCreateMatrix(odm, &pJ);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(pJ, J);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)*J);CHKERRQ(ierr);

  /* overlap IS */
  ierr = MatISGetLocalToGlobalMapping(pJ, &l2g, NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(l2g, &n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(l2g, &idxs);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)odm), n, idxs, PETSC_COPY_VALUES, ovl);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(l2g, &idxs);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*ovl, "_DM_Overlap_HPDDM_MATIS", (PetscObject)pJ);CHKERRQ(ierr);
  ierr = DMDestroy(&odm);CHKERRQ(ierr);
  ierr = MatDestroy(&pJ);CHKERRQ(ierr);

  /* special purpose setup function (composed in DMPlexSetSNESLocalFEM) */
  ierr = PetscObjectQueryFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", setup);CHKERRQ(ierr);
  if (*setup) {
    ierr = PetscObjectCompose((PetscObject)*ovl, "_DM_Original_HPDDM", (PetscObject)dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
