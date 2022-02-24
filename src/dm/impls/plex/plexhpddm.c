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

  PetscFunctionBegin;
  *setup     = NULL;
  *setup_ctx = NULL;
  *ovl       = NULL;
  *J         = NULL;

  /* Overlapped mesh
     overlap is a little more generous, since it is not computed starting from the owned (Dirichlet) points, but from the locally owned cells */
  CHKERRQ(DMPlexDistributeOverlap(dm, 1, &sf, &odm));
  if (!odm) {
    CHKERRQ(PetscSFDestroy(&sf));
    PetscFunctionReturn(0);
  }

  /* share discretization */
  CHKERRQ(DMGetLocalSection(dm, &sec));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
  CHKERRQ(PetscSFDistributeSection(sf, sec, NULL, osec));
  /* what to do here? using both is fine? */
  CHKERRQ(DMSetLocalSection(odm, osec));
  CHKERRQ(DMCopyDisc(dm, odm));
  CHKERRQ(DMPlexGetMaxProjectionHeight(dm, &mh));
  CHKERRQ(DMPlexSetMaxProjectionHeight(odm, mh));
  CHKERRQ(PetscSectionDestroy(&osec));

  /* material parameters */
  {
    Vec A;

    CHKERRQ(DMGetAuxiliaryVec(dm, NULL, 0, 0, &A));
    if (A) {
      DM dmAux, ocdm, odmAux;
      Vec oA;

      CHKERRQ(VecGetDM(A, &dmAux));
      CHKERRQ(DMClone(odm, &odmAux));
      CHKERRQ(DMGetCoordinateDM(odm, &ocdm));
      CHKERRQ(DMSetCoordinateDM(odmAux, ocdm));
      CHKERRQ(DMCopyDisc(dmAux, odmAux));

      CHKERRQ(DMGetLocalSection(dmAux, &sec));
      CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
      CHKERRQ(VecCreate(PetscObjectComm((PetscObject)A), &oA));
      CHKERRQ(VecSetDM(oA, odmAux));
      /* TODO: what if these values changes? */
      CHKERRQ(DMPlexDistributeField(dmAux, sf, sec, A, osec, oA));
      CHKERRQ(DMSetLocalSection(odmAux, osec));
      CHKERRQ(PetscSectionDestroy(&osec));
      CHKERRQ(DMSetAuxiliaryVec(odm, NULL, 0, 0, oA));
      CHKERRQ(VecDestroy(&oA));
      CHKERRQ(DMDestroy(&odmAux));
    }
  }
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_plex_view_neumann_original"));
  CHKERRQ(PetscObjectSetName((PetscObject)odm, "OVL"));
  CHKERRQ(DMViewFromOptions(odm, NULL, "-dm_plex_view_neumann_overlap"));

  /* MATIS for the overlap region
     the HPDDM interface wants local matrices,
     we attach the global MATIS to the overlap IS,
     since we need it to do assembly */
  CHKERRQ(DMSetMatType(odm, MATIS));
  CHKERRQ(DMCreateMatrix(odm, &pJ));
  CHKERRQ(MatISGetLocalMat(pJ, J));
  CHKERRQ(PetscObjectReference((PetscObject)*J));

  /* overlap IS */
  CHKERRQ(MatISGetLocalToGlobalMapping(pJ, &l2g, NULL));
  CHKERRQ(ISLocalToGlobalMappingGetSize(l2g, &n));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(l2g, &idxs));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)odm), n, idxs, PETSC_COPY_VALUES, ovl));
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(l2g, &idxs));
  CHKERRQ(PetscObjectCompose((PetscObject)*ovl, "_DM_Overlap_HPDDM_MATIS", (PetscObject)pJ));
  CHKERRQ(DMDestroy(&odm));
  CHKERRQ(MatDestroy(&pJ));

  /* special purpose setup function (composed in DMPlexSetSNESLocalFEM) */
  CHKERRQ(PetscObjectQueryFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", setup));
  if (*setup) CHKERRQ(PetscObjectCompose((PetscObject)*ovl, "_DM_Original_HPDDM", (PetscObject)dm));
  PetscFunctionReturn(0);
}
