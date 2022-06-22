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

.seealso: `DMCreate()`, `DM`, `MATIS`, `PCHPDDM`, `PCHPDDMSetAuxiliaryMat()`
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
  PetscCall(DMPlexDistributeOverlap(dm, 1, &sf, &odm));
  if (!odm) {
    PetscCall(PetscSFDestroy(&sf));
    PetscFunctionReturn(0);
  }

  /* share discretization */
  PetscCall(DMGetLocalSection(dm, &sec));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
  PetscCall(PetscSFDistributeSection(sf, sec, NULL, osec));
  /* what to do here? using both is fine? */
  PetscCall(DMSetLocalSection(odm, osec));
  PetscCall(DMCopyDisc(dm, odm));
  PetscCall(DMPlexGetMaxProjectionHeight(dm, &mh));
  PetscCall(DMPlexSetMaxProjectionHeight(odm, mh));
  PetscCall(PetscSectionDestroy(&osec));

  /* material parameters */
  {
    Vec A;

    PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &A));
    if (A) {
      DM dmAux, ocdm, odmAux;
      Vec oA;

      PetscCall(VecGetDM(A, &dmAux));
      PetscCall(DMClone(odm, &odmAux));
      PetscCall(DMGetCoordinateDM(odm, &ocdm));
      PetscCall(DMSetCoordinateDM(odmAux, ocdm));
      PetscCall(DMCopyDisc(dmAux, odmAux));

      PetscCall(DMGetLocalSection(dmAux, &sec));
      PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
      PetscCall(VecCreate(PetscObjectComm((PetscObject)A), &oA));
      PetscCall(VecSetDM(oA, odmAux));
      /* TODO: what if these values changes? */
      PetscCall(DMPlexDistributeField(dmAux, sf, sec, A, osec, oA));
      PetscCall(DMSetLocalSection(odmAux, osec));
      PetscCall(PetscSectionDestroy(&osec));
      PetscCall(DMSetAuxiliaryVec(odm, NULL, 0, 0, oA));
      PetscCall(VecDestroy(&oA));
      PetscCall(DMDestroy(&odmAux));
    }
  }
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_plex_view_neumann_original"));
  PetscCall(PetscObjectSetName((PetscObject)odm, "OVL"));
  PetscCall(DMViewFromOptions(odm, NULL, "-dm_plex_view_neumann_overlap"));

  /* MATIS for the overlap region
     the HPDDM interface wants local matrices,
     we attach the global MATIS to the overlap IS,
     since we need it to do assembly */
  PetscCall(DMSetMatType(odm, MATIS));
  PetscCall(DMCreateMatrix(odm, &pJ));
  PetscCall(MatISGetLocalMat(pJ, J));
  PetscCall(PetscObjectReference((PetscObject)*J));

  /* overlap IS */
  PetscCall(MatISGetLocalToGlobalMapping(pJ, &l2g, NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(l2g, &n));
  PetscCall(ISLocalToGlobalMappingGetIndices(l2g, &idxs));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)odm), n, idxs, PETSC_COPY_VALUES, ovl));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(l2g, &idxs));
  PetscCall(PetscObjectCompose((PetscObject)*ovl, "_DM_Overlap_HPDDM_MATIS", (PetscObject)pJ));
  PetscCall(DMDestroy(&odm));
  PetscCall(MatDestroy(&pJ));

  /* special purpose setup function (composed in DMPlexSetSNESLocalFEM) */
  PetscCall(PetscObjectQueryFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", setup));
  if (*setup) PetscCall(PetscObjectCompose((PetscObject)*ovl, "_DM_Original_HPDDM", (PetscObject)dm));
  PetscFunctionReturn(0);
}
