#include <petsc/private/dmpleximpl.h>           /*I      "petscdmplex.h"          I*/

#ifdef PETSC_HAVE_LIBCEED
#include <petscdmplexceed.h>

/* Define the map from the local vector (Lvector) to the cells (Evector) */
PetscErrorCode DMPlexGetCeedRestriction(DM dm, CeedElemRestriction *ERestrict)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ERestrict, 2);
  if (!dm->ceedERestrict) {
    PetscDS      ds;
    PetscFE      fe;
    PetscSpace   sp;
    PetscSection s;
    PetscInt     Nf, *Nc, c, P, cStart, cEnd, Ncell, cell, lsize, *erestrict, eoffset;
    PetscBool    simplex;
    Ceed         ceed;

    ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
    if (simplex) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "LibCEED does not work with simplices");
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &Nf);
    if (Nf != 1) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "LibCEED only works with one field right now");
    ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(ds, 0, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(sp, &P, NULL);CHKERRQ(ierr);
    ++P;
    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart,& cEnd);CHKERRQ(ierr);
    Ncell = cEnd - cStart;

    ierr = PetscMalloc1(Ncell*P*P, &erestrict);CHKERRQ(ierr);
    for (cell = cStart, eoffset = 0; cell < cEnd; ++cell) {
      PetscInt Ni, *ind, i;

      ierr = DMPlexGetClosureIndices(dm, s, s, cell, PETSC_TRUE, &Ni, &ind, NULL, NULL);CHKERRQ(ierr);
      for (i = 0; i < Ni; i += Nc[0]) {
        for (c = 0; c < Nc[0]; ++c) {
          if (ind[i+c] != ind[i] + c) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %D closure indices not interlaced", cell);
        }
        erestrict[eoffset++] = ind[i];
      }
      ierr = DMPlexRestoreClosureIndices(dm, s, s, cell, PETSC_TRUE, &Ni, &ind, NULL, NULL);CHKERRQ(ierr);
    }
    if (eoffset != Ncell*P*P) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Size of array %D != %D restricted dofs", Ncell*P*P, eoffset);

    ierr = DMGetCeed(dm, &ceed);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(s, &lsize);CHKERRQ(ierr);
    ierr = CeedElemRestrictionCreate(ceed, Ncell, P*P, Nc[0], 1, lsize, CEED_MEM_HOST, CEED_COPY_VALUES, erestrict, &dm->ceedERestrict);CHKERRQ(ierr);
    ierr = PetscFree(erestrict);CHKERRQ(ierr);
  }
  *ERestrict = dm->ceedERestrict;
  PetscFunctionReturn(0);
}

#endif
