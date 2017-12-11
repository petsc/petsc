static char help[] = "Test FEM layout with DM\n\n";

#include <petsc.h>
#ifdef PETSC_HAVE_EXODUSII
#include <exodusII.h>
#endif

int main(int argc, char **argv) {
  DM                dm, dmU, dmA, dmS, dmUA;
  IS                isU, isA, isS, isUA;
  PetscSection      section;
  const PetscInt    fieldU = 0;
  const PetscInt    fieldA = 2;
  const PetscInt    fieldS = 1;
  const PetscInt    fieldUA[2] = {0, 2};
  char              ifilename[PETSC_MAX_PATH_LEN];
  IS                csIS;
  const PetscInt   *csID;
  PetscInt         *pStartDepth, *pEndDepth;
  PetscInt          order = 1;
  PetscInt          sdim, d, pStart, pEnd, p, numCS, set;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv,NULL, help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex26");CHKERRQ(ierr);
  ierr = PetscOptionsString("-i", "Filename to read", "ex26", ifilename, ifilename, sizeof(ifilename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-order", "FEM polynomial order", "ex26", order, &order, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

#ifdef PETSC_HAVE_EXODUSII
  ex_opts(EX_VERBOSE+EX_DEBUG);
#endif
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* Create the main section containning all fields */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(section, 3);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldU, "U");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldA, "Alpha");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldS, "Sigma");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &sdim);CHKERRQ(ierr);
  ierr = PetscMalloc2(sdim+1, &pStartDepth, sdim+1, &pEndDepth);CHKERRQ(ierr);
  for (d = 0; d <= sdim; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStartDepth[d], &pEndDepth[d]);CHKERRQ(ierr);}
  /* Vector field U, Scalar field Alpha, Tensor field Sigma */
  ierr = PetscSectionSetFieldComponents(section, fieldU, sdim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(section, fieldA, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2);CHKERRQ(ierr);

  /* Going through cell sets then cells, and setting up storage for the sections */
  ierr = DMGetLabelSize(dm, "Cell Sets", &numCS);CHKERRQ(ierr);
  ierr = DMGetLabelIdIS(dm, "Cell Sets", &csIS);CHKERRQ(ierr);
  ierr = ISGetIndices(csIS, &csID);CHKERRQ(ierr);
  for (set = 0; set < numCS; set++) {
    IS                cellIS;
    const PetscInt   *cellID;
    PetscInt          numCells, cell, closureSize, *closureA = NULL;

    ierr = DMGetStratumSize(dm, "Cell Sets", csID[set], &numCells);CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "Cell Sets", csID[set], &cellIS);CHKERRQ(ierr);
    if (numCells > 0) {
      /* dof layout ordered by increasing height in the DAG: cell, face, edge, vertex */
      PetscInt          dofUP1Tri[]  = {2, 0, 0};
      PetscInt          dofAP1Tri[]  = {1, 0, 0};
      PetscInt          dofUP2Tri[]  = {2, 2, 0};
      PetscInt          dofAP2Tri[]  = {1, 1, 0};
      PetscInt          dofUP1Quad[] = {2, 0, 0};
      PetscInt          dofAP1Quad[] = {1, 0, 0};
      PetscInt          dofUP2Quad[] = {2, 2, 2};
      PetscInt          dofAP2Quad[] = {1, 1, 1};
      PetscInt          dofS2D[]     = {0, 0, 3};
      PetscInt          dofUP1Tet[]  = {3, 0, 0, 0};
      PetscInt          dofAP1Tet[]  = {1, 0, 0, 0};
      PetscInt          dofUP2Tet[]  = {3, 3, 0, 0};
      PetscInt          dofAP2Tet[]  = {1, 1, 0, 0};
      PetscInt          dofUP1Hex[]  = {3, 0, 0, 0};
      PetscInt          dofAP1Hex[]  = {1, 0, 0, 0};
      PetscInt          dofUP2Hex[]  = {3, 3, 3, 3};
      PetscInt          dofAP2Hex[]  = {1, 1, 1, 1};
      PetscInt          dofS3D[]     = {0, 0, 0, 6};
      PetscInt         *dofU, *dofA, *dofS;

      switch (sdim) {
      case 2: dofS = dofS2D;break;
      case 3: dofS = dofS3D;break;
      default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %D", sdim);
      }

      /* Identify cell type based on closure size only. This works for Tri/Tet/Quad/Hex meshes
         It will not be enough to identify more exotic elements like pyramid or prisms...  */
      ierr = ISGetIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA);CHKERRQ(ierr);
      switch (closureSize) {
        case 7: /* Tri */
        if (order == 1) {
          dofU = dofUP1Tri;
          dofA = dofAP1Tri;
        } else {
          dofU = dofUP2Tri;
          dofA = dofAP2Tri;
        }
        break;
        case 9: /* Quad */
        if (order == 1) {
          dofU = dofUP1Quad;
          dofA = dofAP1Quad;
        } else {
          dofU = dofUP2Quad;
          dofA = dofAP2Quad;
        }
        break;
        case 15: /* Tet */
        if (order == 1) {
          dofU = dofUP1Tet;
          dofA = dofAP1Tet;
        } else {
          dofU = dofUP2Tet;
          dofA = dofAP2Tet;
        }
        break;
        case 27: /* Hex */
        if (order == 1) {
          dofU = dofUP1Hex;
          dofA = dofAP1Hex;
        } else {
          dofU = dofUP2Hex;
          dofA = dofAP2Hex;
        }
        break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Unkown element with closure size %D\n", closureSize);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA);CHKERRQ(ierr);

      for (cell = 0; cell < numCells; cell++) {
        PetscInt *closure = NULL;

        ierr = DMPlexGetTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (p = 0; p < closureSize; ++p) {
          /* Find depth of p */
          for (d = 0; d <= sdim; ++d) {
            if ((closure[2*p] >= pStartDepth[d]) && (closure[2*p] < pEndDepth[d])) {
              ierr = PetscSectionSetDof(section, closure[2*p], dofU[d]+dofA[d]+dofS[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldU, dofU[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldA, dofA[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldS, dofS[d]);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(csIS, &csID);CHKERRQ(ierr);
  ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  {
    DM pdm;

    ierr = DMSetUseNatural(dm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm = pdm;
    }
  }
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Get DM and IS for each field of dm */
  ierr = DMCreateSubDM(dm, 1, &fieldU, &isU,  &dmU);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 1, &fieldA, &isA,  &dmA);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 1, &fieldS, &isS,  &dmS);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 2, fieldUA, &isUA, &dmUA);CHKERRQ(ierr);

  ierr = DMDestroy(&dmU);CHKERRQ(ierr); ierr = ISDestroy(&isU);CHKERRQ(ierr);
  ierr = DMDestroy(&dmA);CHKERRQ(ierr); ierr = ISDestroy(&isA);CHKERRQ(ierr);
  ierr = DMDestroy(&dmS);CHKERRQ(ierr); ierr = ISDestroy(&isS);CHKERRQ(ierr);
  ierr = DMDestroy(&dmUA);CHKERRQ(ierr);ierr = ISDestroy(&isUA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree2(pStartDepth, pEndDepth);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
    requires: exodusii
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo -order 2

TEST*/
