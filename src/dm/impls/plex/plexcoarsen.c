#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Plex"
PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  DM_Plex           *mesh = (DM_Plex *) dm->data;
  const PetscReal    coarseRatio = PetscSqr(0.5);
  DM                 udm, coordDM;
  Mat                A;
  Vec                metricVec, coordinates, mb, mx;
  PetscSection       coordSection;
  PetscScalar       *metric;
  PetscScalar       *eqns;
  PetscInt           dim, cStart, cEnd, c, vStart, vEnd, numVertices, v, size;
  char               bdLabelName[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!dm->coarseMesh) {
    /* Create metric */
    ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(coordDM, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    numVertices = vEnd - vStart;
    ierr = VecCreateSeq(PETSC_COMM_SELF, numVertices*PetscSqr(dim), &metricVec);CHKERRQ(ierr);
    ierr = VecGetArray(metricVec, &metric);CHKERRQ(ierr);
    size = (dim*(dim+1))/2;
    ierr = PetscMalloc1(PetscSqr(size), &eqns);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, size, size, eqns, &A);CHKERRQ(ierr);
    ierr = MatCreateVecs(A, &mx, &mb);CHKERRQ(ierr);
    ierr = VecSet(mb, 1.0);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscScalar *sol;
      PetscScalar       *cellCoords = NULL;
      PetscReal          e[3], vol;
      const PetscInt    *cone;
      PetscInt           coneSize, cl, i, j, d, r;

      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
      /* Only works for simplices */
      for (i = 0, r = 0; i < dim+1; ++i) {
        for (j = 0; j < i; ++j, ++r) {
          for (d = 0; d < dim; ++d) e[d] = PetscRealPart(cellCoords[i*dim+d] - cellCoords[j*dim+d]);
          /* FORTRAN ORDERING */
          if (dim == 2) {
            eqns[0*size+r] = PetscSqr(e[0]);
            eqns[1*size+r] = 2.0*e[0]*e[1];
            eqns[2*size+r] = PetscSqr(e[1]);
          } else {
            eqns[0*size+r] = PetscSqr(e[0]);
            eqns[1*size+r] = 2.0*e[0]*e[1];
            eqns[2*size+r] = 2.0*e[0]*e[2];
            eqns[3*size+r] = PetscSqr(e[1]);
            eqns[4*size+r] = 2.0*e[1]*e[2];
            eqns[5*size+r] = PetscSqr(e[2]);
          }
        }
      }
      ierr = MatSetUnfactored(A);CHKERRQ(ierr);
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
      ierr = MatLUFactor(A, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = MatSolve(A, mb, mx);CHKERRQ(ierr);
      ierr = VecGetArrayRead(mx, &sol);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
      for (cl = 0; cl < coneSize; ++cl) {
        const PetscInt v = cone[cl] - vStart;

        if (dim == 2) {
          metric[v*4+0] += vol*coarseRatio*sol[0];
          metric[v*4+1] += vol*coarseRatio*sol[1];
          metric[v*4+2] += vol*coarseRatio*sol[1];
          metric[v*4+3] += vol*coarseRatio*sol[2];
        } else {
          metric[v*9+0] += vol*coarseRatio*sol[0];
          metric[v*9+1] += vol*coarseRatio*sol[1];
          metric[v*9+3] += vol*coarseRatio*sol[1];
          metric[v*9+2] += vol*coarseRatio*sol[2];
          metric[v*9+6] += vol*coarseRatio*sol[2];
          metric[v*9+4] += vol*coarseRatio*sol[3];
          metric[v*9+5] += vol*coarseRatio*sol[4];
          metric[v*9+7] += vol*coarseRatio*sol[4];
          metric[v*9+8] += vol*coarseRatio*sol[5];
        }
      }
      ierr = VecRestoreArrayRead(mx, &sol);CHKERRQ(ierr);
    }
    for (v = 0; v < numVertices; ++v) {
      const PetscInt *support;
      PetscInt        supportSize, s;
      PetscReal       vol, totVol = 0.0;

      ierr = DMPlexGetSupport(udm, v+vStart, &support);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(udm, v+vStart, &supportSize);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {ierr = DMPlexComputeCellGeometryFVM(dm, support[s], &vol, NULL, NULL);CHKERRQ(ierr); totVol += vol;}
      for (s = 0; s < PetscSqr(dim); ++s) metric[v*PetscSqr(dim)+s] /= totVol;
    }
    ierr = VecRestoreArray(metricVec, &metric);CHKERRQ(ierr);
    ierr = VecDestroy(&mx);CHKERRQ(ierr);
    ierr = VecDestroy(&mb);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = DMDestroy(&udm);CHKERRQ(ierr);
    ierr = PetscFree(eqns);CHKERRQ(ierr);

    bdLabelName[0] = '\0';
    ierr = PetscOptionsGetString(NULL, dm->hdr.prefix, "-dm_plex_coarsen_bd_label", bdLabelName, PETSC_MAX_PATH_LEN-1, NULL);CHKERRQ(ierr);
    ierr = DMPlexRemesh_Internal(dm, metricVec, bdLabelName, mesh->remeshBd, &dm->coarseMesh);CHKERRQ(ierr);
    ierr = VecDestroy(&metricVec);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject) dm->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_Plex"
PetscErrorCode DMCoarsenHierarchy_Plex(DM dm, PetscInt nlevels, DM dmCoarsened[])
{
  DM             rdm = dm;
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (c = nlevels-1; c >= 0; --c) {
    ierr = DMCoarsen(rdm, PetscObjectComm((PetscObject) dm), &dmCoarsened[c]);CHKERRQ(ierr);
    ierr = DMCopyBoundary(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    ierr = DMSetCoarseDM(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    rdm  = dmCoarsened[c];
  }
  PetscFunctionReturn(0);
}
