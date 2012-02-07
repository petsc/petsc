#include <private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMComplexView_Ascii"
PetscErrorCode DMComplexView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const char *name;
    PetscInt    maxConeSize, maxSupportSize;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh '%s':\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Max sizes cone: %d support: %d\n", maxConeSize, maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "orientation is missing\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "cap --> base:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for(s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d ----> %d\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "base <-- cap:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for(c = off; c < off+dof; ++c) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d <---- %d\n", rank, p, mesh->cones[c]);CHKERRQ(ierr);
        /* ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d <---- %d: %d\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]);CHKERRQ(ierr); */
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscSectionVecView(mesh->coordSection, mesh->coordinates, viewer);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm = ((PetscObject) dm)->comm;
    PetscInt   *sizes;
    PetscInt    locDepth, depth, dim, d;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt size;

    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    ierr = DMComplexGetLabelSize(dm, "depth", &locDepth);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &sizes);CHKERRQ(ierr);
    ierr = DMComplexGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
    if (depth == 2) {
      ierr = DMComplexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", 0);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMComplexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", dim);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for(d = 0; d <= dim; d++) {
        ierr = DMComplexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd = pEnd - pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", d);CHKERRQ(ierr);
        for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(sizes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Complex"
PetscErrorCode DMView_Complex(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, isbinary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMComplexView_Ascii(dm, viewer);CHKERRQ(ierr);
#if 0
  } else if (isbinary) {
    ierr = DMComplexView_Binary(dm, viewer);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Complex"
PetscErrorCode DMDestroy_Complex(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionDestroy(&mesh->defaultSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->defaultGlobalSection);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&mesh->sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&mesh->sfDefault);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->cones);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->coordSection);CHKERRQ(ierr);
  ierr = VecDestroy(&mesh->coordinates);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->meetTmpA,mesh->meetTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->joinTmpA,mesh->joinTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->closureTmpA,mesh->closureTmpB);CHKERRQ(ierr);
  while(next) {
    SieveLabel tmp;

    ierr = PetscFree(next->name);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumValues);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumOffsets);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumSizes);CHKERRQ(ierr);
    ierr = PetscFree(next->points);CHKERRQ(ierr);
    tmp  = next->next;
    ierr = PetscFree(next);CHKERRQ(ierr);
    next = tmp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Complex"
PetscErrorCode DMCreateGlobalVector_Complex(DM dm, Vec *gvec)
{
  PetscSection   s;
  PetscInt       localSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDefaultGlobalSection(dm, &s);CHKERRQ(ierr);
  /* ierr = PetscSectionGetStorageSize(s, &localSize);CHKERRQ(ierr); */
  ierr = PetscSectionGetConstrainedStorageSize(s, &localSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, localSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *gvec, "DM", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Complex"
PetscErrorCode DMCreateLocalVector_Complex(DM dm, Vec *lvec)
{
  PetscSection   s;
  PetscInt       size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDefaultSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(s, &size);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, lvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*lvec, size, size);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*lvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *lvec, "DM", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPreallocateOperator"
PetscErrorCode DMComplexPreallocateOperator(DM dm, PetscInt bs, PetscSection section, PetscSection sectionGlobal, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  MPI_Comm           comm = ((PetscObject) dm)->comm;
  PetscSF            sf   = mesh->sf, sfDof, sfAdj;
  PetscSection       leafSectionAdj, rootSectionAdj, sectionAdj;
  PetscInt           nleaves, l, p;
  const PetscInt    *leaves;
  const PetscSFNode *remotes;
  PetscInt           pStart, pEnd, numDof, numOwnedDof, numCols;
  PetscInt          *tmpAdj, *adj, *rootAdj, *cols;
  PetscInt           maxConeSize, maxSupportSize, maxAdjSize, adjSize;
  PetscLayout        rLayout;
  PetscInt           locRows, rStart, rEnd, r;
  PetscMPIInt        size, debug = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) debug = 1;
  /* Create dof SF based on point SF */
  if (debug) {
    ierr = PetscPrintf(comm, "Input Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Input Global Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(sectionGlobal, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Input SF for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSFView(sf, PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscSFCreateSectionSF(sf, section, PETSC_NULL, section, &sfDof);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Dof SF for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSFView(sfDof, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Create section for dof adjacency (dof ==> # adj dof) */
  /*   Two points p and q are adjacent if q \in closure(star(p)) */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &numDof);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(leafSectionAdj, 0, numDof);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &rootSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSectionAdj, 0, numDof);CHKERRQ(ierr);
  /*   Fill in the ghost dofs on the interface */
  ierr = PetscSFGetGraph(sf, PETSC_NULL, &nleaves, &leaves, &remotes);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  maxAdjSize = maxSupportSize*maxConeSize;
  ierr = PetscMalloc(maxAdjSize * sizeof(PetscInt), &tmpAdj);CHKERRQ(ierr);
  for(l = 0; l < nleaves; ++l) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscInt        dof, off, d, q;
    PetscInt        p = leaves[l], numAdj = 0;

    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj; ++q) {
          if (cone[c] == tmpAdj[q]) break;
        }
        if (q == numAdj) {
          if (numAdj >= maxAdjSize) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid mesh exceeded adjacency allocation (%d)", maxAdjSize);}
          tmpAdj[q] = cone[c];
          ++numAdj;
        }
      }
    }
    ierr = PetscSortRemoveDupsInt(&numAdj, tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    for(q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
      for(d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(leafSectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(leafSectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjacency Section for Preallocation on Leaves:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(leafSectionAdj, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Get maximum remote adjacency sizes for owned dofs on interface (roots) */
  if (size > 1) {
    ierr = PetscSFReduceBegin(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM);CHKERRQ(ierr);
  }
  if (debug) {
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Add in local adjacency sizes for owned dofs on interface (roots) */
  for(p = pStart; p < pEnd; ++p) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscInt        numAdj = 0, adof, dof, off, d, q;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj; ++q) {
          if (cone[c] == tmpAdj[q]) break;
        }
        if (q == numAdj) {
          if (numAdj >= maxAdjSize) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid mesh exceeded adjacency allocation (%d)", maxAdjSize);}
          tmpAdj[q] = cone[c];
          ++numAdj;
        }
      }
    }
    ierr = PetscSortRemoveDupsInt(&numAdj, tmpAdj);CHKERRQ(ierr);
    for(q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      for(d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(rootSectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(rootSectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after local additions:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Create adj SF based on dof SF */
  ierr = PetscSFCreateSectionSF(sfDof, rootSectionAdj, PETSC_NULL, leafSectionAdj, &sfAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjacency SF for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSFView(sfAdj, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Create leaf adjacency */
  ierr = PetscSectionSetUp(leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(leafSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscMalloc(adjSize * sizeof(PetscInt), &adj);CHKERRQ(ierr);
  ierr = PetscMemzero(adj, adjSize * sizeof(PetscInt));CHKERRQ(ierr);
  for(l = 0; l < nleaves; ++l) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscInt        dof, off, d, q;
    PetscInt        p = leaves[l], numAdj = 0;

    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj; ++q) {
          if (cone[c] == tmpAdj[q]) break;
        }
        if (q == numAdj) {
          if (numAdj >= maxAdjSize) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid mesh exceeded adjacency allocation (%d)", maxAdjSize);}
          tmpAdj[q] = cone[c];
          ++numAdj;
        }
      }
    }
    ierr = PetscSortRemoveDupsInt(&numAdj, tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    for(d = off; d < off+dof; ++d) {
      PetscInt aoff, i = 0;

      ierr = PetscSectionGetOffset(leafSectionAdj, d, &aoff);CHKERRQ(ierr);
      for(q = 0; q < numAdj; ++q) {
        PetscInt  ndof, ncdof, ngoff, nd;

        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for(nd = 0; nd < ndof-ncdof; ++nd) {
          adj[aoff+i] = (ngoff < 0 ? -(ngoff+1) : ngoff) + nd;
          ++i;
        }
      }
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = ISCreateGeneral(comm, adjSize, adj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Gather adjacenct indices to root */
  ierr = PetscSectionGetStorageSize(rootSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscMalloc(adjSize * sizeof(PetscInt), &rootAdj);CHKERRQ(ierr);
  if (size > 1) {
    ierr = PetscSFGatherBegin(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
  }
  /* Compress indices */
  ierr = PetscSectionSetUp(rootSectionAdj);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    for(d = off; d < off+dof-cdof; ++d) {
      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&adof, &rootAdj[off]);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(rootSectionAdj, d, adof);CHKERRQ(ierr);
    }
  }
  /* Build adjacency section */
  ierr = PetscSectionGetStorageSize(sectionGlobal, &numOwnedDof);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &sectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionAdj, 0, numOwnedDof);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscInt        numAdj = 0, dof, cdof, off, goff, d, q;
    PetscBool       found = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for(d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
        ierr = PetscSectionSetDof(sectionAdj, goff+d, -(ldof+1));CHKERRQ(ierr);
      } else if (rdof > 0) {
        ierr = PetscSectionSetDof(sectionAdj, goff+d, rdof);CHKERRQ(ierr);
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj; ++q) {
          if (cone[c] == tmpAdj[q]) break;
        }
        if (q == numAdj) {
          if (numAdj >= maxAdjSize) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid mesh exceeded adjacency allocation (%d)", maxAdjSize);}
          tmpAdj[q] = cone[c];
          ++numAdj;
        }
      }
    }
    ierr = PetscSortRemoveDupsInt(&numAdj, tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for(q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof, noff;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, tmpAdj[q], &noff);CHKERRQ(ierr);
      for(d = goff; d < goff+dof-cdof; ++d) {
        ierr = PetscSectionAddDof(sectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(sectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjacency Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(sectionAdj, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Get adjacent indices */
  ierr = PetscSectionGetStorageSize(sectionAdj, &numCols);CHKERRQ(ierr);
  ierr = PetscMalloc(numCols * sizeof(PetscInt), &cols);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscInt        numAdj = 0, dof, cdof, off, goff, d, q;
    PetscBool       found = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for(d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
      } else if (rdof > 0) {
        PetscInt aoff, roff;

        ierr = PetscSectionGetOffset(sectionAdj, goff+d, &aoff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(rootSectionAdj, off+d, &roff);CHKERRQ(ierr);
        ierr = PetscMemcpy(&cols[aoff], &rootAdj[roff], rdof * sizeof(PetscInt));CHKERRQ(ierr);
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj; ++q) {
          if (cone[c] == tmpAdj[q]) break;
        }
        if (q == numAdj) {
          if (numAdj >= maxAdjSize) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid mesh exceeded adjacency allocation (%d)", maxAdjSize);}
          tmpAdj[q] = cone[c];
          ++numAdj;
        }
      }
    }
    ierr = PetscSortRemoveDupsInt(&numAdj, tmpAdj);CHKERRQ(ierr);
    for(d = goff; d < goff+dof-cdof; ++d) {
      PetscInt adof, aoff, i = 0;

      ierr = PetscSectionGetDof(sectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, d, &aoff);CHKERRQ(ierr);
      for(q = 0; q < numAdj; ++q) {
        PetscInt  ndof, ncdof, ngoff, nd;
        PetscInt *ncind;

        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(section, tmpAdj[q], &ncind);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for(nd = 0; nd < ndof-ncdof; ++nd, ++i) {
          cols[aoff+i] = ngoff+nd;
        }
      }
      if (i != adof) {SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %d != %d for dof %d (point %d)", i, adof, d, p);}
    }
  }
  /* Create allocation vectors from adjacency graph */
  ierr = MatGetLocalSize(A, &locRows, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(((PetscObject) A)->comm, &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  for(r = rStart; r < rEnd; ++r) {
    PetscInt numCols, cStart, c;

    ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
    for(c = cStart; c < cStart+numCols; ++c) {
      if ((cols[c] >= rStart) && (cols[c] < rEnd)) {
        ++dnz[r-rStart];
        if (cols[c] >= r) {++dnzu[r-rStart];}
      } else {
        ++onz[r-rStart];
        if (cols[c] >= r) {++onzu[r-rStart];}
      }
    }
  }
  /* Set matrix pattern */
  ierr = MatXAIJSetPreallocation(A, bs, 0, dnz, 0, onz, 0, dnzu, 0, onzu);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* Fill matrix with zeros */
  if (fillMatrix) {
    PetscScalar *values;
    PetscInt     maxRowLen = 0;

    for(r = rStart; r < rEnd; ++r) {
      PetscInt len;

      ierr = PetscSectionGetDof(sectionAdj, r, &len);CHKERRQ(ierr);
      maxRowLen = PetscMax(maxRowLen, len);
    }
    ierr = PetscMalloc(maxRowLen * sizeof(PetscScalar), &values);CHKERRQ(ierr);
    ierr = PetscMemzero(values, maxRowLen * sizeof(PetscScalar));CHKERRQ(ierr);
    for(r = rStart; r < rEnd; ++r) {
      PetscInt numCols, cStart;

      ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
      ierr = MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Complex"
PetscErrorCode DMCreateMatrix_Complex(DM dm, const MatType mtype, Mat *J)
{
  PetscSection   section, sectionGlobal;
  PetscInt       bs = -1;
  PetscInt       localSize;
  PetscBool      isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock, isSymmetric;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  if (!mtype) mtype = MATAIJ;
  ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMComplexGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  /* ierr = PetscSectionGetStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr); */
  ierr = PetscSectionGetConstrainedStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject) dm)->comm, J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSHELL, &isShell);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATBAIJ, &isBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSBAIJ, &isSymBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock);CHKERRQ(ierr);
  /* Check for symmetric storage */
  isSymmetric = (PetscBool) (isSymBlock || isSymSeqBlock || isSymMPIBlock);
  if (isSymmetric) {
    ierr = MatSetOption(*J, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!isShell) {
    PetscBool fillMatrix = (PetscBool) !dm->prealloc_only;
    PetscInt *dnz, *onz, *dnzu, *onzu, bsLocal;

    if (bs < 0) {
      if (isBlock || isSeqBlock || isMPIBlock || isSymBlock || isSymSeqBlock || isSymMPIBlock) {
        PetscInt pStart, pEnd, p, dof;

        ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
        for(p = pStart; p < pEnd; ++p) {
          ierr = PetscSectionGetDof(sectionGlobal, p, &dof);CHKERRQ(ierr);
          if (dof) {
            bs = dof;
            break;
          }
        }
      } else {
        bs = 1;
      }
      /* Must have same blocksize on all procs (some might have no points) */
      bsLocal = bs;
      ierr = MPI_Allreduce(&bsLocal, &bs, 1, MPIU_INT, MPI_MAX, ((PetscObject) dm)->comm);CHKERRQ(ierr);
    }
    ierr = PetscMalloc4(localSize/bs, PetscInt, &dnz, localSize/bs, PetscInt, &onz, localSize/bs, PetscInt, &dnzu, localSize/bs, PetscInt, &onzu);CHKERRQ(ierr);
    ierr = PetscMemzero(dnz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(onz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(dnzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(onzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = DMComplexPreallocateOperator(dm, bs, section, sectionGlobal, dnz, onz, dnzu, onzu, *J, fillMatrix);CHKERRQ(ierr);
    ierr = PetscFree4(dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDimension"
/*@
  DMComplexGetDimension - Return the topological mesh dimension

  Not collective

  Input Parameter:
. mesh - The DMComplex

  Output Parameter:
. dim - The topological mesh dimension

  Level: beginner

.seealso: DMComplexCreate()
@*/
PetscErrorCode DMComplexGetDimension(DM dm, PetscInt *dim)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = mesh->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetDimension"
/*@
  DMComplexSetDimension - Set the topological mesh dimension

  Collective on mesh

  Input Parameters:
+ mesh - The DMComplex
- dim - The topological mesh dimension

  Level: beginner

.seealso: DMComplexCreate()
@*/
PetscErrorCode DMComplexSetDimension(DM dm, PetscInt dim)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  mesh->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetChart"
/*@
  DMComplexGetChart - Return the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameter:
. mesh - The DMComplex

  Output Parameters:
+ pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexGetChart(DM dm, PetscInt *pStart, PetscInt *pEnd)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetChart"
/*@
  DMComplexSetChart - Set the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Output Parameters:

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetChart()
@*/
PetscErrorCode DMComplexSetChart(DM dm, PetscInt pStart, PetscInt pEnd)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetConeSize"
/*@
  DMComplexGetConeSize - Return the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
- p - The Sieve point, which must lie in the chart set with DMComplexSetChart()

  Output Parameter:
. size - The cone size for point p

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetConeSize(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexGetConeSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetConeSize"
/*@
  DMComplexSetConeSize - Set the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- size - The cone size for point p

  Output Parameter:

  Note:
  This should be called after DMComplexSetChart().

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetConeSize(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexSetConeSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
  mesh->maxConeSize = PetscMax(mesh->maxConeSize, size);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCone"
/*@C
  DMComplexGetCone - Return the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
- p - The Sieve point, which must lie in the chart set with DMComplexSetChart()

  Output Parameter:
. cone - An array of points which are on the in-edges for point p

  Level: beginner

  Note:
  This routine is not available in Fortran.

.seealso: DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexGetCone(DM dm, PetscInt p, const PetscInt *cone[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  *cone = &mesh->cones[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetCone"
/*@
  DMComplexSetCone - Set the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- cone - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMComplexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetCone(), DMComplexSetChart(), DMComplexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMComplexSetCone(DM dm, PetscInt p, const PetscInt cone[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) {SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %d is not in the valid range [%d, %d)", p, pStart, pEnd);}
  for(c = 0; c < dof; ++c) {
    if ((cone[c] < pStart) || (cone[c] >= pEnd)) {SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %d is not in the valid range [%d. %d)", cone[c], pStart, pEnd);}
    mesh->cones[off+c] = cone[c];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexInsertCone"
PetscErrorCode DMComplexInsertCone(DM dm, PetscInt p, PetscInt conePos, PetscInt conePoint)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) {SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %d is not in the valid range [%d, %d)", p, pStart, pEnd);}
  if ((conePoint < pStart) || (conePoint >= pEnd)) {SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %d is not in the valid range [%d, %d)", conePoint, pStart, pEnd);}
  if (conePos >= dof) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone position %d is not in the valid range [0, %d)", conePos, dof);}
  mesh->cones[off+conePos] = conePoint;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetSupportSize"
/*@
  DMComplexGetSupportSize - Return the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
- p - The Sieve point, which must lie in the chart set with DMComplexSetChart()

  Output Parameter:
. size - The support size for point p

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetConeSize(), DMComplexSetChart(), DMComplexGetConeSize()
@*/
PetscErrorCode DMComplexGetSupportSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->supportSection, p, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetSupport"
/*@C
  DMComplexGetSupport - Return the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
- p - The Sieve point, which must lie in the chart set with DMComplexSetChart()

  Output Parameter:
. support - An array of points which are on the out-edges for point p

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart(), DMComplexGetCone()
@*/
PetscErrorCode DMComplexGetSupport(DM dm, PetscInt p, const PetscInt *support[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  *support = &mesh->supports[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetTransitiveClosure"
/*@C
  DMComplexGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- useCone - PETSC_TRUE for in-edges,  otherwise use out-edges

  Output Parameters:
+ numPoints - The number of points in the closure
- points - The points

 Note: THIS SHOULD ALSO RETURN THE POINT ORIENTATIONS

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart(), DMComplexGetCone()
@*/
PetscErrorCode DMComplexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, const PetscInt *points[])
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  const PetscInt *tmp;
  PetscInt        tmpSize, t;
  PetscInt        closureSize = 1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->closureTmpA) {
    PetscInt maxSize = PetscMax(mesh->maxConeSize, mesh->maxSupportSize)+1;
    /* ierr = DMComplexGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
    --depth;
    pow(maxSize, depth)+1; */

    ierr = PetscMalloc2(maxSize,PetscInt,&mesh->closureTmpA,maxSize,PetscInt,&mesh->closureTmpB);CHKERRQ(ierr);
  }
  mesh->closureTmpA[0] = p;
  /* This is only 1-level */
  if (useCone) {
    ierr = DMComplexGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(dm, p, &tmp);CHKERRQ(ierr);
  } else {
    ierr = DMComplexGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &tmp);CHKERRQ(ierr);
  }
  for(t = 0; t < tmpSize; ++t) {
    mesh->closureTmpA[closureSize++] = tmp[t];
  }
  if (numPoints) *numPoints = closureSize;
  if (points)    *points    = mesh->closureTmpA;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetMaxSizes"
/*@
  DMComplexGetMaxSizes - Return the maximum number of in-edges (cone) and out-edges (support) for any point in the Sieve DAG

  Not collective

  Input Parameter:
. mesh - The DMComplex

  Output Parameters:
+ maxConeSize - The maximum number of in-edges
- maxSupportSize - The maximum number of out-edges

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetConeSize(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexGetMaxSizes(DM dm, PetscInt *maxConeSize, PetscInt *maxSupportSize)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (maxConeSize)    *maxConeSize    = mesh->maxConeSize;
  if (maxSupportSize) *maxSupportSize = mesh->maxSupportSize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Complex"
PetscErrorCode DMSetUp_Complex(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetUp(mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coneSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(PetscInt), &mesh->cones);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSymmetrize"
/*@
  DMComplexSymmetrize - Creates support (out-edge) information from cone (in-edge) inoformation

  Not collective

  Input Parameter:
. mesh - The DMComplex

  Output Parameter:

  Note:
  This should be called after all calls to DMComplexSetCone()

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetChart(), DMComplexSetConeSize(), DMComplexSetCone()
@*/
PetscErrorCode DMComplexSymmetrize(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt      *offsets;
  PetscInt       supportSize;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* Calculate support sizes */
  ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(mesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for(c = off; c < off+dof; ++c) {
      ierr = PetscSectionAddDof(mesh->supportSection, mesh->cones[c], 1);CHKERRQ(ierr);
    }
  }
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
    mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, dof);
  }
  ierr = PetscSectionSetUp(mesh->supportSection);CHKERRQ(ierr);
  /* Calculate supports */
  ierr = PetscSectionGetStorageSize(mesh->supportSection, &supportSize);CHKERRQ(ierr);
  ierr = PetscMalloc(supportSize * sizeof(PetscInt), &mesh->supports);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &offsets);CHKERRQ(ierr);
  ierr = PetscMemzero(offsets, (pEnd - pStart) * sizeof(PetscInt));CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for(c = off; c < off+dof; ++c) {
      const PetscInt q = mesh->cones[c];
      PetscInt       offS;

      ierr = PetscSectionGetOffset(mesh->supportSection, q, &offS);CHKERRQ(ierr);
      mesh->supports[offS+offsets[q]] = p;
      ++offsets[q];
    }
  }
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexStratify"
/*@
  DMComplexStratify - The Sieve DAG for most topologies is a graded poset (http://en.wikipedia.org/wiki/Graded_poset), and
  can be illustrated by Hasse Diagram (a http://en.wikipedia.org/wiki/Hasse_diagram). The strata group all points of the
  same grade, and this function calculates the strata. This grade can be seen as the height (or depth) of the point in
  the DAG.

  Not collective

  Input Parameter:
. mesh - The DMComplex

  Output Parameter:

  Notes:
  The normal association for the point grade is element dimension (or co-dimension). For instance, all vertices would
  have depth 0, and all edges depth 1. Likewise, all cells heights would have height 0, and all faces height 1.

  This should be called after all calls to DMComplexSymmetrize()

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSymmetrize()
@*/
PetscErrorCode DMComplexStratify(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd, p;
  PetscInt       numRoots = 0, numLeaves = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* Calculate depth */
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  /* Initialize roots and count leaves */
  for(p = pStart; p < pEnd; ++p) {
    PetscInt coneSize, supportSize;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &coneSize);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(mesh->supportSection, p, &supportSize);CHKERRQ(ierr);
    if (!coneSize && supportSize) {
      ++numRoots;
      ierr = DMComplexSetLabelValue(dm, "depth", p, 0);CHKERRQ(ierr);
    } else if (!supportSize && coneSize) {
      ++numLeaves;
    }
  }
  if (numRoots + numLeaves == (pEnd - pStart)) {
    for(p = pStart; p < pEnd; ++p) {
      PetscInt coneSize, supportSize;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &coneSize);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(mesh->supportSection, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        ierr = DMComplexSetLabelValue(dm, "depth", p, 1);CHKERRQ(ierr);
      }
    }
  } else {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Have not yet coded general stratification");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLabelValue"
/*@C
  DMComplexGetLabelValue - Get the value in a Sieve Label for the given point, with 0 as the default

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
. name - The label name
- point - The mesh point

  Output Parameter:
. value - The label value for this point, or 0 if the point is not in the label

  Level: beginner

.keywords: mesh
.seealso: DMComplexSetLabelValue(), DMComplexGetLabelStratum()
@*/
PetscErrorCode DMComplexGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  *value = 0;
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);CHKERRQ(ierr);}
  /* Find, or add, label value */
  for(v = 0; v < next->numStrata; ++v) {
    for(p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
      if (next->points[p] == point) {
        *value = next->stratumValues[v];
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetLabelValue"
/*@C
  DMComplexSetLabelValue - Add a point to a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMComplexGetLabelStratum()
@*/
PetscErrorCode DMComplexSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  /* Find, or create, label */
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {
    SieveLabel tmpLabel = mesh->labels;
    ierr = PetscNew(struct Sieve_Label, &mesh->labels);CHKERRQ(ierr);
    mesh->labels->next = tmpLabel;
    next = mesh->labels;
    ierr = PetscStrallocpy(name, &next->name);CHKERRQ(ierr);
  }
  /* Find, or add, label value */
  for(v = 0; v < next->numStrata; ++v) {
    if (next->stratumValues[v] == value) break;
  }
  if (v >= next->numStrata) {
    PetscInt *tmpV, *tmpO, *tmpS;
    ierr = PetscMalloc3(next->numStrata+1,PetscInt,&tmpV,next->numStrata+2,PetscInt,&tmpO,next->numStrata+1,PetscInt,&tmpS);CHKERRQ(ierr);
    for(v = 0; v < next->numStrata; ++v) {
      tmpV[v] = next->stratumValues[v];
      tmpO[v] = next->stratumOffsets[v];
      tmpS[v] = next->stratumSizes[v];
    }
    tmpV[v] = value;
    tmpO[v] = v == 0 ? 0 : next->stratumOffsets[v];
    tmpS[v] = 0;
    tmpO[v+1] = tmpO[v];
    ++next->numStrata;
    ierr = PetscFree3(next->stratumValues,next->stratumOffsets,next->stratumSizes);CHKERRQ(ierr);
    next->stratumValues  = tmpV;
    next->stratumOffsets = tmpO;
    next->stratumSizes   = tmpS;
  }
  /* Check whether point exists */
  for(p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
    if (next->points[p] == point) {
      break;
    }
  }
  /* Add point: NEED TO OPTIMIZE */
  if (p >= next->stratumOffsets[v]+next->stratumSizes[v]) {
    /* Check for reallocation */
    if (next->stratumSizes[v] >= next->stratumOffsets[v+1]-next->stratumOffsets[v]) {
      PetscInt  oldSize   = next->stratumOffsets[v+1]-next->stratumOffsets[v];
      PetscInt  newSize   = PetscMax(10, 2*oldSize); /* Double the size, since 2 is the optimal base for this online algorithm */
      PetscInt  shift     = newSize - oldSize;
      PetscInt  allocSize = next->stratumOffsets[next->numStrata] + shift;
      PetscInt *newPoints;
      PetscInt  w, q;

      ierr = PetscMalloc(allocSize * sizeof(PetscInt), &newPoints);CHKERRQ(ierr);
      for(q = 0; q < next->stratumOffsets[v]+next->stratumSizes[v]; ++q) {
        newPoints[q] = next->points[q];
      }
      for(w = v+1; w < next->numStrata; ++w) {
        for(q = next->stratumOffsets[w]; q < next->stratumOffsets[w]+next->stratumSizes[w]; ++q) {
          newPoints[q+shift] = next->points[q];
        }
        next->stratumOffsets[w] += shift;
      }
      next->stratumOffsets[next->numStrata] += shift;
      ierr = PetscFree(next->points);CHKERRQ(ierr);
      next->points = newPoints;
    }
    /* Insert point and resort */
    next->points[next->stratumOffsets[v]+next->stratumSizes[v]] = point;
    ++next->stratumSizes[v];
    ierr = PetscSortInt(next->stratumSizes[v], &next->points[next->stratumOffsets[v]]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLabelSize"
/*@C
  DMComplexGetLabelSize - Get the number of different integer ids in a Label

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
- name - The label name

  Output Parameter:
. size - The label size (number of different integer ids)

  Level: beginner

.keywords: mesh
.seealso: DMComplexSetLabelValue()
@*/
PetscErrorCode DMComplexGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 3);
  *size = 0;
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) {
      *size = next->numStrata;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLabelIdIS"
/*@C
  DMComplexGetLabelIdIS - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The DMComplex object
- name - The label name

  Output Parameter:
. ids - The integer ids

  Level: beginner

.keywords: mesh
.seealso: DMComplexGetLabelSize()
@*/
PetscErrorCode DMComplexGetLabelIdIS(DM dm, const char name[], IS *ids)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscInt      *values;
  PetscInt       size, i = 0;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ids, 3);
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) {
      size = next->numStrata;
      ierr = PetscMalloc(size * sizeof(PetscInt), &values);CHKERRQ(ierr);
      for(i = 0; i < next->numStrata; ++i) {
        values[i] = next->stratumValues[i];
      }
      break;
    }
    next = next->next;
  }
  ierr = ISCreateGeneral(((PetscObject) dm)->comm, size, values, PETSC_OWN_POINTER, ids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetStratumSize"
/*@C
  DMComplexGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The stratum size

  Level: beginner

.keywords: mesh
.seealso: DMComplexGetLabelSize(), DMComplexGetLabelIds()
@*/
PetscErrorCode DMComplexGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 4);
  *size = 0;
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) {
      PetscInt v;

      for(v = 0; v < next->numStrata; ++v) {
        if (next->stratumValues[v] == value) {
          *size = next->stratumSizes[v];
          break;
        }
      }
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetStratumIS"
/*@C
  DMComplexGetStratumIS - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. name - The label name
- value - The stratum value

  Output Parameter:
. is - The stratum points

  Level: beginner

.keywords: mesh
.seealso: DMComplexGetStratumSize()
@*/
PetscErrorCode DMComplexGetStratumIS(DM dm, const char name[], PetscInt value, IS *is) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(is, 4);
  *is = PETSC_NULL;
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) {
      PetscInt v;

      for(v = 0; v < next->numStrata; ++v) {
        if (next->stratumValues[v] == value) {
          ierr = ISCreateGeneral(PETSC_COMM_SELF, next->stratumSizes[v], &next->points[next->stratumOffsets[v]], PETSC_COPY_VALUES, is);CHKERRQ(ierr);
          break;
        }
      }
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexJoinPoints"
/* This is a 1-level join */
PetscErrorCode DMComplexJoinPoints(DM dm, const PetscInt points[], PetscInt *coveredPoint)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(coveredPoint, 3);
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not yet supported");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexMeetPoints"
/* This is a 1-level meet */
PetscErrorCode DMComplexMeetPoints(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveringPoints, const PetscInt **coveringPoints)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt      *meet[2];
  PetscInt       meetSize, i = 0;
  PetscInt       dof, off, p, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveringPoints, 3);
  PetscValidPointer(coveringPoints, 4);
  if (!mesh->meetTmpA) {ierr = PetscMalloc2(mesh->maxConeSize,PetscInt,&mesh->meetTmpA,mesh->maxConeSize,PetscInt,&mesh->meetTmpB);CHKERRQ(ierr);}
  meet[0] = mesh->meetTmpA; meet[1] = mesh->meetTmpB;
  /* Copy in cone of first point */
  ierr = PetscSectionGetDof(mesh->coneSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, points[0], &off);CHKERRQ(ierr);
  for(meetSize = 0; meetSize < dof; ++meetSize) {
    meet[i][meetSize] = mesh->cones[off+meetSize];
  }
  /* Check each successive cone */
  for(p = 1; p < numPoints; ++p) {
    PetscInt newMeetSize = 0;

    ierr = PetscSectionGetDof(mesh->coneSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, points[p], &off);CHKERRQ(ierr);
    for(c = 0; c < dof; ++c) {
      const PetscInt point = mesh->cones[off+c];

      for(m = 0; m < meetSize; ++m) {
        if (point == meet[i][m]) {
          meet[1-i][newMeetSize++] = point;
          break;
        }
      }
    }
    meetSize = newMeetSize;
    i = 1-i;
  }
  *numCoveringPoints = meetSize;
  *coveringPoints    = meet[1-i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateNeighborCSR"
PetscErrorCode DMComplexCreateNeighborCSR(DM dm, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency) {
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[30]; /* maxFaceCases, C89 sucks sucks sucks */
  PetscInt      *off, *adj;
  PetscInt       dim, depth, cStart, cEnd, c, numCells, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
  --depth;
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (cEnd - cStart == 0) {
    if (numVertices) *numVertices = 0;
    if (offsets)     *offsets     = PETSC_NULL;
    if (adjacency)   *adjacency   = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  numCells = cEnd - cStart;
  /* Setup face recognition */
  {
    PetscInt cornersSeen[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* Could use PetscBT */

    for(c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      ierr = DMComplexGetConeSize(dm, c, &corners);CHKERRQ(ierr);
      if (!cornersSeen[corners]) {
        if (numFaceCases >= maxFaceCases) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");}
        cornersSeen[corners] = 1;
        if (corners == dim+1) {
          numFaceVertices[numFaceCases] = dim;
          PetscInfo(dm, "Recognizing simplices\n");
        } else if ((dim == 1) && (corners == 3)) {
          numFaceVertices[numFaceCases] = 3;
          PetscInfo(dm, "Recognizing quadratic edges\n");
        } else if ((dim == 2) && (corners == 4)) {
          numFaceVertices[numFaceCases] = 2;
          PetscInfo(dm, "Recognizing quads\n");
        } else if ((dim == 2) && (corners == 6)) {
          numFaceVertices[numFaceCases] = 3;
          PetscInfo(dm, "Recognizing tri and quad cohesive Lagrange cells\n");
        } else if ((dim == 2) && (corners == 9)) {
          numFaceVertices[numFaceCases] = 3;
          PetscInfo(dm, "Recognizing quadratic quads and quadratic quad cohesive Lagrange cells\n");
        } else if ((dim == 3) && (corners == 6)) {
          numFaceVertices[numFaceCases] = 4;
          PetscInfo(dm, "Recognizing tet cohesive cells\n");
        } else if ((dim == 3) && (corners == 8)) {
          numFaceVertices[numFaceCases] = 4;
          PetscInfo(dm, "Recognizing hexes\n");
        } else if ((dim == 3) && (corners == 9)) {
          numFaceVertices[numFaceCases] = 6;
          PetscInfo(dm, "Recognizing tet cohesive Lagrange cells\n");
        } else if ((dim == 3) && (corners == 10)) {
          numFaceVertices[numFaceCases] = 6;
          PetscInfo(dm, "Recognizing quadratic tets\n");
        } else if ((dim == 3) && (corners == 12)) {
          numFaceVertices[numFaceCases] = 6;
          PetscInfo(dm, "Recognizing hex cohesive Lagrange cells\n");
        } else if ((dim == 3) && (corners == 18)) {
          numFaceVertices[numFaceCases] = 6;
          PetscInfo(dm, "Recognizing quadratic tet cohesive Lagrange cells\n");
        } else if ((dim == 3) && (corners == 27)) {
          numFaceVertices[numFaceCases] = 9;
          PetscInfo(dm, "Recognizing quadratic hexes and quadratic hex cohesive Lagrange cells\n");
        } else {
          SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not recognize number of face vertices for %d corners", corners);
        }
        ++numFaceCases;
      }
    }
  }
  /* Check for optimized depth 1 construction */
  ierr = PetscMalloc((numCells+1) * sizeof(PetscInt), &off);CHKERRQ(ierr);
  ierr = PetscMemzero(off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
  if (depth == 1) {
    PetscInt *neighborCells;
    PetscInt  n;
    PetscInt  maxConeSize, maxSupportSize;

    /* Temp space for point adj <= maxConeSize*maxSupportSize */
    ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc(maxConeSize*maxSupportSize * sizeof(PetscInt), &neighborCells);CHKERRQ(ierr);
    /* Count neighboring cells */
    for(cell = cStart; cell < cEnd; ++cell) {
      const PetscInt *cone;
      PetscInt        numNeighbors = 0;
      PetscInt        coneSize, c;

      /* Get support of the cone, and make a set of the cells */
      ierr = DMComplexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMComplexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
        ierr = DMComplexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
        for(s = 0; s < supportSize; ++s) {
          const PetscInt point = support[s];

          if (point == cell) continue;
          for(n = 0; n < numNeighbors; ++n) {
            if (neighborCells[n] == point) break;
          }
          if (n == numNeighbors) {
            neighborCells[n] = point;
            ++numNeighbors;
          }
        }
      }
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for(n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2] = {cell, neighborCells[n]};
        PetscInt        meetSize;
        const PetscInt *meet;

        ierr = DMComplexMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for(f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              ++off[cell-cStart+1];
              break;
            }
          }
        }
      }
    }
    /* Prefix sum */
    for(cell = 1; cell <= numCells; ++cell) {
      off[cell] += off[cell-1];
    }
    if (adjacency) {
      ierr = PetscMalloc(off[numCells] * sizeof(PetscInt), &adj);CHKERRQ(ierr);
      /* Get neighboring cells */
      for(cell = cStart; cell < cEnd; ++cell) {
        const PetscInt *cone;
        PetscInt        numNeighbors = 0;
        PetscInt        cellOffset   = 0;
        PetscInt        coneSize, c;

        /* Get support of the cone, and make a set of the cells */
        ierr = DMComplexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(dm, cell, &cone);CHKERRQ(ierr);
        for(c = 0; c < coneSize; ++c) {
          const PetscInt *support;
          PetscInt        supportSize, s;

          ierr = DMComplexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
          ierr = DMComplexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
          for(s = 0; s < supportSize; ++s) {
            const PetscInt point = support[s];

            if (point == cell) continue;
            for(n = 0; n < numNeighbors; ++n) {
              if (neighborCells[n] == point) break;
            }
            if (n == numNeighbors) {
              neighborCells[n] = point;
              ++numNeighbors;
            }
          }
        }
        /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
        for(n = 0; n < numNeighbors; ++n) {
          PetscInt        cellPair[2] = {cell, neighborCells[n]};
          PetscInt        meetSize;
          const PetscInt *meet;

          ierr = DMComplexMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for(f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                adj[off[cell-cStart]+cellOffset] = neighborCells[n];
                ++cellOffset;
                break;
              }
            }
          }
        }
      }
    }
    ierr = PetscFree(neighborCells);CHKERRQ(ierr);
  } else if (depth == dim) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Neighbor graph creation not implemented for interpolated meshes");
#if 0
    OffsetVisitor<typename Mesh::sieve_type> oV(*sieve, *overlapSieve, off);
    PetscInt p;

    for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
      sieve->cone(*c_iter, oV);
    }
    for(p = 1; p <= numCells; ++p) {
      off[p] = off[p] + off[p-1];
    }
    ierr = PetscMalloc(off[numCells] * sizeof(PetscInt), &adj);CHKERRQ(ierr);
    AdjVisitor<typename Mesh::sieve_type> aV(adj, zeroBase);
    ISieveVisitor::SupportVisitor<typename Mesh::sieve_type, AdjVisitor<typename Mesh::sieve_type> > sV(*sieve, aV);
    ISieveVisitor::SupportVisitor<typename Mesh::sieve_type, AdjVisitor<typename Mesh::sieve_type> > ovSV(*overlapSieve, aV);

    for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
      aV.setCell(*c_iter);
      sieve->cone(*c_iter, sV);
      sieve->cone(*c_iter, ovSV);
    }
    offset = aV.getOffset();
#endif
  } else {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Neighbor graph creation not defined for partially interpolated meshes");
  }
  if (numVertices) *numVertices = numCells;
  if (offsets)     *offsets     = off;
  if (adjacency)   *adjacency   = adj;
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_CHACO
#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif
/* Chaco does not have an include file */
PETSC_EXTERN_C int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

extern int FREE_GRAPH;

#undef __FUNCT__
#define __FUNCT__ "DMComplexPartition_Chaco"
PetscErrorCode DMComplexPartition_Chaco(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
  MPI_Comm comm = ((PetscObject) dm)->comm;
  int nvtxs = numVertices;                /* number of vertices in full graph */
  int *vwgts = NULL;                      /* weights for all vertices */
  float *ewgts = NULL;                    /* weights for all edges */
  float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
  char *outassignname = NULL;             /*  name of assignment output file */
  char *outfilename = NULL;               /* output file name */
  int architecture = 1;                   /* 0 => hypercube, d => d-dimensional mesh */
  int ndims_tot = 0;                      /* total number of cube dimensions to divide */
  int mesh_dims[3];                       /* dimensions of mesh of processors */
  double *goal = NULL;                    /* desired set sizes for each set */
  int global_method = 1;                  /* global partitioning algorithm */
  int local_method = 1;                   /* local partitioning algorithm */
  int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
  int vmax = 200;                         /* how many vertices to coarsen down to? */
  int ndims = 1;                          /* number of eigenvectors (2^d sets) */
  double eigtol = 0.001;                  /* tolerance on eigenvectors */
  long seed = 123636512;                  /* for random graph mutations */
  short int *assignment;                  /* Output partition */
  int fd_stdout, fd_pipe[2];
  PetscInt      *points;
  PetscMPIInt    commSize;
  int            i, v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize);CHKERRQ(ierr);
  if (!numVertices) {
    ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, 0, PETSC_NULL, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
  for(i = 0; i < start[numVertices]; ++i) {
    ++adjacency[i];
  }
  if (global_method == INERTIAL_METHOD) {
    /* manager.createCellCoordinates(nvtxs, &x, &y, &z); */
    SETERRQ(comm, PETSC_ERR_SUP, "Inertial partitioning not yet supported");
  }
  mesh_dims[0] = commSize;
  mesh_dims[1] = 1;
  mesh_dims[2] = 1;
  ierr = PetscMalloc(nvtxs * sizeof(short int), &assignment);CHKERRQ(ierr);
  /* Chaco outputs to stdout. We redirect this to a buffer. */
  /* TODO: check error codes for UNIX calls */
#ifdef PETSC_HAVE_UNISTD_H
  {
    fd_stdout = dup(1);
    pipe(fd_pipe);
    close(1);
    dup2(fd_pipe[1], 1);
  }
#endif
  ierr = interface(nvtxs, (int *) start, (int *) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                   assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                   vmax, ndims, eigtol, seed);
#ifdef PETSC_HAVE_UNISTD_H
  {
    char msgLog[10000];
    int  count;

    fflush(stdout);
    count = read(fd_pipe[0], msgLog, (10000-1)*sizeof(char));
    if (count < 0) count = 0;
    msgLog[count] = 0;
    close(1);
    dup2(fd_stdout, 1);
    close(fd_stdout);
    close(fd_pipe[0]);
    close(fd_pipe[1]);
    if (ierr) {SETERRQ1(comm, PETSC_ERR_LIB, "Error in Chaco library: %s", msgLog);}
  }
#endif
  /* Convert to PetscSection+IS */
  ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
  for(v = 0; v < nvtxs; ++v) {
    ierr = PetscSectionAddDof(*partSection, assignment[v], 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc(nvtxs * sizeof(PetscInt), &points);CHKERRQ(ierr);
  for(p = 0, i = 0; p < commSize; ++p) {
    for(v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) {SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %d should be %d", i, nvtxs);}
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  if (global_method == INERTIAL_METHOD) {
    /* manager.destroyCellCoordinates(nvtxs, &x, &y, &z); */
  }
  ierr = PetscFree(assignment);CHKERRQ(ierr);
  for(i = 0; i < start[numVertices]; ++i) {
    --adjacency[i];
  }
  PetscFunctionReturn(0);
}
#endif

#ifdef PETSC_HAVE_PARMETIS
#undef __FUNCT__
#define __FUNCT__ "DMComplexPartition_ParMetis"
PetscErrorCode DMComplexPartition_ParMetis(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreatePartition"
PetscErrorCode DMComplexCreatePartition(DM dm, PetscSection *partSection, IS *partition, PetscInt height) {
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject) dm)->comm, &size);CHKERRQ(ierr);
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscSectionCreate(((PetscObject) dm)->comm, partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, size);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*partSection, 0, cEnd-cStart);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = PetscMalloc((cEnd - cStart) * sizeof(PetscInt), &points);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c) {
      points[c] = c;
    }
    ierr = ISCreateGeneral(((PetscObject) dm)->comm, cEnd-cStart, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (height == 0) {
    PetscInt  numVertices;
    PetscInt *start     = PETSC_NULL;
    PetscInt *adjacency = PETSC_NULL;

    if (1) {
      ierr = DMComplexCreateNeighborCSR(dm, &numVertices, &start, &adjacency);CHKERRQ(ierr);
#ifdef PETSC_HAVE_CHACO
      ierr = DMComplexPartition_Chaco(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#endif
    } else {
      ierr = DMComplexCreateNeighborCSR(dm, &numVertices, &start, &adjacency);CHKERRQ(ierr);
#ifdef PETSC_HAVE_PARMETIS
      ierr = DMComplexPartition_ParMetis(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#endif
    }
    ierr = PetscFree(start);CHKERRQ(ierr);
    ierr = PetscFree(adjacency);CHKERRQ(ierr);
# if 0
  } else if (height == 1) {
    /* Build the dual graph for faces and partition the hypergraph */
    PetscInt numEdges;

    buildFaceCSRV(mesh, mesh->getFactory()->getNumbering(mesh, mesh->depth()-1), &numEdges, &start, &adjacency, GraphPartitioner::zeroBase());
    GraphPartitioner().partition(numEdges, start, adjacency, partition, manager);
    destroyCSR(numEdges, start, adjacency);
#endif
  } else {
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid partition height %d", height);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreatePartitionClosure"
PetscErrorCode DMComplexCreatePartitionClosure(DM dm, PetscSection pointSection, IS pointPartition, PetscSection *section, IS *partition) {
  /* const PetscInt  height = 0; */
  const PetscInt *partArray;
  PetscInt       *allPoints, *partPoints = PETSC_NULL;
  PetscInt        rStart, rEnd, rank, maxPartSize = 0, newSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(pointSection, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, rStart, rEnd);CHKERRQ(ierr);
  for(rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0;
    PetscInt numPoints, offset, p;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      PetscInt        point = partArray[offset+p], closureSize;
      const PetscInt *closure;

      /* TODO Include support for height > 0 case */
      ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Merge into existing points */
      if (partSize+closureSize > maxPartSize) {
        PetscInt *tmpPoints;

        maxPartSize = PetscMax(partSize+closureSize, 2*maxPartSize);
        ierr = PetscMalloc(maxPartSize * sizeof(PetscInt), &tmpPoints);CHKERRQ(ierr);
        ierr = PetscMemcpy(tmpPoints, partPoints, partSize * sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscFree(partPoints);CHKERRQ(ierr);
        partPoints = tmpPoints;
      }
      ierr = PetscMemcpy(&partPoints[partSize], closure, closureSize * sizeof(PetscInt));CHKERRQ(ierr);
      partSize += closureSize;
      ierr = PetscSortRemoveDupsInt(&partSize, partPoints);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(*section, rank, partSize);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(*section, &newSize);CHKERRQ(ierr);
  ierr = PetscMalloc(newSize * sizeof(PetscInt), &allPoints);CHKERRQ(ierr);

  for(rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0, newOffset;
    PetscInt numPoints, offset, p;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      PetscInt        point = partArray[offset+p], closureSize;
      const PetscInt *closure;

      /* TODO Include support for height > 0 case */
      ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Merge into existing points */
      ierr = PetscMemcpy(&partPoints[partSize], closure, closureSize * sizeof(PetscInt));CHKERRQ(ierr);
      partSize += closureSize;
      ierr = PetscSortRemoveDupsInt(&partSize, partPoints);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetOffset(*section, rank, &newOffset);CHKERRQ(ierr);
    ierr = PetscMemcpy(&allPoints[newOffset], partPoints, partSize * sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = PetscFree(partPoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(((PetscObject) dm)->comm, newSize, allPoints, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexDistributeField"
/*
  Input Parameters:
. originalSection
, originalVec

  Output Parameters:
. newSection
. newVec
*/
PetscErrorCode DMComplexDistributeField(DM dm, PetscSF pointSF, PetscSection originalSection, Vec originalVec, PetscSection newSection, Vec newVec)
{
  PetscSF         fieldSF;
  PetscInt       *remoteOffsets, fieldSize;
  PetscScalar    *originalValues, *newValues;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newVec, fieldSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(newVec);CHKERRQ(ierr);

  ierr = VecGetArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = VecGetArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = VecRestoreArray(originalVec, &originalValues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexDistribute"
/*@C
  DMComplexDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ dm  - The original DMComplex object
- partitioner - The partitioning package, or NULL for the default

  Output Parameter:
. parallelMesh - The distributed DMComplex object

  Level: intermediate

.keywords: mesh, elements

.seealso: DMComplexCreate(), DMComplexDistributeByFace()
@*/
PetscErrorCode DMComplexDistribute(DM dm, const char partitioner[], DM *dmParallel)
{
  DM_Complex    *mesh   = (DM_Complex *) dm->data, *pmesh;
  MPI_Comm       comm   = ((PetscObject) dm)->comm;
  const PetscInt height = 0;
  PetscInt       dim, numRemoteRanks;
  IS             cellPart,        part;
  PetscSection   cellPartSection, partSection;
  PetscSFNode   *remoteRanks;
  PetscSF        partSF, pointSF, coneSF;
  ISLocalToGlobalMapping renumbering;
  PetscSection   originalConeSection, newConeSection;
  PetscInt      *remoteOffsets;
  PetscInt      *cones, *newCones, newConesSize;
  PetscBool      flg;
  PetscMPIInt    rank, numProcs, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel,3);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  if (numProcs == 1) PetscFunctionReturn(0);

  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = DMComplexCreatePartition(dm, &cellPartSection, &cellPart, height);CHKERRQ(ierr);
  /* Create SF assuming a serial partition for all processes: Could check for IS length here */
  if (!rank) {
    numRemoteRanks = numProcs;
  } else {
    numRemoteRanks = 0;
  }
  ierr = PetscMalloc(numRemoteRanks * sizeof(PetscSFNode), &remoteRanks);CHKERRQ(ierr);
  for(p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, PETSC_NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscSFView(partSF, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMComplexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr = DMComplexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dmParallel, dim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
  pmesh = (DM_Complex *) (*dmParallel)->data;
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, &pointSF);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(part, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscSFView(pointSF, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(renumbering, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Distribute cone section */
  ierr = DMComplexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMComplexGetConeSection(*dmParallel, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = DMSetUp(*dmParallel);CHKERRQ(ierr);
  {
    PetscInt pStart, pEnd, p;

    ierr = PetscSectionGetChart(newConeSection, &pStart, &pEnd);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt coneSize;
      ierr = PetscSectionGetDof(newConeSection, p, &coneSize);CHKERRQ(ierr);
      pmesh->maxConeSize = PetscMax(pmesh->maxConeSize, coneSize);
    }
  }
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(pointSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
  ierr = DMComplexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMComplexGetCones(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newConesSize, newCones, PETSC_NULL, newCones);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFView(coneSF, PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  ierr = DMComplexSymmetrize(*dmParallel);CHKERRQ(ierr);
  ierr = DMComplexStratify(*dmParallel);CHKERRQ(ierr);
  /* Distribute Coordinates */
  {
    PetscSection originalCoordSection, newCoordSection;
    Vec          originalCoordinates, newCoordinates;

    ierr = DMComplexGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateVec(dm, &originalCoordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateVec(*dmParallel, &newCoordinates);CHKERRQ(ierr);

    ierr = DMComplexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
  }
  /* Distribute labels */
  {
    SieveLabel next      = mesh->labels, newNext = PETSC_NULL;
    PetscInt   numLabels = 0, l;

    /* Bcast number of labels */
    while(next) {++numLabels; next = next->next;}
    ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    next = mesh->labels;
    for(l = 0; l < numLabels; ++l) {
      SieveLabel      newLabel;
      const PetscInt *partArray;
      PetscInt       *stratumSizes = PETSC_NULL, *points = PETSC_NULL;
      PetscMPIInt    *sendcnts = PETSC_NULL, *offsets = PETSC_NULL, *displs = PETSC_NULL;
      PetscInt        nameSize, s, p;
      size_t          len = 0;

      ierr = PetscNew(struct Sieve_Label, &newLabel);CHKERRQ(ierr);
      /* Bcast name (could filter for no points) */
      if (!rank) {ierr = PetscStrlen(next->name, &len);CHKERRQ(ierr);}
      nameSize = len;
      ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscMalloc(nameSize+1, &newLabel->name);CHKERRQ(ierr);
      if (!rank) {ierr = PetscMemcpy(newLabel->name, next->name, nameSize+1);CHKERRQ(ierr);}
      ierr = MPI_Bcast(newLabel->name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
      /* Bcast numStrata (could filter for no points in stratum) */
      if (!rank) {newLabel->numStrata = next->numStrata;}
      ierr = MPI_Bcast(&newLabel->numStrata, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscMalloc(newLabel->numStrata * sizeof(PetscInt), &newLabel->stratumValues);CHKERRQ(ierr);
      ierr = PetscMalloc(newLabel->numStrata * sizeof(PetscInt), &newLabel->stratumSizes);CHKERRQ(ierr);
      ierr = PetscMalloc((newLabel->numStrata+1) * sizeof(PetscInt), &newLabel->stratumOffsets);CHKERRQ(ierr);
      /* Bcast stratumValues (could filter for no points in stratum) */
      if (!rank) {ierr = PetscMemcpy(newLabel->stratumValues, next->stratumValues, next->numStrata * sizeof(PetscInt));CHKERRQ(ierr);}
      ierr = MPI_Bcast(newLabel->stratumValues, newLabel->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
      /* Find size on each process and Scatter */
      if (!rank) {
        ierr = ISGetIndices(part, &partArray);CHKERRQ(ierr);
        ierr = PetscMalloc(numProcs*next->numStrata * sizeof(PetscInt), &stratumSizes);CHKERRQ(ierr);
        ierr = PetscMemzero(stratumSizes, numProcs*next->numStrata * sizeof(PetscInt));CHKERRQ(ierr);
        for(s = 0; s < next->numStrata; ++s) {
          for(p = next->stratumOffsets[s]; p < next->stratumOffsets[s]+next->stratumSizes[s]; ++p) {
            const PetscInt point = next->points[p];
            PetscInt       proc;

            for(proc = 0; proc < numProcs; ++proc) {
              PetscInt dof, off, pPart;

              ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
              for(pPart = off; pPart < off+dof; ++pPart) {
                if (partArray[pPart] == point) {
                  ++stratumSizes[proc*next->numStrata+s];
                  break;
                }
              }
            }
          }
        }
        ierr = ISRestoreIndices(part, &partArray);CHKERRQ(ierr);
      }
      ierr = MPI_Scatter(stratumSizes, newLabel->numStrata, MPI_INT, newLabel->stratumSizes, newLabel->numStrata, MPI_INT, 0, comm);CHKERRQ(ierr);
      /* Calculate stratumOffsets */
      newLabel->stratumOffsets[0] = 0;
      for(s = 0; s < newLabel->numStrata; ++s) {
        newLabel->stratumOffsets[s+1] = newLabel->stratumSizes[s] + newLabel->stratumOffsets[s];
      }
      /* Pack points and Scatter */
      if (!rank) {
        ierr = PetscMalloc3(numProcs,PetscMPIInt,&sendcnts,numProcs,PetscMPIInt,&offsets,numProcs+1,PetscMPIInt,&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for(p = 0; p < numProcs; ++p) {
          sendcnts[p] = 0;
          for(s = 0; s < next->numStrata; ++s) {
            sendcnts[p] += stratumSizes[p*next->numStrata+s];
          }
          offsets[p]  = displs[p];
          displs[p+1] = displs[p] + sendcnts[p];
        }
        ierr = PetscMalloc(displs[numProcs] * sizeof(PetscInt), &points);CHKERRQ(ierr);
        for(s = 0; s < next->numStrata; ++s) {
          for(p = next->stratumOffsets[s]; p < next->stratumOffsets[s]+next->stratumSizes[s]; ++p) {
            const PetscInt point = next->points[p];
            PetscInt       proc;

            for(proc = 0; proc < numProcs; ++proc) {
              PetscInt dof, off, pPart;

              ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
              for(pPart = off; pPart < off+dof; ++pPart) {
                if (partArray[pPart] == point) {
                  points[offsets[proc]++] = point;
                  break;
                }
              }
            }
          }
        }
      }
      ierr = PetscMalloc(newLabel->stratumOffsets[newLabel->numStrata] * sizeof(PetscInt), &newLabel->points);CHKERRQ(ierr);
      ierr = MPI_Scatterv(points, sendcnts, displs, MPIU_INT, newLabel->points, newLabel->stratumOffsets[newLabel->numStrata], MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscFree(points);CHKERRQ(ierr);
      ierr = PetscFree3(sendcnts,offsets,displs);CHKERRQ(ierr);
      ierr = PetscFree(stratumSizes);CHKERRQ(ierr);
      /* Renumber points */
      ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newLabel->stratumOffsets[newLabel->numStrata], newLabel->points, PETSC_NULL, newLabel->points);CHKERRQ(ierr);
      /* Sort points */
      for(s = 0; s < newLabel->numStrata; ++s) {
        ierr = PetscSortInt(newLabel->stratumSizes[s], &newLabel->points[newLabel->stratumOffsets[s]]);CHKERRQ(ierr);
      }
      /* Insert into list */
      if (newNext) {
        newNext->next = newLabel;
      } else {
        pmesh->labels = newLabel;
      }
      newNext = newLabel;
      if (!rank) {next = next->next;}
    }
  }
  /* Cleanup Partition */
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Create point SF for parallel mesh */
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints;
    PetscInt       *rowners, *lowners, *ghostPoints;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp;

    ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &leaves, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots*2,PetscInt,&rowners,numLeaves*2,PetscInt,&lowners);CHKERRQ(ierr);
    for(p = 0; p < numRoots*2; ++p) {
      rowners[p] = 0;
    }
    for(p = 0; p < numLeaves; ++p) {
      lowners[p*2+0] = rank;
      lowners[p*2+1] = leaves ? leaves[p] : p;
    }
#if 0 /* Why doesn't this datatype work */
    ierr = PetscSFFetchAndOpBegin(pointSF, MPIU_2INT, rowners, lowners, lowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(pointSF, MPIU_2INT, rowners, lowners, lowners, MPI_MAXLOC);CHKERRQ(ierr);
#endif
    ierr = PetscSFFetchAndOpBegin(pointSF, MPI_2INT, rowners, lowners, lowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(pointSF, MPI_2INT, rowners, lowners, lowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for(p = 0; p < numLeaves; ++p) {
      if (lowners[p*2+0] != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscInt),    &ghostPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscSFNode), &remotePoints);CHKERRQ(ierr);
    for(p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p*2+0] != rank) {
        ghostPoints[gp]       = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p*2+0];
        remotePoints[gp].index = lowners[p*2+1];
        ++gp;
      }
    }
    ierr = PetscFree2(rowners,lowners);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(pmesh->sf, PETSC_DETERMINE, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(pmesh->sf);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dmParallel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>

#undef __FUNCT__
#define __FUNCT__ "InitInput_Triangle"
PetscErrorCode InitInput_Triangle(struct triangulateio *inputCtx) {
  PetscFunctionBegin;
  inputCtx->numberofpoints = 0;
  inputCtx->numberofpointattributes = 0;
  inputCtx->pointlist = PETSC_NULL;
  inputCtx->pointattributelist = PETSC_NULL;
  inputCtx->pointmarkerlist = PETSC_NULL;
  inputCtx->numberofsegments = 0;
  inputCtx->segmentlist = PETSC_NULL;
  inputCtx->segmentmarkerlist = PETSC_NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->trianglelist = PETSC_NULL;
  inputCtx->numberofholes = 0;
  inputCtx->holelist = PETSC_NULL;
  inputCtx->numberofregions = 0;
  inputCtx->regionlist = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitOutput_Triangle"
PetscErrorCode InitOutput_Triangle(struct triangulateio *outputCtx) {
  PetscFunctionBegin;
  outputCtx->numberofpoints = 0;
  outputCtx->pointlist = PETSC_NULL;
  outputCtx->pointattributelist = PETSC_NULL;
  outputCtx->pointmarkerlist = PETSC_NULL;
  outputCtx->numberoftriangles = 0;
  outputCtx->trianglelist = PETSC_NULL;
  outputCtx->triangleattributelist = PETSC_NULL;
  outputCtx->neighborlist = PETSC_NULL;
  outputCtx->segmentlist = PETSC_NULL;
  outputCtx->segmentmarkerlist = PETSC_NULL;
  outputCtx->edgelist = PETSC_NULL;
  outputCtx->edgemarkerlist = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FiniOutput_Triangle"
PetscErrorCode FiniOutput_Triangle(struct triangulateio *outputCtx) {
  PetscFunctionBegin;
  free(outputCtx->pointmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGenerate_Triangle"
PetscErrorCode DMComplexGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm             comm = ((PetscObject) boundary)->comm;
  DM_Complex          *bd   = (DM_Complex *) boundary->data;
  PetscInt             dim              = 2;
  const PetscBool      createConvexHull = PETSC_FALSE;
  const PetscBool      constrained      = PETSC_FALSE;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             vStart, vEnd, v, eStart, eEnd, e;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr  = DMComplexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(bd->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(bd->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(bd->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &eStart, &eEnd);CHKERRQ(ierr);
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    ierr = PetscMalloc(in.numberofsegments*2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofsegments   * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
    for(e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;

      ierr = DMComplexGetCone(boundary, e, &cone);CHKERRQ(ierr);
      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;
      ierr = DMComplexGetLabelValue(boundary, "marker", e, &in.segmentmarkerlist[idx]);CHKERRQ(ierr);
    }
  }
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMComplexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
  if (in.numberofholes > 0) {
    ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRQ(ierr);
    for(h = 0; h < in.numberofholes; ++h) {
      for(d = 0; d < dim; ++d) {
        in.holelist[h*dim+d] = holeCoords[h*dim+d];
      }
    }
  }
#endif
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    if (createConvexHull) {
      ierr = PetscStrcat(args, "c");CHKERRQ(ierr);
    }
    if (constrained) {
      ierr = PetscStrcpy(args, "zepDQ");CHKERRQ(ierr);
    }
    triangulate(args, &in, &out, PETSC_NULL);
  }
  ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
  ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.holelist);CHKERRQ(ierr);

  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dm)->data;
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    int           *cells       = out.trianglelist;
    double        *meshCoords  = out.pointlist;
    PetscInt       coordSize, c;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[3] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells};

      ierr = DMComplexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dm);CHKERRQ(ierr);
    if (interpolate) {
      DM        imesh;
      PetscInt *off, *adj, *cellConeSizes;
      PetscInt  firstEdge = numCells+numVertices, numEdges, edge, e;

      /* Count edges using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dm, PETSC_NULL, &off, &adj);CHKERRQ(ierr);
      numEdges = off[numCells]/2;
      /* Account for boundary edges */
      for(c = 0; c < numCells; ++c) {
        numEdges += 3 - (off[c+1] - off[c]);
      }
      /* Create interpolated mesh */
      ierr = DMCreate(comm, &imesh);CHKERRQ(ierr);
      ierr = DMSetType(imesh, DMCOMPLEX);CHKERRQ(ierr);
      ierr = DMComplexSetDimension(imesh, dim);CHKERRQ(ierr);
      ierr = DMComplexSetChart(imesh, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        ierr = DMComplexSetConeSize(imesh, c, numCorners);CHKERRQ(ierr);
      }
      for(e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMComplexSetConeSize(imesh, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(imesh);CHKERRQ(ierr);
      ierr = PetscMalloc(numCells * sizeof(PetscInt), &cellConeSizes);CHKERRQ(ierr);
      for(c = 0, edge = firstEdge; c < numCells; ++c) {
        PetscInt n;

        for(n = off[c]; n < off[c+1]; ++n) {
          PetscInt        cellPair[2] = {c, adj[n]};
          PetscInt        meetSize, e;
          const PetscInt *meet;
          PetscBool       found = PETSC_FALSE;

          ierr = DMComplexMeetPoints(*dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize != 2) {SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have meet of size %d", meetSize);}
          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((meet[0] == cone[0]) && (meet[1] == cone[1])) || ((meet[0] == cone[1]) && (meet[1] == cone[0]))) {found = PETSC_TRUE;}
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, meet);CHKERRQ(ierr);
            ierr = DMComplexInsertCone(imesh, cellPair[0], cellConeSizes[cellPair[0]], edge);CHKERRQ(ierr);
            ierr = DMComplexInsertCone(imesh, cellPair[1], cellConeSizes[cellPair[1]], edge);CHKERRQ(ierr);
            ++edge;
            ++cellConeSizes[cellPair[0]];
            ++cellConeSizes[cellPair[1]];
          }
        }
      }
      ierr = PetscFree(cellConeSizes);CHKERRQ(ierr);
      if (edge != firstEdge+numEdges) {SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %d should be %d", edge-firstEdge, numEdges);}
      /* TODO Need to set cell orientations, get from original mesh */
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = PetscFree(adj);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      ierr = DMSetFromOptions(imesh);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = imesh;
      SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Interpolation (creation of faces and edges) is not yet supported.");
    }
    ierr = PetscSectionSetChart(mesh->coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
    for(v = numCells; v < numCells+numVertices; ++v) {
      ierr = PetscSectionSetDof(mesh->coordSection, v, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      coords[v*dim+0] = meshCoords[v*dim+0];
      coords[v*dim+1] = meshCoords[v*dim+1];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      for(e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          PetscInt       edge;

          ierr = DMComplexJoinPoints(*dm, vertices, &edge);CHKERRQ(ierr); /* 1-level join */
          ierr = DMComplexSetLabelValue(*dm, "marker", edge, out.edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
    }
  }
#if 0 /* Do not currently support holes */
  ierr = DMComplexCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = FiniOutput_Triangle(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMComplexGenerate"
/*@C
  DMComplexGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The DMComplex boundary object
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The DMComplex object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMComplexCreate(), DMRefine()
@*/
PetscErrorCode DMComplexGenerate(DM boundary, PetscBool  interpolate, DM *mesh)
{
#ifdef PETSC_HAVE_TRIANGLE
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(boundary, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(boundary, interpolate, 2);
#ifdef PETSC_HAVE_TRIANGLE
  ierr = DMComplexGenerate_Triangle(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Mesh generation needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetRefinementLimit"
PetscErrorCode DMComplexSetRefinementLimit(DM dm, PetscReal refinementLimit)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementLimit = refinementLimit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetRefinementLimit"
PetscErrorCode DMComplexGetRefinementLimit(DM dm, PetscReal *refinementLimit)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementLimit,  2);
  /* if (mesh->refinementLimit < 0) = getMaxVolume()/2.0; */
  *refinementLimit = mesh->refinementLimit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefine_Complex"
PetscErrorCode DMRefine_Complex(DM dm, MPI_Comm comm, DM *dmRefined)
{
  PetscReal      refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetRefinementLimit(dm, &refinementLimit);CHKERRQ(ierr);
  if (refinementLimit == 0.0) PetscFunctionReturn(0);
  SETERRQ(comm, PETSC_ERR_SUP, "Refinement not yet implemented");
  ierr = DMComplexCreate(comm, dmRefined);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDepthStratum"
PetscErrorCode DMComplexGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (stratumValue < 0) {
    ierr = DMComplexGetChart(dm, start, end);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    PetscInt pStart, pEnd;

    if (start) {*start = 0;}
    if (end)   {*end   = 0;}
    ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart == pEnd) {PetscFunctionReturn(0);}
  }
  while(next) {
    ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);}
  /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
  depth = stratumValue;
  if ((depth < 0) || (depth >= next->numStrata)) {
    if (start) {*start = 0;}
    if (end)   {*end   = 0;}
  } else {
    if (start) {*start = next->points[next->stratumOffsets[depth]];}
    if (end)   {*end   = next->points[next->stratumOffsets[depth]+next->stratumSizes[depth]-1]+1;}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetHeightStratum"
PetscErrorCode DMComplexGetHeightStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (stratumValue < 0) {
    ierr = DMComplexGetChart(dm, start, end);CHKERRQ(ierr);
  } else {
    PetscInt pStart, pEnd;

    if (start) {*start = 0;}
    if (end)   {*end   = 0;}
    ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart == pEnd) {PetscFunctionReturn(0);}
  }
  while(next) {
    ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);}
  /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
  depth = next->stratumValues[next->numStrata-1] - stratumValue;
  if ((depth < 0) || (depth >= next->numStrata)) {
    if (start) {*start = 0;}
    if (end)   {*end   = 0;}
  } else {
    if (start) {*start = next->points[next->stratumOffsets[depth]];}
    if (end)   {*end   = next->points[next->stratumOffsets[depth]+next->stratumSizes[depth]-1]+1;}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSection"
PetscErrorCode DMComplexCreateSection(DM dm, PetscInt dim, PetscInt numFields, PetscInt numComp[], PetscInt numDof[], PetscInt numBC, PetscInt bcField[], IS bcPoints[], PetscSection *section) {
  PetscInt      *numDofTot, *maxConstraints;
  PetscInt       pStart = 0, pEnd = 0;
  PetscInt       p, d, f, bc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(dim+1,PetscInt,&numDofTot,numFields+1,PetscInt,&maxConstraints);CHKERRQ(ierr);
  for(d = 0; d <= dim; ++d) {
    numDofTot[d] = 0;
    for(f = 0; f < numFields; ++f) {
      numDofTot[d] += numDof[f*(dim+1)+d];
    }
  }
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
  if (numFields > 1) {
    ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
    if (numComp) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
      }
    }
  } else {
    numFields = 0;
  }
  ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
  for(d = 0; d <= dim; ++d) {
    ierr = DMComplexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(*section, p, f, numDof[f*(dim+1)+d]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(*section, p, numDofTot[d]);CHKERRQ(ierr);
    }
  }
  if (numBC) {
    for(f = 0; f <= numFields; ++f) {maxConstraints[f] = 0;}
    for(bc = 0; bc < numBC; ++bc) {
      PetscInt        field = 0;
      const PetscInt *idx;
      PetscInt        n, i;

      if (numFields) {field = bcField[bc];}
      ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
      for(i = 0; i < n; ++i) {
        const PetscInt p = idx[i];
        PetscInt       depth, numConst;

        ierr = DMComplexGetLabelValue(dm, "depth", p, &depth);CHKERRQ(ierr);
        numConst              = numDof[field*(dim+1)+depth];
        maxConstraints[field] = PetscMax(maxConstraints[field], numConst);
        if (numFields) {
          ierr = PetscSectionSetFieldConstraintDof(*section, p, field, numConst);CHKERRQ(ierr);
        }
        ierr = PetscSectionAddConstraintDof(*section, p, numConst);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    }
    for(f = 0; f < numFields; ++f) {
      maxConstraints[numFields] += maxConstraints[f];
    }
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  if (maxConstraints[numFields]) {
    PetscInt *indices;

    ierr = PetscMalloc(maxConstraints[numFields] * sizeof(PetscInt), &indices);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(*section, &pStart, &pEnd);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt cDof;

      ierr = PetscSectionGetConstraintDof(*section, p, &cDof);CHKERRQ(ierr);
      if (cDof) {
        if (cDof > maxConstraints[numFields]) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB, "Likely memory corruption, point %d cDof %d > maxConstraints %d", p, cDof, maxConstraints);}
        if (numFields) {
          PetscInt numConst = 0, fOff = 0;

          for(f = 0; f < numFields; ++f) {
            PetscInt cfDof, fDof;

            ierr = PetscSectionGetFieldDof(*section, p, f, &fDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldConstraintDof(*section, p, f, &cfDof);CHKERRQ(ierr);
            for(d = 0; d < cfDof; ++d) {
              indices[numConst+d] = fOff+d;
            }
            ierr = PetscSectionSetFieldConstraintIndices(*section, p, f, &indices[numConst]);CHKERRQ(ierr);
            numConst += cfDof;
            fOff     += fDof;
          }
          if (cDof != numConst) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of field constraints %d should be %d", numConst, cDof);}
        } else {
          for(d = 0; d < cDof; ++d) {
            indices[d] = d;
          }
        }
        ierr = PetscSectionSetConstraintIndices(*section, p, indices);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  ierr = PetscFree2(numDofTot,maxConstraints);CHKERRQ(ierr);
  {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-section_view", &view);CHKERRQ(ierr);
    if (view) {
      ierr = PetscSectionView(*section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDefaultSection"
/*
  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.
*/
PetscErrorCode DMComplexGetDefaultSection(DM dm, PetscSection *section) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  *section = mesh->defaultSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetDefaultSection"
PetscErrorCode DMComplexSetDefaultSection(DM dm, PetscSection section) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionDestroy(&mesh->defaultSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->defaultGlobalSection);CHKERRQ(ierr);
  mesh->defaultSection = section;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDefaultGlobalSection"
/*
  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.
*/
PetscErrorCode DMComplexGetDefaultGlobalSection(DM dm, PetscSection *section) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!mesh->defaultGlobalSection) {
    ierr = PetscSectionCreateGlobalSection(mesh->defaultSection, mesh->sf, &mesh->defaultGlobalSection);CHKERRQ(ierr);
  }
  *section = mesh->defaultGlobalSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDefaultSF"
/*
  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.
*/
PetscErrorCode DMComplexGetDefaultSF(DM dm, PetscSF *sf) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  ierr = PetscSFGetGraph(mesh->sfDefault, &nroots, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (nroots < 0) {
    ierr = DMComplexCreateDefaultSF(dm);CHKERRQ(ierr);
  }
  *sf = mesh->sfDefault;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCoordinateSection"
PetscErrorCode DMComplexGetCoordinateSection(DM dm, PetscSection *section) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  *section = mesh->coordSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetCoordinateSection"
PetscErrorCode DMComplexSetCoordinateSection(DM dm, PetscSection section) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionDestroy(&mesh->coordSection);CHKERRQ(ierr);
  mesh->coordSection = section;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetConeSection"
PetscErrorCode DMComplexGetConeSection(DM dm, PetscSection *section) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) *section = mesh->coneSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCones"
PetscErrorCode DMComplexGetCones(DM dm, PetscInt *cones[]) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cones) *cones = mesh->cones;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCoordinateVec"
PetscErrorCode DMComplexGetCoordinateVec(DM dm, Vec *coordinates) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coordinates, 2);
  *coordinates = mesh->coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateDefaultSF"
PetscErrorCode DMComplexCreateDefaultSF(DM dm)
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  PetscSection    section, gSection;
  PetscLayout     layout;
  const PetscInt *ranges;
  PetscInt       *local;
  PetscSFNode    *remote;
  PetscInt        pStart, pEnd, p, nroots, nleaves, l;
  PetscMPIInt     size, rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(((PetscObject) dm)->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMComplexGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(gSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gSection, &nroots);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(((PetscObject) dm)->comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, nroots);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
  for(p = pStart, nleaves = 0; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(gSection, p, &cdof);CHKERRQ(ierr);
    nleaves += dof < 0 ? -(dof+1)-cdof : dof-cdof;
  }
  ierr = PetscMalloc(nleaves * sizeof(PetscInt), &local);CHKERRQ(ierr);
  ierr = PetscMalloc(nleaves * sizeof(PetscSFNode), &remote);CHKERRQ(ierr);
  for(p = pStart, l = 0; p < pEnd; ++p) {
    PetscInt *cind;
    PetscInt  dof, gdof, cdof, dim, off, goff, d, c;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(section, p, &cind);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(gSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gSection, p, &goff);CHKERRQ(ierr);
    dim  = dof-cdof;
    for(d = 0, c = 0; d < dof; ++d) {
      if ((c < cdof) && (cind[c] == d)) {++c; continue;}
      local[l+d-c] = off+d;
    }
    if (gdof < 0) {
      for(d = 0; d < dim; ++d, ++l) {
        PetscInt offset = -(goff+1) + d, r;

        for(r = 0; r < size; ++r) {
          if ((offset >= ranges[r]) && (offset < ranges[r+1])) break;
        }
        remote[l].rank  = r;
        remote[l].index = offset - ranges[r];
      }
    } else {
      for(d = 0; d < dim; ++d, ++l) {
        remote[l].rank  = rank;
        remote[l].index = goff+d;
      }
    }
  }
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(mesh->sfDefault, nroots, nleaves, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_Complex"
PetscErrorCode  DMLocalToGlobalBegin_Complex(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscSF        sf;
  MPI_Op         op;
  PetscScalar   *lArray, *gArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMComplexGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  switch(mode) {
  case INSERT_VALUES:
    op = MPI_REPLACE; break;
  case ADD_VALUES:
    op = MPI_SUM; break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);
  }
  ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf, MPIU_SCALAR, lArray, gArray, op);CHKERRQ(ierr);
  ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_Complex"
PetscErrorCode  DMLocalToGlobalEnd_Complex(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscSF        sf;
  MPI_Op         op;
  PetscScalar   *lArray, *gArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMComplexGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  switch(mode) {
  case INSERT_VALUES:
    op = MPI_REPLACE; break;
  case ADD_VALUES:
    op = MPI_SUM; break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);
  }
  ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, op);CHKERRQ(ierr);
  ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_Complex"
PetscErrorCode  DMGlobalToLocalBegin_Complex(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscSF        sf;
  PetscScalar   *lArray, *gArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mode == ADD_VALUES) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);}
  ierr = DMComplexGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_Complex"
PetscErrorCode  DMGlobalToLocalEnd_Complex(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscSF        sf;
  PetscScalar   *lArray, *gArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mode == ADD_VALUES) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);}
  ierr = DMComplexGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalToGlobalMapping_Complex"
PetscErrorCode DMCreateLocalToGlobalMapping_Complex(DM dm)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *ltog;
  PetscInt       pStart, pEnd, size, p, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMComplexGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(PetscInt), &ltog);CHKERRQ(ierr); /* We want the local+overlap size */
  for(p = pStart, l = 0; p < pEnd; ++p) {
    PetscInt dof, off, c;

    /* Should probably use constrained dofs */
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &off);CHKERRQ(ierr);
    for(c = 0; c < dof; ++c, ++l) {
      ltog[l] = off+c;
    }
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, size, ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(dm, dm->ltogmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/******************************** FEM Support **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLocalFunction"
PetscErrorCode DMComplexGetLocalFunction(DM dm, PetscErrorCode (**lf)(DM, Vec, Vec, void *))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lf) *lf = mesh->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetLocalFunction"
PetscErrorCode DMComplexSetLocalFunction(DM dm, PetscErrorCode (*lf)(DM, Vec, Vec, void *))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLocalJacobian"
PetscErrorCode DMComplexGetLocalJacobian(DM dm, PetscErrorCode (**lj)(DM, Vec, Mat, void *))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lj) *lj = mesh->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetLocalJacobian"
PetscErrorCode DMComplexSetLocalJacobian(DM dm, PetscErrorCode (*lj)(DM, Vec, Mat, void *))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVecGetClosure"
/*@C
  DMComplexVecGetClosure - Get an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v
- point - The sieve point in the DM

  Output Parameters:
. values - The array of values, which is a borrowed array and should not be freed

  Level: intermediate

.seealso DMComplexVecSetClosure(), DMComplexMatSetClosure()
*/
PetscErrorCode DMComplexVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar *values[]) {
  PetscScalar    *array, *vArray;
  PetscInt       *points;
  PetscInt        offsets[32];
  PetscInt        orientation = 0;
  PetscInt        numFields, size, numPoints, pStart, pEnd, p, q, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {
    ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 32) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %d limited to 32", numFields);}
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, (const PetscInt **) &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints; ++p) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q] = points[p];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0, size = 0; p < numPoints; ++p) {
    PetscInt dof, fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for(f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    size += dof;
  }
  ierr = DMGetWorkArray(dm, 2*size, &array);CHKERRQ(ierr);
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  for(p = 0; p < numPoints; ++p) {
    PetscInt     dof, off, d;
    PetscScalar *varr;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
    varr = &vArray[off];
    if (numFields) {
      PetscInt fdof, foff, fcomp, f, c;

      for(f = 0, foff = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
        if (orientation >= 0) {
          for(d = 0; d < fdof; ++d, ++offsets[f]) {
            array[offsets[f]] = varr[foff+d];
          }
        } else {
          ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
          for(d = fdof/fcomp-1; d >= 0; --d) {
            for(c = 0; c < fcomp; ++c, ++offsets[f]) {
              array[offsets[f]] = varr[foff+d*fcomp+c];
            }
          }
        }
        foff += fdof;
      }
    } else {
      if (orientation >= 0) {
        for(d = 0; d < dof; ++d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      } else {
        for(d = dof-1; d >= 0; --d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      }
    }
  }
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  *values = array;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void add   (PetscScalar *x, PetscScalar y) {*x += y;}
PETSC_STATIC_INLINE void insert(PetscScalar *x, PetscScalar y) {*x  = y;}

#undef __FUNCT__
#define __FUNCT__ "updatePoint_private"
PetscErrorCode updatePoint_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar *, PetscScalar), PetscBool setBC, PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
  PetscInt       cdof;  /* The number of constraints on this point */
  PetscInt      *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar   *a;
  PetscInt       off, cind = 0, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
  a    = &array[off];
  if (!cdof || setBC) {
    if (orientation >= 0) {
      for(k = 0; k < dof; ++k) {
        fuse(&a[k], values[k]);
      }
    } else {
      for(k = 0; k < dof; ++k) {
        fuse(&a[k], values[dof-k-1]);
      }
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for(k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
        fuse(&a[k], values[k]);
      }
    } else {
      for(k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
        fuse(&a[k], values[dof-k-1]);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updatePointFields_private"
PetscErrorCode updatePointFields_private(PetscSection section, PetscInt point, PetscInt foffs[], void (*fuse)(PetscScalar *, PetscScalar), PetscBool setBC, PetscInt orientation, const PetscScalar values[], PetscScalar array[]) {
  PetscScalar   *a;
  PetscInt       numFields, off, foff, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
  a    = &array[off];
  for(f = 0, foff = 0; f < numFields; ++f) {
    PetscInt  fdof, fcomp, fcdof;
    PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt  cind = 0, k, c;

    ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &fcdof);CHKERRQ(ierr);
    if (!fcdof || setBC) {
      if (orientation >= 0) {
        for(k = 0; k < fdof; ++k) {
          fuse(&a[foff+k], values[foffs[f]+k]);
        }
      } else {
        for(k = fdof/fcomp-1; k >= 0; --k) {
          for(c = 0; c < fcomp; ++c) {
            fuse(&a[foff+(fdof/fcomp-1-k)*fcomp+c], values[foffs[f]+k*fcomp+c]);
          }
        }
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (orientation >= 0) {
        for(k = 0; k < fdof; ++k) {
          if ((cind < fcdof) && (k == fcdofs[cind])) {++cind; continue;}
          fuse(&a[foff+k], values[foffs[f]+k]);
        }
      } else {
        for(k = fdof/fcomp-1; k >= 0; --k) {
          for(c = 0; c < fcomp; ++c) {
            if ((cind < fcdof) && (k*fcomp+c == fcdofs[cind])) {++cind; continue;}
            fuse(&a[foff+(fdof/fcomp-1-k)*fcomp+c], values[foffs[f]+k*fcomp+c]);
          }
        }
      }
    }
    foff     += fdof;
    foffs[f] += fdof;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVecSetClosure"
PetscErrorCode DMComplexVecSetClosure(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode) {
  PetscScalar    *array;
  PetscInt       *points;
  PetscInt        offsets[32];
  PetscInt        orientation = 0;
  PetscInt        numFields, numPoints, off, dof, pStart, pEnd, p, q, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {
    ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 32) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %d limited to 32", numFields);}
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, (const PetscInt **) &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints; ++p) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q] = points[p];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0; p < numPoints; ++p) {
    PetscInt fdof;

    for(f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
  }
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  if (numFields) {
    switch(mode) {
    case INSERT_VALUES:
      for(p = 0; p < numPoints; ++p) {
        updatePointFields_private(section, points[p], offsets, insert, PETSC_FALSE, orientation, values, array);
      } break;
    case INSERT_ALL_VALUES:
      for(p = 0; p < numPoints; ++p) {
        updatePointFields_private(section, points[p], offsets, insert, PETSC_TRUE,  orientation, values, array);
      } break;
    case ADD_VALUES:
      for(p = 0; p < numPoints; ++p) {
        updatePointFields_private(section, points[p], offsets, add,    PETSC_FALSE, orientation, values, array);
      } break;
    case ADD_ALL_VALUES:
      for(p = 0; p < numPoints; ++p) {
        updatePointFields_private(section, points[p], offsets, add,    PETSC_TRUE,  orientation, values, array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
  } else {
    switch(mode) {
    case INSERT_VALUES:
      for(p = 0, off = 0; p < numPoints; ++p, off += dof) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_FALSE, orientation, &values[off], array);
      } break;
    case INSERT_ALL_VALUES:
      for(p = 0, off = 0; p < numPoints; ++p, off += dof) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_TRUE,  orientation, &values[off], array);
      } break;
    case ADD_VALUES:
      for(p = 0, off = 0; p < numPoints; ++p, off += dof) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_FALSE, orientation, &values[off], array);
      } break;
    case ADD_ALL_VALUES:
      for(p = 0, off = 0; p < numPoints; ++p, off += dof) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_TRUE,  orientation, &values[off], array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
  }
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPrintMatSetValues"
PetscErrorCode DMComplexPrintMatSetValues(Mat A, PetscInt point, PetscInt numIndices, const PetscInt indices[], PetscScalar values[])
{
  PetscMPIInt    rank;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) A)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]mat for sieve point %d\n", rank, point);CHKERRQ(ierr);
  for(i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]mat indices[%d] = %d\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  for(i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]", rank);CHKERRQ(ierr);
    for(j = 0; j < numIndices; j++) {
#ifdef PETSC_USE_COMPLEX
      ierr = PetscPrintf(PETSC_COMM_SELF, " (%g,%g)", PetscRealPart(values[i*numIndices+j]), PetscImaginaryPart(values[i*numIndices+j]));CHKERRQ(ierr);
#else
      ierr = PetscPrintf(PETSC_COMM_SELF, " %g", values[i*numIndices+j]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "indicesPoint_private"
/* . off - The global offset of this point */
PetscErrorCode indicesPoint_private(PetscSection section, PetscInt point, PetscInt dof, PetscInt off, PetscBool setBC, PetscInt orientation, PetscInt indices[]) {
  PetscInt       cdof;  /* The number of constraints on this point */
  PetscInt      *cdofs; /* The indices of the constrained dofs on this point */
  PetscInt       cind = 0, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  if (!cdof || setBC) {
    if (orientation >= 0) {
      for(k = 0; k < dof; ++k) {
        indices[k] = off+k;
      }
    } else {
      for(k = 0; k < dof; ++k) {
        indices[dof-k-1] = off+k;
      }
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for(k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[k] = -(off+k+1);
          ++cind;
        } else {
          indices[k] = off+k;
        }
      }
    } else {
      for(k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[dof-k-1] = -(off+k+1);
          ++cind;
        } else {
          indices[dof-k-1] = off+k;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "indicesPointFields_private"
/* . off - The global offset of this point */
PetscErrorCode indicesPointFields_private(PetscSection section, PetscInt point, PetscInt off, PetscInt foffs[], PetscBool setBC, PetscInt orientation, PetscInt indices[]) {
  PetscInt       numFields, foff, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for(f = 0, foff = 0; f < numFields; ++f) {
    PetscInt  fdof, fcomp, cfdof;
    PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt  cind = 0, k, c;

    ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cfdof);CHKERRQ(ierr);
    if (!cfdof || setBC) {
      if (orientation >= 0) {
        for(k = 0; k < fdof; ++k) {
          indices[foffs[f]+k] = off+foff+k;
        }
      } else {
        for(k = fdof/fcomp-1; k >= 0; --k) {
          for(c = 0; c < fcomp; ++c) {
            indices[foffs[f]+k*fcomp+c] = off+foff+(fdof/fcomp-1-k)*fcomp+c;
          }
        }
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (orientation >= 0) {
        for(k = 0; k < fdof; ++k) {
          if ((cind < cfdof) && (k == fcdofs[cind])) {
            indices[foffs[f]+k] = -(off+foff+k+1);
            ++cind;
          } else {
            indices[foffs[f]+k] = off+foff+k;
          }
        }
      } else {
        for(k = fdof/fcomp-1; k >= 0; --k) {
          for(c = 0; c < fcomp; ++c) {
            if ((cind < cfdof) && ((fdof/fcomp-1-k)*fcomp+c == fcdofs[cind])) {
              indices[foffs[f]+k*fcomp+c] = -(off+foff+(fdof/fcomp-1-k)*fcomp+c+1);
              ++cind;
            } else {
              indices[foffs[f]+k*fcomp+c] = off+foff+(fdof/fcomp-1-k)*fcomp+c;
            }
          }
        }
      }
    }
    foff     += fdof - cfdof;
    foffs[f] += fdof;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexMatSetClosure"
PetscErrorCode DMComplexMatSetClosure(DM dm, PetscSection section, PetscSection globalSection, Mat A, PetscInt point, PetscScalar values[], InsertMode mode)
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  PetscInt       *points;
  PetscInt       *indices;
  PetscInt        offsets[32];
  PetscInt        orientation = 0;
  PetscInt        numFields, numPoints, numIndices, dof, off, globalOff, pStart, pEnd, p, q, f;
  PetscBool       useDefault       =       !section ? PETSC_TRUE : PETSC_FALSE;
  PetscBool       useGlobalDefault = !globalSection ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 3);
  if (useDefault) {
    ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  if (useGlobalDefault) {
    if (useDefault) {
      ierr = DMComplexGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
    } else {
      ierr = PetscSectionCreateGlobalSection(section, mesh->sf, &globalSection);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 32) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %d limited to 32", numFields);}
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, (const PetscInt **) &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints; ++p) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q] = points[p];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0, numIndices = 0; p < numPoints; ++p) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for(f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    numIndices += dof;
  }
  ierr = DMGetWorkArray(dm, numIndices, (PetscScalar **) &indices);CHKERRQ(ierr);
  if (numFields) {
    for(p = 0; p < numPoints; ++p) {
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPointFields_private(section, points[p], globalOff, offsets, PETSC_FALSE, orientation, indices);
    }
  } else {
    for(p = 0, off = 0; p < numPoints; ++p, off += dof) {
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPoint_private(section, points[p], dof, globalOff, PETSC_FALSE, orientation, &indices[off]);
    }
  }
  if (useGlobalDefault && !useDefault) {
    ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
  }
  if (mesh->printSetValues) {ierr = DMComplexPrintMatSetValues(A, point, numIndices, indices, values);CHKERRQ(ierr);}
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, values, mode);
  if (ierr) {
    PetscMPIInt    rank;
    PetscErrorCode ierr2;

    ierr2 = MPI_Comm_rank(((PetscObject) A)->comm, &rank);CHKERRQ(ierr2);
    ierr2 = PetscPrintf(PETSC_COMM_SELF, "[%d]ERROR in DMComplexMatSetClosure\n", rank);CHKERRQ(ierr2);
    ierr2 = DMComplexPrintMatSetValues(A, point, numIndices, indices, values);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeTriangleGeometry_private"
PetscErrorCode DMComplexComputeTriangleGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  const PetscScalar *coords;
  const PetscInt     dim = 2;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMComplexVecGetClosure(dm, mesh->coordSection, mesh->coordinates, e, &coords);CHKERRQ(ierr);
  if (v0) {
    for(d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for(d = 0; d < dim; d++) {
      for(f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    *detJ = J[0]*J[3] - J[1]*J[2];
#if 0
    if (detJ < 0.0) {
      const PetscReal xLength = mesh->periodicity[0];

      if (xLength != 0.0) {
        PetscReal v0x = coords[0*dim+0];

        if (v0x == 0.0) {
          v0x = v0[0] = xLength;
        }
        for(f = 0; f < dim; f++) {
          const PetscReal px = coords[(f+1)*dim+0] == 0.0 ? xLength : coords[(f+1)*dim+0];

          J[0*dim+f] = 0.5*(px - v0x);
        }
      }
      detJ = J[0]*J[3] - J[1]*J[2];
    }
#endif
    PetscLogFlops(8.0 + 3.0);
  }
  if (invJ) {
    const PetscReal invDet = 1.0/(*detJ);

    invJ[0] =  invDet*J[3];
    invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2];
    invJ[3] =  invDet*J[0];
    PetscLogFlops(5.0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeRectangleGeometry_private"
PetscErrorCode DMComplexComputeRectangleGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  const PetscScalar *coords;
  const PetscInt     dim = 2;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMComplexVecGetClosure(dm, mesh->coordSection, mesh->coordinates, e, &coords);CHKERRQ(ierr);
  if (v0) {
    for(d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for(d = 0; d < dim; d++) {
      for(f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    *detJ = J[0]*J[3] - J[1]*J[2];
    PetscLogFlops(8.0 + 3.0);
  }
  if (invJ) {
    const PetscReal invDet = 1.0/(*detJ);

    invJ[0] =  invDet*J[3];
    invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2];
    invJ[3] =  invDet*J[0];
    PetscLogFlopsNoError(5.0);
  }
  *detJ *= 2.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeTetrahedronGeometry_private"
PetscErrorCode DMComplexComputeTetrahedronGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  const PetscScalar *coords;
  const PetscInt     dim = 3;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMComplexVecGetClosure(dm, mesh->coordSection, mesh->coordinates, e, &coords);CHKERRQ(ierr);
  if (v0) {
    for(d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for(d = 0; d < dim; d++) {
      for(f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    /* The minus sign is here since I orient the first face to get the outward normal */
    *detJ = -(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
              J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
              J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
    PetscLogFlops(18.0 + 12.0);
  }
  if (invJ) {
    const PetscReal invDet = -1.0/(*detJ);

    invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
    invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
    invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
    invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
    invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
    invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
    invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
    invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
    PetscLogFlops(37.0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeHexahedronGeometry_private"
PetscErrorCode DMComplexComputeHexahedronGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  DM_Complex        *mesh = (DM_Complex *) dm->data;
  const PetscScalar *coords;
  const PetscInt     dim = 3;
  PetscInt           d;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMComplexVecGetClosure(dm, mesh->coordSection, mesh->coordinates, e, &coords);CHKERRQ(ierr);
  if (v0) {
    for(d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for(d = 0; d < dim; d++) {
      J[d*dim+0] = 0.5*(PetscRealPart(coords[(0+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+1] = 0.5*(PetscRealPart(coords[(1+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+2] = 0.5*(PetscRealPart(coords[(3+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
    }
    *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
             J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
             J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
    PetscLogFlops(18.0 + 12.0);
  }
  if (invJ) {
    const PetscReal invDet = -1.0/(*detJ);

    invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
    invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
    invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
    invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
    invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
    invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
    invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
    invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
    PetscLogFlops(37.0);
  }
  *detJ *= 8.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeCellGeometry"
PetscErrorCode DMComplexComputeCellGeometry(DM dm, PetscInt cell, PetscReal *v0, PetscReal *J, PetscReal *invJ, PetscReal *detJ) {
  PetscInt       dim, maxConeSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, PETSC_NULL);CHKERRQ(ierr);
  switch(dim) {
  case 2:
    switch(maxConeSize) {
    case 3:
      ierr = DMComplexComputeTriangleGeometry_private(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
    case 4:
      ierr = DMComplexComputeRectangleGeometry_private(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported number of cell vertices %d for element geometry computation", maxConeSize);
    }
    break;
  case 3:
    switch(maxConeSize) {
    case 4:
      ierr = DMComplexComputeTetrahedronGeometry_private(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
    case 8:
      ierr = DMComplexComputeHexahedronGeometry_private(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported number of cell vertices %d for element geometry computation", maxConeSize);
    }
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported dimension %d for element geometry computation", dim);
  }
  PetscFunctionReturn(0);
}
