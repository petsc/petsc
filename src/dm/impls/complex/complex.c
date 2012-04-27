#include <petsc-private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

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
    PetscBool   hasLabel;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh '%s':\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Max sizes cone: %D support: %D\n", maxConeSize, maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "orientation is missing\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "cap --> base:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for(s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D ----> %D\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "base <-- cap:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for(c = off; c < off+dof; ++c) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D <---- %D (%D)\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscSectionVecView(mesh->coordSection, mesh->coordinates, viewer);CHKERRQ(ierr);
    ierr = DMComplexHasLabel(dm, "marker", &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      const char     *name = "marker";
      IS              ids;
      const PetscInt *markers;
      PetscInt        num, i;

      ierr = PetscViewerASCIIPrintf(viewer, "Label '%s':\n", name);CHKERRQ(ierr);
      ierr = DMComplexGetLabelIdIS(dm, name, &ids);CHKERRQ(ierr);
      ierr = ISGetSize(ids, &num);CHKERRQ(ierr);
      ierr = ISGetIndices(ids, &markers);CHKERRQ(ierr);
      for(i = 0; i < num; ++i) {
        IS              pIS;
        const PetscInt *points;
        PetscInt        size, p;

        ierr = DMComplexGetStratumIS(dm, name, markers[i], &pIS);
        ierr = ISGetSize(pIS, &size);CHKERRQ(ierr);
        ierr = ISGetIndices(pIS, &points);CHKERRQ(ierr);
        for(p = 0; p < size; ++p) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D (%D)\n", rank, points[p], markers[i]);CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(pIS, &points);CHKERRQ(ierr);
        ierr = ISDestroy(&pIS);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(ids, &markers);CHKERRQ(ierr);
      ierr = ISDestroy(&ids);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_LATEX) {
    const char  *name;
    const char  *colors[3] = {"red", "blue", "green"};
    const int    numColors = 3;
    PetscScalar *coords;
    PetscInt     cStart, cEnd, c, vStart, vEnd, v, p;
    PetscMPIInt  rank, size;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject) dm)->comm, &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\documentclass{beamer}\n\n\
\\usepackage{tikz}\n\
\\usepackage{pgflibraryshapes}\n\
\\usetikzlibrary{backgrounds}\n\
\\usetikzlibrary{arrows}\n\
\\newenvironment{changemargin}[2]{%%\n\
  \\begin{list}{}{%%\n\
    \\setlength{\\topsep}{0pt}%%\n\
    \\setlength{\\leftmargin}{#1}%%\n\
    \\setlength{\\rightmargin}{#2}%%\n\
    \\setlength{\\listparindent}{\\parindent}%%\n\
    \\setlength{\\itemindent}{\\parindent}%%\n\
    \\setlength{\\parsep}{\\parskip}%%\n\
  }%%\n\
  \\item[]}{\\end{list}}\n\n\
\\begin{document}\n\
\\begin{frame}{%s}\n\
\\begin{changemargin}{-1cm}{0cm}\n\
\\begin{center}\n\
\\begin{tikzpicture}[scale = 5.00,font=\\fontsize{8}{8}\\selectfont]\n", name);CHKERRQ(ierr);
    /* Plot vertices */
    ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      PetscInt off, dof, d;

      ierr = PetscSectionGetDof(mesh->coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
      for(d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%G", PetscRealPart(coords[off+d]));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%D) [draw,shape=circle,color=%s] {%D} --\n", v, rank, colors[rank%numColors], v);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "(0,0);\n");CHKERRQ(ierr);
    /* Plot cells */
    ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c) {
      PetscInt *closure = PETSC_NULL;
      PetscInt  closureSize, firstPoint = -1;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] ", colors[rank%numColors]);CHKERRQ(ierr);
      for(p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = closure[p];

        if ((point < vStart) || (point >= vEnd)) continue;
        if (firstPoint >= 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(%D_%D)", point, rank);CHKERRQ(ierr);
        if (firstPoint < 0) firstPoint = point;
      }
      /* Why doesn't this work? ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- cycle;\n");CHKERRQ(ierr); */
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- (%D_%D);\n", firstPoint, rank);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n\\end{center}\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh for processes ");CHKERRQ(ierr);
    for(p = 0; p < size; ++p) {
      if (p == size-1) {
        ierr = PetscViewerASCIIPrintf(viewer, ", and ", colors[p%numColors], p);CHKERRQ(ierr);
      } else if (p > 0) {
        ierr = PetscViewerASCIIPrintf(viewer, ", ", colors[p%numColors], p);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "{\\textcolor{%s}%D}", colors[p%numColors], p);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, ".\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{changemargin}\n\
\\end{frame}\n\
\\end{document}\n", name);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm = ((PetscObject) dm)->comm;
    PetscInt   *sizes;
    PetscInt    locDepth, depth, dim, d;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt size;

    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %D dimensions:\n", dim);CHKERRQ(ierr);
    ierr = DMComplexGetDepth(dm, &locDepth);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &sizes);CHKERRQ(ierr);
    if (depth == 1) {
      ierr = DMComplexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", 0);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMComplexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", dim);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for(d = 0; d <= dim; d++) {
        ierr = DMComplexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd = pEnd - pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", d);CHKERRQ(ierr);
        for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
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
  } else SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
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
  ierr = PetscFree(mesh->coneOrientations);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->coordSection);CHKERRQ(ierr);
  ierr = VecDestroy(&mesh->coordinates);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->meetTmpA,mesh->meetTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->joinTmpA,mesh->joinTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->closureTmpA,mesh->closureTmpB);CHKERRQ(ierr);
  ierr = PetscFree(mesh->facesTmp);CHKERRQ(ierr);
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
#define __FUNCT__ "DMComplexGetAdjacencySingleLevel_Private"
PetscErrorCode DMComplexGetAdjacencySingleLevel_Private(DM dm, PetscInt p, PetscBool useClosure, const PetscInt *tmpClosure, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = PETSC_NULL;
  PetscInt        numAdj  = 0, maxAdjSize = *adjSize, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (useClosure) {
    ierr = DMComplexGetConeSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone = PETSC_NULL;
      PetscInt        coneSize, c, q;

      ierr = DMComplexGetSupportSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetSupport(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
          if (cone[c] == adj[q]) break;
        }
        if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
      }
    }
  } else {
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for(s = 0; s < supportSize; ++s) {
      const PetscInt *cone = PETSC_NULL;
      PetscInt        coneSize, c, q;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        for(q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
          if (cone[c] == adj[q]) break;
        }
        if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
      }
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetAdjacency_Private"
PetscErrorCode DMComplexGetAdjacency_Private(DM dm, PetscInt p, PetscBool useClosure, const PetscInt *tmpClosure, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *star   = tmpClosure;
  PetscInt        numAdj = 0, maxAdjSize = *adjSize, starSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetTransitiveClosure(dm, p, useClosure, &starSize, (PetscInt **) &star);CHKERRQ(ierr);
  for(s = 2; s < starSize*2; s += 2) {
    const PetscInt *closure = PETSC_NULL;
    PetscInt        closureSize, c, q;

    ierr = DMComplexGetTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt **) &closure);CHKERRQ(ierr);
    for(c = 0; c < closureSize*2; c += 2) {
      for(q = 0; q < numAdj || (adj[numAdj++] = closure[c],0); ++q) {
        if (closure[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
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
  PetscInt           pStart, pEnd, numDof, globalOffStart, globalOffEnd, numCols;
  PetscInt          *tmpClosure, *tmpAdj, *adj, *rootAdj, *cols;
  PetscInt           depth, maxConeSize, maxSupportSize, maxClosureSize, maxAdjSize, adjSize;
  PetscLayout        rLayout;
  PetscInt           locRows, rStart, rEnd, r;
  PetscMPIInt        size;
  PetscBool          debug = PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-dm_view_preallocation", &debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
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
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  maxClosureSize = 2*PetscMax(pow((PetscReal) mesh->maxConeSize, depth)+1, pow((PetscReal) mesh->maxSupportSize, depth)+1);
  maxAdjSize = pow((PetscReal) mesh->maxConeSize, depth)*pow((PetscReal) mesh->maxSupportSize, depth)+1;
  ierr = PetscMalloc2(maxClosureSize,PetscInt,&tmpClosure,maxAdjSize,PetscInt,&tmpAdj);CHKERRQ(ierr);

  /*
   ** The bootstrapping process involves six rounds with similar structure of visiting neighbors of each point.
    1. Visit unowned points on interface, count adjacencies placing in leafSectionAdj
       Reduce those counts to rootSectionAdj (now redundantly counting some interface points)
    2. Visit owned points on interface, count adjacencies placing in rootSectionAdj
       Create sfAdj connecting rootSectionAdj and leafSectionAdj
    3. Visit unowned points on interface, write adjacencies to adj
       Gather adj to rootAdj (note that there is redundancy in rootAdj when multiple procs find the same adjacencies)
    4. Visit owned points on interface, write adjacencies to rootAdj
       Remove redundancy in rootAdj
   ** The last two traversals use transitive closure
    5. Visit all owned points in the subdomain, count dofs for each point (sectionAdj)
       Allocate memory addressed by sectionAdj (cols)
    6. Visit all owned points in the subdomain, insert dof adjacencies into cols
   ** Knowing all the column adjacencies, check ownership and sum into dnz and onz
  */

  for(l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q;
    PetscInt p = leaves[l], numAdj = maxAdjSize;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
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
    PetscInt numAdj = maxAdjSize, adof, dof, off, d, q;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for(q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
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
    PetscInt dof, off, d, q;
    PetscInt p = leaves[l], numAdj = maxAdjSize;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
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
    ierr = PetscPrintf(comm, "Leaf adjacency indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, adj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Gather adjacenct indices to root */
  ierr = PetscSectionGetStorageSize(rootSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscMalloc(adjSize * sizeof(PetscInt), &rootAdj);CHKERRQ(ierr);
  for(r = 0; r < adjSize; ++r) {
    rootAdj[r] = -1;
  }
  if (size > 1) {
    ierr = PetscSFGatherBegin(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
  }
  ierr = PetscFree(adj);CHKERRQ(ierr);
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Root adjacency indices after gather\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Add in local adjacency indices for owned dofs on interface (roots) */
  for(p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = maxAdjSize, adof, dof, off, d, q;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for(d = off; d < off+dof; ++d) {
      PetscInt adof, aoff, i;

      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      i    = adof-1;
      for(q = 0; q < numAdj; ++q) {
        PetscInt ndof, ncdof, ngoff, nd;

        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for(nd = 0; nd < ndof-ncdof; ++nd) {
          rootAdj[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd: ngoff+nd;
          --i;
        }
      }
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Root adjacency indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Compress indices */
  ierr = PetscSectionSetUp(rootSectionAdj);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    for(d = off; d < off+dof-cdof; ++d) {
      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&adof, &rootAdj[aoff]);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(rootSectionAdj, d, adof);CHKERRQ(ierr);
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after compression:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Root adjacency indices after compression\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Build adjacency section: Maps global indices to sets of adjacent global indices */
  ierr = PetscSectionGetOffsetRange(sectionGlobal, &globalOffStart, &globalOffEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &sectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionAdj, globalOffStart, globalOffEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = maxAdjSize, dof, cdof, off, goff, d, q;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for(d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        ierr = PetscSectionSetDof(sectionAdj, goff+d, rdof);CHKERRQ(ierr);
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
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
    PetscInt  numAdj = maxAdjSize, dof, cdof, off, goff, d, q;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for(d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
        /* We do not own this point */
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
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
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
          cols[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd: ngoff+nd;
        }
      }
      if (i != adof) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %D != %D for dof %D (point %D)", i, adof, d, p);
    }
  }
  ierr = PetscFree(rootAdj);CHKERRQ(ierr);
  ierr = PetscFree2(tmpClosure, tmpAdj);CHKERRQ(ierr);
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Column indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, numCols, cols, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
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
  ierr = MatXAIJSetPreallocation(A, bs, dnz, onz, dnzu, onzu);CHKERRQ(ierr);
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
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "DMComplexPreallocateOperator_2"
PetscErrorCode DMComplexPreallocateOperator_2(DM dm, PetscInt bs, PetscSection section, PetscSection sectionGlobal, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd,pStart,pEnd;
  PetscInt *tmpClosure,*tmpAdj,*visits;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  maxClosureSize = 2*PetscMax(pow(mesh->maxConeSize, depth)+1, pow(mesh->maxSupportSize, depth)+1);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  npoints = pEnd - pStart;
  ierr = PetscMalloc3(maxClosureSize,PetscInt,&tmpClosure,npoints,PetscInt,&lvisits,npoints,PetscInt,&visits);CHKERRQ(ierr);
  ierr = PetscMemzero(lvisits,(pEnd-pStart)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(visits,(pEnd-pStart)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    PetscInt *support = tmpClosure;
    ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_FALSE, &supportSize, (PetscInt**)&support);CHKERRQ(ierr);
    for (p=0; p<supportSize; p++) {
      lvisits[support[p]]++;
    }
  }
  ierr = PetscSFReduceBegin(sf,MPIU_INT,lvisits,visits,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,lvisits,visits,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,visits,lvisits);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,visits,lvisits);CHKERRQ(ierr);

  ierr = PetscSFGetRanks();CHKERRQ(ierr);


  ierr = PetscMalloc2(maxClosureSize*maxClosureSize,PetscInt,&cellmat,npoints,PetscInt,&owner);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    ierr = PetscMemzero(cellmat,maxClosureSize*maxClosureSize*sizeof(PetscInt));CHKERRQ(ierr);
    /*
     Depth-first walk of transitive closure.
     At each leaf frame f of transitive closure that we see, add 1/visits[f] to each pair (p,q) not marked as done in cellmat.
     This contribution is added to dnz if owning ranks of p and q match, to onz otherwise.
     */
  }

  ierr = PetscSFReduceBegin(sf,MPIU_INT,ldnz,dnz,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,lonz,onz,MPI_SUM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

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
#define __FUNCT__ "DMCreateFieldIS_Complex"
PetscErrorCode DMCreateFieldIS_Complex(DM dm, PetscInt *numFields, char ***fieldNames, IS **fields)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *fieldSizes, **fieldIndices;
  PetscInt       nF, f, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMComplexGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &nF);CHKERRQ(ierr);
  ierr = PetscMalloc2(nF,PetscInt,&fieldSizes,nF,PetscInt *,&fieldIndices);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
  for(f = 0; f < nF; ++f) {
    fieldSizes[f] = 0;
  }
  for(p = pStart; p < pEnd; ++p) {
    PetscInt gdof;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      for(f = 0; f < nF; ++f) {
        PetscInt fdof, fcdof;

        ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        fieldSizes[f] += fdof-fcdof;
      }
    }
  }
  for(f = 0; f < nF; ++f) {
    ierr = PetscMalloc(fieldSizes[f] * sizeof(PetscInt), &fieldIndices[f]);CHKERRQ(ierr);
    fieldSizes[f] = 0;
  }
  for(p = pStart; p < pEnd; ++p) {
    PetscInt gdof, goff;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
      for(f = 0; f < nF; ++f) {
        PetscInt fdof, fcdof, fc;

        ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        for(fc = 0; fc < fdof-fcdof; ++fc, ++fieldSizes[f]) {
          fieldIndices[f][fieldSizes[f]] = goff++;
        }
      }
    }
  }
  if (numFields) {*numFields = nF;}
  if (fieldNames) {
    ierr = PetscMalloc(nF * sizeof(char *), fieldNames);CHKERRQ(ierr);
    for(f = 0; f < nF; ++f) {
      const char *fieldName;

      ierr = PetscSectionGetFieldName(section, f, &fieldName);CHKERRQ(ierr);
      ierr = PetscStrallocpy(fieldName, (char **) &(*fieldNames)[f]);CHKERRQ(ierr);
    }
  }
  if (fields) {
    ierr = PetscMalloc(nF * sizeof(IS), fields);CHKERRQ(ierr);
    for(f = 0; f < nF; ++f) {
      ierr = ISCreateGeneral(((PetscObject) dm)->comm, fieldSizes[f], fieldIndices[f], PETSC_OWN_POINTER, &(*fields)[f]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(fieldSizes,fieldIndices);CHKERRQ(ierr);
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
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for(c = 0; c < dof; ++c) {
    if ((cone[c] < pStart) || (cone[c] >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D. %D)", cone[c], pStart, pEnd);
    mesh->cones[off+c] = cone[c];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetConeOrientation"
/*@C
  DMComplexGetConeOrientation - Return the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
- p - The Sieve point, which must lie in the chart set with DMComplexSetChart()

  Output Parameter:
. coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Level: beginner

  Note:
  This routine is not available in Fortran.

.seealso: DMComplexCreate(), DMComplexGetCone(), DMComplexSetCone(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexGetConeOrientation(DM dm, PetscInt p, const PetscInt *coneOrientation[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coneOrientation, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  *coneOrientation = &mesh->coneOrientations[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetConeOrientation"
/*@
  DMComplexSetConeOrientation - Set the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Output Parameter:

  Note:
  This should be called after all calls to DMComplexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetConeOrientation(), DMComplexSetCone(), DMComplexSetChart(), DMComplexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMComplexSetConeOrientation(DM dm, PetscInt p, const PetscInt coneOrientation[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coneOrientation, 3);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for(c = 0; c < dof; ++c) {
    PetscInt cdof, o = coneOrientation[c];

    ierr = PetscSectionGetDof(mesh->coneSection, mesh->cones[off+c], &cdof);CHKERRQ(ierr);
    if ((o < -(cdof+1)) || (o >= cdof)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone orientation %D is not in the valid range [%D. %D)", o, -(cdof+1), cdof);
    mesh->coneOrientations[off+c] = o;
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
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((conePoint < pStart) || (conePoint >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", conePoint, pStart, pEnd);
  if (conePos >= dof) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D is not in the valid range [0, %D)", conePos, dof);
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
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
- points - If points is PETSC_NULL on input, internal storage will be returned, otherwise the provided array is used

  Output Parameters:
+ numPoints - The number of points in the closure, so points[] is of size 2*numPoints
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...]

  Note:
  If using internal storage (points is PETSC_NULL on input), each call overwrites the last output.

  Level: beginner

.seealso: DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart(), DMComplexGetCone()
@*/
PetscErrorCode DMComplexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  PetscInt       *closure, *fifo;
  const PetscInt *tmp, *tmpO = PETSC_NULL;
  PetscInt        tmpSize, t;
  PetscInt        closureSize = 2, fifoSize = 0, fifoStart = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->closureTmpA) {
    PetscInt depth, maxSize;

    ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
    maxSize = 2*PetscMax(pow((PetscReal) mesh->maxConeSize, depth)+1, pow((PetscReal) mesh->maxSupportSize, depth)+1);
    ierr = PetscMalloc2(maxSize,PetscInt,&mesh->closureTmpA,maxSize,PetscInt,&mesh->closureTmpB);CHKERRQ(ierr);
  }
  closure = *points ? *points : mesh->closureTmpA;
  fifo    = mesh->closureTmpB;
  closure[0] = p; closure[1] = 0;
  /* This is only 1-level */
  if (useCone) {
    ierr = DMComplexGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(dm, p, &tmp);CHKERRQ(ierr);
    ierr = DMComplexGetConeOrientation(dm, p, &tmpO);CHKERRQ(ierr);
  } else {
    ierr = DMComplexGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &tmp);CHKERRQ(ierr);
  }
  for(t = 0; t < tmpSize; ++t, closureSize += 2, fifoSize += 2) {
    const PetscInt cp = tmp[t];
    const PetscInt co = tmpO ? tmpO[t] : 0;

    closure[closureSize]   = cp;
    closure[closureSize+1] = co;
    fifo[fifoSize]         = cp;
    fifo[fifoSize+1]       = co;
  }
  while(fifoSize - fifoStart) {
    const PetscInt q   = fifo[fifoStart];
    const PetscInt o   = fifo[fifoStart+1];
    const PetscInt rev = o >= 0 ? 0 : 1;
    const PetscInt off = rev ? -(o+1) : o;

    if (useCone) {
      ierr = DMComplexGetConeSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, q, &tmp);CHKERRQ(ierr);
      ierr = DMComplexGetConeOrientation(dm, q, &tmpO);CHKERRQ(ierr);
    } else {
      ierr = DMComplexGetSupportSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMComplexGetSupport(dm, q, &tmp);CHKERRQ(ierr);
      tmpO = PETSC_NULL;
    }
    for(t = 0; t < tmpSize; ++t) {
      const PetscInt i  = ((rev ? tmpSize-t : t) + off)%tmpSize;
      const PetscInt cp = tmp[i];
      const PetscInt co = tmpO ? tmpO[i] : 0;
      PetscInt       c;

      /* Check for duplicate */
      for(c = 0; c < closureSize; c += 2) {
        if (closure[c] == cp) break;
      }
      if (c == closureSize) {
        closure[closureSize]   = cp;
        closure[closureSize+1] = co;
        fifo[fifoSize]         = cp;
        fifo[fifoSize+1]       = co;
        closureSize += 2;
        fifoSize    += 2;
      }
    }
    fifoStart += 2;
  }
  if (numPoints) *numPoints = closureSize/2;
  if (points)    *points    = closure;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetFaces"
/*
  DMComplexGetFaces - 

  Note: This will only work for cell-vertex meshes.
*/
PetscErrorCode DMComplexGetFaces(DM dm, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  const PetscInt *cone;
  PetscInt        depth, dim, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth > 1) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Faces can only be returned for cell-vertex meshes.");
  if (!mesh->facesTmp) {ierr = PetscMalloc(mesh->maxConeSize*mesh->maxSupportSize * sizeof(PetscInt), &mesh->facesTmp);CHKERRQ(ierr);}
  ierr = DMComplexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMComplexGetCone(dm, p, &cone);CHKERRQ(ierr);
  switch(dim) {
  case 2:
    switch(coneSize) {
    case 3:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[0];
      *numFaces = 3;
      *faceSize = 2;
      *faces    = mesh->facesTmp;
      break;
    default:
      SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension % not supported", dim);
  }
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
  ierr = PetscMalloc(size * sizeof(PetscInt), &mesh->coneOrientations);CHKERRQ(ierr);
  ierr = PetscMemzero(mesh->coneOrientations, size * sizeof(PetscInt));CHKERRQ(ierr);
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
#define __FUNCT__ "DMComplexSetDepth_Private"
PetscErrorCode DMComplexSetDepth_Private(DM dm, PetscInt p, PetscInt *depth)
{
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetLabelValue(dm, "depth", p, &d);CHKERRQ(ierr);
  if (d < 0) {
    /* We are guaranteed that the point has a cone since the depth was not yet set */
    const PetscInt *cone;
    PetscInt        dCone;

    ierr = DMComplexGetCone(dm, p, &cone);CHKERRQ(ierr);
    ierr = DMComplexSetDepth_Private(dm, cone[0], &dCone);CHKERRQ(ierr);
    d    = dCone+1;
    ierr = DMComplexSetLabelValue(dm, "depth", p, d);CHKERRQ(ierr);
  }
  *depth = d;
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

    ierr = DMComplexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
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

      ierr = DMComplexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        ierr = DMComplexSetLabelValue(dm, "depth", p, 1);CHKERRQ(ierr);
      }
    }
  } else {
    /* This might be slow since lookup is not fast */
    for(p = pStart; p < pEnd; ++p) {
      PetscInt depth;

      ierr = DMComplexSetDepth_Private(dm, p, &depth);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexHasLabel"
/*@C
  DMComplexHasLabel - Determine whether the mesh has a label of a given name

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
- name - The label name

  Output Parameter:
. hasLabel - PETSC_TRUE if the label is present

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetLabelValue(), DMComplexSetLabelValue(), DMComplexGetLabelStratum()
@*/
PetscErrorCode DMComplexHasLabel(DM dm, const char name[], PetscBool *hasLabel)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(hasLabel, 3);
  *hasLabel = PETSC_FALSE;
  while(next) {
    ierr = PetscStrcmp(name, next->name, hasLabel);CHKERRQ(ierr);
    if (*hasLabel) break;
    next = next->next;
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
. value - The label value for this point, or -1 if the point is not in the label

  Level: beginner

.keywords: mesh
.seealso: DMComplexSetLabelValue(), DMComplexGetLabelStratum()
@*/
PetscErrorCode DMComplexGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscBool      flg;
  PetscInt       v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  *value = -1;
  ierr = DMComplexHasLabel(dm, name, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);CHKERRQ(ierr);
  /* We should have a generic GetLabel() and a Label class */
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
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
  PetscInt       size=-1, i = 0;
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
PetscErrorCode DMComplexJoinPoints(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt      *join[2];
  PetscInt       joinSize, i = 0;
  PetscInt       dof, off, p, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);
  if (!mesh->joinTmpA) {ierr = PetscMalloc2(mesh->maxSupportSize,PetscInt,&mesh->joinTmpA,mesh->maxSupportSize,PetscInt,&mesh->joinTmpB);CHKERRQ(ierr);}
  join[0] = mesh->joinTmpA; join[1] = mesh->joinTmpB;
  /* Copy in support of first point */
  ierr = PetscSectionGetDof(mesh->supportSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, points[0], &off);CHKERRQ(ierr);
  for(joinSize = 0; joinSize < dof; ++joinSize) {
    join[i][joinSize] = mesh->supports[off+joinSize];
  }
  /* Check each successive cone */
  for(p = 1; p < numPoints; ++p) {
    PetscInt newJoinSize = 0;

    ierr = PetscSectionGetDof(mesh->supportSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->supportSection, points[p], &off);CHKERRQ(ierr);
    for(c = 0; c < dof; ++c) {
      const PetscInt point = mesh->supports[off+c];

      for(m = 0; m < joinSize; ++m) {
        if (point == join[i][m]) {
          join[1-i][newJoinSize++] = point;
          break;
        }
      }
    }
    joinSize = newJoinSize;
    i = 1-i;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
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
  *coveringPoints    = meet[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateNeighborCSR"
PetscErrorCode DMComplexCreateNeighborCSR(DM dm, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency) {
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[30]; /* maxFaceCases, C89 sucks sucks sucks */
  PetscInt      *off, *adj;
  PetscInt      *neighborCells, *tmpClosure;
  PetscInt       maxConeSize, maxSupportSize, maxClosure, maxNeighbors;
  PetscInt       dim, depth, cStart, cEnd, c, numCells, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
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
        if (numFaceCases >= maxFaceCases) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
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
        } else SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not recognize number of face vertices for %D corners", corners);
        ++numFaceCases;
      }
    }
  }
  maxClosure   = 2*PetscMax(pow((PetscReal) maxConeSize, depth)+1, pow((PetscReal) maxSupportSize, depth)+1);
  maxNeighbors = pow((PetscReal) maxConeSize, depth)*pow((PetscReal) maxSupportSize, depth)+1;
  ierr = PetscMalloc2(maxNeighbors,PetscInt,&neighborCells,maxClosure,PetscInt,&tmpClosure);CHKERRQ(ierr);
  ierr = PetscMalloc((numCells+1) * sizeof(PetscInt), &off);CHKERRQ(ierr);
  ierr = PetscMemzero(off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
  /* Count neighboring cells */
  for(cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = maxNeighbors, n;

    ierr = DMComplexGetAdjacencySingleLevel_Private(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for(n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2] = {cell, neighborCells[n]};
      PetscBool       found       = depth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize;
      const PetscInt *meet;

      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        ierr = DMComplexMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for(f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
      }
      if (found) {
        ++off[cell-cStart+1];
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
      PetscInt numNeighbors = maxNeighbors, n;
      PetscInt cellOffset   = 0;

      ierr = DMComplexGetAdjacencySingleLevel_Private(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for(n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2] = {cell, neighborCells[n]};
        PetscBool       found       = depth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize;
        const PetscInt *meet;

        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          ierr = DMComplexMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for(f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
        }
        if (found) {
          adj[off[cell-cStart]+cellOffset] = neighborCells[n];
          ++cellOffset;
        }
      }
    }
  }
  ierr = PetscFree2(neighborCells,tmpClosure);CHKERRQ(ierr);
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
    if (ierr) SETERRQ1(comm, PETSC_ERR_LIB, "Error in Chaco library: %s", msgLog);
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
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
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
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "ParMetis not yet supported");
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
  } else SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid partition height %D", height);
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
      PetscInt  point   = partArray[offset+p], closureSize, c;
      PetscInt *closure = PETSC_NULL;

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
      for(c = 0; c < closureSize; ++c) {
        partPoints[partSize+c] = closure[c*2];
      }
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
      PetscInt  point   = partArray[offset+p], closureSize, c;
      PetscInt *closure = PETSC_NULL;

      /* TODO Include support for height > 0 case */
      ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Merge into existing points */
      for(c = 0; c < closureSize; ++c) {
        partPoints[partSize+c] = closure[c*2];
      }
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
. parallelMesh - The distributed DMComplex object, or PETSC_NULL

  Note: If the mesh was not distributed, the return value is PETSC_NULL

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
  ierr = DMComplexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
  ierr = DMComplexGetConeOrientations(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
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
    PetscInt        pStart, pEnd;

    ierr = DMComplexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
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
    ierr = PetscSFSetGraph(pmesh->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
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
  outputCtx->numberofedges = 0;
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
#define __FUNCT__ "DMComplexInterpolate_2D"
PetscErrorCode DMComplexInterpolate_2D(DM dm, DM *dmInt)
{
  DM             idm;
  DM_Complex    *mesh;
  PetscInt      *off;
  const PetscInt numCorners = 3;
  PetscInt       dim, numCells, cStart, cEnd, c, numVertices, vStart, vEnd;
  PetscInt       numEdges, firstEdge, edge, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  firstEdge   = numCells + numVertices;
  numEdges    = 0 ;
  /* Count edges using algorithm from CreateNeighborCSR */
  ierr = DMComplexCreateNeighborCSR(dm, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
  if (off) {
    numEdges = off[numCells]/2;
    /* Account for boundary edges: \sum_c 3 - neighbors = 3*numCells - totalNeighbors */
    numEdges += 3*numCells - off[numCells];
  }
  /* Create interpolated mesh */
  ierr = DMCreate(((PetscObject) dm)->comm, &idm);CHKERRQ(ierr);
  ierr = DMSetType(idm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(idm, dim);CHKERRQ(ierr);
  ierr = DMComplexSetChart(idm, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
  for(c = 0; c < numCells; ++c) {
    ierr = DMComplexSetConeSize(idm, c, numCorners);CHKERRQ(ierr);
  }
  for(e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMComplexSetConeSize(idm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  for(c = 0, edge = firstEdge; c < numCells; ++c) {
    const PetscInt *faces;
    PetscInt        numFaces, faceSize, f;

    ierr = DMComplexGetFaces(dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
    if (faceSize != 2) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
    for(f = 0; f < numFaces; ++f) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for(e = firstEdge; e < edge; ++e) {
        const PetscInt *cone;

        ierr = DMComplexGetCone(idm, e, &cone);CHKERRQ(ierr);
        if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
            ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMComplexSetCone(idm, edge, &faces[f*faceSize]);CHKERRQ(ierr);
        ++edge;
      }
      ierr = DMComplexInsertCone(idm, c, f, e);CHKERRQ(ierr);
    }
  }
  if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
  ierr = PetscFree(off);CHKERRQ(ierr);
  ierr = DMComplexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMComplexStratify(idm);CHKERRQ(ierr);
  mesh = (DM_Complex *) (idm)->data;
  for(c = 0; c < numCells; ++c) {
    const PetscInt *cone, *faces;
    PetscInt        coneSize, coff, numFaces, faceSize, f;

    ierr = DMComplexGetConeSize(idm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(idm, c, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
    ierr = DMComplexGetFaces(dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
    if (coneSize != numFaces) SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);
    for(f = 0; f < numFaces; ++f) {
      const PetscInt *econe;
      PetscInt        esize;

      ierr = DMComplexGetConeSize(idm, cone[f], &esize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(idm, cone[f], &econe);CHKERRQ(ierr);
      if (esize != 2) SETERRQ2(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);
      if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+f] = 0;
      } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
        /* Start at index 1, and reverse orientation */
        mesh->coneOrientations[coff+f] = -(1+1);
      }
    }
  }
  *dmInt  = idm;
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
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
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
      DM idm;

      ierr = DMComplexInterpolate_2D(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
      mesh = (DM_Complex *) (idm)->data;
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
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMComplexRefine_Triangle"
PetscErrorCode DMComplexRefine_Triangle(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm             comm = ((PetscObject) dm)->comm;
  DM_Complex          *mesh = (DM_Complex *) dm->data;
  PetscInt             dim  = 2;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(mesh->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mesh->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;
  in.trianglearealist  = (double *) maxVolumes;
  if (in.numberoftriangles > 0) {
    ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 4) && (closureSize != 7)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a triangle, %D vertices in closure", closureSize);
      for(v = 0; v < 3; ++v) {
        in.trianglelist[idx*in.numberofcorners + v] = closure[(v+closureSize-3)*2] - vStart;
      }
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
    ierr = PetscStrcpy(args, "pqezQra");CHKERRQ(ierr);
    triangulate(args, &in, &out, PETSC_NULL);
  }
  ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
  ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.trianglelist);CHKERRQ(ierr);

  ierr = DMCreate(comm, dmRefined);CHKERRQ(ierr);
  ierr = DMSetType(*dmRefined, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dmRefined, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dmRefined)->data;
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    int           *cells       = out.trianglelist;
    double        *meshCoords  = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt       coordSize, c, e;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dmRefined, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dmRefined, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dmRefined);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[3] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells};

      ierr = DMComplexSetCone(*dmRefined, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dmRefined);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dmRefined);CHKERRQ(ierr);

    if (interpolate) {
      DM idm;

      ierr = DMComplexInterpolate_2D(*dmRefined, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dmRefined);CHKERRQ(ierr);
      *dmRefined = idm;
      mesh = (DM_Complex *) (idm)->data;
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
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      for(e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
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

#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#undef __FUNCT__
#define __FUNCT__ "DMComplexGenerate_Tetgen"
PetscErrorCode DMComplexGenerate_Tetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm = ((PetscObject) boundary)->comm;
  DM_Complex    *bd   = (DM_Complex *) boundary->data;
  const PetscInt dim  = 3;
  ::tetgenio     in;
  ::tetgenio     out;
  PetscInt       vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr  = DMComplexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];
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
  ierr  = DMComplexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in.numberoffacets = fEnd - fStart;
  if (in.numberoffacets > 0) {
    in.facetlist       = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    for(f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = PETSC_NULL, numPoints, p, numVertices = 0, v;

      in.facetlist[idx].numberofpolygons = 1;
      in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
      in.facetlist[idx].numberofholes    = 0;
      in.facetlist[idx].holelist         = NULL;

      ierr = DMComplexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for(p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) {
          points[numVertices++] = point;
        }
      }

      tetgenio::polygon *poly = in.facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      poly->vertexlist       = new int[poly->numberofvertices];
      for(v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", f, &in.facetmarkerlist[idx]);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    ::tetrahedralize(args, &in, &out);
  }
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dm)->data;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    int           *cells       = out.tetrahedronlist;
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
      PetscInt cone[4] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells, cells[c*numCorners+3]+numCells};

      ierr = DMComplexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dm);CHKERRQ(ierr);
    if (interpolate) {
      DM        imesh;
      PetscInt *off;
      PetscInt  firstFace = numCells+numVertices, numFaces = 0, face, f, firstEdge, numEdges = 0, edge, e;

      SETERRQ(comm, PETSC_ERR_SUP, "Interpolation is not yet implemented in 3D");
      /* TODO: Rewrite algorithm here to do all meets with neighboring cells and return counts */
      /* Count faces using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dm, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
      if (off) {
        numFaces = off[numCells]/2;
        /* Account for boundary faces: \sum_c 4 - neighbors = 4*numCells - totalNeighbors */
        numFaces += 4*numCells - off[numCells];
      }
      firstEdge = firstFace+numFaces;
      /* Create interpolated mesh */
      ierr = DMCreate(comm, &imesh);CHKERRQ(ierr);
      ierr = DMSetType(imesh, DMCOMPLEX);CHKERRQ(ierr);
      ierr = DMComplexSetDimension(imesh, dim);CHKERRQ(ierr);
      ierr = DMComplexSetChart(imesh, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        ierr = DMComplexSetConeSize(imesh, c, numCorners);CHKERRQ(ierr);
      }
      for(f = firstFace; f < firstFace+numFaces; ++f) {
        ierr = DMComplexSetConeSize(imesh, f, 3);CHKERRQ(ierr);
      }
      for(e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMComplexSetConeSize(imesh, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(imesh);CHKERRQ(ierr);
      for(c = 0, face = firstFace; c < numCells; ++c) {
        const PetscInt *faces;
        PetscInt        numFaces, faceSize, f;

        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (faceSize != 2) SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
        for(f = 0; f < numFaces; ++f) {
          PetscBool found = PETSC_FALSE;

          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
                ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
              found = PETSC_TRUE;
              break;
            }
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, &faces[f*faceSize]);CHKERRQ(ierr);
            ++edge;
          }
          ierr = DMComplexInsertCone(imesh, c, f, e);CHKERRQ(ierr);
        }
      }
      if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      mesh = (DM_Complex *) (imesh)->data;
      for(c = 0; c < numCells; ++c) {
        const PetscInt *cone, *faces;
        PetscInt        coneSize, coff, numFaces, faceSize, f;

        ierr = DMComplexGetConeSize(imesh, c, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(imesh, c, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (coneSize != numFaces) SETERRQ3(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);
        for(f = 0; f < numFaces; ++f) {
          const PetscInt *econe;
          PetscInt        esize;

          ierr = DMComplexGetConeSize(imesh, cone[f], &esize);CHKERRQ(ierr);
          ierr = DMComplexGetCone(imesh, cone[f], &econe);CHKERRQ(ierr);
          if (esize != 2) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);
          if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
            /* Correctly oriented */
            mesh->coneOrientations[coff+f] = 0;
          } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
            /* Start at index 1, and reverse orientation */
            mesh->coneOrientations[coff+f] = -(1+1);
          }
        }
      }
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = imesh;
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
      coords[v*dim+2] = meshCoords[v*dim+2];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for(e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
      for(f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexJoinPoints(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dm, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexRefine_Tetgen"
PetscErrorCode DMComplexRefine_Tetgen(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  const PetscInt dim  = 3;
  ::tetgenio     in;
  ::tetgenio     out;
  PetscInt       vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];
    ierr = VecGetArray(mesh->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(mesh->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mesh->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in.numberofcorners       = 4;
  in.numberoftetrahedra    = cEnd - cStart;
  in.tetrahedronvolumelist = (double *) maxVolumes;
  if (in.numberoftetrahedra > 0) {
    in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
    for(c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for(v = 0; v < 4; ++v) {
        in.tetrahedronlist[idx*in.numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
    }
  }
  // TODO: Put in boundary faces with markers
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    //ierr = PetscStrcpy(args, "qezQra");CHKERRQ(ierr);
    ierr = PetscStrcpy(args, "qezraVVVV");CHKERRQ(ierr);
    ::tetrahedralize(args, &in, &out);
  }
  in.tetrahedronvolumelist = NULL;

  ierr = DMCreate(comm, dmRefined);CHKERRQ(ierr);
  ierr = DMSetType(*dmRefined, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dmRefined, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dmRefined)->data;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    int           *cells       = out.tetrahedronlist;
    double        *meshCoords  = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt       coordSize, c, e;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dmRefined, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dmRefined, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dmRefined);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[4] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells, cells[c*numCorners+3]+numCells};

      ierr = DMComplexSetCone(*dmRefined, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dmRefined);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dmRefined);CHKERRQ(ierr);

    if (interpolate) {
      DM        imesh;
      PetscInt *off;
      PetscInt  firstEdge = numCells+numVertices, numEdges, edge, e;

      SETERRQ(comm, PETSC_ERR_SUP, "Interpolation is not yet implemented in 3D");
      /* Count edges using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dmRefined, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
      if (off) {
        numEdges = off[numCells]/2;
        /* Account for boundary edges: \sum_c 3 - neighbors = 3*numCells - totalNeighbors */
        numEdges += 3*numCells - off[numCells];
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
      for(c = 0, edge = firstEdge; c < numCells; ++c) {
        const PetscInt *faces;
        PetscInt        numFaces, faceSize, f;

        ierr = DMComplexGetFaces(*dmRefined, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (faceSize != 2) SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
        for(f = 0; f < numFaces; ++f) {
          PetscBool found = PETSC_FALSE;

          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
                ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
              found = PETSC_TRUE;
              break;
            }
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, &faces[f*faceSize]);CHKERRQ(ierr);
            ++edge;
          }
          ierr = DMComplexInsertCone(imesh, c, f, e);CHKERRQ(ierr);
        }
      }
      if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      mesh = (DM_Complex *) (imesh)->data;
      for(c = 0; c < numCells; ++c) {
        const PetscInt *cone, *faces;
        PetscInt        coneSize, coff, numFaces, faceSize, f;

        ierr = DMComplexGetConeSize(imesh, c, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(imesh, c, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
        ierr = DMComplexGetFaces(*dmRefined, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (coneSize != numFaces) SETERRQ3(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);
        for(f = 0; f < numFaces; ++f) {
          const PetscInt *econe;
          PetscInt        esize;

          ierr = DMComplexGetConeSize(imesh, cone[f], &esize);CHKERRQ(ierr);
          ierr = DMComplexGetCone(imesh, cone[f], &econe);CHKERRQ(ierr);
          if (esize != 2) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);
          if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
            /* Correctly oriented */
            mesh->coneOrientations[coff+f] = 0;
          } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
            /* Start at index 1, and reverse orientation */
            mesh->coneOrientations[coff+f] = -(1+1);
          }
        }
      }
      ierr = DMDestroy(dmRefined);CHKERRQ(ierr);
      *dmRefined  = imesh;
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
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt f;

      for(e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
      for(f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexJoinPoints(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}
#endif

#ifdef PETSC_HAVE_CTETGEN
#undef __FUNCT__
#define __FUNCT__ "DMComplexGenerate_CTetgen"
PetscErrorCode DMComplexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm = ((PetscObject) boundary)->comm;
  DM_Complex    *bd   = (DM_Complex *) boundary->data;
  const PetscInt dim  = 3;
  PLC           *in, *out;
  PetscInt       verbose = 0, vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(((PetscObject) boundary)->prefix, "-ctetgen_verbose", &verbose, PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in->numberofpoints*dim * sizeof(PetscReal), &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberofpoints     * sizeof(int),       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(bd->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(bd->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(bd->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc(in->numberoffacets * sizeof(facet), &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberoffacets * sizeof(int),   &in->facetmarkerlist);CHKERRQ(ierr);
    for(f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = PETSC_NULL, numPoints, p, numVertices = 0, v, m;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;
      ierr = PetscMalloc(in->facetlist[idx].numberofpolygons * sizeof(polygon), &in->facetlist[idx].polygonlist);CHKERRQ(ierr);
      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = PETSC_NULL;

      ierr = DMComplexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for(p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) {
          points[numVertices++] = point;
        }
      }

      poly = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr = PetscMalloc(poly->numberofvertices * sizeof(int), &poly->vertexlist);CHKERRQ(ierr);
      for(v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", f, &m);CHKERRQ(ierr);
      in->facetmarkerlist[idx] = (int) m;
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose;
    ierr = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dm)->data;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    int           *cells       = out->tetrahedronlist;
    PetscReal     *meshCoords  = out->pointlist;
    PetscInt       coordSize, c;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[4] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells, cells[c*numCorners+3]+numCells};

      ierr = DMComplexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dm);CHKERRQ(ierr);
    if (interpolate) {
      DM        imesh;
      PetscInt *off;
      PetscInt  firstFace = numCells+numVertices, numFaces = 0, f, firstEdge, numEdges = 0, edge, e;

      SETERRQ(comm, PETSC_ERR_SUP, "Interpolation is not yet implemented in 3D");
      /* TODO: Rewrite algorithm here to do all meets with neighboring cells and return counts */
      /* Count faces using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dm, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
      if (off) {
        numFaces = off[numCells]/2;
        /* Account for boundary faces: \sum_c 4 - neighbors = 4*numCells - totalNeighbors */
        numFaces += 4*numCells - off[numCells];
      }
      firstEdge = firstFace+numFaces;
      /* Create interpolated mesh */
      ierr = DMCreate(comm, &imesh);CHKERRQ(ierr);
      ierr = DMSetType(imesh, DMCOMPLEX);CHKERRQ(ierr);
      ierr = DMComplexSetDimension(imesh, dim);CHKERRQ(ierr);
      ierr = DMComplexSetChart(imesh, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        ierr = DMComplexSetConeSize(imesh, c, numCorners);CHKERRQ(ierr);
      }
      for(f = firstFace; f < firstFace+numFaces; ++f) {
        ierr = DMComplexSetConeSize(imesh, f, 3);CHKERRQ(ierr);
      }
      for(e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMComplexSetConeSize(imesh, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(imesh);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        const PetscInt *faces;
        PetscInt        numFaces, faceSize, f;

        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (faceSize != 2) SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
        for(f = 0; f < numFaces; ++f) {
          PetscBool found = PETSC_FALSE;

          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
                ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
              found = PETSC_TRUE;
              break;
            }
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, &faces[f*faceSize]);CHKERRQ(ierr);
            ++edge;
          }
          ierr = DMComplexInsertCone(imesh, c, f, e);CHKERRQ(ierr);
        }
      }
      if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      mesh = (DM_Complex *) (imesh)->data;
      for(c = 0; c < numCells; ++c) {
        const PetscInt *cone, *faces;
        PetscInt        coneSize, coff, numFaces, faceSize, f;

        ierr = DMComplexGetConeSize(imesh, c, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(imesh, c, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (coneSize != numFaces) SETERRQ3(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);
        for(f = 0; f < numFaces; ++f) {
          const PetscInt *econe;
          PetscInt        esize;

          ierr = DMComplexGetConeSize(imesh, cone[f], &esize);CHKERRQ(ierr);
          ierr = DMComplexGetCone(imesh, cone[f], &econe);CHKERRQ(ierr);
          if (esize != 2) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);
          if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
            /* Correctly oriented */
            mesh->coneOrientations[coff+f] = 0;
          } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
            /* Start at index 1, and reverse orientation */
            mesh->coneOrientations[coff+f] = -(1+1);
          }
        }
      }
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = imesh;
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
      coords[v*dim+2] = meshCoords[v*dim+2];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for(e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
      for(f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexJoinPoints(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dm, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexRefine_CTetgen"
PetscErrorCode DMComplexRefine_CTetgen(DM dm, PetscReal *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  const PetscInt dim  = 3;
  PLC           *in, *out;
  PetscInt       verbose = 0, vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(((PetscObject) dm)->prefix, "-ctetgen_verbose", &verbose, PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in->numberofpoints*dim * sizeof(PetscReal), &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberofpoints     * sizeof(int),       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(mesh->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(mesh->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    ierr = PetscMalloc(in->numberoftetrahedra*in->numberofcorners * sizeof(int), &in->tetrahedronlist);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for(v = 0; v < 4; ++v) {
        in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = dm; /* Should go away */
    t.refine    = 1;
    t.varvolume = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose; /* Change this */
    ierr = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }

  ierr = DMCreate(comm, dmRefined);CHKERRQ(ierr);
  ierr = DMSetType(*dmRefined, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dmRefined, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dmRefined)->data;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    int           *cells       = out->tetrahedronlist;
    PetscReal     *meshCoords  = out->pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt       coordSize, c, e;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dmRefined, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dmRefined, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dmRefined);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[4] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells, cells[c*numCorners+3]+numCells};

      ierr = DMComplexSetCone(*dmRefined, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dmRefined);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dmRefined);CHKERRQ(ierr);

    if (interpolate) {
      DM        imesh;
      PetscInt *off;
      PetscInt  firstEdge = numCells+numVertices, numEdges, edge, e;

      SETERRQ(comm, PETSC_ERR_SUP, "Interpolation is not yet implemented in 3D");
      /* Count edges using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dmRefined, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
      if (off) {
        numEdges = off[numCells]/2;
        /* Account for boundary edges: \sum_c 3 - neighbors = 3*numCells - totalNeighbors */
        numEdges += 3*numCells - off[numCells];
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
      for(c = 0, edge = firstEdge; c < numCells; ++c) {
        const PetscInt *faces;
        PetscInt        numFaces, faceSize, f;

        ierr = DMComplexGetFaces(*dmRefined, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (faceSize != 2) SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
        for(f = 0; f < numFaces; ++f) {
          PetscBool found = PETSC_FALSE;

          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
                ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
              found = PETSC_TRUE;
              break;
            }
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, &faces[f*faceSize]);CHKERRQ(ierr);
            ++edge;
          }
          ierr = DMComplexInsertCone(imesh, c, f, e);CHKERRQ(ierr);
        }
      }
      if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      mesh = (DM_Complex *) (imesh)->data;
      for(c = 0; c < numCells; ++c) {
        const PetscInt *cone, *faces;
        PetscInt        coneSize, coff, numFaces, faceSize, f;

        ierr = DMComplexGetConeSize(imesh, c, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(imesh, c, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
        ierr = DMComplexGetFaces(*dmRefined, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (coneSize != numFaces) SETERRQ3(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);
        for(f = 0; f < numFaces; ++f) {
          const PetscInt *econe;
          PetscInt        esize;

          ierr = DMComplexGetConeSize(imesh, cone[f], &esize);CHKERRQ(ierr);
          ierr = DMComplexGetCone(imesh, cone[f], &econe);CHKERRQ(ierr);
          if (esize != 2) SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);
          if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
            /* Correctly oriented */
            mesh->coneOrientations[coff+f] = 0;
          } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
            /* Start at index 1, and reverse orientation */
            mesh->coneOrientations[coff+f] = -(1+1);
          }
        }
      }
      ierr = DMDestroy(dmRefined);CHKERRQ(ierr);
      *dmRefined  = imesh;
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
      coords[v*dim+2] = meshCoords[v*dim+2];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt f;

      for(e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
      for(f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexJoinPoints(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
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
. name - The mesh generation package name
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The DMComplex object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMComplexCreate(), DMRefine()
@*/
PetscErrorCode DMComplexGenerate(DM boundary, const char name[], PetscBool interpolate, DM *mesh)
{
  PetscInt       dim;
  char           genname[1024];
  PetscBool      isTriangle = PETSC_FALSE, isTetgen = PETSC_FALSE, isCTetgen = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(boundary, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(boundary, interpolate, 2);
  ierr = DMComplexGetDimension(boundary, &dim);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) boundary)->prefix, "-dm_complex_generator", genname, 1024, &flg);CHKERRQ(ierr);
  if (flg) {name = genname;}
  if (name) {
    ierr = PetscStrcmp(name, "triangle", &isTriangle);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "tetgen",   &isTetgen);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "ctetgen",  &isCTetgen);CHKERRQ(ierr);
  }
  switch(dim) {
  case 1:
    if (!name || isTriangle) {
#ifdef PETSC_HAVE_TRIANGLE
      ierr = DMComplexGenerate_Triangle(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Mesh generation needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
    } else SETERRQ1(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Unknown 2D mesh generation package %s", name);
    break;
  case 2:
    if (!name || isCTetgen) {
#ifdef PETSC_HAVE_CTETGEN
      ierr = DMComplexGenerate_CTetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --with-ctetgen.");
#endif
    } else if (isTetgen) {
#ifdef PETSC_HAVE_TETGEN
      ierr = DMComplexGenerate_Tetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Tetgen needs external package support.\nPlease reconfigure with --with-c-language=cxx --download-tetgen.");
#endif
    } else SETERRQ1(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Unknown 3D mesh generation package %s", name);
    break;
  default:
    SETERRQ1(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Mesh generation for a dimension %d boundary is not supported.", dim);
  }
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
  PetscInt       dim, cStart, cEnd, c;
  char           genname[1024], *name = PETSC_NULL;
  PetscBool      isTriangle = PETSC_FALSE, isTetgen = PETSC_FALSE, isCTetgen = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetRefinementLimit(dm, &refinementLimit);CHKERRQ(ierr);
  if (refinementLimit == 0.0) PetscFunctionReturn(0);
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) dm)->prefix, "-dm_complex_generator", genname, 1024, &flg);CHKERRQ(ierr);
  if (flg) {name = genname;}
  if (name) {
    ierr = PetscStrcmp(name, "triangle", &isTriangle);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "tetgen",   &isTetgen);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "ctetgen",  &isCTetgen);CHKERRQ(ierr);
  }
  switch(dim) {
  case 2:
    if (!name || isTriangle) {
#ifdef PETSC_HAVE_TRIANGLE
      double *maxVolumes;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for(c = 0; c < cEnd-cStart; ++c) {
        maxVolumes[c] = refinementLimit;
      }
      ierr = DMComplexRefine_Triangle(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Mesh refinement needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
    } else SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unknown 2D mesh generation package %s", name);
    break;
  case 3:
    if (!name || isCTetgen) {
#ifdef PETSC_HAVE_CTETGEN
      PetscReal *maxVolumes;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(PetscReal), &maxVolumes);CHKERRQ(ierr);
      for(c = 0; c < cEnd-cStart; ++c) {
        maxVolumes[c] = refinementLimit;
      }
      ierr = DMComplexRefine_CTetgen(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --with-ctetgen.");
#endif
    } else if (isTetgen) {
#ifdef PETSC_HAVE_TETGEN
      double *maxVolumes;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for(c = 0; c < cEnd-cStart; ++c) {
        maxVolumes[c] = refinementLimit;
      }
      ierr = DMComplexRefine_Tetgen(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Tetgen needs external package support.\nPlease reconfigure with --with-c-language=cxx --download-tetgen.");
#endif
    } else SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unknown 3D mesh generation package %s", name);
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Mesh refinement in dimension %d is not supported.", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetDepth"
PetscErrorCode DMComplexGetDepth(DM dm, PetscInt *depth) {
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depth, 2);
  ierr = DMComplexGetLabelSize(dm, "depth", &d);CHKERRQ(ierr);
  *depth = d-1;
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
  ierr = DMComplexHasLabel(dm, "depth", &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);
  /* We should have a generic GetLabel() and a Label class */
  while(next) {
    ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
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
  ierr = DMComplexHasLabel(dm, "depth", &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);
  /* We should have a generic GetLabel() and a Label class */
  while(next) {
    ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
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
  for(f = 0; f <= numFields; ++f) {maxConstraints[f] = 0;}
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
        if (cDof > maxConstraints[numFields]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB, "Likely memory corruption, point %D cDof %D > maxConstraints %D", p, cDof, maxConstraints);
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
          if (cDof != numConst) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of field constraints %D should be %D", numConst, cDof);
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
#define __FUNCT__ "DMComplexGetConeOrientations"
PetscErrorCode DMComplexGetConeOrientations(DM dm, PetscInt *coneOrientations[]) {
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (coneOrientations) *coneOrientations = mesh->coneOrientations;
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
        remote[l].index = goff+d - ranges[rank];
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
#if defined(PETSC_HAVE_MPI_REPLACE)
    op = MPI_REPLACE; break;
#else
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No support for INSERT_VALUES without an MPI-2 implementation");
#endif
  case ADD_VALUES:
    op = MPI_SUM; break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
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
#if defined(PETSC_HAVE_MPI_REPLACE)
    op = MPI_REPLACE; break;
#else
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No support for INSERT_VALUES without an MPI-2 implementation");
#endif
  case ADD_VALUES:
    op = MPI_SUM; break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
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
  if (mode == ADD_VALUES) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
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
  if (mode == ADD_VALUES) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
  ierr = DMComplexGetDefaultSF(dm, &sf);CHKERRQ(ierr);
  ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
  ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
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
PetscErrorCode DMComplexGetLocalJacobian(DM dm, PetscErrorCode (**lj)(DM, Vec, Mat, Mat, void *))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lj) *lj = mesh->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetLocalJacobian"
PetscErrorCode DMComplexSetLocalJacobian(DM dm, PetscErrorCode (*lj)(DM, Vec, Mat,  Mat, void *))
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
@*/
PetscErrorCode DMComplexVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar *values[]) {
  PetscScalar    *array, *vArray;
  PetscInt       *points = PETSC_NULL;
  PetscInt        offsets[32];
  PetscInt        numFields, size, numPoints, pStart, pEnd, p, q, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {
    ierr = DMComplexGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 32) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 32", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0, size = 0; p < numPoints*2; p += 2) {
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
  for(p = 0; p < numPoints*2; p += 2) {
    PetscInt     o = points[p+1];
    PetscInt     dof, off, d;
    PetscScalar *varr;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
    varr = &vArray[off];
    if (numFields) {
      PetscInt fdof, foff, fcomp, f, c;

      for(f = 0, foff = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
        if (o >= 0) {
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
      if (o >= 0) {
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
  PetscInt       *points = PETSC_NULL;
  PetscInt        offsets[32];
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
  if (numFields > 32) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 32", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0; p < numPoints*2; p += 2) {
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
      for(p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, insert, PETSC_FALSE, o, values, array);
      } break;
    case INSERT_ALL_VALUES:
      for(p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, insert, PETSC_TRUE,  o, values, array);
      } break;
    case ADD_VALUES:
      for(p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, add,    PETSC_FALSE, o, values, array);
      } break;
    case ADD_ALL_VALUES:
      for(p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, add,    PETSC_TRUE,  o, values, array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
    }
  } else {
    switch(mode) {
    case INSERT_VALUES:
      for(p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_FALSE, o, &values[off], array);
      } break;
    case INSERT_ALL_VALUES:
      for(p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_TRUE,  o, &values[off], array);
      } break;
    case ADD_VALUES:
      for(p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_FALSE, o, &values[off], array);
      } break;
    case ADD_ALL_VALUES:
      for(p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_TRUE,  o, &values[off], array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
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
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%D]mat for sieve point %D\n", rank, point);CHKERRQ(ierr);
  for(i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%D]mat indices[%D] = %D\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  for(i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%D]", rank);CHKERRQ(ierr);
    for(j = 0; j < numIndices; j++) {
#ifdef PETSC_USE_COMPLEX
      ierr = PetscPrintf(PETSC_COMM_SELF, " (%G,%G)", PetscRealPart(values[i*numIndices+j]), PetscImaginaryPart(values[i*numIndices+j]));CHKERRQ(ierr);
#else
      ierr = PetscPrintf(PETSC_COMM_SELF, " %G", values[i*numIndices+j]);CHKERRQ(ierr);
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
  DM_Complex     *mesh   = (DM_Complex *) dm->data;
  PetscInt       *points = PETSC_NULL;
  PetscInt       *indices;
  PetscInt        offsets[32];
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
  if (numFields > 32) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 32", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for(p = 0, numIndices = 0; p < numPoints*2; p += 2) {
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
    for(p = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPointFields_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, o, indices);
    }
  } else {
    for(p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPoint_private(section, points[p], dof, globalOff < 0 ? -(globalOff+1) : globalOff, PETSC_FALSE, o, &indices[off]);
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
    ierr2 = PetscPrintf(PETSC_COMM_SELF, "[%D]ERROR in DMComplexMatSetClosure\n", rank);CHKERRQ(ierr2);
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
    /* ??? This does not work with CTetGen: The minus sign is here since I orient the first face to get the outward normal */
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
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported number of cell vertices %D for element geometry computation", maxConeSize);
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
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported number of cell vertices %D for element geometry computation", maxConeSize);
    }
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Unsupported dimension %D for element geometry computation", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetFaceOrientation"
PetscErrorCode DMComplexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented) {
  MPI_Comm       comm      = ((PetscObject) dm)->comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;
  PetscErrorCode ierr;

  ierr = DMComplexGetDimension(dm, &cellDim);CHKERRQ(ierr);
  if (debug) {PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners);CHKERRQ(ierr);}

  if (cellDim == numCorners-1) {
    /* Simplices */
    faceSize  = numCorners-1;
    posOrient = !(oppositeVertex%2) ? PETSC_TRUE : PETSC_FALSE;
  } else if (cellDim == 1 && numCorners == 3) {
    /* Quadratic line */
    faceSize  = 1;
    posOrient = PETSC_TRUE;
  } else if (cellDim == 2 && numCorners == 4) {
    /* Quads */
    faceSize  = 2;
    if ((indices[1] > indices[0]) && (indices[1] - indices[0] == 1)) {
      posOrient = PETSC_TRUE;
    } else if ((indices[0] == 3) && (indices[1] == 0)) {
      posOrient = PETSC_TRUE;
    } else {
      if (((indices[0] > indices[1]) && (indices[0] - indices[1] == 1)) || ((indices[0] == 0) && (indices[1] == 3))) {
        posOrient = PETSC_FALSE;
      } else {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossedge");
      }
    }
  } else if (cellDim == 2 && numCorners == 6) {
    /* Quadratic triangle (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeTri = 3;
    PetscInt  sortedIndices[3], i, iFace;
    PetscBool found = PETSC_FALSE;
    PetscInt  faceVerticesTriSorted[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 3,  5, /* left */
    };
    PetscInt  faceVerticesTri[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 5,  3, /* left */
    };

    faceSize = faceSizeTri;
    for(i = 0; i < faceSizeTri; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTri, sortedIndices);CHKERRQ(ierr);
    for(iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeTri;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTriSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTriSorted[ii+1])) {
        for(fVertex = 0; fVertex < faceSizeTri; ++fVertex) {
          for(cVertex = 0; cVertex < faceSizeTri; ++cVertex) {
            if (indices[cVertex] == faceVerticesTri[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tri crossface");
    if (posOriented) {*posOriented = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (cellDim == 2 && numCorners == 9) {
    /* Quadratic quad (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeQuad = 3;
    PetscInt  sortedIndices[3], i, iFace;
    PetscBool found = PETSC_FALSE;
    PetscInt  faceVerticesQuadSorted[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      0, 3,  7, /* left */
    };
    PetscInt  faceVerticesQuad[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      3, 0,  7, /* left */
    };

    faceSize = faceSizeQuad;
    for(i = 0; i < faceSizeQuad; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuad, sortedIndices);CHKERRQ(ierr);
    for(iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuad;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadSorted[ii+1])) {
        for(fVertex = 0; fVertex < faceSizeQuad; ++fVertex) {
          for(cVertex = 0; cVertex < faceSizeQuad; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuad[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossface");
    if (posOriented) {*posOriented = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 8) {
    /* Hexes
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

          7---6
         /|  /|
        4---5 |
        | 3-|-2
        |/  |/
        0---1

        Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeHex = 4;
    PetscInt  sortedIndices[4], i, iFace;
    PetscBool found = PETSC_FALSE;
    PetscInt faceVerticesHexSorted[24] = {
      0, 1, 2, 3,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 1, 4, 5,  /* front */
      1, 2, 5, 6,  /* right */
      2, 3, 6, 7,  /* back */
      0, 3, 4, 7,  /* left */
    };
    PetscInt faceVerticesHex[24] = {
      3, 2, 1, 0,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 1, 5, 4,  /* front */
      1, 2, 6, 5,  /* right */
      2, 3, 7, 6,  /* back */
      3, 0, 4, 7,  /* left */
    };

    faceSize = faceSizeHex;
    for(i = 0; i < faceSizeHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeHex, sortedIndices);CHKERRQ(ierr);
    for(iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesHexSorted[ii+3])) {
        for(fVertex = 0; fVertex < faceSizeHex; ++fVertex) {
          for(cVertex = 0; cVertex < faceSizeHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) {*posOriented = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 10) {
    /* Quadratic tet */
    /* Faces are determined by the first 3 vertices (corners of faces) */
    const PetscInt faceSizeTet = 6;
    PetscInt  sortedIndices[6], i, iFace;
    PetscBool found = PETSC_FALSE;
    PetscInt faceVerticesTetSorted[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 3, 4,  6, 7, 9,  /* front */
      1, 4, 5,  7, 8, 9,  /* right */
      2, 3, 5,  6, 8, 9,  /* left */
    };
    PetscInt faceVerticesTet[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 4, 3,  6, 7, 9,  /* front */
      1, 5, 4,  7, 8, 9,  /* right */
      2, 3, 5,  8, 6, 9,  /* left */
    };

    faceSize = faceSizeTet;
    for(i = 0; i < faceSizeTet; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTet, sortedIndices);CHKERRQ(ierr);
    for(iFace=0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeTet;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTetSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTetSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesTetSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesTetSorted[ii+3])) {
        for(fVertex = 0; fVertex < faceSizeTet; ++fVertex) {
          for(cVertex = 0; cVertex < faceSizeTet; ++cVertex) {
            if (indices[cVertex] == faceVerticesTet[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tet crossface");
    if (posOriented) {*posOriented = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 27) {
    /* Quadratic hexes (I hate this)
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

         7---6
        /|  /|
       4---5 |
       | 3-|-2
       |/  |/
       0---1

       Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeQuadHex = 9;
    PetscInt  sortedIndices[9], i, iFace;
    PetscBool found = PETSC_FALSE;
    PetscInt faceVerticesQuadHexSorted[54] = {
      0, 1, 2, 3,  8, 9, 10, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 4, 5,  8, 12, 16, 17,  22, /* front */
      1, 2, 5, 6,  9, 13, 17, 18,  21, /* right */
      2, 3, 6, 7,  10, 14, 18, 19,  23, /* back */
      0, 3, 4, 7,  11, 15, 16, 19,  20, /* left */
    };
    PetscInt faceVerticesQuadHex[54] = {
      3, 2, 1, 0,  10, 9, 8, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 5, 4,  8, 17, 12, 16,  22, /* front */
      1, 2, 6, 5,  9, 18, 13, 17,  21, /* right */
      2, 3, 7, 6,  10, 19, 14, 18,  23, /* back */
      3, 0, 4, 7,  11, 16, 15, 19,  20 /* left */
    };

    faceSize = faceSizeQuadHex;
    for(i = 0; i < faceSizeQuadHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuadHex, sortedIndices);CHKERRQ(ierr);
    for(iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuadHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesQuadHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesQuadHexSorted[ii+3])) {
        for(fVertex = 0; fVertex < faceSizeQuadHex; ++fVertex) {
          for(cVertex = 0; cVertex < faceSizeQuadHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuadHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) {SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");}
    if (posOriented) {*posOriented = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown cell type for faceOrientation().");
  if (!posOrient) {
    if (debug) {ierr = PetscPrintf(comm, "  Reversing initial face orientation\n");CHKERRQ(ierr);}
    for(f = 0; f < faceSize; ++f) {
      faceVertices[f] = origVertices[faceSize-1 - f];
    }
  } else {
    if (debug) {ierr = PetscPrintf(comm, "  Keeping initial face orientation\n");CHKERRQ(ierr);}
    for(f = 0; f < faceSize; ++f) {
      faceVertices[f] = origVertices[f];
    }
  }
  if (posOriented) {*posOriented = posOrient;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetOrientedFace"
PetscErrorCode DMComplexGetOrientedFace(DM dm, PetscInt cell, PetscInt faceSize, const PetscInt face[], PetscInt numCorners, PetscInt indices[], PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  const PetscInt *cone;
  PetscInt        coneSize, v, f, v2;
  PetscInt        oppositeVertex = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMComplexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  for(v = 0, v2 = 0; v < coneSize; ++v) {
    PetscBool found  = PETSC_FALSE;

    for(f = 0; f < faceSize; ++f) {
      if (face[f] == cone[v]) {found = PETSC_TRUE; break;}
    }
    if (found) {
      indices[v2]      = v;
      origVertices[v2] = cone[v];
      ++v2;
    } else {
      oppositeVertex = v;
    }
  }
  ierr = DMComplexGetFaceOrientation(dm, cell, numCorners, indices, oppositeVertex, origVertices, faceVertices, posOriented);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
