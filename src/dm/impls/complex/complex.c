#include <petsc-private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

/* Logging support */
PetscLogEvent DMCOMPLEX_Distribute;

#undef __FUNCT__
#define __FUNCT__ "DMComplexViewLabel_Ascii"
PetscErrorCode DMComplexViewLabel_Ascii(DM dm, const char name[], PetscViewer viewer)
{
  IS              ids;
  const PetscInt *markers;
  PetscInt        num, i;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Label '%s':\n", name);CHKERRQ(ierr);
  ierr = DMComplexGetLabelIdIS(dm, name, &ids);CHKERRQ(ierr);
  ierr = ISGetSize(ids, &num);CHKERRQ(ierr);
  ierr = ISGetIndices(ids, &markers);CHKERRQ(ierr);
  for (i = 0; i < num; ++i) {
    IS              pIS;
    const PetscInt *points;
    PetscInt        size, p;

    ierr = DMComplexGetStratumIS(dm, name, markers[i], &pIS);
    ierr = ISGetSize(pIS, &size);CHKERRQ(ierr);
    ierr = ISGetIndices(pIS, &points);CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D (%D)\n", rank, points[p], markers[i]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ids, &markers);CHKERRQ(ierr);
  ierr = ISDestroy(&ids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexView_Ascii"
PetscErrorCode DMComplexView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Complex       *mesh = (DM_Complex *) dm->data;
  DM                cdm;
  PetscSection      coordSection;
  Vec               coordinates;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
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
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for (s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D ----> %D\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "base <-- cap:\n", name);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for (c = off; c < off+dof; ++c) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D <---- %D (%D)\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(coordSection, &pStart, PETSC_NULL);CHKERRQ(ierr);
    if (pStart >= 0) {ierr = PetscSectionVecView(coordSection, coordinates, viewer);CHKERRQ(ierr);}
    ierr = DMComplexHasLabel(dm, "marker", &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      ierr = DMComplexViewLabel_Ascii(dm, "marker", viewer);CHKERRQ(ierr);
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
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off, dof, d;

      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%G", PetscRealPart(coords[off+d]));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%D) [draw,shape=circle,color=%s] {%D} --\n", v, rank, colors[rank%numColors], v);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "(0,0);\n");CHKERRQ(ierr);
    /* Plot cells */
    ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = PETSC_NULL;
      PetscInt  closureSize, firstPoint = -1;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] ", colors[rank%numColors]);CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = closure[p];

        if ((point < vStart) || (point >= vEnd)) continue;
        if (firstPoint >= 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(%D_%D)", point, rank);CHKERRQ(ierr);
        if (firstPoint < 0) firstPoint = point;
      }
      /* Why doesn't this work? ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- cycle;\n");CHKERRQ(ierr); */
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- (%D_%D);\n", firstPoint, rank);CHKERRQ(ierr);
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n\\end{center}\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh for processes ");CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
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
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMComplexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", dim);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for (d = 0; d <= dim; d++) {
        ierr = DMComplexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd = pEnd - pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", d);CHKERRQ(ierr);
        for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
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
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
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
  DMLabel        next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) {PetscFunctionReturn(0);}
  ierr = PetscSectionDestroy(&mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->cones);CHKERRQ(ierr);
  ierr = PetscFree(mesh->coneOrientations);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscFree(mesh->facesTmp);CHKERRQ(ierr);
  while(next) {
    DMLabel tmp;

    ierr = PetscFree(next->name);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumValues);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumOffsets);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumSizes);CHKERRQ(ierr);
    ierr = PetscFree(next->points);CHKERRQ(ierr);
    tmp  = next->next;
    ierr = PetscFree(next);CHKERRQ(ierr);
    next = tmp;
  }
  ierr = ISDestroy(&mesh->subpointMap);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalCellNumbers);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(mesh);CHKERRQ(ierr);
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
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone = PETSC_NULL;
      PetscInt        coneSize, c, q;

      ierr = DMComplexGetSupportSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetSupport(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        for (q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
          if (cone[c] == adj[q]) break;
        }
        if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
      }
    }
  } else {
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone = PETSC_NULL;
      PetscInt        coneSize, c, q;

      ierr = DMComplexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        for (q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
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
  for (s = 2; s < starSize*2; s += 2) {
    const PetscInt *closure = PETSC_NULL;
    PetscInt        closureSize, c, q;

    ierr = DMComplexGetTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt **) &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      for (q = 0; q < numAdj || (adj[numAdj++] = closure[c],0); ++q) {
        if (closure[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
    ierr = DMComplexRestoreTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt **) &closure);CHKERRQ(ierr);
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
  PetscSF            sf   = dm->sf, sfDof, sfAdj;
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
  maxClosureSize = (PetscInt) (2*PetscMax(pow((PetscReal) mesh->maxConeSize, depth)+1, pow((PetscReal) mesh->maxSupportSize, depth)+1));
  maxAdjSize     = (PetscInt) (pow((PetscReal) mesh->maxConeSize, depth)*pow((PetscReal) mesh->maxSupportSize, depth)+1);
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

  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q;
    PetscInt p = leaves[l], numAdj = maxAdjSize;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for (q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
      for (d = off; d < off+dof; ++d) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = maxAdjSize, adof, dof, off, d, q;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for (q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof;

      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
      for (d = off; d < off+dof; ++d) {
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
  ierr = PetscSFDestroy(&sfDof);CHKERRQ(ierr);
  /* Create leaf adjacency */
  ierr = PetscSectionSetUp(leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(leafSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscMalloc(adjSize * sizeof(PetscInt), &adj);CHKERRQ(ierr);
  ierr = PetscMemzero(adj, adjSize * sizeof(PetscInt));CHKERRQ(ierr);
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q;
    PetscInt p = leaves[l], numAdj = maxAdjSize;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for (d = off; d < off+dof; ++d) {
      PetscInt aoff, i = 0;

      ierr = PetscSectionGetOffset(leafSectionAdj, d, &aoff);CHKERRQ(ierr);
      for (q = 0; q < numAdj; ++q) {
        PetscInt  ndof, ncdof, ngoff, nd;

        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for (nd = 0; nd < ndof-ncdof; ++nd) {
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
  for (r = 0; r < adjSize; ++r) {
    rootAdj[r] = -1;
  }
  if (size > 1) {
    ierr = PetscSFGatherBegin(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sfAdj, MPIU_INT, adj, rootAdj);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfAdj);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Root adjacency indices after gather\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, PETSC_NULL);CHKERRQ(ierr);
  }
  /* Add in local adjacency indices for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = maxAdjSize, adof, dof, off, d, q;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMComplexGetAdjacency_Private(dm, p, PETSC_FALSE, tmpClosure, &numAdj, tmpAdj);CHKERRQ(ierr);
    for (d = off; d < off+dof; ++d) {
      PetscInt adof, aoff, i;

      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      i    = adof-1;
      for (q = 0; q < numAdj; ++q) {
        PetscInt ndof, ncdof, ngoff, nd;

        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for (nd = 0; nd < ndof-ncdof; ++nd) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    for (d = off; d < off+dof-cdof; ++d) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = maxAdjSize, dof, cdof, off, goff, d, q;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for (d = 0; d < dof-cdof; ++d) {
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
    for (q = 0; q < numAdj; ++q) {
      PetscInt ndof, ncdof, noff;

      /* Adjacent points may not be in the section chart */
      if ((tmpAdj[q] < pStart) || (tmpAdj[q] >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, tmpAdj[q], &noff);CHKERRQ(ierr);
      for (d = goff; d < goff+dof-cdof; ++d) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = maxAdjSize, dof, cdof, off, goff, d, q;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for (d = 0; d < dof-cdof; ++d) {
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
    for (d = goff; d < goff+dof-cdof; ++d) {
      PetscInt adof, aoff, i = 0;

      ierr = PetscSectionGetDof(sectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, d, &aoff);CHKERRQ(ierr);
      for (q = 0; q < numAdj; ++q) {
        PetscInt        ndof, ncdof, ngoff, nd;
        const PetscInt *ncind;

        /* Adjacent points may not be in the section chart */
        if ((tmpAdj[q] < pStart) || (tmpAdj[q] >= pEnd)) continue;
        ierr = PetscSectionGetDof(section, tmpAdj[q], &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, tmpAdj[q], &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(section, tmpAdj[q], &ncind);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, tmpAdj[q], &ngoff);CHKERRQ(ierr);
        for (nd = 0; nd < ndof-ncdof; ++nd, ++i) {
          cols[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd: ngoff+nd;
        }
      }
      if (i != adof) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %D != %D for dof %D (point %D)", i, adof, d, p);
    }
  }
  ierr = PetscSectionDestroy(&leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSectionAdj);CHKERRQ(ierr);
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
  /* Only loop over blocks of rows */
  if (rStart%bs || rEnd%bs) SETERRQ3(((PetscObject) A)->comm, PETSC_ERR_ARG_WRONG, "Invalid layout [%d, %d) for matrix, must be divisible by block size %d", rStart, rEnd, bs);
  for (r = rStart/bs; r < rEnd/bs; ++r) {
    const PetscInt row = r*bs;
    PetscInt numCols, cStart, c;

    ierr = PetscSectionGetDof(sectionAdj, row, &numCols);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionAdj, row, &cStart);CHKERRQ(ierr);
    for (c = cStart; c < cStart+numCols; ++c) {
      if ((cols[c] >= rStart*bs) && (cols[c] < rEnd*bs)) {
        ++dnz[r-rStart];
        if (cols[c] >= row) {++dnzu[r-rStart];}
      } else {
        ++onz[r-rStart];
        if (cols[c] >= row) {++onzu[r-rStart];}
      }
    }
  }
  if (bs > 1) {
    for (r = 0; r < locRows/bs; ++r) {
      dnz[r]  /= bs;
      onz[r]  /= bs;
      dnzu[r] /= bs;
      onzu[r] /= bs;
    }
  }
  /* Set matrix pattern */
  ierr = MatXAIJSetPreallocation(A, bs, dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* Fill matrix with zeros */
  if (fillMatrix) {
    PetscScalar *values;
    PetscInt     maxRowLen = 0;

    for (r = rStart; r < rEnd; ++r) {
      PetscInt len;

      ierr = PetscSectionGetDof(sectionAdj, r, &len);CHKERRQ(ierr);
      maxRowLen = PetscMax(maxRowLen, len);
    }
    ierr = PetscMalloc(maxRowLen * sizeof(PetscScalar), &values);CHKERRQ(ierr);
    ierr = PetscMemzero(values, maxRowLen * sizeof(PetscScalar));CHKERRQ(ierr);
    for (r = rStart; r < rEnd; ++r) {
      PetscInt numCols, cStart;

      ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
      ierr = MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&sectionAdj);CHKERRQ(ierr);
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
PetscErrorCode DMCreateMatrix_Complex(DM dm, MatType mtype, Mat *J)
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
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
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

        ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
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
  ierr = PetscSectionSetChart(mesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
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
  for (c = 0; c < dof; ++c) {
    if ((cone[c] < pStart) || (cone[c] >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", cone[c], pStart, pEnd);
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
  for (c = 0; c < dof; ++c) {
    PetscInt cdof, o = coneOrientation[c];

    ierr = PetscSectionGetDof(mesh->coneSection, mesh->cones[off+c], &cdof);CHKERRQ(ierr);
    if (o && ((o < -(cdof+1)) || (o >= cdof))) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone orientation %D is not in the valid range [%D. %D)", o, -(cdof+1), cdof);
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
  if (conePos >= dof) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
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
#define __FUNCT__ "DMComplexSetSupportSize"
/*@
  DMComplexSetSupportSize - Set the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- size - The support size for point p

  Output Parameter:

  Note:
  This should be called after DMComplexSetChart().

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetSupportSize(), DMComplexSetChart()
@*/
PetscErrorCode DMComplexSetSupportSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->supportSection, p, size);CHKERRQ(ierr);
  mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, size);
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
#define __FUNCT__ "DMComplexSetSupport"
/*@
  DMComplexSetSupport - Set the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMComplex
. p - The Sieve point, which must lie in the chart set with DMComplexSetChart()
- support - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMComplexSetSupportSize() and DMSetUp().

  Level: beginner

.seealso: DMComplexCreate(), DMComplexGetSupport(), DMComplexSetChart(), DMComplexSetSupportSize(), DMSetUp()
@*/
PetscErrorCode DMComplexSetSupport(DM dm, PetscInt p, const PetscInt support[])
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    if ((support[c] < pStart) || (support[c] >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", support[c], pStart, pEnd);
    mesh->supports[off+c] = support[c];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexInsertSupport"
PetscErrorCode DMComplexInsertSupport(DM dm, PetscInt p, PetscInt supportPos, PetscInt supportPoint)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((supportPoint < pStart) || (supportPoint >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", supportPoint, pStart, pEnd);
  if (supportPos >= dof) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Support position %D of point %D is not in the valid range [0, %D)", supportPos, p, dof);
  mesh->supports[off+supportPos] = supportPoint;
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

.seealso: DMComplexRestoreTransitiveClosure(), DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart(), DMComplexGetCone()
@*/
PetscErrorCode DMComplexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DM_Complex     *mesh = (DM_Complex *) dm->data;
  PetscInt       *closure, *fifo;
  const PetscInt *tmp, *tmpO = PETSC_NULL;
  PetscInt        tmpSize, t;
  PetscInt        depth, maxSize;
  PetscInt        closureSize = 2, fifoSize = 0, fifoStart = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  maxSize = (PetscInt) (2*PetscMax(pow((PetscReal) mesh->maxConeSize, depth)+1, pow((PetscReal) mesh->maxSupportSize, depth)+1));
  ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
  }
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
  for (t = 0; t < tmpSize; ++t, closureSize += 2, fifoSize += 2) {
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
    for (t = 0; t < tmpSize; ++t) {
      const PetscInt i  = ((rev ? tmpSize-t : t) + off)%tmpSize;
      const PetscInt cp = tmp[i];
      /* Must propogate orientation */
      const PetscInt co = tmpO ? (rev ? -(tmpO[i]+1) : tmpO[i]) : 0;
      PetscInt       c;

      /* Check for duplicate */
      for (c = 0; c < closureSize; c += 2) {
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
  ierr = DMRestoreWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexRestoreTransitiveClosure"
/*@C
  DMComplexRestoreTransitiveClosure - Restore the array of points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

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
  If not using internal storage (points is not PETSC_NULL on input), this call is unnecessary

  Level: beginner

.seealso: DMComplexGetTransitiveClosure(), DMComplexCreate(), DMComplexSetCone(), DMComplexSetChart(), DMComplexGetCone()
@*/
PetscErrorCode DMComplexRestoreTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, points);CHKERRQ(ierr);
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
  if (!mesh->facesTmp) {ierr = PetscMalloc(PetscSqr(PetscMax(mesh->maxConeSize, mesh->maxSupportSize)) * sizeof(PetscInt), &mesh->facesTmp);CHKERRQ(ierr);}
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
    case 4:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[3];
      mesh->facesTmp[6] = cone[3]; mesh->facesTmp[7] = cone[0];
      *numFaces = 4;
      *faceSize = 2;
      *faces    = mesh->facesTmp;
      break;
    default:
      SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch(coneSize) {
    case 3:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[0];
      *numFaces = 3;
      *faceSize = 2;
      *faces    = mesh->facesTmp;
      break;
    case 4:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1]  = cone[1]; mesh->facesTmp[2]  = cone[2];
      mesh->facesTmp[3] = cone[0]; mesh->facesTmp[4]  = cone[2]; mesh->facesTmp[5]  = cone[3];
      mesh->facesTmp[6] = cone[0]; mesh->facesTmp[7]  = cone[3]; mesh->facesTmp[8]  = cone[1];
      mesh->facesTmp[9] = cone[1]; mesh->facesTmp[10] = cone[3]; mesh->facesTmp[11] = cone[2];
      *numFaces = 4;
      *faceSize = 3;
      *faces    = mesh->facesTmp;
      break;
    default:
      SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
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
  if (mesh->maxSupportSize) {
    ierr = PetscSectionSetUp(mesh->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->supportSection, &size);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &mesh->supports);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateSubDM_Complex"
PetscErrorCode DMCreateSubDM_Complex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *subIndices;
  PetscInt       subSize = 0, subOff = 0, nF, f, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(0);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  if (!section) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Must set default section for DMComplex before splitting fields");
  if (!sectionGlobal) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Must set default global section for DMComplex before splitting fields");
  ierr = PetscSectionGetNumFields(section, &nF);CHKERRQ(ierr);
  if (numFields > nF) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Number of requested fields %d greater than number of DM fields %d", numFields, nF);
  ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      for (f = 0; f < numFields; ++f) {
        PetscInt fdof, fcdof;

        ierr = PetscSectionGetFieldDof(section, p, fields[f], &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof);CHKERRQ(ierr);
        subSize += fdof-fcdof;
      }
    }
  }
  ierr = PetscMalloc(subSize * sizeof(PetscInt), &subIndices);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, goff;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        PetscInt fdof, fcdof, fc, f2, poff = 0;

        /* Can get rid of this loop by storing field information in the global section */
        for (f2 = 0; f2 < fields[f]; ++f2) {
          ierr = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f2, &fcdof);CHKERRQ(ierr);
          poff += fdof-fcdof;
        }
        ierr = PetscSectionGetFieldDof(section, p, fields[f], &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof);CHKERRQ(ierr);
        for (fc = 0; fc < fdof-fcdof; ++fc, ++subOff) {
          subIndices[subOff] = goff+poff+fc;
        }
      }
    }
  }
  if (is) {ierr = ISCreateGeneral(((PetscObject) dm)->comm, subSize, subIndices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);}
  if (subdm) {
    PetscSection subsection;
    PetscBool    haveNull = PETSC_FALSE;
    PetscInt     f, nf = 0;

    ierr = DMComplexClone(dm, subdm);CHKERRQ(ierr);
    ierr = PetscSectionCreateSubsection(section, numFields, fields, &subsection);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(*subdm, subsection);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[fields[f]];
      if ((*subdm)->nullspaceConstructors[f]) {
        haveNull = PETSC_TRUE;
        nf       = f;
      }
    }
    if (haveNull) {
      MatNullSpace nullSpace;

      ierr = (*(*subdm)->nullspaceConstructors[nf])(*subdm, nf, &nullSpace);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    }
    if (dm->fields) {
      if (nF != dm->numFields) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "The number of DM fields %d does not match the number of Section fields %d", dm->numFields, nF);
      ierr = DMSetNumFields(*subdm, numFields);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        ierr = PetscOListDuplicate(dm->fields[fields[f]]->olist, &(*subdm)->fields[f]->olist);
      }
      if (numFields == 1) {
        MatNullSpace space;
        Mat          pmat;

        ierr = PetscObjectQuery((*subdm)->fields[0], "nullspace", (PetscObject *) &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery((*subdm)->fields[0], "nearnullspace", (PetscObject *) &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nearnullspace", (PetscObject) space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery((*subdm)->fields[0], "pmat", (PetscObject *) &pmat);CHKERRQ(ierr);
        if (pmat) {ierr = PetscObjectCompose((PetscObject) *is, "pmat", (PetscObject) pmat);CHKERRQ(ierr);}
      }
    }
  }
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
  if (mesh->supports) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Supports were already setup in this DMComplex");
  /* Calculate support sizes */
  ierr = DMComplexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for (c = off; c < off+dof; ++c) {
      ierr = PetscSectionAddDof(mesh->supportSection, mesh->cones[c], 1);CHKERRQ(ierr);
    }
  }
  for (p = pStart; p < pEnd; ++p) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for (c = off; c < off+dof; ++c) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt coneSize, supportSize;

    ierr = DMComplexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    if (!coneSize && supportSize) {
      ++numRoots;
      ierr = DMComplexSetLabelValue(dm, "depth", p, 0);CHKERRQ(ierr);
    } else if (!supportSize && coneSize) {
      ++numLeaves;
    } else if (!supportSize && !coneSize) {
      /* Isolated points */
      ierr = DMComplexSetLabelValue(dm, "depth", p, 0);CHKERRQ(ierr);
    }
  }
  if (numRoots + numLeaves == (pEnd - pStart)) {
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize, supportSize;

      ierr = DMComplexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        ierr = DMComplexSetLabelValue(dm, "depth", p, 1);CHKERRQ(ierr);
      }
    }
  } else {
    /* This might be slow since lookup is not fast */
    for (p = pStart; p < pEnd; ++p) {
      PetscInt depth;

      ierr = DMComplexSetDepth_Private(dm, p, &depth);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetNumLabels"
/*@
  DMComplexGetNumLabels - Return the number of labels defined by the mesh

  Not Collective

  Input Parameter:
. dm   - The DMComplex object

  Output Parameter:
. numLabels - the number of Labels

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetLabelValue(), DMComplexSetLabelValue(), DMComplexGetStratumIS()
@*/
PetscErrorCode DMComplexGetNumLabels(DM dm, PetscInt *numLabels)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;
  DMLabel     next = mesh->labels;
  PetscInt    n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numLabels, 2);
  while(next) {
    ++n;
    next = next->next;
  }
  *numLabels = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLabelName"
/*@C
  DMComplexGetLabelName - Return the name of nth label

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
- n  - the label number

  Output Parameter:
. name - the label name

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetLabelValue(), DMComplexSetLabelValue(), DMComplexGetStratumIS()
@*/
PetscErrorCode DMComplexGetLabelName(DM dm, PetscInt n, const char **name)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;
  DMLabel     next = mesh->labels;
  PetscInt    l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 3);
  while(next) {
    if (l == n) {
      *name = next->name;
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Label %d does not exist in this DM", n);
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
.seealso: DMComplexGetLabelValue(), DMComplexSetLabelValue(), DMComplexGetStratumIS()
@*/
PetscErrorCode DMComplexHasLabel(DM dm, const char name[], PetscBool *hasLabel)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
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
.seealso: DMComplexSetLabelValue(), DMComplexGetStratumIS()
@*/
PetscErrorCode DMComplexGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
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
  for (v = 0; v < next->numStrata; ++v) {
    for (p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
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
.seealso: DMComplexGetStratumIS(), DMComplexClearLabelValue()
@*/
PetscErrorCode DMComplexSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
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
    DMLabel tmpLabel = mesh->labels;
    ierr = PetscNew(struct _n_DMLabel, &mesh->labels);CHKERRQ(ierr);
    mesh->labels->next = tmpLabel;
    next = mesh->labels;
    ierr = PetscStrallocpy(name, &next->name);CHKERRQ(ierr);
  }
  /* Find, or add, label value */
  for (v = 0; v < next->numStrata; ++v) {
    if (next->stratumValues[v] == value) break;
  }
  if (v >= next->numStrata) {
    PetscInt *tmpV, *tmpO, *tmpS;
    ierr = PetscMalloc3(next->numStrata+1,PetscInt,&tmpV,next->numStrata+2,PetscInt,&tmpO,next->numStrata+1,PetscInt,&tmpS);CHKERRQ(ierr);
    for (v = 0; v < next->numStrata; ++v) {
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
  for (p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
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
      for (q = 0; q < next->stratumOffsets[v]+next->stratumSizes[v]; ++q) {
        newPoints[q] = next->points[q];
      }
      for (w = v+1; w < next->numStrata; ++w) {
        for (q = next->stratumOffsets[w]; q < next->stratumOffsets[w]+next->stratumSizes[w]; ++q) {
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
#define __FUNCT__ "DMComplexClearLabelValue"
/*@C
  DMComplexClearLabelValue - Remove a point from a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMComplexSetLabelValue(), DMComplexGetStratumIS()
@*/
PetscErrorCode DMComplexClearLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  /* Find label */
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) PetscFunctionReturn(0);
  /* Find label value */
  for (v = 0; v < next->numStrata; ++v) {
    if (next->stratumValues[v] == value) break;
  }
  if (v >= next->numStrata) PetscFunctionReturn(0);
  /* Check whether point exists */
  for (p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
    if (next->points[p] == point) {
      /* Found point */
      PetscInt  q;

      for (q = p+1; q < next->stratumOffsets[v]+next->stratumSizes[v]; ++q) {
        next->points[q-1] = next->points[q];
      }
      --next->stratumSizes[v];
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexClearLabelStratum"
/*@C
  DMComplexClearLabelStratum - Remove all points from a stratum from a Sieve Label

  Not Collective

  Input Parameters:
+ dm   - The DMComplex object
. name - The label name
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMComplexSetLabelValue(), DMComplexGetStratumIS(), DMComplexClearLabelValue()
@*/
PetscErrorCode DMComplexClearLabelStratum(DM dm, const char name[], PetscInt value)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
  PetscBool      flg  = PETSC_FALSE;
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  /* Find label */
  while(next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) PetscFunctionReturn(0);
  /* Find label value */
  for (v = 0; v < next->numStrata; ++v) {
    if (next->stratumValues[v] == value) break;
  }
  if (v >= next->numStrata) PetscFunctionReturn(0);
  next->stratumSizes[v] = 0;
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
  DMLabel        next = mesh->labels;
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
  DMLabel        next = mesh->labels;
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
      for (i = 0; i < next->numStrata; ++i) {
        values[i] = next->stratumValues[i];
      }
      break;
    }
    next = next->next;
  }
  if (!next) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label with name %s exists in this mesh", name);
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
  DMLabel        next = mesh->labels;
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

      for (v = 0; v < next->numStrata; ++v) {
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
  DMLabel        next = mesh->labels;
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

      for (v = 0; v < next->numStrata; ++v) {
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
#define __FUNCT__ "DMComplexGetJoin"
/*@C
  DMComplexGetJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Level: intermediate

  Note: Currently, this is restricted to a single level join

.keywords: mesh
.seealso: DMComplexRestoreJoin(), DMComplexGetMeet()
@*/
PetscErrorCode DMComplexGetJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
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
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1]);CHKERRQ(ierr);
  /* Copy in support of first point */
  ierr = PetscSectionGetDof(mesh->supportSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, points[0], &off);CHKERRQ(ierr);
  for (joinSize = 0; joinSize < dof; ++joinSize) {
    join[i][joinSize] = mesh->supports[off+joinSize];
  }
  /* Check each successive support */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newJoinSize = 0;

    ierr = PetscSectionGetDof(mesh->supportSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->supportSection, points[p], &off);CHKERRQ(ierr);
    for (c = 0; c < dof; ++c) {
      const PetscInt point = mesh->supports[off+c];

      for (m = 0; m < joinSize; ++m) {
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
  ierr = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexRestoreJoin"
/*@C
  DMComplexRestoreJoin - Restore an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetJoin(), DMComplexGetFullJoin(), DMComplexGetMeet()
@*/
PetscErrorCode DMComplexRestoreJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coveredPoints, 4);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, coveredPoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetFullJoin"
/*@C
  DMComplexGetFullJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetJoin(), DMComplexRestoreJoin(), DMComplexGetMeet()
@*/
PetscErrorCode DMComplexGetFullJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *join[2];
  PetscInt       depth, maxSize, joinSize = 0, i = 0;
  PetscInt       p, d, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr = DMComplexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints * sizeof(PetscInt *), &closures);CHKERRQ(ierr);
  ierr = PetscMemzero(closures,numPoints*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  maxSize = (PetscInt) (pow((PetscReal) mesh->maxSupportSize, depth)+1);
  ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMComplexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closures[p]);CHKERRQ(ierr);
    offsets[p*(depth+2)+0] = 0;
    for (d = 0; d < depth+1; ++d) {
      PetscInt pStart, pEnd, i;

      ierr = DMComplexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(depth+2)+d]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(depth+2)+d+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(depth+2)+d+1] = i;
    }
    if (offsets[p*(depth+2)+depth+1] != closureSize) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(depth+2)+depth+1], closureSize);
  }
  for (d = 0; d < depth+1; ++d) {
    PetscInt dof;

    /* Copy in support of first point */
    dof = offsets[d+1] - offsets[d];
    for (joinSize = 0; joinSize < dof; ++joinSize) {
      join[i][joinSize] = closures[0][(offsets[d]+joinSize)*2];
    }
    /* Check each successive cone */
    for (p = 1; p < numPoints && joinSize; ++p) {
      PetscInt newJoinSize = 0;

      dof = offsets[p*(depth+2)+d+1] - offsets[p*(depth+2)+d];
      for (c = 0; c < dof; ++c) {
        const PetscInt point = closures[p][(offsets[p*(depth+2)+d]+c)*2];

        for (m = 0; m < joinSize; ++m) {
          if (point == join[i][m]) {
            join[1-i][newJoinSize++] = point;
            break;
          }
        }
      }
      joinSize = newJoinSize;
      i = 1-i;
    }
    if (joinSize) break;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
  for (p = 0; p < numPoints; ++p) {
    ierr = DMComplexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, PETSC_NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetMeet"
/*@C
  DMComplexGetMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Note: Currently, this is restricted to a single level meet

.keywords: mesh
.seealso: DMComplexRestoreMeet(), DMComplexGetJoin()
@*/
PetscErrorCode DMComplexGetMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveringPoints, const PetscInt **coveringPoints)
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
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1]);CHKERRQ(ierr);
  /* Copy in cone of first point */
  ierr = PetscSectionGetDof(mesh->coneSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, points[0], &off);CHKERRQ(ierr);
  for (meetSize = 0; meetSize < dof; ++meetSize) {
    meet[i][meetSize] = mesh->cones[off+meetSize];
  }
  /* Check each successive cone */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newMeetSize = 0;

    ierr = PetscSectionGetDof(mesh->coneSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, points[p], &off);CHKERRQ(ierr);
    for (c = 0; c < dof; ++c) {
      const PetscInt point = mesh->cones[off+c];

      for (m = 0; m < meetSize; ++m) {
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
  ierr = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexRestoreMeet"
/*@C
  DMComplexRestoreMeet - Restore an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetMeet(), DMComplexGetFullMeet(), DMComplexGetJoin()
@*/
PetscErrorCode DMComplexRestoreMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coveredPoints, 4);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, coveredPoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetFullMeet"
/*@C
  DMComplexGetFullMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMComplex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

.keywords: mesh
.seealso: DMComplexGetMeet(), DMComplexRestoreMeet(), DMComplexGetJoin()
@*/
PetscErrorCode DMComplexGetFullMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *meet[2];
  PetscInt       height, maxSize, meetSize = 0, i = 0;
  PetscInt       p, h, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr = DMComplexGetDepth(dm, &height);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints * sizeof(PetscInt *), &closures);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  maxSize = (PetscInt) (pow((PetscReal) mesh->maxConeSize, height)+1);
  ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMComplexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closures[p]);CHKERRQ(ierr);
    offsets[p*(height+2)+0] = 0;
    for (h = 0; h < height+1; ++h) {
      PetscInt pStart, pEnd, i;

      ierr = DMComplexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(height+2)+h]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(height+2)+h+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(height+2)+h+1] = i;
    }
    if (offsets[p*(height+2)+height+1] != closureSize) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(height+2)+height+1], closureSize);
  }
  for (h = 0; h < height+1; ++h) {
    PetscInt dof;

    /* Copy in cone of first point */
    dof = offsets[h+1] - offsets[h];
    for (meetSize = 0; meetSize < dof; ++meetSize) {
      meet[i][meetSize] = closures[0][(offsets[h]+meetSize)*2];
    }
    /* Check each successive cone */
    for (p = 1; p < numPoints && meetSize; ++p) {
      PetscInt newMeetSize = 0;

      dof = offsets[p*(height+2)+h+1] - offsets[p*(height+2)+h];
      for (c = 0; c < dof; ++c) {
        const PetscInt point = closures[p][(offsets[p*(height+2)+h]+c)*2];

        for (m = 0; m < meetSize; ++m) {
          if (point == meet[i][m]) {
            meet[1-i][newMeetSize++] = point;
            break;
          }
        }
      }
      meetSize = newMeetSize;
      i = 1-i;
    }
    if (meetSize) break;
  }
  *numCoveredPoints = meetSize;
  *coveredPoints    = meet[i];
  for (p = 0; p < numPoints; ++p) {
    ierr = DMComplexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, PETSC_NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetNumFaceVertices"
static PetscErrorCode DMComplexGetNumFaceVertices(DM dm, PetscInt numCorners, PetscInt *numFaceVertices) {
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscInt       cellDim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(numFaceVertices,3);
  ierr = DMComplexGetDimension(dm, &cellDim);CHKERRQ(ierr);
  switch(cellDim) {
  case 0:
    *numFaceVertices = 0;
    break;
  case 1:
    *numFaceVertices = 1;
    break;
  case 2:
    switch(numCorners) {
    case 3: // triangle
      *numFaceVertices = 2; // Edge has 2 vertices
      break;
    case 4: // quadrilateral
      *numFaceVertices = 2; // Edge has 2 vertices
      break;
    case 6: // quadratic triangle, tri and quad cohesive Lagrange cells
      *numFaceVertices = 3; // Edge has 3 vertices
      break;
    case 9: // quadratic quadrilateral, quadratic quad cohesive Lagrange cells
      *numFaceVertices = 3; // Edge has 3 vertices
      break;
    default:
      SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %d for dimension %d", numCorners, cellDim);
    }
    break;
  case 3:
    switch(numCorners)	{
    case 4: // tetradehdron
      *numFaceVertices = 3; // Face has 3 vertices
      break;
    case 6: // tet cohesive cells
      *numFaceVertices = 4; // Face has 4 vertices
      break;
    case 8: // hexahedron
      *numFaceVertices = 4; // Face has 4 vertices
      break;
    case 9: // tet cohesive Lagrange cells
      *numFaceVertices = 6; // Face has 6 vertices
      break;
    case 10: // quadratic tetrahedron
      *numFaceVertices = 6; // Face has 6 vertices
      break;
    case 12: // hex cohesive Lagrange cells
      *numFaceVertices = 6; // Face has 6 vertices
      break;
    case 18: // quadratic tet cohesive Lagrange cells
      *numFaceVertices = 6; // Face has 6 vertices
      break;
    case 27: // quadratic hexahedron, quadratic hex cohesive Lagrange cells
      *numFaceVertices = 9; // Face has 9 vertices
      break;
    default:
      SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %d for dimension %d", numCorners, cellDim);
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell dimension %d", cellDim);
  }
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

    for (c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      ierr = DMComplexGetConeSize(dm, c, &corners);CHKERRQ(ierr);
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        if (numFaceCases >= maxFaceCases) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;
        ierr = DMComplexGetNumFaceVertices(dm, corners, &nFV);CHKERRQ(ierr);
        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  maxClosure   = (PetscInt) (2*PetscMax(pow((PetscReal) maxConeSize, depth)+1, pow((PetscReal) maxSupportSize, depth)+1));
  maxNeighbors = (PetscInt) (pow((PetscReal) maxConeSize, depth)*pow((PetscReal) maxSupportSize, depth)+1);
  ierr = PetscMalloc2(maxNeighbors,PetscInt,&neighborCells,maxClosure,PetscInt,&tmpClosure);CHKERRQ(ierr);
  ierr = PetscMalloc((numCells+1) * sizeof(PetscInt), &off);CHKERRQ(ierr);
  ierr = PetscMemzero(off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = maxNeighbors, n;

    ierr = DMComplexGetAdjacencySingleLevel_Private(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2] = {cell, neighborCells[n]};
      PetscBool       found       = depth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize    = 0;
      const PetscInt *meet        = PETSC_NULL;

      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        ierr = DMComplexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        ierr = DMComplexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
      }
      if (found) {
        ++off[cell-cStart+1];
      }
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) {
    off[cell] += off[cell-1];
  }
  if (adjacency) {
    ierr = PetscMalloc(off[numCells] * sizeof(PetscInt), &adj);CHKERRQ(ierr);
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = maxNeighbors, n;
      PetscInt cellOffset   = 0;

      ierr = DMComplexGetAdjacencySingleLevel_Private(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2] = {cell, neighborCells[n]};
        PetscBool       found       = depth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize    = 0;
        const PetscInt *meet        = PETSC_NULL;

        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          ierr = DMComplexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          ierr = DMComplexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
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
  for (i = 0; i < start[numVertices]; ++i) {
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
  for (v = 0; v < nvtxs; ++v) {
    ierr = PetscSectionAddDof(*partSection, assignment[v], 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc(nvtxs * sizeof(PetscInt), &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < commSize; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  if (global_method == INERTIAL_METHOD) {
    /* manager.destroyCellCoordinates(nvtxs, &x, &y, &z); */
  }
  ierr = PetscFree(assignment);CHKERRQ(ierr);
  for (i = 0; i < start[numVertices]; ++i) {
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
    for (c = cStart; c < cEnd; ++c) {
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
  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0;
    PetscInt numPoints, offset, p;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
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
      for (c = 0; c < closureSize; ++c) {
        partPoints[partSize+c] = closure[c*2];
      }
      partSize += closureSize;
      ierr = PetscSortRemoveDupsInt(&partSize, partPoints);CHKERRQ(ierr);
      ierr = DMComplexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(*section, rank, partSize);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(*section, &newSize);CHKERRQ(ierr);
  ierr = PetscMalloc(newSize * sizeof(PetscInt), &allPoints);CHKERRQ(ierr);

  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0, newOffset;
    PetscInt numPoints, offset, p;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt  point   = partArray[offset+p], closureSize, c;
      PetscInt *closure = PETSC_NULL;

      /* TODO Include support for height > 0 case */
      ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Merge into existing points */
      for (c = 0; c < closureSize; ++c) {
        partPoints[partSize+c] = closure[c*2];
      }
      partSize += closureSize;
      ierr = PetscSortRemoveDupsInt(&partSize, partPoints);CHKERRQ(ierr);
      ierr = DMComplexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
  ierr = PetscLogEventBegin(DMCOMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
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
  for (p = 0; p < numRemoteRanks; ++p) {
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
    for (p = pStart; p < pEnd; ++p) {
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
  {
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMComplexSymmetrize(*dmParallel);CHKERRQ(ierr);
  ierr = DMComplexStratify(*dmParallel);CHKERRQ(ierr);
  /* Distribute Coordinates */
  {
    PetscSection originalCoordSection, newCoordSection;
    Vec          originalCoordinates, newCoordinates;

    ierr = DMComplexGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
    ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);

    ierr = DMComplexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dmParallel, newCoordinates);CHKERRQ(ierr);
  }
  /* Distribute labels */
  {
    DMLabel    next      = mesh->labels, newNext = PETSC_NULL;
    PetscInt   numLabels = 0, l;

    /* Bcast number of labels */
    while(next) {++numLabels; next = next->next;}
    ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    next = mesh->labels;
    for (l = 0; l < numLabels; ++l) {
      DMLabel         newLabel;
      const PetscInt *partArray;
      PetscInt       *stratumSizes = PETSC_NULL, *points = PETSC_NULL;
      PetscMPIInt    *sendcnts = PETSC_NULL, *offsets = PETSC_NULL, *displs = PETSC_NULL;
      PetscInt        nameSize, s, p;
      size_t          len = 0;

      ierr = PetscNew(struct _n_DMLabel, &newLabel);CHKERRQ(ierr);
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
        for (s = 0; s < next->numStrata; ++s) {
          for (p = next->stratumOffsets[s]; p < next->stratumOffsets[s]+next->stratumSizes[s]; ++p) {
            const PetscInt point = next->points[p];
            PetscInt       proc;

            for (proc = 0; proc < numProcs; ++proc) {
              PetscInt dof, off, pPart;

              ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
              for (pPart = off; pPart < off+dof; ++pPart) {
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
      for (s = 0; s < newLabel->numStrata; ++s) {
        newLabel->stratumOffsets[s+1] = newLabel->stratumSizes[s] + newLabel->stratumOffsets[s];
      }
      /* Pack points and Scatter */
      if (!rank) {
        ierr = PetscMalloc3(numProcs,PetscMPIInt,&sendcnts,numProcs,PetscMPIInt,&offsets,numProcs+1,PetscMPIInt,&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (p = 0; p < numProcs; ++p) {
          sendcnts[p] = 0;
          for (s = 0; s < next->numStrata; ++s) {
            sendcnts[p] += stratumSizes[p*next->numStrata+s];
          }
          offsets[p]  = displs[p];
          displs[p+1] = displs[p] + sendcnts[p];
        }
        ierr = PetscMalloc(displs[numProcs] * sizeof(PetscInt), &points);CHKERRQ(ierr);
        for (s = 0; s < next->numStrata; ++s) {
          for (p = next->stratumOffsets[s]; p < next->stratumOffsets[s]+next->stratumSizes[s]; ++p) {
            const PetscInt point = next->points[p];
            PetscInt       proc;

            for (proc = 0; proc < numProcs; ++proc) {
              PetscInt dof, off, pPart;

              ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
              for (pPart = off; pPart < off+dof; ++pPart) {
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
      for (s = 0; s < newLabel->numStrata; ++s) {
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
    for (p = 0; p < numRoots*2; ++p) {
      rowners[p] = 0;
    }
    for (p = 0; p < numLeaves; ++p) {
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
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p*2+0] != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscInt),    &ghostPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscSFNode), &remotePoints);CHKERRQ(ierr);
    for (p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p*2+0] != rank) {
        ghostPoints[gp]       = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p*2+0];
        remotePoints[gp].index = lowners[p*2+1];
        ++gp;
      }
    }
    ierr = PetscFree2(rowners,lowners);CHKERRQ(ierr);
    ierr = PetscSFSetGraph((*dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions((*dmParallel)->sf);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dmParallel);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMCOMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexInterpolate_2D"
PetscErrorCode DMComplexInterpolate_2D(DM dm, DM *dmInt)
{
  DM             idm;
  DM_Complex    *mesh;
  PetscInt      *off;
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
    PetscInt numCorners = 0;

    numEdges = off[numCells]/2;
#if 0
    /* Account for boundary edges: \sum_c 3 - neighbors = 3*numCells - totalNeighbors */
    numEdges += 3*numCells - off[numCells];
#else
    /* Account for boundary edges: \sum_c #faces - #neighbors = \sum_c #cellVertices - #neighbors = totalCorners - totalNeighbors */
    for(c = cStart; c < cEnd; ++c) {
      PetscInt coneSize;

      ierr = DMComplexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      numCorners += coneSize;
    }
    numEdges += numCorners - off[numCells];
#endif
  }
  /* Check Euler characteristic V - E + F = 1 */
  if (numVertices && (numVertices-numEdges+numCells != 1)) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Euler characteristic of mesh is %d  != 1", numVertices-numEdges+numCells);
  /* Create interpolated mesh */
  ierr = DMCreate(((PetscObject) dm)->comm, &idm);CHKERRQ(ierr);
  ierr = DMSetType(idm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(idm, dim);CHKERRQ(ierr);
  ierr = DMComplexSetChart(idm, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    PetscInt numCorners;

    ierr = DMComplexGetConeSize(dm, c, &numCorners);CHKERRQ(ierr);
    ierr = DMComplexSetConeSize(idm, c, numCorners);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMComplexSetConeSize(idm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get edge cones from subsets of cell vertices */
  for (c = 0, edge = firstEdge; c < numCells; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMComplexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 2) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (e = firstEdge; e < edge; ++e) {
        const PetscInt *cone;

        ierr = DMComplexGetCone(idm, e, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMComplexSetCone(idm, edge, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++edge;
      }
      ierr = DMComplexInsertCone(idm, c, cf, e);CHKERRQ(ierr);
    }
  }
  if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
  ierr = PetscFree(off);CHKERRQ(ierr);
  ierr = DMComplexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMComplexStratify(idm);CHKERRQ(ierr);
  mesh = (DM_Complex *) (idm)->data;
  /* Orient edges */
  for (c = 0; c < numCells; ++c) {
    const PetscInt *cone, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMComplexGetConeSize(idm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(idm, c, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
    ierr = DMComplexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt *econe;
      PetscInt        esize;

      ierr = DMComplexGetConeSize(idm, cone[cf], &esize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(idm, cone[cf], &econe);CHKERRQ(ierr);
      if (esize != 2) SETERRQ2(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[cf]);
      if ((cellFaces[cf*faceSize+0] == econe[0]) && (cellFaces[cf*faceSize+1] == econe[1])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == econe[1]) && (cellFaces[cf*faceSize+1] == econe[0])) {
        /* Start at index 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      }
    }
  }
  *dmInt  = idm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexInterpolate_3D"
PetscErrorCode DMComplexInterpolate_3D(DM dm, DM *dmInt)
{
  DM             idm, fdm;
  DM_Complex    *mesh;
  PetscInt      *off;
  const PetscInt numCorners = 4;
  PetscInt       dim, numCells, cStart, cEnd, c, numVertices, vStart, vEnd;
  PetscInt       numFaces, firstFace, face, f, numEdges, firstEdge, edge, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  firstFace   = numCells + numVertices;
  numFaces    = 0 ;
  /* Count faces using algorithm from CreateNeighborCSR */
  ierr = DMComplexCreateNeighborCSR(dm, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
  if (off) {
    numFaces = off[numCells]/2;
    /* Account for boundary faces: \sum_c 4 - neighbors = 4*numCells - totalNeighbors */
    numFaces += 4*numCells - off[numCells];
  }
  /* Use Euler characteristic to get edges V - E + F - C = 1 */
  firstEdge = firstFace + numFaces;
  numEdges  = numVertices + numFaces - numCells - 1;
  /* Create interpolated mesh */
  ierr = DMCreate(((PetscObject) dm)->comm, &idm);CHKERRQ(ierr);
  ierr = DMSetType(idm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(idm, dim);CHKERRQ(ierr);
  ierr = DMComplexSetChart(idm, 0, numCells+numVertices+numFaces+numEdges);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    ierr = DMComplexSetConeSize(idm, c, numCorners);CHKERRQ(ierr);
  }
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    ierr = DMComplexSetConeSize(idm, f, 3);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMComplexSetConeSize(idm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get face cones from subsets of cell vertices */
  ierr = DMCreate(((PetscObject) dm)->comm, &fdm);CHKERRQ(ierr);
  ierr = DMSetType(fdm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(fdm, dim);CHKERRQ(ierr);
  ierr = DMComplexSetChart(fdm, numCells, firstFace+numFaces);CHKERRQ(ierr);
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    ierr = DMComplexSetConeSize(fdm, f, 3);CHKERRQ(ierr);
  }
  ierr = DMSetUp(fdm);CHKERRQ(ierr);
  for (c = 0, face = firstFace; c < numCells; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMComplexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 3) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Tetrahedra cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (f = firstFace; f < face; ++f) {
        const PetscInt *cone;

        ierr = DMComplexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1]) && (cellFaces[cf*faceSize+2] == cone[2])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[2]) && (cellFaces[cf*faceSize+2] == cone[0])) ||
            ((cellFaces[cf*faceSize+0] == cone[2]) && (cellFaces[cf*faceSize+1] == cone[0]) && (cellFaces[cf*faceSize+2] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[2]) && (cellFaces[cf*faceSize+2] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[2]) && (cellFaces[cf*faceSize+1] == cone[1]) && (cellFaces[cf*faceSize+2] == cone[0])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]) && (cellFaces[cf*faceSize+2] == cone[2]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMComplexSetCone(idm, face, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        /* Save the vertices for orientation calculation */
        ierr = DMComplexSetCone(fdm, face, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++face;
      }
      ierr = DMComplexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
    }
  }
  if (face != firstFace+numFaces) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid number of faces %D should be %D", face-firstFace, numFaces);
  /* Get edge cones from subsets of face vertices */
  for (f = firstFace, edge = firstEdge; f < firstFace+numFaces; ++f) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMComplexGetFaces(idm, f, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 2) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (e = firstEdge; e < edge; ++e) {
        const PetscInt *cone;

        ierr = DMComplexGetCone(idm, e, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMComplexSetCone(idm, edge, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++edge;
      }
      ierr = DMComplexInsertCone(idm, f, cf, e);CHKERRQ(ierr);
    }
  }
  if (edge != firstEdge+numEdges) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
  ierr = PetscFree(off);CHKERRQ(ierr);
  ierr = DMComplexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMComplexStratify(idm);CHKERRQ(ierr);
  mesh = (DM_Complex *) (idm)->data;
  /* Orient edges */
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    const PetscInt *cone, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMComplexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(idm, f, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, f, &coff);CHKERRQ(ierr);
    ierr = DMComplexGetFaces(fdm, f, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for face %D should be %D", coneSize, f, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt *econe;
      PetscInt        esize;

      ierr = DMComplexGetConeSize(idm, cone[cf], &esize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(idm, cone[cf], &econe);CHKERRQ(ierr);
      if (esize != 2) SETERRQ2(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[cf]);
      if ((cellFaces[cf*faceSize+0] == econe[0]) && (cellFaces[cf*faceSize+1] == econe[1])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == econe[1]) && (cellFaces[cf*faceSize+1] == econe[0])) {
        /* Start at index 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      }
    }
  }
  ierr = DMDestroy(&fdm);CHKERRQ(ierr);
  /* Orient faces */
  for (c = 0; c < numCells; ++c) {
    const PetscInt *cone, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMComplexGetConeSize(idm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMComplexGetCone(idm, c, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
    ierr = DMComplexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscInt *origClosure = PETSC_NULL, *closure;
      PetscInt  closureSize, i;

      ierr = DMComplexGetTransitiveClosure(idm, cone[cf], PETSC_TRUE, &closureSize, &origClosure);CHKERRQ(ierr);
      if (closureSize != 7) SETERRQ2(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid closure size %D for face %D should be 7", closureSize, cone[cf]);
      for (i = 4; i < 7; ++i) {
        if ((origClosure[i*2] < vStart) || (origClosure[i*2] >= vEnd)) SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Invalid closure point %D should be a vertex in [%D, %D)", origClosure[i*2], vStart, vEnd);
      }
      closure = &origClosure[4*2];
      /* Remember that this is the orientation for edges, not vertices */
      if        ((cellFaces[cf*faceSize+0] == closure[0*2]) && (cellFaces[cf*faceSize+1] == closure[1*2]) && (cellFaces[cf*faceSize+2] == closure[2*2])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == closure[1*2]) && (cellFaces[cf*faceSize+1] == closure[2*2]) && (cellFaces[cf*faceSize+2] == closure[0*2])) {
        /* Shifted by 1 */
        mesh->coneOrientations[coff+cf] = 1;
      } else if ((cellFaces[cf*faceSize+0] == closure[2*2]) && (cellFaces[cf*faceSize+1] == closure[0*2]) && (cellFaces[cf*faceSize+2] == closure[1*2])) {
        /* Shifted by 2 */
        mesh->coneOrientations[coff+cf] = 2;
      } else if ((cellFaces[cf*faceSize+0] == closure[2*2]) && (cellFaces[cf*faceSize+1] == closure[1*2]) && (cellFaces[cf*faceSize+2] == closure[0*2])) {
        /* Start at edge 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      } else if ((cellFaces[cf*faceSize+0] == closure[1*2]) && (cellFaces[cf*faceSize+1] == closure[0*2]) && (cellFaces[cf*faceSize+2] == closure[2*2])) {
        /* Start at index 0, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(0+1);
      } else if ((cellFaces[cf*faceSize+0] == closure[0*2]) && (cellFaces[cf*faceSize+1] == closure[2*2]) && (cellFaces[cf*faceSize+2] == closure[1*2])) {
        /* Start at index 2, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(2+1);
      } else SETERRQ3(((PetscObject) idm)->comm, PETSC_ERR_PLIB, "Face %D did not match local face %D in cell %D for any orientation", cone[cf], cf, c);
      ierr = DMComplexRestoreTransitiveClosure(idm, cone[cf], PETSC_TRUE, &closureSize, &origClosure);CHKERRQ(ierr);
    }
  }
  {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(idm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  *dmInt  = idm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexBuildFromCellList_Private"
/*
  This takes as input the common mesh generator output, a list of the vertices for each cell
*/
PetscErrorCode DMComplexBuildFromCellList_Private(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, const int cells[])
{
  PetscInt      *cone, c, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    ierr = DMComplexSetConeSize(dm, c, numCorners);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      cone[p] = cells[c*numCorners+p]+numCells;
    }
    ierr = DMComplexSetCone(dm, c, cone);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexBuildCoordinates_Private"
/*
  This takes as input the coordinates for each vertex
*/
PetscErrorCode DMComplexBuildCoordinates_Private(DM dm, PetscInt spaceDim, PetscInt numCells, PetscInt numVertices, const double vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       coordSize, v, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < numVertices; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coords[v*spaceDim+d] = vertexCoords[v*spaceDim+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateFromCellList"
/*
  This takes as input the common mesh generator output, a list of the vertices for each cell
*/
PetscErrorCode DMComplexCreateFromCellList(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, PetscBool interpolate, const int cells[], const double vertexCoords[], DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMComplexBuildFromCellList_Private(*dm, numCells, numVertices, numCorners, cells);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    switch(dim) {
    case 2:
      ierr = DMComplexInterpolate_2D(*dm, &idm);CHKERRQ(ierr);break;
    case 3:
      ierr = DMComplexInterpolate_3D(*dm, &idm);CHKERRQ(ierr);break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No mesh interpolation support for dimension %D", dim);
    }
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  ierr = DMComplexBuildCoordinates_Private(*dm, dim, numCells, numVertices, vertexCoords);CHKERRQ(ierr);
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
#define __FUNCT__ "DMComplexGenerate_Triangle"
PetscErrorCode DMComplexGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm             comm = ((PetscObject) boundary)->comm;
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &eStart, &eEnd);CHKERRQ(ierr);
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    ierr = PetscMalloc(in.numberofsegments*2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofsegments   * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
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
    for (h = 0; h < in.numberofholes; ++h) {
      for (d = 0; d < dim; ++d) {
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

  {
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells       = out.trianglelist;
    const double  *meshCoords  = out.pointlist;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;
  in.trianglearealist  = (double *) maxVolumes;
  if (in.numberoftriangles > 0) {
    ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 4) && (closureSize != 7)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a triangle, %D vertices in closure", closureSize);
      for (v = 0; v < 3; ++v) {
        in.trianglelist[idx*in.numberofcorners + v] = closure[(v+closureSize-3)*2] - vStart;
      }
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  /* TODO: Segment markers are missing on input */
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMComplexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
  if (in.numberofholes > 0) {
    ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRQ(ierr);
    for (h = 0; h < in.numberofholes; ++h) {
      for (d = 0; d < dim; ++d) {
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

  {
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells       = out.trianglelist;
    const double  *meshCoords  = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in.numberoffacets = fEnd - fStart;
  if (in.numberoffacets > 0) {
    in.facetlist       = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = PETSC_NULL, numPoints, p, numVertices = 0, v;

      in.facetlist[idx].numberofpolygons = 1;
      in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
      in.facetlist[idx].numberofholes    = 0;
      in.facetlist[idx].holelist         = NULL;

      ierr = DMComplexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) {
          points[numVertices++] = point;
        }
      }

      tetgenio::polygon *poly = in.facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      poly->vertexlist       = new int[poly->numberofvertices];
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", f, &in.facetmarkerlist[idx]);CHKERRQ(ierr);
      ierr = DMComplexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    ::tetrahedralize(args, &in, &out);
  }
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells       = out.tetrahedronlist;
    const double  *meshCoords  = out.pointlist;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexGetJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dm, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in.numberofcorners       = 4;
  in.numberoftetrahedra    = cEnd - cStart;
  in.tetrahedronvolumelist = (double *) maxVolumes;
  if (in.numberoftetrahedra > 0) {
    in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in.tetrahedronlist[idx*in.numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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

  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells       = out.tetrahedronlist;
    const double  *meshCoords  = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexGetJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}
#endif

#ifdef PETSC_HAVE_CTETGEN
#include "ctetgen.h"

#undef __FUNCT__
#define __FUNCT__ "DMComplexGenerate_CTetgen"
PetscErrorCode DMComplexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm = ((PetscObject) boundary)->comm;
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in->numberofpoints*dim * sizeof(PetscReal), &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberofpoints     * sizeof(int),       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc(in->numberoffacets * sizeof(facet), &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberoffacets * sizeof(int),   &in->facetmarkerlist);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = PETSC_NULL, numPoints, p, numVertices = 0, v, m;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;
      ierr = PetscMalloc(in->facetlist[idx].numberofpolygons * sizeof(polygon), &in->facetlist[idx].polygonlist);CHKERRQ(ierr);
      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = PETSC_NULL;

      ierr = DMComplexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) {
          points[numVertices++] = point;
        }
      }

      poly = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr = PetscMalloc(poly->numberofvertices * sizeof(int), &poly->vertexlist);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", f, &m);CHKERRQ(ierr);
      in->facetmarkerlist[idx] = (int) m;
      ierr = DMComplexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
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
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    const int     *cells       = out->tetrahedronlist;
    const double  *meshCoords  = out->pointlist;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexGetFullJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dm, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
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
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in->numberofpoints*dim * sizeof(PetscReal), &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberofpoints     * sizeof(int),       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMComplexGetLabelValue(dm, "marker", v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    ierr = PetscMalloc(in->numberoftetrahedra*in->numberofcorners * sizeof(int), &in->tetrahedronlist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize;

      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    const int     *cells       = out->tetrahedronlist;
    const double  *meshCoords  = out->pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMComplexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dmRefined, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMComplexSetLabelValue(*dmRefined, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMComplexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
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
      SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --download-ctetgen.");
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
  PetscInt       dim, cStart, cEnd;
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
      double  *maxVolumes;
      PetscInt c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) {
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
      PetscInt   c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(PetscReal), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) {
        maxVolumes[c] = refinementLimit;
      }
      ierr = DMComplexRefine_CTetgen(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --download-ctetgen.");
#endif
    } else if (isTetgen) {
#ifdef PETSC_HAVE_TETGEN
      double  *maxVolumes;
      PetscInt c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) {
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
/*@
  DMComplexGetDepthStratum - Get the bounds [start, end) for all points at a certain depth.

  Not Collective

  Input Parameters:
+ dm           - The DMComplex object
- stratumValue - The requested depth

  Output Parameters:
+ start - The first point at this depth
- end   - One beyond the last point at this depth

  Level: developer

.keywords: mesh, points
.seealso: DMComplexGetHeightStratum()
@*/
PetscErrorCode DMComplexGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
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
/*@
  DMComplexGetHeightStratum - Get the bounds [start, end) for all points at a certain height.

  Not Collective

  Input Parameters:
+ dm           - The DMComplex object
- stratumValue - The requested height

  Output Parameters:
+ start - The first point at this height
- end   - One beyond the last point at this height

  Level: developer

.keywords: mesh, points
.seealso: DMComplexGetDepthStratum()
@*/
PetscErrorCode DMComplexGetHeightStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  DMLabel        next = mesh->labels;
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
#define __FUNCT__ "DMComplexCreateSectionInitial"
/* Set the number of dof on each point and separate by fields */
PetscErrorCode DMComplexCreateSectionInitial(DM dm, PetscInt dim, PetscInt numFields, PetscInt numComp[], PetscInt numDof[], PetscSection *section) {
  PetscInt      *numDofTot;
  PetscInt       pStart = 0, pEnd = 0;
  PetscInt       p, d, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &numDofTot);CHKERRQ(ierr);
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
  ierr = PetscFree(numDofTot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSectionBCDof"
/* Set the number of dof on each point and separate by fields
   If constDof is PETSC_DETERMINE, constrain every dof on the point
*/
PetscErrorCode DMComplexCreateSectionBCDof(DM dm, PetscInt numBC, PetscInt bcField[], IS bcPoints[], PetscInt constDof, PetscSection section) {
  PetscInt       numFields;
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *idx;
    PetscInt        n, i;

    if (numFields) {field = bcField[bc];}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt p = idx[i];
      PetscInt       numConst = constDof;

      /* Constrain every dof on the point */
      if (numConst < 0) {
        if (numFields) {
          ierr = PetscSectionGetFieldDof(section, p, field, &numConst);CHKERRQ(ierr);
        } else {
          ierr = PetscSectionGetDof(section, p, &numConst);CHKERRQ(ierr);
        }
      }
      if (numFields) {
        ierr = PetscSectionAddFieldConstraintDof(section, p, field, numConst);CHKERRQ(ierr);
      }
      ierr = PetscSectionAddConstraintDof(section, p, numConst);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSectionBCIndicesAll"
/* Set the constrained indices on each point and separate by fields */
PetscErrorCode DMComplexCreateSectionBCIndicesAll(DM dm, PetscSection section) {
  PetscInt      *maxConstraints;
  PetscInt       numFields, f, pStart = 0, pEnd = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc((numFields+1) * sizeof(PetscInt), &maxConstraints);CHKERRQ(ierr);
  for(f = 0; f <= numFields; ++f) {maxConstraints[f] = 0;}
  for(p = pStart; p < pEnd; ++p) {
    PetscInt cdof;

    if (numFields) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &cdof);CHKERRQ(ierr);
        maxConstraints[f] = PetscMax(maxConstraints[f], cdof);
      }
    } else {
      ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
      maxConstraints[0] = PetscMax(maxConstraints[0], cdof);
    }
  }
  for (f = 0; f < numFields; ++f) {
    maxConstraints[numFields] += maxConstraints[f];
  }
  if (maxConstraints[numFields]) {
    PetscInt *indices;

    ierr = PetscMalloc(maxConstraints[numFields] * sizeof(PetscInt), &indices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof, d;

      ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
      if (cdof) {
        if (cdof > maxConstraints[numFields]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB, "Likely memory corruption, point %D cDof %D > maxConstraints %D", p, cdof, maxConstraints[numFields]);
        if (numFields) {
          PetscInt numConst = 0, foff = 0;

          for (f = 0; f < numFields; ++f) {
            PetscInt cfdof, fdof;

            ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldConstraintDof(section, p, f, &cfdof);CHKERRQ(ierr);
            /* Change constraint numbering from absolute local dof number to field relative local dof number */
            for(d = 0; d < cfdof; ++d) {
              indices[numConst+d] = d;
            }
            ierr = PetscSectionSetFieldConstraintIndices(section, p, f, &indices[numConst]);CHKERRQ(ierr);
            for(d = 0; d < cfdof; ++d) {
              indices[numConst+d] += foff;
            }
            numConst += cfdof;
            foff     += fdof;
          }
          if (cdof != numConst) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of field constraints %D should be %D", numConst, cdof);
        } else {
          for (d = 0; d < cdof; ++d) {
            indices[d] = d;
          }
        }
        ierr = PetscSectionSetConstraintIndices(section, p, indices);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  ierr = PetscFree(maxConstraints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSectionBCIndicesField"
/* Set the constrained field indices on each point */
PetscErrorCode DMComplexCreateSectionBCIndicesField(DM dm, PetscInt field, IS bcPoints, IS constraintIndices, PetscSection section) {
  const PetscInt *points, *indices;
  PetscInt        numFields, maxDof, numPoints, p, numConstraints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if ((field < 0) || (field >= numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, numFields);
  }
  ierr = ISGetLocalSize(bcPoints, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(bcPoints, &points);CHKERRQ(ierr);
  if (!constraintIndices) {
    PetscInt *idx, i;

    ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
    ierr = PetscMalloc(maxDof * sizeof(PetscInt), &idx);CHKERRQ(ierr);
    for(i = 0; i < maxDof; ++i) {idx[i] = i;}
    for(p = 0; p < numPoints; ++p) {
      ierr = PetscSectionSetFieldConstraintIndices(section, points[p], field, idx);CHKERRQ(ierr);
    }
    ierr = PetscFree(idx);CHKERRQ(ierr);
  } else {
    ierr = ISGetLocalSize(constraintIndices, &numConstraints);CHKERRQ(ierr);
    ierr = ISGetIndices(constraintIndices, &indices);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      PetscInt fcdof;

      ierr = PetscSectionGetFieldConstraintDof(section, points[p], field, &fcdof);CHKERRQ(ierr);
      if (fcdof != numConstraints) SETERRQ4(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Section point %d field %d has %d constraints, but yo ugave %d indices", p, field, fcdof, numConstraints);
      ierr = PetscSectionSetFieldConstraintIndices(section, points[p], field, indices);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(constraintIndices, &indices);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(bcPoints, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSectionBCIndices"
/* Set the constrained indices on each point and separate by fields */
PetscErrorCode DMComplexCreateSectionBCIndices(DM dm, PetscSection section) {
  PetscInt      *indices;
  PetscInt       numFields, maxDof, f, pStart = 0, pEnd = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = PetscMalloc(maxDof * sizeof(PetscInt), &indices);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (!numFields) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This function only works after users have set field constraint indices.");
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      PetscInt numConst = 0, foff = 0;

      for (f = 0; f < numFields; ++f) {
        const PetscInt *fcind;
        PetscInt        fdof, fcdof;

        ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        if (fcdof) {ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &fcind);CHKERRQ(ierr);}
        /* Change constraint numbering from field relative local dof number to absolute local dof number */
        for(d = 0; d < fcdof; ++d) {
          indices[numConst+d] = fcind[d]+foff;
        }
        foff     += fdof;
        numConst += fcdof;
      }
      if (cdof != numConst) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_LIB, "Total number of field constraints %D should be %D", numConst, cdof);
      ierr = PetscSectionSetConstraintIndices(section, p, indices);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSection"
/*@C
  DMComplexCreateSection - Create a PetscSection based upon the dof layout specification provided.

  Not Collective

  Input Parameters:
+ dm        - The DMComplex object
. dim       - The spatial dimension of the problem
. numFields - The number of fields in the problem
. numComp   - An array of size numFields that holds the number of components for each field
. numDof    - An array of size numFields*(dim+1) which holds the number of dof for each field on a mesh piece of dimension d
. numBC     - The number of boundary conditions
. bcField   - An array of size numBC giving the field number for each boundry condition
- bcPoints  - An array of size numBC giving an IS holding the sieve points to which each boundary condition applies

  Output Parameter:
. section - The PetscSection object

  Notes: numDof[f*(dim+1)+d] gives the number of dof for field f on sieve points of dimension d. For instance, numDof[1] is the
  nubmer of dof for field 0 on each edge.

  Level: developer

.keywords: mesh, elements
.seealso: DMComplexCreate(), PetscSectionCreate()
@*/
PetscErrorCode DMComplexCreateSection(DM dm, PetscInt dim, PetscInt numFields, PetscInt numComp[], PetscInt numDof[], PetscInt numBC, PetscInt bcField[], IS bcPoints[], PetscSection *section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexCreateSectionInitial(dm, dim, numFields, numComp, numDof, section);CHKERRQ(ierr);
  ierr = DMComplexCreateSectionBCDof(dm, numBC, bcField, bcPoints, PETSC_DETERMINE, *section);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  if (numBC) {ierr = DMComplexCreateSectionBCIndicesAll(dm, *section);CHKERRQ(ierr);}
  {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-section_view", &view);CHKERRQ(ierr);
    if (view) {ierr = PetscSectionView(*section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateCoordinateDM_Complex"
PetscErrorCode DMCreateCoordinateDM_Complex(DM dm, DM *cdm) {
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexClone(dm, cdm);CHKERRQ(ierr);
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*cdm, section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCoordinateSection"
/*@
  DMComplexGetCoordinateSection - Retrieve the layout of coordinate values over the mesh.

  Not Collective

  Input Parameter:
. dm - The DMComplex object

  Output Parameter:
. section - The PetscSection object

  Level: intermediate

.keywords: mesh, coordinates
.seealso: DMGetCoordinateDM(), DMComplexGetDefaultSection(), DMComplexSetDefaultSection()
@*/
PetscErrorCode DMComplexGetCoordinateSection(DM dm, PetscSection *section) {
  DM cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetCoordinateSection"
/*@
  DMComplexSetCoordinateSection - Set the layout of coordinate values over the mesh.

  Not Collective

  Input Parameters:
+ dm      - The DMComplex object
- section - The PetscSection object

  Level: intermediate

.keywords: mesh, coordinates
.seealso: DMComplexGetCoordinateSection(), DMComplexGetDefaultSection(), DMComplexSetDefaultSection()
@*/
PetscErrorCode DMComplexSetCoordinateSection(DM dm, PetscSection section) {
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(cdm, section);CHKERRQ(ierr);
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

/******************************** FEM Support **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMComplexVecGetClosure"
/*@C
  DMComplexVecGetClosure - Get an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or PETSC_NULL to use the default section
. v - The local vector
- point - The sieve point in the DM

  Output Parameters:
+ csize - The number of values in the closure, or PETSC_NULL
- values - The array of values, which is a borrowed array and should not be freed

  Level: intermediate

.seealso DMComplexVecRestoreClosure(), DMComplexVecSetClosure(), DMComplexMatSetClosure()
@*/
PetscErrorCode DMComplexVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, const PetscScalar *values[]) {
  PetscScalar    *array, *vArray;
  PetscInt       *points = PETSC_NULL;
  PetscInt        offsets[32];
  PetscInt        numFields, size, numPoints, pStart, pEnd, p, q, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {
    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for (p = 0, size = 0; p < numPoints*2; p += 2) {
    PetscInt dof, fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    size += dof;
  }
  for (f = 1; f < numFields; ++f) {
    offsets[f+1] += offsets[f];
  }
  if (numFields && offsets[numFields] != size) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], size);
  ierr = DMGetWorkArray(dm, size, PETSC_SCALAR, &array);CHKERRQ(ierr);
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  for (p = 0; p < numPoints*2; p += 2) {
    PetscInt     o = points[p+1];
    PetscInt     dof, off, d;
    PetscScalar *varr;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
    varr = &vArray[off];
    if (numFields) {
      PetscInt fdof, foff, fcomp, f, c;

      for (f = 0, foff = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
        if (o >= 0) {
          for (d = 0; d < fdof; ++d, ++offsets[f]) {
            array[offsets[f]] = varr[foff+d];
          }
        } else {
          ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
          for (d = fdof/fcomp-1; d >= 0; --d) {
            for (c = 0; c < fcomp; ++c, ++offsets[f]) {
              array[offsets[f]] = varr[foff+d*fcomp+c];
            }
          }
        }
        foff += fdof;
      }
    } else {
      if (o >= 0) {
        for (d = 0; d < dof; ++d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      } else {
        for (d = dof-1; d >= 0; --d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      }
    }
  }
  ierr = DMComplexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  if (csize) *csize = size;
  *values = array;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVecRestoreClosure"
/*@C
  DMComplexVecRestoreClosure - Restore the array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or PETSC_NULL to use the default section
. v - The local vector
. point - The sieve point in the DM
. csize - The number of values in the closure, or PETSC_NULL
- values - The array of values, which is a borrowed array and should not be freed

  Level: intermediate

.seealso DMComplexVecGetClosure(), DMComplexVecSetClosure(), DMComplexMatSetClosure()
@*/
PetscErrorCode DMComplexVecRestoreClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, const PetscScalar *values[]) {
  PetscInt        size = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Should work without recalculating size */
  ierr = DMRestoreWorkArray(dm, size, PETSC_SCALAR, values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void add   (PetscScalar *x, PetscScalar y) {*x += y;}
PETSC_STATIC_INLINE void insert(PetscScalar *x, PetscScalar y) {*x  = y;}

#undef __FUNCT__
#define __FUNCT__ "updatePoint_private"
PetscErrorCode updatePoint_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar *, PetscScalar), PetscBool setBC, PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
  PetscInt        cdof;  /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
  a    = &array[off];
  if (!cdof || setBC) {
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        fuse(&a[k], values[k]);
      }
    } else {
      for (k = 0; k < dof; ++k) {
        fuse(&a[k], values[dof-k-1]);
      }
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
        fuse(&a[k], values[k]);
      }
    } else {
      for (k = 0; k < dof; ++k) {
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
  for (f = 0, foff = 0; f < numFields; ++f) {
    PetscInt        fdof, fcomp, fcdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, k, c;

    ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &fcdof);CHKERRQ(ierr);
    if (!fcdof || setBC) {
      if (orientation >= 0) {
        for (k = 0; k < fdof; ++k) {
          fuse(&a[foff+k], values[foffs[f]+k]);
        }
      } else {
        for (k = fdof/fcomp-1; k >= 0; --k) {
          for (c = 0; c < fcomp; ++c) {
            fuse(&a[foff+(fdof/fcomp-1-k)*fcomp+c], values[foffs[f]+k*fcomp+c]);
          }
        }
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (orientation >= 0) {
        for (k = 0; k < fdof; ++k) {
          if ((cind < fcdof) && (k == fcdofs[cind])) {++cind; continue;}
          fuse(&a[foff+k], values[foffs[f]+k]);
        }
      } else {
        for (k = fdof/fcomp-1; k >= 0; --k) {
          for (c = 0; c < fcomp; ++c) {
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
/*@C
  DMComplexVecSetClosure - Set an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or PETSC_NULL to use the default sectionw
. v - The local vector
. point - The sieve point in the DM
. values - The array of values, which is a borrowed array and should not be freed
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

  Level: intermediate

.seealso DMComplexVecGetClosure(), DMComplexMatSetClosure()
@*/
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
    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for (p = 0; p < numPoints*2; p += 2) {
    PetscInt fdof;

    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
  }
  for (f = 1; f < numFields; ++f) {
    offsets[f+1] += offsets[f];
  }
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  if (numFields) {
    switch(mode) {
    case INSERT_VALUES:
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, insert, PETSC_FALSE, o, values, array);
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, insert, PETSC_TRUE,  o, values, array);
      } break;
    case ADD_VALUES:
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, add,    PETSC_FALSE, o, values, array);
      } break;
    case ADD_ALL_VALUES:
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFields_private(section, points[p], offsets, add,    PETSC_TRUE,  o, values, array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
    }
  } else {
    switch(mode) {
    case INSERT_VALUES:
      for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_FALSE, o, &values[off], array);
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, insert, PETSC_TRUE,  o, &values[off], array);
      } break;
    case ADD_VALUES:
      for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_FALSE, o, &values[off], array);
      } break;
    case ADD_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePoint_private(section, points[p], dof, add,    PETSC_TRUE,  o, &values[off], array);
      } break;
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
    }
  }
  ierr = DMComplexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
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
  for (i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%D]mat indices[%D] = %D\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  for (i = 0; i < numIndices; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%D]", rank);CHKERRQ(ierr);
    for (j = 0; j < numIndices; j++) {
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
  PetscInt        cdof;  /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscInt        cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  if (!cdof || setBC) {
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        indices[k] = off+k;
      }
    } else {
      for (k = 0; k < dof; ++k) {
        indices[dof-k-1] = off+k;
      }
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[k] = -(off+k+1);
          ++cind;
        } else {
          indices[k] = off+k-cind;
        }
      }
    } else {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[dof-k-1] = -(off+k+1);
          ++cind;
        } else {
          indices[dof-k-1] = off+k-cind;
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
  for (f = 0, foff = 0; f < numFields; ++f) {
    PetscInt        fdof, fcomp, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, k, c;

    ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cfdof);CHKERRQ(ierr);
    if (!cfdof || setBC) {
      if (orientation >= 0) {
        for (k = 0; k < fdof; ++k) {
          indices[foffs[f]+k] = off+foff+k;
        }
      } else {
        for (k = fdof/fcomp-1; k >= 0; --k) {
          for (c = 0; c < fcomp; ++c) {
            indices[foffs[f]+k*fcomp+c] = off+foff+(fdof/fcomp-1-k)*fcomp+c;
          }
        }
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (orientation >= 0) {
        for (k = 0; k < fdof; ++k) {
          if ((cind < cfdof) && (k == fcdofs[cind])) {
            indices[foffs[f]+k] = -(off+foff+k+1);
            ++cind;
          } else {
            indices[foffs[f]+k] = off+foff+k-cind;
          }
        }
      } else {
        for (k = fdof/fcomp-1; k >= 0; --k) {
          for (c = 0; c < fcomp; ++c) {
            if ((cind < cfdof) && ((fdof/fcomp-1-k)*fcomp+c == fcdofs[cind])) {
              indices[foffs[f]+k*fcomp+c] = -(off+foff+(fdof/fcomp-1-k)*fcomp+c+1);
              ++cind;
            } else {
              indices[foffs[f]+k*fcomp+c] = off+foff+(fdof/fcomp-1-k)*fcomp+c-cind;
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
    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  if (useGlobalDefault) {
    if (useDefault) {
      ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
    } else {
      ierr = PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMComplexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  for (p = 0, numIndices = 0; p < numPoints*2; p += 2) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    numIndices += dof;
  }
  for (f = 1; f < numFields; ++f) {
    offsets[f+1] += offsets[f];
  }
  if (numFields && offsets[numFields] != numIndices) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], numIndices);
  ierr = DMGetWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  if (numFields) {
    for (p = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPointFields_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, o, indices);
    }
  } else {
    for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
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
    ierr2 = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
    CHKERRQ(ierr);
  }
  ierr = DMComplexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeTriangleGeometry_private"
PetscErrorCode DMComplexComputeTriangleGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  const PetscInt     dim = 2;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMComplexVecGetClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
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
        for (f = 0; f < dim; f++) {
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
  ierr = DMComplexVecRestoreClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeRectangleGeometry_private"
PetscErrorCode DMComplexComputeRectangleGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  const PetscInt     dim = 2;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMComplexVecGetClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f*2+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
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
    PetscLogFlops(5.0);
  }
  ierr = DMComplexVecRestoreClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeTetrahedronGeometry_private"
PetscErrorCode DMComplexComputeTetrahedronGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  const PetscInt     dim = 3;
  PetscInt           d, f;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMComplexVecGetClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
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
  ierr = DMComplexVecRestoreClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeHexahedronGeometry_private"
PetscErrorCode DMComplexComputeHexahedronGeometry_private(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  const PetscInt     dim = 3;
  PetscInt           d;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMComplexVecGetClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) {
      v0[d] = PetscRealPart(coords[d]);
    }
  }
  if (J) {
    for (d = 0; d < dim; d++) {
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
  ierr = DMComplexVecRestoreClosure(dm, coordSection, coordinates, e, PETSC_NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeCellGeometry"
/*@C
  DMComplexComputeCellGeometry - Compute the Jacobian, inverse Jacobian, and Jacobian determinant for a given cell

  Collective on DM

  Input Arguments:
+ dm   - the DM
- cell - the cell

  Output Arguments:
+ v0   - the translation part of this affine transform
. J    - the Jacobian of the transform to the reference element
. invJ - the inverse of the Jacobian
- detJ - the Jacobian determinant

  Level: advanced

.seealso: DMComplexGetCoordinateSection(), DMComplexGetCoordinateVec()
@*/
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
    for (i = 0; i < faceSizeTri; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTri, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeTri;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTriSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTriSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeTri; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTri; ++cVertex) {
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
    for (i = 0; i < faceSizeQuad; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuad, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuad;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeQuad; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuad; ++cVertex) {
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
    for (i = 0; i < faceSizeHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeHex; ++cVertex) {
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
    for (i = 0; i < faceSizeTet; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTet, sortedIndices);CHKERRQ(ierr);
    for (iFace=0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeTet;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTetSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTetSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesTetSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesTetSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeTet; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTet; ++cVertex) {
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
    for (i = 0; i < faceSizeQuadHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuadHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuadHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesQuadHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesQuadHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeQuadHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuadHex; ++cVertex) {
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
    for (f = 0; f < faceSize; ++f) {
      faceVertices[f] = origVertices[faceSize-1 - f];
    }
  } else {
    if (debug) {ierr = PetscPrintf(comm, "  Keeping initial face orientation\n");CHKERRQ(ierr);}
    for (f = 0; f < faceSize; ++f) {
      faceVertices[f] = origVertices[f];
    }
  }
  if (posOriented) {*posOriented = posOrient;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetOrientedFace"
/*
    Given a cell and a face, as a set of vertices,
      return the oriented face, as a set of vertices, in faceVertices
    The orientation is such that the face normal points out of the cell
*/
PetscErrorCode DMComplexGetOrientedFace(DM dm, PetscInt cell, PetscInt faceSize, const PetscInt face[], PetscInt numCorners, PetscInt indices[], PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  const PetscInt *cone;
  PetscInt        coneSize, v, f, v2;
  PetscInt        oppositeVertex = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMComplexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  for (v = 0, v2 = 0; v < coneSize; ++v) {
    PetscBool found  = PETSC_FALSE;

    for (f = 0; f < faceSize; ++f) {
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

PETSC_STATIC_INLINE PetscInt epsilon(PetscInt i, PetscInt j, PetscInt k)
{
  switch(i) {
  case 0:
    switch(j) {
    case 0: return 0;
    case 1:
      switch(k) {
      case 0: return 0;
      case 1: return 0;
      case 2: return 1;
      }
    case 2:
      switch(k) {
      case 0: return 0;
      case 1: return -1;
      case 2: return 0;
      }
    }
  case 1:
    switch(j) {
    case 0:
      switch(k) {
      case 0: return 0;
      case 1: return 0;
      case 2: return -1;
      }
    case 1: return 0;
    case 2:
      switch(k) {
      case 0: return 1;
      case 1: return 0;
      case 2: return 0;
      }
    }
  case 2:
    switch(j) {
    case 0:
      switch(k) {
      case 0: return 0;
      case 1: return 1;
      case 2: return 0;
      }
    case 1:
      switch(k) {
      case 0: return -1;
      case 1: return 0;
      case 2: return 0;
      }
    case 2: return 0;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateRigidBody"
/*@C
  DMComplexCreateRigidBody - create rigid body modes from coordinates

  Collective on DM

  Input Arguments:
+ dm - the DM
. section - the local section associated with the rigid field, or PETSC_NULL for the default section
- globalSection - the global section associated with the rigid field, or PETSC_NULL for the default section

  Output Argument:
. sp - the null space

  Note: This is necessary to take account of Dirichlet conditions on the displacements

  Level: advanced

.seealso: MatNullSpaceCreate()
@*/
PetscErrorCode DMComplexCreateRigidBody(DM dm, PetscSection section, PetscSection globalSection, MatNullSpace *sp)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  Vec            coordinates, localMode, mode[6];
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       dim, vStart, vEnd, v, n, m, d, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL, sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!section)       {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!globalSection) {ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);}
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &n);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  m    = (dim*(dim+1))/2;
  ierr = VecCreate(comm, &mode[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(mode[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(mode[0]);CHKERRQ(ierr);
  for (i = 1; i < m; ++i) {ierr = VecDuplicate(mode[0], &mode[i]);CHKERRQ(ierr);}
  /* Assume P1 */
  ierr = DMGetLocalVector(dm, &localMode);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    PetscScalar values[3] = {0.0, 0.0, 0.0};

    values[d] = 1.0;
    ierr = VecSet(localMode, 0.0);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      ierr = DMComplexVecSetClosure(dm, section, localMode, v, values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobalBegin(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (d = dim; d < dim*(dim+1)/2; ++d) {
    PetscInt i, j, k = dim > 2 ? d - dim : d;

    ierr = VecSet(localMode, 0.0);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscScalar values[3] = {0.0, 0.0, 0.0};
      PetscInt    off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          values[j] += epsilon(i, j, k)*PetscRealPart(coords[off+i]);
        }
      }
      ierr = DMComplexVecSetClosure(dm, section, localMode, v, values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobalBegin(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localMode);CHKERRQ(ierr);
  for (i = 0; i < dim; ++i) {ierr = VecNormalize(mode[i], PETSC_NULL);CHKERRQ(ierr);}
  /* Orthonormalize system */
  for (i = dim; i < m; ++i) {
    PetscScalar dots[6];

    ierr = VecMDot(mode[i], i, mode, dots);CHKERRQ(ierr);
    for (j = 0; j < i; ++j) dots[j] *= -1.0;
    ierr = VecMAXPY(mode[i], i, dots, mode);CHKERRQ(ierr);
    ierr = VecNormalize(mode[i], PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, m, mode, sp);CHKERRQ(ierr);
  for (i = 0; i< m; ++i) {ierr = VecDestroy(&mode[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetVTKBounds"
PetscErrorCode DMComplexGetVTKBounds(DM dm, PetscInt *cMax, PetscInt *vMax)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cMax) *cMax = mesh->vtkCellMax;
  if (vMax) *vMax = mesh->vtkVertexMax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetVTKBounds"
PetscErrorCode DMComplexSetVTKBounds(DM dm, PetscInt cMax, PetscInt vMax)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cMax >= 0) mesh->vtkCellMax   = cMax;
  if (vMax >= 0) mesh->vtkVertexMax = vMax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetVTKCellHeight"
PetscErrorCode DMComplexGetVTKCellHeight(DM dm, PetscInt *cellHeight)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cellHeight, 2);
  *cellHeight = mesh->vtkCellHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetVTKCellHeight"
PetscErrorCode DMComplexSetVTKCellHeight(DM dm, PetscInt cellHeight)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->vtkCellHeight = cellHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexInsertFace_Private"
/*
  DMComplexInsertFace_Private - Puts a face into the mesh

  Not collective

  Input Parameters:
  + dm              - The DMComplex
  . numFaceVertex   - The number of vertices in the face
  . faceVertices    - The vertices in the face for dm
  . subfaceVertices - The vertices in the face for subdm
  . numCorners      - The number of vertices in the cell
  . cell            - A cell in dm containing the face
  . subcell         - A cell in subdm containing the face
  . firstFace       - First face in the mesh
  - newFacePoint    - Next face in the mesh

  Output Parameters:
  . newFacePoint - Contains next face point number on input, updated on output

  Level: developer
*/
PetscErrorCode DMComplexInsertFace_Private(DM dm, DM subdm, PetscInt numFaceVertices, const PetscInt faceVertices[], const PetscInt subfaceVertices[], PetscInt numCorners, PetscInt cell, PetscInt subcell, PetscInt firstFace, PetscInt *newFacePoint)
{
  MPI_Comm        comm    = ((PetscObject) dm)->comm;
  DM_Complex     *submesh = (DM_Complex *) subdm->data;
  const PetscInt *faces;
  PetscInt        numFaces, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetConeSize(subdm, subcell, &coneSize);CHKERRQ(ierr);
  if (coneSize != 1) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size of cell %d is %d != 1", cell, coneSize);
#if 0
  /* Cannot use this because support() has not been constructed yet */
  ierr = DMComplexGetJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
#else
  {
    PetscInt f;

    numFaces = 0;
    ierr = DMGetWorkArray(subdm, 1, PETSC_INT, &faces);CHKERRQ(ierr);
    for(f = firstFace; f < *newFacePoint; ++f) {
      PetscInt dof, off, d;

      ierr = PetscSectionGetDof(submesh->coneSection, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(submesh->coneSection, f, &off);CHKERRQ(ierr);
      /* Yes, I know this is quadratic, but I expect the sizes to be <5 */
      for(d = 0; d < dof; ++d) {
        const PetscInt p = submesh->cones[off+d];
        PetscInt       v;

        for(v = 0; v < numFaceVertices; ++v) {
          if (subfaceVertices[v] == p) break;
        }
        if (v == numFaceVertices) break;
      }
      if (d == dof) {
        numFaces = 1;
        ((PetscInt *) faces)[0] = f;
      }
    }
  }
#endif
  if (numFaces > 1) {
    SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Vertex set had %d faces, not one", numFaces);
  } else if (numFaces == 1) {
    /* Add the other cell neighbor for this face */
    ierr = DMComplexSetCone(subdm, cell, faces);CHKERRQ(ierr);
  } else {
    PetscInt *indices, *origVertices, *orientedVertices, *orientedSubVertices, v, ov;
    PetscBool posOriented;

    ierr = DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
    origVertices = &orientedVertices[numFaceVertices];
    indices      = &orientedVertices[numFaceVertices*2];
    orientedSubVertices = &orientedVertices[numFaceVertices*3];
    ierr = DMComplexGetOrientedFace(dm, cell, numFaceVertices, faceVertices, numCorners, indices, origVertices, orientedVertices, &posOriented);CHKERRQ(ierr);
    /* TODO: I know that routine should return a permutation, not the indices */
    for(v = 0; v < numFaceVertices; ++v) {
      const PetscInt vertex = faceVertices[v], subvertex = subfaceVertices[v];
      for(ov = 0; ov < numFaceVertices; ++ov) {
        if (orientedVertices[ov] == vertex) {
          orientedSubVertices[ov] = subvertex;
          break;
        }
      }
      if (ov == numFaceVertices) SETERRQ1(comm, PETSC_ERR_PLIB, "Could not find face vertex %d in orientated set", vertex);
    }
    ierr = DMComplexSetCone(subdm, *newFacePoint, orientedSubVertices);CHKERRQ(ierr);
    ierr = DMComplexSetCone(subdm, subcell, newFacePoint);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
    ++(*newFacePoint);
  }
  ierr = DMComplexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSubmesh"
PetscErrorCode DMComplexCreateSubmesh(DM dm, const char label[], DM *subdm)
{
  MPI_Comm        comm = ((PetscObject) dm)->comm;
  DM_Complex     *submesh;
  PetscBool       boundaryFaces = PETSC_FALSE;
  PetscSection    coordSection, subCoordSection;
  Vec             coordinates, subCoordinates;
  PetscScalar    *coords, *subCoords;
  IS              labelIS;
  const PetscInt *subVertices;
  PetscInt       *subVerticesActive, *tmpPoints;
  PetscInt       *subCells = PETSC_NULL;
  PetscInt        numSubVertices, numSubVerticesActive, firstSubVertex, numSubCells = 0, maxSubCells = 0, numOldSubCells;
  PetscInt       *face, *subface, maxConeSize, numSubFaces = 0, firstSubFace, newFacePoint, nFV = 0, coordSize;
  PetscInt        dim; /* Right now, do not specify dimension */
  PetscInt        cStart, cEnd, cMax, c, vStart, vEnd, vMax, v, p, corner, i, d, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetMaxSizes(dm, &maxConeSize, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, &cMax, &vMax);CHKERRQ(ierr);
  if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
  if (vMax >= 0) {vEnd = PetscMin(vEnd, vMax);}
  ierr = DMGetWorkArray(dm, 2*maxConeSize, PETSC_INT, &face);CHKERRQ(ierr);
  subface = &face[maxConeSize];
  ierr = DMCreate(comm, subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  ierr = DMComplexGetStratumIS(dm, label, 1, &labelIS);CHKERRQ(ierr);
  ierr = ISGetSize(labelIS, &numSubVertices);CHKERRQ(ierr);
  ierr = ISGetIndices(labelIS, &subVertices);CHKERRQ(ierr);
  maxSubCells = numSubVertices;
  ierr = PetscMalloc(maxSubCells * sizeof(PetscInt), &subCells);CHKERRQ(ierr);
  ierr = PetscMalloc(numSubVertices * sizeof(PetscInt), &subVerticesActive);CHKERRQ(ierr);
  ierr = PetscMemzero(subVerticesActive, numSubVertices * sizeof(PetscInt));CHKERRQ(ierr);
  for(v = 0; v < numSubVertices; ++v) {
    const PetscInt vertex = subVertices[v];
    PetscInt *star = PETSC_NULL;
    PetscInt  starSize, numCells = 0;

    ierr = DMComplexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for(p = 0; p < starSize*2; p += 2) {
      const PetscInt point = star[p];
      if ((point >= cStart) && (point < cEnd)) {
        star[numCells++] = point;
      }
    }
    numOldSubCells = numSubCells;
    for(c = 0; c < numCells; ++c) {
      const PetscInt cell    = star[c];
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize, numCorners = 0, faceSize = 0;
      PetscInt       cellLoc;

      ierr = PetscFindInt(cell, numOldSubCells, subCells, &cellLoc);CHKERRQ(ierr);
      if (cellLoc >= 0) continue;
      ierr = DMComplexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for(p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = closure[p];
        if ((point >= vStart) && (point < vEnd)) {
          closure[numCorners++] = point;
        }
      }
      if (!nFV) {ierr = DMComplexGetNumFaceVertices(dm, numCorners, &nFV);CHKERRQ(ierr);}
      for(corner = 0; corner < numCorners; ++corner) {
        const PetscInt cellVertex = closure[corner];
        PetscInt       subVertex;

        ierr = PetscFindInt(cellVertex, numSubVertices, subVertices, &subVertex);CHKERRQ(ierr);
        if (subVertex >= 0) { /* contains submesh vertex */
          for(i = 0; i < faceSize; ++i) {if (cellVertex == face[i]) break;}
          if (i == faceSize) {
            if (faceSize >= maxConeSize) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices in face %d should not exceed %d", faceSize+1, maxConeSize);
            face[faceSize]    = cellVertex;
            subface[faceSize] = subVertex;
            ++faceSize;
          }
        }
      }
      ierr = DMComplexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if (faceSize >= nFV) {
        if (faceSize > nFV && !boundaryFaces) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
        if (numSubCells >= maxSubCells) {
          PetscInt *tmpCells;
          maxSubCells *= 2;
          ierr = PetscMalloc(maxSubCells * sizeof(PetscInt), &tmpCells);CHKERRQ(ierr);
          ierr = PetscMemcpy(tmpCells, subCells, numSubCells * sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscFree(subCells);CHKERRQ(ierr);
          subCells = tmpCells;
        }
        /* TOOD: Maybe overestimate then squeeze out empty faces */
        if (faceSize > nFV) {
          /* TODO: This is tricky. Maybe just add all faces */
          numSubFaces++;
        } else {
          numSubFaces++;
        }
        for(f = 0; f < faceSize; ++f) {
          subVerticesActive[subface[f]] = 1;
        }
        subCells[numSubCells++] = cell;
      }
    }
    ierr = DMComplexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    ierr = PetscSortRemoveDupsInt(&numSubCells, subCells);CHKERRQ(ierr);
  }
  /* Pick out active subvertices */
  for(v = 0, numSubVerticesActive = 0; v < numSubVertices; ++v) {
    if (subVerticesActive[v]) {
      subVerticesActive[numSubVerticesActive++] = subVertices[v];
    }
  }
  ierr = DMComplexSetChart(*subdm, 0, numSubCells+numSubFaces+numSubVerticesActive);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVerticesActive;
  newFacePoint   = firstSubFace;
  for(c = 0; c < numSubCells; ++c) {
    ierr = DMComplexSetConeSize(*subdm, c, 1);CHKERRQ(ierr);
  }
  for(f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    ierr = DMComplexSetConeSize(*subdm, f, nFV);CHKERRQ(ierr);
  }
  ierr = DMSetUp(*subdm);CHKERRQ(ierr);
  /* Create face cones */
  for(c = 0; c < numSubCells; ++c) {
    const PetscInt cell    = subCells[c];
    PetscInt      *closure = PETSC_NULL;
    PetscInt       closureSize, numCorners = 0, faceSize = 0;

    ierr = DMComplexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for(p = 0; p < closureSize*2; p += 2) {
      const PetscInt point = closure[p];
      if ((point >= vStart) && (point < vEnd)) {
        closure[numCorners++] = point;
      }
    }
    for(corner = 0; corner < numCorners; ++corner) {
      const PetscInt cellVertex = closure[corner];
      PetscInt       subVertex;

      ierr = PetscFindInt(cellVertex, numSubVerticesActive, subVerticesActive, &subVertex);CHKERRQ(ierr);
      if (subVertex >= 0) { /* contains submesh vertex */
        for(i = 0; i < faceSize; ++i) {if (cellVertex == face[i]) break;}
        if (i == faceSize) {
          face[faceSize]    = cellVertex;
          subface[faceSize] = numSubCells+subVertex;
          ++faceSize;
        }
      }
    }
    ierr = DMComplexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (faceSize >= nFV) {
      if (faceSize > nFV && !boundaryFaces) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
      // Here we allow a set of vertices to lie completely on a boundary cell (like a corner tetrahedron)
      //   We have to take all the faces, and discard those in the interior
      //   We check the join of the face vertices, which produces 2 cells if in the interior
#if 0
      // This object just calls insert on each face that comes from subsets()
      // In fact, we can just always acll subsets(), since when we pass a single face it is a single call
      FaceInserterV<FlexMesh::sieve_type> inserter(mesh, sieve, subSieve, f, *c_iter, numCorners, indices, &origVertices, &faceVertices, &submeshCells);
      PointArray                          faceVec(face->begin(), face->end());

      subsets(faceVec, nFV, inserter);
#endif
      ierr = DMComplexInsertFace_Private(dm, *subdm, faceSize, face, subface, numCorners, cell, c, firstSubFace, &newFacePoint);CHKERRQ(ierr);
    }
  }
  ierr = DMComplexSymmetrize(*subdm);CHKERRQ(ierr);
  ierr = DMComplexStratify(*subdm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(*subdm, &subCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVerticesActive);CHKERRQ(ierr);
  for (v = firstSubVertex; v < firstSubVertex+numSubVerticesActive; ++v) {
    ierr = PetscSectionSetDof(subCoordSection, v, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &subCoordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(subCoordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
  ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
  for(v = 0; v < numSubVerticesActive; ++v) {
    const PetscInt vertex    = subVerticesActive[v];
    const PetscInt subVertex = firstSubVertex+v;
    PetscInt dof, off, sdof, soff;

    ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(subCoordSection, subVertex, &sdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(subCoordSection, subVertex, &soff);CHKERRQ(ierr);
    if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subVertex, vertex, dof);
    for(d = 0; d < dof; ++d) {
      subCoords[soff+d] = coords[off+d];
    }
  }
  ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*subdm, subCoordinates);CHKERRQ(ierr);

  ierr = DMComplexSetVTKCellHeight(*subdm, 1);CHKERRQ(ierr);
  /* Create map from submesh points to original mesh points */
  submesh = (DM_Complex *) (*subdm)->data;
  ierr = PetscMalloc((numSubCells+numSubVerticesActive) * sizeof(PetscInt), &tmpPoints);CHKERRQ(ierr);
  for(c = 0; c < numSubCells; ++c) {
    tmpPoints[c] = subCells[c];
  }
  for(v = numSubCells; v < numSubCells+numSubVerticesActive; ++v) {
    tmpPoints[v] = subVerticesActive[v-numSubCells];
  }
  ierr = ISCreateGeneral(comm, numSubCells+numSubVerticesActive, tmpPoints, PETSC_OWN_POINTER, &submesh->subpointMap);CHKERRQ(ierr);

  ierr = PetscFree(subCells);CHKERRQ(ierr);
  ierr = PetscFree(subVerticesActive);CHKERRQ(ierr);
  ierr = ISRestoreIndices(labelIS, &subVertices);CHKERRQ(ierr);
  ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 2*maxConeSize, PETSC_INT, &face);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateNumbering_Private"
/* We can easily have a form that takes an IS instead */
PetscErrorCode DMComplexCreateNumbering_Private(DM dm, PetscInt pStart, PetscInt pEnd, PetscSF sf, IS *numbering)
{
  PetscSection   section, globalSection;
  PetscInt      *numbers, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(section, p, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(section, sf, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &numbers);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(((PetscObject) dm)->comm, pEnd - pStart, numbers, PETSC_OWN_POINTER, numbering);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetCellNumbering"
PetscErrorCode DMComplexGetCellNumbering(DM dm, IS *globalCellNumbers)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       cellHeight, cStart, cEnd, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalCellNumbers) {
    ierr = DMComplexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
    ierr = DMComplexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMComplexGetVTKBounds(dm, &cMax, PETSC_NULL);CHKERRQ(ierr);
    if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
    ierr = DMComplexCreateNumbering_Private(dm, cStart, cEnd, dm->sf, &mesh->globalCellNumbers);CHKERRQ(ierr);
  }
  *globalCellNumbers = mesh->globalCellNumbers;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetVertexNumbering"
PetscErrorCode DMComplexGetVertexNumbering(DM dm, IS *globalVertexNumbers)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscInt       vStart, vEnd, vMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalVertexNumbers) {
    ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMComplexGetVTKBounds(dm, PETSC_NULL, &vMax);CHKERRQ(ierr);
    if (vMax >= 0) {vEnd = PetscMin(vEnd, vMax);}
    ierr = DMComplexCreateNumbering_Private(dm, vStart, vEnd, dm->sf, &mesh->globalVertexNumbers);CHKERRQ(ierr);
  }
  *globalVertexNumbers = mesh->globalVertexNumbers;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetSubpointMap"
PetscErrorCode DMComplexGetSubpointMap(DM dm, IS *subpointMap)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointMap, 2);
  *subpointMap = mesh->subpointMap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetSubpointMap"
/* Note: Should normally not be called by the user, since it is set in DMComplexCreateSubmesh() */
PetscErrorCode DMComplexSetSubpointMap(DM dm, IS subpointMap)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(subpointMap, IS_CLASSID, 2);
  mesh->subpointMap = subpointMap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetScale"
PetscErrorCode DMComplexGetScale(DM dm, PetscUnit unit, PetscReal *scale)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(scale, 3);
  *scale = mesh->scale[unit];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexSetScale"
PetscErrorCode DMComplexSetScale(DM dm, PetscUnit unit, PetscReal scale)
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->scale[unit] = scale;
  PetscFunctionReturn(0);
}


/*******************************************************************************
This should be in a separate Discretization object, but I am not sure how to lay
it out yet, so I am stuffing things here while I experiment.
*******************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "DMComplexSetFEMIntegration"
PetscErrorCode DMComplexSetFEMIntegration(DM dm,
                                          PetscErrorCode (*integrateResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                                                                 const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]),
                                          PetscErrorCode (*integrateJacobianActionFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[], const PetscScalar[],
                                                                                       const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]),
                                          PetscErrorCode (*integrateJacobianFEM)(PetscInt, PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                                                                 const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]))
{
  DM_Complex *mesh = (DM_Complex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->integrateResidualFEM       = integrateResidualFEM;
  mesh->integrateJacobianActionFEM = integrateJacobianActionFEM;
  mesh->integrateJacobianFEM       = integrateJacobianFEM;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexProjectFunction"
/*@C
  DMComplexProjectFunction - This projects the given function into the function space provided.

  Input Parameters:
+ dm      - The DM
. numComp - The number of components (functions)
. funcs   - The coordinate functions to evaluate
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

  Level: developer

  Note:
  This currently just calls the function with the coordinates of each vertex and edge midpoint, and stores the result in a vector.
  We will eventually fix it.

,seealso: DMComplexComputeL2Diff()
*/
PetscErrorCode DMComplexProjectFunction(DM dm, PetscInt numComp, PetscScalar (**funcs)(const PetscReal []), InsertMode mode, Vec X)
{
  Vec            localX, coordinates;
  PetscSection   section, cSection;
  PetscInt       dim, vStart, vEnd, v, c, d;
  PetscScalar   *values, *cArray;
  PetscReal     *coords;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscMalloc(numComp * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &cArray);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(cSection, vStart, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal),&coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(cSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
    if (dof > dim) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Cannot have more coordinates %d then dimensions %d", dof, dim);
    for(d = 0; d < dof; ++d) {
      coords[d] = PetscRealPart(cArray[off+d]);
    }
    for(c = 0; c < numComp; ++c) {
      values[c] = (*funcs[c])(coords);
    }
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &cArray);CHKERRQ(ierr);
  /* Temporary, must be replaced by a projection on the finite element basis */
  {
    PetscInt eStart = 0, eEnd = 0, e, depth;

    ierr = DMComplexGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
    --depth;
    if (depth > 1) {ierr = DMComplexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);}
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt *cone;
      PetscInt        coneSize, d;
      PetscScalar    *coordsA, *coordsB;

      ierr = DMComplexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, e, &cone);CHKERRQ(ierr);
      if (coneSize != 2) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Cone size %d for point %d should be 2", coneSize, e);
      ierr = VecGetValuesSection(coordinates, cSection, cone[0], &coordsA);CHKERRQ(ierr);
      ierr = VecGetValuesSection(coordinates, cSection, cone[1], &coordsB);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coords[d] = 0.5*(PetscRealPart(coordsA[d]) + PetscRealPart(coordsB[d]));
      }
      for (c = 0; c < numComp; ++c) {
        values[c] = (*funcs[c])(coords);
      }
      ierr = VecSetValuesSection(localX, section, e, values, mode);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
#if 0
  const PetscInt localDof = this->_mesh->sizeWithBC(s, *cells->begin());
  PetscReal      detJ;

  ierr = PetscMalloc(localDof * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(this->_mesh->getSieve()->getMaxConeSize(), dim+1)+1, true);

  for (PetscInt c = cStart; c < cEnd; ++c) {
    ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), c, pV);
    const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
    const int                          oSize   = pV.getSize();
    int                                v       = 0;

    ierr = DMComplexComputeCellGeometry(dm, c, v0, J, PETSC_NULL, &detJ);CHKERRQ(ierr);
    for (PetscInt cl = 0; cl < oSize; ++cl) {
      const PetscInt fDim;

      ierr = PetscSectionGetDof(oPoints[cl], &fDim);CHKERRQ(ierr);
      if (pointDim) {
        for (PetscInt d = 0; d < fDim; ++d, ++v) {
          values[v] = (*this->_options.integrate)(v0, J, v, initFunc);
        }
      }
    }
    ierr = DMComplexVecSetClosure(dm, PETSC_NULL, localX, c, values);CHKERRQ(ierr);
    pV.clear();
  }
  ierr = PetscFree2(v0,J);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeL2Diff"
/*@C
  DMComplexComputeL2Diff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Input Parameters:
+ dm    - The DM
. quad  - The PetscQuadrature object for each field
. funcs - The functions to evaluate for each field component
- X     - The coefficient vector u_h

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMComplexProjectFunction()
*/
PetscErrorCode DMComplexComputeL2Diff(DM dm, PetscQuadrature quad[], PetscScalar (**funcs)(const PetscReal []), Vec X, PetscReal *diff) {
  const PetscInt   debug = 0;
  PetscSection     section;
  Vec              localX;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal        localDiff;
  PetscInt         dim, numFields, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscReal          elemDiff = 0.0;

    ierr = DMComplexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMComplexVecGetClosure(dm, PETSC_NULL, localX, c, PETSC_NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      const PetscInt   numQuadPoints = quad[field].numQuadPoints;
      const PetscReal *quadPoints    = quad[field].quadPoints;
      const PetscReal *quadWeights   = quad[field].quadWeights;
      const PetscInt   numBasisFuncs = quad[field].numBasisFuncs;
      const PetscInt   numBasisComps = quad[field].numComponents;
      const PetscReal *basis         = quad[field].basis;
      PetscInt         q, d, e, fc, f;

      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, numBasisFuncs*numBasisComps, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        for (fc = 0; fc < numBasisComps; ++fc) {
          const PetscReal funcVal     = PetscRealPart((*funcs[comp+fc])(coords));
          PetscReal       interpolant = 0.0;
          for (f = 0; f < numBasisFuncs; ++f) {
            const PetscInt fidx = f*numBasisComps+fc;
            interpolant += PetscRealPart(x[fieldOffset+fidx])*basis[q*numBasisFuncs*numBasisComps+fidx];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ;
        }
      }
      comp        += numBasisComps;
      fieldOffset += numBasisFuncs*numBasisComps;
    }
    ierr = DMComplexVecRestoreClosure(dm, PETSC_NULL, localX, c, PETSC_NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeResidualFEM"
/*@
  DMComplexComputeResidualFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: DMComplexComputeJacobianActionFEM()
*/
PetscErrorCode DMComplexComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM_Complex      *mesh = (DM_Complex *) dm->data;
  PetscFEM        *fem  = (PetscFEM *) &((DM *) user)[1];
  PetscQuadrature *quad = fem->quad;
  PetscSection     section;
  PetscReal       *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u;
  PetscInt         dim, numFields, field, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(ResidualFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscInt           i;

    ierr = DMComplexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMComplexVecGetClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);

    for (i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
    ierr = DMComplexVecRestoreClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);
  }
  for (field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = quad[field].numQuadPoints;
    const PetscInt numBasisFuncs = quad[field].numBasisFuncs;
    void (*f0)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]) = fem->f0Funcs[field];
    void (*f1)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]) = fem->f1Funcs[field];
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = numBatchesTmp;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    ierr = (*mesh->integrateResidualFEM)(numChunks*numBatches*batchSize, numFields, field, quad, u, v0, J, invJ, detJ, f0, f1, elemVec);CHKERRQ(ierr);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;
    ierr = (*mesh->integrateResidualFEM)(numRemainder, numFields, field, quad, &u[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                         f0, f1, &elemVec[offset*cellDof]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMComplexVecSetClosure(dm, PETSC_NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  if (mesh->printFEM) {
    PetscMPIInt rank, numProcs;
    PetscInt    p;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject) dm)->comm, &numProcs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
    for (p = 0; p < numProcs; ++p) {
      if (p == rank) {
        Vec f;

        ierr = VecDuplicate(F, &f);CHKERRQ(ierr);
        ierr = VecCopy(F, f);CHKERRQ(ierr);
        ierr = VecChop(f, 1.0e-10);CHKERRQ(ierr);
        ierr = VecView(f, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&f);CHKERRQ(ierr);
      }
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  /* ierr = PetscLogEventEnd(ResidualFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeJacobianActionFEM"
/*@C
  DMComplexComputeJacobianActionFEM - Form the local action of Jacobian J(u) on the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. J  - The Jacobian shell matrix
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: DMComplexComputeResidualFEM()
*/
PetscErrorCode DMComplexComputeJacobianActionFEM(DM dm, Mat Jac, Vec X, Vec F, void *user)
{
  DM_Complex      *mesh = (DM_Complex *) dm->data;
  PetscFEM        *fem  = (PetscFEM *) &((DM *) user)[1];
  PetscQuadrature *quad = fem->quad;
  PetscSection     section;
  JacActionCtx    *jctx;
  PetscReal       *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u, *a;
  PetscInt         dim, numFields, field, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof  = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(JacobianActionFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  ierr = PetscMalloc7(numCells*cellDof,PetscScalar,&u,numCells*cellDof,PetscScalar,&a,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscInt           i;

    ierr = DMComplexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMComplexVecGetClosure(dm, PETSC_NULL, jctx->u, c, PETSC_NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
    ierr = DMComplexVecRestoreClosure(dm, PETSC_NULL, jctx->u, c, PETSC_NULL, &x);CHKERRQ(ierr);
    ierr = DMComplexVecGetClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) {
      a[c*cellDof+i] = x[i];
    }
    ierr = DMComplexVecRestoreClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);
  }
  for (field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = quad[field].numQuadPoints;
    const PetscInt numBasisFuncs = quad[field].numBasisFuncs;
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = numBatchesTmp;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    ierr = (*mesh->integrateJacobianActionFEM)(numChunks*numBatches*batchSize, numFields, field, quad, u, a, v0, J, invJ, detJ, fem->g0Funcs, fem->g1Funcs, fem->g2Funcs, fem->g3Funcs, elemVec);CHKERRQ(ierr);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;
    ierr = (*mesh->integrateJacobianActionFEM)(numRemainder, numFields, field, quad, &u[offset*cellDof], &a[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                               fem->g0Funcs, fem->g1Funcs, fem->g2Funcs, fem->g3Funcs, &elemVec[offset*cellDof]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMComplexVecSetClosure(dm, PETSC_NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree7(u,a,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  if (mesh->printFEM) {
    PetscMPIInt rank, numProcs;
    PetscInt    p;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject) dm)->comm, &numProcs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian Action:\n");CHKERRQ(ierr);
    for (p = 0; p < numProcs; ++p) {
      if (p == rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  /* ierr = PetscLogEventEnd(JacobianActionFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexComputeJacobianFEM"
/*@
  DMComplexComputeJacobianFEM - Form the local portion of the Jacobian matrix J at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormFunctionLocal()
*/
PetscErrorCode DMComplexComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP, void *user)
{
  DM_Complex      *mesh = (DM_Complex *) dm->data;
  PetscFEM        *fem  = (PetscFEM *) &((DM *) user)[1];
  PetscQuadrature *quad = fem->quad;
  PetscSection     section;
  PetscReal       *v0, *J, *invJ, *detJ;
  PetscScalar     *elemMat, *u;
  PetscInt         dim, numFields, field, fieldI, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof = 0;
  PetscBool        isShell;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(JacobianFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof*cellDof,PetscScalar,&elemMat);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscInt           i;

    ierr = DMComplexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMComplexVecGetClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);

    for (i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
    ierr = DMComplexVecRestoreClosure(dm, PETSC_NULL, X, c, PETSC_NULL, &x);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(elemMat, numCells*cellDof*cellDof * sizeof(PetscScalar));CHKERRQ(ierr);
  for (fieldI = 0; fieldI < numFields; ++fieldI) {
    const PetscInt numQuadPoints = quad[fieldI].numQuadPoints;
    const PetscInt numBasisFuncs = quad[fieldI].numBasisFuncs;
    PetscInt       fieldJ;

    for (fieldJ = 0; fieldJ < numFields; ++fieldJ) {
      void (*g0)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[]) = fem->g0Funcs[fieldI*numFields+fieldJ];
      void (*g1)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]) = fem->g1Funcs[fieldI*numFields+fieldJ];
      void (*g2)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]) = fem->g2Funcs[fieldI*numFields+fieldJ];
      void (*g3)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]) = fem->g3Funcs[fieldI*numFields+fieldJ];
      /* Conforming batches */
      PetscInt blockSize  = numBasisFuncs*numQuadPoints;
      PetscInt numBlocks  = 1;
      PetscInt batchSize  = numBlocks * blockSize;
      PetscInt numBatches = numBatchesTmp;
      PetscInt numChunks  = numCells / (numBatches*batchSize);
      ierr = (*mesh->integrateJacobianFEM)(numChunks*numBatches*batchSize, numFields, fieldI, fieldJ, quad, u, v0, J, invJ, detJ, g0, g1, g2, g3, elemMat);CHKERRQ(ierr);
      /* Remainder */
      PetscInt numRemainder = numCells % (numBatches * batchSize);
      PetscInt offset       = numCells - numRemainder;
      ierr = (*mesh->integrateJacobianFEM)(numRemainder, numFields, fieldI, fieldJ, quad, &u[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                           g0, g1, g2, g3, &elemMat[offset*cellDof*cellDof]);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, "Jacobian", cellDof, cellDof, &elemMat[c*cellDof*cellDof]);CHKERRQ(ierr);}
    ierr = DMComplexMatSetClosure(dm, PETSC_NULL, PETSC_NULL, JacP, c, &elemMat[c*cellDof*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemMat);CHKERRQ(ierr);

  /* Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd(). */
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian:\n");CHKERRQ(ierr);
    ierr = MatChop(JacP, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(JacP, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* ierr = PetscLogEventEnd(JacobianFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr = PetscObjectTypeCompare((PetscObject)Jac, MATSHELL, &isShell);CHKERRQ(ierr);
  if (isShell) {
    JacActionCtx *jctx;

    ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
    ierr = VecCopy(X, jctx->u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
