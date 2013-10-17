#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc-private/isimpl.h>
#include <petscsf.h>

/* Logging support */
PetscLogEvent DMPLEX_Partition, DMPLEX_Distribute, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_Stratify, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM;

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

#undef __FUNCT__
#define __FUNCT__ "VecView_Plex_Local"
PetscErrorCode VecView_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (isvtk) {
    PetscViewerVTKFieldType ft = PETSC_VTK_POINT_FIELD;
    PetscSection            section;
    PetscInt                dim, pStart, pEnd, cStart, fStart, vStart, cdof = 0, fdof = 0, vdof = 0;

    ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    /* Assumes that numer of dofs per point of each stratum is constant, natural for VTK */
    if ((cStart >= pStart) && (cStart < pEnd)) {ierr = PetscSectionGetDof(section, cStart, &cdof);CHKERRQ(ierr);}
    if ((fStart >= pStart) && (fStart < pEnd)) {ierr = PetscSectionGetDof(section, fStart, &fdof);CHKERRQ(ierr);}
    if ((vStart >= pStart) && (vStart < pEnd)) {ierr = PetscSectionGetDof(section, vStart, &vdof);CHKERRQ(ierr);}
    if (cdof && fdof && vdof) { /* Actually Q2 or some such, but visualize as Q1 */
      ft = (cdof == dim) ? PETSC_VTK_POINT_VECTOR_FIELD : PETSC_VTK_POINT_FIELD;
    } else if (cdof && vdof) {
      SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"No support for viewing mixed space with dofs at both vertices and cells");
    } else if (cdof) {
      /* TODO: This assumption should be removed when there is a way of identifying whether a space is conceptually a
       * vector or just happens to have the same number of dofs as the dimension. */
      if (cdof == dim) {
        ft = PETSC_VTK_CELL_VECTOR_FIELD;
      } else {
        ft = PETSC_VTK_CELL_FIELD;
      }
    } else if (vdof) {
      if (vdof == dim) {
        ft = PETSC_VTK_POINT_VECTOR_FIELD;
      } else {
        ft = PETSC_VTK_POINT_FIELD;
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Could not classify input Vec for VTK");

    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr); /* viewer drops reference */
    ierr = PetscObjectReference((PetscObject) v);CHKERRQ(ierr);  /* viewer drops reference */
    ierr = PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, ft, (PetscObject) v);CHKERRQ(ierr);
  } else {
    PetscBool isseq;

    ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
    if (isseq) {
      ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_Plex"
PetscErrorCode VecView_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (isvtk) {
    Vec         locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  } else {
    PetscBool isseq;

    ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
    if (isseq) {
      ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexView_Ascii"
PetscErrorCode DMPlexView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Plex          *mesh = (DM_Plex*) dm->data;
  DM                cdm;
  DMLabel           markers;
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
    PetscMPIInt rank, size;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
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
    ierr = PetscSectionGetChart(coordSection, &pStart, NULL);CHKERRQ(ierr);
    if (pStart >= 0) {ierr = PetscSectionVecView(coordSection, coordinates, viewer);CHKERRQ(ierr);}
    ierr = DMPlexGetLabel(dm, "marker", &markers);CHKERRQ(ierr);
    ierr = DMLabelView(markers,viewer);CHKERRQ(ierr);
    if (size > 1) {
      PetscSF sf;

      ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
      ierr = PetscSFView(sf, viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_LATEX) {
    const char  *name;
    const char  *colors[3] = {"red", "blue", "green"};
    const int    numColors  = 3;
    PetscReal    scale      = 2.0;
    PetscScalar *coords;
    PetscInt     depth, cStart, cEnd, c, vStart, vEnd, v, eStart = 0, eEnd = 0, e, p;
    PetscMPIInt  rank, size;

    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\
\\documentclass[crop,multi=false]{standalone}\n\n\
\\usepackage{tikz}\n\
\\usepackage{pgflibraryshapes}\n\
\\usetikzlibrary{backgrounds}\n\
\\usetikzlibrary{arrows}\n\
\\begin{document}\n\
\\section{%s}\n\
\\begin{center}\n", name, 8.0/scale);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh for process ");CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      if (p > 0 && p == size-1) {
        ierr = PetscViewerASCIIPrintf(viewer, ", and ", colors[p%numColors], p);CHKERRQ(ierr);
      } else if (p > 0) {
        ierr = PetscViewerASCIIPrintf(viewer, ", ", colors[p%numColors], p);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "{\\textcolor{%s}%D}", colors[p%numColors], p);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, ".\n\n\n\
\\begin{tikzpicture}[scale = %g,font=\\fontsize{8}{8}\\selectfont]\n");CHKERRQ(ierr);
    /* Plot vertices */
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off, dof, d;

      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%G", scale*PetscRealPart(coords[off+d]));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%D) [draw,shape=circle,color=%s] {%D} --\n", v, rank, colors[rank%numColors], v);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "(0,0);\n");CHKERRQ(ierr);
    /* Plot edges */
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
    if (depth > 1) {ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);}
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt *cone;
      PetscInt        coneSize, offA, offB, dof, d;

      ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      if (coneSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Edge %d cone should have two vertices, not %d", e, coneSize);
      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(coordSection, cone[0], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, cone[0], &offA);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, cone[1], &offB);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%G", scale*0.5*PetscRealPart(coords[offA+d]+coords[offB+d]));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%D) [draw,shape=circle,color=%s] {%D} --\n", e, rank, colors[rank%numColors], e);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "(0,0);\n");CHKERRQ(ierr);
    /* Plot cells */
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, firstPoint = -1;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n\\end{center}\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{document}\n", name);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm;
    PetscInt   *sizes, *hybsizes;
    PetscInt    locDepth, depth, dim, d, pMax[4];
    PetscInt    pStart, pEnd, p;
    PetscInt    numLabels, l;
    const char *name;
    PetscMPIInt size;

    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    if (name) {ierr = PetscViewerASCIIPrintf(viewer, "%s in %D dimensions:\n", name, dim);CHKERRQ(ierr);}
    else      {ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %D dimensions:\n", dim);CHKERRQ(ierr);}
    ierr = DMPlexGetDepth(dm, &locDepth);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(dm, &pMax[depth], &pMax[depth-1], &pMax[1], &pMax[0]);CHKERRQ(ierr);
    ierr = PetscMalloc2(size,PetscInt,&sizes,size,PetscInt,&hybsizes);CHKERRQ(ierr);
    if (depth == 1) {
      ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", 0);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", dim);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for (d = 0; d <= dim; d++) {
        ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd    -= pStart;
        pMax[d] -= pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = MPI_Gather(&pMax[d], 1, MPIU_INT, hybsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", d);CHKERRQ(ierr);
        for (p = 0; p < size; ++p) {
          if (hybsizes[p] >= 0) {ierr = PetscViewerASCIIPrintf(viewer, " %D (%D)", sizes[p], sizes[p] - hybsizes[p]);CHKERRQ(ierr);}
          else                  {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(sizes,hybsizes);CHKERRQ(ierr);
    ierr = DMPlexGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
    if (numLabels) {ierr = PetscViewerASCIIPrintf(viewer, "Labels:\n");CHKERRQ(ierr);}
    for (l = 0; l < numLabels; ++l) {
      DMLabel         label;
      const char     *name;
      IS              valueIS;
      const PetscInt *values;
      PetscInt        numValues, v;

      ierr = DMPlexGetLabelName(dm, l, &name);CHKERRQ(ierr);
      ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
      ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %s: %d strata of sizes (", name, numValues);CHKERRQ(ierr);
      ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
      ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
      for (v = 0; v < numValues; ++v) {
        PetscInt size;

        ierr = DMLabelGetStratumSize(label, values[v], &size);CHKERRQ(ierr);
        if (v > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "%d", size);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, ")\n");CHKERRQ(ierr);
      ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
      ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Plex"
PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, isbinary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMPlexView_Ascii(dm, viewer);CHKERRQ(ierr);
#if 0
  } else if (isbinary) {
    ierr = DMPlexView_Binary(dm, viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Plex"
PetscErrorCode DMDestroy_Plex(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  DMLabel        next  = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  ierr = PetscSectionDestroy(&mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->cones);CHKERRQ(ierr);
  ierr = PetscFree(mesh->coneOrientations);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscFree(mesh->facesTmp);CHKERRQ(ierr);
  while (next) {
    DMLabel tmp = next->next;

    ierr = DMLabelDestroy(&next);CHKERRQ(ierr);
    next = tmp;
  }
  ierr = DMLabelDestroy(&mesh->subpointMap);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalCellNumbers);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Plex"
PetscErrorCode DMCreateMatrix_Plex(DM dm, Mat *J)
{
  PetscSection   section, sectionGlobal;
  PetscInt       bs = -1;
  PetscInt       localSize;
  PetscBool      isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock;
  PetscErrorCode ierr;
  MatType        mtype;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  mtype = dm->mattype;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  /* ierr = PetscSectionGetStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr); */
  ierr = PetscSectionGetConstrainedStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm), J);CHKERRQ(ierr);
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
  if (!isShell) {
    PetscBool fillMatrix = (PetscBool) !dm->prealloc_only;
    PetscInt *dnz, *onz, *dnzu, *onzu, bsLocal, bsMax, bsMin;

    if (bs < 0) {
      if (isBlock || isSeqBlock || isMPIBlock || isSymBlock || isSymSeqBlock || isSymMPIBlock) {
        PetscInt pStart, pEnd, p, dof, cdof;

        ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          ierr = PetscSectionGetDof(sectionGlobal, p, &dof);CHKERRQ(ierr);
          ierr = PetscSectionGetConstraintDof(sectionGlobal, p, &cdof);CHKERRQ(ierr);
          if (dof-cdof) {
            if (bs < 0) {
              bs = dof-cdof;
            } else if (bs != dof-cdof) {
              /* Layout does not admit a pointwise block size */
              bs = 1;
              break;
            }
          }
        }
        /* Must have same blocksize on all procs (some might have no points) */
        bsLocal = bs;
        ierr = MPI_Allreduce(&bsLocal, &bsMax, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
        bsLocal = bs < 0 ? bsMax : bs;
        ierr = MPI_Allreduce(&bsLocal, &bsMin, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
        if (bsMin != bsMax) {
          bs = 1;
        } else {
          bs = bsMax;
        }
      } else {
        bs = 1;
      }
    }
    ierr = PetscMalloc4(localSize/bs, PetscInt, &dnz, localSize/bs, PetscInt, &onz, localSize/bs, PetscInt, &dnzu, localSize/bs, PetscInt, &onzu);CHKERRQ(ierr);
    ierr = PetscMemzero(dnz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(onz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(dnzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(onzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = DMPlexPreallocateOperator(dm, bs, section, sectionGlobal, dnz, onz, dnzu, onzu, *J, fillMatrix);CHKERRQ(ierr);
    ierr = PetscFree4(dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetDimension"
/*@
  DMPlexGetDimension - Return the topological mesh dimension

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:
. dim - The topological mesh dimension

  Level: beginner

.seealso: DMPlexCreate()
@*/
PetscErrorCode DMPlexGetDimension(DM dm, PetscInt *dim)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = mesh->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetDimension"
/*@
  DMPlexSetDimension - Set the topological mesh dimension

  Collective on mesh

  Input Parameters:
+ mesh - The DMPlex
- dim - The topological mesh dimension

  Level: beginner

.seealso: DMPlexCreate()
@*/
PetscErrorCode DMPlexSetDimension(DM dm, PetscInt dim)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  mesh->dim               = dim;
  mesh->preallocCenterDim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetChart"
/*@
  DMPlexGetChart - Return the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameters:
+ pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetChart(DM dm, PetscInt *pStart, PetscInt *pEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetChart"
/*@
  DMPlexSetChart - Set the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Output Parameters:

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetChart()
@*/
PetscErrorCode DMPlexSetChart(DM dm, PetscInt pStart, PetscInt pEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(mesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetConeSize"
/*@
  DMPlexGetConeSize - Return the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. size - The cone size for point p

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetConeSize"
/*@
  DMPlexSetConeSize - Set the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- size - The cone size for point p

  Output Parameter:

  Note:
  This should be called after DMPlexSetChart().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexSetConeSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->coneSection, p, size);CHKERRQ(ierr);

  mesh->maxConeSize = PetscMax(mesh->maxConeSize, size);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetCone"
/*@C
  DMPlexGetCone - Return the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. cone - An array of points which are on the in-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  You must also call DMPlexRestoreCone() after you finish using the returned array.

.seealso: DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetCone(DM dm, PetscInt p, const PetscInt *cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  ierr  = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  *cone = &mesh->cones[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetCone"
/*@
  DMPlexSetCone - Set the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- cone - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetCone(DM dm, PetscInt p, const PetscInt cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(cone, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    if ((cone[c] < pStart) || (cone[c] >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", cone[c], pStart, pEnd);
    mesh->cones[off+c] = cone[c];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetConeOrientation"
/*@C
  DMPlexGetConeOrientation - Return the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  You must also call DMPlexRestoreConeOrientation() after you finish using the returned array.

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetCone(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeOrientation(DM dm, PetscInt p, const PetscInt *coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt dof;
    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    if (dof) PetscValidPointer(coneOrientation, 3);
  }
#endif
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);

  *coneOrientation = &mesh->coneOrientations[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetConeOrientation"
/*@
  DMPlexSetConeOrientation - Set the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetConeOrientation(), DMPlexSetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetConeOrientation(DM dm, PetscInt p, const PetscInt coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(coneOrientation, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    PetscInt cdof, o = coneOrientation[c];

    ierr = PetscSectionGetDof(mesh->coneSection, mesh->cones[off+c], &cdof);CHKERRQ(ierr);
    if (o && ((o < -(cdof+1)) || (o >= cdof))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone orientation %D is not in the valid range [%D. %D)", o, -(cdof+1), cdof);
    mesh->coneOrientations[off+c] = o;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertCone"
PetscErrorCode DMPlexInsertCone(DM dm, PetscInt p, PetscInt conePos, PetscInt conePoint)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((conePoint < pStart) || (conePoint >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", conePoint, pStart, pEnd);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((conePos < 0) || (conePos >= dof)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
  mesh->cones[off+conePos] = conePoint;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertConeOrientation"
PetscErrorCode DMPlexInsertConeOrientation(DM dm, PetscInt p, PetscInt conePos, PetscInt coneOrientation)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((conePos < 0) || (conePos >= dof)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
  mesh->coneOrientations[off+conePos] = coneOrientation;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetSupportSize"
/*@
  DMPlexGetSupportSize - Return the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. size - The support size for point p

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart(), DMPlexGetConeSize()
@*/
PetscErrorCode DMPlexGetSupportSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->supportSection, p, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetSupportSize"
/*@
  DMPlexSetSupportSize - Set the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- size - The support size for point p

  Output Parameter:

  Note:
  This should be called after DMPlexSetChart().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetSupportSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexSetSupportSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->supportSection, p, size);CHKERRQ(ierr);

  mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, size);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetSupport"
/*@C
  DMPlexGetSupport - Return the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. support - An array of points which are on the out-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  You must also call DMPlexRestoreSupport() after you finish using the returned array.

.seealso: DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetSupport(DM dm, PetscInt p, const PetscInt *support[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  ierr     = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  *support = &mesh->supports[off];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetSupport"
/*@
  DMPlexSetSupport - Set the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- support - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetSupportSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetSupport(), DMPlexSetChart(), DMPlexSetSupportSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetSupport(DM dm, PetscInt p, const PetscInt support[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(support, 3);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    if ((support[c] < pStart) || (support[c] >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", support[c], pStart, pEnd);
    mesh->supports[off+c] = support[c];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertSupport"
PetscErrorCode DMPlexInsertSupport(DM dm, PetscInt p, PetscInt supportPos, PetscInt supportPoint)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((supportPoint < pStart) || (supportPoint >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", supportPoint, pStart, pEnd);
  if (supportPos >= dof) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support position %D of point %D is not in the valid range [0, %D)", supportPos, p, dof);
  mesh->supports[off+supportPos] = supportPoint;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetTransitiveClosure"
/*@C
  DMPlexGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
- points - If points is NULL on input, internal storage will be returned, otherwise the provided array is used

  Output Parameters:
+ numPoints - The number of points in the closure, so points[] is of size 2*numPoints
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...]

  Note:
  If using internal storage (points is NULL on input), each call overwrites the last output.

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexRestoreTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PetscInt       *closure, *fifo;
  const PetscInt *tmp = NULL, *tmpO = NULL;
  PetscInt        tmpSize, t;
  PetscInt        depth       = 0, maxSize;
  PetscInt        closureSize = 2, fifoSize = 0, fifoStart = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* This is only 1-level */
  if (useCone) {
    ierr = DMPlexGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &tmp);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &tmpO);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &tmp);CHKERRQ(ierr);
  }
  if (depth == 1) {
    if (*points) {
      closure = *points;
    } else {
      maxSize = 2*(PetscMax(mesh->maxConeSize, mesh->maxSupportSize)+1);
      ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
    }
    closure[0] = p; closure[1] = 0;
    for (t = 0; t < tmpSize; ++t, closureSize += 2) {
      closure[closureSize]   = tmp[t];
      closure[closureSize+1] = tmpO ? tmpO[t] : 0;
    }
    if (numPoints) *numPoints = closureSize/2;
    if (points)    *points    = closure;
    PetscFunctionReturn(0);
  }
  maxSize = 2*PetscMax(PetscMax(PetscPowInt(mesh->maxConeSize,depth+1),PetscPowInt(mesh->maxSupportSize,depth+1)),depth+1);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
  }
  closure[0] = p; closure[1] = 0;
  for (t = 0; t < tmpSize; ++t, closureSize += 2, fifoSize += 2) {
    const PetscInt cp = tmp[t];
    const PetscInt co = tmpO ? tmpO[t] : 0;

    closure[closureSize]   = cp;
    closure[closureSize+1] = co;
    fifo[fifoSize]         = cp;
    fifo[fifoSize+1]       = co;
  }
  /* Should kick out early when depth is reached, rather than checking all vertices for empty cones */
  while (fifoSize - fifoStart) {
    const PetscInt q   = fifo[fifoStart];
    const PetscInt o   = fifo[fifoStart+1];
    const PetscInt rev = o >= 0 ? 0 : 1;
    const PetscInt off = rev ? -(o+1) : o;

    if (useCone) {
      ierr = DMPlexGetConeSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, q, &tmp);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, q, &tmpO);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetSupportSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, q, &tmp);CHKERRQ(ierr);
      tmpO = NULL;
    }
    for (t = 0; t < tmpSize; ++t) {
      const PetscInt i  = ((rev ? tmpSize-t : t) + off)%tmpSize;
      const PetscInt cp = tmp[i];
      /* Must propogate orientation: When we reverse orientation, we both reverse the direction of iteration and start at the other end of the chain. */
      /* HACK: It is worse to get the size here, than to change the interpretation of -(*+1)
       const PetscInt co = tmpO ? (rev ? -(tmpO[i]+1) : tmpO[i]) : 0; */
      PetscInt       co = tmpO ? tmpO[i] : 0;
      PetscInt       c;

      if (rev) {
        PetscInt childSize, coff;
        ierr = DMPlexGetConeSize(dm, cp, &childSize);CHKERRQ(ierr);
        coff = tmpO[i] < 0 ? -(tmpO[i]+1) : tmpO[i];
        co   = childSize ? -(((coff+childSize-1)%childSize)+1) : 0;
      }
      /* Check for duplicate */
      for (c = 0; c < closureSize; c += 2) {
        if (closure[c] == cp) break;
      }
      if (c == closureSize) {
        closure[closureSize]   = cp;
        closure[closureSize+1] = co;
        fifo[fifoSize]         = cp;
        fifo[fifoSize+1]       = co;
        closureSize           += 2;
        fifoSize              += 2;
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
#define __FUNCT__ "DMPlexRestoreTransitiveClosure"
/*@C
  DMPlexRestoreTransitiveClosure - Restore the array of points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
. numPoints - The number of points in the closure, so points[] is of size 2*numPoints, zeroed on exit
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...], zeroed on exit

  Note:
  If not using internal storage (points is not NULL on input), this call is unnecessary

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexGetTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexRestoreTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numPoints) PetscValidIntPointer(numPoints,4);
  if (points) PetscValidPointer(points,5);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, points);CHKERRQ(ierr);
  if (numPoints) *numPoints = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetMaxSizes"
/*@
  DMPlexGetMaxSizes - Return the maximum number of in-edges (cone) and out-edges (support) for any point in the Sieve DAG

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameters:
+ maxConeSize - The maximum number of in-edges
- maxSupportSize - The maximum number of out-edges

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetMaxSizes(DM dm, PetscInt *maxConeSize, PetscInt *maxSupportSize)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (maxConeSize)    *maxConeSize    = mesh->maxConeSize;
  if (maxSupportSize) *maxSupportSize = mesh->maxSupportSize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Plex"
PetscErrorCode DMSetUp_Plex(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
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
#define __FUNCT__ "DMCreateSubDM_Plex"
PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMClone(dm, subdm);CHKERRQ(ierr);}
  ierr = DMCreateSubDM_Section_Private(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSymmetrize"
/*@
  DMPlexSymmetrize - Creates support (out-edge) information from cone (in-edge) inoformation

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetCone()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetChart(), DMPlexSetConeSize(), DMPlexSetCone()
@*/
PetscErrorCode DMPlexSymmetrize(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets;
  PetscInt       supportSize;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->supports) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Supports were already setup in this DMPlex");
  /* Calculate support sizes */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexStratify"
/*@
  DMPlexStratify - The Sieve DAG for most topologies is a graded poset (http://en.wikipedia.org/wiki/Graded_poset), and
  can be illustrated by Hasse Diagram (a http://en.wikipedia.org/wiki/Hasse_diagram). The strata group all points of the
  same grade, and this function calculates the strata. This grade can be seen as the height (or depth) of the point in
  the DAG.

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:

  Notes:
  The normal association for the point grade is element dimension (or co-dimension). For instance, all vertices would
  have depth 0, and all edges depth 1. Likewise, all cells heights would have height 0, and all faces height 1.

  This should be called after all calls to DMPlexSymmetrize()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSymmetrize()
@*/
PetscErrorCode DMPlexStratify(DM dm)
{
  DMLabel        label;
  PetscInt       pStart, pEnd, p;
  PetscInt       numRoots = 0, numLeaves = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);
  /* Calculate depth */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dm, "depth");CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  /* Initialize roots and count leaves */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt coneSize, supportSize;

    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    if (!coneSize && supportSize) {
      ++numRoots;
      ierr = DMLabelSetValue(label, p, 0);CHKERRQ(ierr);
    } else if (!supportSize && coneSize) {
      ++numLeaves;
    } else if (!supportSize && !coneSize) {
      /* Isolated points */
      ierr = DMLabelSetValue(label, p, 0);CHKERRQ(ierr);
    }
  }
  if (numRoots + numLeaves == (pEnd - pStart)) {
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize, supportSize;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        ierr = DMLabelSetValue(label, p, 1);CHKERRQ(ierr);
      }
    }
  } else {
    IS       pointIS;
    PetscInt numPoints = 0, level = 0;

    ierr = DMLabelGetStratumIS(label, level, &pointIS);CHKERRQ(ierr);
    if (pointIS) {ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);}
    while (numPoints) {
      const PetscInt *points;
      const PetscInt  newLevel = level+1;

      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        const PetscInt  point = points[p];
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMLabelSetValue(label, support[s], newLevel);CHKERRQ(ierr);
        }
      }
      ++level;
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, level, &pointIS);CHKERRQ(ierr);
      if (pointIS) {ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);}
      else         {numPoints = 0;}
    }
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetJoin"
/*@C
  DMPlexGetJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Level: intermediate

  Note: Currently, this is restricted to a single level join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.keywords: mesh
.seealso: DMPlexRestoreJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexGetJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
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
    i        = 1-i;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
  ierr              = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRestoreJoin"
/*@C
  DMPlexRestoreJoin - Restore an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.keywords: mesh
.seealso: DMPlexGetJoin(), DMPlexGetFullJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexRestoreJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints, 5);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void*) coveredPoints);CHKERRQ(ierr);
  if (numCoveredPoints) *numCoveredPoints = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFullJoin"
/*@C
  DMPlexGetFullJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.keywords: mesh
.seealso: DMPlexGetJoin(), DMPlexRestoreJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexGetFullJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *join[2];
  PetscInt       depth = 0, maxSize, joinSize = 0, i = 0;
  PetscInt       p, d, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr    = PetscMalloc(numPoints * sizeof(PetscInt*), &closures);CHKERRQ(ierr);
  ierr    = PetscMemzero(closures,numPoints*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  maxSize = PetscPowInt(mesh->maxSupportSize,depth+1);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closures[p]);CHKERRQ(ierr);

    offsets[p*(depth+2)+0] = 0;
    for (d = 0; d < depth+1; ++d) {
      PetscInt pStart, pEnd, i;

      ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(depth+2)+d]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(depth+2)+d+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(depth+2)+d+1] = i;
    }
    if (offsets[p*(depth+2)+depth+1] != closureSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(depth+2)+depth+1], closureSize);
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
      i        = 1-i;
    }
    if (joinSize) break;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
  for (p = 0; p < numPoints; ++p) {
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetMeet"
/*@C
  DMPlexGetMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Note: Currently, this is restricted to a single level meet

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.keywords: mesh
.seealso: DMPlexRestoreMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexGetMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveringPoints, const PetscInt **coveringPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
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
    i        = 1-i;
  }
  *numCoveringPoints = meetSize;
  *coveringPoints    = meet[i];
  ierr               = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRestoreMeet"
/*@C
  DMPlexRestoreMeet - Restore an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.keywords: mesh
.seealso: DMPlexGetMeet(), DMPlexGetFullMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexRestoreMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints,5);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void*) coveredPoints);CHKERRQ(ierr);
  if (numCoveredPoints) *numCoveredPoints = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFullMeet"
/*@C
  DMPlexGetFullMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.keywords: mesh
.seealso: DMPlexGetMeet(), DMPlexRestoreMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexGetFullMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *meet[2];
  PetscInt       height = 0, maxSize, meetSize = 0, i = 0;
  PetscInt       p, h, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr    = DMPlexGetDepth(dm, &height);CHKERRQ(ierr);
  ierr    = PetscMalloc(numPoints * sizeof(PetscInt*), &closures);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  maxSize = PetscPowInt(mesh->maxConeSize,height+1);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closures[p]);CHKERRQ(ierr);

    offsets[p*(height+2)+0] = 0;
    for (h = 0; h < height+1; ++h) {
      PetscInt pStart, pEnd, i;

      ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(height+2)+h]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(height+2)+h+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(height+2)+h+1] = i;
    }
    if (offsets[p*(height+2)+height+1] != closureSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(height+2)+height+1], closureSize);
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
      i        = 1-i;
    }
    if (meetSize) break;
  }
  *numCoveredPoints = meetSize;
  *coveredPoints    = meet[i];
  for (p = 0; p < numPoints; ++p) {
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexEqual"
/*@C
  DMPlexEqual - Determine if two DMs have the same topology

  Not Collective

  Input Parameters:
+ dmA - A DMPlex object
- dmB - A DMPlex object

  Output Parameters:
. equal - PETSC_TRUE if the topologies are identical

  Level: intermediate

  Notes:
  We are not solving graph isomorphism, so we do not permutation.

.keywords: mesh
.seealso: DMPlexGetCone()
@*/
PetscErrorCode DMPlexEqual(DM dmA, DM dmB, PetscBool *equal)
{
  PetscInt       depth, depthB, pStart, pEnd, pStartB, pEndB, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidPointer(equal, 3);

  *equal = PETSC_FALSE;
  ierr = DMPlexGetDepth(dmA, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dmB, &depthB);CHKERRQ(ierr);
  if (depth != depthB) PetscFunctionReturn(0);
  ierr = DMPlexGetChart(dmA, &pStart,  &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmB, &pStartB, &pEndB);CHKERRQ(ierr);
  if ((pStart != pStartB) || (pEnd != pEndB)) PetscFunctionReturn(0);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *coneB, *ornt, *orntB, *support, *supportB;
    PetscInt        coneSize, coneSizeB, c, supportSize, supportSizeB, s;

    ierr = DMPlexGetConeSize(dmA, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmA, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmA, p, &ornt);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dmB, p, &coneSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmB, p, &coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmB, p, &orntB);CHKERRQ(ierr);
    if (coneSize != coneSizeB) PetscFunctionReturn(0);
    for (c = 0; c < coneSize; ++c) {
      if (cone[c] != coneB[c]) PetscFunctionReturn(0);
      if (ornt[c] != orntB[c]) PetscFunctionReturn(0);
    }
    ierr = DMPlexGetSupportSize(dmA, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmA, p, &support);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmB, p, &supportSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmB, p, &supportB);CHKERRQ(ierr);
    if (supportSize != supportSizeB) PetscFunctionReturn(0);
    for (s = 0; s < supportSize; ++s) {
      if (support[s] != supportB[s]) PetscFunctionReturn(0);
    }
  }
  *equal = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetNumFaceVertices"
PetscErrorCode DMPlexGetNumFaceVertices(DM dm, PetscInt cellDim, PetscInt numCorners, PetscInt *numFaceVertices)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  PetscValidPointer(numFaceVertices,3);
  switch (cellDim) {
  case 0:
    *numFaceVertices = 0;
    break;
  case 1:
    *numFaceVertices = 1;
    break;
  case 2:
    switch (numCorners) {
    case 3: /* triangle */
      *numFaceVertices = 2; /* Edge has 2 vertices */
      break;
    case 4: /* quadrilateral */
      *numFaceVertices = 2; /* Edge has 2 vertices */
      break;
    case 6: /* quadratic triangle, tri and quad cohesive Lagrange cells */
      *numFaceVertices = 3; /* Edge has 3 vertices */
      break;
    case 9: /* quadratic quadrilateral, quadratic quad cohesive Lagrange cells */
      *numFaceVertices = 3; /* Edge has 3 vertices */
      break;
    default:
      SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %d for dimension %d", numCorners, cellDim);
    }
    break;
  case 3:
    switch (numCorners) {
    case 4: /* tetradehdron */
      *numFaceVertices = 3; /* Face has 3 vertices */
      break;
    case 6: /* tet cohesive cells */
      *numFaceVertices = 4; /* Face has 4 vertices */
      break;
    case 8: /* hexahedron */
      *numFaceVertices = 4; /* Face has 4 vertices */
      break;
    case 9: /* tet cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 10: /* quadratic tetrahedron */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 12: /* hex cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 18: /* quadratic tet cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 27: /* quadratic hexahedron, quadratic hex cohesive Lagrange cells */
      *numFaceVertices = 9; /* Face has 9 vertices */
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
#define __FUNCT__ "DMPlexOrient"
/* Trys to give the mesh a consistent orientation */
PetscErrorCode DMPlexOrient(DM dm)
{
  PetscBT        seenCells, flippedCells, seenFaces;
  PetscInt      *faceFIFO, fTop, fBottom;
  PetscInt       dim, h, cStart, cEnd, c, fStart, fEnd, face, maxConeSize, *revcone, *revconeO;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Truth Table
     mismatch    flips   do action   mismatch   flipA ^ flipB   action
         F       0 flips     no         F             F           F
         F       1 flip      yes        F             T           T
         F       2 flips     no         T             F           T
         T       0 flips     yes        T             T           F
         T       1 flip      no
         T       2 flips     yes
  */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &h);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h+1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &seenCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, seenCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &flippedCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, flippedCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(fEnd - fStart, &seenFaces);CHKERRQ(ierr);
  ierr = PetscBTMemzero(fEnd - fStart, seenFaces);CHKERRQ(ierr);
  ierr = PetscMalloc((fEnd - fStart) * sizeof(PetscInt), &faceFIFO);CHKERRQ(ierr);
  fTop = fBottom = 0;
  /* Initialize FIFO with first cell */
  if (cEnd > cStart) {
    const PetscInt *cone;
    PetscInt        coneSize;

    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cStart, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      faceFIFO[fBottom++] = cone[c];
      ierr = PetscBTSet(seenFaces, cone[c]-fStart);CHKERRQ(ierr);
    }
  }
  /* Consider each face in FIFO */
  while (fTop < fBottom) {
    const PetscInt *support, *coneA, *coneB, *coneOA, *coneOB;
    PetscInt        supportSize, coneSizeA, coneSizeB, posA = -1, posB = -1;
    PetscInt        seenA, flippedA, seenB, flippedB, mismatch;

    face = faceFIFO[fTop++];
    ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
    if (supportSize < 2) continue;
    if (supportSize != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Faces should separate only two cells, not %d", supportSize);
    seenA    = PetscBTLookup(seenCells,    support[0]-cStart);
    flippedA = PetscBTLookup(flippedCells, support[0]-cStart) ? 1 : 0;
    seenB    = PetscBTLookup(seenCells,    support[1]-cStart);
    flippedB = PetscBTLookup(flippedCells, support[1]-cStart) ? 1 : 0;

    ierr = DMPlexGetConeSize(dm, support[0], &coneSizeA);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, support[1], &coneSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[0], &coneA);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[1], &coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, support[0], &coneOA);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, support[1], &coneOB);CHKERRQ(ierr);
    for (c = 0; c < coneSizeA; ++c) {
      if (!PetscBTLookup(seenFaces, coneA[c]-fStart)) {
        faceFIFO[fBottom++] = coneA[c];
        ierr = PetscBTSet(seenFaces, coneA[c]-fStart);CHKERRQ(ierr);
      }
      if (coneA[c] == face) posA = c;
      if (fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], fBottom, fEnd-fStart);
    }
    if (posA < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[0]);
    for (c = 0; c < coneSizeB; ++c) {
      if (!PetscBTLookup(seenFaces, coneB[c]-fStart)) {
        faceFIFO[fBottom++] = coneB[c];
        ierr = PetscBTSet(seenFaces, coneB[c]-fStart);CHKERRQ(ierr);
      }
      if (coneB[c] == face) posB = c;
      if (fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], fBottom, fEnd-fStart);
    }
    if (posB < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[1]);

    if (dim == 1) {
      mismatch = posA == posB;
    } else {
      mismatch = coneOA[posA] == coneOB[posB];
    }

    if (mismatch ^ (flippedA ^ flippedB)) {
      if (seenA && seenB) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen cells %d and %d do not match: Fault mesh is non-orientable", support[0], support[1]);
      if (!seenA && !flippedA) {
        ierr = PetscBTSet(flippedCells, support[0]-cStart);CHKERRQ(ierr);
      } else if (!seenB && !flippedB) {
        ierr = PetscBTSet(flippedCells, support[1]-cStart);CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
    } else if (mismatch && flippedA && flippedB) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
    ierr = PetscBTSet(seenCells, support[0]-cStart);CHKERRQ(ierr);
    ierr = PetscBTSet(seenCells, support[1]-cStart);CHKERRQ(ierr);
  }

  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, PETSC_INT, &revcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, PETSC_INT, &revconeO);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt *cone, *coneO, *support;
    PetscInt        coneSize, supportSize, faceSize, cp, sp;

    if (!PetscBTLookup(flippedCells, c-cStart)) continue;
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, c, &coneO);CHKERRQ(ierr);
    for (cp = 0; cp < coneSize; ++cp) {
      const PetscInt rcp = coneSize-cp-1;

      ierr = DMPlexGetConeSize(dm, cone[rcp], &faceSize);CHKERRQ(ierr);
      revcone[cp]  = cone[rcp];
      revconeO[cp] = coneO[rcp] >= 0 ? -(faceSize-coneO[rcp]) : faceSize+coneO[rcp];
    }
    ierr = DMPlexSetCone(dm, c, revcone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, c, revconeO);CHKERRQ(ierr);
    /* Reverse orientations of support */
    faceSize = coneSize;
    ierr = DMPlexGetSupportSize(dm, c, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, c, &support);CHKERRQ(ierr);
    for (sp = 0; sp < supportSize; ++sp) {
      ierr = DMPlexGetConeSize(dm, support[sp], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[sp], &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, support[sp], &coneO);CHKERRQ(ierr);
      for (cp = 0; cp < coneSize; ++cp) {
        if (cone[cp] != c) continue;
        ierr = DMPlexInsertConeOrientation(dm, support[sp], cp, coneO[cp] >= 0 ? -(faceSize-coneO[cp]) : faceSize+coneO[cp]);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, maxConeSize, PETSC_INT, &revcone);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, maxConeSize, PETSC_INT, &revconeO);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&seenCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&flippedCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&seenFaces);CHKERRQ(ierr);
  ierr = PetscFree(faceFIFO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencySingleLevel_Internal"
static PetscErrorCode DMPlexGetAdjacencySingleLevel_Internal(DM dm, PetscInt p, PetscBool useClosure, const PetscInt *tmpClosure, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = NULL;
  PetscInt        numAdj   = 0, maxAdjSize = *adjSize, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (useClosure) {
    ierr = DMPlexGetConeSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone = NULL;
      PetscInt        coneSize, c, q;

      ierr = DMPlexGetSupportSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        for (q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
          if (cone[c] == adj[q]) break;
        }
        if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
      }
    }
  } else {
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone = NULL;
      PetscInt        coneSize, c, q;

      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexCreateNeighborCSR"
PetscErrorCode DMPlexCreateNeighborCSR(DM dm, PetscInt cellHeight, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency)
{
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[30]; /* maxFaceCases, C89 sucks sucks sucks */
  PetscInt      *off, *adj;
  PetscInt      *neighborCells, *tmpClosure;
  PetscInt       maxConeSize, maxSupportSize, maxClosure, maxNeighbors;
  PetscInt       dim, cellDim, depth = 0, faceDepth, cStart, cEnd, c, numCells, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  cellDim = dim - cellHeight;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (cEnd - cStart == 0) {
    if (numVertices) *numVertices = 0;
    if (offsets)   *offsets   = NULL;
    if (adjacency) *adjacency = NULL;
    PetscFunctionReturn(0);
  }
  numCells  = cEnd - cStart;
  faceDepth = depth - cellHeight;
  /* Setup face recognition */
  if (faceDepth == 1) {
    PetscInt cornersSeen[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* Could use PetscBT */

    for (c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      ierr = DMPlexGetConeSize(dm, c, &corners);CHKERRQ(ierr);
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        if (numFaceCases >= maxFaceCases) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;

        ierr = DMPlexGetNumFaceVertices(dm, cellDim, corners, &nFV);CHKERRQ(ierr);

        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  maxClosure   = 2*PetscMax(PetscPowInt(maxConeSize,depth+1),PetscPowInt(maxSupportSize,depth+1));
  maxNeighbors = PetscPowInt(maxConeSize,depth+1)*PetscPowInt(maxSupportSize,depth+1);
  ierr         = PetscMalloc2(maxNeighbors,PetscInt,&neighborCells,maxClosure,PetscInt,&tmpClosure);CHKERRQ(ierr);
  ierr         = PetscMalloc((numCells+1) * sizeof(PetscInt), &off);CHKERRQ(ierr);
  ierr         = PetscMemzero(off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = maxNeighbors, n;

    ierr = DMPlexGetAdjacencySingleLevel_Internal(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2];
      PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize = 0;
      const PetscInt *meet    = NULL;

      cellPair[0] = cell; cellPair[1] = neighborCells[n];
      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
      }
      if (found) ++off[cell-cStart+1];
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) off[cell] += off[cell-1];

  if (adjacency) {
    ierr = PetscMalloc(off[numCells] * sizeof(PetscInt), &adj);CHKERRQ(ierr);
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = maxNeighbors, n;
      PetscInt cellOffset   = 0;

      ierr = DMPlexGetAdjacencySingleLevel_Internal(dm, cell, PETSC_TRUE, tmpClosure, &numNeighbors, neighborCells);CHKERRQ(ierr);
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2];
        PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize = 0;
        const PetscInt *meet    = NULL;

        cellPair[0] = cell; cellPair[1] = neighborCells[n];
        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
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
  if (offsets)   *offsets   = off;
  if (adjacency) *adjacency = adj;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHACO)
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
/* Chaco does not have an include file */
PETSC_EXTERN int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

extern int FREE_GRAPH;

#undef __FUNCT__
#define __FUNCT__ "DMPlexPartition_Chaco"
PetscErrorCode DMPlexPartition_Chaco(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
  MPI_Comm       comm;
  int            nvtxs          = numVertices; /* number of vertices in full graph */
  int           *vwgts          = NULL;   /* weights for all vertices */
  float         *ewgts          = NULL;   /* weights for all edges */
  float         *x              = NULL, *y = NULL, *z = NULL; /* coordinates for inertial method */
  char          *outassignname  = NULL;   /*  name of assignment output file */
  char          *outfilename    = NULL;   /* output file name */
  int            architecture   = 1;      /* 0 => hypercube, d => d-dimensional mesh */
  int            ndims_tot      = 0;      /* total number of cube dimensions to divide */
  int            mesh_dims[3];            /* dimensions of mesh of processors */
  double        *goal          = NULL;    /* desired set sizes for each set */
  int            global_method = 1;       /* global partitioning algorithm */
  int            local_method  = 1;       /* local partitioning algorithm */
  int            rqi_flag      = 0;       /* should I use RQI/Symmlq eigensolver? */
  int            vmax          = 200;     /* how many vertices to coarsen down to? */
  int            ndims         = 1;       /* number of eigenvectors (2^d sets) */
  double         eigtol        = 0.001;   /* tolerance on eigenvectors */
  long           seed          = 123636512; /* for random graph mutations */
  short int     *assignment;              /* Output partition */
  int            fd_stdout, fd_pipe[2];
  PetscInt      *points;
  PetscMPIInt    commSize;
  int            i, v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commSize);CHKERRQ(ierr);
  if (!numVertices) {
    ierr = PetscSectionCreate(comm, partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, commSize);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
  for (i = 0; i < start[numVertices]; ++i) ++adjacency[i];

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
#if defined(PETSC_HAVE_UNISTD_H)
  {
    int piperet;
    piperet = pipe(fd_pipe);
    if (piperet) SETERRQ(comm,PETSC_ERR_SYS,"Could not create pipe");
    fd_stdout = dup(1);
    close(1);
    dup2(fd_pipe[1], 1);
  }
#endif
  ierr = interface(nvtxs, (int*) start, (int*) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                   assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                   vmax, ndims, eigtol, seed);
#if defined(PETSC_HAVE_UNISTD_H)
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
  for (i = 0; i < start[numVertices]; ++i) --adjacency[i];
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_PARMETIS)
#include <parmetis.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexPartition_ParMetis"
PetscErrorCode DMPlexPartition_ParMetis(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  MPI_Comm       comm;
  PetscInt       nvtxs      = numVertices; // The number of vertices in full graph
  PetscInt      *vtxdist;                  // Distribution of vertices across processes
  PetscInt      *xadj       = start;       // Start of edge list for each vertex
  PetscInt      *adjncy     = adjacency;   // Edge lists for all vertices
  PetscInt      *vwgt       = NULL;        // Vertex weights
  PetscInt      *adjwgt     = NULL;        // Edge weights
  PetscInt       wgtflag    = 0;           // Indicates which weights are present
  PetscInt       numflag    = 0;           // Indicates initial offset (0 or 1)
  PetscInt       ncon       = 1;           // The number of weights per vertex
  PetscInt       nparts;                   // The number of partitions
  PetscReal     *tpwgts;                   // The fraction of vertex weights assigned to each partition
  PetscReal     *ubvec;                    // The balance intolerance for vertex weights
  PetscInt       options[5];               // Options
  // Outputs
  PetscInt       edgeCut;                  // The number of edges cut by the partition
  PetscInt      *assignment, *points;
  PetscMPIInt    commSize, rank, p, v, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commSize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  nparts = commSize;
  options[0] = 0; /* Use all defaults */
  /* Calculate vertex distribution */
  ierr = PetscMalloc4(nparts+1,PetscInt,&vtxdist,nparts*ncon,PetscReal,&tpwgts,ncon,PetscReal,&ubvec,nvtxs,PetscInt,&assignment);CHKERRQ(ierr);
  vtxdist[0] = 0;
  ierr = MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm);CHKERRQ(ierr);
  for (p = 2; p <= nparts; ++p) {
    vtxdist[p] += vtxdist[p-1];
  }
  /* Calculate weights */
  for (p = 0; p < nparts; ++p) {
    tpwgts[p] = 1.0/nparts;
  }
  ubvec[0] = 1.05;

  if (nparts == 1) {
    ierr = PetscMemzero(assignment, nvtxs * sizeof(PetscInt));
  } else {
    if (vtxdist[1] == vtxdist[nparts]) {
      if (!rank) {
        PetscStackPush("METIS_PartGraphKway");
        ierr = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, NULL, &edgeCut, assignment);
        PetscStackPop;
        if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      }
    } else {
      PetscStackPush("ParMETIS_V3_PartKway");
      ierr = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
      PetscStackPop;
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_PartKway()");
    }
  }
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
  ierr = PetscFree4(vtxdist,tpwgts,ubvec,assignment);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexEnlargePartition"
/* Expand the partition by BFS on the adjacency graph */
PetscErrorCode DMPlexEnlargePartition(DM dm, const PetscInt start[], const PetscInt adjacency[], PetscSection origPartSection, IS origPartition, PetscSection *partSection, IS *partition)
{
  PetscHashI      h;
  const PetscInt *points;
  PetscInt      **tmpPoints, *newPoints, totPoints = 0;
  PetscInt        pStart, pEnd, part, q;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscHashICreate(h);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), partSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(origPartSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*partSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(origPartition, &points);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt*), &tmpPoints);CHKERRQ(ierr);
  for (part = pStart; part < pEnd; ++part) {
    PetscInt numPoints, nP, numNewPoints, off, p, n = 0;

    PetscHashIClear(h);
    ierr = PetscSectionGetDof(origPartSection, part, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(origPartSection, part, &off);CHKERRQ(ierr);
    /* Add all existing points to h */
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[off+p];
      PetscHashIAdd(h, point, 1);
    }
    PetscHashISize(h, nP);
    if (nP != numPoints) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid partition has %d points, but only %d were unique", numPoints, nP);
    /* Add all points in next BFS level */
    /*   TODO We are brute forcing here, but could check the adjacency size to find the boundary */
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[off+p];
      PetscInt       s     = start[point], e = start[point+1], a;

      for (a = s; a < e; ++a) PetscHashIAdd(h, adjacency[a], 1);
    }
    PetscHashISize(h, numNewPoints);
    ierr = PetscSectionSetDof(*partSection, part, numNewPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numNewPoints * sizeof(PetscInt), &tmpPoints[part]);CHKERRQ(ierr);
    if (numNewPoints) PetscHashIGetKeys(h, n, tmpPoints[part]); /* Should not need this conditional */
    totPoints += numNewPoints;
  }
  ierr = ISRestoreIndices(origPartition, &points);CHKERRQ(ierr);
  PetscHashIDestroy(h);
  ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
  ierr = PetscMalloc(totPoints * sizeof(PetscInt), &newPoints);CHKERRQ(ierr);
  for (part = pStart, q = 0; part < pEnd; ++part) {
    PetscInt numPoints, p;

    ierr = PetscSectionGetDof(*partSection, part, &numPoints);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p, ++q) newPoints[q] = tmpPoints[part][p];
    ierr = PetscFree(tmpPoints[part]);CHKERRQ(ierr);
  }
  ierr = PetscFree(tmpPoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), totPoints, newPoints, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartition"
/*
  DMPlexCreatePartition - Create a non-overlapping partition of the points at the given height

  Collective on DM

  Input Parameters:
  + dm - The DM
  . height - The height for points in the partition
  - enlarge - Expand each partition with neighbors

  Output Parameters:
  + partSection - The PetscSection giving the division of points by partition
  . partition - The list of points by partition
  . origPartSection - If enlarge is true, the PetscSection giving the division of points before enlarging by partition, otherwise NULL
  - origPartition - If enlarge is true, the list of points before enlarging by partition, otherwise NULL

  Level: developer

.seealso DMPlexDistribute()
*/
PetscErrorCode DMPlexCreatePartition(DM dm, const char name[], PetscInt height, PetscBool enlarge, PetscSection *partSection, IS *partition, PetscSection *origPartSection, IS *origPartition)
{
  char           partname[1024];
  PetscBool      isChaco = PETSC_FALSE, isMetis = PETSC_FALSE, flg;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);

  *origPartSection = NULL;
  *origPartition   = NULL;
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*partSection, 0, size);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*partSection, 0, cEnd-cStart);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(*partSection);CHKERRQ(ierr);
    ierr = PetscMalloc((cEnd - cStart) * sizeof(PetscInt), &points);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), cEnd-cStart, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscOptionsGetString(((PetscObject) dm)->prefix, "-dm_plex_partitioner", partname, 1024, &flg);CHKERRQ(ierr);
  if (flg) name = partname;
  if (name) {
    ierr = PetscStrcmp(name, "chaco", &isChaco);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "metis", &isMetis);CHKERRQ(ierr);
  }
  if (height == 0) {
    PetscInt  numVertices;
    PetscInt *start     = NULL;
    PetscInt *adjacency = NULL;

    ierr = DMPlexCreateNeighborCSR(dm, 0, &numVertices, &start, &adjacency);CHKERRQ(ierr);
    if (!name || isChaco) {
#if defined(PETSC_HAVE_CHACO)
      ierr = DMPlexPartition_Chaco(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-chaco.");
#endif
    } else if (isMetis) {
#if defined(PETSC_HAVE_PARMETIS)
      ierr = DMPlexPartition_ParMetis(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Unknown mesh partitioning package %s", name);
    if (enlarge) {
      *origPartSection = *partSection;
      *origPartition   = *partition;

      ierr = DMPlexEnlargePartition(dm, start, adjacency, *origPartSection, *origPartition, partSection, partition);CHKERRQ(ierr);
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
  } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid partition height %D", height);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePartitionClosure"
PetscErrorCode DMPlexCreatePartitionClosure(DM dm, PetscSection pointSection, IS pointPartition, PetscSection *section, IS *partition)
{
  /* const PetscInt  height = 0; */
  const PetscInt *partArray;
  PetscInt       *allPoints, *packPoints;
  PetscInt        rStart, rEnd, rank, pStart, pEnd, newSize;
  PetscErrorCode  ierr;
  PetscBT         bt;
  PetscSegBuffer  segpack,segpart;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(pointSection, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, rStart, rEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscBTCreate(pEnd-pStart,&bt);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&segpack);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&segpart);CHKERRQ(ierr);
  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt partSize = 0, numPoints, offset, p, *PETSC_RESTRICT placePoints;

    ierr = PetscSectionGetDof(pointSection, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointSection, rank, &offset);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt  point   = partArray[offset+p], closureSize, c;
      PetscInt *closure = NULL;

      /* TODO Include support for height > 0 case */
      ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c=0; c<closureSize; c++) {
        PetscInt cpoint = closure[c*2];
        if (!PetscBTLookupSet(bt,cpoint-pStart)) {
          PetscInt *PETSC_RESTRICT pt;
          partSize++;
          ierr = PetscSegBufferGetInts(segpart,1,&pt);CHKERRQ(ierr);
          *pt = cpoint;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(*section, rank, partSize);CHKERRQ(ierr);
    ierr = PetscSegBufferGetInts(segpack,partSize,&placePoints);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractTo(segpart,placePoints);CHKERRQ(ierr);
    ierr = PetscSortInt(partSize,placePoints);CHKERRQ(ierr);
    for (p=0; p<partSize; p++) {ierr = PetscBTClear(bt,placePoints[p]-pStart);CHKERRQ(ierr);}
  }
  ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&segpart);CHKERRQ(ierr);

  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(*section, &newSize);CHKERRQ(ierr);
  ierr = PetscMalloc(newSize * sizeof(PetscInt), &allPoints);CHKERRQ(ierr);

  ierr = PetscSegBufferExtractInPlace(segpack,&packPoints);CHKERRQ(ierr);
  for (rank = rStart; rank < rEnd; ++rank) {
    PetscInt numPoints, offset;

    ierr = PetscSectionGetDof(*section, rank, &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(*section, rank, &offset);CHKERRQ(ierr);
    ierr = PetscMemcpy(&allPoints[offset], packPoints, numPoints * sizeof(PetscInt));CHKERRQ(ierr);
    packPoints += numPoints;
  }

  ierr = PetscSegBufferDestroy(&segpack);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pointPartition, &partArray);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), newSize, allPoints, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeField"
/*@
  DMPlexDistributeField - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
- originalVec - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newVec - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeData()
@*/
PetscErrorCode DMPlexDistributeField(DM dm, PetscSF pointSF, PetscSection originalSection, Vec originalVec, PetscSection newSection, Vec newVec)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscScalar   *originalValues, *newValues;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newVec, fieldSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newVec,dm->vectype);CHKERRQ(ierr);

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
#define __FUNCT__ "DMPlexDistributeData"
/*@
  DMPlexDistributeData - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
. datatype - The type of data
- originalData - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newData - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeField()
@*/
PetscErrorCode DMPlexDistributeData(DM dm, PetscSF pointSF, PetscSection originalSection, MPI_Datatype datatype, void *originalData, PetscSection newSection, void **newData)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscMPIInt    dataSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = MPI_Type_size(datatype, &dataSize);CHKERRQ(ierr);
  ierr = PetscMalloc(fieldSize * dataSize, newData);CHKERRQ(ierr);

  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistribute"
/*@C
  DMPlexDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ dm  - The original DMPlex object
. partitioner - The partitioning package, or NULL for the default
- overlap - The overlap of partitions, 0 is the default

  Output Parameter:
+ sf - The PetscSF used for point distribution
- parallelMesh - The distributed DMPlex object, or NULL

  Note: If the mesh was not distributed, the return value is NULL

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMPlexDistributeByFace()
@*/
PetscErrorCode DMPlexDistribute(DM dm, const char partitioner[], PetscInt overlap, PetscSF *sf, DM *dmParallel)
{
  DM_Plex               *mesh   = (DM_Plex*) dm->data, *pmesh;
  MPI_Comm               comm;
  const PetscInt         height = 0;
  PetscInt               dim, numRemoteRanks;
  IS                     origCellPart,        cellPart,        part;
  PetscSection           origCellPartSection, cellPartSection, partSection;
  PetscSFNode           *remoteRanks;
  PetscSF                partSF, pointSF, coneSF;
  ISLocalToGlobalMapping renumbering;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets;
  PetscInt              *cones, *newCones, newConesSize;
  PetscBool              flg;
  PetscMPIInt            rank, numProcs, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidPointer(sf,4);
  PetscValidPointer(dmParallel,5);

  ierr = PetscLogEventBegin(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);

  *dmParallel = NULL;
  if (numProcs == 1) PetscFunctionReturn(0);

  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  if (overlap > 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Overlap > 1 not yet implemented");
  ierr = DMPlexCreatePartition(dm, partitioner, height, overlap > 0 ? PETSC_TRUE : PETSC_FALSE, &cellPartSection, &cellPart, &origCellPartSection, &origCellPart);CHKERRQ(ierr);
  /* Create SF assuming a serial partition for all processes: Could check for IS length here */
  if (!rank) numRemoteRanks = numProcs;
  else       numRemoteRanks = 0;
  ierr = PetscMalloc(numRemoteRanks * sizeof(PetscSFNode), &remoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, NULL);CHKERRQ(ierr);
    if (origCellPart) {
      ierr = PetscPrintf(comm, "Original Cell Partition:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(origCellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = ISView(origCellPart, NULL);CHKERRQ(ierr);
    }
    ierr = PetscSFView(partSF, NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMPlexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr  = DMPlexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr  = DMPlexSetDimension(*dmParallel, dim);CHKERRQ(ierr);
  ierr  = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
  pmesh = (DM_Plex*) (*dmParallel)->data;
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, &pointSF);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(part, NULL);CHKERRQ(ierr);
    ierr = PetscSFView(pointSF, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(renumbering, NULL);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  /* Distribute cone section */
  ierr = DMPlexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(*dmParallel, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = DMSetUp(*dmParallel);CHKERRQ(ierr);
  {
    PetscInt pStart, pEnd, p;

    ierr = PetscSectionGetChart(newConeSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize;
      ierr               = PetscSectionGetDof(newConeSection, p, &coneSize);CHKERRQ(ierr);
      pmesh->maxConeSize = PetscMax(pmesh->maxConeSize, coneSize);
    }
  }
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(pointSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetCones(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFView(coneSF, NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientations(*dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  {
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(*dmParallel);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dmParallel);CHKERRQ(ierr);
  /* Distribute Coordinates */
  {
    PetscSection originalCoordSection, newCoordSection;
    Vec          originalCoordinates, newCoordinates;
    const char  *name;

    ierr = DMPlexGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
    ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

    ierr = DMPlexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dmParallel, newCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
  }
  /* Distribute labels */
  ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  {
    DMLabel  next      = mesh->labels, newNext = pmesh->labels;
    PetscInt numLabels = 0, l;

    /* Bcast number of labels */
    while (next) {
      ++numLabels; next = next->next;
    }
    ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    next = mesh->labels;
    for (l = 0; l < numLabels; ++l) {
      DMLabel         newLabel;
      const PetscInt *partArray;
      char           *name;
      PetscInt       *stratumSizes = NULL, *points = NULL;
      PetscMPIInt    *sendcnts     = NULL, *offsets = NULL, *displs = NULL;
      PetscInt        nameSize, s, p, proc;
      PetscBool       isdepth;
      size_t          len = 0;

      /* Bcast name (could filter for no points) */
      if (!rank) {ierr = PetscStrlen(next->name, &len);CHKERRQ(ierr);}
      nameSize = len;
      ierr     = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr     = PetscMalloc(nameSize+1, &name);CHKERRQ(ierr);
      if (!rank) {ierr = PetscMemcpy(name, next->name, nameSize+1);CHKERRQ(ierr);}
      ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
      ierr = PetscStrcmp(name, "depth", &isdepth);CHKERRQ(ierr);
      if (isdepth) {            /* skip because "depth" is not distributed */
        ierr = PetscFree(name);CHKERRQ(ierr);
        if (!rank) next = next->next;
        continue;
      }
      ierr           = PetscNew(struct _n_DMLabel, &newLabel);CHKERRQ(ierr);
      newLabel->name = name;
      /* Bcast numStrata (could filter for no points in stratum) */
      if (!rank) newLabel->numStrata = next->numStrata;
      ierr = MPI_Bcast(&newLabel->numStrata, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscMalloc3(newLabel->numStrata,PetscInt,&newLabel->stratumValues,
                          newLabel->numStrata,PetscInt,&newLabel->stratumSizes,
                          newLabel->numStrata+1,PetscInt,&newLabel->stratumOffsets);CHKERRQ(ierr);
      /* Bcast stratumValues (could filter for no points in stratum) */
      if (!rank) {ierr = PetscMemcpy(newLabel->stratumValues, next->stratumValues, next->numStrata * sizeof(PetscInt));CHKERRQ(ierr);}
      ierr = MPI_Bcast(newLabel->stratumValues, newLabel->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
      /* Find size on each process and Scatter
           we use the fact that both the stratum points and partArray are sorted */
      if (!rank) {
        ierr = ISGetIndices(part, &partArray);CHKERRQ(ierr);
        ierr = PetscMalloc(numProcs*next->numStrata * sizeof(PetscInt), &stratumSizes);CHKERRQ(ierr);
        ierr = PetscMemzero(stratumSizes, numProcs*next->numStrata * sizeof(PetscInt));CHKERRQ(ierr);
        /* TODO We should switch to using binary search if the label is a lot smaller than partitions */
        for (proc = 0; proc < numProcs; ++proc) {
          PetscInt dof, off;

          ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
          for (s = 0; s < next->numStrata; ++s) {
            PetscInt lStart = next->stratumOffsets[s], lEnd = next->stratumOffsets[s]+next->stratumSizes[s];
            PetscInt pStart = off,                     pEnd = off+dof;

            while (pStart < pEnd && lStart < lEnd) {
              if (partArray[pStart] > next->points[lStart]) {
                ++lStart;
              } else if (next->points[lStart] > partArray[pStart]) {
                ++pStart;
              } else {
                ++stratumSizes[proc*next->numStrata+s];
                ++pStart; ++lStart;
              }
            }
          }
        }
        ierr = ISRestoreIndices(part, &partArray);CHKERRQ(ierr);
      }
      ierr = MPI_Scatter(stratumSizes, newLabel->numStrata, MPIU_INT, newLabel->stratumSizes, newLabel->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
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
        /* TODO We should switch to using binary search if the label is a lot smaller than partitions */
        for (proc = 0; proc < numProcs; ++proc) {
          PetscInt dof, off;

          ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
          for (s = 0; s < next->numStrata; ++s) {
            PetscInt lStart = next->stratumOffsets[s], lEnd = next->stratumOffsets[s]+next->stratumSizes[s];
            PetscInt pStart = off,                     pEnd = off+dof;

            while (pStart < pEnd && lStart < lEnd) {
              if (partArray[pStart] > next->points[lStart]) {
                ++lStart;
              } else if (next->points[lStart] > partArray[pStart]) {
                ++pStart;
              } else {
                points[offsets[proc]++] = next->points[lStart];
                ++pStart; ++lStart;
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
      ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newLabel->stratumOffsets[newLabel->numStrata], newLabel->points, NULL, newLabel->points);CHKERRQ(ierr);
      /* Sort points */
      for (s = 0; s < newLabel->numStrata; ++s) {
        ierr = PetscSortInt(newLabel->stratumSizes[s], &newLabel->points[newLabel->stratumOffsets[s]]);CHKERRQ(ierr);
      }
      /* Insert into list */
      if (newNext) newNext->next = newLabel;
      else pmesh->labels = newLabel;
      newNext = newLabel;
      if (!rank) next = next->next;
    }
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  /* Setup hybrid structure */
  {
    const PetscInt *gpoints;
    PetscInt        depth, n, d;

    for (d = 0; d <= dim; ++d) {pmesh->hybridPointMax[d] = mesh->hybridPointMax[d];}
    ierr = MPI_Bcast(pmesh->hybridPointMax, dim+1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(renumbering, &n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(renumbering, &gpoints);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) {
      PetscInt pmax = pmesh->hybridPointMax[d], newmax = 0, pEnd, stratum[2], p;

      if (pmax < 0) continue;
      ierr = DMPlexGetDepthStratum(dm, d > depth ? depth : d, &stratum[0], &stratum[1]);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(*dmParallel, d, NULL, &pEnd);CHKERRQ(ierr);
      ierr = MPI_Bcast(stratum, 2, MPIU_INT, 0, comm);CHKERRQ(ierr);
      for (p = 0; p < n; ++p) {
        const PetscInt point = gpoints[p];

        if ((point >= stratum[0]) && (point < stratum[1]) && (point >= pmax)) ++newmax;
      }
      if (newmax > 0) pmesh->hybridPointMax[d] = pEnd - newmax;
      else            pmesh->hybridPointMax[d] = -1;
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(renumbering, &gpoints);CHKERRQ(ierr);
  }
  /* Cleanup Partition */
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Create point SF for parallel mesh */
  ierr = PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    ierr = DMPlexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots,PetscSFNode,&rowners,numLeaves,PetscSFNode,&lowners);CHKERRQ(ierr);
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    if (origCellPart) {
      /* Make sure cells in the original partition are not assigned to other procs */
      const PetscInt *origCells;

      ierr = ISGetIndices(origCellPart, &origCells);CHKERRQ(ierr);
      for (p = 0; p < numProcs; ++p) {
        PetscInt dof, off, d;

        ierr = PetscSectionGetDof(origCellPartSection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(origCellPartSection, p, &off);CHKERRQ(ierr);
        for (d = off; d < off+dof; ++d) {
          rowners[origCells[d]].rank = p;
        }
      }
      ierr = ISRestoreIndices(origCellPart, &origCells);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&origCellPart);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&origCellPartSection);CHKERRQ(ierr);

    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].rank == rank) { /* Either put in a bid or we know we own it */
        lowners[p].rank  = rank;
        lowners[p].index = leaves ? leaves[p] : p;
      } else if (lowners[p].rank >= 0) { /* Point already claimed so flag so that MAXLOC does not listen to us */
        lowners[p].rank  = -2;
        lowners[p].index = -2;
      }
    }
    for (p=0; p<numRoots; p++) { /* Root must not participate in the rediction, flag so that MAXLOC does not use */
      rowners[p].rank  = -3;
      rowners[p].index = -3;
    }
    ierr = PetscSFReduceBegin(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell partition corrupt: point not claimed");
      if (lowners[p].rank != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscInt),    &ghostPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numGhostPoints * sizeof(PetscSFNode), &remotePoints);CHKERRQ(ierr);
    for (p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p].rank != rank) {
        ghostPoints[gp]        = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p].rank;
        remotePoints[gp].index = lowners[p].index;
        ++gp;
      }
    }
    ierr = PetscFree2(rowners,lowners);CHKERRQ(ierr);
    ierr = PetscSFSetGraph((*dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions((*dmParallel)->sf);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  /* Cleanup */
  if (sf) {*sf = pointSF;}
  else    {ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);}
  ierr = DMSetFromOptions(*dmParallel);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInvertCell"
/*@C
  DMPlexInvertCell - This flips tetrahedron and hexahedron orientation since Plex stores them internally with outward normals. Other cells are left untouched.

  Input Parameters:
+ numCorners - The number of vertices in a cell
- cone - The incoming cone

  Output Parameter:
. cone - The inverted cone (in-place)

  Level: developer

.seealso: DMPlexGenerate()
@*/
PetscErrorCode DMPlexInvertCell(PetscInt dim, PetscInt numCorners, int cone[])
{
  int tmpc;

  PetscFunctionBegin;
  if (dim != 3) PetscFunctionReturn(0);
  switch (numCorners) {
  case 4:
    tmpc    = cone[0];
    cone[0] = cone[1];
    cone[1] = tmpc;
    break;
  case 8:
    tmpc    = cone[1];
    cone[1] = cone[3];
    cone[3] = tmpc;
    break;
  default: break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInvertCells_Internal"
/* This is to fix the tetrahedron orientation from TetGen */
PETSC_UNUSED static PetscErrorCode DMPlexInvertCells_Internal(PetscInt dim, PetscInt numCells, PetscInt numCorners, int cells[])
{
  PetscInt       bound = numCells*numCorners, coff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (coff = 0; coff < bound; coff += numCorners) {
    ierr = DMPlexInvertCell(dim, numCorners, &cells[coff]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_TRIANGLE)
#include <triangle.h>

#undef __FUNCT__
#define __FUNCT__ "InitInput_Triangle"
PetscErrorCode InitInput_Triangle(struct triangulateio *inputCtx)
{
  PetscFunctionBegin;
  inputCtx->numberofpoints             = 0;
  inputCtx->numberofpointattributes    = 0;
  inputCtx->pointlist                  = NULL;
  inputCtx->pointattributelist         = NULL;
  inputCtx->pointmarkerlist            = NULL;
  inputCtx->numberofsegments           = 0;
  inputCtx->segmentlist                = NULL;
  inputCtx->segmentmarkerlist          = NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->trianglelist               = NULL;
  inputCtx->numberofholes              = 0;
  inputCtx->holelist                   = NULL;
  inputCtx->numberofregions            = 0;
  inputCtx->regionlist                 = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitOutput_Triangle"
PetscErrorCode InitOutput_Triangle(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  outputCtx->numberofpoints        = 0;
  outputCtx->pointlist             = NULL;
  outputCtx->pointattributelist    = NULL;
  outputCtx->pointmarkerlist       = NULL;
  outputCtx->numberoftriangles     = 0;
  outputCtx->trianglelist          = NULL;
  outputCtx->triangleattributelist = NULL;
  outputCtx->neighborlist          = NULL;
  outputCtx->segmentlist           = NULL;
  outputCtx->segmentmarkerlist     = NULL;
  outputCtx->numberofedges         = 0;
  outputCtx->edgelist              = NULL;
  outputCtx->edgemarkerlist        = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FiniOutput_Triangle"
PetscErrorCode FiniOutput_Triangle(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  free(outputCtx->pointmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGenerate_Triangle"
PetscErrorCode DMPlexGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm             comm;
  PetscInt             dim              = 2;
  const PetscBool      createConvexHull = PETSC_FALSE;
  const PetscBool      constrained      = PETSC_FALSE;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             vStart, vEnd, v, eStart, eEnd, e;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMPlexGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &eStart, &eEnd);CHKERRQ(ierr);
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    ierr = PetscMalloc(in.numberofsegments*2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofsegments   * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;

      ierr = DMPlexGetCone(boundary, e, &cone);CHKERRQ(ierr);

      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;

      ierr = DMPlexGetLabelValue(boundary, "marker", e, &in.segmentmarkerlist[idx]);CHKERRQ(ierr);
    }
  }
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMPlexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
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
    triangulate(args, &in, &out, NULL);
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
    const int     *cells      = out.trianglelist;
    const double  *meshCoords = out.pointlist;

    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
#if 0 /* Do not currently support holes */
  ierr = DMPlexCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = FiniOutput_Triangle(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRefine_Triangle"
PetscErrorCode DMPlexRefine_Triangle(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm             comm;
  PetscInt             dim  = 2;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMPlexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;

  in.trianglearealist  = (double*) maxVolumes;
  if (in.numberoftriangles > 0) {
    ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 4) && (closureSize != 7)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a triangle, %D vertices in closure", closureSize);
      for (v = 0; v < 3; ++v) {
        in.trianglelist[idx*in.numberofcorners + v] = closure[(v+closureSize-3)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  /* TODO: Segment markers are missing on input */
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMPlexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
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
    triangulate(args, &in, &out, NULL);
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
    const int     *cells      = out.trianglelist;
    const double  *meshCoords = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
#if 0 /* Do not currently support holes */
  ierr = DMPlexCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = FiniOutput_Triangle(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_TETGEN)
#include <tetgen.h>
#undef __FUNCT__
#define __FUNCT__ "DMPlexGenerate_Tetgen"
PetscErrorCode DMPlexGenerate_Tetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm;
  const PetscInt dim  = 3;
  ::tetgenio     in;
  ::tetgenio     out;
  PetscInt       vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr              = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr              = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr              = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];

    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in.pointlist[idx*dim + d] = array[off+d];
      ierr = DMPlexGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);

  in.numberoffacets = fEnd - fStart;
  if (in.numberoffacets > 0) {
    in.facetlist       = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx     = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v;

      in.facetlist[idx].numberofpolygons = 1;
      in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
      in.facetlist[idx].numberofholes    = 0;
      in.facetlist[idx].holelist         = NULL;

      ierr = DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      tetgenio::polygon *poly = in.facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      poly->vertexlist       = new int[poly->numberofvertices];
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMPlexGetLabelValue(boundary, "marker", f, &in.facetmarkerlist[idx]);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
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
    const double   *meshCoords = out.pointlist;
    int            *cells      = out.tetrahedronlist;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dm, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMPlexSetLabelValue(*dm, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRefine_Tetgen"
PetscErrorCode DMPlexRefine_Tetgen(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm;
  const PetscInt dim  = 3;
  ::tetgenio     in;
  ::tetgenio     out;
  PetscInt       vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];

    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in.pointlist[idx*dim + d] = array[off+d];
      ierr = DMPlexGetLabelValue(dm, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in.numberofcorners       = 4;
  in.numberoftetrahedra    = cEnd - cStart;
  in.tetrahedronvolumelist = (double*) maxVolumes;
  if (in.numberoftetrahedra > 0) {
    in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in.tetrahedronlist[idx*in.numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  /* TODO: Put in boundary faces with markers */
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    /*ierr = PetscStrcpy(args, "qezQra");CHKERRQ(ierr); */
    ierr = PetscStrcpy(args, "qezraVVVV");CHKERRQ(ierr);
    ::tetrahedralize(args, &in, &out);
  }
  in.tetrahedronvolumelist = NULL;

  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    const double   *meshCoords = out.pointlist;
    int            *cells      = out.tetrahedronlist;

    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dmRefined, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dmRefined, "marker", edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMPlexSetLabelValue(*dmRefined, "marker", faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_CTETGEN)
#include "ctetgen.h"

#undef __FUNCT__
#define __FUNCT__ "DMPlexGenerate_CTetgen"
PetscErrorCode DMPlexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm;
  const PetscInt dim  = 3;
  PLC           *in, *out;
  PetscInt       verbose = 0, vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject) boundary)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
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
    ierr = DMPlexGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMPlexGetLabelValue(boundary, "marker", v, &m);CHKERRQ(ierr);

      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);

  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc(in->numberoffacets * sizeof(facet), &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberoffacets * sizeof(int),   &in->facetmarkerlist);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx     = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v, m;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;

      ierr = PetscMalloc(in->facetlist[idx].numberofpolygons * sizeof(polygon), &in->facetlist[idx].polygonlist);CHKERRQ(ierr);

      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = NULL;

      ierr = DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      poly                   = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr                   = PetscMalloc(poly->numberofvertices * sizeof(int), &poly->vertexlist);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr                     = DMPlexGetLabelValue(boundary, "marker", f, &m);CHKERRQ(ierr);
      in->facetmarkerlist[idx] = (int) m;
      ierr                     = DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr        = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose;
    ierr        = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr        = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    const double   *meshCoords = out->pointlist;
    int            *cells      = out->tetrahedronlist;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dm, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dm, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMPlexSetLabelValue(*dm, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRefine_CTetgen"
PetscErrorCode DMPlexRefine_CTetgen(DM dm, PetscReal *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm;
  const PetscInt dim  = 3;
  PLC           *in, *out;
  PetscInt       verbose = 0, vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject) dm)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
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
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      ierr = DMPlexGetLabelValue(dm, "marker", v, &m);CHKERRQ(ierr);

      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    ierr = PetscMalloc(in->numberoftetrahedra*in->numberofcorners * sizeof(int), &in->tetrahedronlist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
    const double   *meshCoords = out->pointlist;
    int            *cells      = out->tetrahedronlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMPlexSetLabelValue(*dmRefined, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMPlexSetLabelValue(*dmRefined, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMPlexSetLabelValue(*dmRefined, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexGenerate"
/*@C
  DMPlexGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The DMPlex boundary object
. name - The mesh generation package name
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The DMPlex object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMRefine()
@*/
PetscErrorCode DMPlexGenerate(DM boundary, const char name[], PetscBool interpolate, DM *mesh)
{
  PetscInt       dim;
  char           genname[1024];
  PetscBool      isTriangle = PETSC_FALSE, isTetgen = PETSC_FALSE, isCTetgen = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(boundary, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(boundary, interpolate, 2);
  ierr = DMPlexGetDimension(boundary, &dim);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) boundary)->prefix, "-dm_plex_generator", genname, 1024, &flg);CHKERRQ(ierr);
  if (flg) name = genname;
  if (name) {
    ierr = PetscStrcmp(name, "triangle", &isTriangle);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "tetgen",   &isTetgen);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "ctetgen",  &isCTetgen);CHKERRQ(ierr);
  }
  switch (dim) {
  case 1:
    if (!name || isTriangle) {
#if defined(PETSC_HAVE_TRIANGLE)
      ierr = DMPlexGenerate_Triangle(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Mesh generation needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Unknown 2D mesh generation package %s", name);
    break;
  case 2:
    if (!name || isCTetgen) {
#if defined(PETSC_HAVE_CTETGEN)
      ierr = DMPlexGenerate_CTetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --download-ctetgen.");
#endif
    } else if (isTetgen) {
#if defined(PETSC_HAVE_TETGEN)
      ierr = DMPlexGenerate_Tetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Tetgen needs external package support.\nPlease reconfigure with --with-c-language=cxx --download-tetgen.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Unknown 3D mesh generation package %s", name);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Mesh generation for a dimension %d boundary is not supported.", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefine_Plex"
PetscErrorCode DMRefine_Plex(DM dm, MPI_Comm comm, DM *dmRefined)
{
  PetscReal      refinementLimit;
  PetscInt       dim, cStart, cEnd;
  char           genname[1024], *name = NULL;
  PetscBool      isUniform, isTriangle = PETSC_FALSE, isTetgen = PETSC_FALSE, isCTetgen = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetRefinementUniform(dm, &isUniform);CHKERRQ(ierr);
  if (isUniform) {
    CellRefiner cellRefiner;

    ierr = DMPlexGetCellRefiner_Internal(dm, &cellRefiner);CHKERRQ(ierr);
    ierr = DMPlexRefineUniform_Internal(dm, cellRefiner, dmRefined);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetRefinementLimit(dm, &refinementLimit);CHKERRQ(ierr);
  if (refinementLimit == 0.0) PetscFunctionReturn(0);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) dm)->prefix, "-dm_plex_generator", genname, 1024, &flg);CHKERRQ(ierr);
  if (flg) name = genname;
  if (name) {
    ierr = PetscStrcmp(name, "triangle", &isTriangle);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "tetgen",   &isTetgen);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "ctetgen",  &isCTetgen);CHKERRQ(ierr);
  }
  switch (dim) {
  case 2:
    if (!name || isTriangle) {
#if defined(PETSC_HAVE_TRIANGLE)
      double  *maxVolumes;
      PetscInt c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
      ierr = DMPlexRefine_Triangle(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
      ierr = PetscFree(maxVolumes);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh refinement needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unknown 2D mesh generation package %s", name);
    break;
  case 3:
    if (!name || isCTetgen) {
#if defined(PETSC_HAVE_CTETGEN)
      PetscReal *maxVolumes;
      PetscInt   c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(PetscReal), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
      ierr = DMPlexRefine_CTetgen(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --download-ctetgen.");
#endif
    } else if (isTetgen) {
#if defined(PETSC_HAVE_TETGEN)
      double  *maxVolumes;
      PetscInt c;

      ierr = PetscMalloc((cEnd - cStart) * sizeof(double), &maxVolumes);CHKERRQ(ierr);
      for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
      ierr = DMPlexRefine_Tetgen(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Tetgen needs external package support.\nPlease reconfigure with --with-c-language=cxx --download-tetgen.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unknown 3D mesh generation package %s", name);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh refinement in dimension %d is not supported.", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetDepthLabel"
/*@
  DMPlexGetDepthLabel - Get the DMLabel recording the depth of each point

  Not Collective

  Input Parameter:
. dm    - The DMPlex object

  Output Parameter:
. depthLabel - The DMLabel recording point depth

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetDepth(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum()
@*/
PetscErrorCode DMPlexGetDepthLabel(DM dm, DMLabel *depthLabel)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depthLabel, 2);
  if (!mesh->depthLabel) {
    ierr = DMPlexGetLabel(dm, "depth", &mesh->depthLabel);CHKERRQ(ierr);
  }
  *depthLabel = mesh->depthLabel;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetDepth"
/*@
  DMPlexGetDepth - Get the depth of the DAG representing this mesh

  Not Collective

  Input Parameter:
. dm    - The DMPlex object

  Output Parameter:
. depth - The number of strata (breadth first levels) in the DAG

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetDepthLabel(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum()
@*/
PetscErrorCode DMPlexGetDepth(DM dm, PetscInt *depth)
{
  DMLabel        label;
  PetscInt       d = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depth, 2);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (label) {ierr = DMLabelGetNumValues(label, &d);CHKERRQ(ierr);}
  *depth = d-1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetDepthStratum"
/*@
  DMPlexGetDepthStratum - Get the bounds [start, end) for all points at a certain depth.

  Not Collective

  Input Parameters:
+ dm           - The DMPlex object
- stratumValue - The requested depth

  Output Parameters:
+ start - The first point at this depth
- end   - One beyond the last point at this depth

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetHeightStratum(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end)
{
  DMLabel        label;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (stratumValue < 0) {
    ierr = DMPlexGetChart(dm, start, end);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    PetscInt pStart, pEnd;

    if (start) *start = 0;
    if (end)   *end   = 0;
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart == pEnd) PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (!label) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);
  /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
  depth = stratumValue;
  if ((depth < 0) || (depth >= label->numStrata)) {
    if (start) *start = 0;
    if (end)   *end   = 0;
  } else {
    if (start) *start = label->points[label->stratumOffsets[depth]];
    if (end)   *end   = label->points[label->stratumOffsets[depth]+label->stratumSizes[depth]-1]+1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetHeightStratum"
/*@
  DMPlexGetHeightStratum - Get the bounds [start, end) for all points at a certain height.

  Not Collective

  Input Parameters:
+ dm           - The DMPlex object
- stratumValue - The requested height

  Output Parameters:
+ start - The first point at this height
- end   - One beyond the last point at this height

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetDepthStratum(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetHeightStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end)
{
  DMLabel        label;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (stratumValue < 0) {
    ierr = DMPlexGetChart(dm, start, end);CHKERRQ(ierr);
  } else {
    PetscInt pStart, pEnd;

    if (start) *start = 0;
    if (end)   *end   = 0;
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart == pEnd) PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (!label) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);
  /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
  depth = label->stratumValues[label->numStrata-1] - stratumValue;
  if ((depth < 0) || (depth >= label->numStrata)) {
    if (start) *start = 0;
    if (end)   *end   = 0;
  } else {
    if (start) *start = label->points[label->stratumOffsets[depth]];
    if (end)   *end   = label->points[label->stratumOffsets[depth]+label->stratumSizes[depth]-1]+1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSectionInitial"
/* Set the number of dof on each point and separate by fields */
PetscErrorCode DMPlexCreateSectionInitial(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscSection *section)
{
  PetscInt      *numDofTot;
  PetscInt       pStart = 0, pEnd = 0;
  PetscInt       p, d, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &numDofTot);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    numDofTot[d] = 0;
    for (f = 0; f < numFields; ++f) numDofTot[d] += numDof[f*(dim+1)+d];
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  if (numFields > 0) {
    ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
    if (numComp) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(*section, p, f, numDof[f*(dim+1)+d]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(*section, p, numDofTot[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(numDofTot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSectionBCDof"
/* Set the number of dof on each point and separate by fields
   If constDof is PETSC_DETERMINE, constrain every dof on the point
*/
PetscErrorCode DMPlexCreateSectionBCDof(DM dm, PetscInt numBC,const PetscInt bcField[],const IS bcPoints[], PetscInt constDof, PetscSection section)
{
  PetscInt       numFields;
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *idx;
    PetscInt        n, i;

    if (numFields) field = bcField[bc];
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt p        = idx[i];
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
#define __FUNCT__ "DMPlexCreateSectionBCIndicesAll"
/* Set the constrained indices on each point and separate by fields */
PetscErrorCode DMPlexCreateSectionBCIndicesAll(DM dm, PetscSection section)
{
  PetscInt      *maxConstraints;
  PetscInt       numFields, f, pStart = 0, pEnd = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc((numFields+1) * sizeof(PetscInt), &maxConstraints);CHKERRQ(ierr);
  for (f = 0; f <= numFields; ++f) maxConstraints[f] = 0;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof;

    if (numFields) {
      for (f = 0; f < numFields; ++f) {
        ierr              = PetscSectionGetFieldConstraintDof(section, p, f, &cdof);CHKERRQ(ierr);
        maxConstraints[f] = PetscMax(maxConstraints[f], cdof);
      }
    } else {
      ierr              = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
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
            for (d = 0; d < cfdof; ++d) indices[numConst+d] = d;
            ierr = PetscSectionSetFieldConstraintIndices(section, p, f, &indices[numConst]);CHKERRQ(ierr);
            for (d = 0; d < cfdof; ++d) indices[numConst+d] += foff;
            numConst += cfdof;
            foff     += fdof;
          }
          if (cdof != numConst) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of field constraints %D should be %D", numConst, cdof);
        } else {
          for (d = 0; d < cdof; ++d) indices[d] = d;
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
#define __FUNCT__ "DMPlexCreateSectionBCIndicesField"
/* Set the constrained field indices on each point */
PetscErrorCode DMPlexCreateSectionBCIndicesField(DM dm, PetscInt field, IS bcPoints, IS constraintIndices, PetscSection section)
{
  const PetscInt *points, *indices;
  PetscInt        numFields, maxDof, numPoints, p, numConstraints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if ((field < 0) || (field >= numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, numFields);

  ierr = ISGetLocalSize(bcPoints, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(bcPoints, &points);CHKERRQ(ierr);
  if (!constraintIndices) {
    PetscInt *idx, i;

    ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
    ierr = PetscMalloc(maxDof * sizeof(PetscInt), &idx);CHKERRQ(ierr);
    for (i = 0; i < maxDof; ++i) idx[i] = i;
    for (p = 0; p < numPoints; ++p) {
      ierr = PetscSectionSetFieldConstraintIndices(section, points[p], field, idx);CHKERRQ(ierr);
    }
    ierr = PetscFree(idx);CHKERRQ(ierr);
  } else {
    ierr = ISGetLocalSize(constraintIndices, &numConstraints);CHKERRQ(ierr);
    ierr = ISGetIndices(constraintIndices, &indices);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt fcdof;

      ierr = PetscSectionGetFieldConstraintDof(section, points[p], field, &fcdof);CHKERRQ(ierr);
      if (fcdof != numConstraints) SETERRQ4(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Section point %d field %d has %d constraints, but yo ugave %d indices", p, field, fcdof, numConstraints);
      ierr = PetscSectionSetFieldConstraintIndices(section, points[p], field, indices);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(constraintIndices, &indices);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(bcPoints, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSectionBCIndices"
/* Set the constrained indices on each point and separate by fields */
PetscErrorCode DMPlexCreateSectionBCIndices(DM dm, PetscSection section)
{
  PetscInt      *indices;
  PetscInt       numFields, maxDof, f, pStart = 0, pEnd = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = PetscMalloc(maxDof * sizeof(PetscInt), &indices);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (!numFields) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This function only works after users have set field constraint indices.");
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
        for (d = 0; d < fcdof; ++d) indices[numConst+d] = fcind[d]+foff;
        foff     += fdof;
        numConst += fcdof;
      }
      if (cdof != numConst) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_LIB, "Total number of field constraints %D should be %D", numConst, cdof);
      ierr = PetscSectionSetConstraintIndices(section, p, indices);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSection"
/*@C
  DMPlexCreateSection - Create a PetscSection based upon the dof layout specification provided.

  Not Collective

  Input Parameters:
+ dm        - The DMPlex object
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

  Fortran Notes:
  A Fortran 90 version is available as DMPlexCreateSectionF90()

.keywords: mesh, elements
.seealso: DMPlexCreate(), PetscSectionCreate()
@*/
PetscErrorCode DMPlexCreateSection(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscInt numBC,const PetscInt bcField[],const IS bcPoints[], PetscSection *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateSectionInitial(dm, dim, numFields, numComp, numDof, section);CHKERRQ(ierr);
  ierr = DMPlexCreateSectionBCDof(dm, numBC, bcField, bcPoints, PETSC_DETERMINE, *section);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  if (numBC) {ierr = DMPlexCreateSectionBCIndicesAll(dm, *section);CHKERRQ(ierr);}
  {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-section_view", &view);CHKERRQ(ierr);
    if (view) {ierr = PetscSectionView(*section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateCoordinateDM_Plex"
PetscErrorCode DMCreateCoordinateDM_Plex(DM dm, DM *cdm)
{
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone(dm, cdm);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*cdm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetCoordinateSection"
/*@
  DMPlexGetCoordinateSection - Retrieve the layout of coordinate values over the mesh.

  Not Collective

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. section - The PetscSection object

  Level: intermediate

.keywords: mesh, coordinates
.seealso: DMGetCoordinateDM(), DMPlexGetDefaultSection(), DMPlexSetDefaultSection()
@*/
PetscErrorCode DMPlexGetCoordinateSection(DM dm, PetscSection *section)
{
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetCoordinateSection"
/*@
  DMPlexSetCoordinateSection - Set the layout of coordinate values over the mesh.

  Not Collective

  Input Parameters:
+ dm      - The DMPlex object
- section - The PetscSection object

  Level: intermediate

.keywords: mesh, coordinates
.seealso: DMPlexGetCoordinateSection(), DMPlexGetDefaultSection(), DMPlexSetDefaultSection()
@*/
PetscErrorCode DMPlexSetCoordinateSection(DM dm, PetscSection section)
{
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,2);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(cdm, section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetConeSection"
PetscErrorCode DMPlexGetConeSection(DM dm, PetscSection *section)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) *section = mesh->coneSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetSupportSection"
PetscErrorCode DMPlexGetSupportSection(DM dm, PetscSection *section)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) *section = mesh->supportSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetCones"
PetscErrorCode DMPlexGetCones(DM dm, PetscInt *cones[])
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cones) *cones = mesh->cones;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetConeOrientations"
PetscErrorCode DMPlexGetConeOrientations(DM dm, PetscInt *coneOrientations[])
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (coneOrientations) *coneOrientations = mesh->coneOrientations;
  PetscFunctionReturn(0);
}

/******************************** FEM Support **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMPlexVecGetClosure"
/*@C
  DMPlexVecGetClosure - Get an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
- point - The sieve point in the DM

  Output Parameters:
+ csize - The number of values in the closure, or NULL
- values - The array of values, which is a borrowed array and should not be freed

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The csize argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso DMPlexVecRestoreClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscSection   clSection;
  IS             clIndices;
  PetscScalar   *array, *vArray;
  PetscInt      *points = NULL;
  PetscInt       offsets[32];
  PetscInt       depth, numFields, size = 0, numPoints, pStart, pEnd, p, q, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  ierr = PetscSectionGetClosureIndex(section, (PetscObject) dm, &clSection, &clIndices);CHKERRQ(ierr);
  if (clSection) {
    const PetscInt *idx;
    PetscInt        dof, off;

    ierr = PetscSectionGetDof(clSection, point, &dof);CHKERRQ(ierr);
    if (csize) *csize = dof;
    if (values) {
      if (!*values) {
        ierr = DMGetWorkArray(dm, dof, PETSC_SCALAR, &array);CHKERRQ(ierr);
        *values = array;
      } else {
        array = *values;
      }
      ierr = PetscSectionGetOffset(clSection, point, &off);CHKERRQ(ierr);
      ierr = ISGetIndices(clIndices, &idx);CHKERRQ(ierr);
      ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
      for (p = 0; p < dof; ++p) array[p] = vArray[idx[off+p]];
      ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
      ierr = ISRestoreIndices(clIndices, &idx);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  if (depth == 1 && numFields < 2) {
    const PetscInt *cone, *coneO;

    ierr = DMPlexGetConeSize(dm, point, &numPoints);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, point, &coneO);CHKERRQ(ierr);
    if (!values || !*values) {
      if ((point >= pStart) && (point < pEnd)) {
        PetscInt dof;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        size += dof;
      }
      for (p = 0; p < numPoints; ++p) {
        const PetscInt cp = cone[p];
        PetscInt       dof;

        if ((cp < pStart) || (cp >= pEnd)) continue;
        ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
        size += dof;
      }
      if (!values) {
        if (csize) *csize = size;
        PetscFunctionReturn(0);
      }
      ierr = DMGetWorkArray(dm, size, PETSC_SCALAR, &array);CHKERRQ(ierr);
    } else {
      array = *values;
    }
    size = 0;
    ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
    if ((point >= pStart) && (point < pEnd)) {
      PetscInt     dof, off, d;
      PetscScalar *varr;
      ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
      varr = &vArray[off];
      for (d = 0; d < dof; ++d, ++offsets[0]) {
        array[offsets[0]] = varr[d];
      }
      size += dof;
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt cp = cone[p];
      PetscInt       o  = coneO[p];
      PetscInt       dof, off, d;
      PetscScalar   *varr;

      if ((cp < pStart) || (cp >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, cp, &off);CHKERRQ(ierr);
      varr = &vArray[off];
      if (o >= 0) {
        for (d = 0; d < dof; ++d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      } else {
        for (d = dof-1; d >= 0; --d, ++offsets[0]) {
          array[offsets[0]] = varr[d];
        }
      }
      size += dof;
    }
    ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
    if (!*values) {
      if (csize) *csize = size;
      *values = array;
    } else {
      if (size > *csize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %d < actual size %d", *csize, size);
      *csize = size;
    }
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
  for (p = 0, q = 0; p < numPoints*2; p += 2) {
    if ((points[p] >= pStart) && (points[p] < pEnd)) {
      points[q*2]   = points[p];
      points[q*2+1] = points[p+1];
      ++q;
    }
  }
  numPoints = q;
  if (!values || !*values) {
    for (p = 0, size = 0; p < numPoints*2; p += 2) {
      PetscInt dof, fdof;

      ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        ierr          = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
        offsets[f+1] += fdof;
      }
      size += dof;
    }
    if (!values) {
      ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      if (csize) *csize = size;
      PetscFunctionReturn(0);
    }
    ierr = DMGetWorkArray(dm, size, PETSC_SCALAR, &array);CHKERRQ(ierr);
  } else {
    array = *values;
  }
  for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];
  if (numFields && offsets[numFields] != size) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], size);
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
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  if (!*values) {
    if (csize) *csize = size;
    *values = array;
  } else {
    if (size > *csize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %d < actual size %d", *csize, size);
    *csize = size;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexVecRestoreClosure"
/*@C
  DMPlexVecRestoreClosure - Restore the array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
. point - The sieve point in the DM
. csize - The number of values in the closure, or NULL
- values - The array of values, which is a borrowed array and should not be freed

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The csize argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecRestoreClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscInt       size = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Should work without recalculating size */
  ierr = DMRestoreWorkArray(dm, size, PETSC_SCALAR, (void*) values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void add   (PetscScalar *x, PetscScalar y) {*x += y;}
PETSC_STATIC_INLINE void insert(PetscScalar *x, PetscScalar y) {*x  = y;}

#undef __FUNCT__
#define __FUNCT__ "updatePoint_private"
PetscErrorCode updatePoint_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
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
#define __FUNCT__ "updatePointBC_private"
PetscErrorCode updatePointBC_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
  a    = &array[off];
  if (cdof) {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          fuse(&a[k], values[k]);
          ++cind;
        }
      }
    } else {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          fuse(&a[k], values[dof-k-1]);
          ++cind;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updatePointFields_private"
PetscErrorCode updatePointFields_private(PetscSection section, PetscInt point, PetscInt foffs[], void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
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
#define __FUNCT__ "updatePointFieldsBC_private"
PetscErrorCode updatePointFieldsBC_private(PetscSection section, PetscInt point, PetscInt foffs[], void (*fuse)(PetscScalar*, PetscScalar), PetscInt orientation, const PetscScalar values[], PetscScalar array[])
{
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
    if (fcdof) {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (orientation >= 0) {
        for (k = 0; k < fdof; ++k) {
          if ((cind < fcdof) && (k == fcdofs[cind])) {
            fuse(&a[foff+k], values[foffs[f]+k]);
            ++cind;
          }
        }
      } else {
        for (k = fdof/fcomp-1; k >= 0; --k) {
          for (c = 0; c < fcomp; ++c) {
            if ((cind < fcdof) && (k*fcomp+c == fcdofs[cind])) {
              fuse(&a[foff+(fdof/fcomp-1-k)*fcomp+c], values[foffs[f]+k*fcomp+c]);
              ++cind;
            }
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
#define __FUNCT__ "DMPlexVecSetClosure"
/*@C
  DMPlexVecSetClosure - Set an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
. point - The sieve point in the DM
. values - The array of values
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

  Fortran Notes:
  This routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecSetClosure(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscScalar   *array;
  PetscInt      *points = NULL;
  PetscInt       offsets[32];
  PetscInt       depth, numFields, numPoints, off, dof, pStart, pEnd, p, q, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (!section) {
    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  if (depth == 1 && numFields < 2 && mode == ADD_VALUES) {
    const PetscInt *cone, *coneO;

    ierr = DMPlexGetConeSize(dm, point, &numPoints);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, point, &coneO);CHKERRQ(ierr);
    ierr = VecGetArray(v, &array);CHKERRQ(ierr);
    for (p = 0, off = 0; p <= numPoints; ++p, off += dof) {
      const PetscInt cp = !p ? point : cone[p-1];
      const PetscInt o  = !p ? 0     : coneO[p-1];

      if ((cp < pStart) || (cp >= pEnd)) {dof = 0; continue;}
      ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
      /* ADD_VALUES */
      {
        const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
        PetscScalar    *a;
        PetscInt        cdof, coff, cind = 0, k;

        ierr = PetscSectionGetConstraintDof(section, cp, &cdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(section, cp, &coff);CHKERRQ(ierr);
        a    = &array[coff];
        if (!cdof) {
          if (o >= 0) {
            for (k = 0; k < dof; ++k) {
              a[k] += values[off+k];
            }
          } else {
            for (k = 0; k < dof; ++k) {
              a[k] += values[off+dof-k-1];
            }
          }
        } else {
          ierr = PetscSectionGetConstraintIndices(section, cp, &cdofs);CHKERRQ(ierr);
          if (o >= 0) {
            for (k = 0; k < dof; ++k) {
              if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
              a[k] += values[off+k];
            }
          } else {
            for (k = 0; k < dof; ++k) {
              if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
              a[k] += values[off+dof-k-1];
            }
          }
        }
      }
    }
    ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  /* Compress out points not in the section */
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
      ierr          = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
  }
  for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  if (numFields) {
    switch (mode) {
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
    case INSERT_BC_VALUES:
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt o = points[p+1];
        updatePointFieldsBC_private(section, points[p], offsets, insert,  o, values, array);
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
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
    }
  } else {
    switch (mode) {
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
    case INSERT_BC_VALUES:
      for (p = 0, off = 0; p < numPoints*2; p += 2, off += dof) {
        PetscInt o = points[p+1];
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        updatePointBC_private(section, points[p], dof, insert,  o, &values[off], array);
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
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %D", mode);
    }
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateClosureIndex"
/*@
  DMPlexCreateClosureIndex - Calculate an index for the given PetscSection for the closure operation on the DM

  Not collective

  Input Parameters:
+ dm - The DM
- section - The section describing the layout in v, or NULL to use the default section

  Note:
  This should greatly improve the performance of the closure operations, at the cost of additional memory.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexVecRestoreClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexCreateClosureIndex(DM dm, PetscSection section)
{
  PetscSection   closureSection;
  IS             closureIS;
  PetscInt       offsets[32], *clIndices;
  PetscInt       depth, numFields, pStart, pEnd, point, clSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) section), &closureSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(closureSection, pStart, pEnd);CHKERRQ(ierr);
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, dof, cldof = 0;

    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= pStart) && (points[p] < pEnd)) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        cldof += dof;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(closureSection, point, cldof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(closureSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(closureSection, &clSize);CHKERRQ(ierr);
  ierr = PetscMalloc(clSize * sizeof(PetscInt), &clIndices);CHKERRQ(ierr);
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, q, cldof, cloff, fdof, f;

    ierr = PetscSectionGetDof(closureSection, point, &cldof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(closureSection, point, &cloff);CHKERRQ(ierr);
    ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    /* Compress out points not in the section, and create field offsets */
    for (p = 0, q = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= pStart) && (points[p] < pEnd)) {
        points[q*2]   = points[p];
        points[q*2+1] = points[p+1];
        for (f = 0; f < numFields; ++f) {
          ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
          offsets[f+1] += fdof;
        }
        ++q;
      }
    }
    numPoints = q;
    for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];
    if (numFields && offsets[numFields] != cldof) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], cldof);
    /* Create indices */
    for (p = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1], dof, off, d;

      ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
      if (numFields) {
        PetscInt fdof, foff, fcomp, f, c;

        for (f = 0, foff = 0; f < numFields; ++f) {
          ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
          if (o >= 0) {
            for (d = 0; d < fdof; ++d, ++offsets[f]) clIndices[cloff+offsets[f]] = off+foff+d;
          } else {
            ierr = PetscSectionGetFieldComponents(section, f, &fcomp);CHKERRQ(ierr);
            for (d = fdof/fcomp-1; d >= 0; --d) {
              for (c = 0; c < fcomp; ++c, ++offsets[f]) clIndices[cloff+offsets[f]] = off+foff+d*fcomp+c;
            }
          }
          foff += fdof;
        }
      } else {
        if (o >= 0) for (d = 0;     d < dof; ++d) clIndices[cloff+d] = off+d;
        else        for (d = dof-1; d >= 0;  --d) clIndices[cloff+d] = off+d;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, clSize, clIndices, PETSC_OWN_POINTER, &closureIS);CHKERRQ(ierr);
  ierr = PetscSectionSetClosureIndex(section, (PetscObject) dm, closureSection, closureIS);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPrintMatSetValues"
PetscErrorCode DMPlexPrintMatSetValues(PetscViewer viewer, Mat A, PetscInt point, PetscInt numIndices, const PetscInt indices[], const PetscScalar values[])
{
  PetscMPIInt    rank;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "[%D]mat for sieve point %D\n", rank, point);CHKERRQ(ierr);
  for (i = 0; i < numIndices; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "[%D]mat indices[%D] = %D\n", rank, i, indices[i]);CHKERRQ(ierr);
  }
  for (i = 0; i < numIndices; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "[%D]", rank);CHKERRQ(ierr);
    for (j = 0; j < numIndices; j++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer, " (%G,%G)", PetscRealPart(values[i*numIndices+j]), PetscImaginaryPart(values[i*numIndices+j]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer, " %G", values[i*numIndices+j]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "indicesPoint_private"
/* . off - The global offset of this point */
PetscErrorCode indicesPoint_private(PetscSection section, PetscInt point, PetscInt off, PetscInt *loff, PetscBool setBC, PetscInt orientation, PetscInt indices[])
{
  PetscInt        dof;    /* The number of unknowns on this point */
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscInt        cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  if (!cdof || setBC) {
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) indices[*loff+k] = off+k;
    } else {
      for (k = 0; k < dof; ++k) indices[*loff+dof-k-1] = off+k;
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (orientation >= 0) {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[*loff+k] = -(off+k+1);
          ++cind;
        } else {
          indices[*loff+k] = off+k-cind;
        }
      }
    } else {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[*loff+dof-k-1] = -(off+k+1);
          ++cind;
        } else {
          indices[*loff+dof-k-1] = off+k-cind;
        }
      }
    }
  }
  *loff += dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "indicesPointFields_private"
/* . off - The global offset of this point */
PetscErrorCode indicesPointFields_private(PetscSection section, PetscInt point, PetscInt off, PetscInt foffs[], PetscBool setBC, PetscInt orientation, PetscInt indices[])
{
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
        for (k = 0; k < fdof; ++k) indices[foffs[f]+k] = off+foff+k;
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
#define __FUNCT__ "DMPlexMatSetClosure"
/*@C
  DMPlexMatSetClosure - Set an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v
. globalSection - The section describing the layout in v
. A - The matrix
. point - The sieve point in the DM
. values - The array of values
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

  Fortran Notes:
  This routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexVecSetClosure()
@*/
PetscErrorCode DMPlexMatSetClosure(DM dm, PetscSection section, PetscSection globalSection, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex       *mesh   = (DM_Plex*) dm->data;
  PetscInt      *points = NULL;
  PetscInt      *indices;
  PetscInt       offsets[32];
  PetscInt       numFields, numPoints, numIndices, dof, off, globalOff, pStart, pEnd, p, q, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
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
      ierr          = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    numIndices += dof;
  }
  for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];

  if (numFields && offsets[numFields] != numIndices) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], numIndices);
  ierr = DMGetWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  if (numFields) {
    for (p = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPointFields_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, o, indices);
    }
  } else {
    for (p = 0, off = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPoint_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, o, indices);
    }
  }
  if (mesh->printSetValues) {ierr = DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numIndices, indices, values);CHKERRQ(ierr);}
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, values, mode);
  if (ierr) {
    PetscMPIInt    rank;
    PetscErrorCode ierr2;

    ierr2 = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank);CHKERRQ(ierr2);
    ierr2 = (*PetscErrorPrintf)("[%D]ERROR in DMPlexMatSetClosure\n", rank);CHKERRQ(ierr2);
    ierr2 = DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numIndices, indices, values);CHKERRQ(ierr2);
    ierr2 = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetHybridBounds"
PetscErrorCode DMPlexGetHybridBounds(DM dm, PetscInt *cMax, PetscInt *fMax, PetscInt *eMax, PetscInt *vMax)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (cMax) *cMax = mesh->hybridPointMax[dim];
  if (fMax) *fMax = mesh->hybridPointMax[dim-1];
  if (eMax) *eMax = mesh->hybridPointMax[1];
  if (vMax) *vMax = mesh->hybridPointMax[0];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetHybridBounds"
PetscErrorCode DMPlexSetHybridBounds(DM dm, PetscInt cMax, PetscInt fMax, PetscInt eMax, PetscInt vMax)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (cMax >= 0) mesh->hybridPointMax[dim]   = cMax;
  if (fMax >= 0) mesh->hybridPointMax[dim-1] = fMax;
  if (eMax >= 0) mesh->hybridPointMax[1]     = eMax;
  if (vMax >= 0) mesh->hybridPointMax[0]     = vMax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetVTKCellHeight"
PetscErrorCode DMPlexGetVTKCellHeight(DM dm, PetscInt *cellHeight)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cellHeight, 2);
  *cellHeight = mesh->vtkCellHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetVTKCellHeight"
PetscErrorCode DMPlexSetVTKCellHeight(DM dm, PetscInt cellHeight)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->vtkCellHeight = cellHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateNumbering_Private"
/* We can easily have a form that takes an IS instead */
PetscErrorCode DMPlexCreateNumbering_Private(DM dm, PetscInt pStart, PetscInt pEnd, PetscSF sf, IS *numbering)
{
  PetscSection   section, globalSection;
  PetscInt      *numbers, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(section, p, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(section, sf, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &numbers);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), pEnd - pStart, numbers, PETSC_OWN_POINTER, numbering);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetCellNumbering"
PetscErrorCode DMPlexGetCellNumbering(DM dm, IS *globalCellNumbers)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       cellHeight, cStart, cEnd, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalCellNumbers) {
    ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
    if (cMax >= 0) cEnd = PetscMin(cEnd, cMax);
    ierr = DMPlexCreateNumbering_Private(dm, cStart, cEnd, dm->sf, &mesh->globalCellNumbers);CHKERRQ(ierr);
  }
  *globalCellNumbers = mesh->globalCellNumbers;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetVertexNumbering"
PetscErrorCode DMPlexGetVertexNumbering(DM dm, IS *globalVertexNumbers)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       vStart, vEnd, vMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalVertexNumbers) {
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(dm, NULL, NULL, NULL, &vMax);CHKERRQ(ierr);
    if (vMax >= 0) vEnd = PetscMin(vEnd, vMax);
    ierr = DMPlexCreateNumbering_Private(dm, vStart, vEnd, dm->sf, &mesh->globalVertexNumbers);CHKERRQ(ierr);
  }
  *globalVertexNumbers = mesh->globalVertexNumbers;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreateGlobalSectionLabel"
/*@C
  PetscSectionCreateGlobalSectionLabel - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Input Parameters:
  + s - The PetscSection for the local field layout
  . sf - The SF describing parallel layout of the section points
  . includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not possess constrained dofs
  . label - The label specifying the points
  - labelValue - The label stratum specifying the points

  Output Parameter:
  . gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: developer

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection s, PetscSF sf, PetscBool includeConstraints, DMLabel label, PetscInt labelValue, PetscSection *gsection)
{
  PetscInt      *neg = NULL, *tmpOff = NULL;
  PetscInt       pStart, pEnd, p, dof, cdof, off, globalOff = 0, nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(s->atlasLayout.comm, gsection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*gsection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    if (nroots < pEnd-pStart) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %d < %d section size", nroots, pEnd-pStart);
    ierr = PetscMalloc(nroots * sizeof(PetscInt), &neg);CHKERRQ(ierr);
    ierr = PetscMemzero(neg, nroots * sizeof(PetscInt));CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      ierr = PetscMalloc(nroots * sizeof(PetscInt), &tmpOff);CHKERRQ(ierr);
      ierr = PetscMemzero(tmpOff, nroots * sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt value;

    ierr = DMLabelGetValue(label, p, &value);CHKERRQ(ierr);
    if (value != labelValue) continue;
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*gsection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (!includeConstraints && cdof > 0) {ierr = PetscSectionSetConstraintDof(*gsection, p, cdof);CHKERRQ(ierr);}
    if (neg) neg[p] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(*gsection);CHKERRQ(ierr);
  if (nroots >= 0) {
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasDof[p-pStart] = tmpOff[p];}
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[p] : 0;
    (*gsection)->atlasOff[p] = off;
    off += (*gsection)->atlasDof[p] > 0 ? (*gsection)->atlasDof[p]-cdof : 0;
  }
  ierr       = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, s->atlasLayout.comm);CHKERRQ(ierr);
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p] = -((*gsection)->atlasOff[p]+1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasOff[p-pStart] = tmpOff[p];}
    }
  }
  if (nroots >= 0 && nroots > pEnd-pStart) {ierr = PetscFree(tmpOff);CHKERRQ(ierr);}
  ierr = PetscFree(neg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCheckSymmetry"
/*@
  DMPlexCheckSymmetry - Check that the adjacency information in the mesh is symmetric.

  Input Parameters:
  + dm - The DMPlex object

  Note: This is a useful diagnostic when creating meshes programmatically.

  Level: developer

.seealso: DMCreate()
@*/
PetscErrorCode DMPlexCheckSymmetry(DM dm)
{
  PetscSection    coneSection, supportSection;
  const PetscInt *cone, *support;
  PetscInt        coneSize, c, supportSize, s;
  PetscInt        pStart, pEnd, p, csize, ssize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSection(dm, &supportSection);CHKERRQ(ierr);
  /* Check that point p is found in the support of its cone points, and vice versa */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      ierr = DMPlexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        if (support[s] == p) break;
      }
      if (s >= supportSize) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %d cone: ", p);
        for (s = 0; s < coneSize; ++s) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%d, ", cone[s]);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %d support: ", cone[c]);
        for (s = 0; s < supportSize; ++s) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%d, ", support[s]);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %d not found in support of cone point %d", p, cone[c]);
      }
    }
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        if (cone[c] == p) break;
      }
      if (c >= coneSize) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %d support: ", p);
        for (c = 0; c < supportSize; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%d, ", support[c]);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %d cone: ", support[s]);
        for (c = 0; c < coneSize; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%d, ", cone[c]);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %d not found in cone of support point %d", p, support[s]);
      }
    }
  }
  ierr = PetscSectionGetStorageSize(coneSection, &csize);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(supportSection, &ssize);CHKERRQ(ierr);
  if (csize != ssize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Total cone size %d != Total support size %d", csize, ssize);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCheckSkeleton"
/*@
  DMPlexCheckSkeleton - Check that each cell has the correct number of vertices

  Input Parameters:
  + dm - The DMPlex object

  Note: This is a useful diagnostic when creating meshes programmatically.

  Level: developer

.seealso: DMCreate()
@*/
PetscErrorCode DMPlexCheckSkeleton(DM dm, PetscBool isSimplex)
{
  DM             udm;
  PetscInt       dim, numCorners, coneSize, cStart, cEnd, cMax, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  switch (dim) {
  case 2: numCorners = isSimplex ? 3 : 4; break;
  case 3: numCorners = isSimplex ? 4 : 8; break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle meshes of dimension %d", dim);
  }
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) udm, "un_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(udm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  cMax = cMax >= 0 ? cMax : cEnd;
  for (c = cStart; c < cMax; ++c) {
    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    if (coneSize != numCorners) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %d has  %d vertices != %d", c, coneSize, numCorners);
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
