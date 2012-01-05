static char help[] = "Mesh Distribution with SF.\n\n";
#include <petscdmmesh.h>
#include <petscsf.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  char          filename[2048];    /* Optional filename to read mesh from */
  char          partitioner[2048]; /* The graph partitioner */
  PetscLogEvent createMeshEvent;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Mesh Distribution Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex1.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The input filename", "ex1.c", options->filename, options->filename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "ex1.c", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",    DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMeshCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMMeshRefine(*dm, refinementLimit, interpolate, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFConvertPartition"
PetscErrorCode PetscSFConvertPartition(PetscSF sfPart, PetscSection partSection, IS partition, ISLocalToGlobalMapping *renumbering, PetscSF *sf)
{
  MPI_Comm       comm = ((PetscObject)sfPart)->comm;
  PetscSF        sfPoints;
  PetscInt       *partSizes,*partOffsets,p,i,numParts,numMyPoints,numPoints,count;
  const PetscInt *partArray;
  PetscSFNode    *sendPoints;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* Get the number of parts and sizes that I have to distribute */
  ierr = PetscSFGetGraph(sfPart,PETSC_NULL,&numParts,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(numParts,PetscInt,&partSizes,numParts,PetscInt,&partOffsets);CHKERRQ(ierr);
  for (p=0,numPoints=0; p<numParts; p++) {
    ierr = PetscSectionGetDof(partSection, p, &partSizes[p]);CHKERRQ(ierr);
    numPoints += partSizes[p];
  }
  numMyPoints = 0;
  ierr = PetscSFFetchAndOpBegin(sfPart,MPIU_INT,&numMyPoints,partSizes,partOffsets,MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(sfPart,MPIU_INT,&numMyPoints,partSizes,partOffsets,MPIU_SUM);CHKERRQ(ierr);
  /* I will receive numMyPoints. I will send a total of numPoints, to be placed on remote procs at partOffsets */

  /* Create SF mapping locations (addressed through partition, as indexed leaves) to new owners (roots) */
  ierr = PetscMalloc(numPoints*sizeof(PetscSFNode),&sendPoints);CHKERRQ(ierr);
  for (p=0,count=0; p<numParts; p++) {
    for (i=0; i<partSizes[p]; i++) {
      sendPoints[count].rank = p;
      sendPoints[count].index = partOffsets[p]+i;
      count++;
    }
  }
  if (count != numPoints) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count %D should equal numPoints=%D",count,numPoints);
  ierr = PetscFree2(partSizes,partOffsets);CHKERRQ(ierr);
  ierr = ISGetIndices(partition,&partArray);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sfPoints);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfPoints,numMyPoints,numPoints,partArray,PETSC_USE_POINTER,sendPoints,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* Invert SF so that the new owners are leaves and the locations indexed through partition are the roots */
  ierr = PetscSFCreateInverseSF(sfPoints,sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPoints);CHKERRQ(ierr);
  ierr = ISRestoreIndices(partition,&partArray);CHKERRQ(ierr);

  /* Create the new local-to-global mapping */
  ierr = ISLocalToGlobalMappingCreateSF(*sf,0,renumbering);CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(*renumbering, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFView(*sf,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFDistributeSection"
PetscErrorCode PetscSFDistributeSection(PetscSF sf, PetscSection originalSection, PetscInt **remoteOffsets, PetscSection newSection)
{
  PetscSF         embedSF;
  const PetscInt *ilocal, *indices;
  IS              selected;
  PetscInt        nleaves, rpStart, rpEnd, pStart = PETSC_MAX_INT, pEnd = -1, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(originalSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
  ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&selected);CHKERRQ(ierr);
  ierr = PetscSFView(embedSF, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(embedSF, PETSC_NULL, &nleaves, &ilocal, PETSC_NULL);CHKERRQ(ierr);
  if (ilocal) {
    for(i = 0; i < nleaves; ++i) {
      pStart = PetscMin(pStart, ilocal[i]);
      pEnd   = PetscMax(pEnd,   ilocal[i]);
    }
  } else {
    pStart = 0;
    pEnd   = nleaves;
  }
  ++pEnd;
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newSection, pStart, pEnd);CHKERRQ(ierr);
  /* Could fuse these at the cost of a copy and extra allocation */
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &originalSection->atlasDof[-originalSection->atlasLayout.pStart], &newSection->atlasDof[-newSection->atlasLayout.pStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &originalSection->atlasDof[-originalSection->atlasLayout.pStart], &newSection->atlasDof[-newSection->atlasLayout.pStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &originalSection->atlasOff[-originalSection->atlasLayout.pStart], &(*remoteOffsets)[-pStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &originalSection->atlasOff[-originalSection->atlasLayout.pStart], &(*remoteOffsets)[-pStart]);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(newSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreateSectionSF"
/*
  . section       - Data layout of local points for incoming data
  . remoteOffsets - Offsets for point data on remote processes
*/
PetscErrorCode PetscSFCreateSectionSF(PetscSF sf, PetscSection section, const PetscInt remoteOffsets[], PetscSF *sectionSF)
{
  MPI_Comm           comm = ((PetscObject) sf)->comm;
  const PetscInt    *localPoints;
  PetscInt           pStart, pEnd;
  PetscInt           numRanks;
  const PetscInt    *ranks, *rankOffsets;
  PetscInt           numPoints, numIndices = 0;
  PetscInt          *localIndices;
  PetscSFNode       *remoteIndices;
  PetscInt           i, r, ind;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFView(sf, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, PETSC_NULL, &numPoints, &localPoints, PETSC_NULL);CHKERRQ(ierr);
  for(i = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt dof;

    if ((localPoint >= pStart) && (localPoint < pEnd)) {
      ierr = PetscSectionGetDof(section, localPoint, &dof);CHKERRQ(ierr);
      numIndices += dof;
    }
  }
  ierr = PetscMalloc(numIndices * sizeof(PetscInt), &localIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(numIndices * sizeof(PetscSFNode), &remoteIndices);CHKERRQ(ierr);
  /* Create new index graph */
  ierr = PetscSFGetRanks(sf, &numRanks, &ranks, &rankOffsets, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  for(r = 0, ind = 0; r < numRanks; ++r) {
    PetscInt rank = ranks[r];

    for(i = rankOffsets[r]; i < rankOffsets[r+1]; ++i) {
      PetscInt localPoint   = localPoints ? localPoints[i] : i;

      if ((localPoint >= pStart) && (localPoint < pEnd)) {
        PetscInt remoteOffset = remoteOffsets[localPoint-pStart];
        PetscInt localOffset, dof, d;
        ierr = PetscSectionGetOffset(section, localPoint, &localOffset);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(section, localPoint, &dof);CHKERRQ(ierr);
        for(d = 0; d < dof; ++d, ++ind) {
          localIndices[ind]        = localOffset+d;
          remoteIndices[ind].rank  = rank;
          remoteIndices[ind].index = remoteOffset+d;
        }
      }
    }
  }
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (numIndices != ind) {SETERRQ2(comm, PETSC_ERR_PLIB, "Inconsistency in indices, %d should be %d", ind, numIndices);}
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sectionSF, numIndices, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFView(*sectionSF, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateNeighborCSR"
PetscErrorCode DMMeshCreateNeighborCSR(DM dm, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency) {
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[maxFaceCases];
  PetscInt      *off, *adj;
  PetscInt       dim, depth, cStart, cEnd, c, numCells, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMMeshGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMMeshGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
  --depth;
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (cEnd - cStart == 0) {
    *numVertices = 0;
    *offsets     = PETSC_NULL;
    *adjacency   = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  numCells = cEnd - cStart;
  /* Setup face recognition */
  {
    PetscInt cornersSeen[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* Could use PetscBT */

    for(c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      ierr = DMMeshGetConeSize(dm, c, &corners);CHKERRQ(ierr);
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
    ierr = DMMeshGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc(maxConeSize*maxSupportSize * sizeof(PetscInt), &neighborCells);CHKERRQ(ierr);
    /* Count neighboring cells */
    for(cell = cStart; cell < cEnd; ++cell) {
      const PetscInt *cone;
      PetscInt        numNeighbors = 0;
      PetscInt        coneSize, c;

      /* Get support of the cone, and make a set of the cells */
      ierr = DMMeshGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMMeshGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMMeshGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
        ierr = DMMeshGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
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

        ierr = DMMeshMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
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
    ierr = PetscMalloc(off[numCells] * sizeof(PetscInt), &adj);CHKERRQ(ierr);
    /* Get neighboring cells */
    for(cell = cStart; cell < cEnd; ++cell) {
      const PetscInt *cone;
      PetscInt        numNeighbors = 0;
      PetscInt        cellOffset   = 0;
      PetscInt        coneSize, c;

      /* Get support of the cone, and make a set of the cells */
      ierr = DMMeshGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMMeshGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for(c = 0; c < coneSize; ++c) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMMeshGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
        ierr = DMMeshGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
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

        ierr = DMMeshMeetPoints(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
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
  *numVertices = numCells;
  *offsets     = off;
  *adjacency   = adj;
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_CHACO
#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif
/* Chaco does not have an include file */
extern "C" {
  extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

  extern int FREE_GRAPH;
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshPartition_Chaco"
PetscErrorCode DMMeshPartition_Chaco(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
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
#define __FUNCT__ "DMMeshPartition_ParMetis"
PetscErrorCode DMMeshPartition_ParMetis(DM dm, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection *partSection, IS *partition)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreatePartition"
PetscErrorCode DMMeshCreatePartition(DM dm, PetscSection *partSection, IS *partition, PetscInt height) {
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject) dm)->comm, &size);CHKERRQ(ierr);
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
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
      ierr = DMMeshCreateNeighborCSR(dm, &numVertices, &start, &adjacency);CHKERRQ(ierr);
#ifdef PETSC_HAVE_CHACO
      ierr = DMMeshPartition_Chaco(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
#endif
    } else {
      ierr = DMMeshCreateNeighborCSR(dm, &numVertices, &start, &adjacency);CHKERRQ(ierr);
#ifdef PETSC_HAVE_PARMETIS
      ierr = DMMeshPartition_ParMetis(dm, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMeshCreatePartitionClosure"
PetscErrorCode DMMeshCreatePartitionClosure(DM dm, PetscSection pointSection, IS pointPartition, PetscSection *section, IS *partition) {
  const PetscInt  height = 0;
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
      ierr = DMMeshGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
      ierr = DMMeshGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
#define __FUNCT__ "DistributeMesh"
/* Distribute cones
   - Partitioning:         input partition point map and naive sf, output sf with inverse of map, distribute points
   - Distribute section:   input current sf, communicate sizes and offsets, output local section and offsets (only use for new sf)
   - Create SF for values: input current sf and offsets, output new sf
   - Distribute values:    input new sf, communicate values
 */
PetscErrorCode DistributeMesh(DM dm, AppCtx *user, PetscSF *pointSF, DM *parallelDM)
{
  MPI_Comm       comm   = ((PetscObject) dm)->comm;
  const PetscInt height = 0;
  PetscInt       dim, numRemoteRanks;
  IS             cellPart,        part;
  PetscSection   cellPartSection, partSection;
  PetscSFNode   *remoteRanks;
  PetscSF        partSF;
  ISLocalToGlobalMapping renumbering;
  PetscSF        coneSF;
  PetscSection   originalConeSection, newConeSection;
  PetscInt      *remoteOffsets, newConesSize;
  PetscInt      *cones, *newCones;
  PetscBool      flg;
  PetscMPIInt    numProcs, rank, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMMeshGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = DMMeshCreatePartition(dm, &cellPartSection, &cellPart, height);CHKERRQ(ierr);
  ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(cellPart, PETSC_NULL);CHKERRQ(ierr);
  /* Debugging */
  ierr = PetscOptionsHasName(PETSC_NULL, "-output_partition", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer    viewer;
    PetscErrorCode ierr;

    ierr = PetscViewerCreate(comm, &viewer);CHKERRXX(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
    ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRXX(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dm, viewer);CHKERRQ(ierr);
    ierr = ISView(cellPart, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
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
  ierr = PetscSFView(partSF, PETSC_NULL);CHKERRQ(ierr);
  /* Close the partition over the mesh */
  ierr = DMMeshCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(part, PETSC_NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr = DMMeshCreate(comm, parallelDM);CHKERRQ(ierr);
  ierr = DMMeshSetDimension(*parallelDM, dim);CHKERRQ(ierr);
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, pointSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Distribute cone section */
  ierr = DMMeshGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMMeshGetConeSection(*parallelDM, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(*pointSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMMeshSetUp(*parallelDM);CHKERRQ(ierr);
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(*pointSF, newConeSection, remoteOffsets, &coneSF);CHKERRQ(ierr);
  ierr = DMMeshGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMMeshGetCones(*parallelDM, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, newConesSize, newCones, PETSC_NULL, newCones);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  ierr = DMMeshSymmetrize(*parallelDM);CHKERRQ(ierr);
  ierr = DMMeshStratify(*parallelDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeCoordinates"
PetscErrorCode DistributeCoordinates(DM dm, PetscSF pointSF, DM parallelDM)
{
  PetscSF        coordSF;
  PetscSection   originalCoordSection, newCoordSection;
  Vec            coordinates, newCoordinates;
  PetscScalar   *coords,     *newCoords;
  PetscInt      *remoteOffsets, coordSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateSection(parallelDM, &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalCoordSection, &remoteOffsets, newCoordSection);CHKERRQ(ierr);

  ierr = DMMeshGetCoordinateVec(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateVec(parallelDM, &newCoordinates);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newCoordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(newCoordinates);CHKERRQ(ierr);

  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, newCoordSection, remoteOffsets, &coordSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coordSF, MPIU_SCALAR, coords, newCoords);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coordSF, MPIU_SCALAR, coords, newCoords);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coordSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             dm, parallelDM;
  PetscSF        pointSF;
  AppCtx         user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DistributeMesh(dm, &user, &pointSF, &parallelDM);CHKERRQ(ierr);
  ierr = DistributeCoordinates(dm, pointSF, parallelDM);CHKERRQ(ierr);
  ierr = DMSetFromOptions(parallelDM);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  ierr = DMDestroy(&parallelDM);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
