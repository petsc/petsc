#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

const char * const CellRefiners[] = {"NOOP", "SIMPLEX_1D", "SIMPLEX_2D", "HYBRID_SIMPLEX_2D", "SIMPLEX_TO_HEX_2D", "HYBRID_SIMPLEX_TO_HEX_2D", "HEX_2D", "HYBRID_HEX_2D",
                                     "SIMPLEX_3D", "HYBRID_SIMPLEX_3D", "SIMPLEX_TO_HEX_3D", "HYBRID_SIMPLEX_TO_HEX_3D", "HEX_3D", "HYBRID_HEX_3D", "CellRefiners", "REFINER_", 0};

PETSC_STATIC_INLINE PetscErrorCode GetDepthStart_Private(PetscInt depth, PetscInt depthSize[], PetscInt *cStart, PetscInt *fStart, PetscInt *eStart, PetscInt *vStart)
{
  PetscFunctionBegin;
  if (cStart) *cStart = 0;
  if (vStart) *vStart = depth < 0 ? 0 : depthSize[depth];
  if (fStart) *fStart = depth < 0 ? 0 : depthSize[depth] + depthSize[0];
  if (eStart) *eStart = depth < 0 ? 0 : depthSize[depth] + depthSize[0] + depthSize[depth-1];
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode GetDepthEnd_Private(PetscInt depth, PetscInt depthSize[], PetscInt *cEnd, PetscInt *fEnd, PetscInt *eEnd, PetscInt *vEnd)
{
  PetscFunctionBegin;
  if (cEnd) *cEnd = depth < 0 ? 0 : depthSize[depth];
  if (vEnd) *vEnd = depth < 0 ? 0 : depthSize[depth] + depthSize[0];
  if (fEnd) *fEnd = depth < 0 ? 0 : depthSize[depth] + depthSize[0] + depthSize[depth-1];
  if (eEnd) *eEnd = depth < 0 ? 0 : depthSize[depth] + depthSize[0] + depthSize[depth-1] + depthSize[1];
  PetscFunctionReturn(0);
}

/*
  Note that j and invj are non-square:
         v0 + j x_face = x_cell
    invj (x_cell - v0) = x_face
*/
PetscErrorCode CellRefinerGetAffineFaceTransforms_Internal(CellRefiner refiner, PetscInt *numFaces, PetscReal *v0[], PetscReal *jac[], PetscReal *invjac[], PetscReal *detj[])
{
  PetscReal     *v = NULL, *j = NULL, *invj = NULL, *dj = NULL;
  PetscInt       cdim, fdim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (refiner) {
  case REFINER_NOOP: break;
  case REFINER_SIMPLEX_2D:
    /*
     2
     |\
     | \
     |  \
     |   \
     |    \
     |     \
     |      \
     2       1
     |        \
     |         \
     |          \
     0---0-------1
     */
    cdim = 2;
    fdim = 1;
    if (numFaces) *numFaces = 3;
    if (v0) {
      ierr = PetscMalloc1(3*cdim,      &v);CHKERRQ(ierr);
      ierr = PetscMalloc1(3*cdim*fdim, &j);CHKERRQ(ierr);
      ierr = PetscMalloc1(3*cdim*fdim, &invj);CHKERRQ(ierr);
      ierr = PetscMalloc1(3,           &dj);CHKERRQ(ierr);
      /* 0 */
      v[0+0] =  0.0; v[0+1] = -1.0;
      j[0+0] =  1.0;
      j[0+1] =  0.0;
      invj[0+0] = 1.0; invj[0+1] = 0.0;
      dj[0]  = 1.0;
      /* 1 */
      v[2+0] =  0.0; v[2+1] =  0.0;
      j[2+0] = -1.0;
      j[2+1] =  1.0;
      invj[2+0] = -0.5; invj[2+1] = 0.5;
      dj[1]  = 1.414213562373095;
      /* 2 */
      v[4+0] = -1.0; v[4+1] =  0.0;
      j[4+0] =  0.0;
      j[4+1] = -1.0;
      invj[4+0] = 0.0; invj[4+1] = -1.0;
      dj[2]  = 1.0;
    }
    break;
  case REFINER_HEX_2D:
    /*
     3---------2---------2
     |                   |
     |                   |
     |                   |
     3                   1
     |                   |
     |                   |
     |                   |
     0---------0---------1
     */
    cdim = 2;
    fdim = 1;
    if (numFaces) *numFaces = 4;
    if (v0) {
      ierr = PetscMalloc1(4*cdim,      &v);CHKERRQ(ierr);
      ierr = PetscMalloc1(4*cdim*fdim, &j);CHKERRQ(ierr);
      ierr = PetscMalloc1(4*cdim*fdim, &invj);CHKERRQ(ierr);
      ierr = PetscMalloc1(4,           &dj);CHKERRQ(ierr);
      /* 0 */
      v[0+0] =  0.0; v[0+1] = -1.0;
      j[0+0] =  1.0;
      j[0+1] =  0.0;
      invj[0+0] =  1.0; invj[0+1] =  0.0;
      dj[0]  = 1.0;
      /* 1 */
      v[2+0] =  1.0; v[2+1] =  0.0;
      j[2+0] =  0.0;
      j[2+1] =  1.0;
      invj[2+0] =  0.0; invj[2+1] =  1.0;
      dj[1]  = 1.0;
      /* 2 */
      v[4+0] =  0.0; v[4+1] = 1.0;
      j[4+0] = -1.0;
      j[4+1] =  0.0;
      invj[4+0] = -1.0; invj[4+1] =  0.0;
      dj[2]  = 1.0;
      /* 3 */
      v[6+0] = -1.0; v[6+1] =  0.0;
      j[6+0] =  0.0;
      j[6+1] = -1.0;
      invj[6+0] =  0.0; invj[6+1] = -1.0;
      dj[3]  = 1.0;
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  if (v0)     {*v0     = v;}
  else        {ierr = PetscFree(v);CHKERRQ(ierr);}
  if (jac)    {*jac    = j;}
  else        {ierr = PetscFree(j);CHKERRQ(ierr);}
  if (invjac) {*invjac = invj;}
  else        {ierr = PetscFree(invj);CHKERRQ(ierr);}
  if (invjac) {*invjac = invj;}
  else        {ierr = PetscFree(invj);CHKERRQ(ierr);}
  if (detj)   {*detj   = dj;}
  else        {ierr = PetscFree(dj);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Gets the affine map from the original cell to each subcell */
PetscErrorCode CellRefinerGetAffineTransforms_Internal(CellRefiner refiner, PetscInt *numSubcells, PetscReal *v0[], PetscReal *jac[], PetscReal *invjac[])
{
  PetscReal     *v = NULL, *j = NULL, *invj = NULL, detJ;
  PetscInt       dim, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (refiner) {
  case REFINER_NOOP: break;
  case REFINER_SIMPLEX_2D:
    /*
     2
     |\
     | \
     |  \
     |   \
     | C  \
     |     \
     |      \
     2---1---1
     |\  D  / \
     | 2   0   \
     |A \ /  B  \
     0---0-------1
     */
    dim = 2;
    if (numSubcells) *numSubcells = 4;
    if (v0) {
      ierr = PetscMalloc3(4*dim,&v,4*dim*dim,&j,4*dim*dim,&invj);CHKERRQ(ierr);
      /* A */
      v[0+0] = -1.0; v[0+1] = -1.0;
      j[0+0] =  0.5; j[0+1] =  0.0;
      j[0+2] =  0.0; j[0+3] =  0.5;
      /* B */
      v[2+0] =  0.0; v[2+1] = -1.0;
      j[4+0] =  0.5; j[4+1] =  0.0;
      j[4+2] =  0.0; j[4+3] =  0.5;
      /* C */
      v[4+0] = -1.0; v[4+1] =  0.0;
      j[8+0] =  0.5; j[8+1] =  0.0;
      j[8+2] =  0.0; j[8+3] =  0.5;
      /* D */
      v[6+0]  =  0.0; v[6+1]  = -1.0;
      j[12+0] =  0.0; j[12+1] = -0.5;
      j[12+2] =  0.5; j[12+3] =  0.5;
      for (s = 0; s < 4; ++s) {
        DMPlex_Det2D_Internal(&detJ, &j[s*dim*dim]);
        DMPlex_Invert2D_Internal(&invj[s*dim*dim], &j[s*dim*dim], detJ);
      }
    }
    break;
  case REFINER_HEX_2D:
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    dim = 2;
    if (numSubcells) *numSubcells = 4;
    if (v0) {
      ierr = PetscMalloc3(4*dim,&v,4*dim*dim,&j,4*dim*dim,&invj);CHKERRQ(ierr);
      /* A */
      v[0+0] = -1.0; v[0+1] = -1.0;
      j[0+0] =  0.5; j[0+1] =  0.0;
      j[0+2] =  0.0; j[0+3] =  0.5;
      /* B */
      v[2+0] =  0.0; v[2+1] = -1.0;
      j[4+0] =  0.5; j[4+1] =  0.0;
      j[4+2] =  0.0; j[4+3] =  0.5;
      /* C */
      v[4+0] =  0.0; v[4+1] =  0.0;
      j[8+0] =  0.5; j[8+1] =  0.0;
      j[8+2] =  0.0; j[8+3] =  0.5;
      /* D */
      v[6+0]  = -1.0; v[6+1]  =  0.0;
      j[12+0] =  0.5; j[12+1] =  0.0;
      j[12+2] =  0.0; j[12+3] =  0.5;
      for (s = 0; s < 4; ++s) {
        DMPlex_Det2D_Internal(&detJ, &j[s*dim*dim]);
        DMPlex_Invert2D_Internal(&invj[s*dim*dim], &j[s*dim*dim], detJ);
      }
    }
    break;
  case REFINER_HEX_3D:
    /*
     Bottom (viewed from top)    Top
     1---------2---------2       7---------2---------6
     |         |         |       |         |         |
     |    B    2    C    |       |    H    2    G    |
     |         |         |       |         |         |
     3----3----0----1----1       3----3----0----1----1
     |         |         |       |         |         |
     |    A    0    D    |       |    E    0    F    |
     |         |         |       |         |         |
     0---------0---------3       4---------0---------5
     */
    break;
    dim = 3;
    if (numSubcells) *numSubcells = 8;
    if (v0) {
      ierr = PetscMalloc3(4*dim,&v,4*dim*dim,&j,4*dim*dim,&invj);CHKERRQ(ierr);
      /* A */
      v[0+0] = -1.0; v[0+1] = -1.0; v[0+2] = -1.0;
      j[0+0] =  0.5; j[0+1] =  0.0; j[0+2] =  0.0;
      j[0+3] =  0.0; j[0+4] =  0.5; j[0+5] =  0.0;
      j[0+6] =  0.0; j[0+7] =  0.0; j[0+8] =  0.5;
      /* B */
      v[3+0] = -1.0; v[3+1] =  0.0; v[3+2] = -1.0;
      j[9+0] =  0.5; j[9+1] =  0.0; j[9+2] =  0.0;
      j[9+3] =  0.0; j[9+4] =  0.5; j[9+5] =  0.0;
      j[9+6] =  0.0; j[9+7] =  0.0; j[9+8] =  0.5;
      /* C */
      v[6+0] =  0.0; v[6+1] =  0.0; v[6+2] = -1.0;
      j[18+0] = 0.5; j[18+1] = 0.0; j[18+2] = 0.0;
      j[18+3] = 0.0; j[18+4] = 0.5; j[18+5] = 0.0;
      j[18+6] = 0.0; j[18+7] = 0.0; j[18+8] = 0.5;
      /* D */
      v[9+0] =  0.0; v[9+1] = -1.0; v[9+2] = -1.0;
      j[27+0] = 0.5; j[27+1] = 0.0; j[27+2] = 0.0;
      j[27+3] = 0.0; j[27+4] = 0.5; j[27+5] = 0.0;
      j[27+6] = 0.0; j[27+7] = 0.0; j[27+8] = 0.5;
      /* E */
      v[12+0] = -1.0; v[12+1] = -1.0; v[12+2] =  0.0;
      j[36+0] =  0.5; j[36+1] =  0.0; j[36+2] =  0.0;
      j[36+3] =  0.0; j[36+4] =  0.5; j[36+5] =  0.0;
      j[36+6] =  0.0; j[36+7] =  0.0; j[36+8] =  0.5;
      /* F */
      v[15+0] =  0.0; v[15+1] = -1.0; v[15+2] =  0.0;
      j[45+0] =  0.5; j[45+1] =  0.0; j[45+2] =  0.0;
      j[45+3] =  0.0; j[45+4] =  0.5; j[45+5] =  0.0;
      j[45+6] =  0.0; j[45+7] =  0.0; j[45+8] =  0.5;
      /* G */
      v[18+0] =  0.0; v[18+1] =  0.0; v[18+2] =  0.0;
      j[54+0] =  0.5; j[54+1] =  0.0; j[54+2] =  0.0;
      j[54+3] =  0.0; j[54+4] =  0.5; j[54+5] =  0.0;
      j[54+6] =  0.0; j[54+7] =  0.0; j[54+8] =  0.5;
      /* H */
      v[21+0] = -1.0; v[21+1] =  0.0; v[21+2] =  0.0;
      j[63+0] =  0.5; j[63+1] =  0.0; j[63+2] =  0.0;
      j[63+3] =  0.0; j[63+4] =  0.5; j[63+5] =  0.0;
      j[63+6] =  0.0; j[63+7] =  0.0; j[63+8] =  0.5;
      for (s = 0; s < 8; ++s) {
        DMPlex_Det3D_Internal(&detJ, &j[s*dim*dim]);
        DMPlex_Invert3D_Internal(&invj[s*dim*dim], &j[s*dim*dim], detJ);
      }
    }
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  if (v0) {*v0 = v; *jac = j; *invjac = invj;}
  PetscFunctionReturn(0);
}

PetscErrorCode CellRefinerRestoreAffineTransforms_Internal(CellRefiner refiner, PetscInt *numSubcells, PetscReal *v0[], PetscReal *jac[], PetscReal *invjac[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree3(*v0,*jac,*invjac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Should this be here or in the DualSpace somehow? */
PetscErrorCode CellRefinerInCellTest_Internal(CellRefiner refiner, const PetscReal point[], PetscBool *inside)
{
  PetscReal sum = 0.0;
  PetscInt  d;

  PetscFunctionBegin;
  *inside = PETSC_TRUE;
  switch (refiner) {
  case REFINER_NOOP: break;
  case REFINER_SIMPLEX_2D:
    for (d = 0; d < 2; ++d) {
      if (point[d] < -1.0) {*inside = PETSC_FALSE; break;}
      sum += point[d];
    }
    if (sum > 1.0e-10) {*inside = PETSC_FALSE; break;}
    break;
  case REFINER_HEX_2D:
    for (d = 0; d < 2; ++d) if ((point[d] < -1.00000000001) || (point[d] > 1.000000000001)) {*inside = PETSC_FALSE; break;}
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerGetSizes(CellRefiner refiner, DM dm, PetscInt depthSize[])
{
  PetscInt       cStart, cEnd, cMax, vStart, vEnd, vMax, fStart, fEnd, fMax, eStart, eEnd, eMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  switch (refiner) {
  case REFINER_NOOP:
    break;
  case REFINER_SIMPLEX_1D:
    depthSize[0] = vEnd - vStart + cEnd - cStart;         /* Add a vertex on every cell. */
    depthSize[1] = 2*(cEnd - cStart);                     /* Split every cell in 2. */
    break;
  case REFINER_SIMPLEX_2D:
    depthSize[0] = vEnd - vStart + fEnd - fStart;         /* Add a vertex on every face */
    depthSize[1] = 2*(fEnd - fStart) + 3*(cEnd - cStart); /* Every face is split into 2 faces and 3 faces are added for each cell */
    depthSize[2] = 4*(cEnd - cStart);                     /* Every cell split into 4 cells */
    break;
  case REFINER_HYBRID_SIMPLEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    depthSize[0] = vEnd - vStart + fMax - fStart;                                         /* Add a vertex on every face, but not hybrid faces */
    depthSize[1] = 2*(fMax - fStart) + 3*(cMax - cStart) + (fEnd - fMax) + (cEnd - cMax); /* Every interior face is split into 2 faces, 3 faces are added for each interior cell, and one in each hybrid cell */
    depthSize[2] = 4*(cMax - cStart) + 2*(cEnd - cMax);                                   /* Interior cells split into 4 cells, hybrid cells split into 2 cells */
    break;
  case REFINER_SIMPLEX_TO_HEX_2D:
    depthSize[0] = vEnd - vStart + fEnd - fStart + cEnd - cStart; /* Add a vertex on every face and cell */
    depthSize[1] = 2*(fEnd - fStart) + 3*(cEnd - cStart);         /* Every face is split into 2 faces and 3 faces are added for each cell */
    depthSize[2] = 3*(cEnd - cStart);                             /* Every cell split into 3 cells */
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    depthSize[0] = vEnd - vStart + fEnd - fStart + cEnd - cStart;           /* Add a vertex on every face and cell */
    depthSize[1] = 2*(fEnd - fStart) + 3*(cMax - cStart) + 4*(cEnd - cMax); /* Every face is split into 2 faces and 3 faces are added for each cell. 4 for each hybrid cell */
    depthSize[2] = 3*(cMax - cStart) + 4*(cEnd - cMax);                     /* Every cell split into 3 cells, hybrid cells split in 4 */
    break;
  case REFINER_HEX_2D:
    depthSize[0] = vEnd - vStart + fEnd - fStart + cEnd - cStart; /* Add a vertex on every face and cell */
    depthSize[1] = 2*(fEnd - fStart) + 4*(cEnd - cStart);         /* Every face is split into 2 faces and 4 faces are added for each cell */
    depthSize[2] = 4*(cEnd - cStart);                             /* Every cell split into 4 cells */
    break;
  case REFINER_HYBRID_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    /* Quadrilateral */
    depthSize[0] = vEnd - vStart + fMax - fStart + cMax - cStart;                 /* Add a vertex on every face and cell */
    depthSize[1] = 2*(fMax - fStart) + 4*(cMax - cStart);                         /* Every face is split into 2 faces, and 4 faces are added for each cell */
    depthSize[2] = 4*(cMax - cStart);                                             /* Every cell split into 4 cells */
    /* Segment Prisms */
    depthSize[0] += 0;                                                            /* No hybrid vertices */
    depthSize[1] +=   (fEnd - fMax)  +   (cEnd - cMax);                           /* Every hybrid face remains and 1 faces is added for each hybrid cell */
    depthSize[2] += 2*(cEnd - cMax);                                              /* Every hybrid cell split into 2 cells */
    break;
  case REFINER_SIMPLEX_3D:
    depthSize[0] =    vEnd - vStart  +    eEnd - eStart;                    /* Add a vertex on every edge */
    depthSize[1] = 2*(eEnd - eStart) + 3*(fEnd - fStart) + (cEnd - cStart); /* Every edge is split into 2 edges, 3 edges are added for each face, and 1 edge for each cell */
    depthSize[2] = 4*(fEnd - fStart) + 8*(cEnd - cStart);                   /* Every face split into 4 faces and 8 faces are added for each cell */
    depthSize[3] = 8*(cEnd - cStart);                                       /* Every cell split into 8 cells */
    break;
  case REFINER_HYBRID_SIMPLEX_3D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
    /* Tetrahedra */
    depthSize[0]  =    vEnd - vStart  +    eMax - eStart;                    /* Add a vertex on every interior edge */
    depthSize[1]  = 2*(eMax - eStart) + 3*(fMax - fStart) + (cMax - cStart); /* Every interior edge split into 2 edges, 3 edges added for each interior face, 1 edge for each interior cell */
    depthSize[2]  = 4*(fMax - fStart) + 8*(cMax - cStart);                   /* Every interior face split into 4 faces, 8 faces added for each interior cell */
    depthSize[3]  = 8*(cMax - cStart);                                       /* Every interior cell split into 8 cells */
    /* Triangular Prisms */
    depthSize[0] += 0;                                                       /* No hybrid vertices */
    depthSize[1] +=   (eEnd - eMax)   +   (fEnd - fMax);                     /* Every hybrid edge remains, 1 edge for every hybrid face */
    depthSize[2] += 2*(fEnd - fMax)   + 3*(cEnd - cMax);                     /* Every hybrid face split into 2 faces and 3 faces are added for each hybrid cell */
    depthSize[3] += 4*(cEnd - cMax);                                         /* Every hybrid cell split into 4 cells */
    break;
  case REFINER_SIMPLEX_TO_HEX_3D:
    depthSize[0] = vEnd - vStart + fEnd - fStart + eEnd - eStart + cEnd - cStart; /* Add a vertex on every face, edge and cell */
    depthSize[1] = 2*(eEnd - eStart) + 3*(fEnd - fStart) + 4*(cEnd - cStart);     /* Every edge is split into 2 edges, 3 edges are added for each face, and 4 for each cell */
    depthSize[2] = 3*(fEnd - fStart) + 6*(cEnd - cStart);                         /* Every face is split into 3 faces and 6 faces are added for each cell */
    depthSize[3] = 4*(cEnd - cStart);                                             /* Every cell split into 4 cells */
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
    /* Tetrahedra */
    depthSize[0]  =    vEnd - vStart  +    eMax - eStart  + fMax - fStart + cMax - cStart; /* Add a vertex on every interior edge, face and cell */
    depthSize[1]  = 2*(eMax - eStart) + 3*(fMax - fStart) + 4*(cMax - cStart);             /* Every interior edge split into 2 edges, 3 edges added for each interior face, 4 edges for each interior cell */
    depthSize[2]  = 3*(fMax - fStart) + 6*(cMax - cStart);                                 /* Every interior face split into 3 faces, 6 faces added for each interior cell */
    depthSize[3]  = 4*(cMax - cStart);                                                     /* Every interior cell split into 8 cells */
    /* Triangular Prisms */
    depthSize[0] += 0;                                                 /* No hybrid vertices */
    depthSize[1] +=   (eEnd - eMax) +   (fEnd - fMax) + (cEnd - cMax); /* Every hybrid edge remains, 1 edge for every hybrid face and cell */
    depthSize[2] += 2*(fEnd - fMax) + 3*(cEnd - cMax);                 /* Every hybrid face split into 2 faces and 3 faces are added for each hybrid cell */
    depthSize[3] += 3*(cEnd - cMax);                                   /* Every hybrid cell split into 3 cells */
    break;
  case REFINER_HEX_3D:
    depthSize[0] = vEnd - vStart + eEnd - eStart + fEnd - fStart + cEnd - cStart; /* Add a vertex on every edge, face and cell */
    depthSize[1] = 2*(eEnd - eStart) +  4*(fEnd - fStart) + 6*(cEnd - cStart);    /* Every edge is split into 2 edge, 4 edges are added for each face, and 6 edges for each cell */
    depthSize[2] = 4*(fEnd - fStart) + 12*(cEnd - cStart);                        /* Every face is split into 4 faces, and 12 faces are added for each cell */
    depthSize[3] = 8*(cEnd - cStart);                                             /* Every cell split into 8 cells */
    break;
  case REFINER_HYBRID_HEX_3D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
    /* Hexahedra */
    depthSize[0] = vEnd - vStart + eMax - eStart + fMax - fStart + cMax - cStart; /* Add a vertex on every edge, face and cell */
    depthSize[1] = 2*(eMax - eStart) +  4*(fMax - fStart) + 6*(cMax - cStart);    /* Every edge is split into 2 edge, 4 edges are added for each face, and 6 edges for each cell */
    depthSize[2] = 4*(fMax - fStart) + 12*(cMax - cStart);                        /* Every face is split into 4 faces, and 12 faces are added for each cell */
    depthSize[3] = 8*(cMax - cStart);                                             /* Every cell split into 8 cells */
    /* Quadrilateral Prisms */
    depthSize[0] += 0;                                                            /* No hybrid vertices */
    depthSize[1] +=   (eEnd - eMax)   +   (fEnd - fMax)   +   (cEnd - cMax);      /* Every hybrid edge remains, 1 edge for every hybrid face and hybrid cell */
    depthSize[2] += 2*(fEnd - fMax)   + 4*(cEnd - cMax);                          /* Every hybrid face split into 2 faces and 4 faces are added for each hybrid cell */
    depthSize[3] += 4*(cEnd - cMax);                                              /* Every hybrid cell split into 4 cells */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  PetscFunctionReturn(0);
}

/* Return triangle edge for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetTriEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 2-(o+r) : o+r)%3;
}
PETSC_STATIC_INLINE PetscInt GetTriEdgeInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 2-(o+s) : 3+s-o)%3;
}

/* Return triangle subface for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetTriSubface_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 3-(o+r) : o+r)%3;
}
PETSC_STATIC_INLINE PetscInt GetTriSubfaceInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 3-(o+s) : 3+s-o)%3;
}

/* Return the interior edge number connecting the midpoints of the triangle edges r
   and r+1 in the transitive closure for triangle orientation o */
PETSC_STATIC_INLINE PetscInt GetTriMidEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 1-(o+r) : o+r)%3;
}
PETSC_STATIC_INLINE PetscInt GetTriMidEdgeInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 1-(o+s) : 3+s-o)%3;
}

/* Return the interior edge number connecting the midpoint of the triangle edge r
   (in the transitive closure) and the vertex in the interior of the face for triangle orientation o */
PETSC_STATIC_INLINE PetscInt GetTriInteriorEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 2-(o+r) : o+r)%3;
}
PETSC_STATIC_INLINE PetscInt GetTriInteriorEdgeInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 2-(o+s) : 3+s-o)%3;
}

/* Return quad edge for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetQuadEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 3-(o+r) : o+r)%4;
}
PETSC_STATIC_INLINE PetscInt GetQuadEdgeInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 3-(o+s) : 4+s-o)%4;
}

/* Return quad subface for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetQuadSubface_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 4-(o+r) : o+r)%4;
}
PETSC_STATIC_INLINE PetscInt GetQuadSubfaceInverse_Static(PetscInt o, PetscInt s) {
  return (o < 0 ? 4-(o+s) : 4+s-o)%4;
}

static PetscErrorCode DMLabelSetStratumBounds(DMLabel label, PetscInt value, PetscInt cStart, PetscInt cEnd)
{
  IS             cIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cIS);CHKERRQ(ierr);
  ierr = DMLabelSetStratumIS(label, value, cIS);CHKERRQ(ierr);
  ierr = ISDestroy(&cIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerSetConeSizes(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscInt       depth, cStart, cStartNew, cEnd, cEndNew, cMax, c, vStart, vStartNew, vEnd, vEndNew, vMax, v, fStart, fStartNew, fEnd, fEndNew, fMax, f, eStart, eStartNew, eEnd, eEndNew, eMax, e, r;
  DMLabel        depthLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);
  ierr = GetDepthEnd_Private(depth, depthSize, &cEndNew, &fEndNew, &eEndNew, &vEndNew);CHKERRQ(ierr);
  ierr = DMCreateLabel(rdm,"depth");CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(rdm,&depthLabel);CHKERRQ(ierr);
  ierr = DMLabelSetStratumBounds(depthLabel, 0, vStartNew, vEndNew);CHKERRQ(ierr);
  if (depth > 2) ierr = DMLabelSetStratumBounds(depthLabel, 1, eStartNew, eEndNew);CHKERRQ(ierr);
  if (depth > 1) ierr = DMLabelSetStratumBounds(depthLabel, depth - 1, fStartNew, fEndNew);CHKERRQ(ierr);
  if (depth > 0) ierr = DMLabelSetStratumBounds(depthLabel, depth, cStartNew, cEndNew);CHKERRQ(ierr);
  {
    DM_Plex *plex = (DM_Plex *) rdm->data;
    ierr = PetscObjectStateGet((PetscObject) depthLabel, &plex->depthState);CHKERRQ(ierr);
  }
  if (!refiner) PetscFunctionReturn(0);
  switch (refiner) {
  case REFINER_SIMPLEX_1D:
    /* All cells have 2 vertices */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*2 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Cell vertices have support 2 */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
    }
    break;
  case REFINER_SIMPLEX_2D:
    /* All cells have 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + cells*2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size*2);CHKERRQ(ierr);
    }
    break;
  case REFINER_SIMPLEX_TO_HEX_2D:
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Split-face vertices have cells + 2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size + 2);CHKERRQ(ierr);
    }
    /* Interior vertices have 3 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      ierr = DMPlexSetSupportSize(rdm, newp, 3);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    /* the mesh is no longer hybrid */
    cMax = PetscMin(cEnd, cMax);
    /* All cells have 4 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = (cMax - cStart)*3 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid interior faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Split-face vertices have cells + 2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size + 2);CHKERRQ(ierr);
    }
    /* Interior vertices have 3 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      ierr = DMPlexSetSupportSize(rdm, newp, 3);CHKERRQ(ierr);
    }
    /* Hybrid interior vertices have 4 supports */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case REFINER_HEX_2D:
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_SIMPLEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + (cMax - cStart)*4, fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 3 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid cell faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (c - cMax);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + (2 interior, 1 hybrid) supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *support;
      PetscInt       size, newSize = 2, s;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] >= cMax) newSize += 1;
        else newSize += 2;
      }
      ierr = DMPlexSetSupportSize(rdm, newp, newSize);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + (cMax - cStart)*4, fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 4 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid cell faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (fEnd - fMax) + (c - cMax);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fMax - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case REFINER_SIMPLEX_3D:
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 3 edges and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 3 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Split edges have 2 vertices and the same faces */
    for (e = eStart; e < eEnd; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        const PetscInt *cone, *ornt, *support, eint[4] = {1, 0, 2, 0};
        PetscInt        coneSize, c, supportSize, s, er, intFaces = 0;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
          er = GetTriMidEdgeInverse_Static(ornt[c], r);
          if (er == eint[c]) {
            intFaces += 1;
          } else {
            intFaces += 2;
          }
        }
        ierr = DMPlexSetSupportSize(rdm, newp, 2+intFaces);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Edge vertices have 2 + faces*2 + cells*0/1 supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size, *star = NULL, starSize, s, cellSize = 0;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cEnd)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) ++cellSize;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size*2 + cellSize);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_SIMPLEX_3D:
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + 8*(cMax-cStart), fStartNew + 4*(fMax - fStart) + 8*(cMax - cStart),
                                 eStartNew + 2*(eMax - eStart) + 3*(fMax - fStart) + (cMax - cStart), PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 4 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 5 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 5);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 3 edges and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 3 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid split faces have 4 edges and the same cells as the parent */
    for (f = fMax; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Interior split edges have 2 vertices and the same faces */
    for (e = eStart; e < eMax; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        const PetscInt *cone, *ornt, *support, eint[4] = {1, 0, 2, 0};
        PetscInt        coneSize, c, supportSize, s, er, intFaces = 0;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          if (support[s] < cMax) {
            /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
            er = GetTriMidEdgeInverse_Static(ornt[c], r);
            if (er == eint[c]) {
              intFaces += 1;
            } else {
              intFaces += 2;
            }
          } else {
            intFaces += 1;
          }
        }
        ierr = DMPlexSetSupportSize(rdm, newp, 2+intFaces);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    /* Hybrid edges have 2 vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (e - eMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid face edges have 2 vertices and 2+2*cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2+2*size);CHKERRQ(ierr);
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Interior edge vertices have 2 + interior face*2 + hybrid face + cells*0/1 supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *support;
      PetscInt        size, *star = NULL, starSize, s, faceSize = 0, cellSize = 0;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] < fMax) faceSize += 2;
        else                   faceSize += 1;
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cMax)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) ++cellSize;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + faceSize + cellSize);CHKERRQ(ierr);
    }
    break;
  case REFINER_SIMPLEX_TO_HEX_3D:
    /* All cells have 6 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Split faces have 4 edges and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*3 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*3 + (c - cStart)*6 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Split edges have 2 vertices and the same faces */
    for (e = eStart; e < eEnd; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Face edges have 2 vertices and 2 + cell faces supports */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Face vertices have 3 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + f - fStart;
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 3 + size);CHKERRQ(ierr);
    }
    /* Interior cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + fEnd - fStart + c - cStart;

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
    /* the mesh is no longer hybrid */
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
    eMax = PetscMin(eEnd, eMax);
    /* All cells have 6 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 4 edges and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*3 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 4 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*3 + (c - cStart)*6 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid split faces have 4 edges and the same cells as the parent */
    for (f = fMax; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (f - fMax)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Hybrid cell faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Interior split edges have 2 vertices and the same faces */
    for (e = eStart; e < eMax; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior face edges have 2 vertices and 2 + cell faces supports */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 3 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Hybrid edges have 2 vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (e - eMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid face edges have 2 vertices and 2+cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (f - fMax);
      PetscInt        size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
    }
    /* Hybrid cell edges have 2 vertices and 3 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + (c - cMax);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 3);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Interior edge vertices have 2 + faces supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Interior face vertices have 3 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + f - fStart;
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 3 + size);CHKERRQ(ierr);
    }
    /* Interior cell vertices have 4 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + c - cStart;

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case REFINER_HEX_3D:
    /* All cells have 6 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Split faces have 4 edges and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 12; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Split edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eEnd; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
      }
    }
    /* Cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Face vertices have 4 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4 + size);CHKERRQ(ierr);
    }
    /* Cell vertices have 6 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 6);CHKERRQ(ierr);
    }
    break;
  case REFINER_HYBRID_HEX_3D:
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + 8*(cMax-cStart), fStartNew + 4*(fMax - fStart) + 12*(cMax - cStart),
                                 eStartNew + 2*(eMax - eStart) + 4*(fMax - fStart) + 6*(cMax - cStart), PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 6 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 6 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 4 edges and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 4 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 12; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid split faces have 4 edges and the same cells as the parent */
    for (f = fMax; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (f - fMax)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Interior split edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eMax; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (c - cStart)*6 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Hybrid edges have 2 vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (e - eMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid face edges have 2 vertices and 2+cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
    }
    /* Hybrid cell edges have 2 vertices and 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (fEnd - fMax) + (c - cMax);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Interior edge vertices have 2 + faces supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Interior face vertices have 4 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4 + size);CHKERRQ(ierr);
    }
    /* Interior cell vertices have 6 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 6);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerSetCones(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  const PetscInt *faces, cellInd[4] = {0, 1, 2, 3};
  PetscInt        cStart,    cEnd,    cMax,    vStart,    vEnd, vMax, fStart,    fEnd,    fMax,    eStart,    eEnd,    eMax;
  PetscInt        cStartNew, cEndNew, cMaxNew, vStartNew, vEndNew,    fStartNew, fEndNew, fMaxNew, eStartNew, eEndNew, eMaxNew;
  PetscInt        depth, maxSupportSize, *supportRef, c, f, e, v, r;
#if defined(PETSC_USE_DEBUG)
  PetscInt        p;
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!refiner) PetscFunctionReturn(0);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);
  ierr = GetDepthEnd_Private(depth, depthSize, &cEndNew, &fEndNew, &eEndNew, &vEndNew);CHKERRQ(ierr);
  switch (refiner) {
  case REFINER_SIMPLEX_1D:
    /* Max support size of refined mesh is 2 */
    ierr = PetscMalloc1(2, &supportRef);CHKERRQ(ierr);
    /* All cells have 2 vertices */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (c - cStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*2 + r;
        const PetscInt *cone;
        PetscInt        coneNew[2];

        ierr             = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < cStartNew) || (newp >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp, cStartNew, cEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = cStartNew + (support[s] - cStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Cell vertices have support of 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (c - cStart);

      supportRef[0] = cStartNew + (c - cStart)*2 + 0;
      supportRef[1] = cStartNew + (c - cStart)*2 + 1;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_SIMPLEX_2D:
    /*
     2
     |\
     | \
     |  \
     |   \
     | C  \
     |     \
     |      \
     2---1---1
     |\  D  / \
     | 2   0   \
     |A \ /  B  \
     0---0-------1
     */
    /* All cells have 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C triangle */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D triangle */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r]       - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - fStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + (r+1)%3;
        supportNew[1] = (c - cStart)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + cells*2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if      (cone[1] == f) r = 1;
        else if (cone[2] == f) r = 2;
        supportRef[2+s*2+0] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + (r+2)%3;
        supportRef[2+s*2+1] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size*2; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_SIMPLEX_TO_HEX_2D:
    /*
     2
     |\
     | \
     |  \
     |   \
     | C  \
     |     \
     2      1
     |\    / \
     | 2  1   \
     |  \/     \
     |   |      \
     |A  |   B   \
     |   0        \
     |   |         \
     0---0----------1
     */
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[3] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[0] = ornt[1];
      coneNew[1] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[1] = ornt[2];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*3 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (c - cStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*3 + r%3;
        supportNew[1] = (c - cStart)*3 + (r+1)%3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split-face vertices have cells + 2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if      (cone[1] == f) r = 1;
        else if (cone[2] == f) r = 2;
        supportRef[2+s+0] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Interior vertices have 3 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      supportRef[0] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 0;
      supportRef[1] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 1;
      supportRef[2] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 2;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[3] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[0] = ornt[1];
      coneNew[1] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[1] = ornt[2];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /*
     2---------1---------3
     |         |         |
     |    D    1    C    |
     |         |         |
     2----2----0----3----3
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    /* Parent cells are input as prisms but children are quads, since the mesh is no longer hybrid */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp  = cStartNew + (cMax - cStart)*3 + (c - cMax)*4;
      const PetscInt  newpt = (cMax - cStart)*3 + (c - cMax)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + newpt + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + newpt + 2;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[3] = ornt[2] < 0 ? 0 : -2;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 1 : 0);
      orntNew[1] = ornt[3];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + newpt + 3;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + newpt + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + newpt + 3;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 0 : 1);
      orntNew[1] = ornt[3];
      coneNew[2] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + newpt + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + newpt + 2;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + newpt + 1;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      coneNew[3] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[3] = ornt[2] < 0 ? 0 : -2;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          const PetscInt p2q[4][2] = { {0, 1},
                                       {3, 2},
                                       {0, 3},
                                       {1, 2} };

          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          if (coneSize == 3)      supportRef[s] = cStartNew + (support[s] - cStart)*3 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
          else if (coneSize == 4) supportRef[s] = cStartNew + (cMax - cStart)*3 + (support[s] - cMax)*4 + (ornt[c] < 0 ? p2q[c][(r+1)%2] : p2q[c][r]);
          else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected cone size %D", coneSize);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (c - cStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*3 + r%3;
        supportNew[1] = (c - cStart)*3 + (r+1)%3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Hybrid interior faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (cMax - cStart) + (c - cMax);
	ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        if (r==0) {
          supportNew[0] = (cMax - cStart)*3 + (c - cMax)*4 + 0;
          supportNew[1] = (cMax - cStart)*3 + (c - cMax)*4 + 1;
        } else if (r==1) {
          supportNew[0] = (cMax - cStart)*3 + (c - cMax)*4 + 2;
          supportNew[1] = (cMax - cStart)*3 + (c - cMax)*4 + 3;
        } else if (r==2) {
          supportNew[0] = (cMax - cStart)*3 + (c - cMax)*4 + 0;
          supportNew[1] = (cMax - cStart)*3 + (c - cMax)*4 + 3;
        } else {
          supportNew[0] = (cMax - cStart)*3 + (c - cMax)*4 + 1;
          supportNew[1] = (cMax - cStart)*3 + (c - cMax)*4 + 2;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split-face vertices have cells + 2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0, coneSize;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        if (coneSize == 3) {
          if      (cone[1] == f) r = 1;
          else if (cone[2] == f) r = 2;
          supportRef[2+s] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + r;
        } else if (coneSize == 4) {
          if      (cone[1] == f) r = 1;
          else if (cone[2] == f) r = 2;
          else if (cone[3] == f) r = 3;
          supportRef[2+s] = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (support[s] - cMax)*4 + r;
        } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected cone size %D", coneSize);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Interior vertices have 3 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      supportRef[0] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 0;
      supportRef[1] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 1;
      supportRef[2] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + 2;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
    }
    /* Hybrid interior vertices have 4 supports */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + c - cStart;

      supportRef[0] = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + 0;
      supportRef[1] = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + 1;
      supportRef[2] = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + 2;
      supportRef[3] = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (c - cMax)*4 + 3;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_HEX_2D:
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 0 : 1);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 1 : 0);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%4 : (c+r)%4);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;

	if (r==1 || r==2) {
          coneNew[0] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (c - cStart);
          coneNew[1] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
	} else {
          coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
          coneNew[1] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (c - cStart);
	}
	ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + r;
        supportNew[1] = (c - cStart)*4 + (r+1)%4;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if      (cone[1] == f) r = 1;
        else if (cone[2] == f) r = 2;
        else if (cone[3] == f) r = 3;
        supportRef[2+s] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (c - cStart);
      PetscInt       supportNew[4];

      for (r = 0; r < 4; ++r) {
        supportNew[r] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_SIMPLEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    ierr = DMPlexGetHybridBounds(rdm, &cMaxNew, &fMaxNew, NULL, NULL);CHKERRQ(ierr);
    /* Interior cells have 3 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+0, cStartNew, cMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+1, cStartNew, cMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C triangle */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+2, cStartNew, cMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D triangle */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+3, cStartNew, cMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /*
     2----3----3
     |         |
     |    B    |
     |         |
     0----4--- 1
     |         |
     |    A    |
     |         |
     0----2----1
     */
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4], r;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      r    = (ornt[0] < 0 ? 1 : 0);
      /* A quad */
      coneNew[0]   = fStartNew + (cone[0] - fStart)*2 + r;
      orntNew[0]   = ornt[0];
      coneNew[1]   = fStartNew + (cone[1] - fStart)*2 + r;
      orntNew[1]   = ornt[1];
      coneNew[2+r] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (cone[2+r] - fMax);
      orntNew[2+r] = 0;
      coneNew[3-r] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (fEnd    - fMax) + (c - cMax);
      orntNew[3-r] = 0;
      ierr = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0]   = fStartNew + (cone[0] - fStart)*2 + 1-r;
      orntNew[0]   = ornt[0];
      coneNew[1]   = fStartNew + (cone[1] - fStart)*2 + 1-r;
      orntNew[1]   = ornt[1];
      coneNew[2+r] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (fEnd    - fMax) + (c - cMax);
      orntNew[2+r] = 0;
      coneNew[3-r] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (cone[3-r] - fMax);
      orntNew[3-r] = 0;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (cone[c] == f) break;
          if (support[s] >= cMax) {
            supportRef[s] = cStartNew + (cMax - cStart)*4 + (support[s] - cMax)*2 + (ornt[c] < 0 ? 1-r : r);
          } else {
            supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r]       - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - fStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + (r+1)%3;
        supportNew[1] = (c - cStart)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (f - fMax);
      const PetscInt *cone, *ornt;
      const PetscInt *support;
      PetscInt        coneNew[2];
      PetscInt        supportNew[2];
      PetscInt        size, s, r;

      ierr       = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
        for (r = 0; r < 2; ++r) {
          if (cone[r+2] == f) break;
        }
        supportNew[s] = (cMax - cStart)*4 + (support[s] - cMax)*2 + (ornt[0] < 0 ? 1-r : r);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Cell hybrid faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (c - cMax);
      const PetscInt *cone;
      PetscInt        coneNew[2];
      PetscInt        supportNew[2];

      ierr       = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - fStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - fStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = (cMax - cStart)*4 + (c - cMax)*2 + 0;
      supportNew[1] = (cMax - cStart)*4 + (c - cMax)*2 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] >= fMax) {
          supportRef[s] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (support[s] - fMax);
        } else {
          PetscInt r = 0;

          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          if (cone[1] == v) r = 1;
          supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + (2 interior, 1 hybrid) supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, newSize = 2, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (support[s] >= cMax) {
          supportRef[newSize+0] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (support[s] - cMax);

          newSize += 1;
        } else {
          if      (cone[1] == f) r = 1;
          else if (cone[2] == f) r = 2;
          supportRef[newSize+0] = fStartNew + (fMax - fStart)*2 + (support[s] - cStart)*3 + (r+2)%3;
          supportRef[newSize+1] = fStartNew + (fMax - fStart)*2 + (support[s] - cStart)*3 + r;

          newSize += 2;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < newSize; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_HEX_2D:
    /* Hybrid Hex 2D */
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    ierr = DMPlexGetHybridBounds(rdm, &cMaxNew, &fMaxNew, NULL, NULL);CHKERRQ(ierr);
    /* Interior cells have 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 0 : 1);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+0, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+1, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+2, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D quad */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 1 : 0);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior cell [%D, %D)", newp+3, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /*
     2----3----3
     |         |
     |    B    |
     |         |
     0----4--- 1
     |         |
     |    A    |
     |         |
     0----2----1
     */
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*4 + (cone[2] - fMax);
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*4 + (fEnd    - fMax) + (c - cMax);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*4 + (fEnd    - fMax) + (c - cMax);
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*4 + (cone[3] - fMax);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          if (support[s] >= cMax) {
            supportRef[s] = cStartNew + (cMax - cStart)*4 + (support[s] - cMax)*2 + r;
          } else {
            ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
            ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
            for (c = 0; c < coneSize; ++c) {
              if (cone[c] == f) break;
            }
            supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%4 : (c+r)%4);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*4 + r;
        PetscInt       coneNew[2], supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (fMax    - fStart) + (c - cStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + r;
        supportNew[1] = (c - cStart)*4 + (r+1)%4;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (f - fMax);
      const PetscInt *cone, *support;
      PetscInt        coneNew[2], supportNew[2];
      PetscInt        size, s, r;

      ierr       = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 2; ++r) {
          if (cone[r+2] == f) break;
        }
        supportNew[s] = (cMax - cStart)*4 + (support[s] - cMax)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Cell hybrid faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (fEnd - fMax) + (c - cMax);
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[2];

      ierr       = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - fStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - fStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = (cMax - cStart)*4 + (c - cMax)*2 + 0;
      supportNew[1] = (cMax - cStart)*4 + (c - cMax)*2 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] >= fMax) {
          supportRef[s] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (support[s] - fMax);
        } else {
          PetscInt r = 0;

          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          if (cone[1] == v) r = 1;
          supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (support[s] >= cMax) {
          supportRef[2+s] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (fEnd - fMax) + (support[s] - cMax);
        } else {
          if      (cone[1] == f) r = 1;
          else if (cone[2] == f) r = 2;
          else if (cone[3] == f) r = 3;
          supportRef[2+s] = fStartNew + (fMax - fStart)*2 + (support[s] - cStart)*4 + r;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fMax - fStart) + (c - cStart);
      PetscInt       supportNew[4];

      for (r = 0; r < 4; ++r) {
        supportNew[r] = fStartNew + (fMax - fStart)*2 + (c - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_SIMPLEX_3D:
    /* All cells have 4 faces: Tet face order is prescribed in DMPlexGetFaces_Internal() */
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A tetrahedron: {0, a, c, d} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 0); /* A */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 0); /* A */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 0); /* A */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B tetrahedron: {a, 1, b, e} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 2); /* C */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 1); /* B */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C tetrahedron: {c, b, 2, f} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 2); /* C */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 1); /* B */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 0); /* A */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D tetrahedron: {d, e, f, 3} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 1); /* B */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 2); /* C */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 2); /* C */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* A' tetrahedron: {c, d, a, f} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[2] - fStart)*4 + 3;
      orntNew[1] = ornt[2] < 0 ? -(GetTriMidEdge_Static(ornt[2], 0)+1) : GetTriMidEdge_Static(ornt[2], 0);
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[3] = 2;
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+4 < cStartNew) || (newp+4 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+4, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B' tetrahedron: {e, b, a, f} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[3] - fStart)*4 + 3;
      orntNew[1] = ornt[3] < 0 ? -(GetTriMidEdge_Static(ornt[3], 1)+1) : GetTriMidEdge_Static(ornt[3], 1);
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+5 < cStartNew) || (newp+5 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+5, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C' tetrahedron: {f, a, c, b} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[3] = ornt[0] < 0 ? -(GetTriMidEdge_Static(ornt[0], 2)+1) : GetTriMidEdge_Static(ornt[0], 2);
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+6 < cStartNew) || (newp+6 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+6, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D' tetrahedron: {f, a, e, d} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[1] = -1;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[3] = ornt[1] < 0 ? -(GetTriMidEdge_Static(ornt[1], 1)+1) : GetTriMidEdge_Static(ornt[1], 1);
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+7 < cStartNew) || (newp+7 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+7, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 3 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*3, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*4;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[3], orntNew[3], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < fStartNew) || (newp+0 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+0, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C triangle */
      coneNew[0] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < fStartNew) || (newp+2 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+2, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* D triangle */
      coneNew[0] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < fStartNew) || (newp+3 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+3, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        for (s = 0; s < supportSize; ++s) {
          PetscInt subf;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          subf = GetTriSubfaceInverse_Static(ornt[c], r);
          supportRef[s] = cStartNew + (support[s] - cStart)*8 + (r==3 ? (c+2)%4 + 4 : faces[c*3+subf]);
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fStartNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+r, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 3 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face A: {c, a, d} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 2);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 2);
      orntNew[1] = ornt[1] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 2);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0;
      supportNew[1] = (c - cStart)*8 + 0+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face B: {a, b, e} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 0);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 0);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 1);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1;
      supportNew[1] = (c - cStart)*8 + 1+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face C: {c, f, b} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 0);
      orntNew[0] = ornt[2] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 2);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 1);
      orntNew[2] = ornt[0] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 2;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face D: {d, e, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 0);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 1);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 1);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 3;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face E: {d, f, a} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 1);
      orntNew[0] = ornt[2] < 0 ? 0 : -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 2);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face F: {c, a, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 2);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 0);
      orntNew[2] = ornt[2] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face G: {e, a, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 1);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 1);
      orntNew[2] = ornt[3] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face H: {a, b, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 0);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 2);
      orntNew[1] = ornt[3] < 0 ? 0 : -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[2] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
    }
    /* Split Edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          supportRef[s] = fStartNew + (support[s] - fStart)*4 + (c + (ornt[c] < 0 ? 1-r : r))%3;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2], intFaces = 0, er, eint[4] = {1, 0, 2, 0};
        PetscInt        fint[24] = { 1,  7, -1, -1,  0,  5,
                                    -1, -1,  1,  6,  0,  4,
                                     2,  5,  3,  4, -1, -1,
                                    -1, -1,  3,  6,  2,  7};

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[(r+0)%3] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - eStart);
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*4 + (r+1)%3;
        supportRef[1] = fStartNew + (f - fStart)*4 + 3;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
          er = GetTriMidEdgeInverse_Static(ornt[c], r);
          if (er == eint[c]) {
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + (c + 2)%4;
          } else {
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 0];
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 1];
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < intFaces; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      const PetscInt *cone, *ornt, *fcone;
      PetscInt        coneNew[2], supportNew[4], find;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cone[0], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[0], 0);
      coneNew[0] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexGetCone(dm, cone[2], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[2], 1);
      coneNew[1] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 4;
      supportNew[1] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 5;
      supportNew[2] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 6;
      supportNew[3] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 7;
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 4; ++p) {
        if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Edge vertices have 2 + face*2 + 0/1 supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt       *star = NULL, starSize, cellSize = 0, coneSize, size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
        supportRef[2+s*2+0] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*3 + (r+0)%3;
        supportRef[2+s*2+1] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*3 + (r+2)%3;
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cEnd)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) {supportRef[2+size*2+cellSize++] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (star[s] - cStart);}
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size*2+cellSize; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_SIMPLEX_3D:
    ierr = DMPlexGetHybridBounds(rdm, &cMaxNew, &fMaxNew, &eMaxNew, NULL);CHKERRQ(ierr);
    /* Interior cells have 4 faces: Tet face order is prescribed in DMPlexGetFaces_Internal() */
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A tetrahedron: {0, a, c, d} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 0); /* A */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 0); /* A */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 0); /* A */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B tetrahedron: {a, 1, b, e} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 2); /* C */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 1); /* B */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C tetrahedron: {c, b, 2, f} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 2); /* C */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 1); /* B */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 0); /* A */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D tetrahedron: {d, e, f, 3} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 1); /* B */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 2); /* C */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 2); /* C */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* A' tetrahedron: {d, a, c, f} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[2] - fStart)*4 + 3;
      orntNew[1] = ornt[2] < 0 ? -(GetTriMidEdge_Static(ornt[2], 0)+1) : GetTriMidEdge_Static(ornt[2], 0);
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[3] = 2;
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+4 < cStartNew) || (newp+4 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+4, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B' tetrahedron: {e, b, a, f} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[1] = 1;
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + 3;
      orntNew[3] = ornt[3] < 0 ? -(GetTriMidEdge_Static(ornt[3], 0)+1) : GetTriMidEdge_Static(ornt[3], 0);
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+5 < cStartNew) || (newp+5 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+5, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C' tetrahedron: {b, f, c, a} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[1] = ornt[0] < 0 ? -(GetTriMidEdge_Static(ornt[0], 2)+1) : GetTriMidEdge_Static(ornt[0], 2);
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[2] = -3;
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+6 < cStartNew) || (newp+6 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+6, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D' tetrahedron: {f, e, d, a} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[1] = -3;
      coneNew[2] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[2] = ornt[1] < 0 ? -(GetTriMidEdge_Static(ornt[1], 0)+1) : GetTriMidEdge_Static(ornt[1], 0);
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[3] = -3;
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+7 < cStartNew) || (newp+7 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+7, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /* Hybrid cells have 5 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*8 + (c - cMax)*4;
      const PetscInt *cone, *ornt, *fornt;
      PetscInt        coneNew[5], orntNew[5], o, of, i;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, cone[0], &fornt);CHKERRQ(ierr);
      o = ornt[0] < 0 ? -1 : 1;
      for (r = 0; r < 3; ++r) {
        coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], r);
        orntNew[0] = ornt[0];
        coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], r);
        orntNew[1] = ornt[1];
        of = fornt[GetTriEdge_Static(ornt[0], r)]       < 0 ? -1 : 1;
        i  = GetTriEdgeInverse_Static(ornt[0], r)       + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (cone[2+GetTriEdge_Static(ornt[0], r)]       - fMax)*2 + (o*of < 0 ? 1 : 0);
        orntNew[i] = 0;
        i  = GetTriEdgeInverse_Static(ornt[0], (r+1)%3) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + GetTriSubface_Static(ornt[0], r);
        orntNew[i] = 0;
        of = fornt[GetTriEdge_Static(ornt[0], (r+2)%3)] < 0 ? -1 : 1;
        i  = GetTriEdgeInverse_Static(ornt[0], (r+2)%3) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (cone[2+GetTriEdge_Static(ornt[0], (r+2)%3)] - fMax)*2 + (o*of < 0 ? 0 : 1);
        orntNew[i] = 0;
        ierr = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < cMaxNew) || (newp+r >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", newp+r, cMaxNew, cEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
        }
        for (p = 2; p < 5; ++p) {
          if ((coneNew[p] < fMaxNew)   || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", coneNew[p], fMaxNew, fEndNew);
        }
#endif
      }
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 2;
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 0;
      orntNew[4] = 0;
      ierr = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cMaxNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", newp+3, cMaxNew, cEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
      for (p = 2; p < 5; ++p) {
        if ((coneNew[p] < fMaxNew)   || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", coneNew[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Split faces have 3 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*4;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[3], orntNew[3], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < fStartNew) || (newp+0 >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+0, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* B triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* C triangle */
      coneNew[0] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < fStartNew) || (newp+2 >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+2, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* D triangle */
      coneNew[0] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < fStartNew) || (newp+3 >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+3, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        for (s = 0; s < supportSize; ++s) {
          PetscInt subf;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          subf = GetTriSubfaceInverse_Static(ornt[c], r);
          if (support[s] < cMax) {
            supportRef[s] = cStartNew + (support[s] - cStart)*8 + (r==3 ? (c+2)%4 + 4 : faces[c*3+subf]);
          } else {
            supportRef[s] = cStartNew + (cMax - cStart)*8 + (support[s] - cMax)*4 + (r==3 ? r : subf);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fStartNew) || (newp+r >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+r, fStartNew, fMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior or hybrid cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 3 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face A: {c, a, d} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 2);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 2);
      orntNew[1] = ornt[1] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 2);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0;
      supportNew[1] = (c - cStart)*8 + 0+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face B: {a, b, e} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 0);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 0);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 1);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1;
      supportNew[1] = (c - cStart)*8 + 1+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face C: {c, f, b} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 0);
      orntNew[0] = ornt[2] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 2);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 1);
      orntNew[2] = ornt[0] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 2;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face D: {d, e, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 0);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 1);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 1);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 3;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face E: {d, f, a} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 1);
      orntNew[0] = ornt[2] < 0 ? 0 : -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 2);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face F: {c, a, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 2);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriMidEdge_Static(ornt[2], 0);
      orntNew[2] = ornt[2] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face G: {e, a, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriMidEdge_Static(ornt[1], 1);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 1);
      orntNew[2] = ornt[3] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face H: {a, b, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriMidEdge_Static(ornt[0], 0);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriMidEdge_Static(ornt[3], 2);
      orntNew[1] = ornt[3] < 0 ? 0 : -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[2] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
    }
    /* Hybrid split faces have 4 edges and same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2], size, s, c;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + r;

        coneNew[0]   = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1-r : r);
        orntNew[0]   = ornt[0];
        coneNew[1]   = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1-r : r);
        orntNew[1]   = ornt[1];
        coneNew[2+r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (cone[2+r] - eMax);
        orntNew[2+r] = 0;
        coneNew[3-r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd      - eMax) + (f - fMax);
        orntNew[3-r] = 0;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        for (s = 0; s < size; ++s) {
          const PetscInt *coneCell, *orntCell, *fornt;
          PetscInt        o, of;

          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          o = orntCell[0] < 0 ? -1 : 1;
          for (c = 2; c < 5; ++c) if (coneCell[c] == f) break;
          if (c >= 5) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of cell %D", f, support[s]);
          ierr = DMPlexGetConeOrientation(dm, coneCell[0], &fornt);CHKERRQ(ierr);
          of = fornt[c-2] < 0 ? -1 : 1;
          supportNew[s] = cStartNew + (cMax - cStart)*8 + (support[s] - cMax)*4 + (GetTriEdgeInverse_Static(orntCell[0], c-2) + (o*of < 0 ? 1-r : r))%3;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp, fMaxNew, fEndNew);
        for (p = 0; p < size; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Hybrid cell faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (r+2)%3;
        orntNew[0] = 0;
        coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (r+2)%3;
        orntNew[1] = 0;
        coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (cone[2+(r+2)%3] - fMax);
        orntNew[2] = 0;
        coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (cone[2+r]       - fMax);
        orntNew[3] = 0;
        ierr = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        supportNew[0] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + GetTriSubface_Static(ornt[0], r);
        supportNew[1] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp+r, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Interior split edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (cone[c] == e) break;
          if (support[s] < fMax) {
            supportRef[s] = fStartNew + (support[s] - fStart)*4 + (c + (ornt[c] < 0 ? 1-r : r))%3;
          } else {
            supportRef[s] = fStartNew + (fMax       - fStart)*4 + (cMax - cStart)*8 + (support[s] - fMax)*2 + (ornt[c] < 0 ? 1-r : r);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior or hybrid face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2], intFaces = 0, er, eint[4] = {1, 0, 2, 0};
        PetscInt        fint[24] = { 1,  7, -1, -1,  0,  5,
                                    -1, -1,  1,  6,  0,  4,
                                     2,  5,  3,  4, -1, -1,
                                    -1, -1,  3,  6,  2,  7};

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[(r+0)%3] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - eStart);
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*4 + (r+1)%3;
        supportRef[1] = fStartNew + (f - fStart)*4 + 3;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          if (support[s] < cMax) {
            /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
            er = GetTriMidEdgeInverse_Static(ornt[c], r);
            if (er == eint[c]) {
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + (c + 2)%4;
            } else {
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 0];
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 1];
            }
          } else {
            supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + (r + 1)%3;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < intFaces; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior or hybrid face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      const PetscInt *cone, *ornt, *fcone;
      PetscInt        coneNew[2], supportNew[4], find;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cone[0], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[0], 0);
      coneNew[0] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexGetCone(dm, cone[2], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[2], 1);
      coneNew[1] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 4;
      supportNew[1] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 5;
      supportNew[2] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 6;
      supportNew[3] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 7;
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((supportNew[p] < fStartNew) || (supportNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /* Hybrid edges have two vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (e - eMax);
      const PetscInt *cone, *support, *fcone;
      PetscInt        coneNew[2], size, fsize, s;

      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &fsize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &fcone);CHKERRQ(ierr);
        for (c = 0; c < fsize; ++c) if (fcone[c] == e) break;
        if ((c < 2) || (c > 3)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D not found in cone of face %D", e, support[s]);
        supportRef[s] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (support[s] - fMax)*2 + c-2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Hybrid face edges have 2 vertices and 2+2*cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (f - fMax);
      const PetscInt *cone, *support, *ccone, *cornt;
      PetscInt        coneNew[2], size, csize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - eStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportRef[0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + 0;
      supportRef[1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + 1;
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &csize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &ccone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[s], &cornt);CHKERRQ(ierr);
        for (c = 0; c < csize; ++c) if (ccone[c] == f) break;
        if ((c < 2) || (c >= csize)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hybrid face %D is not in cone of hybrid cell %D", f, support[s]);
        supportRef[2+s*2+0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + c-2;
        supportRef[2+s*2+1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + (c-1)%3;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2+size*2; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        if (support[s] < eMax) supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
        else                   supportRef[s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (support[s] - eMax);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior or hybrid edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior edge vertices have 2 + interior face*2 + hybrid face + cells*0/1 supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt       *star = NULL, starSize, faceSize = 0, cellSize = 0, coneSize, size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        if (support[s] < fMax) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
          supportRef[2+faceSize+0] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*3 + (r+0)%3;
          supportRef[2+faceSize+1] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*3 + (r+2)%3;
          faceSize += 2;
        } else {
          supportRef[2+faceSize+0] = eStartNew + (eMax - eStart)*2 + (fMax       - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (support[s] - fMax);
          ++faceSize;
        }
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cMax)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) {supportRef[2+faceSize+cellSize++] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (star[s] - cStart);}
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+faceSize+cellSize; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an interior or hybrid edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case REFINER_SIMPLEX_TO_HEX_3D:
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    /* All cells have 6 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6];
      PetscInt        orntNew[6];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 0); /* B */
      orntNew[0] = ornt[0] < 0 ? -1 : 1;
      coneNew[1] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 3;               /* T */
      orntNew[1] = -4;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 0); /* F */
      orntNew[2] = ornt[2] < 0 ? -1 : 1;
      coneNew[3] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 0;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 2;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 0); /* L */
      orntNew[5] = ornt[1] < 0 ? -1 : 1;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 4;               /* T */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 0;               /* F */
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 1); /* K */
      orntNew[3] = ornt[3] < 0 ? -2 : 0;
      coneNew[4] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 1;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 2); /* L */
      orntNew[5] = ornt[1] < 0 ? -4 : 2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 2); /* B */
      orntNew[0] = ornt[0] < 0 ? -4 : 2;
      coneNew[1] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 5;               /* T */
      orntNew[1] = -4;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 1); /* F */
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 1;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 0); /* R */
      orntNew[4] = ornt[3] < 0 ? -1 : 1;
      coneNew[5] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 2;               /* L */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 3;               /* B */
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 2); /* T */
      orntNew[1] = ornt[3] < 0 ? -1 : 1;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 2); /* F */
      orntNew[2] = ornt[2] < 0 ? -4 : 2;
      coneNew[3] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 4;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (fEnd    - fStart)*3 + (c - cStart)*6 + 5;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 1); /* L */
      orntNew[5] = ornt[1] < 0 ? -2 : 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 4 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*3;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[0] = ornt[2];
      coneNew[1] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[1] = ornt[0];
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < fStartNew) || (newp+0 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+0, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[0] = ornt[1];
      coneNew[1] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[1] = ornt[2];
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < fStartNew) || (newp+2 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+2, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        for (s = 0; s < supportSize; ++s) {
          PetscInt subf;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          subf = GetTriSubfaceInverse_Static(ornt[c], r);
          supportRef[s] = cStartNew + (support[s] - cStart)*4 + faces[c*3+subf];
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fStartNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+r, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fEnd - fStart)*3 + (c - cStart)*6;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face {a, g, m, h} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],0);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],2);
      orntNew[3] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 0;
      supportNew[1] = (c - cStart)*4 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {g, b, l , m} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[3] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 1;
      supportNew[1] = (c - cStart)*4 + 2;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {c, g, m, i} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],2);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],0);
      orntNew[3] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 0;
      supportNew[1] = (c - cStart)*4 + 2;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {d, h, m, i} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],0);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],2);
      orntNew[3] = -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 0;
      supportNew[1] = (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {h, m, l, e} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],1);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],1);
      orntNew[3] = 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 1;
      supportNew[1] = (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {i, m, l, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],2);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],1);
      orntNew[3] = 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*4 + 2;
      supportNew[1] = (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
    }
    /* Split Edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          supportRef[s] = fStartNew + (support[s] - fStart)*3 + (c + (ornt[c] < 0 ? 1-r : r))%3;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face edges have 2 vertices and 2 + cell faces supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2];
        PetscInt        fint[4][3] = { {0, 1, 2},
                                       {3, 4, 0},
                                       {2, 5, 3},
                                       {1, 4, 5} };

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (eEnd - eStart) + f - fStart;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*3 + (r+0)%3;
        supportRef[1] = fStartNew + (f - fStart)*3 + (r+1)%3;
        for (s = 0; s < supportSize; ++s) {
          PetscInt er;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          er = GetTriInteriorEdgeInverse_Static(ornt[c], r);
          supportRef[2+s] = fStartNew + (fEnd - fStart)*3 + (support[s] - cStart)*6 + fint[c][er];
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize + 2; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior cell edges have 2 vertices and 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt       fint[4][3] = { {0,1,2},
                                    {0,3,4},
                                    {2,3,5},
                                    {1,4,5} } ;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; r++) {
        PetscInt       coneNew[2], supportNew[3];
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd -fStart) + c - cStart;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = fStartNew + (fEnd - fStart)*3 + (c - cStart)*6 + fint[r][0];
        supportNew[1] = fStartNew + (fEnd - fStart)*3 + (c - cStart)*6 + fint[r][1];
        supportNew[2] = fStartNew + (fEnd - fStart)*3 + (c - cStart)*6 + fint[r][2];
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 3; ++p) {
          if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0, coneSize;

        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
        supportRef[2+s] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*3 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Face vertices have 3 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + 0;
      supportRef[1] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + 1;
      supportRef[2] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + 2;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0, coneSize;

        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == f) break;}
        supportRef[3+s] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (support[s] - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 3+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + c - cStart;
      supportRef[0] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 0;
      supportRef[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 1;
      supportRef[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 2;
      supportRef[3] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart)*4 + 3;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 4; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    eMax = PetscMin(eEnd, eMax);
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    /* All cells have 6 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6];
      PetscInt        orntNew[6];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 4; ++p) {
        if (cone[p] >= fMax) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position %D for cell %D", cone[p], p, fMax, c);
      }
#endif
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 0); /* B */
      orntNew[0] = ornt[0] < 0 ? -1 : 1;
      coneNew[1] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 3;               /* T */
      orntNew[1] = -4;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 0); /* F */
      orntNew[2] = ornt[2] < 0 ? -1 : 1;
      coneNew[3] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 0;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 2;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 0); /* L */
      orntNew[5] = ornt[1] < 0 ? -1 : 1;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 4;               /* T */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 0;               /* F */
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 1); /* K */
      orntNew[3] = ornt[3] < 0 ? -2 : 0;
      coneNew[4] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 1;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 2); /* L */
      orntNew[5] = ornt[1] < 0 ? -4 : 2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 2); /* B */
      orntNew[0] = ornt[0] < 0 ? -4 : 2;
      coneNew[1] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 5;               /* T */
      orntNew[1] = -4;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 1); /* F */
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 1;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 0); /* R */
      orntNew[4] = ornt[3] < 0 ? -1 : 1;
      coneNew[5] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 2;               /* L */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D hex */
      coneNew[0] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 3;               /* B */
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[3] - fStart)*3 + GetTriSubface_Static(ornt[3], 2); /* T */
      orntNew[1] = ornt[3] < 0 ? -1 : 1;
      coneNew[2] = fStartNew + (cone[2] - fStart)*3 + GetTriSubface_Static(ornt[2], 2); /* F */
      orntNew[2] = ornt[2] < 0 ? -4 : 2;
      coneNew[3] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 4;               /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (fMax    - fStart)*3 + (c - cStart)*6 + 5;               /* R */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 1); /* L */
      orntNew[5] = ornt[1] < 0 ? -2 : 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6], orntNew[6];
      PetscInt        o, of, cf;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if (cone[0] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 0 for cell %D", cone[0], fMax, c);
      if (cone[1] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 1 for cell %D", cone[1], fMax, c);
      if (cone[2] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 2 for cell %D", cone[2], fMax, c);
      if (cone[3] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 3 for cell %D", cone[3], fMax, c);
      if (cone[4] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 4 for cell %D", cone[4], fMax, c);
#endif
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      o    = ornt[0] < 0 ? -1 : 1;
      o    = 1;
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 0);                            /* B */
      orntNew[0] = ornt[0] < 0 ? -1 :  1;
      coneNew[1] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 0);                            /* T */
      orntNew[1] = ornt[1] < 0 ?  1 : -1;
      cf         = 2;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[2] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 0 : 1); /* F */
      orntNew[2] = o*of < 0 ? 0 : -1;
      coneNew[3] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 0;         /* K */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 2;         /* R */
      orntNew[4] = 0;
      cf         = 0;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[5] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 1 : 0); /* L */
      orntNew[5] = o*of < 0 ? 1 : -4;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 1);                            /* B */
      orntNew[0] = ornt[0] < 0 ? -2 :  0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 1);                            /* T */
      orntNew[1] = ornt[1] < 0 ?  2 : -4;
      coneNew[2] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 0;         /* F */
      orntNew[2] = 0;
      cf         = 1;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[3] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 1 : 0); /* K */
      orntNew[3] = o*of < 0 ? 0 : -1;
      coneNew[4] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 1;         /* R */
      orntNew[4] = -1;
      cf         = 0;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[5] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 0 : 1); /* L */
      orntNew[5] = o*of < 0 ? 1 : -4;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], 2);                            /* B */
      orntNew[0] = ornt[0] < 0 ? -4 : 2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], 2);                            /* T */
      orntNew[1] = ornt[1] < 0 ? 0 : -2;
      cf         = 2;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[2] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 1 : 0); /* F */
      orntNew[2] = o*of < 0 ? 0 : -1;
      coneNew[3] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 1;         /* K */
      orntNew[3] = 0;
      cf         = 1;
      of         = ornt[2+cf] < 0 ? -1 : 1;
      coneNew[4] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (cone[2+cf] - fMax)*2 + (o*of < 0 ? 0 : 1); /* R */
      orntNew[4] = o*of < 0 ? 0 : -1;
      coneNew[5] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 2;         /* L */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }

    /* Split faces have 4 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(2 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*3;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 3; ++p) {
        if (cone[p] >= eMax) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid edge %D (eMax %D) in cone position %D for face %D", cone[p], p, eMax, f);
      }
#endif
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[0] = ornt[2];
      coneNew[1] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[1] = ornt[0];
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < fStartNew) || (newp+0 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+0, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[0] = ornt[1];
      coneNew[1] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[1] = ornt[2];
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < fStartNew) || (newp+2 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+2, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        for (s = 0; s < supportSize; ++s) {
          PetscInt subf;

          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          if (coneSize != 5 && support[s] >= cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
          if (coneSize != 4 && support[s] <  cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          subf = GetTriSubfaceInverse_Static(ornt[c], r);
          if (coneSize == 4) {
            supportRef[s] = cStartNew + (support[s] - cStart)*4 + faces[c*3+subf];
          } else if (coneSize == 5) {
            if (c != 0 && c != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position %D in cone %D of cell %D (cMax %D) for face %D", c, support[s], cMax, f);
            supportRef[s] = cStartNew + (cMax - cStart)*4 + (support[s] - cMax)*3 + subf;
          } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fStartNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+r, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*3 + (c - cStart)*6;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 4; ++p) {
        if (cone[p] >= fMax) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position %D for face %D", cone[p], p, fMax, f);
      }
#endif
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face {a, g, m, h} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],0);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],2);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 0;
      supportNew[1] = cStartNew + (c - cStart)*4 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {g, b, l , m} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 1;
      supportNew[1] = cStartNew + (c - cStart)*4 + 2;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {c, g, m, i} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],2);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 0;
      supportNew[1] = cStartNew + (c - cStart)*4 + 2;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {d, h, m, i} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],0);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],2);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 0;
      supportNew[1] = cStartNew + (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {h, m, l, e} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 1;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],1);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],1);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 1;
      supportNew[1] = cStartNew + (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {i, m, l, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 2;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + GetTriInteriorEdge_Static(ornt[3],2);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + GetTriInteriorEdge_Static(ornt[2],1);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (c - cStart)*4 + 2;
      supportNew[1] = cStartNew + (c - cStart)*4 + 3;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
    }
    /* Hybrid split faces have 4 edges and same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        size, s;
      const PetscInt  newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (f - fMax)*2;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if (cone[0] >= eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid edge %D (eMax %D) in cone position 0 for face %D", cone[0], eMax, f);
      if (cone[1] >= eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid edge %D (eMax %D) in cone position 1 for face %D", cone[1], eMax, f);
      if (cone[2] <  eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected edge %D (eMax %D) in cone position 2 for face %D", cone[2], eMax, f);
      if (cone[3] <  eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected edge %D (eMax %D) in cone position 3 for face %D", cone[3], eMax, f);
#endif
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A face */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (f - fMax);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (cone[2] - eMax);
      orntNew[3] = ornt[2] < 0 ? 0 : -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif

      /* B face */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (cone[3] - eMax);
      orntNew[1] = ornt[3];
      coneNew[2] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (f - fMax);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 2; ++r) {
        for (s = 0; s < size; ++s) {
          const PetscInt *coneCell, *orntCell;
          PetscInt        coneSize, o, of, c;

          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          if (coneSize != 5 || support[s] < cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          o = orntCell[0] < 0 ? -1 : 1;
          o = 1;
          for (c = 0; c < coneSize; ++c) if (coneCell[c] == f) break;
          if (c == 0 || c == 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position in cone %D of cell %D (cMax %D) for face %D", c, support[s], cMax, f);
          if (c == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of cell %D", f, support[s]);
          of = orntCell[c] < 0 ? -1 : 1;
          supportRef[s] = cStartNew + (cMax - cStart)*4 + (support[s] - cMax)*3 + (c-2 + (o*of < 0 ? 1-r : r))%3;
        }
        ierr = DMPlexSetSupport(rdm, newp + r, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        for (p = 0; p < size; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior hybrid faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if (cone[0] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 0 for cell %D", cone[0], fMax, c);
      if (cone[1] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 1 for cell %D", cone[1], fMax, c);
      if (cone[2] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 2 for cell %D", cone[2], fMax, c);
      if (cone[3] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 3 for cell %D", cone[3], fMax, c);
      if (cone[4] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected face %D (fMax %D) in cone position 3 for cell %D", cone[3], fMax, c);
#endif
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face {a, g, h, d} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],0);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + c - cMax;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],0);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (cone[2] - fMax);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 0;
      supportNew[1] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {b, g, h, l} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],1);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + c - cMax;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],1);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (cone[3] - fMax);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 1;
      supportNew[1] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 2;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face {c, g, h, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriInteriorEdge_Static(ornt[0],2);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + c - cMax;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriInteriorEdge_Static(ornt[1],2);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (cone[4] - fMax);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 2;
      supportNew[1] = cStartNew + (cMax - cStart)*4 + (c - cMax)*3 + 0;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Face edges have 2 vertices and 2 + cell faces supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2];
        PetscInt        fint[4][3] = { {0, 1, 2},
                                       {3, 4, 0},
                                       {2, 5, 3},
                                       {1, 4, 5} };

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        if (cone[r] >= eMax) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected cone point %D in position %D for face %D (eMax %D)", cone[r], r, f, eMax);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (eMax - eStart) + f - fStart;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*3 + (r+0)%3;
        supportRef[1] = fStartNew + (f - fStart)*3 + (r+1)%3;
        for (s = 0; s < supportSize; ++s) {
          PetscInt er;

          supportRef[2+s] = -1;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          if (coneSize != 5 && support[s] >= cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
          if (coneSize != 4 && support[s] <  cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          er = GetTriInteriorEdgeInverse_Static(ornt[c], r);
          if (coneSize == 4) {
            supportRef[2+s] = fStartNew + (fMax - fStart)*3 + (support[s] - cStart)*6 + fint[c][er];
          } else if (coneSize == 5) {
            if (c != 0 && c != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position %D in cone %D of cell %D (cMax %D) for face %D", c, support[s], cMax, f);
            supportRef[2+s] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + er;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize + 2; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior cell edges have 2 vertices and 3 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt *cone;
      PetscInt       fint[4][3] = { {0,1,2},
                                    {0,3,4},
                                    {2,3,5},
                                    {1,4,5} } ;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; r++) {
        PetscInt       coneNew[2], supportNew[3];
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + r;

        if (cone[r] >= fMax) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position %D for cell %D", cone[r], r, fMax, c);
        coneNew[0] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax     -fStart) + c - cStart;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = fStartNew + (fMax - fStart)*3 + (c - cStart)*6 + fint[r][0];
        supportNew[1] = fStartNew + (fMax - fStart)*3 + (c - cStart)*6 + fint[r][1];
        supportNew[2] = fStartNew + (fMax - fStart)*3 + (c - cStart)*6 + fint[r][2];
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 3; ++p) {
          if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Hybrid edges have two vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (e - eMax);
      const PetscInt *cone, *support, *fcone;
      PetscInt        coneNew[2], size, fsize, s;

      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is a edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &fsize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &fcone);CHKERRQ(ierr);
        for (c = 0; c < fsize; ++c) if (fcone[c] == e) break;
        if ((c < 2) || (c > 3)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D not found in cone of face %D", e, support[s]);
        supportRef[s] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (support[s] - fMax)*2 + c-2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Hybrid face edges have 2 vertices and 2 + cell faces supports */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize;
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + f - fMax;
      PetscInt        coneNew[2], s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if (cone[0] >= eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected cone point %D in position 0 for face %D (eMax %D)", cone[0], f, eMax);
      if (cone[1] >= eMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected cone point %D in position 1 for face %D (eMax %D)", cone[1], f, eMax);
#endif
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - eStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportRef[0] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (f- fMax)*2 + 0;
      supportRef[1] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (f- fMax)*2 + 1;
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        if (coneSize != 5 || support[s] < cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
        for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
        if (c == 0 || c == 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position in cone %D of cell %D (cMax %D) for face %D", c, support[s], cMax, f);
        supportRef[2+s] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + c - 2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < supportSize + 2; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Hybrid cell edges have 2 vertices and 3 faces */
    for (c = cMax; c < cEnd; ++c) {
      PetscInt       coneNew[2], supportNew[3];
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + c - cMax;
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if (cone[0] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 0 for cell %D", cone[0], fMax, c);
      if (cone[1] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected hybrid face %D (fMax %D) in cone position 1 for cell %D", cone[1], fMax, c);
#endif
      coneNew[0] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[0] - fStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[1] - fStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 0;
      supportNew[1] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 1;
      supportNew[2] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (c - cMax)*3 + 2;
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
      for (p = 0; p < 3; ++p) {
        if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        const PetscInt e = support[s];

        supportRef[s] = -1;
        if (eStart <= e) {
          if (e < eMax) {
            ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
            supportRef[s] = eStartNew + (e - eStart)*2 + (cone[1] == v ? 1 : 0);
          } else if (e < eEnd) {
            supportRef[s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + e - eMax;
          } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", e, eStart, eEnd);
        } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", e, eStart, eEnd);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior edge vertices have 2 + faces supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r, coneSize;

        supportRef[2+s] = -1;
        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        if (coneSize != 4 && support[s] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
        if (coneSize != 3 && support[s] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
        if (coneSize == 3) supportRef[2+s] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*3 + r;
        else if (coneSize == 4) {
          if (r != 0 && r != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position in cone %D of face %D (fMax %D) for edge %D", r, support[s], fMax, e);
          supportRef[2+s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + support[s] - fMax;
        } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Split Edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          if (coneSize != 4 && support[s] >= fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
          if (coneSize != 3 && support[s] <  fMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          if (coneSize == 3) supportRef[s] = fStartNew + (support[s] - fStart)*3 + (c + (ornt[c] < 0 ? 1-r : r))%3;
          else if (coneSize == 4) {
            if (c != 0 && c != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position in cone %D of face %D (fMax %D) for edge %D", c, support[s], fMax, e);
            supportRef[s] = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (support[s] - fMax)*2 + (ornt[c] < 0 ? 1-r : r);
          } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for face %D (fMax %D)", coneSize, support[s], fMax);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face vertices have 3 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + 0;
      supportRef[1] = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + 1;
      supportRef[2] = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + 2;
      for (s = 0; s < size; ++s) {
        PetscInt r, coneSize;

        supportRef[3+s] = -1;
        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        if (coneSize != 5 && support[s] >= cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
        if (coneSize != 4 && support[s] <  cMax) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected conesize %D for cell %D (cMax %D)", coneSize, support[s], cMax);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == f) break;}
        if (coneSize == 4) supportRef[3+s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (support[s] - cStart)*4 + r;
        else if (coneSize == 5) {
          if (r != 0 && r != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected position in cone %D of cell %D (cMax %D) for face %D", r, support[s], cMax, f);
          supportRef[3+s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + support[s] - cMax;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 3+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior cell vertices have 4 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + c - cStart;

      supportRef[0] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 0;
      supportRef[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 1;
      supportRef[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 2;
      supportRef[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart)*4 + 3;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 4; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case REFINER_HEX_3D:
    /*
     Bottom (viewed from top)    Top
     1---------2---------2       7---------2---------6
     |         |         |       |         |         |
     |    B    2    C    |       |    H    2    G    |
     |         |         |       |         |         |
     3----3----0----1----1       3----3----0----1----1
     |         |         |       |         |         |
     |    A    0    D    |       |    E    0    F    |
     |         |         |       |         |         |
     0---------0---------3       4---------0---------5
     */
    /* All cells have 6 faces: Bottom, Top, Front, Back, Right, Left */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6], orntNew[6];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 0);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 1);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 3);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 2);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 0);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 1);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 3);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 0);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* E hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 3);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[4] = -1;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 1);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+4 < cStartNew) || (newp+4 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+4, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* F hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 2);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 3);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[5] = 1;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+5 < cStartNew) || (newp+5 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+5, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* G hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 2);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 3);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 2);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[5] = -3;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+6 < cStartNew) || (newp+6 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+6, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* H hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 3);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 2);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[4] = 3;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 2);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+7 < cStartNew) || (newp+7 >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+7, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 4 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(4 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        /* TODO: This can come from GetFaces_Internal() */
        const PetscInt  newCells[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 3, 5, 4,  2, 1, 7, 6,  3, 2, 6, 5,  0, 4, 7, 1};
        const PetscInt  newp = fStartNew + (f - fStart)*4 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[4], orntNew[4], coneSize, c, supportSize, s;

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
        coneNew[(r+3)%4] = eStartNew + (cone[(r+3)%4] - eStart)*2 + (ornt[(r+3)%4] < 0 ? 0 : 1);
        orntNew[(r+3)%4] = ornt[(r+3)%4];
        coneNew[(r+0)%4] = eStartNew + (cone[r]       - eStart)*2 + (ornt[r] < 0 ? 1 : 0);
        orntNew[(r+0)%4] = ornt[r];
        coneNew[(r+1)%4] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;
        orntNew[(r+1)%4] = 0;
        coneNew[(r+2)%4] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + (r+3)%4;
        orntNew[(r+2)%4] = -2;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 4; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*8 + newCells[c*4+GetQuadSubfaceInverse_Static(ornt[c], r)];
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newCells[24] = {0, 3,  2, 3,  1, 2,  0, 1,  4, 5,  5, 6,  6, 7,  4, 7,  0, 4,  3, 5,  2, 6,  1, 7};
      const PetscInt *cone, *ornt;
      PetscInt        newp, coneNew[4], orntNew[4], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A-D face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 0;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 3);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C-D face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 1;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 2);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B-C face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 2;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* A-B face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 3;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* E-F face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 4;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 2);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 0);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* F-G face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 5;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 2);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 1);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* G-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 6;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 2);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 2);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* E-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 7;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 1);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 3);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* A-E face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 8;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 3);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* D-F face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 9;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C-G face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 10;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 1);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 3);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 11;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 1);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 2);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      for (r = 0; r < 12; ++r) {
        newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + r;
        supportNew[0] = cStartNew + (c - cStart)*8 + newCells[r*2+0];
        supportNew[1] = cStartNew + (c - cStart)*8 + newCells[r*2+1];
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Split edges have 2 vertices and the same faces as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          supportRef[s] = fStartNew + (support[s] - fStart)*4 + (ornt[c] < 0 ? (c+1-r)%4 : (c+r)%4);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newFaces[24] = {3, 2, 1, 0,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  8, 7, 11, 3};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      const PetscInt *cone, *coneCell, *orntCell, *support;
      PetscInt        coneNew[2], coneSize, c, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - eStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        supportRef[0] = fStartNew + (f - fStart)*4 + r;
        supportRef[1] = fStartNew + (f - fStart)*4 + (r+1)%4;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (coneCell[c] == f) break;
          supportRef[2+s] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*12 + newFaces[c*4 + GetQuadEdgeInverse_Static(orntCell[c], r)];
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2+supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newFaces[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  3, 8, 7, 11};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (cone[r] - fStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        for (f = 0; f < 4; ++f) supportNew[f] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + newFaces[r*4+f];
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eEndNew);
        for (p = 0; p < 4; ++p) {
          if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 4; ++r) if (cone[r] == e) break;
        supportRef[2+s] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Face vertices have 4 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) supportRef[r] = eStartNew + (eEnd - eStart)*2 +  (f - fStart)*4 + r;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 6; ++r) if (cone[r] == f) break;
        supportRef[4+s] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (support[s] - cStart)*6 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 4+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Cell vertices have 6 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);
      PetscInt       supportNew[6];

      for (r = 0; r < 6; ++r) {
        supportNew[r] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case REFINER_HYBRID_HEX_3D:
    ierr = DMPlexGetHybridBounds(rdm, &cMaxNew, &fMaxNew, &eMaxNew, NULL);CHKERRQ(ierr);
    /*
     Bottom (viewed from top)    Top
     1---------2---------2       7---------2---------6
     |         |         |       |         |         |
     |    B    2    C    |       |    H    2    G    |
     |         |         |       |         |         |
     3----3----0----1----1       3----3----0----1----1
     |         |         |       |         |         |
     |    A    0    D    |       |    E    0    F    |
     |         |         |       |         |         |
     0---------0---------3       4---------0---------5
     */
    /* Interior cells have 6 faces: Bottom, Top, Front, Back, Right, Left */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6], orntNew[6];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 0);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+0 < cStartNew) || (newp+0 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+0, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 1);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 3);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+1 < cStartNew) || (newp+1 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+1, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 2);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 0);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 1);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+2 < cStartNew) || (newp+2 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+2, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 3);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 0);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[5] = -4;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+3 < cStartNew) || (newp+3 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+3, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* E hex */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 3);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[4] = -1;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 1);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+4 < cStartNew) || (newp+4 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+4, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* F hex */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 2);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[3] = -1;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 3);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[5] = 1;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+5 < cStartNew) || (newp+5 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+5, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* G hex */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 2);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 3);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 2);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[5] = -3;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+6 < cStartNew) || (newp+6 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+6, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* H hex */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[0] = -4;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 3);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[2] = -1;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 2);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[4] = 3;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 2);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp+7 < cStartNew) || (newp+7 >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", newp+7, cStartNew, cMaxNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /* Hybrid cells have 6 faces: Front, Back, Sides */
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = (cMax - cStart)*8 + (c - cMax)*4;
      const PetscInt *cone, *ornt, *fornt;
      PetscInt        coneNew[6], orntNew[6], o, of, i;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, cone[0], &fornt);CHKERRQ(ierr);
      o = ornt[0] < 0 ? -1 : 1;
      for (r = 0; r < 4; ++r) {
        PetscInt subfA = GetQuadSubface_Static(ornt[0], r);
        PetscInt edgeA = GetQuadEdge_Static(ornt[0], r);
        PetscInt edgeB = GetQuadEdge_Static(ornt[0], (r+3)%4);
        if (ornt[0] != ornt[1]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent ordering for matching ends of hybrid cell %D: %D != %D", c, ornt[0], ornt[1]);
        coneNew[0]         = fStartNew + (cone[0] - fStart)*4 + subfA;
        orntNew[0]         = ornt[0];
        coneNew[1]         = fStartNew + (cone[1] - fStart)*4 + subfA;
        orntNew[1]         = ornt[0];
        of = fornt[edgeA] < 0 ? -1 : 1;
        i  = GetQuadEdgeInverse_Static(ornt[0], r) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (cone[2+edgeA] - fMax)*2 + (o*of < 0 ? 1 : 0);
        orntNew[i] = ornt[edgeA];
        i  = GetQuadEdgeInverse_Static(ornt[0], (r+1)%4) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd          - fMax)*2 + (c - cMax)*4 + edgeA;
        orntNew[i] = 0;
        i  = GetQuadEdgeInverse_Static(ornt[0], (r+2)%4) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd          - fMax)*2 + (c - cMax)*4 + edgeB;
        orntNew[i] = -2;
        of = fornt[edgeB] < 0 ? -1 : 1;
        i  = GetQuadEdgeInverse_Static(ornt[0], (r+3)%4) + 2;
        coneNew[i] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (cone[2+edgeB] - fMax)*2 + (o*of < 0 ? 0 : 1);
        orntNew[i] = ornt[edgeB];
        ierr       = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr       = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < cMaxNew) || (newp+r >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", newp+r, cMaxNew, cEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", coneNew[p], fStartNew, fMaxNew);
        }
        for (p = 2; p < 6; ++p) {
          if ((coneNew[p] < fMaxNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", coneNew[p], fMaxNew, fEndNew);
        }
#endif
      }
    }
    /* Interior split faces have 4 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(4 + maxSupportSize*2, &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 4; ++r) {
        /* TODO: This can come from GetFaces_Internal() */
        const PetscInt  newCells[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 3, 5, 4,  2, 1, 7, 6,  3, 2, 6, 5,  0, 4, 7, 1};
        const PetscInt  newp = fStartNew + (f - fStart)*4 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[4], orntNew[4], coneSize, c, supportSize, s;

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
        coneNew[(r+3)%4] = eStartNew + (cone[(r+3)%4] - eStart)*2 + (ornt[(r+3)%4] < 0 ? 0 : 1);
        orntNew[(r+3)%4] = ornt[(r+3)%4];
        coneNew[(r+0)%4] = eStartNew + (cone[r]       - eStart)*2 + (ornt[r] < 0 ? 1 : 0);
        orntNew[(r+0)%4] = ornt[r];
        coneNew[(r+1)%4] = eStartNew + (eMax - eStart)*2 + (f - fStart)*4 + r;
        orntNew[(r+1)%4] = 0;
        coneNew[(r+2)%4] = eStartNew + (eMax - eStart)*2 + (f - fStart)*4 + (r+3)%4;
        orntNew[(r+2)%4] = -2;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
        for (p = 0; p < 4; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          PetscInt subf;
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          subf = GetQuadSubfaceInverse_Static(ornt[c], r);
          if (support[s] < cMax) {
            supportRef[s] = cStartNew + (support[s] - cStart)*8 + newCells[c*4+subf];
          } else {
            supportRef[s] = cStartNew + (cMax       - cStart)*8 + (support[s] - cMax)*4 + subf;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 4 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newCells[24] = {0, 3,  2, 3,  1, 2,  0, 1,  4, 5,  5, 6,  6, 7,  4, 7,  0, 4,  3, 5,  2, 6,  1, 7};
      const PetscInt *cone, *ornt;
      PetscInt        newp, coneNew[4], orntNew[4], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A-D face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 0;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 3);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* C-D face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 1;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 2);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* B-C face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 2;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* A-B face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 3;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* E-F face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 4;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 2);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 0);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* F-G face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 5;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 2);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 1);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* G-H face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 6;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 2);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 2);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* E-H face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 7;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 1);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 3);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* A-E face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 8;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 3);
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 0);
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* D-F face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 9;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* C-G face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 10;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 1);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 3);
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* B-H face */
      newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + 11;
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 1);
      orntNew[2] = -2;
      coneNew[3] = eStartNew + (eMax - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 2);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      for (r = 0; r < 12; ++r) {
        newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + r;
        supportNew[0] = cStartNew + (c - cStart)*8 + newCells[r*2+0];
        supportNew[1] = cStartNew + (c - cStart)*8 + newCells[r*2+1];
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", newp, fStartNew, fMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a cell [%D, %D)", supportNew[p], cStartNew, cMaxNew);
        }
#endif
      }
    }
    /* Hybrid split faces have 4 edges and same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2], size, s, c;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (f - fMax)*2 + r;

        coneNew[0]   = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1-r : r);
        orntNew[0]   = ornt[0];
        coneNew[1]   = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1-r : r);
        orntNew[1]   = ornt[1];
        coneNew[2+r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (cone[2+r] - eMax);
        orntNew[2+r] = 0;
        coneNew[3-r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd      - eMax) + (f - fMax);
        orntNew[3-r] = 0;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        for (s = 0; s < size; ++s) {
          const PetscInt *coneCell, *orntCell, *fornt;
          PetscInt        o, of;

          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          o = orntCell[0] < 0 ? -1 : 1;
          for (c = 2; c < 6; ++c) if (coneCell[c] == f) break;
          if (c >= 6) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of cell %D", f, support[s]);
          ierr = DMPlexGetConeOrientation(dm, coneCell[0], &fornt);CHKERRQ(ierr);
          of = fornt[c-2] < 0 ? -1 : 1;
          supportNew[s] = cStartNew + (cMax - cStart)*8 + (support[s] - cMax)*4 + (GetQuadEdgeInverse_Static(orntCell[0], c-2) + (o*of < 0 ? 1-r : r))%4;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp, fMaxNew, fEndNew);
        for (p = 0; p < size; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Hybrid cell faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
#if 0
        coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], r);
        orntNew[0] = 0;
        coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], r);
        orntNew[1] = 0;
        coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (cone[2+GetQuadEdge_Static(ornt[0], r)] - fMax);
        orntNew[2] = 0;
        coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (fEnd                                   - fMax) + (c - cMax);
        orntNew[3] = 0;
#else
        coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*4 + r;
        orntNew[0] = 0;
        coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*4 + r;
        orntNew[1] = 0;
        coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (cone[2+r] - fMax);
        orntNew[2] = 0;
        coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (fEnd      - fMax) + (c - cMax);
        orntNew[3] = 0;
#endif
        ierr = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        supportNew[0] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + GetQuadSubface_Static(ornt[0], r);
        supportNew[1] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + GetQuadSubface_Static(ornt[0], (r+1)%4);
        ierr          = DMPlexSetSupport(rdm, newp+r, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid cell [%D, %D)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Interior split edges have 2 vertices and the same faces as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          if (support[s] < fMax) {
            supportRef[s] = fStartNew + (support[s] - fStart)*4 + (c + (ornt[c] < 0 ? 1-r : r))%4;
          } else {
            supportRef[s] = fStartNew + (fMax       - fStart)*4 + (cMax - cStart)*12 + (support[s] - fMax)*2 + (ornt[c] < 0 ? 1-r : r);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newFaces[24] = {3, 2, 1, 0,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  8, 7, 11, 3};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eMax - eStart) + (f - fStart);
      const PetscInt *cone, *coneCell, *orntCell, *support;
      PetscInt        coneNew[2], coneSize, c, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - eStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        supportRef[0] = fStartNew + (f - fStart)*4 + r;
        supportRef[1] = fStartNew + (f - fStart)*4 + (r+1)%4;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (coneCell[c] == f) break;
          if (support[s] < cMax) {
            supportRef[2+s] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*12 + newFaces[c*4 + GetQuadEdgeInverse_Static(orntCell[c], r)];
          } else {
            supportRef[2+s] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (support[s] - cMax)*4 + r;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2+supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newFaces[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  3, 8, 7, 11};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + (c - cStart);
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (c - cStart)*6 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[r] - fStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        for (f = 0; f < 4; ++f) supportNew[f] = fStartNew + (fMax - fStart)*4 + (c - cStart)*12 + newFaces[r*4+f];
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 4; ++p) {
          if ((supportNew[p] < fStartNew) || (supportNew[p] >= fMaxNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a face [%D, %D)", supportNew[p], fStartNew, fMaxNew);
        }
#endif
      }
    }
    /* Hybrid edges have two vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (e - eMax);
      const PetscInt *cone, *support, *fcone;
      PetscInt        coneNew[2], size, fsize, s;

      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &fsize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &fcone);CHKERRQ(ierr);
        for (c = 0; c < fsize; ++c) if (fcone[c] == e) break;
        if ((c < 2) || (c > 3)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D not found in cone of face %D", e, support[s]);
        supportRef[s] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (support[s] - fMax)*2 + c-2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Hybrid face edges have 2 vertices and 2+cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (f - fMax);
      const PetscInt *cone, *support, *ccone, *cornt;
      PetscInt        coneNew[2], size, csize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - eStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportRef[0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (f - fMax)*2 + 0;
      supportRef[1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (f - fMax)*2 + 1;
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &csize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &ccone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[s], &cornt);CHKERRQ(ierr);
        for (c = 0; c < csize; ++c) if (ccone[c] == f) break;
        if ((c < 2) || (c >= csize)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hybrid face %D is not in cone of hybrid cell %D", f, support[s]);
        supportRef[2+s] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (support[s] - cMax)*4 + c-2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Hybrid cell edges have 2 vertices and 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (fEnd - fMax) + (c - cMax);
      const PetscInt *cone, *support;
      PetscInt        coneNew[2], size;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, c, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[0] - fStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (eMax - eStart) + (cone[1] - fStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportRef[0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4 + 0;
      supportRef[1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4 + 1;
      supportRef[2] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4 + 2;
      supportRef[3] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (c - cMax)*4 + 3;
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid edge [%D, %D)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 4; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a hybrid face [%D, %D)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        if (support[s] < eMax) supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
        else                   supportRef[s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (support[s] - eMax);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior edge vertices have 2 + faces supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 4; ++r) if (cone[r] == e) break;
        if (support[s] < fMax) {
          supportRef[2+s] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*4 + r;
        } else {
          supportRef[2+s] = eStartNew + (eMax - eStart)*2 + (fMax       - fStart)*4 + (cMax - cStart)*6 + (eEnd - eMax) + (support[s] - fMax);
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior face vertices have 4 + cells supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) supportRef[r] = eStartNew + (eMax - eStart)*2 +  (f - fStart)*4 + r;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 6; ++r) if (cone[r] == f) break;
        if (support[s] < cMax) {
          supportRef[4+s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (support[s] - cStart)*6 + r;
        } else {
          supportRef[4+s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax       - cStart)*6 + (eEnd - eMax) + (fEnd - fMax) + (support[s] - cMax);
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a vertex [%D, %D)", newp, vStartNew, vEndNew);
      for (p = 0; p < 4+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not an edge [%D, %D)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Cell vertices have 6 supports */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + (c - cStart);
      PetscInt       supportNew[6];

      for (r = 0; r < 6; ++r) {
        supportNew[r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (c - cStart)*6 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerSetCoordinates(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscSection          coordSection, coordSectionNew;
  Vec                   coordinates, coordinatesNew;
  PetscScalar          *coords, *coordsNew;
  const PetscInt        numVertices = depthSize ? depthSize[0] : 0;
  PetscInt              dim, spaceDim, depth, bs, coordSizeNew, cStart, cEnd, cMax;
  PetscInt              c, vStart, vStartNew, vEnd, v, eStart, eEnd, eMax, e, fStart, fEnd, fMax, f;
  PetscInt              cStartNew, cEndNew, vEndNew, *parentId = NULL;
  VecType               vtype;
  PetscBool             isperiodic, localize = PETSC_FALSE, needcoords = PETSC_FALSE;
  const PetscReal      *maxCell, *L;
  const DMBoundaryType *bd;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, NULL);CHKERRQ(ierr);
  if (cMax < 0) cMax = cEnd;
  if (fMax < 0) fMax = fEnd;
  if (eMax < 0) eMax = eEnd;
  ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, NULL, NULL, &vStartNew);CHKERRQ(ierr);
  ierr = GetDepthEnd_Private(depth, depthSize, &cEndNew, NULL, NULL, &vEndNew);CHKERRQ(ierr);
  ierr = DMGetPeriodicity(dm, &isperiodic, &maxCell, &L, &bd);CHKERRQ(ierr);
  /* Determine if we need to localize coordinates when generating them */
  if (isperiodic && !maxCell) {
    ierr = DMGetCoordinatesLocalized(dm, &localize);CHKERRQ(ierr);
    if (!localize) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Cannot refine if coordinates have not been localized");
  }
  if (isperiodic) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex coords refinement options","DM");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_plex_refine_localize","Automatically localize from parent cells",NULL,localize,&localize,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (localize) {
      ierr = DMLocalizeCoordinates(dm);CHKERRQ(ierr);
    }
  }
  ierr = DMSetPeriodicity(rdm, isperiodic,  maxCell,  L,  bd);CHKERRQ(ierr);

  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSection, 0, &spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &coordSectionNew);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSectionNew, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionNew, 0, spaceDim);CHKERRQ(ierr);

  if (localize) {
    PetscInt p, r, newp, *pi;

    /* New coordinates will be already localized on the cell */
    ierr = PetscSectionSetChart(coordSectionNew, 0, vStartNew+numVertices);CHKERRQ(ierr);

    /* We need the parentId to properly localize coordinates */
    ierr = PetscMalloc1(cEndNew-cStartNew,&pi);CHKERRQ(ierr);
    switch (refiner) {
    case REFINER_NOOP:
      break;
    case REFINER_SIMPLEX_1D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 2; ++r) {
          newp     = (p - cStart)*2 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_SIMPLEX_2D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HEX_2D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_SIMPLEX_TO_HEX_2D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 3; ++r) {
          newp     = (p - cStart)*3 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 3; ++r) {
          newp     = (p - cStart)*3 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (cMax - cStart)*3 + (p - cMax)*4 + r;
          pi[newp] = p;
        }
      }
      /* The refiner needs midpoint vertices on hybrid edges and hybrid cells */
      cMax = cEnd;
      eMax = eEnd;
      break;
    case REFINER_HYBRID_SIMPLEX_2D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 2; ++r) {
          newp     = (cMax - cStart)*4 + (p - cMax)*2 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HYBRID_HEX_2D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 2; ++r) {
          newp     = (cMax - cStart)*4 + (p - cMax)*2 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_SIMPLEX_3D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 8; ++r) {
          newp     = (p - cStart)*8 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HYBRID_SIMPLEX_3D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 8; ++r) {
          newp     = (p - cStart)*8 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (cMax - cStart)*8 + (p - cMax)*4 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_SIMPLEX_TO_HEX_3D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 4; ++r) {
          newp     = (p - cStart)*4 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 3; ++r) {
          newp     = (cMax - cStart)*4 + (p - cMax)*3 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HEX_3D:
      for (p = cStart; p < cEnd; ++p) {
        for (r = 0; r < 8; ++r) {
          newp = (p - cStart)*8 + r;
          pi[newp] = p;
        }
      }
      break;
    case REFINER_HYBRID_HEX_3D:
      for (p = cStart; p < cMax; ++p) {
        for (r = 0; r < 8; ++r) {
          newp = (p - cStart)*8 + r;
          pi[newp] = p;
        }
      }
      for (p = cMax; p < cEnd; ++p) {
        for (r = 0; r < 4; ++r) {
          newp = (cMax - cStart)*8 + (p - cMax)*4 + r;
          pi[newp] = p;
        }
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
    }
    parentId = pi;
  } else {
    /* The refiner needs midpoint vertices on hybrid edges and hybrid cells */
    if (REFINER_HYBRID_SIMPLEX_TO_HEX_2D == refiner) { cMax = cEnd; eMax = eEnd; }
    ierr = PetscSectionSetChart(coordSectionNew, vStartNew, vStartNew+numVertices);CHKERRQ(ierr);
  }

  /* All vertices have the spaceDim coordinates */
  if (localize) {
    PetscInt c;

    for (c = cStartNew; c < cEndNew; ++c) {
      PetscInt *cone = NULL;
      PetscInt  closureSize, coneSize = 0, p, pdof;

      ierr = PetscSectionGetDof(coordSection, parentId[c], &pdof); CHKERRQ(ierr);
      if (pdof) { /* localize on all cells that are refinement of a localized parent cell */
        ierr = DMPlexGetTransitiveClosure(rdm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
        for (p = 0; p < closureSize*2; p += 2) {
          const PetscInt point = cone[p];
          if ((point >= vStartNew) && (point < vEndNew)) coneSize++;
        }
        ierr = DMPlexRestoreTransitiveClosure(rdm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(coordSectionNew, c, coneSize*spaceDim);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(coordSectionNew, c, 0, coneSize*spaceDim);CHKERRQ(ierr);
      }
    }
  }
  for (v = vStartNew; v < vStartNew+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSectionNew, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionNew, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSectionNew);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(rdm, PETSC_DETERMINE, coordSectionNew);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionNew, &coordSizeNew);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinatesNew);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesNew, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesNew, coordSizeNew, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinatesNew, bs);CHKERRQ(ierr);
  ierr = VecGetType(coordinates, &vtype);CHKERRQ(ierr);
  ierr = VecSetType(coordinatesNew, vtype);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);

  switch (refiner) {
  case REFINER_NOOP: break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
  case REFINER_SIMPLEX_TO_HEX_3D:
  case REFINER_HEX_3D:
  case REFINER_HYBRID_HEX_3D:
    /* Face vertices have the average of corner coordinates */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (eMax - eStart) + (f - fStart);
      PetscInt      *cone = NULL;
      PetscInt       closureSize, coneSize = 0, off[8], offnew, p, d;

      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = cone[p];
        if ((point >= vStart) && (point < vEnd)) cone[coneSize++] = point;
      }
      if (localize) {
        const PetscInt *support = NULL;
        PetscInt       *rStar = NULL;
        PetscInt        supportSize, rStarSize, coff, s, ccoff[8];
        PetscBool       cellfound = PETSC_FALSE;

        ierr = DMPlexGetTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm,f,&supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm,f,&support);CHKERRQ(ierr);
        /* Compute average of coordinates for each cell sharing the face */
        for (s = 0; s < supportSize; ++s) {
          PetscScalar     coordsNewAux[3] = { 0.0, 0.0, 0.0 };
          PetscInt       *cellCone = NULL;
          PetscInt        cellClosureSize, cellConeSize = 0, cdof;
          const PetscInt  cell = support[s];
          PetscBool       copyoff = PETSC_FALSE;

          ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &cellClosureSize, &cellCone);CHKERRQ(ierr);
          for (p = 0; p < cellClosureSize*2; p += 2) {
            const PetscInt point = cellCone[p];
            if ((point >= vStart) && (point < vEnd)) cellCone[cellConeSize++] = point;
          }
          ierr = PetscSectionGetDof(coordSection, cell, &cdof);CHKERRQ(ierr);
          if (!cdof) { /* the parent cell does not have localized coordinates */
            cellfound = PETSC_TRUE;
            for (v = 0; v < coneSize; ++v) {
              ierr = PetscSectionGetOffset(coordSection, cone[v], &off[v]);CHKERRQ(ierr);
              for (d = 0; d < spaceDim; ++d) coordsNewAux[d] += coords[off[v]+d];
            }
            for (d = 0; d < spaceDim; ++d) coordsNewAux[d] /= coneSize;
          } else {
            ierr = PetscSectionGetOffset(coordSection, cell, &coff);CHKERRQ(ierr);
            for (p = 0; p < coneSize; ++p) {
              const PetscInt tv = cone[p];
              PetscInt       cv, voff;
              PetscBool      locv = PETSC_TRUE;

              for (cv = 0; cv < cellConeSize; ++cv) {
                if (cellCone[cv] == tv) {
                  ccoff[p] = spaceDim*cv + coff;
                  break;
                }
              }
              if (cv == cellConeSize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map vertex %D",tv);

              ierr = PetscSectionGetOffset(coordSection, cone[p], &voff);CHKERRQ(ierr);
              for (d = 0; d < spaceDim; ++d) {
                coordsNewAux[d] += coords[ccoff[p]+d];
                if (!cellfound && coords[voff+d] != coords[ccoff[p]+d]) locv = PETSC_FALSE;
              }
              if (locv && !cellfound) {
                cellfound = PETSC_TRUE;
                copyoff   = PETSC_TRUE;
              }
            }
            for (d = 0; d < spaceDim; ++d) coordsNewAux[d] /= coneSize;

            /* Found a valid face for the "vertex" part of the Section (physical space)
               i.e., a face that has at least one corner in the physical space */
            if (copyoff) for (p = 0; p < coneSize; ++p) off[p] = ccoff[p];
          }

          /* Localize new coordinates on each refined cell */
          for (v = 0; v < rStarSize*2; v += 2) {
            if ((rStar[v] >= cStartNew) && (rStar[v] < cEndNew) && parentId[rStar[v]-cStartNew] == cell) {
              PetscInt       *rcone = NULL, rclosureSize, lid, rcdof, rcoff;
              const PetscInt  rcell = rStar[v];

              ierr = PetscSectionGetDof(coordSectionNew, rcell, &rcdof);CHKERRQ(ierr);
              if (!rcdof) continue;
              ierr = PetscSectionGetOffset(coordSectionNew, rcell, &rcoff);CHKERRQ(ierr);
              ierr = DMPlexGetTransitiveClosure(rdm, rcell, PETSC_TRUE, &rclosureSize, &rcone);CHKERRQ(ierr);
              for (p = 0, lid = 0; p < rclosureSize*2; p += 2) {
                if (rcone[p] == newv) {
                  for (d = 0; d < spaceDim; d++) coordsNew[rcoff + lid*spaceDim + d] = coordsNewAux[d];
                  break;
                }
                if (rcone[p] >= vStartNew && rcone[p] < vEndNew) lid++;
              }
              ierr = DMPlexRestoreTransitiveClosure(rdm, rcell, PETSC_TRUE, &rclosureSize, &rcone);CHKERRQ(ierr);
              if (p == closureSize*2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map new vertex %D",newv);
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &cellClosureSize, &cellCone);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        if (!cellfound) {
          /* Could not find a valid face for the vertex part, we will get this vertex later (final reduction) */
          needcoords = PETSC_TRUE;
          coneSize   = 0;
        }
      } else {
        for (v = 0; v < coneSize; ++v) {
          ierr = PetscSectionGetOffset(coordSection, cone[v], &off[v]);CHKERRQ(ierr);
        }
      }
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      if (coneSize) {
        for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] = 0.0;
        for (v = 0; v < coneSize; ++v) {ierr = DMLocalizeAddCoordinate_Internal(dm, spaceDim, &coords[off[0]], &coords[off[v]], &coordsNew[offnew]);CHKERRQ(ierr);}
        for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] /= coneSize;
      } else {
        for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] = PETSC_MIN_REAL;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
    }
  case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
  case REFINER_SIMPLEX_TO_HEX_2D:
  case REFINER_HEX_2D:
  case REFINER_HYBRID_HEX_2D:
  case REFINER_SIMPLEX_1D:
    /* Cell vertices have the average of corner coordinates */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (dim > 1 ? (eMax - eStart) : 0) + (c - cStart) + (dim > 2 ? (fMax - fStart) : 0);
      PetscInt      *cone = NULL;
      PetscInt       closureSize, coneSize = 0, off[8], offnew, p, d, cdof = 0;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = cone[p];
        if ((point >= vStart) && (point < vEnd)) cone[coneSize++] = point;
      }
      if (localize) {
        ierr = PetscSectionGetDof(coordSection, c, &cdof);CHKERRQ(ierr);
      }
      if (cdof) {
        PetscInt coff;

        ierr = PetscSectionGetOffset(coordSection, c, &coff);CHKERRQ(ierr);
        for (v = 0; v < coneSize; ++v) off[v] = spaceDim*v + coff;
      } else {
        for (v = 0; v < coneSize; ++v) {
          ierr = PetscSectionGetOffset(coordSection, cone[v], &off[v]);CHKERRQ(ierr);
        }
      }
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] = 0.0;
      for (v = 0; v < coneSize; ++v) {ierr = DMLocalizeAddCoordinate_Internal(dm, spaceDim, &coords[off[0]], &coords[off[v]], &coordsNew[offnew]);CHKERRQ(ierr);}
      for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] /= coneSize;
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);

      /* Localize new coordinates on each refined cell */
      if (cdof) {
        PetscInt *rStar = NULL, rStarSize;

        ierr = DMPlexGetTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        for (v = 0; v < rStarSize*2; v += 2) {
          if ((rStar[v] >= cStartNew) && (rStar[v] < cEndNew)) {
            PetscInt *cone = NULL, closureSize, lid, coff, rc, rcdof;

            rc   = rStar[v];
            ierr = PetscSectionGetDof(coordSectionNew, rc, &rcdof);CHKERRQ(ierr);
            if (!rcdof) continue;
            ierr = PetscSectionGetOffset(coordSectionNew, rc, &coff);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(rdm, rc, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
            for (p = 0, lid = 0; p < closureSize*2; p += 2) {
              if (cone[p] == newv) {
                for (d = 0; d < spaceDim; d++) coordsNew[coff + lid*spaceDim + d] = coordsNew[offnew + d];
                break;
              }
              if (cone[p] >= vStartNew && cone[p] < vEndNew) lid++;
            }
            if (p == closureSize*2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map new vertex %D",newv);
            ierr = DMPlexRestoreTransitiveClosure(rdm, rc, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
      }
    }
  case REFINER_SIMPLEX_2D:
  case REFINER_HYBRID_SIMPLEX_2D:
  case REFINER_SIMPLEX_3D:
  case REFINER_HYBRID_SIMPLEX_3D:
    /* Edge vertices have the average of endpoint coordinates */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone;
      PetscInt        coneSize, offA, offB, offnew, d;

      ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      if (coneSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %D cone should have two vertices, not %D", e, coneSize);
      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      if (localize) {
        PetscInt   coff, toffA = -1, toffB = -1, voffA, voffB;
        PetscInt  *eStar = NULL, eStarSize;
        PetscInt  *rStar = NULL, rStarSize;
        PetscBool  cellfound = PETSC_FALSE;

        offA = offB = -1;
        ierr = PetscSectionGetOffset(coordSection, cone[0], &voffA);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[1], &voffB);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &eStarSize, &eStar);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        for (v = 0; v < eStarSize*2; v += 2) {
          if ((eStar[v] >= cStart) && (eStar[v] < cEnd)) {
            PetscScalar     coordsNewAux[3] = {0., 0., 0.};
            PetscInt       *cellCone = NULL;
            PetscInt        cellClosureSize, s, cv, cdof;
            PetscBool       locvA = PETSC_TRUE, locvB = PETSC_TRUE;
            const PetscInt  cell = eStar[v];

            ierr = PetscSectionGetDof(coordSection, cell, &cdof);CHKERRQ(ierr);
            if (!cdof) {
              /* Found a valid edge for the "vertex" part of the Section */
              offA = voffA;
              offB = voffB;
              cellfound = PETSC_TRUE;
            } else {
              ierr = PetscSectionGetOffset(coordSection, cell, &coff);CHKERRQ(ierr);
              ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &cellClosureSize, &cellCone);CHKERRQ(ierr);
              for (s = 0, cv = 0; s < cellClosureSize*2; s += 2) {
                const PetscInt point = cellCone[s];
                if ((point >= vStart) && (point < vEnd)) {
                  if (point == cone[0]) toffA = spaceDim*cv + coff;
                  else if (point == cone[1]) toffB = spaceDim*cv + coff;
                  cv++;
                }
              }
              ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &cellClosureSize, &cellCone);CHKERRQ(ierr);
              for (d = 0; d < spaceDim; ++d) {
                coordsNewAux[d] = 0.5*(coords[toffA+d] + coords[toffB+d]);
                if (coords[toffA+d] != coords[voffA+d]) locvA = PETSC_FALSE;
                if (coords[toffB+d] != coords[voffB+d]) locvB = PETSC_FALSE;
              }
              /* Found a valid edge for the "vertex" part of the Section */
              if (!cellfound && (locvA || locvB)) {
                cellfound = PETSC_TRUE;
                offA = toffA;
                offB = toffB;
              }
            }

            /* Localize new coordinates on each refined cell */
            for (s = 0; s < rStarSize*2; s += 2) {
              if ((rStar[s] >= cStartNew) && (rStar[s] < cEndNew) && parentId[rStar[s]-cStartNew] == cell) {
                PetscInt       *rcone = NULL, rclosureSize, lid, p, rcdof;
                const PetscInt  rcell = rStar[s];

                ierr = PetscSectionGetDof(coordSectionNew, rcell, &rcdof);CHKERRQ(ierr);
                if (!rcdof) continue;
                ierr = PetscSectionGetOffset(coordSectionNew, rcell, &coff);CHKERRQ(ierr);
                ierr = DMPlexGetTransitiveClosure(rdm, rcell, PETSC_TRUE, &rclosureSize, &rcone);CHKERRQ(ierr);
                for (p = 0, lid = 0; p < rclosureSize*2; p += 2) {
                  if (rcone[p] == newv) {
                    for (d = 0; d < spaceDim; d++) coordsNew[coff + lid*spaceDim + d] = coordsNewAux[d];
                    break;
                  }
                  if (rcone[p] >= vStartNew && rcone[p] < vEndNew) lid++;
                }
                ierr = DMPlexRestoreTransitiveClosure(rdm, rcell, PETSC_TRUE, &rclosureSize, &rcone);CHKERRQ(ierr);
                if (p == rclosureSize*2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map new vertex %D",newv);
              }
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &eStarSize, &eStar);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        if (!cellfound) {
          /* Could not find a valid edge for the vertex part, we will get this vertex later (final reduction) */
          needcoords = PETSC_TRUE;
        }
      } else {
        ierr = PetscSectionGetOffset(coordSection, cone[0], &offA);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[1], &offB);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      if (offA != -1 && offB != -1) {
        ierr = DMLocalizeCoordinate_Internal(dm, spaceDim, &coords[offA], &coords[offB], &coordsNew[offnew]);CHKERRQ(ierr);
        for (d = 0; d < spaceDim; ++d) {
          coordsNew[offnew+d] = 0.5*(coords[offA+d] + coordsNew[offnew+d]);
        }
      } else {
        for (d = 0; d < spaceDim; ++d) coordsNew[offnew+d] = PETSC_MIN_REAL;
      }
    }
    /* Old vertices have the same coordinates */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newv = vStartNew + (v - vStart);
      PetscInt       off, offnew, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < spaceDim; ++d) {
        coordsNew[offnew+d] = coords[off+d];
      }

      /* Localize new coordinates on each refined cell */
      if (localize) {
        PetscInt  p;
        PetscInt *rStar = NULL, rStarSize;

        ierr = DMPlexGetTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
        for (p = 0; p < rStarSize*2; p += 2) {
          if ((rStar[p] >= cStartNew) && (rStar[p] < cEndNew)) {
            PetscScalar  ocoords[3] = {0,0,0}; /* dummy values for compiler warnings about uninitialized values */
            PetscInt    *cone = NULL, closureSize, lid, coff, s, oc, cdof;

            c    = rStar[p];
            oc   = parentId[c-cStartNew];
            ierr = PetscSectionGetDof(coordSectionNew, c, &cdof);CHKERRQ(ierr);
            if (!cdof) continue;
            ierr = PetscSectionGetDof(coordSection, oc, &cdof);CHKERRQ(ierr);
            if (!cdof) continue;
            ierr = PetscSectionGetOffset(coordSection, oc, &coff);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(dm, oc, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
            for (s = 0, lid = 0; s < closureSize*2; s += 2) {
              if (cone[s] == v) {
                for (d = 0; d < spaceDim; d++) ocoords[d] = coords[coff + lid*spaceDim + d];
                break;
              }
              if (cone[s] >= vStart && cone[s] < vEnd) lid++;
            }
            if (s == closureSize*2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map old vertex %D",v);
            ierr = DMPlexRestoreTransitiveClosure(dm, oc, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);

            ierr = PetscSectionGetOffset(coordSectionNew, c, &coff);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(rdm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
            for (s = 0, lid = 0; s < closureSize*2; s += 2) {
              if (cone[s] == newv) {
                for (d = 0; d < spaceDim; d++) coordsNew[coff + lid*spaceDim + d] = ocoords[d];
                break;
              }
              if (cone[s] >= vStartNew && cone[s] < vEndNew) lid++;
            }
            if (s == closureSize*2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to map new vertex %D",newv);
            ierr = DMPlexRestoreTransitiveClosure(rdm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(rdm, newv, PETSC_FALSE, &rStarSize, &rStar);CHKERRQ(ierr);
      }
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(rdm, coordinatesNew);CHKERRQ(ierr);

  /* Final reduction (if needed) if we are localizing */
  if (localize) {
    PetscBool gred;

    ierr = MPIU_Allreduce(&needcoords, &gred, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)rdm));CHKERRQ(ierr);
    if (gred) {
      DM                 cdm;
      Vec                aux;
      PetscSF            sf;
      const PetscScalar *lArray;
      PetscScalar       *gArray;
#if defined(PETSC_USE_COMPLEX)
      PetscInt          i, ln, gn;
      PetscReal         *lrArray;
      PetscReal         *grArray;
#endif

      ierr = DMGetCoordinateDM(rdm, &cdm);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(cdm, &aux);CHKERRQ(ierr);
      ierr = DMGetSectionSF(cdm, &sf);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coordinatesNew, &lArray);CHKERRQ(ierr);
      ierr = VecSet(aux, PETSC_MIN_REAL);CHKERRQ(ierr);
      ierr = VecGetArray(aux, &gArray);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = VecGetLocalSize(aux, &gn);CHKERRQ(ierr);
      ierr = VecGetLocalSize(coordinatesNew, &ln);CHKERRQ(ierr);
      ierr = PetscMalloc2(ln,&lrArray,gn,&grArray);CHKERRQ(ierr);
      for (i=0;i<ln;i++) lrArray[i] = PetscRealPart(lArray[i]);
      for (i=0;i<gn;i++) grArray[i] = PetscRealPart(gArray[i]);
      ierr = PetscSFReduceBegin(sf, MPIU_REAL, lrArray, grArray, MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf, MPIU_REAL, lrArray, grArray, MPIU_MAX);CHKERRQ(ierr);
      for (i=0;i<gn;i++) gArray[i] = grArray[i];
      ierr = PetscFree2(lrArray,grArray);CHKERRQ(ierr);
#else
      ierr = PetscSFReduceBegin(sf, MPIU_SCALAR, lArray, gArray, MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, MPIU_MAX);CHKERRQ(ierr);
#endif
      ierr = VecRestoreArrayRead(coordinatesNew, &lArray);CHKERRQ(ierr);
      ierr = VecRestoreArray(aux, &gArray);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(cdm, aux, INSERT_VALUES, coordinatesNew);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(cdm, aux, INSERT_VALUES, coordinatesNew);CHKERRQ(ierr);
      ierr = VecDestroy(&aux);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&coordinatesNew);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&coordSectionNew);CHKERRQ(ierr);
  ierr = PetscFree(parentId);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateProcessSF - Create an SF which just has process connectivity

  Collective on dm

  Input Parameters:
+ dm      - The DM
- sfPoint - The PetscSF which encodes point connectivity

  Output Parameters:
+ processRanks - A list of process neighbors, or NULL
- sfProcess    - An SF encoding the process connectivity, or NULL

  Level: developer

.seealso: PetscSFCreate(), DMPlexCreateTwoSidedProcessSF()
@*/
PetscErrorCode DMPlexCreateProcessSF(DM dm, PetscSF sfPoint, IS *processRanks, PetscSF *sfProcess)
{
  PetscInt           numRoots, numLeaves, l;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt          *ranks, *ranksNew;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 3);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 4);}
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &ranks);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranks[l] = remotePoints[l].rank;
  }
  ierr = PetscSortRemoveDupsInt(&numLeaves, ranks);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &ranksNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &remotePointsNew);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranksNew[l]              = ranks[l];
    localPointsNew[l]        = l;
    remotePointsNew[l].index = 0;
    remotePointsNew[l].rank  = ranksNew[l];
  }
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  if (processRanks) {ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), numLeaves, ranksNew, PETSC_OWN_POINTER, processRanks);CHKERRQ(ierr);}
  else              {ierr = PetscFree(ranksNew);CHKERRQ(ierr);}
  if (sfProcess) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *sfProcess, "Process SF");CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(*sfProcess);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*sfProcess, size, numLeaves, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerCreateSF(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscSF            sf, sfNew, sfProcess;
  IS                 processRanks;
  MPI_Datatype       depthType;
  PetscInt           numRoots, numLeaves, numLeavesNew = 0, l, m;
  const PetscInt    *localPoints, *neighbors;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt          *depthSizeOld, *rdepthSize, *rdepthSizeOld, *rdepthMaxOld, *rvStart, *rvStartNew, *reStart, *reStartNew, *rfStart, *rfStartNew, *rcStart, *rcStartNew;
  PetscInt           ldepth, depth, numNeighbors, pStartNew, pEndNew, cStart, cEnd, cMax, vStart, vEnd, vMax, fStart, fEnd, fMax, eStart, eEnd, eMax, r, n;
  PetscInt           cStartNew = 0, vStartNew = 0, fStartNew = 0, eStartNew = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(rdm, &pStartNew, &pEndNew);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &ldepth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&ldepth, &depth, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  if ((ldepth >= 0) && (depth != ldepth)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent Plex depth %D != %D", ldepth, depth);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  cMax = cMax < 0 ? cEnd : cMax;
  fMax = fMax < 0 ? fEnd : fMax;
  eMax = eMax < 0 ? eEnd : eMax;
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetPointSF(rdm, &sfNew);CHKERRQ(ierr);
  /* Calculate size of new SF */
  ierr = PetscSFGetGraph(sf, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  for (l = 0; l < numLeaves; ++l) {
    const PetscInt p = localPoints[l];

    switch (refiner) {
    case REFINER_SIMPLEX_1D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= cStart && p < cMax)) {
        /* Interior cells add new cells and interior vertices */
        numLeavesNew += 2 + 1;
      }
      break;
    case REFINER_SIMPLEX_2D:
    case REFINER_HYBRID_SIMPLEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces stay the same */
        ++numLeavesNew;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells and interior faces */
        numLeavesNew += 4 + 3;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and hybrid face */
        numLeavesNew += 2 + 1;
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    case REFINER_SIMPLEX_TO_HEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Interior faces add new faces and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, interior faces, and vertex */
        numLeavesNew += 3 + 3 + 1;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells, interior faces, and vertex */
        numLeavesNew += 4 + 4 + 1;
      }
      break;
    case REFINER_HEX_2D:
    case REFINER_HYBRID_HEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces stay the same */
        ++numLeavesNew;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, interior faces, and vertex */
        numLeavesNew += 4 + 4 + 1;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and hybrid face */
        numLeavesNew += 2 + 1;
      }
      break;
    case REFINER_SIMPLEX_3D:
    case REFINER_HYBRID_SIMPLEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and edges */
        numLeavesNew += 4 + 3;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, and edges */
        numLeavesNew += 8 + 8 + 1;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and faces */
        numLeavesNew += 4 + 3;
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
    case REFINER_SIMPLEX_TO_HEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces, edges and a vertex */
        numLeavesNew += 3 + 3 + 1;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and an edge */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, edges and a vertex */
        numLeavesNew += 4 + 6 + 4 + 1;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells, faces and an edge */
        numLeavesNew += 3 + 3 + 1;
      }
      break;
    case REFINER_HEX_3D:
    case REFINER_HYBRID_HEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges, and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces, edges, and vertex */
        numLeavesNew += 4 + 4 + 1;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, edges, and vertex */
        numLeavesNew += 8 + 12 + 6 + 1;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Hybrid cells add new cells, faces, and edges */
        numLeavesNew += 4 + 4 + 1;
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
    }
  }
  /* Communicate depthSizes for each remote rank */
  ierr = DMPlexCreateProcessSF(dm, sf, &processRanks, &sfProcess);CHKERRQ(ierr);
  ierr = ISGetLocalSize(processRanks, &numNeighbors);CHKERRQ(ierr);
  ierr = PetscMalloc5((depth+1)*numNeighbors,&rdepthSize,numNeighbors,&rvStartNew,numNeighbors,&reStartNew,numNeighbors,&rfStartNew,numNeighbors,&rcStartNew);CHKERRQ(ierr);
  ierr = PetscMalloc7(depth+1,&depthSizeOld,(depth+1)*numNeighbors,&rdepthSizeOld,(depth+1)*numNeighbors,&rdepthMaxOld,numNeighbors,&rvStart,numNeighbors,&reStart,numNeighbors,&rfStart,numNeighbors,&rcStart);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(depth+1, MPIU_INT, &depthType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&depthType);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSize, rdepthSize);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSize, rdepthSize);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    ierr = GetDepthStart_Private(depth, &rdepthSize[n*(depth+1)], &rcStartNew[n], &rfStartNew[n], &reStartNew[n], &rvStartNew[n]);CHKERRQ(ierr);
  }
  depthSizeOld[depth]   = cMax;
  depthSizeOld[0]       = vMax;
  depthSizeOld[depth-1] = fMax;
  depthSizeOld[1]       = eMax;

  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSizeOld, rdepthMaxOld);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSizeOld, rdepthMaxOld);CHKERRQ(ierr);

  depthSizeOld[depth]   = cEnd - cStart;
  depthSizeOld[0]       = vEnd - vStart;
  depthSizeOld[depth-1] = fEnd - fStart;
  depthSizeOld[1]       = eEnd - eStart;

  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSizeOld, rdepthSizeOld);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSizeOld, rdepthSizeOld);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    ierr = GetDepthStart_Private(depth, &rdepthSizeOld[n*(depth+1)], &rcStart[n], &rfStart[n], &reStart[n], &rvStart[n]);CHKERRQ(ierr);
    rdepthMaxOld[n*(depth+1)+depth]   = rdepthMaxOld[n*(depth+1)+depth]   < 0 ? rdepthSizeOld[n*(depth+1)+depth]  +rcStart[n]: rdepthMaxOld[n*(depth+1)+depth];
    rdepthMaxOld[n*(depth+1)+depth-1] = rdepthMaxOld[n*(depth+1)+depth-1] < 0 ? rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n]: rdepthMaxOld[n*(depth+1)+depth-1];
    rdepthMaxOld[n*(depth+1)+1]       = rdepthMaxOld[n*(depth+1)+1]       < 0 ? rdepthSizeOld[n*(depth+1)+1]      +reStart[n]: rdepthMaxOld[n*(depth+1)+1];
  }
  ierr = MPI_Type_free(&depthType);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfProcess);CHKERRQ(ierr);
  /* Calculate new point SF */
  ierr = PetscMalloc1(numLeavesNew, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeavesNew, &remotePointsNew);CHKERRQ(ierr);
  ierr = ISGetIndices(processRanks, &neighbors);CHKERRQ(ierr);
  for (l = 0, m = 0; l < numLeaves; ++l) {
    PetscInt    p     = localPoints[l];
    PetscInt    rp    = remotePoints[l].index, n;
    PetscMPIInt rrank = remotePoints[l].rank;

    ierr = PetscFindInt(rrank, numNeighbors, neighbors, &n);CHKERRQ(ierr);
    if (n < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not locate remote rank %D", rrank);
    switch (refiner) {
    case REFINER_SIMPLEX_1D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old interior cells add new cells and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*2     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - cStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rcStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case REFINER_SIMPLEX_2D:
    case REFINER_HYBRID_SIMPLEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Old interior faces add new faces and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Old hybrid faces stay the same */
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - fMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old interior cells add new cells and interior faces */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - cStart)*3     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rcStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Old hybrid cells add new cells and hybrid face */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (cMax                            - cStart)*3     + (p  - cMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*3 + (rp - rdepthMaxOld[n*(depth+1)+depth]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    case REFINER_SIMPLEX_TO_HEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old interior faces add new faces and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old interior cells add new cells, interior faces, and a vertex */
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*3     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*2                    + (p  - cStart)*3     + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*2 + (rp - rcStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (fEnd - fStart)                    + (p  - cStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + rdepthSizeOld[n*(depth+1)+depth-1] + (rp - rcStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Old interior hybrid cells add new cells, interior faces, and a vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (cMax  - cStart)*3                               + (p  - cMax)*4 + r;
          remotePointsNew[m].index = rcStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*3 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*2                    + (cMax  - cStart)*3                               + (p - cMax)*4 + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*2 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*3 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (fEnd - fStart)                    + (p  - cStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + rdepthSizeOld[n*(depth+1)+depth-1] + (rp - rcStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case REFINER_HEX_2D:
    case REFINER_HYBRID_HEX_2D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Old interior faces add new faces and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Old hybrid faces stay the same */
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - fMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old interior cells add new cells, interior faces, and vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)               + (fMax                              - fStart)     + (p  - cStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0]  + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n]) + (rp - rcStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old hybrid cells add new cells and hybrid face */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r; /* TODO: is this a bug? */
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r; /* TODO: is this a bug? */
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (cMax                            - cStart)*4     + (p  - cMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*4 + (rp - rdepthMaxOld[n*(depth+1)+depth]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case REFINER_SIMPLEX_3D:
    case REFINER_HYBRID_SIMPLEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (cMax                            - cStart)     + (p  - eMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n]) + (rp - rdepthMaxOld[n*(depth+1)+1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and edges */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (p  - fStart)*3     + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rp - rfStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*8     + (p  - fMax)*2                              + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rp - rdepthMaxOld[n*(depth+1)+depth-1])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (cMax                            - cStart)     + (eEnd                                    - eMax)                        + (p  - fMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n]) + (rdepthSizeOld[n*(depth+1)+1]+reStart[n] - rdepthMaxOld[n*(depth+1)+1]) + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, and edges */
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (p  - cStart)*1     + 0;
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rp - rcStart[n])*1 + 0;
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and faces */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (cMax                            - cStart)*8     + (p  - cMax)*4                            + r;
          remotePointsNew[m].index = rcStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*8     + (fEnd                                          - fMax)*2                              + (p  - cMax)*3                            + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
    case REFINER_SIMPLEX_TO_HEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (cMax                            - cStart)*4     + (p  - eMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*4 + (rp - rdepthMaxOld[n*(depth+1)+1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces, edges and a vertex */
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*3     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax - eStart)*2                            + (p  - fStart)*3     + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rp - rfStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (eMax - eStart)                            + (p - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n]) + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Interior hybrid faces add new faces and an edge */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax - fStart)*3                                 + (cMax - cStart)*6                                + (p  - fMax)*2 + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*6 + (rp - rdepthMaxOld[n*(depth+1)+depth-1])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax - eStart)*2                            + (fMax - fStart)*3                                 + (cMax - cStart)*4                                + (eEnd - eMax)                                                           + (p  - fMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*4 + (rdepthSizeOld[n*(depth+1)+1]+reStart[n] - rdepthMaxOld[n*(depth+1)+1]) + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, edges, and a vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 6; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax - fStart)*3                                 + (p  - cStart)*6     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rp - rcStart[n])*6 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax - eStart)*2                           + (fMax - fStart)*3                                 + (p  - cStart)*4 + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1]- reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (eMax - eStart)                           + (fMax - fStart)                                 + (p  - cStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rdepthMaxOld[n*(depth+1)+1]- reStart[n]) + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n]) + (rp - rcStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Interior hybrid cells add new cells, faces and an edge */
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (cMax                            - cStart)*4     + (p  - cMax)*3                            + r;
          remotePointsNew[m].index = rcStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*4 + (rp - rdepthMaxOld[n*(depth+1)+depth])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax - fStart)*3                                 + (cMax - cStart)*6                                + (fEnd  - fMax)*2                                                                      + (p  - cMax)*3 + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*6 + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax - eStart)*2                           + (fMax - fStart)*3                                 + (cMax - cStart)*4                                + (eEnd  - eMax)                                                          + (fEnd  - fMax)                                                                      + (p  - cMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1]- reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1]- rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*4 + (rdepthSizeOld[n*(depth+1)+1]+reStart[n] - rdepthMaxOld[n*(depth+1)+1]) + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1]) + (rp - rdepthMaxOld[n*(depth+1)+depth]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case REFINER_HEX_3D:
    case REFINER_HYBRID_HEX_3D:
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*4     + (cMax                            - cStart)*6     + (p  - eMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*6 + (rp - rdepthMaxOld[n*(depth+1)+1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces, edges, and vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (eMax                        - eStart)     + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n]) + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*12     + (p  - fMax)*2                              + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*12 + (rp - rdepthMaxOld[n*(depth+1)+depth-1])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*4     + (cMax                            - cStart)*6     + (eEnd                                    - eMax)                        + (p  - fMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*6 + (rdepthSizeOld[n*(depth+1)+1]+reStart[n] - rdepthMaxOld[n*(depth+1)+1]) + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, edges, and vertex */
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 12; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (p  - cStart)*12     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rp - rcStart[n])*12 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 6; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*4     + (p  - cStart)*6     + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rp - rcStart[n])*6 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 1; ++r, ++m) {
          localPointsNew[m]        = vStartNew     + (eMax                        - eStart)     + (fMax                              - fStart)     + (p  - cStart)     + r;
          remotePointsNew[m].index = rvStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n]) + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n]) + (rp - rcStart[n]) + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells, faces, and edges */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (cMax                            - cStart)*8     + (p  - cMax)*4                            + r;
          remotePointsNew[m].index = rcStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*12     + (fEnd                                          - fMax)*2                              + (p  - cMax)*4                            + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*12 + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*4     + (cMax                            - cStart)*6     + (eEnd                                    - eMax)                        + (fEnd                                          - fMax)                              + (p  - cMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*6 + (rdepthSizeOld[n*(depth+1)+1]+reStart[n] - rdepthMaxOld[n*(depth+1)+1]) + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1]) + (rp - rdepthMaxOld[n*(depth+1)+depth]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
    }
  }
  if (m != numLeavesNew) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of leaf point %D should be %D", m, numLeavesNew);
  ierr = ISRestoreIndices(processRanks, &neighbors);CHKERRQ(ierr);
  ierr = ISDestroy(&processRanks);CHKERRQ(ierr);
  {
    PetscSFNode *rp, *rtmp;
    PetscInt    *lp, *idx, *ltmp, i;

    /* SF needs sorted leaves to correct calculate Gather */
    ierr = PetscMalloc1(numLeavesNew,&idx);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesNew, &lp);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesNew, &rp);CHKERRQ(ierr);
    for (i = 0; i < numLeavesNew; ++i) {
      if ((localPointsNew[i] < pStartNew) || (localPointsNew[i] >= pEndNew)) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local SF point %D (%D) not in [%D, %D)", localPointsNew[i], i, pStartNew, pEndNew);
      idx[i] = i;
    }
    ierr = PetscSortIntWithPermutation(numLeavesNew, localPointsNew, idx);CHKERRQ(ierr);
    for (i = 0; i < numLeavesNew; ++i) {
      lp[i] = localPointsNew[idx[i]];
      rp[i] = remotePointsNew[idx[i]];
    }
    ltmp            = localPointsNew;
    localPointsNew  = lp;
    rtmp            = remotePointsNew;
    remotePointsNew = rp;
    ierr = PetscFree(idx);CHKERRQ(ierr);
    ierr = PetscFree(ltmp);CHKERRQ(ierr);
    ierr = PetscFree(rtmp);CHKERRQ(ierr);
  }
  ierr = PetscSFSetGraph(sfNew, pEndNew-pStartNew, numLeavesNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree5(rdepthSize,rvStartNew,reStartNew,rfStartNew,rcStartNew);CHKERRQ(ierr);
  ierr = PetscFree7(depthSizeOld,rdepthSizeOld,rdepthMaxOld,rvStart,reStart,rfStart,rcStart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CellRefinerCreateLabels(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscInt       numLabels, l;
  PetscInt       depth, newp, cStart, cEnd, cMax, vStart, vEnd, vMax, fStart, fEnd, fMax, eStart, eEnd, eMax, r;
  PetscInt       cStartNew = 0, vStartNew = 0, fStartNew = 0, eStartNew = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  switch (refiner) {
  case REFINER_NOOP:
  case REFINER_SIMPLEX_1D:
  case REFINER_SIMPLEX_2D:
  case REFINER_SIMPLEX_TO_HEX_2D:
  case REFINER_HEX_2D:
  case REFINER_SIMPLEX_3D:
  case REFINER_HEX_3D:
  case REFINER_SIMPLEX_TO_HEX_3D:
    break;
  case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
  case REFINER_HYBRID_SIMPLEX_3D:
  case REFINER_HYBRID_HEX_3D:
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
  case REFINER_HYBRID_SIMPLEX_2D:
  case REFINER_HYBRID_HEX_2D:
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
  case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
  }
  cMax = cMax < 0 ? cEnd : cMax;
  fMax = fMax < 0 ? fEnd : fMax;
  eMax = eMax < 0 ? eEnd : eMax;
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, labelNew;
    const char     *lname;
    PetscBool       isDepth;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        defVal;
    PetscInt        numValues, val;

    ierr = DMGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = DMCreateLabel(rdm, lname);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, lname, &label);CHKERRQ(ierr);
    ierr = DMGetLabel(rdm, lname, &labelNew);CHKERRQ(ierr);
    ierr = DMLabelGetDefaultValue(label,&defVal);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(labelNew,defVal);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    for (val = 0; val < numValues; ++val) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        numPoints, n;

      ierr = DMLabelGetStratumIS(label, values[val], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      /* Ensure refined label is created with same number of strata as
       * original (even if no entries here). */
      ierr = DMLabelAddStratum(labelNew, values[val]);CHKERRQ(ierr);
      for (n = 0; n < numPoints; ++n) {
        const PetscInt p = points[n];
        switch (refiner) {
        case REFINER_SIMPLEX_1D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = cStartNew + (p - cStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case REFINER_SIMPLEX_2D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case REFINER_HYBRID_SIMPLEX_TO_HEX_2D:
        case REFINER_SIMPLEX_TO_HEX_2D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cMax)) {
            /* Old cells add new cells, interior faces, and a vertex */
            for (r = 0; r < 3; ++r) {
              newp = cStartNew + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + p;
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Old hybrid cells add new cells, interior faces, and a vertex */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (cMax - cStart)*3 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (cMax - cStart)*3 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + p;
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_HEX_2D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces and vertex */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_HYBRID_SIMPLEX_2D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Old interior faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Old hybrid faces stay the same */
            newp = fStartNew + (fMax - fStart)*2 + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Old interior cells add new cells and interior faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Old hybrid cells add new cells and hybrid face */
            for (r = 0; r < 2; ++r) {
              newp = cStartNew + (cMax - cStart)*4 + (p - cMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (p - cMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_HYBRID_HEX_2D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Old interior faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Old hybrid faces stay the same */
            newp = fStartNew + (fMax - fStart)*2 + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Old interior cells add new cells, interior faces, and vertex */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Old hybrid cells add new cells and hybrid face */
            for (r = 0; r < 2; ++r) {
              newp = cStartNew + (cMax - cStart)*4 + (p - cMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*4 + (p - cMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_SIMPLEX_3D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eEnd)) {
            /* Old edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and edges */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces and edges */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 8; ++r) {
              newp = fStartNew + (fEnd - fStart)*4 + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 1; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (p - cStart)*1 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case REFINER_HYBRID_SIMPLEX_TO_HEX_3D:
        case REFINER_SIMPLEX_TO_HEX_3D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eMax)) {
            /* Interior edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eMax) && (p < eEnd)) {
            /* Hybrid edges stay the same */
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + p - eMax;
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Old faces add new faces, edges and a vertex */
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Old hybrid faces add new faces and an edge */
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (p - fMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Old cells add new cells and interior faces and edges and a vertex */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 6; ++r) {
              newp = fStartNew + (fMax - fStart)*3 + (p - cStart)*6 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + p - cStart;
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Old hybrid cells add new cells and interior faces and an edge */
            for (r = 0; r < 3; ++r) {
              newp = cStartNew + (cMax - cStart)*4 + (p - cMax)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fMax - fStart)*3 + (cMax - cStart)*6 + (fEnd - fMax)*2 + (p - cMax)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart)*4 + (eEnd - eMax) + (fEnd - fMax) + p - cMax;
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_HYBRID_SIMPLEX_3D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Interior vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eMax)) {
            /* Interior edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eMax) && (p < eEnd)) {
            /* Hybrid edges stay the same */
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (p - eMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Interior faces add new faces and edges */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Hybrid faces add new faces and edges */
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (p - fMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Interior cells add new cells, faces, and edges */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 8; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Hybrid cells add new cells and faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (cMax - cStart)*8 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (p - cMax)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case REFINER_HEX_3D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eEnd)) {
            /* Old edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces, edges, and vertex */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells, faces, edges, and vertex */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 12; ++r) {
              newp = fStartNew + (fEnd - fStart)*4 + (p - cStart)*12 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 6; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (p - cStart)*6 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case REFINER_HYBRID_HEX_3D:
          if ((p >= vStart) && (p < vEnd)) {
            /* Interior vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eMax)) {
            /* Interior edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eMax) && (p < eEnd)) {
            /* Hybrid edges stay the same */
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (p - eMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Interior faces add new faces, edges, and vertex */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Hybrid faces add new faces and edges */
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (p - fMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Interior cells add new cells, faces, edges, and vertex */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 12; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (p - cStart)*12 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 6; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (p - cStart)*6 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eMax - eStart) + (fMax - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Hybrid cells add new cells, faces, and edges */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (cMax - cStart)*8 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*12 + (fEnd - fMax)*2 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*4 + (cMax - cStart)*6 + (fEnd - fMax) + (p - cMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[refiner]);
        }
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* This will only work for interpolated meshes */
PetscErrorCode DMPlexRefineUniform_Internal(DM dm, CellRefiner cellRefiner, DM *dmRefined)
{
  DM             rdm;
  PetscInt      *depthSize;
  PetscInt       dim, embedDim, depth = 0, d, pStart = 0, pEnd = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(rdm, dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &embedDim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(rdm, embedDim);CHKERRQ(ierr);
  /* Calculate number of new points of each depth */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth >= 0 && dim != depth) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated for regular refinement");
  ierr = PetscMalloc1(depth+1, &depthSize);CHKERRQ(ierr);
  ierr = PetscMemzero(depthSize, (depth+1) * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = CellRefinerGetSizes(cellRefiner, dm, depthSize);CHKERRQ(ierr);
  /* Step 1: Set chart */
  for (d = 0; d <= depth; ++d) pEnd += depthSize[d];
  ierr = DMPlexSetChart(rdm, pStart, pEnd);CHKERRQ(ierr);
  /* Step 2: Set cone/support sizes (automatically stratifies) */
  ierr = CellRefinerSetConeSizes(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 3: Setup refined DM */
  ierr = DMSetUp(rdm);CHKERRQ(ierr);
  /* Step 4: Set cones and supports (automatically symmetrizes) */
  ierr = CellRefinerSetCones(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 5: Create pointSF */
  ierr = CellRefinerCreateSF(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 6: Create labels */
  ierr = CellRefinerCreateLabels(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 7: Set coordinates */
  ierr = CellRefinerSetCoordinates(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  ierr = PetscFree(depthSize);CHKERRQ(ierr);

  *dmRefined = rdm;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateCoarsePointIS - Creates an IS covering the coarse DM chart with the fine points as data

  Input Parameter:
. dm - The coarse DM

  Output Parameter:
. fpointIS - The IS of all the fine points which exist in the original coarse mesh

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementUniform(), DMPlexCreateSubpointIS()
@*/
PetscErrorCode DMPlexCreateCoarsePointIS(DM dm, IS *fpointIS)
{
  CellRefiner    cellRefiner;
  PetscInt      *depthSize, *fpoints;
  PetscInt       cStartNew = 0, vStartNew = 0, fStartNew = 0, eStartNew = 0;
  PetscInt       depth, pStart, pEnd, p, vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetCellRefiner_Internal(dm, &cellRefiner);CHKERRQ(ierr);
  ierr = PetscMalloc1(depth+1, &depthSize);CHKERRQ(ierr);
  ierr = CellRefinerGetSizes(cellRefiner, dm, depthSize);CHKERRQ(ierr);
  if (cellRefiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  ierr = PetscMalloc1(pEnd-pStart,&fpoints);CHKERRQ(ierr);
  for (p = 0; p < pEnd-pStart; ++p) fpoints[p] = -1;
  switch (cellRefiner) {
  case REFINER_SIMPLEX_1D:
  case REFINER_SIMPLEX_2D:
  case REFINER_HYBRID_SIMPLEX_2D:
  case REFINER_HEX_2D:
  case REFINER_HYBRID_HEX_2D:
  case REFINER_SIMPLEX_3D:
  case REFINER_HYBRID_SIMPLEX_3D:
  case REFINER_HEX_3D:
  case REFINER_HYBRID_HEX_3D:
    for (v = vStart; v < vEnd; ++v) fpoints[v-pStart] = vStartNew + (v - vStart);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %s", CellRefiners[cellRefiner]);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, pEnd-pStart, fpoints, PETSC_OWN_POINTER, fpointIS);CHKERRQ(ierr);
  ierr = PetscFree(depthSize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementUniform - Set the flag for uniform refinement

  Input Parameters:
+ dm - The DM
- refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexSetRefinementUniform(DM dm, PetscBool refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementUniform = refinementUniform;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementUniform - Retrieve the flag for uniform refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexGetRefinementUniform(DM dm, PetscBool *refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementUniform,  2);
  *refinementUniform = mesh->refinementUniform;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementLimit - Set the maximum cell volume for refinement

  Input Parameters:
+ dm - The DM
- refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementLimit(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexSetRefinementLimit(DM dm, PetscReal refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementLimit = refinementLimit;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementLimit - Retrieve the maximum cell volume for refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementLimit(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexGetRefinementLimit(DM dm, PetscReal *refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementLimit,  2);
  /* if (mesh->refinementLimit < 0) = getMaxVolume()/2.0; */
  *refinementLimit = mesh->refinementLimit;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementFunction - Set the function giving the maximum cell volume for refinement

  Input Parameters:
+ dm - The DM
- refinementFunc - Function giving the maximum cell volume in the refined mesh

  Note: The calling sequence is refinementFunc(coords, limit)
$ coords - Coordinates of the current point, usually a cell centroid
$ limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementFunction(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexSetRefinementFunction(DM dm, PetscErrorCode (*refinementFunc)(const PetscReal [], PetscReal *))
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementFunc = refinementFunc;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementFunction - Get the function giving the maximum cell volume for refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementFunc - Function giving the maximum cell volume in the refined mesh

  Note: The calling sequence is refinementFunc(coords, limit)
$ coords - Coordinates of the current point, usually a cell centroid
$ limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementFunction(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexGetRefinementFunction(DM dm, PetscErrorCode (**refinementFunc)(const PetscReal [], PetscReal *))
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementFunc,  2);
  *refinementFunc = mesh->refinementFunc;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetCellRefiner_Internal(DM dm, CellRefiner *cellRefiner)
{
  PetscInt       dim, cStart, cEnd, coneSize, cMax, fMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (cEnd <= cStart) {*cellRefiner = REFINER_NOOP; PetscFunctionReturn(0);}
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, NULL, NULL);CHKERRQ(ierr);
  switch (dim) {
  case 1:
    switch (coneSize) {
    case 2:
      *cellRefiner = REFINER_SIMPLEX_1D;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %D in dimension %D for cell refiner", coneSize, dim);
    }
    break;
  case 2:
    switch (coneSize) {
    case 3:
      if (cMax >= 0) *cellRefiner = REFINER_HYBRID_SIMPLEX_2D;
      else *cellRefiner = REFINER_SIMPLEX_2D;
      break;
    case 4:
      if (cMax >= 0 && fMax >= 0) *cellRefiner = REFINER_HYBRID_HEX_2D;
      else *cellRefiner = REFINER_HEX_2D;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %D in dimension %D for cell refiner", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 4:
      if (cMax >= 0) *cellRefiner = REFINER_HYBRID_SIMPLEX_3D;
      else *cellRefiner = REFINER_SIMPLEX_3D;
      break;
    case 5:
      if (cMax == 0) *cellRefiner = REFINER_HYBRID_SIMPLEX_3D;
      else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %D in dimension %D for cell refiner", coneSize, dim);
      break;
    case 6:
      if (cMax >= 0) *cellRefiner = REFINER_HYBRID_HEX_3D;
      else *cellRefiner = REFINER_HEX_3D;
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %D in dimension %D for cell refiner", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown dimension %D for cell refiner", dim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefine_Plex(DM dm, MPI_Comm comm, DM *dmRefined)
{
  PetscBool      isUniform;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetRefinementUniform(dm, &isUniform);CHKERRQ(ierr);
  if (isUniform) {
    CellRefiner cellRefiner;
    PetscBool   localized;

    ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
    ierr = DMPlexGetCellRefiner_Internal(dm, &cellRefiner);CHKERRQ(ierr);
    ierr = DMPlexRefineUniform_Internal(dm, cellRefiner, dmRefined);CHKERRQ(ierr);
    ierr = DMPlexSetRegularRefinement(*dmRefined, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCopyBoundary(dm, *dmRefined);CHKERRQ(ierr);
    if (localized) {ierr = DMLocalizeCoordinates(*dmRefined);CHKERRQ(ierr);}
  } else {
    ierr = DMPlexRefine_Internal(dm, NULL, dmRefined);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefineHierarchy_Plex(DM dm, PetscInt nlevels, DM dmRefined[])
{
  DM             cdm = dm;
  PetscInt       r;
  PetscBool      isUniform, localized;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetRefinementUniform(dm, &isUniform);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  if (isUniform) {
    for (r = 0; r < nlevels; ++r) {
      CellRefiner cellRefiner;

      ierr = DMPlexGetCellRefiner_Internal(cdm, &cellRefiner);CHKERRQ(ierr);
      ierr = DMPlexRefineUniform_Internal(cdm, cellRefiner, &dmRefined[r]);CHKERRQ(ierr);
      ierr = DMSetCoarsenLevel(dmRefined[r], cdm->leveldown);CHKERRQ(ierr);
      ierr = DMSetRefineLevel(dmRefined[r], cdm->levelup+1);CHKERRQ(ierr);
      ierr = DMCopyBoundary(cdm, dmRefined[r]);CHKERRQ(ierr);
      if (localized) {ierr = DMLocalizeCoordinates(dmRefined[r]);CHKERRQ(ierr);}
      ierr = DMSetCoarseDM(dmRefined[r], cdm);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(dmRefined[r], PETSC_TRUE);CHKERRQ(ierr);
      cdm  = dmRefined[r];
    }
  } else {
    for (r = 0; r < nlevels; ++r) {
      ierr = DMRefine(cdm, PetscObjectComm((PetscObject) dm), &dmRefined[r]);CHKERRQ(ierr);
      ierr = DMCopyBoundary(cdm, dmRefined[r]);CHKERRQ(ierr);
      if (localized) {ierr = DMLocalizeCoordinates(dmRefined[r]);CHKERRQ(ierr);}
      ierr = DMSetCoarseDM(dmRefined[r], cdm);CHKERRQ(ierr);
      cdm  = dmRefined[r];
    }
  }
  PetscFunctionReturn(0);
}
