#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformSetUp_BL(DMPlexTransform tr)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  const PetscInt   n  = bl->n;
  DMPolytopeType   ct;
  DM               dm;
  DMLabel          active;
  PetscInt         Nc, No, coff, i, ict;

  PetscFunctionBegin;
  /* If no label is given, split all tensor cells */
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  if (active) {
    IS              refineIS;
    const PetscInt *refineCells;
    PetscInt        c;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
    PetscCall(DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS));
    PetscCall(DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc));
    if (refineIS) PetscCall(ISGetIndices(refineIS, &refineCells));
    for (c = 0; c < Nc; ++c) {
      const PetscInt cell    = refineCells[c];
      PetscInt      *closure = NULL;
      PetscInt       Ncl, cl;

      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
      for (cl = 0; cl < Ncl; cl += 2) {
        const PetscInt point = closure[cl];

        PetscCall(DMPlexGetCellType(dm, point, &ct));
        switch (ct) {
          case DM_POLYTOPE_POINT_PRISM_TENSOR:
          case DM_POLYTOPE_SEG_PRISM_TENSOR:
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
          case DM_POLYTOPE_QUAD_PRISM_TENSOR:
            PetscCall(DMLabelSetValue(tr->trType, point, 1));break;
          default: break;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    }
  }
  /* Cell heights */
  PetscCall(PetscMalloc1(n, &bl->h));
  if (bl->r > 1.) {
    /* initial height h_0 = (r - 1) / (r^{n+1} - 1)
       cell height    h_i = r^i h_0
       so that \sum^n_{i = 0} h_i = h_0 \sum^n_{i=0} r^i = h_0 (r^{n+1} - 1) / (r - 1) = 1
    */
    PetscReal d = (bl->r - 1.)/(PetscPowRealInt(bl->r, n+1) - 1.);

    bl->h[0] = d;
    for (i = 1; i < n; ++i) {
      d *= bl->r;
      bl->h[i] = bl->h[i-1] + d;
    }
  } else {
    /* equal division */
    for (i = 0; i < n; ++i) bl->h[i] = (i + 1.)/(n + 1);
  }

  PetscCall(PetscMalloc5(DM_NUM_POLYTOPES, &bl->Nt, DM_NUM_POLYTOPES, &bl->target, DM_NUM_POLYTOPES, &bl->size, DM_NUM_POLYTOPES, &bl->cone, DM_NUM_POLYTOPES, &bl->ornt));
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    bl->Nt[ict]     = -1;
    bl->target[ict] = NULL;
    bl->size[ict]   = NULL;
    bl->cone[ict]   = NULL;
    bl->ornt[ict]   = NULL;
  }
  /* DM_POLYTOPE_POINT_PRISM_TENSOR produces n points and n+1 tensor segments */
  ct = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->Nt[ct] = 2;
  Nc = 7*2 + 6*(n - 1);
  No = 2*(n + 1);
  PetscCall(PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]));
  bl->target[ct][0] = DM_POLYTOPE_POINT;
  bl->target[ct][1] = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->size[ct][0]   = n;
  bl->size[ct][1]   = n+1;
  /*   cones for tensor segments */
  bl->cone[ct][0] = DM_POLYTOPE_POINT;
  bl->cone[ct][1] = 1;
  bl->cone[ct][2] = 0;
  bl->cone[ct][3] = 0;
  bl->cone[ct][4] = DM_POLYTOPE_POINT;
  bl->cone[ct][5] = 0;
  bl->cone[ct][6] = 0;
  for (i = 0; i < n-1; ++i) {
    bl->cone[ct][7+6*i+0] = DM_POLYTOPE_POINT;
    bl->cone[ct][7+6*i+1] = 0;
    bl->cone[ct][7+6*i+2] = i;
    bl->cone[ct][7+6*i+3] = DM_POLYTOPE_POINT;
    bl->cone[ct][7+6*i+4] = 0;
    bl->cone[ct][7+6*i+5] = i+1;
  }
  bl->cone[ct][7+6*(n-1)+0] = DM_POLYTOPE_POINT;
  bl->cone[ct][7+6*(n-1)+1] = 0;
  bl->cone[ct][7+6*(n-1)+2] = n-1;
  bl->cone[ct][7+6*(n-1)+3] = DM_POLYTOPE_POINT;
  bl->cone[ct][7+6*(n-1)+4] = 1;
  bl->cone[ct][7+6*(n-1)+5] = 1;
  bl->cone[ct][7+6*(n-1)+6] = 0;
  for (i = 0; i < No; i++) bl->ornt[ct][i] = 0;
  /* DM_POLYTOPE_SEG_PRISM_TENSOR produces n segments and n+1 tensor quads */
  ct = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->Nt[ct] = 2;
  Nc = 8*n + 15*2 + 14*(n - 1);
  No = 2*n + 4*(n + 1);
  PetscCall(PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]));
  bl->target[ct][0] = DM_POLYTOPE_SEGMENT;
  bl->target[ct][1] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->size[ct][0]   = n;
  bl->size[ct][1]   = n+1;
  /*   cones for segments */
  for (i = 0; i < n; ++i) {
    bl->cone[ct][8*i+0] = DM_POLYTOPE_POINT;
    bl->cone[ct][8*i+1] = 1;
    bl->cone[ct][8*i+2] = 2;
    bl->cone[ct][8*i+3] = i;
    bl->cone[ct][8*i+4] = DM_POLYTOPE_POINT;
    bl->cone[ct][8*i+5] = 1;
    bl->cone[ct][8*i+6] = 3;
    bl->cone[ct][8*i+7] = i;
  }
  /*   cones for tensor quads */
  coff = 8*n;
  bl->cone[ct][coff+ 0] = DM_POLYTOPE_SEGMENT;
  bl->cone[ct][coff+ 1] = 1;
  bl->cone[ct][coff+ 2] = 0;
  bl->cone[ct][coff+ 3] = 0;
  bl->cone[ct][coff+ 4] = DM_POLYTOPE_SEGMENT;
  bl->cone[ct][coff+ 5] = 0;
  bl->cone[ct][coff+ 6] = 0;
  bl->cone[ct][coff+ 7] = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->cone[ct][coff+ 8] = 1;
  bl->cone[ct][coff+ 9] = 2;
  bl->cone[ct][coff+10] = 0;
  bl->cone[ct][coff+11] = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->cone[ct][coff+12] = 1;
  bl->cone[ct][coff+13] = 3;
  bl->cone[ct][coff+14] = 0;
  for (i = 0; i < n-1; ++i) {
    bl->cone[ct][coff+15+14*i+ 0] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][coff+15+14*i+ 1] = 0;
    bl->cone[ct][coff+15+14*i+ 2] = i;
    bl->cone[ct][coff+15+14*i+ 3] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][coff+15+14*i+ 4] = 0;
    bl->cone[ct][coff+15+14*i+ 5] = i+1;
    bl->cone[ct][coff+15+14*i+ 6] = DM_POLYTOPE_POINT_PRISM_TENSOR;
    bl->cone[ct][coff+15+14*i+ 7] = 1;
    bl->cone[ct][coff+15+14*i+ 8] = 2;
    bl->cone[ct][coff+15+14*i+ 9] = i+1;
    bl->cone[ct][coff+15+14*i+10] = DM_POLYTOPE_POINT_PRISM_TENSOR;
    bl->cone[ct][coff+15+14*i+11] = 1;
    bl->cone[ct][coff+15+14*i+12] = 3;
    bl->cone[ct][coff+15+14*i+13] = i+1;
  }
  bl->cone[ct][coff+15+14*(n-1)+ 0] = DM_POLYTOPE_SEGMENT;
  bl->cone[ct][coff+15+14*(n-1)+ 1] = 0;
  bl->cone[ct][coff+15+14*(n-1)+ 2] = n-1;
  bl->cone[ct][coff+15+14*(n-1)+ 3] = DM_POLYTOPE_SEGMENT;
  bl->cone[ct][coff+15+14*(n-1)+ 4] = 1;
  bl->cone[ct][coff+15+14*(n-1)+ 5] = 1;
  bl->cone[ct][coff+15+14*(n-1)+ 6] = 0;
  bl->cone[ct][coff+15+14*(n-1)+ 7] = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->cone[ct][coff+15+14*(n-1)+ 8] = 1;
  bl->cone[ct][coff+15+14*(n-1)+ 9] = 2;
  bl->cone[ct][coff+15+14*(n-1)+10] = n;
  bl->cone[ct][coff+15+14*(n-1)+11] = DM_POLYTOPE_POINT_PRISM_TENSOR;
  bl->cone[ct][coff+15+14*(n-1)+12] = 1;
  bl->cone[ct][coff+15+14*(n-1)+13] = 3;
  bl->cone[ct][coff+15+14*(n-1)+14] = n;
  for (i = 0; i < No; i++) bl->ornt[ct][i] = 0;
  /* DM_POLYTOPE_TRI_PRISM_TENSOR produces n triangles and n+1 tensor triangular prisms */
  ct = DM_POLYTOPE_TRI_PRISM_TENSOR;
  bl->Nt[ct] = 2;
  Nc = 12*n + 19*2 + 18*(n - 1);
  No = 3*n + 5*(n + 1);
  PetscCall(PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]));
  bl->target[ct][0] = DM_POLYTOPE_TRIANGLE;
  bl->target[ct][1] = DM_POLYTOPE_TRI_PRISM_TENSOR;
  bl->size[ct][0]   = n;
  bl->size[ct][1]   = n+1;
  /*   cones for triangles */
  for (i = 0; i < n; ++i) {
    bl->cone[ct][12*i+ 0] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][12*i+ 1] = 1;
    bl->cone[ct][12*i+ 2] = 2;
    bl->cone[ct][12*i+ 3] = i;
    bl->cone[ct][12*i+ 4] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][12*i+ 5] = 1;
    bl->cone[ct][12*i+ 6] = 3;
    bl->cone[ct][12*i+ 7] = i;
    bl->cone[ct][12*i+ 8] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][12*i+ 9] = 1;
    bl->cone[ct][12*i+10] = 4;
    bl->cone[ct][12*i+11] = i;
  }
  /*   cones for triangular prisms */
  coff = 12*n;
  bl->cone[ct][coff+ 0] = DM_POLYTOPE_TRIANGLE;
  bl->cone[ct][coff+ 1] = 1;
  bl->cone[ct][coff+ 2] = 0;
  bl->cone[ct][coff+ 3] = 0;
  bl->cone[ct][coff+ 4] = DM_POLYTOPE_TRIANGLE;
  bl->cone[ct][coff+ 5] = 0;
  bl->cone[ct][coff+ 6] = 0;
  bl->cone[ct][coff+ 7] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+ 8] = 1;
  bl->cone[ct][coff+ 9] = 2;
  bl->cone[ct][coff+10] = 0;
  bl->cone[ct][coff+11] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+12] = 1;
  bl->cone[ct][coff+13] = 3;
  bl->cone[ct][coff+14] = 0;
  bl->cone[ct][coff+15] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+16] = 1;
  bl->cone[ct][coff+17] = 4;
  bl->cone[ct][coff+18] = 0;
  for (i = 0; i < n-1; ++i) {
    bl->cone[ct][coff+19+18*i+ 0] = DM_POLYTOPE_TRIANGLE;
    bl->cone[ct][coff+19+18*i+ 1] = 0;
    bl->cone[ct][coff+19+18*i+ 2] = i;
    bl->cone[ct][coff+19+18*i+ 3] = DM_POLYTOPE_TRIANGLE;
    bl->cone[ct][coff+19+18*i+ 4] = 0;
    bl->cone[ct][coff+19+18*i+ 5] = i+1;
    bl->cone[ct][coff+19+18*i+ 6] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+19+18*i+ 7] = 1;
    bl->cone[ct][coff+19+18*i+ 8] = 2;
    bl->cone[ct][coff+19+18*i+ 9] = i+1;
    bl->cone[ct][coff+19+18*i+10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+19+18*i+11] = 1;
    bl->cone[ct][coff+19+18*i+12] = 3;
    bl->cone[ct][coff+19+18*i+13] = i+1;
    bl->cone[ct][coff+19+18*i+14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+19+18*i+15] = 1;
    bl->cone[ct][coff+19+18*i+16] = 4;
    bl->cone[ct][coff+19+18*i+17] = i+1;
  }
  bl->cone[ct][coff+19+18*(n-1)+ 0] = DM_POLYTOPE_TRIANGLE;
  bl->cone[ct][coff+19+18*(n-1)+ 1] = 0;
  bl->cone[ct][coff+19+18*(n-1)+ 2] = n-1;
  bl->cone[ct][coff+19+18*(n-1)+ 3] = DM_POLYTOPE_TRIANGLE;
  bl->cone[ct][coff+19+18*(n-1)+ 4] = 1;
  bl->cone[ct][coff+19+18*(n-1)+ 5] = 1;
  bl->cone[ct][coff+19+18*(n-1)+ 6] = 0;
  bl->cone[ct][coff+19+18*(n-1)+ 7] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+19+18*(n-1)+ 8] = 1;
  bl->cone[ct][coff+19+18*(n-1)+ 9] = 2;
  bl->cone[ct][coff+19+18*(n-1)+10] = n;
  bl->cone[ct][coff+19+18*(n-1)+11] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+19+18*(n-1)+12] = 1;
  bl->cone[ct][coff+19+18*(n-1)+13] = 3;
  bl->cone[ct][coff+19+18*(n-1)+14] = n;
  bl->cone[ct][coff+19+18*(n-1)+15] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+19+18*(n-1)+16] = 1;
  bl->cone[ct][coff+19+18*(n-1)+17] = 4;
  bl->cone[ct][coff+19+18*(n-1)+18] = n;
  for (i = 0; i < No; ++i) bl->ornt[ct][i] = 0;
  /* DM_POLYTOPE_QUAD_PRISM_TENSOR produces n quads and n+1 tensor quad prisms */
  ct = DM_POLYTOPE_QUAD_PRISM_TENSOR;
  bl->Nt[ct] = 2;
  Nc = 16*n + 23*2 + 22*(n - 1);
  No = 4*n + 6*(n + 1);
  PetscCall(PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]));
  bl->target[ct][0] = DM_POLYTOPE_QUADRILATERAL;
  bl->target[ct][1] = DM_POLYTOPE_QUAD_PRISM_TENSOR;
  bl->size[ct][0]   = n;
  bl->size[ct][1]   = n+1;
  /*  cones for quads */
  for (i = 0; i < n; ++i) {
    bl->cone[ct][16*i+ 0] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][16*i+ 1] = 1;
    bl->cone[ct][16*i+ 2] = 2;
    bl->cone[ct][16*i+ 3] = i;
    bl->cone[ct][16*i+ 4] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][16*i+ 5] = 1;
    bl->cone[ct][16*i+ 6] = 3;
    bl->cone[ct][16*i+ 7] = i;
    bl->cone[ct][16*i+ 8] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][16*i+ 9] = 1;
    bl->cone[ct][16*i+10] = 4;
    bl->cone[ct][16*i+11] = i;
    bl->cone[ct][16*i+12] = DM_POLYTOPE_SEGMENT;
    bl->cone[ct][16*i+13] = 1;
    bl->cone[ct][16*i+14] = 5;
    bl->cone[ct][16*i+15] = i;
  }
  /*   cones for quad prisms */
  coff = 16*n;
  bl->cone[ct][coff+ 0] = DM_POLYTOPE_QUADRILATERAL;
  bl->cone[ct][coff+ 1] = 1;
  bl->cone[ct][coff+ 2] = 0;
  bl->cone[ct][coff+ 3] = 0;
  bl->cone[ct][coff+ 4] = DM_POLYTOPE_QUADRILATERAL;
  bl->cone[ct][coff+ 5] = 0;
  bl->cone[ct][coff+ 6] = 0;
  bl->cone[ct][coff+ 7] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+ 8] = 1;
  bl->cone[ct][coff+ 9] = 2;
  bl->cone[ct][coff+10] = 0;
  bl->cone[ct][coff+11] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+12] = 1;
  bl->cone[ct][coff+13] = 3;
  bl->cone[ct][coff+14] = 0;
  bl->cone[ct][coff+15] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+16] = 1;
  bl->cone[ct][coff+17] = 4;
  bl->cone[ct][coff+18] = 0;
  bl->cone[ct][coff+19] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+20] = 1;
  bl->cone[ct][coff+21] = 5;
  bl->cone[ct][coff+22] = 0;
  for (i = 0; i < n-1; ++i) {
    bl->cone[ct][coff+23+22*i+ 0] = DM_POLYTOPE_QUADRILATERAL;
    bl->cone[ct][coff+23+22*i+ 1] = 0;
    bl->cone[ct][coff+23+22*i+ 2] = i;
    bl->cone[ct][coff+23+22*i+ 3] = DM_POLYTOPE_QUADRILATERAL;
    bl->cone[ct][coff+23+22*i+ 4] = 0;
    bl->cone[ct][coff+23+22*i+ 5] = i+1;
    bl->cone[ct][coff+23+22*i+ 6] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+23+22*i+ 7] = 1;
    bl->cone[ct][coff+23+22*i+ 8] = 2;
    bl->cone[ct][coff+23+22*i+ 9] = i+1;
    bl->cone[ct][coff+23+22*i+10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+23+22*i+11] = 1;
    bl->cone[ct][coff+23+22*i+12] = 3;
    bl->cone[ct][coff+23+22*i+13] = i+1;
    bl->cone[ct][coff+23+22*i+14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+23+22*i+15] = 1;
    bl->cone[ct][coff+23+22*i+16] = 4;
    bl->cone[ct][coff+23+22*i+17] = i+1;
    bl->cone[ct][coff+23+22*i+18] = DM_POLYTOPE_SEG_PRISM_TENSOR;
    bl->cone[ct][coff+23+22*i+19] = 1;
    bl->cone[ct][coff+23+22*i+20] = 5;
    bl->cone[ct][coff+23+22*i+21] = i+1;
  }
  bl->cone[ct][coff+23+22*(n-1)+ 0] = DM_POLYTOPE_QUADRILATERAL;
  bl->cone[ct][coff+23+22*(n-1)+ 1] = 0;
  bl->cone[ct][coff+23+22*(n-1)+ 2] = n-1;
  bl->cone[ct][coff+23+22*(n-1)+ 3] = DM_POLYTOPE_QUADRILATERAL;
  bl->cone[ct][coff+23+22*(n-1)+ 4] = 1;
  bl->cone[ct][coff+23+22*(n-1)+ 5] = 1;
  bl->cone[ct][coff+23+22*(n-1)+ 6] = 0;
  bl->cone[ct][coff+23+22*(n-1)+ 7] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+23+22*(n-1)+ 8] = 1;
  bl->cone[ct][coff+23+22*(n-1)+ 9] = 2;
  bl->cone[ct][coff+23+22*(n-1)+10] = n;
  bl->cone[ct][coff+23+22*(n-1)+11] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+23+22*(n-1)+12] = 1;
  bl->cone[ct][coff+23+22*(n-1)+13] = 3;
  bl->cone[ct][coff+23+22*(n-1)+14] = n;
  bl->cone[ct][coff+23+22*(n-1)+15] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+23+22*(n-1)+16] = 1;
  bl->cone[ct][coff+23+22*(n-1)+17] = 4;
  bl->cone[ct][coff+23+22*(n-1)+18] = n;
  bl->cone[ct][coff+23+22*(n-1)+19] = DM_POLYTOPE_SEG_PRISM_TENSOR;
  bl->cone[ct][coff+23+22*(n-1)+20] = 1;
  bl->cone[ct][coff+23+22*(n-1)+21] = 5;
  bl->cone[ct][coff+23+22*(n-1)+22] = n;
  for (i = 0; i < No; ++i) bl->ornt[ct][i] = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_BL(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  const PetscInt   n  = bl->n;
  PetscInt         tquad_tquad_o[] = { 0,  1, -2, -1,
                                       1,  0, -1, -2,
                                      -2, -1,  0,  1,
                                      -1, -2,  1,  0};

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = o;
  if (tr->trType) {
    PetscInt rt;

    PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
    if (rt < 0) {
      PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
      PetscFunctionReturn(0);
    }
  }
  switch (sct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      switch (tct) {
        case DM_POLYTOPE_POINT:
          *rnew = !so ? r : n-1 - r;
          break;
        case DM_POLYTOPE_POINT_PRISM_TENSOR:
          if (!so) {*rnew = r;     *onew = o;}
          else     {*rnew = n - r; *onew = -(o+1);}
        default: break;
      }
      break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:
          *onew = (so == 0) || (so ==  1) ? o : -(o+1);
          *rnew = (so == 0) || (so == -1) ? r : n-1 - r;
          break;
        case DM_POLYTOPE_SEG_PRISM_TENSOR:
          *onew = tquad_tquad_o[(so+2)*4+o+2];
          *rnew = (so == 0) || (so == -1) ? r : n - r;
          break;
        default: break;
      }
      break;
    default: PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_BL(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;

  PetscFunctionBeginHot;
  if (rt) *rt = -1;
  if (tr->trType && p >= 0) {
    PetscInt val;

    PetscCall(DMLabelGetValue(tr->trType, p, &val));
    if (val < 0) {
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      PetscFunctionReturn(0);
    }
    if (rt) *rt = val;
  }
  if (bl->Nt[source] < 0) {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
  } else {
    *Nt     = bl->Nt[source];
    *target = bl->target[source];
    *size   = bl->size[source];
    *cone   = bl->cone[source];
    *ornt   = bl->ornt[source];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformMapCoordinates_BL(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  PetscInt         d;

  PetscFunctionBeginHot;
  switch (pct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      PetscCheck(ct == DM_POLYTOPE_POINT,PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for target point type %s", DMPolytopeTypes[ct]);
      PetscCheck(Nv == 2,PETSC_COMM_SELF, PETSC_ERR_SUP, "Number of parent vertices %" PetscInt_FMT " != 2", Nv);
      PetscCheckFalse(r >= bl->n || r < 0,PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid replica %" PetscInt_FMT ", must be in [0, %" PetscInt_FMT ")", r, bl->n);
      for (d = 0; d < dE; ++d) out[d] = in[d] + bl->h[r] * (in[d + dE] - in[d]);
      break;
    default: PetscCall(DMPlexTransformMapCoordinatesBarycenter_Internal(tr, pct, ct, p, r, Nv, dE, in, out));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_BL(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  PetscInt         cells[256], n = 256, i;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  PetscOptionsHeadBegin(PetscOptionsObject,"DMPlexTransform Boundary Layer Options");
  PetscCall(PetscOptionsBoundedInt("-dm_plex_transform_bl_splits", "Number of divisions of a cell", "", bl->n, &bl->n, NULL, 1));
  PetscCall(PetscOptionsReal("-dm_plex_transform_bl_height_factor", "Factor increase for height at each division", "", bl->r, &bl->r, NULL));
  PetscCall(PetscOptionsIntArray("-dm_plex_transform_bl_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    PetscCall(DMPlexTransformSetActive(tr, active));
    PetscCall(DMLabelDestroy(&active));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_BL(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    PetscCall(PetscObjectGetName((PetscObject) tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Boundary Layer refinement %s\n", name ? name : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(DMLabelView(tr->trType, viewer));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_BL(DMPlexTransform tr)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  PetscInt         ict;

  PetscFunctionBegin;
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    PetscCall(PetscFree4(bl->target[ict], bl->size[ict], bl->cone[ict], bl->ornt[ict]));
  }
  PetscCall(PetscFree5(bl->Nt, bl->target, bl->size, bl->cone, bl->ornt));
  PetscCall(PetscFree(bl->h));
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_BL(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view           = DMPlexTransformView_BL;
  tr->ops->setfromoptions = DMPlexTransformSetFromOptions_BL;
  tr->ops->setup          = DMPlexTransformSetUp_BL;
  tr->ops->destroy        = DMPlexTransformDestroy_BL;
  tr->ops->celltransform  = DMPlexTransformCellTransform_BL;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_BL;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinates_BL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_BL(DMPlexTransform tr)
{
  DMPlexRefine_BL *bl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNewLog(tr, &bl));
  tr->data = bl;

  bl->n = 1; /* 1 split -> 2 new cells */
  bl->r = 1; /* linear coordinate progression */

  PetscCall(DMPlexTransformInitialize_BL(tr));
  PetscFunctionReturn(0);
}
