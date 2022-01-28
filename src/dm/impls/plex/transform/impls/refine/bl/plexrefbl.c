#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformSetUp_BL(DMPlexTransform tr)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  const PetscInt   n  = bl->n;
  DMPolytopeType   ct;
  DM               dm;
  DMLabel          active;
  PetscInt         Nc, No, coff, i, ict;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* If no label is given, split all tensor cells */
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMPlexTransformGetActive(tr, &active);CHKERRQ(ierr);
  if (active) {
    IS              refineIS;
    const PetscInt *refineCells;
    PetscInt        c;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc);CHKERRQ(ierr);
    if (refineIS) {ierr = ISGetIndices(refineIS, &refineCells);CHKERRQ(ierr);}
    for (c = 0; c < Nc; ++c) {
      const PetscInt cell    = refineCells[c];
      PetscInt      *closure = NULL;
      PetscInt       Ncl, cl;

      ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < Ncl; cl += 2) {
        const PetscInt point = closure[cl];

        ierr = DMPlexGetCellType(dm, point, &ct);CHKERRQ(ierr);
        switch (ct) {
          case DM_POLYTOPE_POINT_PRISM_TENSOR:
          case DM_POLYTOPE_SEG_PRISM_TENSOR:
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
          case DM_POLYTOPE_QUAD_PRISM_TENSOR:
            ierr = DMLabelSetValue(tr->trType, point, 1);CHKERRQ(ierr);break;
          default: break;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
    }
  }
  /* Cell heights */
  ierr = PetscMalloc1(n, &bl->h);CHKERRQ(ierr);
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

  ierr = PetscMalloc5(DM_NUM_POLYTOPES, &bl->Nt, DM_NUM_POLYTOPES, &bl->target, DM_NUM_POLYTOPES, &bl->size, DM_NUM_POLYTOPES, &bl->cone, DM_NUM_POLYTOPES, &bl->ornt);CHKERRQ(ierr);
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
  ierr = PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]);CHKERRQ(ierr);
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
  ierr = PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]);CHKERRQ(ierr);
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
  ierr = PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]);CHKERRQ(ierr);
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
  ierr = PetscMalloc4(bl->Nt[ct], &bl->target[ct], bl->Nt[ct], &bl->size[ct], Nc, &bl->cone[ct], No, &bl->ornt[ct]);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;
  PetscInt         tquad_tquad_o[] = { 0,  1, -2, -1,
                                       1,  0, -1, -2,
                                      -2, -1,  0,  1,
                                      -1, -2,  1,  0};

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = o;
  if (tr->trType) {
    PetscInt rt;

    ierr = DMLabelGetValue(tr->trType, sp, &rt);CHKERRQ(ierr);
    if (rt < 0) {
      ierr = DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
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
    default: ierr = DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_BL(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  PetscErrorCode   ierr;

  PetscFunctionBeginHot;
  if (rt) *rt = -1;
  if (tr->trType && p >= 0) {
    PetscInt val;

    ierr = DMLabelGetValue(tr->trType, p, &val);CHKERRQ(ierr);
    if (val < 0) {
      ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (rt) *rt = val;
  }
  if (bl->Nt[source] < 0) {
    ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBeginHot;
  switch (pct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      if (ct != DM_POLYTOPE_POINT) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for target point type %s", DMPolytopeTypes[ct]);
      if (Nv != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Number of parent vertices %D != 2", Nv);
      if (r >= bl->n || r < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid replica %D, must be in [0, %D)", r, bl->n);
      for (d = 0; d < dE; ++d) out[d] = in[d] + bl->h[r] * (in[d + dE] - in[d]);
      break;
    default: ierr = DMPlexTransformMapCoordinatesBarycenter_Internal(tr, pct, ct, p, r, Nv, dE, in, out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_BL(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  DMPlexRefine_BL *bl = (DMPlexRefine_BL *) tr->data;
  PetscInt         cells[256], n = 256, i;
  PetscBool        flg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlexTransform Boundary Layer Options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_transform_bl_splits", "Number of divisions of a cell", "", bl->n, &bl->n, NULL, 1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_transform_bl_height_factor", "Factor increase for height at each division", "", bl->r, &bl->r, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dm_plex_transform_bl_ref_cell", "Mark cells for refinement", "", cells, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    DMLabel active;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE);CHKERRQ(ierr);}
    ierr = DMPlexTransformSetActive(tr, active);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&active);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_BL(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    ierr = PetscObjectGetName((PetscObject) tr, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Boundary Layer refinement %s\n", name ? name : "");CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = DMLabelView(tr->trType, viewer);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    ierr = PetscFree4(bl->target[ict], bl->size[ict], bl->cone[ict], bl->ornt[ict]);CHKERRQ(ierr);
  }
  ierr = PetscFree5(bl->Nt, bl->target, bl->size, bl->cone, bl->ornt);CHKERRQ(ierr);
  ierr = PetscFree(bl->h);CHKERRQ(ierr);
  ierr = PetscFree(tr->data);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ierr = PetscNewLog(tr, &bl);CHKERRQ(ierr);
  tr->data = bl;

  bl->n = 1; /* 1 split -> 2 new cells */
  bl->r = 1; /* linear coordinate progression */

  ierr = DMPlexTransformInitialize_BL(tr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
