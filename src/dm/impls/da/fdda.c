
#include <petsc/private/dmdaimpl.h> /*I      "petscdmda.h"     I*/
#include <petscmat.h>
#include <petscbt.h>

extern PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM, ISColoringType, ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM, ISColoringType, ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM, ISColoringType, ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM, ISColoringType, ISColoring *);

/*
   For ghost i that may be negative or greater than the upper bound this
  maps it into the 0:m-1 range using periodicity
*/
#define SetInRange(i, m) ((i < 0) ? m + i : ((i >= m) ? i - m : i))

static PetscErrorCode DMDASetBlockFills_Private(const PetscInt *dfill, PetscInt w, PetscInt **rfill)
{
  PetscInt i, j, nz, *fill;

  PetscFunctionBegin;
  if (!dfill) PetscFunctionReturn(PETSC_SUCCESS);

  /* count number nonzeros */
  nz = 0;
  for (i = 0; i < w; i++) {
    for (j = 0; j < w; j++) {
      if (dfill[w * i + j]) nz++;
    }
  }
  PetscCall(PetscMalloc1(nz + w + 1, &fill));
  /* construct modified CSR storage of nonzero structure */
  /*  fill[0 -- w] marks starts of each row of column indices (and end of last row)
   so fill[1] - fill[0] gives number of nonzeros in first row etc */
  nz = w + 1;
  for (i = 0; i < w; i++) {
    fill[i] = nz;
    for (j = 0; j < w; j++) {
      if (dfill[w * i + j]) {
        fill[nz] = j;
        nz++;
      }
    }
  }
  fill[w] = nz;

  *rfill = fill;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDASetBlockFillsSparse_Private(const PetscInt *dfillsparse, PetscInt w, PetscInt **rfill)
{
  PetscInt nz;

  PetscFunctionBegin;
  if (!dfillsparse) PetscFunctionReturn(PETSC_SUCCESS);

  /* Determine number of non-zeros */
  nz = (dfillsparse[w] - w - 1);

  /* Allocate space for our copy of the given sparse matrix representation. */
  PetscCall(PetscMalloc1(nz + w + 1, rfill));
  PetscCall(PetscArraycpy(*rfill, dfillsparse, nz + w + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDASetBlockFills_Private2(DM_DA *dd)
{
  PetscInt i, k, cnt = 1;

  PetscFunctionBegin;

  /* ofillcount tracks the columns of ofill that have any nonzero in thems; the value in each location is the number of
   columns to the left with any nonzeros in them plus 1 */
  PetscCall(PetscCalloc1(dd->w, &dd->ofillcols));
  for (i = 0; i < dd->w; i++) {
    for (k = dd->ofill[i]; k < dd->ofill[i + 1]; k++) dd->ofillcols[dd->ofill[k]] = 1;
  }
  for (i = 0; i < dd->w; i++) {
    if (dd->ofillcols[i]) dd->ofillcols[i] = cnt++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMDASetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by `DMCreateMatrix()`.

    Logically Collective

    Input Parameters:
+   da - the distributed array
.   dfill - the fill pattern in the diagonal block (may be NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks

    Level: developer

    Notes:
    This only makes sense when you are doing multicomponent problems but using the
       `MATMPIAIJ` matrix format

           The format for dfill and ofill is a 2 dimensional dof by dof matrix with 1 entries
       representing coupling and 0 entries for missing coupling. For example
.vb
            dfill[9] = {1, 0, 0,
                        1, 1, 0,
                        0, 1, 1}
.ve
       means that row 0 is coupled with only itself in the diagonal block, row 1 is coupled with
       itself and row 0 (in the diagonal block) and row 2 is coupled with itself and row 1 (in the
       diagonal block).

     `DMDASetGetMatrix()` allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Glenn Hammond

.seealso: `DM`, `DMDA`, `DMCreateMatrix()`, `DMDASetGetMatrix()`, `DMSetMatrixPreallocateOnly()`
@*/
PetscErrorCode DMDASetBlockFills(DM da, const PetscInt *dfill, const PetscInt *ofill)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /* save the given dfill and ofill information */
  PetscCall(DMDASetBlockFills_Private(dfill, dd->w, &dd->dfill));
  PetscCall(DMDASetBlockFills_Private(ofill, dd->w, &dd->ofill));

  /* count nonzeros in ofill columns */
  PetscCall(DMDASetBlockFills_Private2(dd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMDASetBlockFillsSparse - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by `DMCreateMatrix()`, using sparse representations
    of fill patterns.

    Logically Collective

    Input Parameters:
+   da - the distributed array
.   dfill - the sparse fill pattern in the diagonal block (may be `NULL`, means use dense block)
-   ofill - the sparse fill pattern in the off-diagonal blocks

    Level: developer

    Notes:
    This only makes sense when you are doing multicomponent problems but using the
       `MATMPIAIJ` matrix format

           The format for `dfill` and `ofill` is a sparse representation of a
           dof-by-dof matrix with 1 entries representing coupling and 0 entries
           for missing coupling.  The sparse representation is a 1 dimensional
           array of length nz + dof + 1, where nz is the number of non-zeros in
           the matrix.  The first dof entries in the array give the
           starting array indices of each row's items in the rest of the array,
           the dof+1st item contains the value nz + dof + 1 (i.e. the entire length of the array)
           and the remaining nz items give the column indices of each of
           the 1s within the logical 2D matrix.  Each row's items within
           the array are the column indices of the 1s within that row
           of the 2D matrix.  PETSc developers may recognize that this is the
           same format as that computed by the `DMDASetBlockFills_Private()`
           function from a dense 2D matrix representation.

     `DMDASetGetMatrix()` allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the `dfill`, `ofill` format

   Contributed by Philip C. Roth

.seealso: `DM`, `DMDA`, `DMDASetBlockFills()`, `DMCreateMatrix()`, `DMDASetGetMatrix()`, `DMSetMatrixPreallocateOnly()`
@*/
PetscErrorCode DMDASetBlockFillsSparse(DM da, const PetscInt *dfillsparse, const PetscInt *ofillsparse)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /* save the given dfill and ofill information */
  PetscCall(DMDASetBlockFillsSparse_Private(dfillsparse, dd->w, &dd->dfill));
  PetscCall(DMDASetBlockFillsSparse_Private(ofillsparse, dd->w, &dd->ofill));

  /* count nonzeros in ofill columns */
  PetscCall(DMDASetBlockFills_Private2(dd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateColoring_DA(DM da, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt       dim, m, n, p, nc;
  DMBoundaryType bx, by, bz;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      isBAIJ;
  DM_DA         *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           yn  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     xn                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  PetscCall(DMDAGetInfo(da, &dim, NULL, NULL, NULL, &m, &n, &p, &nc, NULL, &bx, &by, &bz, NULL));

  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (ctype == IS_COLORING_LOCAL) {
    if (size == 1) {
      ctype = IS_COLORING_GLOBAL;
    } else {
      PetscCheck((dim == 1) || !((m == 1 && bx == DM_BOUNDARY_PERIODIC) || (n == 1 && by == DM_BOUNDARY_PERIODIC) || (p == 1 && bz == DM_BOUNDARY_PERIODIC)), PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "IS_COLORING_LOCAL cannot be used for periodic boundary condition having both ends of the domain on the same process");
    }
  }

  /* Tell the DMDA it has 1 degree of freedom per grid point so that the coloring for BAIJ
     matrices is for the blocks, not the individual matrix elements  */
  PetscCall(PetscStrbeginswith(da->mattype, MATBAIJ, &isBAIJ));
  if (!isBAIJ) PetscCall(PetscStrbeginswith(da->mattype, MATMPIBAIJ, &isBAIJ));
  if (!isBAIJ) PetscCall(PetscStrbeginswith(da->mattype, MATSEQBAIJ, &isBAIJ));
  if (isBAIJ) {
    dd->w  = 1;
    dd->xs = dd->xs / nc;
    dd->xe = dd->xe / nc;
    dd->Xs = dd->Xs / nc;
    dd->Xe = dd->Xe / nc;
  }

  /*
     We do not provide a getcoloring function in the DMDA operations because
   the basic DMDA does not know about matrices. We think of DMDA as being
   more low-level then matrices.
  */
  if (dim == 1) PetscCall(DMCreateColoring_DA_1d_MPIAIJ(da, ctype, coloring));
  else if (dim == 2) PetscCall(DMCreateColoring_DA_2d_MPIAIJ(da, ctype, coloring));
  else if (dim == 3) PetscCall(DMCreateColoring_DA_3d_MPIAIJ(da, ctype, coloring));
  else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Not done for %" PetscInt_FMT " dimension, send us mail petsc-maint@mcs.anl.gov for code", dim);
  if (isBAIJ) {
    dd->w  = nc;
    dd->xs = dd->xs * nc;
    dd->xe = dd->xe * nc;
    dd->Xs = dd->Xs * nc;
    dd->Xe = dd->Xe * nc;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM da, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt         xs, ys, nx, ny, i, j, ii, gxs, gys, gnx, gny, m, n, M, N, dim, s, k, nc, col;
  PetscInt         ncolors = 0;
  MPI_Comm         comm;
  DMBoundaryType   bx, by;
  DMDAStencilType  st;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, &M, &N, NULL, &nc, &s, &bx, &by, NULL, &st));
  col = 2 * s + 1;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  /* special case as taught to us by Paul Hovland */
  if (st == DMDA_STENCIL_STAR && s == 1) {
    PetscCall(DMCreateColoring_DA_2d_5pt_MPIAIJ(da, ctype, coloring));
  } else {
    if (ctype == IS_COLORING_GLOBAL) {
      if (!dd->localcoloring) {
        PetscCall(PetscMalloc1(nc * nx * ny, &colors));
        ii = 0;
        for (j = ys; j < ys + ny; j++) {
          for (i = xs; i < xs + nx; i++) {
            for (k = 0; k < nc; k++) colors[ii++] = k + nc * ((i % col) + col * (j % col));
          }
        }
        ncolors = nc + nc * (col - 1 + col * (col - 1));
        PetscCall(ISColoringCreate(comm, ncolors, nc * nx * ny, colors, PETSC_OWN_POINTER, &dd->localcoloring));
      }
      *coloring = dd->localcoloring;
    } else if (ctype == IS_COLORING_LOCAL) {
      if (!dd->ghostedcoloring) {
        PetscCall(PetscMalloc1(nc * gnx * gny, &colors));
        ii = 0;
        for (j = gys; j < gys + gny; j++) {
          for (i = gxs; i < gxs + gnx; i++) {
            for (k = 0; k < nc; k++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = k + nc * ((SetInRange(i, m) % col) + col * (SetInRange(j, n) % col));
            }
          }
        }
        ncolors = nc + nc * (col - 1 + col * (col - 1));
        PetscCall(ISColoringCreate(comm, ncolors, nc * gnx * gny, colors, PETSC_OWN_POINTER, &dd->ghostedcoloring));
        /* PetscIntView(ncolors,(PetscInt*)colors,0); */

        PetscCall(ISColoringSetType(dd->ghostedcoloring, IS_COLORING_LOCAL));
      }
      *coloring = dd->ghostedcoloring;
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Unknown ISColoringType %d", (int)ctype);
  }
  PetscCall(ISColoringReference(*coloring));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM da, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt         xs, ys, nx, ny, i, j, gxs, gys, gnx, gny, m, n, p, dim, s, k, nc, col, zs, gzs, ii, l, nz, gnz, M, N, P;
  PetscInt         ncolors;
  MPI_Comm         comm;
  DMBoundaryType   bx, by, bz;
  DMDAStencilType  st;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, &M, &N, &P, &nc, &s, &bx, &by, &bz, &st));
  col = 2 * s + 1;
  PetscCheck(bx != DM_BOUNDARY_PERIODIC || (m % col) == 0, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  PetscCheck(by != DM_BOUNDARY_PERIODIC || (n % col) == 0, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  PetscCheck(bz != DM_BOUNDARY_PERIODIC || (p % col) == 0, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      PetscCall(PetscMalloc1(nc * nx * ny * nz, &colors));
      ii = 0;
      for (k = zs; k < zs + nz; k++) {
        for (j = ys; j < ys + ny; j++) {
          for (i = xs; i < xs + nx; i++) {
            for (l = 0; l < nc; l++) colors[ii++] = l + nc * ((i % col) + col * (j % col) + col * col * (k % col));
          }
        }
      }
      ncolors = nc + nc * (col - 1 + col * (col - 1) + col * col * (col - 1));
      PetscCall(ISColoringCreate(comm, ncolors, nc * nx * ny * nz, colors, PETSC_OWN_POINTER, &dd->localcoloring));
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      PetscCall(PetscMalloc1(nc * gnx * gny * gnz, &colors));
      ii = 0;
      for (k = gzs; k < gzs + gnz; k++) {
        for (j = gys; j < gys + gny; j++) {
          for (i = gxs; i < gxs + gnx; i++) {
            for (l = 0; l < nc; l++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = l + nc * ((SetInRange(i, m) % col) + col * (SetInRange(j, n) % col) + col * col * (SetInRange(k, p) % col));
            }
          }
        }
      }
      ncolors = nc + nc * (col - 1 + col * (col - 1) + col * col * (col - 1));
      PetscCall(ISColoringCreate(comm, ncolors, nc * gnx * gny * gnz, colors, PETSC_OWN_POINTER, &dd->ghostedcoloring));
      PetscCall(ISColoringSetType(dd->ghostedcoloring, IS_COLORING_LOCAL));
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Unknown ISColoringType %d", (int)ctype);
  PetscCall(ISColoringReference(*coloring));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM da, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt         xs, nx, i, i1, gxs, gnx, l, m, M, dim, s, nc, col;
  PetscInt         ncolors;
  MPI_Comm         comm;
  DMBoundaryType   bx;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, NULL, NULL, &M, NULL, NULL, &nc, &s, &bx, NULL, NULL, NULL));
  col = 2 * s + 1;
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &nx, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, NULL, NULL, &gnx, NULL, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      PetscCall(PetscMalloc1(nc * nx, &colors));
      if (dd->ofillcols) {
        PetscInt tc = 0;
        for (i = 0; i < nc; i++) tc += (PetscInt)(dd->ofillcols[i] > 0);
        i1 = 0;
        for (i = xs; i < xs + nx; i++) {
          for (l = 0; l < nc; l++) {
            if (dd->ofillcols[l] && (i % col)) {
              colors[i1++] = nc - 1 + tc * ((i % col) - 1) + dd->ofillcols[l];
            } else {
              colors[i1++] = l;
            }
          }
        }
        ncolors = nc + 2 * s * tc;
      } else {
        i1 = 0;
        for (i = xs; i < xs + nx; i++) {
          for (l = 0; l < nc; l++) colors[i1++] = l + nc * (i % col);
        }
        ncolors = nc + nc * (col - 1);
      }
      PetscCall(ISColoringCreate(comm, ncolors, nc * nx, colors, PETSC_OWN_POINTER, &dd->localcoloring));
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      PetscCall(PetscMalloc1(nc * gnx, &colors));
      i1 = 0;
      for (i = gxs; i < gxs + gnx; i++) {
        for (l = 0; l < nc; l++) {
          /* the complicated stuff is to handle periodic boundaries */
          colors[i1++] = l + nc * (SetInRange(i, m) % col);
        }
      }
      ncolors = nc + nc * (col - 1);
      PetscCall(ISColoringCreate(comm, ncolors, nc * gnx, colors, PETSC_OWN_POINTER, &dd->ghostedcoloring));
      PetscCall(ISColoringSetType(dd->ghostedcoloring, IS_COLORING_LOCAL));
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Unknown ISColoringType %d", (int)ctype);
  PetscCall(ISColoringReference(*coloring));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM da, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt         xs, ys, nx, ny, i, j, ii, gxs, gys, gnx, gny, m, n, dim, s, k, nc;
  PetscInt         ncolors;
  MPI_Comm         comm;
  DMBoundaryType   bx, by;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, NULL, NULL, NULL, &nc, &s, &bx, &by, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      PetscCall(PetscMalloc1(nc * nx * ny, &colors));
      ii = 0;
      for (j = ys; j < ys + ny; j++) {
        for (i = xs; i < xs + nx; i++) {
          for (k = 0; k < nc; k++) colors[ii++] = k + nc * ((3 * j + i) % 5);
        }
      }
      ncolors = 5 * nc;
      PetscCall(ISColoringCreate(comm, ncolors, nc * nx * ny, colors, PETSC_OWN_POINTER, &dd->localcoloring));
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      PetscCall(PetscMalloc1(nc * gnx * gny, &colors));
      ii = 0;
      for (j = gys; j < gys + gny; j++) {
        for (i = gxs; i < gxs + gnx; i++) {
          for (k = 0; k < nc; k++) colors[ii++] = k + nc * ((3 * SetInRange(j, n) + SetInRange(i, m)) % 5);
        }
      }
      ncolors = 5 * nc;
      PetscCall(ISColoringCreate(comm, ncolors, nc * gnx * gny, colors, PETSC_OWN_POINTER, &dd->ghostedcoloring));
      PetscCall(ISColoringSetType(dd->ghostedcoloring, IS_COLORING_LOCAL));
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Unknown ISColoringType %d", (int)ctype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================== */
extern PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ_Fill(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_1d_SeqAIJ_NoPreallocation(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPISELL(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPISELL(DM, Mat);
extern PetscErrorCode DMCreateMatrix_DA_IS(DM, Mat);

/*@C
   MatSetupDM - Sets the `DMDA` that is to be used by the HYPRE_StructMatrix PETSc matrix

   Logically Collective

   Input Parameters:
+  mat - the matrix
-  da - the da

   Level: intermediate

.seealso: `DMDA`, `Mat`, `MatSetUp()`
@*/
PetscErrorCode MatSetupDM(Mat mat, DM da)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidHeaderSpecificType(da, DM_CLASSID, 2, DMDA);
  PetscTryMethod(mat, "MatSetupDM_C", (Mat, DM), (mat, da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_MPI_DA(Mat A, PetscViewer viewer)
{
  DM                da;
  const char       *prefix;
  Mat               Anatural;
  AO                ao;
  PetscInt          rstart, rend, *petsc, i;
  IS                is;
  MPI_Comm          comm;
  PetscViewerFormat format;

  PetscFunctionBegin;
  /* Check whether we are just printing info, in which case MatView() already viewed everything we wanted to view */
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetDM(A, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix not generated from a DMDA");

  PetscCall(DMDAGetAO(da, &ao));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(PetscMalloc1(rend - rstart, &petsc));
  for (i = rstart; i < rend; i++) petsc[i - rstart] = i;
  PetscCall(AOApplicationToPetsc(ao, rend - rstart, petsc));
  PetscCall(ISCreateGeneral(comm, rend - rstart, petsc, PETSC_OWN_POINTER, &is));

  /* call viewer on natural ordering */
  PetscCall(MatCreateSubMatrix(A, is, is, MAT_INITIAL_MATRIX, &Anatural));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)A, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Anatural, prefix));
  PetscCall(PetscObjectSetName((PetscObject)Anatural, ((PetscObject)A)->name));
  ((PetscObject)Anatural)->donotPetscObjectPrintClassNamePrefixType = PETSC_TRUE;
  PetscCall(MatView(Anatural, viewer));
  ((PetscObject)Anatural)->donotPetscObjectPrintClassNamePrefixType = PETSC_FALSE;
  PetscCall(MatDestroy(&Anatural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_MPI_DA(Mat A, PetscViewer viewer)
{
  DM       da;
  Mat      Anatural, Aapp;
  AO       ao;
  PetscInt rstart, rend, *app, i, m, n, M, N;
  IS       is;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetDM(A, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix not generated from a DMDA");

  /* Load the matrix in natural ordering */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &Anatural));
  PetscCall(MatSetType(Anatural, ((PetscObject)A)->type_name));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatSetSizes(Anatural, m, n, M, N));
  PetscCall(MatLoad(Anatural, viewer));

  /* Map natural ordering to application ordering and create IS */
  PetscCall(DMDAGetAO(da, &ao));
  PetscCall(MatGetOwnershipRange(Anatural, &rstart, &rend));
  PetscCall(PetscMalloc1(rend - rstart, &app));
  for (i = rstart; i < rend; i++) app[i - rstart] = i;
  PetscCall(AOPetscToApplication(ao, rend - rstart, app));
  PetscCall(ISCreateGeneral(comm, rend - rstart, app, PETSC_OWN_POINTER, &is));

  /* Do permutation and replace header */
  PetscCall(MatCreateSubMatrix(Anatural, is, is, MAT_INITIAL_MATRIX, &Aapp));
  PetscCall(MatHeaderReplace(A, &Aapp));
  PetscCall(ISDestroy(&is));
  PetscCall(MatDestroy(&Anatural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA(DM da, Mat *J)
{
  PetscInt    dim, dof, nx, ny, nz, dims[3], starts[3], M, N, P;
  Mat         A;
  MPI_Comm    comm;
  MatType     Atype;
  MatType     mtype;
  PetscMPIInt size;
  DM_DA      *dd    = (DM_DA *)da->data;
  void (*aij)(void) = NULL, (*baij)(void) = NULL, (*sbaij)(void) = NULL, (*sell)(void) = NULL, (*is)(void) = NULL;

  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  mtype = da->mattype;

  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           ny  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     nx                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  M   = dd->M;
  N   = dd->N;
  P   = dd->P;
  dim = da->dim;
  dof = dd->w;
  /* PetscCall(DMDAGetInfo(da,&dim,&M,&N,&P,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL)); */
  PetscCall(DMDAGetCorners(da, NULL, NULL, NULL, &nx, &ny, &nz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, dof * nx * ny * nz, dof * nx * ny * nz, dof * M * N * P, dof * M * N * P));
  PetscCall(MatSetType(A, mtype));
  PetscCall(MatSetFromOptions(A));
  if (dof * nx * ny * nz < da->bind_below) {
    PetscCall(MatSetBindingPropagates(A, PETSC_TRUE));
    PetscCall(MatBindToCPU(A, PETSC_TRUE));
  }
  PetscCall(MatSetDM(A, da));
  if (da->structure_only) PetscCall(MatSetOption(A, MAT_STRUCTURE_ONLY, PETSC_TRUE));
  PetscCall(MatGetType(A, &Atype));
  /*
     We do not provide a getmatrix function in the DMDA operations because
   the basic DMDA does not know about matrices. We think of DMDA as being more
   more low-level than matrices. This is kind of cheating but, cause sometimes
   we think of DMDA has higher level than matrices.

     We could switch based on Atype (or mtype), but we do not since the
   specialized setting routines depend only on the particular preallocation
   details of the matrix, not the type itself.
  */
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", &aij));
  if (!aij) PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqAIJSetPreallocation_C", &aij));
  if (!aij) {
    PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMPIBAIJSetPreallocation_C", &baij));
    if (!baij) PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqBAIJSetPreallocation_C", &baij));
    if (!baij) {
      PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMPISBAIJSetPreallocation_C", &sbaij));
      if (!sbaij) PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqSBAIJSetPreallocation_C", &sbaij));
      if (!sbaij) {
        PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMPISELLSetPreallocation_C", &sell));
        if (!sell) PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqSELLSetPreallocation_C", &sell));
      }
      if (!sell) PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatISSetPreallocation_C", &is));
    }
  }
  if (aij) {
    if (dim == 1) {
      if (dd->ofill) {
        PetscCall(DMCreateMatrix_DA_1d_MPIAIJ_Fill(da, A));
      } else {
        DMBoundaryType bx;
        PetscMPIInt    size;
        PetscCall(DMDAGetInfo(da, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &bx, NULL, NULL, NULL));
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
        if (size == 1 && bx == DM_BOUNDARY_NONE) {
          PetscCall(DMCreateMatrix_DA_1d_SeqAIJ_NoPreallocation(da, A));
        } else {
          PetscCall(DMCreateMatrix_DA_1d_MPIAIJ(da, A));
        }
      }
    } else if (dim == 2) {
      if (dd->ofill) {
        PetscCall(DMCreateMatrix_DA_2d_MPIAIJ_Fill(da, A));
      } else {
        PetscCall(DMCreateMatrix_DA_2d_MPIAIJ(da, A));
      }
    } else if (dim == 3) {
      if (dd->ofill) {
        PetscCall(DMCreateMatrix_DA_3d_MPIAIJ_Fill(da, A));
      } else {
        PetscCall(DMCreateMatrix_DA_3d_MPIAIJ(da, A));
      }
    }
  } else if (baij) {
    if (dim == 2) {
      PetscCall(DMCreateMatrix_DA_2d_MPIBAIJ(da, A));
    } else if (dim == 3) {
      PetscCall(DMCreateMatrix_DA_3d_MPIBAIJ(da, A));
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Not implemented for %" PetscInt_FMT " dimension and Matrix Type: %s in %" PetscInt_FMT " dimension! Send mail to petsc-maint@mcs.anl.gov for code", dim, Atype, dim);
  } else if (sbaij) {
    if (dim == 2) {
      PetscCall(DMCreateMatrix_DA_2d_MPISBAIJ(da, A));
    } else if (dim == 3) {
      PetscCall(DMCreateMatrix_DA_3d_MPISBAIJ(da, A));
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Not implemented for %" PetscInt_FMT " dimension and Matrix Type: %s in %" PetscInt_FMT " dimension! Send mail to petsc-maint@mcs.anl.gov for code", dim, Atype, dim);
  } else if (sell) {
    if (dim == 2) {
      PetscCall(DMCreateMatrix_DA_2d_MPISELL(da, A));
    } else if (dim == 3) {
      PetscCall(DMCreateMatrix_DA_3d_MPISELL(da, A));
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Not implemented for %" PetscInt_FMT " dimension and Matrix Type: %s in %" PetscInt_FMT " dimension! Send mail to petsc-maint@mcs.anl.gov for code", dim, Atype, dim);
  } else if (is) {
    PetscCall(DMCreateMatrix_DA_IS(da, A));
  } else {
    ISLocalToGlobalMapping ltog;

    PetscCall(MatSetBlockSize(A, dof));
    PetscCall(MatSetUp(A));
    PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
    PetscCall(MatSetLocalToGlobalMapping(A, ltog, ltog));
  }
  PetscCall(DMDAGetGhostCorners(da, &starts[0], &starts[1], &starts[2], &dims[0], &dims[1], &dims[2]));
  PetscCall(MatSetStencil(A, dim, dims, starts, dof));
  PetscCall(MatSetDM(A, da));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    /* change viewer to display matrix in natural ordering */
    PetscCall(MatSetOperation(A, MATOP_VIEW, (void (*)(void))MatView_MPI_DA));
    PetscCall(MatSetOperation(A, MATOP_LOAD, (void (*)(void))MatLoad_MPI_DA));
  }
  *J = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatISSetPreallocation_IS(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);

PetscErrorCode DMCreateMatrix_DA_IS(DM dm, Mat J)
{
  DM_DA                 *da = (DM_DA *)dm->data;
  Mat                    lJ, P;
  ISLocalToGlobalMapping ltog;
  IS                     is;
  PetscBT                bt;
  const PetscInt        *e_loc, *idx;
  PetscInt               i, nel, nen, nv, dof, *gidx, n, N;

  /* The l2g map of DMDA has all ghosted nodes, and e_loc is a subset of all the local nodes (including the ghosted)
     We need to filter out the local indices that are not represented through the DMDAGetElements decomposition */
  PetscFunctionBegin;
  dof = da->w;
  PetscCall(MatSetBlockSize(J, dof));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));

  /* flag local elements indices in local DMDA numbering */
  PetscCall(ISLocalToGlobalMappingGetSize(ltog, &nv));
  PetscCall(PetscBTCreate(nv / dof, &bt));
  PetscCall(DMDAGetElements(dm, &nel, &nen, &e_loc)); /* this will throw an error if the stencil type is not DMDA_STENCIL_BOX */
  for (i = 0; i < nel * nen; i++) PetscCall(PetscBTSet(bt, e_loc[i]));

  /* filter out (set to -1) the global indices not used by the local elements */
  PetscCall(PetscMalloc1(nv / dof, &gidx));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(ltog, &idx));
  PetscCall(PetscArraycpy(gidx, idx, nv / dof));
  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(ltog, &idx));
  for (i = 0; i < nv / dof; i++)
    if (!PetscBTLookup(bt, i)) gidx[i] = -1;
  PetscCall(PetscBTDestroy(&bt));
  PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)dm), dof, nv / dof, gidx, PETSC_OWN_POINTER, &is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is, &ltog));
  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscCall(ISDestroy(&is));

  /* Preallocation */
  if (dm->prealloc_skip) {
    PetscCall(MatSetUp(J));
  } else {
    PetscCall(MatISGetLocalMat(J, &lJ));
    PetscCall(MatGetLocalToGlobalMapping(lJ, &ltog, NULL));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)lJ), &P));
    PetscCall(MatSetType(P, MATPREALLOCATOR));
    PetscCall(MatSetLocalToGlobalMapping(P, ltog, ltog));
    PetscCall(MatGetSize(lJ, &N, NULL));
    PetscCall(MatGetLocalSize(lJ, &n, NULL));
    PetscCall(MatSetSizes(P, n, n, N, N));
    PetscCall(MatSetBlockSize(P, dof));
    PetscCall(MatSetUp(P));
    for (i = 0; i < nel; i++) PetscCall(MatSetValuesBlockedLocal(P, nen, e_loc + i * nen, nen, e_loc + i * nen, NULL, INSERT_VALUES));
    PetscCall(MatPreallocatorPreallocate(P, (PetscBool)!da->prealloc_only, lJ));
    PetscCall(MatISRestoreLocalMat(J, &lJ));
    PetscCall(DMDARestoreElements(dm, &nel, &nen, &e_loc));
    PetscCall(MatDestroy(&P));

    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPISELL(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny, m, n, dim, s, *cols = NULL, k, nc, *rows = NULL, col, cnt, l, p;
  PetscInt               lstart, lend, pstart, pend, *dnz, *onz;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, NULL, NULL, NULL, &nc, &s, &bx, &by, NULL, &st));
  col = 2 * s + 1;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc2(nc, &rows, col * col * nc * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  PetscCall(MatSetBlockSize(J, nc));
  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny, nc * nx * ny, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

    for (j = ys; j < ys + ny; j++) {
      slot = i - gxs + gnx * (j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

      cnt = 0;
      for (k = 0; k < nc; k++) {
        for (l = lstart; l < lend + 1; l++) {
          for (p = pstart; p < pend + 1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = k + nc * (slot + gnx * l + p);
            }
          }
        }
        rows[k] = k + nc * (slot);
      }
      PetscCall(MatPreallocateSetLocal(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
    }
  }
  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqSELLSetPreallocation(J, 0, dnz));
  PetscCall(MatMPISELLSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

      for (j = ys; j < ys + ny; j++) {
        slot = i - gxs + gnx * (j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

        cnt = 0;
        for (k = 0; k < nc; k++) {
          for (l = lstart; l < lend + 1; l++) {
            for (p = pstart; p < pend + 1; p++) {
              if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star have either l = 0 or p = 0 */
                cols[cnt++] = k + nc * (slot + gnx * l + p);
              }
            }
          }
          rows[k] = k + nc * (slot);
        }
        PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, values, INSERT_VALUES));
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree2(rows, cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPISELL(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols = NULL, k, nc, *rows = NULL, col, cnt, l, p, *dnz = NULL, *onz = NULL;
  PetscInt               istart, iend, jstart, jend, kstart, kend, zs, nz, gzs, gnz, ii, jj, kk, M, N, P;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by, bz;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, &M, &N, &P, &nc, &s, &bx, &by, &bz, &st));
  col = 2 * s + 1;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc2(nc, &rows, col * col * col * nc * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  PetscCall(MatSetBlockSize(J, nc));
  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny * nz, nc * nx * ny * nz, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      for (k = zs; k < zs + nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

        slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

        cnt = 0;
        for (l = 0; l < nc; l++) {
          for (ii = istart; ii < iend + 1; ii++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (kk = kstart; kk < kend + 1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                  cols[cnt++] = l + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                }
              }
            }
          }
          rows[l] = l + nc * (slot);
        }
        PetscCall(MatPreallocateSetLocal(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
      }
    }
  }
  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqSELLSetPreallocation(J, 0, dnz));
  PetscCall(MatMPISELLSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * col * nc * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        for (k = zs; k < zs + nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

          slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

          cnt = 0;
          for (l = 0; l < nc; l++) {
            for (ii = istart; ii < iend + 1; ii++) {
              for (jj = jstart; jj < jend + 1; jj++) {
                for (kk = kstart; kk < kend + 1; kk++) {
                  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                    cols[cnt++] = l + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                  }
                }
              }
            }
            rows[l] = l + nc * (slot);
          }
          PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, values, INSERT_VALUES));
        }
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree2(rows, cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny, m, n, dim, s, *cols = NULL, k, nc, *rows = NULL, col, cnt, l, p, M, N;
  PetscInt               lstart, lend, pstart, pend, *dnz, *onz;
  MPI_Comm               comm;
  DMBoundaryType         bx, by;
  ISLocalToGlobalMapping ltog, mltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE, alreadyboundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, &M, &N, NULL, &nc, &s, &bx, &by, NULL, &st));
  if (bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_TRUE));
  col = 2 * s + 1;
  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2 * s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2 * s >= n) removedups = PETSC_TRUE;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc2(nc, &rows, col * col * nc * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  PetscCall(MatSetBlockSize(J, nc));
  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny, nc * nx * ny, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

    for (j = ys; j < ys + ny; j++) {
      slot = i - gxs + gnx * (j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

      cnt = 0;
      for (k = 0; k < nc; k++) {
        for (l = lstart; l < lend + 1; l++) {
          for (p = pstart; p < pend + 1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = k + nc * (slot + gnx * l + p);
            }
          }
        }
        rows[k] = k + nc * (slot);
      }
      if (removedups) PetscCall(MatPreallocateSetLocalRemoveDups(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
      else PetscCall(MatPreallocateSetLocal(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
    }
  }
  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqAIJSetPreallocation(J, 0, dnz));
  PetscCall(MatMPIAIJSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatGetLocalToGlobalMapping(J, &mltog, NULL));
  if (!mltog) PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i = xs; i < xs + nx; i++) {
      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

      for (j = ys; j < ys + ny; j++) {
        slot = i - gxs + gnx * (j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

        cnt = 0;
        for (l = lstart; l < lend + 1; l++) {
          for (p = pstart; p < pend + 1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = nc * (slot + gnx * l + p);
              for (k = 1; k < nc; k++) {
                cols[cnt] = 1 + cols[cnt - 1];
                cnt++;
              }
            }
          }
        }
        for (k = 0; k < nc; k++) rows[k] = k + nc * (slot);
        PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, NULL, INSERT_VALUES));
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBoundToCPU(J, &alreadyboundtocpu));
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (!alreadyboundtocpu) PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
    if (bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_FALSE));
  }
  PetscCall(PetscFree2(rows, cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, k, nc, row, col, cnt, maxcnt = 0, l, p, M, N;
  PetscInt               lstart, lend, pstart, pend, *dnz, *onz;
  DM_DA                 *dd = (DM_DA *)da->data;
  PetscInt               ifill_col, *ofill = dd->ofill, *dfill = dd->dfill;
  MPI_Comm               comm;
  DMBoundaryType         bx, by;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, &M, &N, NULL, &nc, &s, &bx, &by, NULL, &st));
  col = 2 * s + 1;
  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2 * s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2 * s >= n) removedups = PETSC_TRUE;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc1(col * col * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  PetscCall(MatSetBlockSize(J, nc));
  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny, nc * nx * ny, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

    for (j = ys; j < ys + ny; j++) {
      slot = i - gxs + gnx * (j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

      for (k = 0; k < nc; k++) {
        cnt = 0;
        for (l = lstart; l < lend + 1; l++) {
          for (p = pstart; p < pend + 1; p++) {
            if (l || p) {
              if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star */
                for (ifill_col = ofill[k]; ifill_col < ofill[k + 1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc * (slot + gnx * l + p);
              }
            } else {
              if (dfill) {
                for (ifill_col = dfill[k]; ifill_col < dfill[k + 1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc * (slot + gnx * l + p);
              } else {
                for (ifill_col = 0; ifill_col < nc; ifill_col++) cols[cnt++] = ifill_col + nc * (slot + gnx * l + p);
              }
            }
          }
        }
        row    = k + nc * (slot);
        maxcnt = PetscMax(maxcnt, cnt);
        if (removedups) PetscCall(MatPreallocateSetLocalRemoveDups(ltog, 1, &row, ltog, cnt, cols, dnz, onz));
        else PetscCall(MatPreallocateSetLocal(ltog, 1, &row, ltog, cnt, cols, dnz, onz));
      }
    }
  }
  PetscCall(MatSeqAIJSetPreallocation(J, 0, dnz));
  PetscCall(MatMPIAIJSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i = xs; i < xs + nx; i++) {
      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));

      for (j = ys; j < ys + ny; j++) {
        slot = i - gxs + gnx * (j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));

        for (k = 0; k < nc; k++) {
          cnt = 0;
          for (l = lstart; l < lend + 1; l++) {
            for (p = pstart; p < pend + 1; p++) {
              if (l || p) {
                if ((st == DMDA_STENCIL_BOX) || (!l || !p)) { /* entries on star */
                  for (ifill_col = ofill[k]; ifill_col < ofill[k + 1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc * (slot + gnx * l + p);
                }
              } else {
                if (dfill) {
                  for (ifill_col = dfill[k]; ifill_col < dfill[k + 1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc * (slot + gnx * l + p);
                } else {
                  for (ifill_col = 0; ifill_col < nc; ifill_col++) cols[cnt++] = ifill_col + nc * (slot + gnx * l + p);
                }
              }
            }
          }
          row = k + nc * (slot);
          PetscCall(MatSetValuesLocal(J, 1, &row, cnt, cols, NULL, INSERT_VALUES));
        }
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols = NULL, k, nc, *rows = NULL, col, cnt, l, p, *dnz = NULL, *onz = NULL;
  PetscInt               istart, iend, jstart, jend, kstart, kend, zs, nz, gzs, gnz, ii, jj, kk, M, N, P;
  MPI_Comm               comm;
  DMBoundaryType         bx, by, bz;
  ISLocalToGlobalMapping ltog, mltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, &M, &N, &P, &nc, &s, &bx, &by, &bz, &st));
  if (bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE && bz == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_TRUE));
  col = 2 * s + 1;

  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2 * s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2 * s >= n) removedups = PETSC_TRUE;
  if (P == 1 && 2 * s >= p) removedups = PETSC_TRUE;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc2(nc, &rows, col * col * col * nc * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  PetscCall(MatSetBlockSize(J, nc));
  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny * nz, nc * nx * ny * nz, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      for (k = zs; k < zs + nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

        slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

        cnt = 0;
        for (l = 0; l < nc; l++) {
          for (ii = istart; ii < iend + 1; ii++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (kk = kstart; kk < kend + 1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                  cols[cnt++] = l + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                }
              }
            }
          }
          rows[l] = l + nc * (slot);
        }
        if (removedups) PetscCall(MatPreallocateSetLocalRemoveDups(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
        else PetscCall(MatPreallocateSetLocal(ltog, nc, rows, ltog, cnt, cols, dnz, onz));
      }
    }
  }
  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqAIJSetPreallocation(J, 0, dnz));
  PetscCall(MatMPIAIJSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatGetLocalToGlobalMapping(J, &mltog, NULL));
  if (!mltog) PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        for (k = zs; k < zs + nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

          slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

          cnt = 0;
          for (kk = kstart; kk < kend + 1; kk++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (ii = istart; ii < iend + 1; ii++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                  cols[cnt++] = nc * (slot + ii + gnx * jj + gnx * gny * kk);
                  for (l = 1; l < nc; l++) {
                    cols[cnt] = 1 + cols[cnt - 1];
                    cnt++;
                  }
                }
              }
            }
          }
          rows[0] = nc * (slot);
          for (l = 1; l < nc; l++) rows[l] = 1 + rows[l - 1];
          PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, NULL, INSERT_VALUES));
        }
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE && bz == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_FALSE));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree2(rows, cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ_Fill(DM da, Mat J)
{
  DM_DA                 *dd = (DM_DA *)da->data;
  PetscInt               xs, nx, i, j, gxs, gnx, row, k, l;
  PetscInt               m, dim, s, *cols = NULL, nc, cnt, maxcnt = 0, *ocols;
  PetscInt              *ofill = dd->ofill, *dfill = dd->dfill;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog;
  PetscMPIInt            rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));

  /*
         nc - number of components per grid point
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, NULL, NULL, NULL, NULL, NULL, &nc, &s, &bx, NULL, NULL, NULL));
  PetscCheck(s <= 1, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Matrix creation for 1d not implemented correctly for stencil width larger than 1");
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &nx, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, NULL, NULL, &gnx, NULL, NULL));

  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(PetscCalloc2(nx * nc, &cols, nx * nc, &ocols));

  /*
        note should be smaller for first and last process with no periodic
        does not handle dfill
  */
  cnt = 0;
  /* coupling with process to the left */
  for (i = 0; i < s; i++) {
    for (j = 0; j < nc; j++) {
      ocols[cnt] = ((rank == 0) ? 0 : (s - i) * (ofill[j + 1] - ofill[j]));
      cols[cnt]  = dfill[j + 1] - dfill[j] + (s + i) * (ofill[j + 1] - ofill[j]);
      if (rank == 0 && (dd->bx == DM_BOUNDARY_PERIODIC)) {
        if (size > 1) ocols[cnt] += (s - i) * (ofill[j + 1] - ofill[j]);
        else cols[cnt] += (s - i) * (ofill[j + 1] - ofill[j]);
      }
      maxcnt = PetscMax(maxcnt, ocols[cnt] + cols[cnt]);
      cnt++;
    }
  }
  for (i = s; i < nx - s; i++) {
    for (j = 0; j < nc; j++) {
      cols[cnt] = dfill[j + 1] - dfill[j] + 2 * s * (ofill[j + 1] - ofill[j]);
      maxcnt    = PetscMax(maxcnt, ocols[cnt] + cols[cnt]);
      cnt++;
    }
  }
  /* coupling with process to the right */
  for (i = nx - s; i < nx; i++) {
    for (j = 0; j < nc; j++) {
      ocols[cnt] = ((rank == (size - 1)) ? 0 : (i - nx + s + 1) * (ofill[j + 1] - ofill[j]));
      cols[cnt]  = dfill[j + 1] - dfill[j] + (s + nx - i - 1) * (ofill[j + 1] - ofill[j]);
      if ((rank == size - 1) && (dd->bx == DM_BOUNDARY_PERIODIC)) {
        if (size > 1) ocols[cnt] += (i - nx + s + 1) * (ofill[j + 1] - ofill[j]);
        else cols[cnt] += (i - nx + s + 1) * (ofill[j + 1] - ofill[j]);
      }
      maxcnt = PetscMax(maxcnt, ocols[cnt] + cols[cnt]);
      cnt++;
    }
  }

  PetscCall(MatSeqAIJSetPreallocation(J, 0, cols));
  PetscCall(MatMPIAIJSetPreallocation(J, 0, cols, 0, ocols));
  PetscCall(PetscFree2(cols, ocols));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscMalloc1(maxcnt, &cols));
    row = xs * nc;
    /* coupling with process to the left */
    for (i = xs; i < xs + s; i++) {
      for (j = 0; j < nc; j++) {
        cnt = 0;
        if (rank) {
          for (l = 0; l < s; l++) {
            for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i - s + l) * nc + ofill[k];
          }
        }
        if (rank == 0 && (dd->bx == DM_BOUNDARY_PERIODIC)) {
          for (l = 0; l < s; l++) {
            for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (m + i - s - l) * nc + ofill[k];
          }
        }
        if (dfill) {
          for (k = dfill[j]; k < dfill[j + 1]; k++) cols[cnt++] = i * nc + dfill[k];
        } else {
          for (k = 0; k < nc; k++) cols[cnt++] = i * nc + k;
        }
        for (l = 0; l < s; l++) {
          for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i + s - l) * nc + ofill[k];
        }
        PetscCall(MatSetValues(J, 1, &row, cnt, cols, NULL, INSERT_VALUES));
        row++;
      }
    }
    for (i = xs + s; i < xs + nx - s; i++) {
      for (j = 0; j < nc; j++) {
        cnt = 0;
        for (l = 0; l < s; l++) {
          for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i - s + l) * nc + ofill[k];
        }
        if (dfill) {
          for (k = dfill[j]; k < dfill[j + 1]; k++) cols[cnt++] = i * nc + dfill[k];
        } else {
          for (k = 0; k < nc; k++) cols[cnt++] = i * nc + k;
        }
        for (l = 0; l < s; l++) {
          for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i + s - l) * nc + ofill[k];
        }
        PetscCall(MatSetValues(J, 1, &row, cnt, cols, NULL, INSERT_VALUES));
        row++;
      }
    }
    /* coupling with process to the right */
    for (i = xs + nx - s; i < xs + nx; i++) {
      for (j = 0; j < nc; j++) {
        cnt = 0;
        for (l = 0; l < s; l++) {
          for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i - s + l) * nc + ofill[k];
        }
        if (dfill) {
          for (k = dfill[j]; k < dfill[j + 1]; k++) cols[cnt++] = i * nc + dfill[k];
        } else {
          for (k = 0; k < nc; k++) cols[cnt++] = i * nc + k;
        }
        if (rank < size - 1) {
          for (l = 0; l < s; l++) {
            for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i + s - l) * nc + ofill[k];
          }
        }
        if ((rank == size - 1) && (dd->bx == DM_BOUNDARY_PERIODIC)) {
          for (l = 0; l < s; l++) {
            for (k = ofill[j]; k < ofill[j + 1]; k++) cols[cnt++] = (i - s - l - m + 2) * nc + ofill[k];
          }
        }
        PetscCall(MatSetValues(J, 1, &row, cnt, cols, NULL, INSERT_VALUES));
        row++;
      }
    }
    PetscCall(PetscFree(cols));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM da, Mat J)
{
  PetscInt               xs, nx, i, i1, slot, gxs, gnx;
  PetscInt               m, dim, s, *cols = NULL, nc, *rows = NULL, col, cnt, l;
  PetscInt               istart, iend;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog, mltog;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, NULL, NULL, NULL, NULL, NULL, &nc, &s, &bx, NULL, NULL, NULL));
  if (bx == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_TRUE));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &nx, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, NULL, NULL, &gnx, NULL, NULL));

  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqAIJSetPreallocation(J, col * nc, NULL));
  PetscCall(MatMPIAIJSetPreallocation(J, col * nc, NULL, col * nc, NULL));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
  PetscCall(MatGetLocalToGlobalMapping(J, &mltog, NULL));
  if (!mltog) PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscMalloc2(nc, &rows, col * nc * nc, &cols));
    for (i = xs; i < xs + nx; i++) {
      istart = PetscMax(-s, gxs - i);
      iend   = PetscMin(s, gxs + gnx - i - 1);
      slot   = i - gxs;

      cnt = 0;
      for (i1 = istart; i1 < iend + 1; i1++) {
        cols[cnt++] = nc * (slot + i1);
        for (l = 1; l < nc; l++) {
          cols[cnt] = 1 + cols[cnt - 1];
          cnt++;
        }
      }
      rows[0] = nc * (slot);
      for (l = 1; l < nc; l++) rows[l] = 1 + rows[l - 1];
      PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, NULL, INSERT_VALUES));
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (bx == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_FALSE));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
    PetscCall(PetscFree2(rows, cols));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_1d_SeqAIJ_NoPreallocation(DM da, Mat J)
{
  PetscInt               xs, nx, i, i1, slot, gxs, gnx;
  PetscInt               m, dim, s, *cols = NULL, nc, *rows = NULL, col, cnt, l;
  PetscInt               istart, iend;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog, mltog;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, NULL, NULL, NULL, NULL, NULL, &nc, &s, &bx, NULL, NULL, NULL));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &nx, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, NULL, NULL, &gnx, NULL, NULL));

  PetscCall(MatSetBlockSize(J, nc));
  PetscCall(MatSeqAIJSetTotalPreallocation(J, nx * nc * col * nc));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
  PetscCall(MatGetLocalToGlobalMapping(J, &mltog, NULL));
  if (!mltog) PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscMalloc2(nc, &rows, col * nc * nc, &cols));
    for (i = xs; i < xs + nx; i++) {
      istart = PetscMax(-s, gxs - i);
      iend   = PetscMin(s, gxs + gnx - i - 1);
      slot   = i - gxs;

      cnt = 0;
      for (i1 = istart; i1 < iend + 1; i1++) {
        cols[cnt++] = nc * (slot + i1);
        for (l = 1; l < nc; l++) {
          cols[cnt] = 1 + cols[cnt - 1];
          cnt++;
        }
      }
      rows[0] = nc * (slot);
      for (l = 1; l < nc; l++) rows[l] = 1 + rows[l - 1];
      PetscCall(MatSetValuesLocal(J, nc, rows, cnt, cols, NULL, INSERT_VALUES));
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (bx == DM_BOUNDARY_NONE) PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_FALSE));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
    PetscCall(PetscFree2(rows, cols));
  }
  PetscCall(MatSetOption(J, MAT_SORTED_FULL, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, nc, col, cnt, *dnz, *onz;
  PetscInt               istart, iend, jstart, jend, ii, jj;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, NULL, NULL, NULL, &nc, &s, &bx, &by, NULL, &st));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc1(col * col * nc * nc, &cols));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nx * ny, nx * ny, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      slot   = i - gxs + gnx * (j - gys);

      /* Find block columns in block row */
      cnt = 0;
      for (ii = istart; ii < iend + 1; ii++) {
        for (jj = jstart; jj < jend + 1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
            cols[cnt++] = slot + ii + gnx * jj;
          }
        }
      }
      PetscCall(MatPreallocateSetLocalBlock(ltog, 1, &slot, ltog, cnt, cols, dnz, onz));
    }
  }
  PetscCall(MatSeqBAIJSetPreallocation(J, nc, 0, dnz));
  PetscCall(MatMPIBAIJSetPreallocation(J, nc, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        slot   = i - gxs + gnx * (j - gys);
        cnt    = 0;
        for (ii = istart; ii < iend + 1; ii++) {
          for (jj = jstart; jj < jend + 1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
              cols[cnt++] = slot + ii + gnx * jj;
            }
          }
        }
        PetscCall(MatSetValuesBlockedLocal(J, 1, &slot, cnt, cols, values, INSERT_VALUES));
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, k, nc, col, cnt, p, *dnz, *onz;
  PetscInt               istart, iend, jstart, jend, kstart, kend, zs, nz, gzs, gnz, ii, jj, kk;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by, bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, NULL, NULL, NULL, &nc, &s, &bx, &by, &bz, &st));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc1(col * col * col, &cols));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nx * ny * nz, nx * ny * nz, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      for (k = zs; k < zs + nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

        slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

        /* Find block columns in block row */
        cnt = 0;
        for (ii = istart; ii < iend + 1; ii++) {
          for (jj = jstart; jj < jend + 1; jj++) {
            for (kk = kstart; kk < kend + 1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                cols[cnt++] = slot + ii + gnx * jj + gnx * gny * kk;
              }
            }
          }
        }
        PetscCall(MatPreallocateSetLocalBlock(ltog, 1, &slot, ltog, cnt, cols, dnz, onz));
      }
    }
  }
  PetscCall(MatSeqBAIJSetPreallocation(J, nc, 0, dnz));
  PetscCall(MatMPIBAIJSetPreallocation(J, nc, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * col * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        for (k = zs; k < zs + nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

          slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

          cnt = 0;
          for (ii = istart; ii < iend + 1; ii++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (kk = kstart; kk < kend + 1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                  cols[cnt++] = slot + ii + gnx * jj + gnx * gny * kk;
                }
              }
            }
          }
          PetscCall(MatSetValuesBlockedLocal(J, 1, &slot, cnt, cols, values, INSERT_VALUES));
        }
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This helper is for of SBAIJ preallocation, to discard the lower-triangular values which are difficult to
  identify in the local ordering with periodic domain.
*/
static PetscErrorCode L2GFilterUpperTriangular(ISLocalToGlobalMapping ltog, PetscInt *row, PetscInt *cnt, PetscInt col[])
{
  PetscInt i, n;

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingApplyBlock(ltog, 1, row, row));
  PetscCall(ISLocalToGlobalMappingApplyBlock(ltog, *cnt, col, col));
  for (i = 0, n = 0; i < *cnt; i++) {
    if (col[i] >= *row) col[n++] = col[i];
  }
  *cnt = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, nc, col, cnt, *dnz, *onz;
  PetscInt               istart, iend, jstart, jend, ii, jj;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, NULL, NULL, NULL, NULL, &nc, &s, &bx, &by, NULL, &st));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &nx, &ny, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gnx, &gny, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc1(col * col * nc * nc, &cols));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nx * ny, nx * ny, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      slot   = i - gxs + gnx * (j - gys);

      /* Find block columns in block row */
      cnt = 0;
      for (ii = istart; ii < iend + 1; ii++) {
        for (jj = jstart; jj < jend + 1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) cols[cnt++] = slot + ii + gnx * jj;
        }
      }
      PetscCall(L2GFilterUpperTriangular(ltog, &slot, &cnt, cols));
      PetscCall(MatPreallocateSymmetricSetBlock(slot, cnt, cols, dnz, onz));
    }
  }
  PetscCall(MatSeqSBAIJSetPreallocation(J, nc, 0, dnz));
  PetscCall(MatMPISBAIJSetPreallocation(J, nc, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        slot   = i - gxs + gnx * (j - gys);

        /* Find block columns in block row */
        cnt = 0;
        for (ii = istart; ii < iend + 1; ii++) {
          for (jj = jstart; jj < jend + 1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) cols[cnt++] = slot + ii + gnx * jj;
          }
        }
        PetscCall(L2GFilterUpperTriangular(ltog, &slot, &cnt, cols));
        PetscCall(MatSetValuesBlocked(J, 1, &slot, cnt, cols, values, INSERT_VALUES));
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, k, nc, col, cnt, p, *dnz, *onz;
  PetscInt               istart, iend, jstart, jend, kstart, kend, zs, nz, gzs, gnz, ii, jj, kk;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by, bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, NULL, NULL, NULL, &nc, &s, &bx, &by, &bz, &st));
  col = 2 * s + 1;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  /* create the matrix */
  PetscCall(PetscMalloc1(col * col * col, &cols));

  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nx * ny * nz, nx * ny * nz, dnz, onz);
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      for (k = zs; k < zs + nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

        slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

        /* Find block columns in block row */
        cnt = 0;
        for (ii = istart; ii < iend + 1; ii++) {
          for (jj = jstart; jj < jend + 1; jj++) {
            for (kk = kstart; kk < kend + 1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) cols[cnt++] = slot + ii + gnx * jj + gnx * gny * kk;
            }
          }
        }
        PetscCall(L2GFilterUpperTriangular(ltog, &slot, &cnt, cols));
        PetscCall(MatPreallocateSymmetricSetBlock(slot, cnt, cols, dnz, onz));
      }
    }
  }
  PetscCall(MatSeqSBAIJSetPreallocation(J, nc, 0, dnz));
  PetscCall(MatMPISBAIJSetPreallocation(J, nc, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(col * col * col * nc * nc, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        for (k = zs; k < zs + nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

          slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

          cnt = 0;
          for (ii = istart; ii < iend + 1; ii++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (kk = kstart; kk < kend + 1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) cols[cnt++] = slot + ii + gnx * jj + gnx * gny * kk;
              }
            }
          }
          PetscCall(L2GFilterUpperTriangular(ltog, &slot, &cnt, cols));
          PetscCall(MatSetValuesBlocked(J, 1, &slot, cnt, cols, values, INSERT_VALUES));
        }
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM da, Mat J)
{
  PetscInt               xs, ys, nx, ny, i, j, slot, gxs, gys, gnx, gny;
  PetscInt               m, n, dim, s, *cols, k, nc, row, col, cnt, maxcnt = 0, l, p, *dnz, *onz;
  PetscInt               istart, iend, jstart, jend, kstart, kend, zs, nz, gzs, gnz, ii, jj, kk, M, N, P;
  DM_DA                 *dd = (DM_DA *)da->data;
  PetscInt               ifill_col, *dfill = dd->dfill, *ofill = dd->ofill;
  MPI_Comm               comm;
  PetscScalar           *values;
  DMBoundaryType         bx, by, bz;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem
  */
  PetscCall(DMDAGetInfo(da, &dim, &m, &n, &p, &M, &N, &P, &nc, &s, &bx, &by, &bz, &st));
  col = 2 * s + 1;

  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2 * s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2 * s >= n) removedups = PETSC_TRUE;
  if (P == 1 && 2 * s >= p) removedups = PETSC_TRUE;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gnx, &gny, &gnz));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));

  PetscCall(PetscMalloc1(col * col * col * nc, &cols));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));

  /* determine the matrix preallocation information */
  MatPreallocateBegin(comm, nc * nx * ny * nz, nc * nx * ny * nz, dnz, onz);

  PetscCall(MatSetBlockSize(J, nc));
  for (i = xs; i < xs + nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
    for (j = ys; j < ys + ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
      for (k = zs; k < zs + nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

        slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

        for (l = 0; l < nc; l++) {
          cnt = 0;
          for (ii = istart; ii < iend + 1; ii++) {
            for (jj = jstart; jj < jend + 1; jj++) {
              for (kk = kstart; kk < kend + 1; kk++) {
                if (ii || jj || kk) {
                  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                    for (ifill_col = ofill[l]; ifill_col < ofill[l + 1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                  }
                } else {
                  if (dfill) {
                    for (ifill_col = dfill[l]; ifill_col < dfill[l + 1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                  } else {
                    for (ifill_col = 0; ifill_col < nc; ifill_col++) cols[cnt++] = ifill_col + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                  }
                }
              }
            }
          }
          row    = l + nc * (slot);
          maxcnt = PetscMax(maxcnt, cnt);
          if (removedups) PetscCall(MatPreallocateSetLocalRemoveDups(ltog, 1, &row, ltog, cnt, cols, dnz, onz));
          else PetscCall(MatPreallocateSetLocal(ltog, 1, &row, ltog, cnt, cols, dnz, onz));
        }
      }
    }
  }
  PetscCall(MatSeqAIJSetPreallocation(J, 0, dnz));
  PetscCall(MatMPIAIJSetPreallocation(J, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatSetLocalToGlobalMapping(J, ltog, ltog));

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    PetscCall(PetscCalloc1(maxcnt, &values));
    for (i = xs; i < xs + nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, m - i - 1));
      for (j = ys; j < ys + ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, n - j - 1));
        for (k = zs; k < zs + nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s, -k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ? s : (PetscMin(s, p - k - 1));

          slot = i - gxs + gnx * (j - gys) + gnx * gny * (k - gzs);

          for (l = 0; l < nc; l++) {
            cnt = 0;
            for (ii = istart; ii < iend + 1; ii++) {
              for (jj = jstart; jj < jend + 1; jj++) {
                for (kk = kstart; kk < kend + 1; kk++) {
                  if (ii || jj || kk) {
                    if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) { /* entries on star*/
                      for (ifill_col = ofill[l]; ifill_col < ofill[l + 1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                    }
                  } else {
                    if (dfill) {
                      for (ifill_col = dfill[l]; ifill_col < dfill[l + 1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                    } else {
                      for (ifill_col = 0; ifill_col < nc; ifill_col++) cols[cnt++] = ifill_col + nc * (slot + ii + gnx * jj + gnx * gny * kk);
                    }
                  }
                }
              }
            }
            row = l + nc * (slot);
            PetscCall(MatSetValuesLocal(J, 1, &row, cnt, cols, values, INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(PetscFree(values));
    /* do not copy values to GPU since they are all zero and not yet needed there */
    PetscCall(MatBindToCPU(J, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatBindToCPU(J, PETSC_FALSE));
    PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscCall(PetscFree(cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}
