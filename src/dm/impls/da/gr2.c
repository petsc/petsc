
/*
   Plots vectors obtained with DMDACreate2d()
*/

#include <petsc/private/dmdaimpl.h> /*I  "petscdmda.h"   I*/
#include <petsc/private/glvisvecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petscdraw.h>

/*
        The data that is passed into the graphics callback
*/
typedef struct {
  PetscMPIInt        rank;
  PetscInt           m, n, dof, k;
  PetscReal          xmin, xmax, ymin, ymax, min, max;
  const PetscScalar *xy, *v;
  PetscBool          showaxis, showgrid;
  const char        *name0, *name1;
} ZoomCtx;

/*
       This does the drawing for one particular field
    in one particular set of coordinates. It is a callback
    called from PetscDrawZoom()
*/
PetscErrorCode VecView_MPI_Draw_DA2d_Zoom(PetscDraw draw, void *ctx)
{
  ZoomCtx           *zctx = (ZoomCtx *)ctx;
  PetscInt           m, n, i, j, k, dof, id, c1, c2, c3, c4;
  PetscReal          min, max, x1, x2, x3, x4, y_1, y2, y3, y4;
  const PetscScalar *xy, *v;

  PetscFunctionBegin;
  m   = zctx->m;
  n   = zctx->n;
  dof = zctx->dof;
  k   = zctx->k;
  xy  = zctx->xy;
  v   = zctx->v;
  min = zctx->min;
  max = zctx->max;

  /* PetscDraw the contour plot patch */
  PetscDrawCollectiveBegin(draw);
  for (j = 0; j < n - 1; j++) {
    for (i = 0; i < m - 1; i++) {
      id  = i + j * m;
      x1  = PetscRealPart(xy[2 * id]);
      y_1 = PetscRealPart(xy[2 * id + 1]);
      c1  = PetscDrawRealToColor(PetscRealPart(v[k + dof * id]), min, max);

      id = i + j * m + 1;
      x2 = PetscRealPart(xy[2 * id]);
      y2 = PetscRealPart(xy[2 * id + 1]);
      c2 = PetscDrawRealToColor(PetscRealPart(v[k + dof * id]), min, max);

      id = i + j * m + 1 + m;
      x3 = PetscRealPart(xy[2 * id]);
      y3 = PetscRealPart(xy[2 * id + 1]);
      c3 = PetscDrawRealToColor(PetscRealPart(v[k + dof * id]), min, max);

      id = i + j * m + m;
      x4 = PetscRealPart(xy[2 * id]);
      y4 = PetscRealPart(xy[2 * id + 1]);
      c4 = PetscDrawRealToColor(PetscRealPart(v[k + dof * id]), min, max);

      PetscCall(PetscDrawTriangle(draw, x1, y_1, x2, y2, x3, y3, c1, c2, c3));
      PetscCall(PetscDrawTriangle(draw, x1, y_1, x3, y3, x4, y4, c1, c3, c4));
      if (zctx->showgrid) {
        PetscCall(PetscDrawLine(draw, x1, y_1, x2, y2, PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw, x2, y2, x3, y3, PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw, x3, y3, x4, y4, PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw, x4, y4, x1, y_1, PETSC_DRAW_BLACK));
      }
    }
  }
  if (zctx->showaxis && !zctx->rank) {
    if (zctx->name0 || zctx->name1) {
      PetscReal xl, yl, xr, yr, x, y;
      PetscCall(PetscDrawGetCoordinates(draw, &xl, &yl, &xr, &yr));
      x  = xl + .30 * (xr - xl);
      xl = xl + .01 * (xr - xl);
      y  = yr - .30 * (yr - yl);
      yl = yl + .01 * (yr - yl);
      if (zctx->name0) PetscCall(PetscDrawString(draw, x, yl, PETSC_DRAW_BLACK, zctx->name0));
      if (zctx->name1) PetscCall(PetscDrawStringVertical(draw, xl, y, PETSC_DRAW_BLACK, zctx->name1));
    }
    /*
       Ideally we would use the PetscDrawAxis object to manage displaying the coordinate limits
       but that may require some refactoring.
    */
    {
      double    xmin = (double)zctx->xmin, ymin = (double)zctx->ymin;
      double    xmax = (double)zctx->xmax, ymax = (double)zctx->ymax;
      char      value[16];
      size_t    len;
      PetscReal w;
      PetscCall(PetscSNPrintf(value, 16, "%0.2e", xmin));
      PetscCall(PetscDrawString(draw, xmin, ymin - .05 * (ymax - ymin), PETSC_DRAW_BLACK, value));
      PetscCall(PetscSNPrintf(value, 16, "%0.2e", xmax));
      PetscCall(PetscStrlen(value, &len));
      PetscCall(PetscDrawStringGetSize(draw, &w, NULL));
      PetscCall(PetscDrawString(draw, xmax - len * w, ymin - .05 * (ymax - ymin), PETSC_DRAW_BLACK, value));
      PetscCall(PetscSNPrintf(value, 16, "%0.2e", ymin));
      PetscCall(PetscDrawString(draw, xmin - .05 * (xmax - xmin), ymin, PETSC_DRAW_BLACK, value));
      PetscCall(PetscSNPrintf(value, 16, "%0.2e", ymax));
      PetscCall(PetscDrawString(draw, xmin - .05 * (xmax - xmin), ymax, PETSC_DRAW_BLACK, value));
    }
  }
  PetscDrawCollectiveEnd(draw);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_MPI_Draw_DA2d(Vec xin, PetscViewer viewer)
{
  DM                  da, dac, dag;
  PetscInt            N, s, M, w, ncoors = 4;
  const PetscInt     *lx, *ly;
  PetscReal           coors[4];
  PetscDraw           draw, popup;
  PetscBool           isnull, useports = PETSC_FALSE;
  MPI_Comm            comm;
  Vec                 xlocal, xcoor, xcoorl;
  DMBoundaryType      bx, by;
  DMDAStencilType     st;
  ZoomCtx             zctx;
  PetscDrawViewPorts *ports = NULL;
  PetscViewerFormat   format;
  PetscInt           *displayfields;
  PetscInt            ndisplayfields, i, nbounds;
  const PetscReal    *bounds;

  PetscFunctionBegin;
  zctx.showgrid = PETSC_FALSE;
  zctx.showaxis = PETSC_TRUE;

  PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(0);

  PetscCall(PetscViewerDrawGetBounds(viewer, &nbounds, &bounds));

  PetscCall(VecGetDM(xin, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)xin), PETSC_ERR_ARG_WRONG, "Vector not generated from a DMDA");

  PetscCall(PetscObjectGetComm((PetscObject)xin, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &zctx.rank));

  PetscCall(DMDAGetInfo(da, NULL, &M, &N, NULL, &zctx.m, &zctx.n, NULL, &w, &s, &bx, &by, NULL, &st));
  PetscCall(DMDAGetOwnershipRanges(da, &lx, &ly, NULL));

  /*
     Obtain a sequential vector that is going to contain the local values plus ONE layer of
     ghosted values to draw the graphics from. We also need its corresponding DMDA (dac) that will
     update the local values plus ONE layer of ghost values.
  */
  PetscCall(PetscObjectQuery((PetscObject)da, "GraphicsGhosted", (PetscObject *)&xlocal));
  if (!xlocal) {
    if (bx != DM_BOUNDARY_NONE || by != DM_BOUNDARY_NONE || s != 1 || st != DMDA_STENCIL_BOX) {
      /*
         if original da is not of stencil width one, or periodic or not a box stencil then
         create a special DMDA to handle one level of ghost points for graphics
      */
      PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, M, N, zctx.m, zctx.n, w, 1, lx, ly, &dac));
      PetscCall(DMSetUp(dac));
      PetscCall(PetscInfo(da, "Creating auxiliary DMDA for managing graphics ghost points\n"));
    } else {
      /* otherwise we can use the da we already have */
      dac = da;
    }
    /* create local vector for holding ghosted values used in graphics */
    PetscCall(DMCreateLocalVector(dac, &xlocal));
    if (dac != da) {
      /* don't keep any public reference of this DMDA, is is only available through xlocal */
      PetscCall(PetscObjectDereference((PetscObject)dac));
    } else {
      /* remove association between xlocal and da, because below we compose in the opposite
         direction and if we left this connect we'd get a loop, so the objects could
         never be destroyed */
      PetscCall(PetscObjectRemoveReference((PetscObject)xlocal, "__PETSc_dm"));
    }
    PetscCall(PetscObjectCompose((PetscObject)da, "GraphicsGhosted", (PetscObject)xlocal));
    PetscCall(PetscObjectDereference((PetscObject)xlocal));
  } else {
    if (bx != DM_BOUNDARY_NONE || by != DM_BOUNDARY_NONE || s != 1 || st != DMDA_STENCIL_BOX) {
      PetscCall(VecGetDM(xlocal, &dac));
    } else {
      dac = da;
    }
  }

  /*
      Get local (ghosted) values of vector
  */
  PetscCall(DMGlobalToLocalBegin(dac, xin, INSERT_VALUES, xlocal));
  PetscCall(DMGlobalToLocalEnd(dac, xin, INSERT_VALUES, xlocal));
  PetscCall(VecGetArrayRead(xlocal, &zctx.v));

  /*
      Get coordinates of nodes
  */
  PetscCall(DMGetCoordinates(da, &xcoor));
  if (!xcoor) {
    PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
    PetscCall(DMGetCoordinates(da, &xcoor));
  }

  /*
      Determine the min and max coordinates in plot
  */
  PetscCall(VecStrideMin(xcoor, 0, NULL, &zctx.xmin));
  PetscCall(VecStrideMax(xcoor, 0, NULL, &zctx.xmax));
  PetscCall(VecStrideMin(xcoor, 1, NULL, &zctx.ymin));
  PetscCall(VecStrideMax(xcoor, 1, NULL, &zctx.ymax));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-draw_contour_axis", &zctx.showaxis, NULL));
  if (zctx.showaxis) {
    coors[0] = zctx.xmin - .05 * (zctx.xmax - zctx.xmin);
    coors[1] = zctx.ymin - .05 * (zctx.ymax - zctx.ymin);
    coors[2] = zctx.xmax + .05 * (zctx.xmax - zctx.xmin);
    coors[3] = zctx.ymax + .05 * (zctx.ymax - zctx.ymin);
  } else {
    coors[0] = zctx.xmin;
    coors[1] = zctx.ymin;
    coors[2] = zctx.xmax;
    coors[3] = zctx.ymax;
  }
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-draw_coordinates", coors, &ncoors, NULL));
  PetscCall(PetscInfo(da, "Preparing DMDA 2d contour plot coordinates %g %g %g %g\n", (double)coors[0], (double)coors[1], (double)coors[2], (double)coors[3]));

  /*
      Get local ghosted version of coordinates
  */
  PetscCall(PetscObjectQuery((PetscObject)da, "GraphicsCoordinateGhosted", (PetscObject *)&xcoorl));
  if (!xcoorl) {
    /* create DMDA to get local version of graphics */
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, M, N, zctx.m, zctx.n, 2, 1, lx, ly, &dag));
    PetscCall(DMSetUp(dag));
    PetscCall(PetscInfo(dag, "Creating auxiliary DMDA for managing graphics coordinates ghost points\n"));
    PetscCall(DMCreateLocalVector(dag, &xcoorl));
    PetscCall(PetscObjectCompose((PetscObject)da, "GraphicsCoordinateGhosted", (PetscObject)xcoorl));
    PetscCall(PetscObjectDereference((PetscObject)dag));
    PetscCall(PetscObjectDereference((PetscObject)xcoorl));
  } else PetscCall(VecGetDM(xcoorl, &dag));
  PetscCall(DMGlobalToLocalBegin(dag, xcoor, INSERT_VALUES, xcoorl));
  PetscCall(DMGlobalToLocalEnd(dag, xcoor, INSERT_VALUES, xcoorl));
  PetscCall(VecGetArrayRead(xcoorl, &zctx.xy));
  PetscCall(DMDAGetCoordinateName(da, 0, &zctx.name0));
  PetscCall(DMDAGetCoordinateName(da, 1, &zctx.name1));

  /*
      Get information about size of area each processor must do graphics for
  */
  PetscCall(DMDAGetInfo(dac, NULL, &M, &N, NULL, NULL, NULL, NULL, &zctx.dof, NULL, &bx, &by, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(dac, NULL, NULL, NULL, &zctx.m, &zctx.n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-draw_contour_grid", &zctx.showgrid, NULL));

  PetscCall(DMDASelectFields(da, &ndisplayfields, &displayfields));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-draw_ports", &useports, NULL));
  if (format == PETSC_VIEWER_DRAW_PORTS) useports = PETSC_TRUE;
  if (useports) {
    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawCheckResizedWindow(draw));
    PetscCall(PetscDrawClear(draw));
    PetscCall(PetscDrawViewPortsCreate(draw, ndisplayfields, &ports));
  }

  /*
      Loop over each field; drawing each in a different window
  */
  for (i = 0; i < ndisplayfields; i++) {
    zctx.k = displayfields[i];

    /* determine the min and max value in plot */
    PetscCall(VecStrideMin(xin, zctx.k, NULL, &zctx.min));
    PetscCall(VecStrideMax(xin, zctx.k, NULL, &zctx.max));
    if (zctx.k < nbounds) {
      zctx.min = bounds[2 * zctx.k];
      zctx.max = bounds[2 * zctx.k + 1];
    }
    if (zctx.min == zctx.max) {
      zctx.min -= 1.e-12;
      zctx.max += 1.e-12;
    }
    PetscCall(PetscInfo(da, "DMDA 2d contour plot min %g max %g\n", (double)zctx.min, (double)zctx.max));

    if (useports) {
      PetscCall(PetscDrawViewPortsSet(ports, i));
    } else {
      const char *title;
      PetscCall(PetscViewerDrawGetDraw(viewer, i, &draw));
      PetscCall(DMDAGetFieldName(da, zctx.k, &title));
      if (title) PetscCall(PetscDrawSetTitle(draw, title));
    }

    PetscCall(PetscDrawGetPopup(draw, &popup));
    PetscCall(PetscDrawScalePopup(popup, zctx.min, zctx.max));
    PetscCall(PetscDrawSetCoordinates(draw, coors[0], coors[1], coors[2], coors[3]));
    PetscCall(PetscDrawZoom(draw, VecView_MPI_Draw_DA2d_Zoom, &zctx));
    if (!useports) PetscCall(PetscDrawSave(draw));
  }
  if (useports) {
    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawSave(draw));
  }

  PetscCall(PetscDrawViewPortsDestroy(ports));
  PetscCall(PetscFree(displayfields));
  PetscCall(VecRestoreArrayRead(xcoorl, &zctx.xy));
  PetscCall(VecRestoreArrayRead(xlocal, &zctx.v));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode VecGetHDF5ChunkSize(DM_DA *da, Vec xin, PetscInt dimension, PetscInt timestep, hsize_t *chunkDims)
{
  PetscMPIInt comm_size;
  hsize_t     chunk_size, target_size, dim;
  hsize_t     vec_size = sizeof(PetscScalar) * da->M * da->N * da->P * da->w;
  hsize_t     avg_local_vec_size, KiB = 1024, MiB = KiB * KiB, GiB = MiB * KiB, min_size = MiB;
  hsize_t     max_chunks     = 64 * KiB; /* HDF5 internal limitation */
  hsize_t     max_chunk_size = 4 * GiB;  /* HDF5 internal limitation */
  PetscInt    zslices = da->p, yslices = da->n, xslices = da->m;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin), &comm_size));
  avg_local_vec_size = (hsize_t)PetscCeilInt(vec_size, comm_size); /* we will attempt to use this as the chunk size */

  target_size = (hsize_t)PetscMin((PetscInt64)vec_size, PetscMin((PetscInt64)max_chunk_size, PetscMax((PetscInt64)avg_local_vec_size, PetscMax(PetscCeilInt64(vec_size, max_chunks), (PetscInt64)min_size))));
  /* following line uses sizeof(PetscReal) instead of sizeof(PetscScalar) because the last dimension of chunkDims[] captures the 2* when complex numbers are being used */
  chunk_size = (hsize_t)PetscMax(1, chunkDims[0]) * PetscMax(1, chunkDims[1]) * PetscMax(1, chunkDims[2]) * PetscMax(1, chunkDims[3]) * PetscMax(1, chunkDims[4]) * PetscMax(1, chunkDims[5]) * sizeof(PetscReal);

  /*
   if size/rank > max_chunk_size, we need radical measures: even going down to
   avg_local_vec_size is not enough, so we simply use chunk size of 4 GiB no matter
   what, composed in the most efficient way possible.
   N.B. this minimises the number of chunks, which may or may not be the optimal
   solution. In a BG, for example, the optimal solution is probably to make # chunks = #
   IO nodes involved, but this author has no access to a BG to figure out how to
   reliably find the right number. And even then it may or may not be enough.
   */
  if (avg_local_vec_size > max_chunk_size) {
    /* check if we can just split local z-axis: is that enough? */
    zslices = PetscCeilInt(vec_size, da->p * max_chunk_size) * zslices;
    if (zslices > da->P) {
      /* lattice is too large in xy-directions, splitting z only is not enough */
      zslices = da->P;
      yslices = PetscCeilInt(vec_size, zslices * da->n * max_chunk_size) * yslices;
      if (yslices > da->N) {
        /* lattice is too large in x-direction, splitting along z, y is not enough */
        yslices = da->N;
        xslices = PetscCeilInt(vec_size, zslices * yslices * da->m * max_chunk_size) * xslices;
      }
    }
    dim = 0;
    if (timestep >= 0) ++dim;
    /* prefer to split z-axis, even down to planar slices */
    if (dimension == 3) {
      chunkDims[dim++] = (hsize_t)da->P / zslices;
      chunkDims[dim++] = (hsize_t)da->N / yslices;
      chunkDims[dim++] = (hsize_t)da->M / xslices;
    } else {
      /* This is a 2D world exceeding 4GiB in size; yes, I've seen them, even used myself */
      chunkDims[dim++] = (hsize_t)da->N / yslices;
      chunkDims[dim++] = (hsize_t)da->M / xslices;
    }
    chunk_size = (hsize_t)PetscMax(1, chunkDims[0]) * PetscMax(1, chunkDims[1]) * PetscMax(1, chunkDims[2]) * PetscMax(1, chunkDims[3]) * PetscMax(1, chunkDims[4]) * PetscMax(1, chunkDims[5]) * sizeof(double);
  } else {
    if (target_size < chunk_size) {
      /* only change the defaults if target_size < chunk_size */
      dim = 0;
      if (timestep >= 0) ++dim;
      /* prefer to split z-axis, even down to planar slices */
      if (dimension == 3) {
        /* try splitting the z-axis to core-size bits, i.e. divide chunk size by # comm_size in z-direction */
        if (target_size >= chunk_size / da->p) {
          /* just make chunks the size of <local_z>x<whole_world_y>x<whole_world_x>x<dof> */
          chunkDims[dim] = (hsize_t)PetscCeilInt(da->P, da->p);
        } else {
          /* oops, just splitting the z-axis is NOT ENOUGH, need to split more; let's be
           radical and let everyone write all they've got */
          chunkDims[dim++] = (hsize_t)PetscCeilInt(da->P, da->p);
          chunkDims[dim++] = (hsize_t)PetscCeilInt(da->N, da->n);
          chunkDims[dim++] = (hsize_t)PetscCeilInt(da->M, da->m);
        }
      } else {
        /* This is a 2D world exceeding 4GiB in size; yes, I've seen them, even used myself */
        if (target_size >= chunk_size / da->n) {
          /* just make chunks the size of <local_z>x<whole_world_y>x<whole_world_x>x<dof> */
          chunkDims[dim] = (hsize_t)PetscCeilInt(da->N, da->n);
        } else {
          /* oops, just splitting the z-axis is NOT ENOUGH, need to split more; let's be
           radical and let everyone write all they've got */
          chunkDims[dim++] = (hsize_t)PetscCeilInt(da->N, da->n);
          chunkDims[dim++] = (hsize_t)PetscCeilInt(da->M, da->m);
        }
      }
      chunk_size = (hsize_t)PetscMax(1, chunkDims[0]) * PetscMax(1, chunkDims[1]) * PetscMax(1, chunkDims[2]) * PetscMax(1, chunkDims[3]) * PetscMax(1, chunkDims[4]) * PetscMax(1, chunkDims[5]) * sizeof(double);
    } else {
      /* precomputed chunks are fine, we don't need to do anything */
    }
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecView_MPI_HDF5_DA(Vec xin, PetscViewer viewer)
{
  PetscViewer_HDF5  *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  DM                 dm;
  DM_DA             *da;
  hid_t              filespace;  /* file dataspace identifier */
  hid_t              chunkspace; /* chunk dataset property identifier */
  hid_t              dset_id;    /* dataset identifier */
  hid_t              memspace;   /* memory dataspace identifier */
  hid_t              file_id;
  hid_t              group;
  hid_t              memscalartype;  /* scalar type for mem (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  hid_t              filescalartype; /* scalar type for file (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  hsize_t            dim;
  hsize_t            maxDims[6] = {0}, dims[6] = {0}, chunkDims[6] = {0}, count[6] = {0}, offset[6] = {0}; /* we depend on these being sane later on  */
  PetscBool          timestepping = PETSC_FALSE, dim2, spoutput;
  PetscInt           timestep     = PETSC_MIN_INT, dimension;
  const PetscScalar *x;
  const char        *vecname;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5OpenGroup(viewer, NULL, &file_id, &group));
  PetscCall(PetscViewerHDF5IsTimestepping(viewer, &timestepping));
  if (timestepping) PetscCall(PetscViewerHDF5GetTimestep(viewer, &timestep));
  PetscCall(PetscViewerHDF5GetBaseDimension2(viewer, &dim2));
  PetscCall(PetscViewerHDF5GetSPOutput(viewer, &spoutput));

  PetscCall(VecGetDM(xin, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)xin), PETSC_ERR_ARG_WRONG, "Vector not generated from a DMDA");
  da = (DM_DA *)dm->data;
  PetscCall(DMGetDimension(dm, &dimension));

  /* Create the dataspace for the dataset.
   *
   * dims - holds the current dimensions of the dataset
   *
   * maxDims - holds the maximum dimensions of the dataset (unlimited
   * for the number of time steps with the current dimensions for the
   * other dimensions; so only additional time steps can be added).
   *
   * chunkDims - holds the size of a single time step (required to
   * permit extending dataset).
   */
  dim = 0;
  if (timestep >= 0) {
    dims[dim]      = timestep + 1;
    maxDims[dim]   = H5S_UNLIMITED;
    chunkDims[dim] = 1;
    ++dim;
  }
  if (dimension == 3) {
    PetscCall(PetscHDF5IntCast(da->P, dims + dim));
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
  }
  if (dimension > 1) {
    PetscCall(PetscHDF5IntCast(da->N, dims + dim));
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
  }
  PetscCall(PetscHDF5IntCast(da->M, dims + dim));
  maxDims[dim]   = dims[dim];
  chunkDims[dim] = dims[dim];
  ++dim;
  if (da->w > 1 || dim2) {
    PetscCall(PetscHDF5IntCast(da->w, dims + dim));
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
  }
  #if defined(PETSC_USE_COMPLEX)
  dims[dim]      = 2;
  maxDims[dim]   = dims[dim];
  chunkDims[dim] = dims[dim];
  ++dim;
  #endif

  PetscCall(VecGetHDF5ChunkSize(da, xin, dimension, timestep, chunkDims));

  PetscCallHDF5Return(filespace, H5Screate_simple, (dim, dims, maxDims));

  #if defined(PETSC_USE_REAL_SINGLE)
  memscalartype  = H5T_NATIVE_FLOAT;
  filescalartype = H5T_NATIVE_FLOAT;
  #elif defined(PETSC_USE_REAL___FLOAT128)
    #error "HDF5 output with 128 bit floats not supported."
  #elif defined(PETSC_USE_REAL___FP16)
    #error "HDF5 output with 16 bit floats not supported."
  #else
  memscalartype = H5T_NATIVE_DOUBLE;
  if (spoutput == PETSC_TRUE) filescalartype = H5T_NATIVE_FLOAT;
  else filescalartype = H5T_NATIVE_DOUBLE;
  #endif

  /* Create the dataset with default properties and close filespace */
  PetscCall(PetscObjectGetName((PetscObject)xin, &vecname));
  if (!H5Lexists(group, vecname, H5P_DEFAULT)) {
    /* Create chunk */
    PetscCallHDF5Return(chunkspace, H5Pcreate, (H5P_DATASET_CREATE));
    PetscCallHDF5(H5Pset_chunk, (chunkspace, dim, chunkDims));

    PetscCallHDF5Return(dset_id, H5Dcreate2, (group, vecname, filescalartype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
  } else {
    PetscCallHDF5Return(dset_id, H5Dopen2, (group, vecname, H5P_DEFAULT));
    PetscCallHDF5(H5Dset_extent, (dset_id, dims));
  }
  PetscCallHDF5(H5Sclose, (filespace));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  if (dimension == 3) PetscCall(PetscHDF5IntCast(da->zs, offset + dim++));
  if (dimension > 1) PetscCall(PetscHDF5IntCast(da->ys, offset + dim++));
  PetscCall(PetscHDF5IntCast(da->xs / da->w, offset + dim++));
  if (da->w > 1 || dim2) offset[dim++] = 0;
  #if defined(PETSC_USE_COMPLEX)
  offset[dim++] = 0;
  #endif
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  if (dimension == 3) PetscCall(PetscHDF5IntCast(da->ze - da->zs, count + dim++));
  if (dimension > 1) PetscCall(PetscHDF5IntCast(da->ye - da->ys, count + dim++));
  PetscCall(PetscHDF5IntCast((da->xe - da->xs) / da->w, count + dim++));
  if (da->w > 1 || dim2) PetscCall(PetscHDF5IntCast(da->w, count + dim++));
  #if defined(PETSC_USE_COMPLEX)
  count[dim++] = 2;
  #endif
  PetscCallHDF5Return(memspace, H5Screate_simple, (dim, count, NULL));
  PetscCallHDF5Return(filespace, H5Dget_space, (dset_id));
  PetscCallHDF5(H5Sselect_hyperslab, (filespace, H5S_SELECT_SET, offset, NULL, count, NULL));

  PetscCall(VecGetArrayRead(xin, &x));
  PetscCallHDF5(H5Dwrite, (dset_id, memscalartype, memspace, filespace, hdf5->dxpl_id, x));
  PetscCallHDF5(H5Fflush, (file_id, H5F_SCOPE_GLOBAL));
  PetscCall(VecRestoreArrayRead(xin, &x));

  #if defined(PETSC_USE_COMPLEX)
  {
    PetscBool tru = PETSC_TRUE;
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)xin, "complex", PETSC_BOOL, &tru));
  }
  #endif
  if (timestepping) PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)xin, "timestepping", PETSC_BOOL, &timestepping));

  /* Close/release resources */
  if (group != file_id) PetscCallHDF5(H5Gclose, (group));
  PetscCallHDF5(H5Sclose, (filespace));
  PetscCallHDF5(H5Sclose, (memspace));
  PetscCallHDF5(H5Dclose, (dset_id));
  PetscCall(PetscInfo(xin, "Wrote Vec object with name %s\n", vecname));
  PetscFunctionReturn(0);
}
#endif

extern PetscErrorCode VecView_MPI_Draw_DA1d(Vec, PetscViewer);

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode DMDAArrayMPIIO(DM da, PetscViewer viewer, Vec xin, PetscBool write)
{
  MPI_File           mfdes;
  PetscMPIInt        gsizes[4], lsizes[4], lstarts[4], asiz, dof;
  MPI_Datatype       view;
  const PetscScalar *array;
  MPI_Offset         off;
  MPI_Aint           ub, ul;
  PetscInt           type, rows, vecrows, tr[2];
  DM_DA             *dd = (DM_DA *)da->data;
  PetscBool          skipheader;

  PetscFunctionBegin;
  PetscCall(VecGetSize(xin, &vecrows));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &skipheader));
  if (!write) {
    /* Read vector header. */
    if (!skipheader) {
      PetscCall(PetscViewerBinaryRead(viewer, tr, 2, NULL, PETSC_INT));
      type = tr[0];
      rows = tr[1];
      PetscCheck(type == VEC_FILE_CLASSID, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Not vector next in file");
      PetscCheck(rows == vecrows, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_SIZ, "Vector in file not same size as DMDA vector");
    }
  } else {
    tr[0] = VEC_FILE_CLASSID;
    tr[1] = vecrows;
    if (!skipheader) PetscCall(PetscViewerBinaryWrite(viewer, tr, 2, PETSC_INT));
  }

  PetscCall(PetscMPIIntCast(dd->w, &dof));
  gsizes[0] = dof;
  PetscCall(PetscMPIIntCast(dd->M, gsizes + 1));
  PetscCall(PetscMPIIntCast(dd->N, gsizes + 2));
  PetscCall(PetscMPIIntCast(dd->P, gsizes + 3));
  lsizes[0] = dof;
  PetscCall(PetscMPIIntCast((dd->xe - dd->xs) / dof, lsizes + 1));
  PetscCall(PetscMPIIntCast(dd->ye - dd->ys, lsizes + 2));
  PetscCall(PetscMPIIntCast(dd->ze - dd->zs, lsizes + 3));
  lstarts[0] = 0;
  PetscCall(PetscMPIIntCast(dd->xs / dof, lstarts + 1));
  PetscCall(PetscMPIIntCast(dd->ys, lstarts + 2));
  PetscCall(PetscMPIIntCast(dd->zs, lstarts + 3));
  PetscCallMPI(MPI_Type_create_subarray(da->dim + 1, gsizes, lsizes, lstarts, MPI_ORDER_FORTRAN, MPIU_SCALAR, &view));
  PetscCallMPI(MPI_Type_commit(&view));

  PetscCall(PetscViewerBinaryGetMPIIODescriptor(viewer, &mfdes));
  PetscCall(PetscViewerBinaryGetMPIIOOffset(viewer, &off));
  PetscCallMPI(MPI_File_set_view(mfdes, off, MPIU_SCALAR, view, (char *)"native", MPI_INFO_NULL));
  PetscCall(VecGetArrayRead(xin, &array));
  asiz = lsizes[1] * (lsizes[2] > 0 ? lsizes[2] : 1) * (lsizes[3] > 0 ? lsizes[3] : 1) * dof;
  if (write) PetscCall(MPIU_File_write_all(mfdes, (PetscScalar *)array, asiz, MPIU_SCALAR, MPI_STATUS_IGNORE));
  else PetscCall(MPIU_File_read_all(mfdes, (PetscScalar *)array, asiz, MPIU_SCALAR, MPI_STATUS_IGNORE));
  PetscCallMPI(MPI_Type_get_extent(view, &ul, &ub));
  PetscCall(PetscViewerBinaryAddMPIIOOffset(viewer, ub));
  PetscCall(VecRestoreArrayRead(xin, &array));
  PetscCallMPI(MPI_Type_free(&view));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecView_MPI_DA(Vec xin, PetscViewer viewer)
{
  DM        da;
  PetscInt  dim;
  Vec       natural;
  PetscBool isdraw, isvtk, isglvis;
#if defined(PETSC_HAVE_HDF5)
  PetscBool ishdf5;
#endif
  const char       *prefix, *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(VecGetDM(xin, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)xin), PETSC_ERR_ARG_WRONG, "Vector not generated from a DMDA");
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERVTK, &isvtk));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis));
  if (isdraw) {
    PetscCall(DMDAGetInfo(da, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    if (dim == 1) {
      PetscCall(VecView_MPI_Draw_DA1d(xin, viewer));
    } else if (dim == 2) {
      PetscCall(VecView_MPI_Draw_DA2d(xin, viewer));
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot graphically view vector associated with this dimensional DMDA %" PetscInt_FMT, dim);
  } else if (isvtk) { /* Duplicate the Vec */
    Vec Y;
    PetscCall(VecDuplicate(xin, &Y));
    if (((PetscObject)xin)->name) {
      /* If xin was named, copy the name over to Y. The duplicate names are safe because nobody else will ever see Y. */
      PetscCall(PetscObjectSetName((PetscObject)Y, ((PetscObject)xin)->name));
    }
    PetscCall(VecCopy(xin, Y));
    {
      PetscObject dmvtk;
      PetscBool   compatible, compatibleSet;
      PetscCall(PetscViewerVTKGetDM(viewer, &dmvtk));
      if (dmvtk) {
        PetscValidHeaderSpecific((DM)dmvtk, DM_CLASSID, 2);
        PetscCall(DMGetCompatibility(da, (DM)dmvtk, &compatible, &compatibleSet));
        PetscCheck(compatibleSet && compatible, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cannot confirm compatibility of DMs associated with Vecs viewed in the same VTK file. Check that grids are the same.");
      }
      PetscCall(PetscViewerVTKAddField(viewer, (PetscObject)da, DMDAVTKWriteAll, PETSC_DEFAULT, PETSC_VTK_POINT_FIELD, PETSC_FALSE, (PetscObject)Y));
    }
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(VecView_MPI_HDF5_DA(xin, viewer));
#endif
  } else if (isglvis) {
    PetscCall(VecView_GLVis(xin, viewer));
  } else {
#if defined(PETSC_HAVE_MPIIO)
    PetscBool isbinary, isMPIIO;

    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
    if (isbinary) {
      PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &isMPIIO));
      if (isMPIIO) {
        PetscCall(DMDAArrayMPIIO(da, viewer, xin, PETSC_TRUE));
        PetscFunctionReturn(0);
      }
    }
#endif

    /* call viewer on natural ordering */
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)xin, &prefix));
    PetscCall(DMDACreateNaturalVector(da, &natural));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)natural, prefix));
    PetscCall(DMDAGlobalToNaturalBegin(da, xin, INSERT_VALUES, natural));
    PetscCall(DMDAGlobalToNaturalEnd(da, xin, INSERT_VALUES, natural));
    PetscCall(PetscObjectGetName((PetscObject)xin, &name));
    PetscCall(PetscObjectSetName((PetscObject)natural, name));

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      /* temporarily remove viewer format so it won't trigger in the VecView() */
      PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT));
    }

    ((PetscObject)natural)->donotPetscObjectPrintClassNamePrefixType = PETSC_TRUE;
    PetscCall(VecView(natural, viewer));
    ((PetscObject)natural)->donotPetscObjectPrintClassNamePrefixType = PETSC_FALSE;

    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      MPI_Comm    comm;
      FILE       *info;
      const char *fieldname;
      char        fieldbuf[256];
      PetscInt    dim, ni, nj, nk, pi, pj, pk, dof, n;

      /* set the viewer format back into the viewer */
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
      PetscCall(PetscViewerBinaryGetInfoPointer(viewer, &info));
      PetscCall(DMDAGetInfo(da, &dim, &ni, &nj, &nk, &pi, &pj, &pk, &dof, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscFPrintf(comm, info, "#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
      PetscCall(PetscFPrintf(comm, info, "#$$ tmp = PetscBinaryRead(fd); \n"));
      if (dim == 1) PetscCall(PetscFPrintf(comm, info, "#$$ tmp = reshape(tmp,%" PetscInt_FMT ",%" PetscInt_FMT ");\n", dof, ni));
      if (dim == 2) PetscCall(PetscFPrintf(comm, info, "#$$ tmp = reshape(tmp,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ");\n", dof, ni, nj));
      if (dim == 3) PetscCall(PetscFPrintf(comm, info, "#$$ tmp = reshape(tmp,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ");\n", dof, ni, nj, nk));

      for (n = 0; n < dof; n++) {
        PetscCall(DMDAGetFieldName(da, n, &fieldname));
        if (!fieldname || !fieldname[0]) {
          PetscCall(PetscSNPrintf(fieldbuf, sizeof fieldbuf, "field%" PetscInt_FMT, n));
          fieldname = fieldbuf;
        }
        if (dim == 1) PetscCall(PetscFPrintf(comm, info, "#$$ Set.%s.%s = squeeze(tmp(%" PetscInt_FMT ",:))';\n", name, fieldname, n + 1));
        if (dim == 2) PetscCall(PetscFPrintf(comm, info, "#$$ Set.%s.%s = squeeze(tmp(%" PetscInt_FMT ",:,:))';\n", name, fieldname, n + 1));
        if (dim == 3) PetscCall(PetscFPrintf(comm, info, "#$$ Set.%s.%s = permute(squeeze(tmp(%" PetscInt_FMT ",:,:,:)),[2 1 3]);\n", name, fieldname, n + 1));
      }
      PetscCall(PetscFPrintf(comm, info, "#$$ clear tmp; \n"));
      PetscCall(PetscFPrintf(comm, info, "#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
    }

    PetscCall(VecDestroy(&natural));
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecLoad_HDF5_DA(Vec xin, PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  DM                da;
  int               dim, rdim;
  hsize_t           dims[6] = {0}, count[6] = {0}, offset[6] = {0};
  PetscBool         dim2 = PETSC_FALSE, timestepping = PETSC_FALSE;
  PetscInt          dimension, timestep              = PETSC_MIN_INT, dofInd;
  PetscScalar      *x;
  const char       *vecname;
  hid_t             filespace; /* file dataspace identifier */
  hid_t             dset_id;   /* dataset identifier */
  hid_t             memspace;  /* memory dataspace identifier */
  hid_t             file_id, group;
  hid_t             scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  DM_DA            *dd;

  PetscFunctionBegin;
  #if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
  #elif defined(PETSC_USE_REAL___FLOAT128)
    #error "HDF5 output with 128 bit floats not supported."
  #elif defined(PETSC_USE_REAL___FP16)
    #error "HDF5 output with 16 bit floats not supported."
  #else
  scalartype = H5T_NATIVE_DOUBLE;
  #endif

  PetscCall(PetscViewerHDF5OpenGroup(viewer, NULL, &file_id, &group));
  PetscCall(PetscObjectGetName((PetscObject)xin, &vecname));
  PetscCall(PetscViewerHDF5CheckTimestepping_Internal(viewer, vecname));
  PetscCall(PetscViewerHDF5IsTimestepping(viewer, &timestepping));
  if (timestepping) PetscCall(PetscViewerHDF5GetTimestep(viewer, &timestep));
  PetscCall(VecGetDM(xin, &da));
  dd = (DM_DA *)da->data;
  PetscCall(DMGetDimension(da, &dimension));

  /* Open dataset */
  PetscCallHDF5Return(dset_id, H5Dopen2, (group, vecname, H5P_DEFAULT));

  /* Retrieve the dataspace for the dataset */
  PetscCallHDF5Return(filespace, H5Dget_space, (dset_id));
  PetscCallHDF5Return(rdim, H5Sget_simple_extent_dims, (filespace, dims, NULL));

  /* Expected dimension for holding the dof's */
  #if defined(PETSC_USE_COMPLEX)
  dofInd = rdim - 2;
  #else
  dofInd = rdim - 1;
  #endif

  /* The expected number of dimensions, assuming basedimension2 = false */
  dim = dimension;
  if (dd->w > 1) ++dim;
  if (timestep >= 0) ++dim;
  #if defined(PETSC_USE_COMPLEX)
  ++dim;
  #endif

  /* In this case the input dataset have one extra, unexpected dimension. */
  if (rdim == dim + 1) {
    /* In this case the block size unity */
    if (dd->w == 1 && dims[dofInd] == 1) dim2 = PETSC_TRUE;

    /* Special error message for the case where dof does not match the input file */
    else PetscCheck(dd->w == (PetscInt)dims[dofInd], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Number of dofs in file is %" PetscInt_FMT ", not %" PetscInt_FMT " as expected", (PetscInt)dims[dofInd], dd->w);

    /* Other cases where rdim != dim cannot be handled currently */
  } else PetscCheck(rdim == dim, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file is %d, not %d as expected with dof = %" PetscInt_FMT, rdim, dim, dd->w);

  /* Set up the hyperslab size */
  dim = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    count[dim]  = 1;
    ++dim;
  }
  if (dimension == 3) {
    PetscCall(PetscHDF5IntCast(dd->zs, offset + dim));
    PetscCall(PetscHDF5IntCast(dd->ze - dd->zs, count + dim));
    ++dim;
  }
  if (dimension > 1) {
    PetscCall(PetscHDF5IntCast(dd->ys, offset + dim));
    PetscCall(PetscHDF5IntCast(dd->ye - dd->ys, count + dim));
    ++dim;
  }
  PetscCall(PetscHDF5IntCast(dd->xs / dd->w, offset + dim));
  PetscCall(PetscHDF5IntCast((dd->xe - dd->xs) / dd->w, count + dim));
  ++dim;
  if (dd->w > 1 || dim2) {
    offset[dim] = 0;
    PetscCall(PetscHDF5IntCast(dd->w, count + dim));
    ++dim;
  }
  #if defined(PETSC_USE_COMPLEX)
  offset[dim] = 0;
  count[dim]  = 2;
  ++dim;
  #endif

  /* Create the memory and filespace */
  PetscCallHDF5Return(memspace, H5Screate_simple, (dim, count, NULL));
  PetscCallHDF5(H5Sselect_hyperslab, (filespace, H5S_SELECT_SET, offset, NULL, count, NULL));

  PetscCall(VecGetArray(xin, &x));
  PetscCallHDF5(H5Dread, (dset_id, scalartype, memspace, filespace, hdf5->dxpl_id, x));
  PetscCall(VecRestoreArray(xin, &x));

  /* Close/release resources */
  if (group != file_id) PetscCallHDF5(H5Gclose, (group));
  PetscCallHDF5(H5Sclose, (filespace));
  PetscCallHDF5(H5Sclose, (memspace));
  PetscCallHDF5(H5Dclose, (dset_id));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecLoad_Binary_DA(Vec xin, PetscViewer viewer)
{
  DM          da;
  Vec         natural;
  const char *prefix;
  PetscInt    bs;
  PetscBool   flag;
  DM_DA      *dd;
#if defined(PETSC_HAVE_MPIIO)
  PetscBool isMPIIO;
#endif

  PetscFunctionBegin;
  PetscCall(VecGetDM(xin, &da));
  dd = (DM_DA *)da->data;
#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &isMPIIO));
  if (isMPIIO) {
    PetscCall(DMDAArrayMPIIO(da, viewer, xin, PETSC_FALSE));
    PetscFunctionReturn(0);
  }
#endif

  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)xin, &prefix));
  PetscCall(DMDACreateNaturalVector(da, &natural));
  PetscCall(PetscObjectSetName((PetscObject)natural, ((PetscObject)xin)->name));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)natural, prefix));
  PetscCall(VecLoad(natural, viewer));
  PetscCall(DMDANaturalToGlobalBegin(da, natural, INSERT_VALUES, xin));
  PetscCall(DMDANaturalToGlobalEnd(da, natural, INSERT_VALUES, xin));
  PetscCall(VecDestroy(&natural));
  PetscCall(PetscInfo(xin, "Loading vector from natural ordering into DMDA\n"));
  PetscCall(PetscOptionsGetInt(NULL, ((PetscObject)xin)->prefix, "-vecload_block_size", &bs, &flag));
  if (flag && bs != dd->w) PetscCall(PetscInfo(xin, "Block size in file %" PetscInt_FMT " not equal to DMDA's dof %" PetscInt_FMT "\n", bs, dd->w));
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Default_DA(Vec xin, PetscViewer viewer)
{
  DM        da;
  PetscBool isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool ishdf5;
#endif

  PetscFunctionBegin;
  PetscCall(VecGetDM(xin, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)xin), PETSC_ERR_ARG_WRONG, "Vector not generated from a DMDA");

#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));

  if (isbinary) {
    PetscCall(VecLoad_Binary_DA(xin, viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(VecLoad_HDF5_DA(xin, viewer));
#endif
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Viewer type %s not supported for vector loading", ((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}
