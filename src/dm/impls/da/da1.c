
/*
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95
*/

#include <petsc/private/dmdaimpl.h>     /*I  "petscdmda.h"   I*/

#include <petscdraw.h>
static PetscErrorCode DMView_DA_1d(DM da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii,isdraw,isglvis,isbinary;
  DM_DA          *dd = (DM_DA*)da->data;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool ismatlab;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_LOAD_BALANCE) {
      PetscInt      i,nmax = 0,nmin = PETSC_MAX_INT,navg = 0,*nz,nzlocal;
      DMDALocalInfo info;
      PetscMPIInt   size;
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)da),&size);CHKERRQ(ierr);
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      nzlocal = info.xm;
      ierr = PetscMalloc1(size,&nz);CHKERRQ(ierr);
      ierr = MPI_Allgather(&nzlocal,1,MPIU_INT,nz,1,MPIU_INT,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
      for (i=0; i<(PetscInt)size; i++) {
        nmax = PetscMax(nmax,nz[i]);
        nmin = PetscMin(nmin,nz[i]);
        navg += nz[i];
      }
      ierr = PetscFree(nz);CHKERRQ(ierr);
      navg = navg/size;
      ierr = PetscViewerASCIIPrintf(viewer,"  Load Balance - Grid Points: Min %D  avg %D  max %D\n",nmin,navg,nmax);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (format != PETSC_VIEWER_ASCII_VTK_DEPRECATED && format != PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED && format != PETSC_VIEWER_ASCII_GLVIS) {
      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D m %D w %D s %D\n",rank,dd->M,dd->m,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D\n",info.xs,info.xs+info.xm);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
      ierr = DMView_DA_GLVis(da,viewer);CHKERRQ(ierr);
    } else {
      ierr = DMView_DA_VTK(da, viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw draw;
    double    ymin = -1,ymax = 1,xmin = -1,xmax = dd->M,x;
    PetscInt  base;
    char      node[10];
    PetscBool isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);

    ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* first processor draws all node lines */
    if (!rank) {
      PetscInt xmin_tmp;
      ymin = 0.0; ymax = 0.3;
      for (xmin_tmp=0; xmin_tmp < dd->M; xmin_tmp++) {
        ierr = PetscDrawLine(draw,(double)xmin_tmp,ymin,(double)xmin_tmp,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = dd->M - 1;
      ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = dd->xs / dd->w; xmax = (dd->xe / dd->w)  - 1;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    /* Put in index numbers */
    base = dd->base / dd->w;
    for (x=xmin; x<=xmax; x++) {
      ierr = PetscSNPrintf(node,sizeof(node),"%d",(int)base++);CHKERRQ(ierr);
      ierr = PetscDrawString(draw,x,ymin,PETSC_DRAW_RED,node);CHKERRQ(ierr);
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
    ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  } else if (isglvis) {
    ierr = DMView_DA_GLVis(da,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = DMView_DA_Binary(da,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = DMView_DA_Matlab(da,viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}


PetscErrorCode  DMSetUp_DA_1D(DM da)
{
  DM_DA            *dd   = (DM_DA*)da->data;
  const PetscInt   M     = dd->M;
  const PetscInt   dof   = dd->w;
  const PetscInt   s     = dd->s;
  const PetscInt   sDist = s;  /* stencil distance in points */
  const PetscInt   *lx   = dd->lx;
  DMBoundaryType   bx    = dd->bx;
  MPI_Comm         comm;
  Vec              local, global;
  VecScatter       gtol;
  IS               to, from;
  PetscBool        flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscMPIInt      rank, size;
  PetscInt         i,*idx,nn,left,xs,xe,x,Xs,Xe,start,m,IXs,IXe;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  dd->p = 1;
  dd->n = 1;
  dd->m = size;
  m     = dd->m;

  if (s > 0) {
    /* if not communicating data then should be ok to have nothing on some processes */
    if (M < m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"More processes than data points! %D %D",m,M);
    if ((M-1) < s && size > 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array is too small for stencil! %D %D",M-1,s);
  }

  /*
     Determine locally owned region
     xs is the first local node number, x is the number of local nodes
  */
  if (!lx) {
    ierr = PetscMalloc1(m, &dd->lx);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)da)->options,((PetscObject)da)->prefix,"-da_partition_blockcomm",&flg1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)da)->options,((PetscObject)da)->prefix,"-da_partition_nodes_at_end",&flg2,NULL);CHKERRQ(ierr);
    if (flg1) {      /* Block Comm type Distribution */
      xs = rank*M/m;
      x  = (rank + 1)*M/m - xs;
    } else if (flg2) { /* The odd nodes are evenly distributed across last nodes */
      x = (M + rank)/m;
      if (M/m == x) xs = rank*x;
      else          xs = rank*(x-1) + (M+rank)%(x*m);
    } else { /* The odd nodes are evenly distributed across the first k nodes */
      /* Regular PETSc Distribution */
      x = M/m + ((M % m) > rank);
      if (rank >= (M % m)) xs = (rank * (PetscInt)(M/m) + M % m);
      else                 xs = rank * (PetscInt)(M/m) + rank;
    }
    ierr = MPI_Allgather(&xs,1,MPIU_INT,dd->lx,1,MPIU_INT,comm);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) dd->lx[i] = dd->lx[i+1] - dd->lx[i];
    dd->lx[m-1] = M - dd->lx[m-1];
  } else {
    x  = lx[rank];
    xs = 0;
    for (i=0; i<rank; i++) xs += lx[i];
    /* verify that data user provided is consistent */
    left = xs;
    for (i=rank; i<size; i++) left += lx[i];
    if (left != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M %D %D",left,M);
  }

  /*
   check if the scatter requires more than one process neighbor or wraps around
   the domain more than once
  */
  if ((x < s) & ((M > 1) | (bx == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s);

  xe  = xs + x;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs-sDist > 0) {
    Xs  = xs - sDist;
    IXs = xs - sDist;
  } else {
    if (bx) Xs = xs - sDist;
    else Xs = 0;
    IXs = 0;
  }
  if (xe+sDist <= M) {
    Xe  = xe + sDist;
    IXe = xe + sDist;
  } else {
    if (bx) Xe = xe + sDist;
    else Xe = M;
    IXe = M;
  }

  if (bx == DM_BOUNDARY_PERIODIC || bx == DM_BOUNDARY_MIRROR) {
    Xs  = xs - sDist;
    Xe  = xe + sDist;
    IXs = xs - sDist;
    IXe = xe + sDist;
  }

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = dof*x;
  ierr       = VecCreateMPIWithArray(comm,dof,dd->Nlocal,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
  dd->nlocal = dof*(Xe-Xs);
  ierr       = VecCreateSeqWithArray(PETSC_COMM_SELF,dof,dd->nlocal,NULL,&local);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(global,&start,NULL);CHKERRQ(ierr);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  ierr = ISCreateStride(comm,dof*(IXe-IXs),dof*(IXs-Xs),1,&to);CHKERRQ(ierr);

  ierr = PetscMalloc1(x+2*sDist,&idx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)da,(x+2*(sDist))*sizeof(PetscInt));CHKERRQ(ierr);

  for (i=0; i<IXs-Xs; i++) idx[i] = -1; /* prepend with -1s if needed for ghosted case*/

  nn = IXs-Xs;
  if (bx == DM_BOUNDARY_PERIODIC) { /* Handle all cases with periodic first */
    for (i=0; i<sDist; i++) {  /* Left ghost points */
      if ((xs-sDist+i)>=0) idx[nn++] = xs-sDist+i;
      else                 idx[nn++] = M+(xs-sDist+i);
    }

    for (i=0; i<x; i++) idx [nn++] = xs + i;  /* Non-ghost points */

    for (i=0; i<sDist; i++) { /* Right ghost points */
      if ((xe+i)<M) idx [nn++] =  xe+i;
      else          idx [nn++] = (xe+i) - M;
    }
  } else if (bx == DM_BOUNDARY_MIRROR) { /* Handle all cases with periodic first */
    for (i=0; i<(sDist); i++) {  /* Left ghost points */
      if ((xs-sDist+i)>=0) idx[nn++] = xs-sDist+i;
      else                 idx[nn++] = sDist - i;
    }

    for (i=0; i<x; i++) idx [nn++] = xs + i;  /* Non-ghost points */

    for (i=0; i<(sDist); i++) { /* Right ghost points */
      if ((xe+i)<M) idx[nn++] =  xe+i;
      else          idx[nn++] = M - (i + 2);
    }
  } else {      /* Now do all cases with no periodicity */
    if (0 <= xs-sDist) {
      for (i=0; i<sDist; i++) idx[nn++] = xs - sDist + i;
    } else {
      for (i=0; i<xs; i++) idx[nn++] = i;
    }

    for (i=0; i<x; i++) idx [nn++] = xs + i;

    if ((xe+sDist)<=M) {
      for (i=0; i<sDist; i++) idx[nn++]=xe+i;
    } else {
      for (i=xe; i<M; i++) idx[nn++]=i;
    }
  }

  ierr = ISCreateBlock(comm,dof,nn-IXs+Xs,&idx[IXs-Xs],PETSC_USE_POINTER,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  dd->xs = dof*xs; dd->xe = dof*xe; dd->ys = 0; dd->ye = 1; dd->zs = 0; dd->ze = 1;
  dd->Xs = dof*Xs; dd->Xe = dof*Xe; dd->Ys = 0; dd->Ye = 1; dd->Zs = 0; dd->Ze = 1;

  dd->gtol      = gtol;
  dd->base      = dof*xs;
  da->ops->view = DMView_DA_1d;

  /*
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  for (i=0; i<Xe-IXe; i++) idx[nn++] = -1; /* pad with -1s if needed for ghosted case*/

  ierr = ISLocalToGlobalMappingCreate(comm,dof,nn,idx,PETSC_OWN_POINTER,&da->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)da->ltogmap);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*@C
   DMDACreate1d - Creates an object that will manage the communication of  one-dimensional
   regular array data that is distributed across some processors.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bx - type of ghost cells at the boundary the array should have, if any. Use
          DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, or DM_BOUNDARY_PERIODIC.
.  M - global dimension of the array (that is the number of grid points)
            from the command line with -da_grid_x <M>)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx - array containing number of nodes in the X direction on each processor,
        or NULL. If non-null, must be of length as the number of processes in the MPI_Comm.
        The sum of these entries must equal M

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Key:
+  -dm_view - Calls DMView() at the conclusion of DMDACreate1d()
.  -da_grid_x <nx> - number of grid points in x direction
.  -da_refine_x <rx> - refinement factor
-  -da_refine <n> - refine the DMDA n times before creating it

   Level: beginner

   Notes:
   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.

   You must call DMSetUp() after this call before using this DM.

   If you wish to use the options database to change values in the DMDA call DMSetFromOptions() after this call
   but before DMSetUp().

.seealso: DMDestroy(), DMView(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(), DMDASetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToLocalBegin(), DMLocalToLocalEnd(), DMDAGetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMLoad(), DMDAGetOwnershipRanges(),
          DMStagCreate1d()

@*/
PetscErrorCode  DMDACreate1d(MPI_Comm comm, DMBoundaryType bx, PetscInt M, PetscInt dof, PetscInt s, const PetscInt lx[], DM *da)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMSetDimension(*da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, 1, 1);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, size, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*da, bx, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
