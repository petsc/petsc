
/*
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95
*/

#include <petsc-private/daimpl.h>     /*I  "petscdmda.h"   I*/

const char *const DMDABoundaryTypes[] = {"BOUNDARY_NONE","BOUNDARY_GHOSTED","BOUNDARY_PERIODIC","DMDA_",0};

#undef __FUNCT__
#define __FUNCT__ "DMView_DA_1d"
PetscErrorCode DMView_DA_1d(DM da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii,isdraw,isbinary;
  DM_DA          *dd = (DM_DA*)da->data;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool      ismatlab;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_VTK && format != PETSC_VIEWER_ASCII_VTK_CELL) {
      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D m %D w %D s %D\n",rank,dd->M,dd->m,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D\n",info.xs,info.xs+info.xm);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = DMView_DA_VTK(da, viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw  draw;
    double     ymin = -1,ymax = 1,xmin = -1,xmax = dd->M,x;
    PetscInt   base;
    char       node[10];
    PetscBool  isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

    ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draws all node lines */
    if (!rank) {
      PetscInt xmin_tmp;
      ymin = 0.0; ymax = 0.3;

      /* ADIC doesn't like doubles in a for loop */
      for (xmin_tmp =0; xmin_tmp < dd->M; xmin_tmp++) {
         ierr = PetscDrawLine(draw,(double)xmin_tmp,ymin,(double)xmin_tmp,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }

      xmin = 0.0; xmax = dd->M - 1;
      ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = 0; ymax = 0.3; xmin = dd->xs / dd->w; xmax = (dd->xe / dd->w)  - 1;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

    /* Put in index numbers */
    base = dd->base / dd->w;
    for (x=xmin; x<=xmax; x++) {
      sprintf(node,"%d",(int)base++);
      ierr = PetscDrawString(draw,x,ymin,PETSC_DRAW_RED,node);CHKERRQ(ierr);
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  } else if (isbinary){
    ierr = DMView_DA_Binary(da,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = DMView_DA_Matlab(da,viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for DMDA 1d",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_DA_Private"
/*
    Processes command line options to determine if/how a DMDA
  is to be viewed. Called by DMDACreateXX()
*/
PetscErrorCode DMView_DA_Private(DM da)
{
  PetscErrorCode ierr;
  PetscBool      flg1 = PETSC_FALSE;
  PetscViewer    view;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DMDA viewing options","DMDA");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-da_view","Print information about the DMDA's distribution","DMView",PETSC_FALSE,&flg1,PETSC_NULL);CHKERRQ(ierr);
    if (flg1) {
      ierr = PetscViewerASCIIGetStdout(((PetscObject)da)->comm,&view);CHKERRQ(ierr);
      ierr = DMView(da,view);CHKERRQ(ierr);
    }
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsBool("-da_view_draw","Draw how the DMDA is distributed","DMView",PETSC_FALSE,&flg1,PETSC_NULL);CHKERRQ(ierr);
    if (flg1) {ierr = DMView(da,PETSC_VIEWER_DRAW_(((PetscObject)da)->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_DA_1D"
PetscErrorCode  DMSetUp_DA_1D(DM da)
{
  DM_DA            *dd = (DM_DA*)da->data;
  const PetscInt   M     = dd->M;
  const PetscInt   dof   = dd->w;
  const PetscInt   s     = dd->s;
  const PetscInt   o     = dd->overlap;
  const PetscInt   sDist = s*dof;  /* absolute stencil distance */
  const PetscInt   oDist = o*dof;
  const PetscInt   *lx    = dd->lx;
  DMDABoundaryType bx  = dd->bx;
  MPI_Comm         comm;
  Vec              local, global;
  VecScatter       ltog, gtol;
  IS               to, from;
  PetscBool        flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscMPIInt      rank, size;
  PetscInt         i,*idx,nn,left,xs,xe,x,Xs,Xe,start,end,m,IXs,IXe;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dof < 1) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dof);
  if (s < 0) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",s);

  dd->dim = 1;
  ierr = PetscMalloc(dof*sizeof(char*),&dd->fieldname);CHKERRQ(ierr);
  ierr = PetscMemzero(dd->fieldname,dof*sizeof(char*));CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  dd->m = size;
  m     = dd->m;

  if (s > 0) {
    /* if not communicating data then should be ok to have nothing on some processes */
    if (M < m)     SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"More processes than data points! %D %D",m,M);
    if ((M-1) < s) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array is too small for stencil! %D %D",M-1,s);
  }

  /*
     Determine locally owned region
     xs is the first local node number, x is the number of local nodes
  */
  if (!lx) {
    ierr = PetscMalloc(m*sizeof(PetscInt), &dd->lx);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(PETSC_NULL,"-da_partition_blockcomm",&flg1,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(PETSC_NULL,"-da_partition_nodes_at_end",&flg2,PETSC_NULL);CHKERRQ(ierr);
    if (flg1) {      /* Block Comm type Distribution */
      xs = rank*M/m;
      x  = (rank + 1)*M/m - xs;
    } else if (flg2) { /* The odd nodes are evenly distributed across last nodes */
      x = (M + rank)/m;
      if (M/m == x) { xs = rank*x; }
      else          { xs = rank*(x-1) + (M+rank)%(x*m); }
    } else { /* The odd nodes are evenly distributed across the first k nodes */
      /* Regular PETSc Distribution */
      x = M/m + ((M % m) > rank);
      if (rank >= (M % m)) {xs = (rank * (PetscInt)(M/m) + M % m);}
      else                 {xs = rank * (PetscInt)(M/m) + rank;}
    }
    ierr = MPI_Allgather(&xs,1,MPIU_INT,dd->lx,1,MPIU_INT,comm);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) dd->lx[i] = dd->lx[i+1] - dd->lx[i];
    dd->lx[m-1] = M - dd->lx[m-1];
  } else {
    x  = lx[rank];
    xs = 0;
    for (i=0; i<rank; i++) {
      xs += lx[i];
    }
    /* verify that data user provided is consistent */
    left = xs;
    for (i=rank; i<size; i++) {
      left += lx[i];
    }
    if (left != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M %D %D",left,M);
  }

  /*
   check if the scatter requires more than one process neighbor or wraps around
   the domain more than once
  */
  if ((x < s+o) & ((M > 1) | (bx == DMDA_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s+o);

  /* From now on x,xs,xe,Xs,Xe are the exact location in the array */
  x  *= dof;
  xs *= dof;
  xe  = xs + x;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs-sDist-oDist > 0) {
    Xs = xs - sDist - oDist;
    IXs = xs - sDist - oDist;
  } else {
    if (bx) {
      Xs = xs - sDist;
    } else {
      Xs = 0;
    }
    IXs = 0;
  }
  if (xe+sDist+oDist <= M*dof) {
    Xe = xe + sDist + oDist;
    IXe = xe + sDist + oDist;
  } else {
    if (bx) {
      Xe = xe + sDist;
    } else {
      Xe = M*dof;
    }
    IXe = M*dof;
  }

  if (bx == DMDA_BOUNDARY_PERIODIC) {
    Xs = xs - sDist - oDist;
    Xe = xe + sDist + oDist;
    IXs = xs - sDist - oDist;
    IXe = xe + sDist + oDist;
  }

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = x;
  ierr = VecCreateMPIWithArray(comm,dof,dd->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  dd->nlocal = (Xe-Xs);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dof,dd->nlocal,0,&local);CHKERRQ(ierr);

  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,start,1,&to);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,xs-Xs,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  ierr = ISCreateStride(comm,(IXe-IXs),IXs-Xs,1,&to);CHKERRQ(ierr);

  ierr = PetscMalloc((x+2*(sDist+oDist))*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(da,(x+2*(sDist+oDist))*sizeof(PetscInt));CHKERRQ(ierr);

  for (i=0; i<IXs-Xs; i++) {idx[i] = -1; } /* prepend with -1s if needed for ghosted case*/

  nn = IXs-Xs;
  if (bx == DMDA_BOUNDARY_PERIODIC) { /* Handle all cases with wrap first */
    for (i=0; i<sDist+oDist; i++) {  /* Left ghost points */
      if ((xs-sDist-oDist+i)>=0) { idx[nn++] = xs-sDist-oDist+i;}
      else                 { idx[nn++] = M*dof+(xs-sDist-oDist+i);}
    }

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}  /* Non-ghost points */

    for (i=0; i<sDist+oDist; i++) { /* Right ghost points */
      if ((xe+i)<M*dof) { idx [nn++] =  xe+i; }
      else              { idx [nn++] = (xe+i) - M*dof;}
    }
  } else {      /* Now do all cases with no wrapping */
    if (0 <= xs-sDist-oDist) {for (i=0; i<sDist+oDist; i++) {idx[nn++] = xs - sDist - oDist + i;}}
    else               {for (i=0; i<xs;    i++) {idx[nn++] = i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}

    if ((xe+sDist+oDist)<=M*dof) {for (i=0;  i<sDist+oDist;   i++) {idx[nn++]=xe+i;}}
    else                   {for (i=xe; i<(M*dof); i++) {idx[nn++]=i;}}
  }

  ierr = ISCreateGeneral(comm,nn-IXs+Xs,&idx[IXs-Xs],PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,to);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  dd->xs = xs; dd->xe = xe; dd->ys = 0; dd->ye = 1; dd->zs = 0; dd->ze = 1;
  dd->Xs = Xs; dd->Xe = Xe; dd->Ys = 0; dd->Ye = 1; dd->Zs = 0; dd->Ze = 1;

  dd->gtol      = gtol;
  dd->ltog      = ltog;
  dd->base      = xs;
  da->ops->view = DMView_DA_1d;

  /*
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  for (i=0; i<Xe-IXe; i++) {idx[nn++] = -1; } /* pad with -1s if needed for ghosted case*/

  ierr = ISLocalToGlobalMappingCreate(comm,nn,idx,PETSC_COPY_VALUES,&da->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(da->ltogmap,dd->w,&da->ltogmapb);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltogmap);CHKERRQ(ierr);

  dd->idx = idx;
  dd->Nl  = nn;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDACreate1d"
/*@C
   DMDACreate1d - Creates an object that will manage the communication of  one-dimensional
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  bx - type of ghost cells at the boundary the array should have, if any. Use
          DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, or DMDA_BOUNDARY_PERIODIC.
.  M - global dimension of the array (use -M to indicate that it may be set to a different value
            from the command line with -da_grid_x <M>)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx - array containing number of nodes in the X direction on each processor,
        or PETSC_NULL. If non-null, must be of length as the number of processes in the MPI_Comm.

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Key:
+  -da_view - Calls DMView() at the conclusion of DMDACreate1d()
.  -da_grid_x <nx> - number of grid points in x direction; can set if M < 0
.  -da_refine_x <rx> - refinement factor
-  -da_refine <n> - refine the DMDA n times before creating it, if M < 0

   Level: beginner

   Notes:
   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.

.keywords: distributed array, create, one-dimensional

.seealso: DMDestroy(), DMView(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(), DMDASetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMDALocalToLocalBegin(), DMDALocalToLocalEnd(), DMDAGetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMLoad(), DMDAGetOwnershipRanges()

@*/
PetscErrorCode  DMDACreate1d(MPI_Comm comm, DMDABoundaryType bx, PetscInt M, PetscInt dof, PetscInt s, const PetscInt lx[], DM *da)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMDASetDim(*da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, 1, 1);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, size, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*da, bx, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  /* This violates the behavior for other classes, but right now users expect negative dimensions to be handled this way */
  ierr = DMSetFromOptions(*da);CHKERRQ(ierr);
  ierr = DMSetUp(*da);CHKERRQ(ierr);
  ierr = DMView_DA_Private(*da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
