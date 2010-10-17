#define PETSCDM_DLL
/* 
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95    
*/

#include "private/daimpl.h"     /*I  "petscdm.h"   I*/

const char *DMDAPeriodicTypes[] = {"NONPERIODIC","XPERIODIC","YPERIODIC","XYPERIODIC",
                                 "XYZPERIODIC","XZPERIODIC","YZPERIODIC","ZPERIODIC","XYZGHOSTED","DMDAPeriodicType","DMDA_",0};

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

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format != PETSC_VIEWER_ASCII_VTK && format != PETSC_VIEWER_ASCII_VTK_CELL) {
      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D m %D w %D s %D\n",rank,dd->M,dd->m,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D\n",info.xs,info.xs+info.xm);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
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
    ierr = PetscOptionsTruth("-da_view","Print information about the DMDA's distribution","DMView",PETSC_FALSE,&flg1,PETSC_NULL);CHKERRQ(ierr);
    if (flg1) {
      ierr = PetscViewerASCIIGetStdout(((PetscObject)da)->comm,&view);CHKERRQ(ierr);
      ierr = DMView(da,view);CHKERRQ(ierr);
    }
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsTruth("-da_view_draw","Draw how the DMDA is distributed","DMView",PETSC_FALSE,&flg1,PETSC_NULL);CHKERRQ(ierr);
    if (flg1) {ierr = DMView(da,PETSC_VIEWER_DRAW_(((PetscObject)da)->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetUp_DA_1D"
PetscErrorCode PETSCDM_DLLEXPORT DMSetUp_DA_1D(DM da)
{
  DM_DA                *dd = (DM_DA*)da->data;
  const PetscInt       M     = dd->M;
  const PetscInt       dof   = dd->w;
  const PetscInt       s     = dd->s;
  const PetscInt       sDist = s*dof;  /* absolute stencil distance */
  const PetscInt      *lx    = dd->lx;
  const DMDAPeriodicType wrap  = dd->wrap;
  MPI_Comm             comm;
  Vec                  local, global;
  VecScatter           ltog, gtol;
  IS                   to, from;
  PetscBool            flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscMPIInt          rank, size;
  PetscInt             i,*idx,nn,left,xs,xe,x,Xs,Xe,start,end,m;
  PetscErrorCode       ierr;

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

  /* From now on x,xs,xe,Xs,Xe are the exact location in the array */
  x  *= dof;
  xs *= dof;
  xe  = xs + x;

  /* determine ghost region */
  if (wrap == DMDA_XPERIODIC || wrap == DMDA_XYZGHOSTED) {
    Xs = xs - sDist; 
    Xe = xe + sDist;
  } else {
    if ((xs-sDist) >= 0)     Xs = xs-sDist;  else Xs = 0; 
    if ((xe+sDist) <= M*dof) Xe = xe+sDist;  else Xe = M*dof;    
  }

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = x;
  ierr = VecCreateMPIWithArray(comm,dd->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  dd->nlocal = (Xe-Xs);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dd->nlocal,0,&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);
    
  /* Create Local to Global Vector Scatter Context */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,start,1,&to);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x,xs-Xs,1,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,ltog);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  if  (wrap == DMDA_XYZGHOSTED) {
    if (size == 1) {
      ierr = ISCreateStride(comm,(xe-xs),sDist,1,&to);CHKERRQ(ierr);
    } else if (!rank) {
      ierr = ISCreateStride(comm,(Xe-xs),sDist,1,&to);CHKERRQ(ierr);
    } else if (rank == size-1) {
      ierr = ISCreateStride(comm,(xe-Xs),0,1,&to);CHKERRQ(ierr);
    } else {
      ierr = ISCreateStride(comm,(Xe-Xs),0,1,&to);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateStride(comm,(Xe-Xs),0,1,&to);CHKERRQ(ierr);
  }
 
  ierr = PetscMalloc((x+2*sDist)*sizeof(PetscInt),&idx);CHKERRQ(ierr);  
  ierr = PetscLogObjectMemory(da,(x+2*sDist)*sizeof(PetscInt));CHKERRQ(ierr);

  nn = 0;
  if (wrap == DMDA_XPERIODIC) {    /* Handle all cases with wrap first */

    for (i=0; i<sDist; i++) {  /* Left ghost points */
      if ((xs-sDist+i)>=0) { idx[nn++] = xs-sDist+i;}
      else                 { idx[nn++] = M*dof+(xs-sDist+i);}
    }

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}  /* Non-ghost points */
    
    for (i=0; i<sDist; i++) { /* Right ghost points */
      if ((xe+i)<M*dof) { idx [nn++] =  xe+i; }
      else              { idx [nn++] = (xe+i) - M*dof;}
    }
  } else if (wrap == DMDA_XYZGHOSTED) { 

    if (sDist <= xs) {for (i=0; i<sDist; i++) {idx[nn++] = xs - sDist + i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}
    
    if ((xe+sDist)<=M*dof) {for (i=0;  i<sDist;     i++) {idx[nn++]=xe+i;}}

  } else {      /* Now do all cases with no wrapping */

    if (sDist <= xs) {for (i=0; i<sDist; i++) {idx[nn++] = xs - sDist + i;}}
    else             {for (i=0; i<xs;    i++) {idx[nn++] = i;}}

    for (i=0; i<x; i++) { idx [nn++] = xs + i;}
    
    if ((xe+sDist)<=M*dof) {for (i=0;  i<sDist;   i++) {idx[nn++]=xe+i;}}
    else                   {for (i=xe; i<(M*dof); i++) {idx[nn++]=i;}}
  }

  ierr = ISCreateGeneral(comm,nn,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,to);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,gtol);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);

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
  if (wrap == DMDA_XYZGHOSTED) {
    PetscInt *tmpidx;
    if (size == 1) {
      ierr = PetscMalloc((nn+2*sDist)*sizeof(PetscInt),&tmpidx);CHKERRQ(ierr);
      for (i=0; i<sDist; i++) tmpidx[i] = -1;
      ierr = PetscMemcpy(tmpidx+sDist,idx,nn*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=nn+sDist; i<nn+2*sDist; i++) tmpidx[i] = -1;
      ierr = PetscFree(idx);CHKERRQ(ierr);
      idx  = tmpidx;
      nn  += 2*sDist;
    } else if (!rank) { /* must preprend -1 marker for ghost location that have no global value */
      ierr = PetscMalloc((nn+sDist)*sizeof(PetscInt),&tmpidx);CHKERRQ(ierr);
      for (i=0; i<sDist; i++) tmpidx[i] = -1;
      ierr = PetscMemcpy(tmpidx+sDist,idx,nn*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscFree(idx);CHKERRQ(ierr);
      idx  = tmpidx;
      nn  += sDist;
    } else if (rank  == size-1) { /* must postpend -1 marker for ghost location that have no global value */
      ierr = PetscMalloc((nn+sDist)*sizeof(PetscInt),&tmpidx);CHKERRQ(ierr);
      ierr = PetscMemcpy(tmpidx,idx,nn*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=nn; i<nn+sDist; i++) tmpidx[i] = -1;
      ierr = PetscFree(idx);CHKERRQ(ierr);
      idx  = tmpidx;
      nn  += sDist;
    }
  }
  ierr = ISLocalToGlobalMappingCreate(comm,nn,idx,PETSC_OWN_POINTER,&dd->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(dd->ltogmap,dd->w,&dd->ltogmapb);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,dd->ltogmap);CHKERRQ(ierr);

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
.  wrap - type of periodicity should the array have, if any. Use 
          either DMDA_NONPERIODIC or DMDA_XPERIODIC
.  M - global dimension of the array (use -M to indicate that it may be set to a different value 
            from the command line with -da_grid_x <M>)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx - array containing number of nodes in the X direction on each processor, 
        or PETSC_NULL. If non-null, must be of length as m.

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Key:
+  -da_view - Calls DMView() at the conclusion of DMDACreate1d()
.  -da_grid_x <nx> - number of grid points in x direction; can set if M < 0
-  -da_refine_x - refinement factor 

   Level: beginner

   Notes:
   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.


.keywords: distributed array, create, one-dimensional

.seealso: DMDestroy(), DMView(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(), DMDASetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMDALocalToLocalBegin(), DMDALocalToLocalEnd(), DMDAGetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMDALoad(), DMDAGetOwnershipRanges()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMDACreate1d(MPI_Comm comm, DMDAPeriodicType wrap, PetscInt M, PetscInt dof, PetscInt s, const PetscInt lx[], DM *da)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMDASetDim(*da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, 1, 1);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, size, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetPeriodicity(*da, wrap);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  /* This violates the behavior for other classes, but right now users expect negative dimensions to be handled this way */
  ierr = DMSetFromOptions(*da);CHKERRQ(ierr);
  ierr = DMSetUp(*da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
