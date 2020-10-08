
#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/
#include <petscdraw.h>

static PetscErrorCode DMView_DA_2d(DM da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii,isdraw,isglvis,isbinary;
  DM_DA          *dd = (DM_DA*)da->data;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool ismatlab;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank);CHKERRMPI(ierr);

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
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)da),&size);CHKERRMPI(ierr);
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      nzlocal = info.xm*info.ym;
      ierr = PetscMalloc1(size,&nz);CHKERRQ(ierr);
      ierr = MPI_Allgather(&nzlocal,1,MPIU_INT,nz,1,MPIU_INT,PetscObjectComm((PetscObject)da));CHKERRMPI(ierr);
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D N %D m %D n %D w %D s %D\n",rank,dd->M,dd->N,dd->m,dd->n,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D, Y range of indices: %D %D\n",info.xs,info.xs+info.xm,info.ys,info.ys+info.ym);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
      ierr = DMView_DA_GLVis(da,viewer);CHKERRQ(ierr);
    } else {
      ierr = DMView_DA_VTK(da,viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw      draw;
    double         ymin = -1*dd->s-1,ymax = dd->N+dd->s;
    double         xmin = -1*dd->s-1,xmax = dd->M+dd->s;
    double         x,y;
    PetscInt       base;
    const PetscInt *idx;
    char           node[10];
    PetscBool      isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);

    ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = dd->N - 1;
      for (xmin=0; xmin<dd->M; xmin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = dd->M - 1;
      for (ymin=0; ymin<dd->N; ymin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* draw my box */
    xmin = dd->xs/dd->w; xmax =(dd->xe-1)/dd->w; ymin = dd->ys; ymax = dd->ye - 1;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    /* put in numbers */
    base = (dd->base)/dd->w;
    for (y=ymin; y<=ymax; y++) {
      for (x=xmin; x<=xmax; x++) {
        ierr = PetscSNPrintf(node,sizeof(node),"%d",(int)base++);CHKERRQ(ierr);
        ierr = PetscDrawString(draw,x,y,PETSC_DRAW_BLACK,node);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* overlay ghost numbers, useful for error checking */
    ierr = ISLocalToGlobalMappingGetBlockIndices(da->ltogmap,&idx);CHKERRQ(ierr);
    base = 0; xmin = dd->Xs; xmax = dd->Xe; ymin = dd->Ys; ymax = dd->Ye;
    for (y=ymin; y<ymax; y++) {
      for (x=xmin; x<xmax; x++) {
        if ((base % dd->w) == 0) {
          ierr = PetscSNPrintf(node,sizeof(node),"%d",(int)(idx[base/dd->w]));CHKERRQ(ierr);
          ierr = PetscDrawString(draw,x/dd->w,y,PETSC_DRAW_BLUE,node);CHKERRQ(ierr);
        }
        base++;
      }
    }
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(da->ltogmap,&idx);CHKERRQ(ierr);
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


#if defined(new)
/*
  DMDAGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix where local
    function lives on a DMDA

        y ~= (F(u + ha) - F(u))/h,
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode DMDAGetDiagonal_MFFD(DM da,Vec U,Vec a)
{
  PetscScalar    h,*aa,*ww,v;
  PetscReal      epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  PetscErrorCode ierr;
  PetscInt       gI,nI;
  MatStencil     stencil;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = (*ctx->func)(0,U,a,ctx->funcctx);CHKERRQ(ierr);
  ierr = (*ctx->funcisetbase)(U,ctx->funcctx);CHKERRQ(ierr);

  ierr = VecGetArray(U,&ww);CHKERRQ(ierr);
  ierr = VecGetArray(a,&aa);CHKERRQ(ierr);

  nI = 0;
  h  = ww[gI];
  if (h == 0.0) h = 1.0;
  if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0) h = umin;
  else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
  h *= epsilon;

  ww[gI] += h;
  ierr    = (*ctx->funci)(i,w,&v,ctx->funcctx);CHKERRQ(ierr);
  aa[nI]  = (v - aa[nI])/h;
  ww[gI] -= h;
  nI++;

  ierr = VecRestoreArray(U,&ww);CHKERRQ(ierr);
  ierr = VecRestoreArray(a,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode  DMSetUp_DA_2D(DM da)
{
  DM_DA            *dd = (DM_DA*)da->data;
  const PetscInt   M            = dd->M;
  const PetscInt   N            = dd->N;
  PetscInt         m            = dd->m;
  PetscInt         n            = dd->n;
  const PetscInt   dof          = dd->w;
  const PetscInt   s            = dd->s;
  DMBoundaryType   bx           = dd->bx;
  DMBoundaryType   by           = dd->by;
  DMDAStencilType  stencil_type = dd->stencil_type;
  PetscInt         *lx          = dd->lx;
  PetscInt         *ly          = dd->ly;
  MPI_Comm         comm;
  PetscMPIInt      rank,size;
  PetscInt         xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,IXs,IXe,IYs,IYe;
  PetscInt         up,down,left,right,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  PetscInt         xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count;
  PetscInt         s_x,s_y; /* s proportionalized to w */
  PetscInt         sn0 = 0,sn2 = 0,sn6 = 0,sn8 = 0;
  Vec              local,global;
  VecScatter       gtol;
  IS               to,from;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (stencil_type == DMDA_STENCIL_BOX && (bx == DM_BOUNDARY_MIRROR || by == DM_BOUNDARY_MIRROR)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Mirror boundary and box stencil");
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (((PetscInt64) M)*((PetscInt64) N)*((PetscInt64) dof) > (PetscInt64) PETSC_MPI_INT_MAX) SETERRQ3(comm,PETSC_ERR_INT_OVERFLOW,"Mesh of %D by %D by %D (dof) is too large for 32 bit indices",M,N,dof);
#endif

  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  dd->p = 1;
  if (m != PETSC_DECIDE) {
    if (m < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %D",m);
    else if (m > size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %D %d",m,size);
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %D",n);
    else if (n > size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %D %d",n,size);
  }

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    if (n != PETSC_DECIDE) {
      m = size/n;
    } else if (m != PETSC_DECIDE) {
      n = size/m;
    } else {
      /* try for squarish distribution */
      m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)N)));
      if (!m) m = 1;
      while (m > 0) {
        n = size/m;
        if (m*n == size) break;
        m--;
      }
      if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}
    }
    if (m*n != size) SETERRQ(comm,PETSC_ERR_PLIB,"Unable to create partition, check the size of the communicator and input m and n ");
  } else if (m*n != size) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition");

  if (M < m) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);

  /*
     Determine locally owned region
     xs is the first local node number, x is the number of local nodes
  */
  if (!lx) {
    ierr = PetscMalloc1(m, &dd->lx);CHKERRQ(ierr);
    lx   = dd->lx;
    for (i=0; i<m; i++) {
      lx[i] = M/m + ((M % m) > i);
    }
  }
  x  = lx[rank % m];
  xs = 0;
  for (i=0; i<(rank % m); i++) {
    xs += lx[i];
  }
  if (PetscDefined(USE_DEBUG)) {
    left = xs;
    for (i=(rank % m); i<m; i++) {
      left += lx[i];
    }
    if (left != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M: %D %D",left,M);
  }

  /*
     Determine locally owned region
     ys is the first local node number, y is the number of local nodes
  */
  if (!ly) {
    ierr = PetscMalloc1(n, &dd->ly);CHKERRQ(ierr);
    ly   = dd->ly;
    for (i=0; i<n; i++) {
      ly[i] = N/n + ((N % n) > i);
    }
  }
  y  = ly[rank/m];
  ys = 0;
  for (i=0; i<(rank/m); i++) {
    ys += ly[i];
  }
  if (PetscDefined(USE_DEBUG)) {
    left = ys;
    for (i=(rank/m); i<n; i++) {
      left += ly[i];
    }
    if (left != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Sum of ly across processors not equal to N: %D %D",left,N);
  }

  /*
   check if the scatter requires more than one process neighbor or wraps around
   the domain more than once
  */
  if ((x < s) && ((m > 1) || (bx == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s);
  if ((y < s) && ((n > 1) || (by == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local y-width of domain y %D is smaller than stencil width s %D",y,s);
  xe = xs + x;
  ye = ys + y;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs-s > 0) {
    Xs = xs - s; IXs = xs - s;
  } else {
    if (bx) {
      Xs = xs - s;
    } else {
      Xs = 0;
    }
    IXs = 0;
  }
  if (xe+s <= M) {
    Xe = xe + s; IXe = xe + s;
  } else {
    if (bx) {
      Xs = xs - s; Xe = xe + s;
    } else {
      Xe = M;
    }
    IXe = M;
  }

  if (bx == DM_BOUNDARY_PERIODIC || bx == DM_BOUNDARY_MIRROR) {
    IXs = xs - s;
    IXe = xe + s;
    Xs  = xs - s;
    Xe  = xe + s;
  }

  if (ys-s > 0) {
    Ys = ys - s; IYs = ys - s;
  } else {
    if (by) {
      Ys = ys - s;
    } else {
      Ys = 0;
    }
    IYs = 0;
  }
  if (ye+s <= N) {
    Ye = ye + s; IYe = ye + s;
  } else {
    if (by) {
      Ye = ye + s;
    } else {
      Ye = N;
    }
    IYe = N;
  }

  if (by == DM_BOUNDARY_PERIODIC || by == DM_BOUNDARY_MIRROR) {
    IYs = ys - s;
    IYe = ye + s;
    Ys  = ys - s;
    Ye  = ye + s;
  }

  /* stencil length in each direction */
  s_x = s;
  s_y = s;

  /* determine starting point of each processor */
  nn       = x*y;
  ierr     = PetscMalloc2(size+1,&bases,size,&ldims);CHKERRQ(ierr);
  ierr     = MPI_Allgather(&nn,1,MPIU_INT,ldims,1,MPIU_INT,comm);CHKERRQ(ierr);
  bases[0] = 0;
  for (i=1; i<=size; i++) {
    bases[i] = ldims[i-1];
  }
  for (i=1; i<=size; i++) {
    bases[i] += bases[i-1];
  }
  base = bases[rank]*dof;

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = x*y*dof;
  ierr       = VecCreateMPIWithArray(comm,dof,dd->Nlocal,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
  dd->nlocal = (Xe-Xs)*(Ye-Ys)*dof;
  ierr       = VecCreateSeqWithArray(PETSC_COMM_SELF,dof,dd->nlocal,NULL,&local);CHKERRQ(ierr);

  /* generate global to local vector scatter and local to global mapping*/

  /* global to local must include ghost points within the domain,
     but not ghost points outside the domain that aren't periodic */
  ierr  = PetscMalloc1((IXe-IXs)*(IYe-IYs),&idx);CHKERRQ(ierr);
  if (stencil_type == DMDA_STENCIL_BOX) {
    left  = IXs - Xs; right = left + (IXe-IXs);
    down  = IYs - Ys; up = down + (IYe-IYs);
    count = 0;
    for (i=down; i<up; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);

  } else {
    /* must drop into cross shape region */
    /*       ---------|
            |  top    |
         |---         ---| up
         |   middle      |
         |               |
         ----         ---- down
            | bottom  |
            -----------
         Xs xs        xe Xe */
    left  = xs - Xs; right = left + x;
    down  = ys - Ys; up = down + y;
    count = 0;
    /* bottom */
    for (i=(IYs-Ys); i<down; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    /* middle */
    for (i=down; i<up; i++) {
      for (j=(IXs-Xs); j<(IXe-Xs); j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    /* top */
    for (i=up; i<up+IYe-ye; i++) {
      for (j=left; j<right; j++) {
        idx[count++] = j + i*(Xe-Xs);
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);
  }


  /* determine who lies on each side of us stored in    n6 n7 n8
                                                        n3    n5
                                                        n0 n1 n2
  */

  /* Assume the Non-Periodic Case */
  n1 = rank - m;
  if (rank % m) {
    n0 = n1 - 1;
  } else {
    n0 = -1;
  }
  if ((rank+1) % m) {
    n2 = n1 + 1;
    n5 = rank + 1;
    n8 = rank + m + 1; if (n8 >= m*n) n8 = -1;
  } else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (rank % m) {
    n3 = rank - 1;
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  } else {
    n3 = -1; n6 = -1;
  }
  n7 = rank + m; if (n7 >= m*n) n7 = -1;

  if (bx == DM_BOUNDARY_PERIODIC && by == DM_BOUNDARY_PERIODIC) {
    /* Modify for Periodic Cases */
    /* Handle all four corners */
    if ((n6 < 0) && (n7 < 0) && (n3 < 0)) n6 = m-1;
    if ((n8 < 0) && (n7 < 0) && (n5 < 0)) n8 = 0;
    if ((n2 < 0) && (n5 < 0) && (n1 < 0)) n2 = size-m;
    if ((n0 < 0) && (n3 < 0) && (n1 < 0)) n0 = size-1;

    /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;

    /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  } else if (by == DM_BOUNDARY_PERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;
  } else if (bx == DM_BOUNDARY_PERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }

  ierr = PetscMalloc1(9,&dd->neighbors);CHKERRQ(ierr);

  dd->neighbors[0] = n0;
  dd->neighbors[1] = n1;
  dd->neighbors[2] = n2;
  dd->neighbors[3] = n3;
  dd->neighbors[4] = rank;
  dd->neighbors[5] = n5;
  dd->neighbors[6] = n6;
  dd->neighbors[7] = n7;
  dd->neighbors[8] = n8;

  if (stencil_type == DMDA_STENCIL_STAR) {
    /* save corner processor numbers */
    sn0 = n0; sn2 = n2; sn6 = n6; sn8 = n8;
    n0  = n2 = n6 = n8 = -1;
  }

  ierr = PetscMalloc1((Xe-Xs)*(Ye-Ys),&idx);CHKERRQ(ierr);

  nn = 0;
  xbase = bases[rank];
  for (i=1; i<=s_y; i++) {
    if (n0 >= 0) { /* left below */
      x_t = lx[n0 % m];
      y_t = ly[(n0/m)];
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    }

    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = ly[(n1/m)];
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<x_t; j++) idx[nn++] = s_t++;
    } else if (by == DM_BOUNDARY_MIRROR) {
      for (j=0; j<x; j++) idx[nn++] = bases[rank] + x*(s_y - i + 1)  + j;
    }

    if (n2 >= 0) { /* right below */
      x_t = lx[n2 % m];
      y_t = ly[(n2/m)];
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    }
  }

  for (i=0; i<y; i++) {
    if (n3 >= 0) { /* directly left */
      x_t = lx[n3 % m];
      /* y_t = y; */
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    } else if (bx == DM_BOUNDARY_MIRROR) {
      for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x*i + s_x - j;
    }

    for (j=0; j<x; j++) idx[nn++] = xbase++; /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m];
      /* y_t = y; */
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    } else if (bx == DM_BOUNDARY_MIRROR) {
      for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x*(i + 1) - 2 - j;
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m];
      /* y_t = ly[(n6/m)]; */
      s_t = bases[n6] + (i)*x_t - s_x;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    }

    if (n7 >= 0) { /* directly above */
      x_t = x;
      /* y_t = ly[(n7/m)]; */
      s_t = bases[n7] + (i-1)*x_t;
      for (j=0; j<x_t; j++) idx[nn++] = s_t++;
    } else if (by == DM_BOUNDARY_MIRROR) {
      for (j=0; j<x; j++) idx[nn++] = bases[rank] + x*(y - i - 1)  + j;
    }

    if (n8 >= 0) { /* right above */
      x_t = lx[n8 % m];
      /* y_t = ly[(n8/m)]; */
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) idx[nn++] = s_t++;
    }
  }

  ierr = ISCreateBlock(comm,dof,nn,idx,PETSC_USE_POINTER,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  if (stencil_type == DMDA_STENCIL_STAR) {
    n0 = sn0; n2 = sn2; n6 = sn6; n8 = sn8;
  }

  if (((stencil_type == DMDA_STENCIL_STAR)  || (bx && bx != DM_BOUNDARY_PERIODIC) || (by && by != DM_BOUNDARY_PERIODIC))) {
    /*
        Recompute the local to global mappings, this time keeping the
      information about the cross corner processor numbers and any ghosted
      but not periodic indices.
    */
    nn    = 0;
    xbase = bases[rank];
    for (i=1; i<=s_y; i++) {
      if (n0 >= 0) { /* left below */
        x_t = lx[n0 % m];
        y_t = ly[(n0/m)];
        s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (xs-Xs > 0 && ys-Ys > 0) {
        for (j=0; j<s_x; j++) idx[nn++] = -1;
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n1/m)];
        s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      } else if (ys-Ys > 0) {
        if (by == DM_BOUNDARY_MIRROR) {
          for (j=0; j<x; j++) idx[nn++] = bases[rank] + x*(s_y - i + 1)  + j;
        } else {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
      }
      if (n2 >= 0) { /* right below */
        x_t = lx[n2 % m];
        y_t = ly[(n2/m)];
        s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (Xe-xe> 0 && ys-Ys > 0) {
        for (j=0; j<s_x; j++) idx[nn++] = -1;
      }
    }

    for (i=0; i<y; i++) {
      if (n3 >= 0) { /* directly left */
        x_t = lx[n3 % m];
        /* y_t = y; */
        s_t = bases[n3] + (i+1)*x_t - s_x;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (xs-Xs > 0) {
        if (bx == DM_BOUNDARY_MIRROR) {
          for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x*i + s_x - j;
        } else {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (j=0; j<x; j++) idx[nn++] = xbase++; /* interior */

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m];
        /* y_t = y; */
        s_t = bases[n5] + (i)*x_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (Xe-xe > 0) {
        if (bx == DM_BOUNDARY_MIRROR) {
          for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x*(i + 1) - 2 - j;
        } else {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m];
        /* y_t = ly[(n6/m)]; */
        s_t = bases[n6] + (i)*x_t - s_x;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (xs-Xs > 0 && Ye-ye > 0) {
        for (j=0; j<s_x; j++) idx[nn++] = -1;
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        /* y_t = ly[(n7/m)]; */
        s_t = bases[n7] + (i-1)*x_t;
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      } else if (Ye-ye > 0) {
        if (by == DM_BOUNDARY_MIRROR) {
          for (j=0; j<x; j++) idx[nn++] = bases[rank] + x*(y - i - 1)  + j;
        } else {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
      }
      if (n8 >= 0) { /* right above */
        x_t = lx[n8 % m];
        /* y_t = ly[(n8/m)]; */
        s_t = bases[n8] + (i-1)*x_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (Xe-xe > 0 && Ye-ye > 0) {
        for (j=0; j<s_x; j++) idx[nn++] = -1;
      }
    }
  }
  /*
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  ierr = ISLocalToGlobalMappingCreate(comm,dof,nn,idx,PETSC_OWN_POINTER,&da->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)da->ltogmap);CHKERRQ(ierr);

  ierr  = PetscFree2(bases,ldims);CHKERRQ(ierr);
  dd->m = m;  dd->n  = n;
  /* note petsc expects xs/xe/Xs/Xe to be multiplied by #dofs in many places */
  dd->xs = xs*dof; dd->xe = xe*dof; dd->ys = ys; dd->ye = ye; dd->zs = 0; dd->ze = 1;
  dd->Xs = Xs*dof; dd->Xe = Xe*dof; dd->Ys = Ys; dd->Ye = Ye; dd->Zs = 0; dd->Ze = 1;

  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  dd->gtol      = gtol;
  dd->base      = base;
  da->ops->view = DMView_DA_2d;
  dd->ltol      = NULL;
  dd->ao        = NULL;
  PetscFunctionReturn(0);
}

/*@C
   DMDACreate2d -  Creates an object that will manage the communication of  two-dimensional
   regular array data that is distributed across some processors.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bx,by - type of ghost nodes the array have.
         Use one of DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_PERIODIC.
.  stencil_type - stencil type.  Use either DMDA_STENCIL_BOX or DMDA_STENCIL_STAR.
.  M,N - global dimension in each direction of the array
.  m,n - corresponding number of processors in each dimension
         (or PETSC_DECIDE to have calculated)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx, ly - arrays containing the number of nodes in each cell along
           the x and y coordinates, or NULL. If non-null, these
           must be of length as m and n, and the corresponding
           m and n cannot be PETSC_DECIDE. The sum of the lx[] entries
           must be M, and the sum of the ly[] entries must be N.

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Key:
+  -dm_view - Calls DMView() at the conclusion of DMDACreate2d()
.  -da_grid_x <nx> - number of grid points in x direction
.  -da_grid_y <ny> - number of grid points in y direction
.  -da_processors_x <nx> - number of processors in x direction
.  -da_processors_y <ny> - number of processors in y direction
.  -da_refine_x <rx> - refinement ratio in x direction
.  -da_refine_y <ry> - refinement ratio in y direction
-  -da_refine <n> - refine the DMDA n times before creating


   Level: beginner

   Notes:
   The stencil type DMDA_STENCIL_STAR with width 1 corresponds to the
   standard 5-pt stencil, while DMDA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.

   You must call DMSetUp() after this call before using this DM.

   If you wish to use the options database to change values in the DMDA call DMSetFromOptions() after this call
   but before DMSetUp().

.seealso: DMDestroy(), DMView(), DMDACreate1d(), DMDACreate3d(), DMGlobalToLocalBegin(), DMDAGetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToLocalBegin(), DMLocalToLocalEnd(), DMDASetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMLoad(), DMDAGetOwnershipRanges(),
          DMStagCreate2d()

@*/

PetscErrorCode  DMDACreate2d(MPI_Comm comm,DMBoundaryType bx,DMBoundaryType by,DMDAStencilType stencil_type,
                             PetscInt M,PetscInt N,PetscInt m,PetscInt n,PetscInt dof,PetscInt s,const PetscInt lx[],const PetscInt ly[],DM *da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMSetDimension(*da, 2);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, N, 1);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, m, n, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*da, bx, by, DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*da, stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, ly, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
