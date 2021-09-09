
/*
   Code for manipulating distributed regular 3d arrays in parallel.
   File created by Peter Mell  7/14/95
 */

#include <petsc/private/dmdaimpl.h>     /*I   "petscdmda.h"    I*/

#include <petscdraw.h>
static PetscErrorCode DMView_DA_3d(DM da,PetscViewer viewer)
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

    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_LOAD_BALANCE) {
      PetscInt      i,nmax = 0,nmin = PETSC_MAX_INT,navg = 0,*nz,nzlocal;
      DMDALocalInfo info;
      PetscMPIInt   size;
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)da),&size);CHKERRMPI(ierr);
      ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
      nzlocal = info.xm*info.ym*info.zm;
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %D N %D P %D m %D n %D p %D w %D s %D\n",rank,dd->M,dd->N,dd->P,dd->m,dd->n,dd->p,dd->w,dd->s);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %D %D, Y range of indices: %D %D, Z range of indices: %D %D\n",
                                                info.xs,info.xs+info.xm,info.ys,info.ys+info.ym,info.zs,info.zs+info.zm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      if (da->coordinates) {
        PetscInt        last;
        const PetscReal *coors;
        ierr = VecGetArrayRead(da->coordinates,&coors);CHKERRQ(ierr);
        ierr = VecGetLocalSize(da->coordinates,&last);CHKERRQ(ierr);
        last = last - 3;
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Lower left corner %g %g %g : Upper right %g %g %g\n",(double)coors[0],(double)coors[1],(double)coors[2],(double)coors[last],(double)coors[last+1],(double)coors[last+2]);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(da->coordinates,&coors);CHKERRQ(ierr);
      }
#endif
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
      ierr = DMView_DA_GLVis(da,viewer);CHKERRQ(ierr);
    } else {
      ierr = DMView_DA_VTK(da,viewer);CHKERRQ(ierr);
    }
  } else if (isdraw) {
    PetscDraw      draw;
    PetscReal      ymin = -1.0,ymax = (PetscReal)dd->N;
    PetscReal      xmin = -1.0,xmax = (PetscReal)((dd->M+2)*dd->P),x,y,ycoord,xcoord;
    PetscInt       k,plane,base;
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
      for (k=0; k<dd->P; k++) {
        ymin = 0.0; ymax = (PetscReal)(dd->N - 1);
        for (xmin=(PetscReal)(k*(dd->M+1)); xmin<(PetscReal)(dd->M+(k*(dd->M+1))); xmin++) {
          ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
        }
        xmin = (PetscReal)(k*(dd->M+1)); xmax = xmin + (PetscReal)(dd->M - 1);
        for (ymin=0; ymin<(PetscReal)dd->N; ymin++) {
          ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /*Go through and draw for each plane*/
    for (k=0; k<dd->P; k++) {
      if ((k >= dd->zs) && (k < dd->ze)) {
        /* draw my box */
        ymin = dd->ys;
        ymax = dd->ye - 1;
        xmin = dd->xs/dd->w    + (dd->M+1)*k;
        xmax =(dd->xe-1)/dd->w + (dd->M+1)*k;

        ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

        xmin = dd->xs/dd->w;
        xmax =(dd->xe-1)/dd->w;

        /* identify which processor owns the box */
        ierr = PetscSNPrintf(node,sizeof(node),"%d",(int)rank);CHKERRQ(ierr);
        ierr = PetscDrawString(draw,xmin+(dd->M+1)*k+.2,ymin+.3,PETSC_DRAW_RED,node);CHKERRQ(ierr);
        /* put in numbers*/
        base = (dd->base+(dd->xe-dd->xs)*(dd->ye-dd->ys)*(k-dd->zs))/dd->w;
        for (y=ymin; y<=ymax; y++) {
          for (x=xmin+(dd->M+1)*k; x<=xmax+(dd->M+1)*k; x++) {
            ierr = PetscSNPrintf(node,sizeof(node),"%d",(int)base++);CHKERRQ(ierr);
            ierr = PetscDrawString(draw,x,y,PETSC_DRAW_BLACK,node);CHKERRQ(ierr);
          }
        }

      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (k=0-dd->s; k<dd->P+dd->s; k++) {
      /* Go through and draw for each plane */
      if ((k >= dd->Zs) && (k < dd->Ze)) {
        /* overlay ghost numbers, useful for error checking */
        base = (dd->Xe-dd->Xs)*(dd->Ye-dd->Ys)*(k-dd->Zs)/dd->w;
        ierr = ISLocalToGlobalMappingGetBlockIndices(da->ltogmap,&idx);CHKERRQ(ierr);
        plane=k;
        /* Keep z wrap around points on the drawing */
        if (k<0) plane=dd->P+k;
        if (k>=dd->P) plane=k-dd->P;
        ymin = dd->Ys; ymax = dd->Ye;
        xmin = (dd->M+1)*plane*dd->w;
        xmax = (dd->M+1)*plane*dd->w+dd->M*dd->w;
        for (y=ymin; y<ymax; y++) {
          for (x=xmin+dd->Xs; x<xmin+dd->Xe; x+=dd->w) {
            sprintf(node,"%d",(int)(idx[base]));
            ycoord = y;
            /*Keep y wrap around points on drawing */
            if (y<0) ycoord = dd->N+y;
            if (y>=dd->N) ycoord = y-dd->N;
            xcoord = x;   /* Keep x wrap points on drawing */
            if (x<xmin) xcoord = xmax - (xmin-x);
            if (x>=xmax) xcoord = xmin + (x-xmax);
            ierr = PetscDrawString(draw,xcoord/dd->w,ycoord,PETSC_DRAW_BLUE,node);CHKERRQ(ierr);
            base++;
          }
        }
        ierr = ISLocalToGlobalMappingRestoreBlockIndices(da->ltogmap,&idx);CHKERRQ(ierr);
      }
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

PetscErrorCode  DMSetUp_DA_3D(DM da)
{
  DM_DA            *dd          = (DM_DA*)da->data;
  const PetscInt   M            = dd->M;
  const PetscInt   N            = dd->N;
  const PetscInt   P            = dd->P;
  PetscInt         m            = dd->m;
  PetscInt         n            = dd->n;
  PetscInt         p            = dd->p;
  const PetscInt   dof          = dd->w;
  const PetscInt   s            = dd->s;
  DMBoundaryType   bx           = dd->bx;
  DMBoundaryType   by           = dd->by;
  DMBoundaryType   bz           = dd->bz;
  DMDAStencilType  stencil_type = dd->stencil_type;
  PetscInt         *lx          = dd->lx;
  PetscInt         *ly          = dd->ly;
  PetscInt         *lz          = dd->lz;
  MPI_Comm         comm;
  PetscMPIInt      rank,size;
  PetscInt         xs = 0,xe,ys = 0,ye,zs = 0,ze,x = 0,y = 0,z = 0;
  PetscInt         Xs,Xe,Ys,Ye,Zs,Ze,IXs,IXe,IYs,IYe,IZs,IZe,pm;
  PetscInt         left,right,up,down,bottom,top,i,j,k,*idx,nn;
  PetscInt         n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n14;
  PetscInt         n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26;
  PetscInt         *bases,*ldims,base,x_t,y_t,z_t,s_t,count,s_x,s_y,s_z;
  PetscInt         sn0  = 0,sn1 = 0,sn2 = 0,sn3 = 0,sn5 = 0,sn6 = 0,sn7 = 0;
  PetscInt         sn8  = 0,sn9 = 0,sn11 = 0,sn15 = 0,sn24 = 0,sn25 = 0,sn26 = 0;
  PetscInt         sn17 = 0,sn18 = 0,sn19 = 0,sn20 = 0,sn21 = 0,sn23 = 0;
  Vec              local,global;
  VecScatter       gtol;
  IS               to,from;
  PetscBool        twod;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (stencil_type == DMDA_STENCIL_BOX && (bx == DM_BOUNDARY_MIRROR || by == DM_BOUNDARY_MIRROR || bz == DM_BOUNDARY_MIRROR)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Mirror boundary and box stencil");
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (((PetscInt64) M)*((PetscInt64) N)*((PetscInt64) P)*((PetscInt64) dof) > (PetscInt64) PETSC_MPI_INT_MAX) SETERRQ4(comm,PETSC_ERR_INT_OVERFLOW,"Mesh of %D by %D by %D by %D (dof) is too large for 32 bit indices",M,N,P,dof);
#endif

  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  if (m != PETSC_DECIDE) {
    if (m < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %D",m);
    else if (m > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %D %d",m,size);
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %D",n);
    else if (n > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %D %d",n,size);
  }
  if (p != PETSC_DECIDE) {
    if (p < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Z direction: %D",p);
    else if (p > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Z direction: %D %d",p,size);
  }
  if ((m > 0) && (n > 0) && (p > 0) && (m*n*p != size)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m %D * n %D * p %D != size %d",m,n,p,size);

  /* Partition the array among the processors */
  if (m == PETSC_DECIDE && n != PETSC_DECIDE && p != PETSC_DECIDE) {
    m = size/(n*p);
  } else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    n = size/(m*p);
  } else if (m != PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    p = size/(m*n);
  } else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)N*p)));
    if (!m) m = 1;
    while (m > 0) {
      n = size/(m*p);
      if (m*n*p == size) break;
      m--;
    }
    if (!m) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad p value: p = %D",p);
    if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}
  } else if (m == PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)P*n)));
    if (!m) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (!m) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad n value: n = %D",n);
    if (M > P && m < p) {PetscInt _m = m; m = p; p = _m;}
  } else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (int)(0.5 + PetscSqrtReal(((PetscReal)N)*((PetscReal)size)/((PetscReal)P*m)));
    if (!n) n = 1;
    while (n > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      n--;
    }
    if (!n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad m value: m = %D",n);
    if (N > P && n < p) {PetscInt _n = n; n = p; p = _n;}
  } else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (PetscInt)(0.5 + PetscPowReal(((PetscReal)N*N)*((PetscReal)size)/((PetscReal)P*M),(PetscReal)(1./3.)));
    if (!n) n = 1;
    while (n > 0) {
      pm = size/n;
      if (n*pm == size) break;
      n--;
    }
    if (!n) n = 1;
    m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)P*n)));
    if (!m) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (M > P && m < p) {PetscInt _m = m; m = p; p = _m;}
  } else if (m*n*p != size) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition");

  if (m*n*p != size) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_PLIB,"Could not find good partition");
  if (M < m) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);
  if (P < p) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Partition in z direction is too fine! %D %D",P,p);

  /*
     Determine locally owned region
     [x, y, or z]s is the first local node number, [x, y, z] is the number of local nodes
  */

  if (!lx) {
    ierr = PetscMalloc1(m, &dd->lx);CHKERRQ(ierr);
    lx   = dd->lx;
    for (i=0; i<m; i++) lx[i] = M/m + ((M % m) > (i % m));
  }
  x  = lx[rank % m];
  xs = 0;
  for (i=0; i<(rank%m); i++) xs += lx[i];
  if ((x < s) && ((m > 1) || (bx == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local x-width of domain x %D is smaller than stencil width s %D",x,s);

  if (!ly) {
    ierr = PetscMalloc1(n, &dd->ly);CHKERRQ(ierr);
    ly   = dd->ly;
    for (i=0; i<n; i++) ly[i] = N/n + ((N % n) > (i % n));
  }
  y = ly[(rank % (m*n))/m];
  if ((y < s) && ((n > 1) || (by == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local y-width of domain y %D is smaller than stencil width s %D",y,s);

  ys = 0;
  for (i=0; i<(rank % (m*n))/m; i++) ys += ly[i];

  if (!lz) {
    ierr = PetscMalloc1(p, &dd->lz);CHKERRQ(ierr);
    lz = dd->lz;
    for (i=0; i<p; i++) lz[i] = P/p + ((P % p) > (i % p));
  }
  z = lz[rank/(m*n)];

  /* note this is different than x- and y-, as we will handle as an important special
   case when p=P=1 and DM_BOUNDARY_PERIODIC and s > z.  This is to deal with 2D problems
   in a 3D code.  Additional code for this case is noted with "2d case" comments */
  twod = PETSC_FALSE;
  if (P == 1) twod = PETSC_TRUE;
  else if ((z < s) && ((p > 1) || (bz == DM_BOUNDARY_PERIODIC))) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local z-width of domain z %D is smaller than stencil width s %D",z,s);
  zs = 0;
  for (i=0; i<(rank/(m*n)); i++) zs += lz[i];
  ye = ys + y;
  xe = xs + x;
  ze = zs + z;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs-s > 0) {
    Xs = xs - s; IXs = xs - s;
  } else {
    if (bx) Xs = xs - s;
    else Xs = 0;
    IXs = 0;
  }
  if (xe+s <= M) {
    Xe = xe + s; IXe = xe + s;
  } else {
    if (bx) {
      Xs = xs - s; Xe = xe + s;
    } else Xe = M;
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
    if (by) Ys = ys - s;
    else Ys = 0;
    IYs = 0;
  }
  if (ye+s <= N) {
    Ye = ye + s; IYe = ye + s;
  } else {
    if (by) Ye = ye + s;
    else Ye = N;
    IYe = N;
  }

  if (by == DM_BOUNDARY_PERIODIC || by == DM_BOUNDARY_MIRROR) {
    IYs = ys - s;
    IYe = ye + s;
    Ys  = ys - s;
    Ye  = ye + s;
  }

  if (zs-s > 0) {
    Zs = zs - s; IZs = zs - s;
  } else {
    if (bz) Zs = zs - s;
    else Zs = 0;
    IZs = 0;
  }
  if (ze+s <= P) {
    Ze = ze + s; IZe = ze + s;
  } else {
    if (bz) Ze = ze + s;
    else Ze = P;
    IZe = P;
  }

  if (bz == DM_BOUNDARY_PERIODIC || bz == DM_BOUNDARY_MIRROR) {
    IZs = zs - s;
    IZe = ze + s;
    Zs  = zs - s;
    Ze  = ze + s;
  }

  /* Resize all X parameters to reflect w */
  s_x = s;
  s_y = s;
  s_z = s;

  /* determine starting point of each processor */
  nn       = x*y*z;
  ierr     = PetscMalloc2(size+1,&bases,size,&ldims);CHKERRQ(ierr);
  ierr     = MPI_Allgather(&nn,1,MPIU_INT,ldims,1,MPIU_INT,comm);CHKERRMPI(ierr);
  bases[0] = 0;
  for (i=1; i<=size; i++) bases[i] = ldims[i-1];
  for (i=1; i<=size; i++) bases[i] += bases[i-1];
  base = bases[rank]*dof;

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = x*y*z*dof;
  ierr       = VecCreateMPIWithArray(comm,dof,dd->Nlocal,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
  dd->nlocal = (Xe-Xs)*(Ye-Ys)*(Ze-Zs)*dof;
  ierr       = VecCreateSeqWithArray(PETSC_COMM_SELF,dof,dd->nlocal,NULL,&local);CHKERRQ(ierr);

  /* generate global to local vector scatter and local to global mapping*/

  /* global to local must include ghost points within the domain,
     but not ghost points outside the domain that aren't periodic */
  ierr   = PetscMalloc1((IXe-IXs)*(IYe-IYs)*(IZe-IZs),&idx);CHKERRQ(ierr);
  if (stencil_type == DMDA_STENCIL_BOX) {
    left   = IXs - Xs; right = left + (IXe-IXs);
    bottom = IYs - Ys; top = bottom + (IYe-IYs);
    down   = IZs - Zs; up  = down + (IZe-IZs);
    count  = 0;
    for (i=down; i<up; i++) {
      for (j=bottom; j<top; j++) {
        for (k=left; k<right; k++) {
          idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
        }
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);
  } else {
    /* This is way ugly! We need to list the funny cross type region */
    left   = xs - Xs; right = left + x;
    bottom = ys - Ys; top = bottom + y;
    down   = zs - Zs;   up  = down + z;
    count  = 0;
    /* the bottom chunk */
    for (i=(IZs-Zs); i<down; i++) {
      for (j=bottom; j<top; j++) {
        for (k=left; k<right; k++) idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
      }
    }
    /* the middle piece */
    for (i=down; i<up; i++) {
      /* front */
      for (j=(IYs-Ys); j<bottom; j++) {
        for (k=left; k<right; k++) idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
      }
      /* middle */
      for (j=bottom; j<top; j++) {
        for (k=IXs-Xs; k<IXe-Xs; k++) idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
      }
      /* back */
      for (j=top; j<top+IYe-ye; j++) {
        for (k=left; k<right; k++) idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
      }
    }
    /* the top piece */
    for (i=up; i<up+IZe-ze; i++) {
      for (j=bottom; j<top; j++) {
        for (k=left; k<right; k++) idx[count++] = (i*(Ye-Ys) + j)*(Xe-Xs) + k;
      }
    }
    ierr = ISCreateBlock(comm,dof,count,idx,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);
  }

  /* determine who lies on each side of use stored in    n24 n25 n26
                                                         n21 n22 n23
                                                         n18 n19 n20

                                                         n15 n16 n17
                                                         n12     n14
                                                         n9  n10 n11

                                                         n6  n7  n8
                                                         n3  n4  n5
                                                         n0  n1  n2
  */

  /* Solve for X,Y, and Z Periodic Case First, Then Modify Solution */
  /* Assume Nodes are Internal to the Cube */
  n0 = rank - m*n - m - 1;
  n1 = rank - m*n - m;
  n2 = rank - m*n - m + 1;
  n3 = rank - m*n -1;
  n4 = rank - m*n;
  n5 = rank - m*n + 1;
  n6 = rank - m*n + m - 1;
  n7 = rank - m*n + m;
  n8 = rank - m*n + m + 1;

  n9  = rank - m - 1;
  n10 = rank - m;
  n11 = rank - m + 1;
  n12 = rank - 1;
  n14 = rank + 1;
  n15 = rank + m - 1;
  n16 = rank + m;
  n17 = rank + m + 1;

  n18 = rank + m*n - m - 1;
  n19 = rank + m*n - m;
  n20 = rank + m*n - m + 1;
  n21 = rank + m*n - 1;
  n22 = rank + m*n;
  n23 = rank + m*n + 1;
  n24 = rank + m*n + m - 1;
  n25 = rank + m*n + m;
  n26 = rank + m*n + m + 1;

  /* Assume Pieces are on Faces of Cube */

  if (xs == 0) { /* First assume not corner or edge */
    n0  = rank       -1 - (m*n);
    n3  = rank + m   -1 - (m*n);
    n6  = rank + 2*m -1 - (m*n);
    n9  = rank       -1;
    n12 = rank + m   -1;
    n15 = rank + 2*m -1;
    n18 = rank       -1 + (m*n);
    n21 = rank + m   -1 + (m*n);
    n24 = rank + 2*m -1 + (m*n);
  }

  if (xe == M) { /* First assume not corner or edge */
    n2  = rank -2*m +1 - (m*n);
    n5  = rank - m  +1 - (m*n);
    n8  = rank      +1 - (m*n);
    n11 = rank -2*m +1;
    n14 = rank - m  +1;
    n17 = rank      +1;
    n20 = rank -2*m +1 + (m*n);
    n23 = rank - m  +1 + (m*n);
    n26 = rank      +1 + (m*n);
  }

  if (ys==0) { /* First assume not corner or edge */
    n0  = rank + m * (n-1) -1 - (m*n);
    n1  = rank + m * (n-1)    - (m*n);
    n2  = rank + m * (n-1) +1 - (m*n);
    n9  = rank + m * (n-1) -1;
    n10 = rank + m * (n-1);
    n11 = rank + m * (n-1) +1;
    n18 = rank + m * (n-1) -1 + (m*n);
    n19 = rank + m * (n-1)    + (m*n);
    n20 = rank + m * (n-1) +1 + (m*n);
  }

  if (ye == N) { /* First assume not corner or edge */
    n6  = rank - m * (n-1) -1 - (m*n);
    n7  = rank - m * (n-1)    - (m*n);
    n8  = rank - m * (n-1) +1 - (m*n);
    n15 = rank - m * (n-1) -1;
    n16 = rank - m * (n-1);
    n17 = rank - m * (n-1) +1;
    n24 = rank - m * (n-1) -1 + (m*n);
    n25 = rank - m * (n-1)    + (m*n);
    n26 = rank - m * (n-1) +1 + (m*n);
  }

  if (zs == 0) { /* First assume not corner or edge */
    n0 = size - (m*n) + rank - m - 1;
    n1 = size - (m*n) + rank - m;
    n2 = size - (m*n) + rank - m + 1;
    n3 = size - (m*n) + rank - 1;
    n4 = size - (m*n) + rank;
    n5 = size - (m*n) + rank + 1;
    n6 = size - (m*n) + rank + m - 1;
    n7 = size - (m*n) + rank + m;
    n8 = size - (m*n) + rank + m + 1;
  }

  if (ze == P) { /* First assume not corner or edge */
    n18 = (m*n) - (size-rank) - m - 1;
    n19 = (m*n) - (size-rank) - m;
    n20 = (m*n) - (size-rank) - m + 1;
    n21 = (m*n) - (size-rank) - 1;
    n22 = (m*n) - (size-rank);
    n23 = (m*n) - (size-rank) + 1;
    n24 = (m*n) - (size-rank) + m - 1;
    n25 = (m*n) - (size-rank) + m;
    n26 = (m*n) - (size-rank) + m + 1;
  }

  if ((xs==0) && (zs==0)) { /* Assume an edge, not corner */
    n0 = size - m*n + rank + m-1 - m;
    n3 = size - m*n + rank + m-1;
    n6 = size - m*n + rank + m-1 + m;
  }

  if ((xs==0) && (ze==P)) { /* Assume an edge, not corner */
    n18 = m*n - (size - rank) + m-1 - m;
    n21 = m*n - (size - rank) + m-1;
    n24 = m*n - (size - rank) + m-1 + m;
  }

  if ((xs==0) && (ys==0)) { /* Assume an edge, not corner */
    n0  = rank + m*n -1 - m*n;
    n9  = rank + m*n -1;
    n18 = rank + m*n -1 + m*n;
  }

  if ((xs==0) && (ye==N)) { /* Assume an edge, not corner */
    n6  = rank - m*(n-1) + m-1 - m*n;
    n15 = rank - m*(n-1) + m-1;
    n24 = rank - m*(n-1) + m-1 + m*n;
  }

  if ((xe==M) && (zs==0)) { /* Assume an edge, not corner */
    n2 = size - (m*n-rank) - (m-1) - m;
    n5 = size - (m*n-rank) - (m-1);
    n8 = size - (m*n-rank) - (m-1) + m;
  }

  if ((xe==M) && (ze==P)) { /* Assume an edge, not corner */
    n20 = m*n - (size - rank) - (m-1) - m;
    n23 = m*n - (size - rank) - (m-1);
    n26 = m*n - (size - rank) - (m-1) + m;
  }

  if ((xe==M) && (ys==0)) { /* Assume an edge, not corner */
    n2  = rank + m*(n-1) - (m-1) - m*n;
    n11 = rank + m*(n-1) - (m-1);
    n20 = rank + m*(n-1) - (m-1) + m*n;
  }

  if ((xe==M) && (ye==N)) { /* Assume an edge, not corner */
    n8  = rank - m*n +1 - m*n;
    n17 = rank - m*n +1;
    n26 = rank - m*n +1 + m*n;
  }

  if ((ys==0) && (zs==0)) { /* Assume an edge, not corner */
    n0 = size - m + rank -1;
    n1 = size - m + rank;
    n2 = size - m + rank +1;
  }

  if ((ys==0) && (ze==P)) { /* Assume an edge, not corner */
    n18 = m*n - (size - rank) + m*(n-1) -1;
    n19 = m*n - (size - rank) + m*(n-1);
    n20 = m*n - (size - rank) + m*(n-1) +1;
  }

  if ((ye==N) && (zs==0)) { /* Assume an edge, not corner */
    n6 = size - (m*n-rank) - m * (n-1) -1;
    n7 = size - (m*n-rank) - m * (n-1);
    n8 = size - (m*n-rank) - m * (n-1) +1;
  }

  if ((ye==N) && (ze==P)) { /* Assume an edge, not corner */
    n24 = rank - (size-m) -1;
    n25 = rank - (size-m);
    n26 = rank - (size-m) +1;
  }

  /* Check for Corners */
  if ((xs==0) && (ys==0) && (zs==0)) n0  = size -1;
  if ((xs==0) && (ys==0) && (ze==P)) n18 = m*n-1;
  if ((xs==0) && (ye==N) && (zs==0)) n6  = (size-1)-m*(n-1);
  if ((xs==0) && (ye==N) && (ze==P)) n24 = m-1;
  if ((xe==M) && (ys==0) && (zs==0)) n2  = size-m;
  if ((xe==M) && (ys==0) && (ze==P)) n20 = m*n-m;
  if ((xe==M) && (ye==N) && (zs==0)) n8  = size-m*n;
  if ((xe==M) && (ye==N) && (ze==P)) n26 = 0;

  /* Check for when not X,Y, and Z Periodic */

  /* If not X periodic */
  if (bx != DM_BOUNDARY_PERIODIC) {
    if (xs==0) n0 = n3 = n6 = n9  = n12 = n15 = n18 = n21 = n24 = -2;
    if (xe==M) n2 = n5 = n8 = n11 = n14 = n17 = n20 = n23 = n26 = -2;
  }

  /* If not Y periodic */
  if (by != DM_BOUNDARY_PERIODIC) {
    if (ys==0) n0 = n1 = n2 = n9  = n10 = n11 = n18 = n19 = n20 = -2;
    if (ye==N) n6 = n7 = n8 = n15 = n16 = n17 = n24 = n25 = n26 = -2;
  }

  /* If not Z periodic */
  if (bz != DM_BOUNDARY_PERIODIC) {
    if (zs==0) n0  = n1  = n2  = n3  = n4  = n5  = n6  = n7  = n8  = -2;
    if (ze==P) n18 = n19 = n20 = n21 = n22 = n23 = n24 = n25 = n26 = -2;
  }

  ierr = PetscMalloc1(27,&dd->neighbors);CHKERRQ(ierr);

  dd->neighbors[0]  = n0;
  dd->neighbors[1]  = n1;
  dd->neighbors[2]  = n2;
  dd->neighbors[3]  = n3;
  dd->neighbors[4]  = n4;
  dd->neighbors[5]  = n5;
  dd->neighbors[6]  = n6;
  dd->neighbors[7]  = n7;
  dd->neighbors[8]  = n8;
  dd->neighbors[9]  = n9;
  dd->neighbors[10] = n10;
  dd->neighbors[11] = n11;
  dd->neighbors[12] = n12;
  dd->neighbors[13] = rank;
  dd->neighbors[14] = n14;
  dd->neighbors[15] = n15;
  dd->neighbors[16] = n16;
  dd->neighbors[17] = n17;
  dd->neighbors[18] = n18;
  dd->neighbors[19] = n19;
  dd->neighbors[20] = n20;
  dd->neighbors[21] = n21;
  dd->neighbors[22] = n22;
  dd->neighbors[23] = n23;
  dd->neighbors[24] = n24;
  dd->neighbors[25] = n25;
  dd->neighbors[26] = n26;

  /* If star stencil then delete the corner neighbors */
  if (stencil_type == DMDA_STENCIL_STAR) {
    /* save information about corner neighbors */
    sn0 = n0; sn1 = n1; sn2 = n2; sn3 = n3; sn5 = n5; sn6 = n6; sn7 = n7;
    sn8 = n8; sn9 = n9; sn11 = n11; sn15 = n15; sn17 = n17; sn18 = n18;
    sn19 = n19; sn20 = n20; sn21 = n21; sn23 = n23; sn24 = n24; sn25 = n25;
    sn26 = n26;
    n0 = n1 = n2 = n3 = n5 = n6 = n7 = n8 = n9 = n11 = n15 = n17 = n18 = n19 = n20 = n21 = n23 = n24 = n25 = n26 = -1;
  }

  ierr = PetscMalloc1((Xe-Xs)*(Ye-Ys)*(Ze-Zs),&idx);CHKERRQ(ierr);

  nn = 0;
  /* Bottom Level */
  for (k=0; k<s_z; k++) {
    for (i=1; i<=s_y; i++) {
      if (n0 >= 0) { /* left below */
        x_t = lx[n0 % m];
        y_t = ly[(n0 % (m*n))/m];
        z_t = lz[n0 / (m*n)];
        s_t = bases[n0] + x_t*y_t*z_t - (s_y-i)*x_t - s_x - (s_z-k-1)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n0] + x_t*y_t*z_t - (s_y-i)*x_t - s_x; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n1 % (m*n))/m];
        z_t = lz[n1 / (m*n)];
        s_t = bases[n1] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n1] + x_t*y_t*z_t - (s_y+1-i)*x_t; /* 2D case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      }
      if (n2 >= 0) { /* right below */
        x_t = lx[n2 % m];
        y_t = ly[(n2 % (m*n))/m];
        z_t = lz[n2 / (m*n)];
        s_t = bases[n2] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n2] + x_t*y_t*z_t - (s_y+1-i)*x_t; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }

    for (i=0; i<y; i++) {
      if (n3 >= 0) { /* directly left */
        x_t = lx[n3 % m];
        y_t = y;
        z_t = lz[n3 / (m*n)];
        s_t = bases[n3] + (i+1)*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n3] + (i+1)*x_t - s_x + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }

      if (n4 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        z_t = lz[n4 / (m*n)];
        s_t = bases[n4] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n4] + i*x_t + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      } else if (bz == DM_BOUNDARY_MIRROR) {
        for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + x*i + (s_z - k - 1)*x*y;
      }

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m];
        y_t = y;
        z_t = lz[n5 / (m*n)];
        s_t = bases[n5] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n5] + i*x_t + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m];
        y_t = ly[(n6 % (m*n))/m];
        z_t = lz[n6 / (m*n)];
        s_t = bases[n6] + i*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n6] + i*x_t - s_x + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        y_t = ly[(n7 % (m*n))/m];
        z_t = lz[n7 / (m*n)];
        s_t = bases[n7] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n7] + (i-1)*x_t + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      }
      if (n8 >= 0) { /* right above */
        x_t = lx[n8 % m];
        y_t = ly[(n8 % (m*n))/m];
        z_t = lz[n8 / (m*n)];
        s_t = bases[n8] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        if (twod && (s_t < 0)) s_t = bases[n8] + (i-1)*x_t + x_t*y_t*z_t - x_t*y_t; /* 2D case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }
  }

  /* Middle Level */
  for (k=0; k<z; k++) {
    for (i=1; i<=s_y; i++) {
      if (n9 >= 0) { /* left below */
        x_t = lx[n9 % m];
        y_t = ly[(n9 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n9] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n10 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n10 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n10] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      }  else if (by == DM_BOUNDARY_MIRROR) {
        for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + k*x*y + (s_y - i)*x;
      }
      if (n11 >= 0) { /* right below */
        x_t = lx[n11 % m];
        y_t = ly[(n11 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n11] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }

    for (i=0; i<y; i++) {
      if (n12 >= 0) { /* directly left */
        x_t = lx[n12 % m];
        y_t = y;
        /* z_t = z; */
        s_t = bases[n12] + (i+1)*x_t - s_x + k*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }  else if (bx == DM_BOUNDARY_MIRROR) {
        for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + s_x - j - 1 + k*x*y + i*x;
      }

      /* Interior */
      s_t = bases[rank] + i*x + k*x*y;
      for (j=0; j<x; j++) idx[nn++] = s_t++;

      if (n14 >= 0) { /* directly right */
        x_t = lx[n14 % m];
        y_t = y;
        /* z_t = z; */
        s_t = bases[n14] + i*x_t + k*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      } else if (bx == DM_BOUNDARY_MIRROR) {
        for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x - j - 1 + k*x*y + i*x;
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n15 >= 0) { /* left above */
        x_t = lx[n15 % m];
        y_t = ly[(n15 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n15] + i*x_t - s_x + k*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n16 >= 0) { /* directly above */
        x_t = x;
        y_t = ly[(n16 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n16] + (i-1)*x_t + k*x_t*y_t;
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      } else if (by == DM_BOUNDARY_MIRROR) {
        for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + k*x*y + (y-i)*x;
      }
      if (n17 >= 0) { /* right above */
        x_t = lx[n17 % m];
        y_t = ly[(n17 % (m*n))/m];
        /* z_t = z; */
        s_t = bases[n17] + (i-1)*x_t + k*x_t*y_t;
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }
  }

  /* Upper Level */
  for (k=0; k<s_z; k++) {
    for (i=1; i<=s_y; i++) {
      if (n18 >= 0) { /* left below */
        x_t = lx[n18 % m];
        y_t = ly[(n18 % (m*n))/m];
        /* z_t = lz[n18 / (m*n)]; */
        s_t = bases[n18] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n18] - (s_y-i)*x_t -s_x + x_t*y_t; /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n19 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n19 % (m*n))/m];
        /* z_t = lz[n19 / (m*n)]; */
        s_t = bases[n19] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n19] - (s_y+1-i)*x_t + x_t*y_t; /* 2d case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      }
      if (n20 >= 0) { /* right below */
        x_t = lx[n20 % m];
        y_t = ly[(n20 % (m*n))/m];
        /* z_t = lz[n20 / (m*n)]; */
        s_t = bases[n20] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n20] - (s_y+1-i)*x_t + x_t*y_t; /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }

    for (i=0; i<y; i++) {
      if (n21 >= 0) { /* directly left */
        x_t = lx[n21 % m];
        y_t = y;
        /* z_t = lz[n21 / (m*n)]; */
        s_t = bases[n21] + (i+1)*x_t - s_x + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n21] + (i+1)*x_t - s_x;  /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }

      if (n22 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        /* z_t = lz[n22 / (m*n)]; */
        s_t = bases[n22] + i*x_t + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n22] + i*x_t; /* 2d case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      } else if (bz == DM_BOUNDARY_MIRROR) {
        for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + (z-k-1)*x*y + i*x;
      }

      if (n23 >= 0) { /* directly right */
        x_t = lx[n23 % m];
        y_t = y;
        /* z_t = lz[n23 / (m*n)]; */
        s_t = bases[n23] + i*x_t + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n23] + i*x_t; /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n24 >= 0) { /* left above */
        x_t = lx[n24 % m];
        y_t = ly[(n24 % (m*n))/m];
        /* z_t = lz[n24 / (m*n)]; */
        s_t = bases[n24] + i*x_t - s_x + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n24] + i*x_t - s_x; /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
      if (n25 >= 0) { /* directly above */
        x_t = x;
        y_t = ly[(n25 % (m*n))/m];
        /* z_t = lz[n25 / (m*n)]; */
        s_t = bases[n25] + (i-1)*x_t + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n25] + (i-1)*x_t; /* 2d case */
        for (j=0; j<x_t; j++) idx[nn++] = s_t++;
      }
      if (n26 >= 0) { /* right above */
        x_t = lx[n26 % m];
        y_t = ly[(n26 % (m*n))/m];
        /* z_t = lz[n26 / (m*n)]; */
        s_t = bases[n26] + (i-1)*x_t + k*x_t*y_t;
        if (twod && (s_t >= M*N*P)) s_t = bases[n26] + (i-1)*x_t; /* 2d case */
        for (j=0; j<s_x; j++) idx[nn++] = s_t++;
      }
    }
  }

  ierr = ISCreateBlock(comm,dof,nn,idx,PETSC_USE_POINTER,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  if (stencil_type == DMDA_STENCIL_STAR) {
    n0  = sn0;  n1  = sn1;  n2  = sn2;  n3  = sn3;  n5  = sn5;  n6  = sn6; n7 = sn7;
    n8  = sn8;  n9  = sn9;  n11 = sn11; n15 = sn15; n17 = sn17; n18 = sn18;
    n19 = sn19; n20 = sn20; n21 = sn21; n23 = sn23; n24 = sn24; n25 = sn25;
    n26 = sn26;
  }

  if (((stencil_type == DMDA_STENCIL_STAR) || (bx != DM_BOUNDARY_PERIODIC && bx) || (by != DM_BOUNDARY_PERIODIC && by) || (bz != DM_BOUNDARY_PERIODIC && bz))) {
    /*
        Recompute the local to global mappings, this time keeping the
      information about the cross corner processor numbers.
    */
    nn = 0;
    /* Bottom Level */
    for (k=0; k<s_z; k++) {
      for (i=1; i<=s_y; i++) {
        if (n0 >= 0) { /* left below */
          x_t = lx[n0 % m];
          y_t = ly[(n0 % (m*n))/m];
          z_t = lz[n0 / (m*n)];
          s_t = bases[n0] + x_t*y_t*z_t - (s_y-i)*x_t - s_x - (s_z-k-1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && Ys-ys < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n1 >= 0) { /* directly below */
          x_t = x;
          y_t = ly[(n1 % (m*n))/m];
          z_t = lz[n1 / (m*n)];
          s_t = bases[n1] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (Ys-ys < 0 && Zs-zs < 0) {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
        if (n2 >= 0) { /* right below */
          x_t = lx[n2 % m];
          y_t = ly[(n2 % (m*n))/m];
          z_t = lz[n2 / (m*n)];
          s_t = bases[n2] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && Ys-ys < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (i=0; i<y; i++) {
        if (n3 >= 0) { /* directly left */
          x_t = lx[n3 % m];
          y_t = y;
          z_t = lz[n3 / (m*n)];
          s_t = bases[n3] + (i+1)*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }

        if (n4 >= 0) { /* middle */
          x_t = x;
          y_t = y;
          z_t = lz[n4 / (m*n)];
          s_t = bases[n4] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (Zs-zs < 0) {
          if (bz == DM_BOUNDARY_MIRROR) {
            for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + x*i + (s_z - k - 1)*x*y;
          } else {
            for (j=0; j<x; j++) idx[nn++] = -1;
          }
        }

        if (n5 >= 0) { /* directly right */
          x_t = lx[n5 % m];
          y_t = y;
          z_t = lz[n5 / (m*n)];
          s_t = bases[n5] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (i=1; i<=s_y; i++) {
        if (n6 >= 0) { /* left above */
          x_t = lx[n6 % m];
          y_t = ly[(n6 % (m*n))/m];
          z_t = lz[n6 / (m*n)];
          s_t = bases[n6] + i*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && ye-Ye < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n7 >= 0) { /* directly above */
          x_t = x;
          y_t = ly[(n7 % (m*n))/m];
          z_t = lz[n7 / (m*n)];
          s_t = bases[n7] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (ye-Ye < 0 && Zs-zs < 0) {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
        if (n8 >= 0) { /* right above */
          x_t = lx[n8 % m];
          y_t = ly[(n8 % (m*n))/m];
          z_t = lz[n8 / (m*n)];
          s_t = bases[n8] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && ye-Ye < 0 && Zs-zs < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }
    }

    /* Middle Level */
    for (k=0; k<z; k++) {
      for (i=1; i<=s_y; i++) {
        if (n9 >= 0) { /* left below */
          x_t = lx[n9 % m];
          y_t = ly[(n9 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n9] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && Ys-ys < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n10 >= 0) { /* directly below */
          x_t = x;
          y_t = ly[(n10 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n10] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (Ys-ys < 0) {
          if (by == DM_BOUNDARY_MIRROR) {
            for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + k*x*y + (s_y - i)*x;
          } else {
            for (j=0; j<x; j++) idx[nn++] = -1;
          }
        }
        if (n11 >= 0) { /* right below */
          x_t = lx[n11 % m];
          y_t = ly[(n11 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n11] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && Ys-ys < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (i=0; i<y; i++) {
        if (n12 >= 0) { /* directly left */
          x_t = lx[n12 % m];
          y_t = y;
          /* z_t = z; */
          s_t = bases[n12] + (i+1)*x_t - s_x + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0) {
          if (bx == DM_BOUNDARY_MIRROR) {
            for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + s_x - j - 1 + k*x*y + i*x;
          } else {
            for (j=0; j<s_x; j++) idx[nn++] = -1;
          }
        }

        /* Interior */
        s_t = bases[rank] + i*x + k*x*y;
        for (j=0; j<x; j++) idx[nn++] = s_t++;

        if (n14 >= 0) { /* directly right */
          x_t = lx[n14 % m];
          y_t = y;
          /* z_t = z; */
          s_t = bases[n14] + i*x_t + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0) {
          if (bx == DM_BOUNDARY_MIRROR) {
            for (j=0; j<s_x; j++) idx[nn++] = bases[rank] + x - j - 1 + k*x*y + i*x;
          } else {
            for (j=0; j<s_x; j++) idx[nn++] = -1;
          }
        }
      }

      for (i=1; i<=s_y; i++) {
        if (n15 >= 0) { /* left above */
          x_t = lx[n15 % m];
          y_t = ly[(n15 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n15] + i*x_t - s_x + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && ye-Ye < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n16 >= 0) { /* directly above */
          x_t = x;
          y_t = ly[(n16 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n16] + (i-1)*x_t + k*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (ye-Ye < 0) {
          if (by == DM_BOUNDARY_MIRROR) {
            for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + k*x*y + (y-i)*x;
          } else {
            for (j=0; j<x; j++) idx[nn++] = -1;
          }
        }
        if (n17 >= 0) { /* right above */
          x_t = lx[n17 % m];
          y_t = ly[(n17 % (m*n))/m];
          /* z_t = z; */
          s_t = bases[n17] + (i-1)*x_t + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && ye-Ye < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }
    }

    /* Upper Level */
    for (k=0; k<s_z; k++) {
      for (i=1; i<=s_y; i++) {
        if (n18 >= 0) { /* left below */
          x_t = lx[n18 % m];
          y_t = ly[(n18 % (m*n))/m];
          /* z_t = lz[n18 / (m*n)]; */
          s_t = bases[n18] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && Ys-ys < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n19 >= 0) { /* directly below */
          x_t = x;
          y_t = ly[(n19 % (m*n))/m];
          /* z_t = lz[n19 / (m*n)]; */
          s_t = bases[n19] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (Ys-ys < 0 && ze-Ze < 0) {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
        if (n20 >= 0) { /* right below */
          x_t = lx[n20 % m];
          y_t = ly[(n20 % (m*n))/m];
          /* z_t = lz[n20 / (m*n)]; */
          s_t = bases[n20] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && Ys-ys < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (i=0; i<y; i++) {
        if (n21 >= 0) { /* directly left */
          x_t = lx[n21 % m];
          y_t = y;
          /* z_t = lz[n21 / (m*n)]; */
          s_t = bases[n21] + (i+1)*x_t - s_x + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }

        if (n22 >= 0) { /* middle */
          x_t = x;
          y_t = y;
          /* z_t = lz[n22 / (m*n)]; */
          s_t = bases[n22] + i*x_t + k*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (ze-Ze < 0) {
          if (bz == DM_BOUNDARY_MIRROR) {
            for (j=0; j<x; j++) idx[nn++] = bases[rank] + j + (z-k-1)*x*y + i*x;
          } else {
            for (j=0; j<x; j++) idx[nn++] = -1;
          }
        }

        if (n23 >= 0) { /* directly right */
          x_t = lx[n23 % m];
          y_t = y;
          /* z_t = lz[n23 / (m*n)]; */
          s_t = bases[n23] + i*x_t + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
      }

      for (i=1; i<=s_y; i++) {
        if (n24 >= 0) { /* left above */
          x_t = lx[n24 % m];
          y_t = ly[(n24 % (m*n))/m];
          /* z_t = lz[n24 / (m*n)]; */
          s_t = bases[n24] + i*x_t - s_x + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (Xs-xs < 0 && ye-Ye < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
        if (n25 >= 0) { /* directly above */
          x_t = x;
          y_t = ly[(n25 % (m*n))/m];
          /* z_t = lz[n25 / (m*n)]; */
          s_t = bases[n25] + (i-1)*x_t + k*x_t*y_t;
          for (j=0; j<x_t; j++) idx[nn++] = s_t++;
        } else if (ye-Ye < 0 && ze-Ze < 0) {
          for (j=0; j<x; j++) idx[nn++] = -1;
        }
        if (n26 >= 0) { /* right above */
          x_t = lx[n26 % m];
          y_t = ly[(n26 % (m*n))/m];
          /* z_t = lz[n26 / (m*n)]; */
          s_t = bases[n26] + (i-1)*x_t + k*x_t*y_t;
          for (j=0; j<s_x; j++) idx[nn++] = s_t++;
        } else if (xe-Xe < 0 && ye-Ye < 0 && ze-Ze < 0) {
          for (j=0; j<s_x; j++) idx[nn++] = -1;
        }
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
  dd->m = m;  dd->n  = n;  dd->p  = p;
  /* note petsc expects xs/xe/Xs/Xe to be multiplied by #dofs in many places */
  dd->xs = xs*dof; dd->xe = xe*dof; dd->ys = ys; dd->ye = ye; dd->zs = zs; dd->ze = ze;
  dd->Xs = Xs*dof; dd->Xe = Xe*dof; dd->Ys = Ys; dd->Ye = Ye; dd->Zs = Zs; dd->Ze = Ze;

  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  dd->gtol      = gtol;
  dd->base      = base;
  da->ops->view = DMView_DA_3d;
  dd->ltol      = NULL;
  dd->ao        = NULL;
  PetscFunctionReturn(0);
}

/*@C
   DMDACreate3d - Creates an object that will manage the communication of three-dimensional
   regular array data that is distributed across some processors.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bx,by,bz - type of ghost nodes the array have.
         Use one of DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_PERIODIC.
.  stencil_type - Type of stencil (DMDA_STENCIL_STAR or DMDA_STENCIL_BOX)
.  M,N,P - global dimension in each direction of the array
.  m,n,p - corresponding number of processors in each dimension
           (or PETSC_DECIDE to have calculated)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx, ly, lz - arrays containing the number of nodes in each cell along
          the x, y, and z coordinates, or NULL. If non-null, these
          must be of length as m,n,p and the corresponding
          m,n, or p cannot be PETSC_DECIDE. Sum of the lx[] entries must be M, sum of
          the ly[] must N, sum of the lz[] must be P

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Key:
+  -dm_view - Calls DMView() at the conclusion of DMDACreate3d()
.  -da_grid_x <nx> - number of grid points in x direction
.  -da_grid_y <ny> - number of grid points in y direction
.  -da_grid_z <nz> - number of grid points in z direction
.  -da_processors_x <MX> - number of processors in x direction
.  -da_processors_y <MY> - number of processors in y direction
.  -da_processors_z <MZ> - number of processors in z direction
.  -da_refine_x <rx> - refinement ratio in x direction
.  -da_refine_y <ry> - refinement ratio in y direction
.  -da_refine_z <rz>- refinement ratio in z directio
-  -da_refine <n> - refine the DMDA n times before creating it

   Level: beginner

   Notes:
   The stencil type DMDA_STENCIL_STAR with width 1 corresponds to the
   standard 7-pt stencil, while DMDA_STENCIL_BOX with width 1 denotes
   the standard 27-pt stencil.

   The array data itself is NOT stored in the DMDA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DMCreateGlobalVector()
   and DMCreateLocalVector() and calls to VecDuplicate() if more are needed.

   You must call DMSetUp() after this call before using this DM.

   If you wish to use the options database to change values in the DMDA call DMSetFromOptions() after this call
   but before DMSetUp().

.seealso: DMDestroy(), DMView(), DMDACreate1d(), DMDACreate2d(), DMGlobalToLocalBegin(), DMDAGetRefinementFactor(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToLocalBegin(), DMLocalToLocalEnd(), DMDASetRefinementFactor(),
          DMDAGetInfo(), DMCreateGlobalVector(), DMCreateLocalVector(), DMDACreateNaturalVector(), DMLoad(), DMDAGetOwnershipRanges(),
          DMStagCreate3d()

@*/
PetscErrorCode  DMDACreate3d(MPI_Comm comm,DMBoundaryType bx,DMBoundaryType by,DMBoundaryType bz,DMDAStencilType stencil_type,PetscInt M,
               PetscInt N,PetscInt P,PetscInt m,PetscInt n,PetscInt p,PetscInt dof,PetscInt s,const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],DM *da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate(comm, da);CHKERRQ(ierr);
  ierr = DMSetDimension(*da, 3);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, N, P);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*da, m, n, p);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*da, bx, by, bz);CHKERRQ(ierr);
  ierr = DMDASetDof(*da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*da, stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*da, s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*da, lx, ly, lz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
