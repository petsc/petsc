/*$Id: da2.c,v 1.180 2001/09/07 20:12:17 bsmith Exp $*/
 
#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetOwnershipRange"
int DAGetOwnershipRange(DA da,int **lx,int **ly,int **lz)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (lx) *lx = da->lx;
  if (ly) *ly = da->ly;
  if (lz) *lz = da->lz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAView_2d"
int DAView_2d(DA da,PetscViewer viewer)
{
  int        rank,ierr;
  PetscTruth isascii,isdraw;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(da->comm,&rank);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Processor [%d] M %d N %d m %d n %d w %d s %d\n",rank,da->M,
                             da->N,da->m,da->n,da->w,da->s);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"X range of indices: %d %d, Y range of indices: %d %d\n",da->xs,da->xe,da->ys,da->ye);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    PetscDraw       draw;
    double     ymin = -1*da->s-1,ymax = da->N+da->s;
    double     xmin = -1*da->s-1,xmax = da->M+da->s;
    double     x,y;
    int        base,*idx;
    char       node[10];
    PetscTruth isnull;
 
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
    ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = da->N - 1;
      for (xmin=0; xmin<da->M; xmin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
      xmin = 0.0; xmax = da->M - 1;
      for (ymin=0; ymin<da->N; ymin++) {
        ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);

    /* draw my box */
    ymin = da->ys; ymax = da->ye - 1; xmin = da->xs/da->w; 
    xmax =(da->xe-1)/da->w;
    ierr = PetscDrawLine(draw,xmin,ymin,xmax,ymin,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymin,xmin,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmin,ymax,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xmax,ymin,xmax,ymax,PETSC_DRAW_RED);CHKERRQ(ierr);

    /* put in numbers */
    base = (da->base)/da->w;
    for (y=ymin; y<=ymax; y++) {
      for (x=xmin; x<=xmax; x++) {
        sprintf(node,"%d",base++);
        ierr = PetscDrawString(draw,x,y,PETSC_DRAW_BLACK,node);CHKERRQ(ierr);
      }
    }

    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
    /* overlay ghost numbers, useful for error checking */
    /* put in numbers */

    base = 0; idx = da->idx;
    ymin = da->Ys; ymax = da->Ye; xmin = da->Xs; xmax = da->Xe;
    for (y=ymin; y<ymax; y++) {
      for (x=xmin; x<xmax; x++) {
        if ((base % da->w) == 0) {
          sprintf(node,"%d",idx[base]/da->w);
          ierr = PetscDrawString(draw,x/da->w,y,PETSC_DRAW_BLUE,node);CHKERRQ(ierr);
        }
        base++;
      }
    }        
    ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for DA2d",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
/*
      This function tells the AMS the layout of the vectors, it is called
   in the VecPublish_xx routines.
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "AMSSetFieldBlock_DA"
int AMSSetFieldBlock_DA(AMS_Memory amem,char *name,Vec vec)
{
  int        ierr,dof,dim,ends[4],shift = 0,starts[] = {0,0,0,0};
  DA         da = 0;
  PetscTruth isseq,ismpi;

  PetscFunctionBegin;
  if (((PetscObject)vec)->amem < 0) PetscFunctionReturn(0); /* return if not published */

  ierr = PetscObjectQuery((PetscObject)vec,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) PetscFunctionReturn(0);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (dof > 1) {dim++; shift = 1; ends[0] = dof;}

  ierr = PetscTypeCompare((PetscObject)vec,VECSEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)vec,VECMPI,&ismpi);CHKERRQ(ierr);
  if (isseq) {
    ierr = DAGetGhostCorners(da,0,0,0,ends+shift,ends+shift+1,ends+shift+2);CHKERRQ(ierr);
    ends[shift]   += starts[shift]-1;
    ends[shift+1] += starts[shift+1]-1;
    ends[shift+2] += starts[shift+2]-1;
    ierr = AMS_Memory_set_field_block(amem,name,dim,starts,ends);CHKERRQ(ierr);
    if (ierr) {
      char *message;
      AMS_Explain_error(ierr,&message);
      SETERRQ(ierr,message);
    }
  } else if (ismpi) {
    ierr = DAGetCorners(da,starts+shift,starts+shift+1,starts+shift+2,
                           ends+shift,ends+shift+1,ends+shift+2);CHKERRQ(ierr);
    ends[shift]   += starts[shift]-1;
    ends[shift+1] += starts[shift+1]-1;
    ends[shift+2] += starts[shift+2]-1;
    ierr = AMS_Memory_set_field_block(amem,name,dim,starts,ends);
    if (ierr) {
      char *message;
      AMS_Explain_error(ierr,&message);
      SETERRQ(ierr,message);
    }
  } else {
    SETERRQ1(1,"Wrong vector type %s for this call",((PetscObject)vec)->type_name);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "DAPublish_Petsc"
int DAPublish_Petsc(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  DA          v = (DA) obj;
  int         ierr;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DACreate2d"
/*@C
   DACreate2d -  Creates an object that will manage the communication of  two-dimensional 
   regular array data that is distributed across some processors.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  wrap - type of periodicity should the array have. 
         Use one of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, or DA_XYPERIODIC.
.  stencil_type - stencil type.  Use either DA_STENCIL_BOX or DA_STENCIL_STAR.
.  M,N - global dimension in each direction of the array
.  m,n - corresponding number of processors in each dimension 
         (or PETSC_DECIDE to have calculated)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx, ly - arrays containing the number of nodes in each cell along
           the x and y coordinates, or PETSC_NULL. If non-null, these
           must be of length as m and n, and the corresponding
           m and n cannot be PETSC_DECIDE. The sum of the lx[] entries
           must be M, and the sum of the ly[] entries must be N.

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
+  -da_view - Calls DAView() at the conclusion of DACreate2d()
.  -da_grid_x <nx> - number of grid points in x direction, if M < 0
.  -da_grid_y <ny> - number of grid points in y direction, if N < 0
.  -da_processors_x <nx> - number of processors in x direction
-  -da_processors_y <ny> - number of processors in y direction

   Level: beginner

   Notes:
   The stencil type DA_STENCIL_STAR with width 1 corresponds to the 
   standard 5-pt stencil, while DA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

   The array data itself is NOT stored in the DA, it is stored in Vec objects;
   The appropriate vector objects can be obtained with calls to DACreateGlobalVector()
   and DACreateLocalVector() and calls to VecDuplicate() if more are needed.

.keywords: distributed array, create, two-dimensional

.seealso: DADestroy(), DAView(), DACreate1d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DALocalToLocalBegin(), DALocalToLocalEnd(),
          DAGetInfo(), DACreateGlobalVector(), DACreateLocalVector(), DACreateNaturalVector(), DALoad(), DAView()

@*/
int DACreate2d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type,
                int M,int N,int m,int n,int dof,int s,int *lx,int *ly,DA *inra)
{
  int           rank,size,xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,ierr,start,end;
  int           up,down,left,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  int           xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count;
  int           s_x,s_y; /* s proportionalized to w */
  int           *flx = 0,*fly = 0;
  int           sn0 = 0,sn2 = 0,sn6 = 0,sn8 = 0,refine_x = 2, refine_y = 2,tM = M,tN = N;
  PetscTruth    flg1,flg2;
  DA            da;
  Vec           local,global;
  VecScatter    ltog,gtol;
  IS            to,from;

  PetscFunctionBegin;
  PetscValidPointer(inra);
  *inra = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);                                                                 CHKERRQ(ierr);
#endif

  if (dof < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %d",dof);
  if (s < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %d",s);

  ierr = PetscOptionsBegin(comm,PETSC_NULL,"2d DA Options","DA");CHKERRQ(ierr);
    if (M < 0){
      tM = -M;
      ierr = PetscOptionsInt("-da_grid_x","Number of grid points in x direction","DACreate2d",tM,&tM,PETSC_NULL);CHKERRQ(ierr);
    }
    if (N < 0){
      tN = -N;
      ierr = PetscOptionsInt("-da_grid_y","Number of grid points in y direction","DACreate2d",tN,&tN,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-da_processors_x","Number of processors in x direction","DACreate2d",m,&m,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_processors_y","Number of processors in y direction","DACreate2d",n,&n,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_refine_x","Refinement ratio in x direction","DACreate2d",refine_x,&refine_x,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_refine_y","Refinement ratio in y direction","DACreate2d",refine_y,&refine_y,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  M = tM; N = tN;

  PetscHeaderCreate(da,_p_DA,struct _DAOps,DA_COOKIE,0,"DA",comm,DADestroy,DAView);
  PetscLogObjectCreate(da);
  da->bops->publish           = DAPublish_Petsc;
  da->ops->createglobalvector = DACreateGlobalVector;
  da->ops->getinterpolation   = DAGetInterpolation;
  da->ops->getcoloring        = DAGetColoring;
  da->ops->getmatrix          = DAGetMatrix;
  da->ops->refine             = DARefine;
  da->ops->getinjection       = DAGetInjection;
  PetscLogObjectMemory(da,sizeof(struct _p_DA));
  da->dim        = 2;
  da->interptype = DA_Q1;
  da->refine_x   = refine_x;
  da->refine_y   = refine_y;
  ierr = PetscMalloc(dof*sizeof(char*),&da->fieldname);CHKERRQ(ierr);
  ierr = PetscMemzero(da->fieldname,dof*sizeof(char*));CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  if (m != PETSC_DECIDE) {
    if (m < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %d",m);}
    else if (m > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %d %d",m,size);}
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %d",n);}
    else if (n > size) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %d %d",n,size);}
  }

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    /* try for squarish distribution */
    /* This should use MPI_Dims_create instead */
    m = (int)(0.5 + sqrt(((double)M)*((double)size)/((double)N)));
    if (!m) m = 1;
    while (m > 0) {
      n = size/m;
      if (m*n == size) break;
      m--;
    }
    if (M > N && m < n) {int _m = m; m = n; n = _m;}
    if (m*n != size) SETERRQ(PETSC_ERR_PLIB,"Internally Created Bad Partition");
  } else if (m*n != size) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition"); 

  if (M < m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %d %d",M,m);
  if (N < n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %d %d",N,n);

  /*
     We should create an MPI Cartesian topology here, with reorder
     set to true.  That would create a NEW communicator that we would
     need to use for operations on this distributed array 
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-da_partition_nodes_at_end",&flg2);CHKERRQ(ierr);

  /* 
     Determine locally owned region 
     xs is the first local node number, x is the number of local nodes 
  */
  if (lx) { /* user sets distribution */
    x  = lx[rank % m];
    xs = 0;
    for (i=0; i<(rank % m); i++) {
      xs += lx[i];
    }
    left = xs;
    for (i=(rank % m); i<m; i++) {
      left += lx[i];
    }
    if (left != M) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Sum of lx across processors not equal to M: %d %d",left,M);
    }
  } else if (flg2) { 
    x = (M + rank%m)/m;
    if (m > 1 && x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column width is too thin for stencil! %d %d",x,s);
    if (M/m == x) { xs = (rank % m)*x; }
    else          { xs = (rank % m)*(x-1) + (M+(rank % m))%(x*m); }
    SETERRQ(PETSC_ERR_SUP,"-da_partition_nodes_at_end not supported");
  } else { /* Normal PETSc distribution */
    x = M/m + ((M % m) > (rank % m));
    if (m > 1 && x < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column width is too thin for stencil! %d %d",x,s);
    if ((M % m) > (rank % m)) { xs = (rank % m)*x; }
    else                      { xs = (M % m)*(x+1) + ((rank % m)-(M % m))*x; }
    ierr = PetscMalloc(m*sizeof(int),&lx);CHKERRQ(ierr);
    flx = lx;
    for (i=0; i<m; i++) {
      lx[i] = M/m + ((M % m) > i);
    }
  }

  /* 
     Determine locally owned region 
     ys is the first local node number, y is the number of local nodes 
  */
  if (ly) { /* user sets distribution */
    y  = ly[rank/m];
    ys = 0;
    for (i=0; i<(rank/m); i++) {
      ys += ly[i];
    }
    left = ys;
    for (i=(rank/m); i<n; i++) {
      left += ly[i];
    }
    if (left != N) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Sum of ly across processors not equal to N: %d %d",left,N);
    }
  } else if (flg2) { 
    y = (N + rank/m)/n;
    if (n > 1 && y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row width is too thin for stencil! %d %d",y,s);
    if (N/n == y) { ys = (rank/m)*y;  }
    else          { ys = (rank/m)*(y-1) + (N+(rank/m))%(y*n); }
    SETERRQ(PETSC_ERR_SUP,"-da_partition_nodes_at_end not supported");
  } else { /* Normal PETSc distribution */
    y = N/n + ((N % n) > (rank/m));
    if (n > 1 && y < s) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row width is too thin for stencil! %d %d",y,s);
    if ((N % n) > (rank/m)) { ys = (rank/m)*y; }
    else                    { ys = (N % n)*(y+1) + ((rank/m)-(N % n))*y; }
    ierr = PetscMalloc(n*sizeof(int),&ly);CHKERRQ(ierr);
    fly  = ly;
    for (i=0; i<n; i++) {
      ly[i] = N/n + ((N % n) > i);
    }
  }

  xe = xs + x;
  ye = ys + y;

  /* determine ghost region */
  /* Assume No Periodicity */
  if (xs-s > 0) Xs = xs - s; else Xs = 0; 
  if (ys-s > 0) Ys = ys - s; else Ys = 0; 
  if (xe+s <= M) Xe = xe + s; else Xe = M; 
  if (ye+s <= N) Ye = ye + s; else Ye = N;

  /* X Periodic */
  if (DAXPeriodic(wrap)){
    Xs = xs - s; 
    Xe = xe + s; 
  }

  /* Y Periodic */
  if (DAYPeriodic(wrap)){
    Ys = ys - s;
    Ye = ye + s;
  }

  /* Resize all X parameters to reflect w */
  x   *= dof;
  xs  *= dof;
  xe  *= dof;
  Xs  *= dof;
  Xe  *= dof;
  s_x = s*dof;
  s_y = s;

  /* determine starting point of each processor */
  nn = x*y;
  ierr = PetscMalloc((2*size+1)*sizeof(int),&bases);CHKERRQ(ierr);
  ldims = (int*)(bases+size+1);
  ierr = MPI_Allgather(&nn,1,MPI_INT,ldims,1,MPI_INT,comm);CHKERRQ(ierr);
  bases[0] = 0;
  for (i=1; i<=size; i++) {
    bases[i] = ldims[i-1];
  }
  for (i=1; i<=size; i++) {
    bases[i] += bases[i-1];
  }

  /* allocate the base parallel and sequential vectors */
  da->Nlocal = x*y;
  ierr = VecCreateMPIWithArray(comm,da->Nlocal,PETSC_DECIDE,0,&global);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,dof);CHKERRQ(ierr);
  da->nlocal = (Xe-Xs)*(Ye-Ys) ;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,da->nlocal,0,&local);CHKERRQ(ierr);
  ierr = VecSetBlockSize(local,dof);CHKERRQ(ierr);


  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,x*y,start,1,&to);CHKERRQ(ierr);

  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  ierr = PetscMalloc(x*(up - down)*sizeof(int),&idx);CHKERRQ(ierr);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<x; j++) {
      idx[count++] = left + i*(Xe-Xs) + j;
    }
  }
  ierr = ISCreateGeneral(comm,count,idx,&from);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);

  ierr = VecScatterCreate(local,from,global,to,&ltog);CHKERRQ(ierr);
  PetscLogObjectParent(da,to);
  PetscLogObjectParent(da,from);
  PetscLogObjectParent(da,ltog);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX) {
    ierr = ISCreateStride(comm,(Xe-Xs)*(Ye-Ys),0,1,&to);CHKERRQ(ierr); 
  } else {
    /* must drop into cross shape region */
    /*       ---------|
            |  top    |
         |---         ---|
         |   middle      |
         |               |
         ----         ----
            | bottom  |
            -----------
        Xs xs        xe  Xe */
    /* bottom */
    left  = xs - Xs; down = ys - Ys; up    = down + y;
    count = down*(xe-xs) + (up-down)*(Xe-Xs) + (Ye-Ys-up)*(xe-xs);
    ierr  = PetscMalloc(count*sizeof(int),&idx);CHKERRQ(ierr);
    count = 0;
    for (i=0; i<down; i++) {
      for (j=0; j<xe-xs; j++) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    /* middle */
    for (i=down; i<up; i++) {
      for (j=0; j<Xe-Xs; j++) {
        idx[count++] = i*(Xe-Xs) + j;
      }
    }
    /* top */
    for (i=up; i<Ye-Ys; i++) {
      for (j=0; j<xe-xs; j++) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    ierr = ISCreateGeneral(comm,count,idx,&to);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
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


  /* Modify for Periodic Cases */
  if (wrap == DA_YPERIODIC) {  /* Handle Top and Bottom Sides */
    if (n1 < 0) n1 = rank + m * (n-1);
    if (n7 < 0) n7 = rank - m * (n-1);
    if ((n3 >= 0) && (n0 < 0)) n0 = size - m + rank - 1;
    if ((n3 >= 0) && (n6 < 0)) n6 = (rank%m)-1;
    if ((n5 >= 0) && (n2 < 0)) n2 = size - m + rank + 1;
    if ((n5 >= 0) && (n8 < 0)) n8 = (rank%m)+1;
  } else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  } else if (wrap == DA_XYPERIODIC) {

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
  }

  if (stencil_type == DA_STENCIL_STAR) {
    /* save corner processor numbers */
    sn0 = n0; sn2 = n2; sn6 = n6; sn8 = n8; 
    n0 = n2 = n6 = n8 = -1;
  }

  ierr = PetscMalloc((x+2*s_x)*(y+2*s_y)*sizeof(int),&idx);CHKERRQ(ierr);
  PetscLogObjectMemory(da,(x+2*s_x)*(y+2*s_y)*sizeof(int));
  nn = 0;

  xbase = bases[rank];
  for (i=1; i<=s_y; i++) {
    if (n0 >= 0) { /* left below */
      x_t = lx[n0 % m]*dof;
      y_t = ly[(n0/m)];
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = ly[(n1/m)];
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = lx[n2 % m]*dof;
      y_t = ly[(n2/m)];
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=0; i<y; i++) {
    if (n3 >= 0) { /* directly left */
      x_t = lx[n3 % m]*dof;
      /* y_t = y; */
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }

    for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = lx[n5 % m]*dof;
      /* y_t = y; */
      s_t = bases[n5] + (i)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  for (i=1; i<=s_y; i++) {
    if (n6 >= 0) { /* left above */
      x_t = lx[n6 % m]*dof;
      /* y_t = ly[(n6/m)]; */
      s_t = bases[n6] + (i)*x_t - s_x;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      /* y_t = ly[(n7/m)]; */
      s_t = bases[n7] + (i-1)*x_t;
      for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = lx[n8 % m]*dof;
      /* y_t = ly[(n8/m)]; */
      s_t = bases[n8] + (i-1)*x_t;
      for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
    }
  }

  base = bases[rank];
  ierr = ISCreateGeneral(comm,nn,idx,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol);CHKERRQ(ierr);
  PetscLogObjectParent(da,to);
  PetscLogObjectParent(da,from);
  PetscLogObjectParent(da,gtol);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

  if (stencil_type == DA_STENCIL_STAR) {
    /*
        Recompute the local to global mappings, this time keeping the 
      information about the cross corner processor numbers.
    */
    n0 = sn0; n2 = sn2; n6 = sn6; n8 = sn8;
    nn = 0;
    xbase = bases[rank];
    for (i=1; i<=s_y; i++) {
      if (n0 >= 0) { /* left below */
        x_t = lx[n0 % m]*dof;
        y_t = ly[(n0/m)];
        s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = ly[(n1/m)];
        s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      }
      if (n2 >= 0) { /* right below */
        x_t = lx[n2 % m]*dof;
        y_t = ly[(n2/m)];
        s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }

    for (i=0; i<y; i++) {
      if (n3 >= 0) { /* directly left */
        x_t = lx[n3 % m]*dof;
        /* y_t = y; */
        s_t = bases[n3] + (i+1)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }

      for (j=0; j<x; j++) { idx[nn++] = xbase++; } /* interior */

      if (n5 >= 0) { /* directly right */
        x_t = lx[n5 % m]*dof;
        /* y_t = y; */
        s_t = bases[n5] + (i)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }

    for (i=1; i<=s_y; i++) {
      if (n6 >= 0) { /* left above */
        x_t = lx[n6 % m]*dof;
        /* y_t = ly[(n6/m)]; */
        s_t = bases[n6] + (i)*x_t - s_x;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        /* y_t = ly[(n7/m)]; */
        s_t = bases[n7] + (i-1)*x_t;
        for (j=0; j<x_t; j++) { idx[nn++] = s_t++;}
      }
      if (n8 >= 0) { /* right above */
        x_t = lx[n8 % m]*dof;
        /* y_t = ly[(n8/m)]; */
        s_t = bases[n8] + (i-1)*x_t;
        for (j=0; j<s_x; j++) { idx[nn++] = s_t++;}
      }
    }
  }
  ierr = PetscFree(bases);CHKERRQ(ierr); 

  da->M  = M;  da->N  = N;  da->m  = m;  da->n  = n;  da->w = dof;  da->s = s;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye; da->zs = 0; da->ze = 1;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye; da->Zs = 0; da->Ze = 1;
  da->P  = 1;  da->p  = 1;

  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);

  da->gtol         = gtol;
  da->ltog         = ltog;
  da->idx          = idx;
  da->Nl           = nn;
  da->base         = base;
  da->wrap         = wrap;
  da->ops->view    = DAView_2d;
  da->stencil_type = stencil_type;

  /* 
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  ierr = ISLocalToGlobalMappingCreateNC(comm,nn,idx,&da->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(da->ltogmap,da->w,&da->ltogmapb);CHKERRQ(ierr);
  PetscLogObjectParent(da,da->ltogmap);

  *inra = da;

  da->ltol = PETSC_NULL;
  da->ao   = PETSC_NULL;


  if (!flx) {
    ierr = PetscMalloc(m*sizeof(int),&flx);CHKERRQ(ierr);
    ierr = PetscMemcpy(flx,lx,m*sizeof(int));CHKERRQ(ierr);
  }
  if (!fly) {
    ierr = PetscMalloc(n*sizeof(int),&fly);CHKERRQ(ierr);
    ierr = PetscMemcpy(fly,ly,n*sizeof(int));CHKERRQ(ierr);
  }
  da->lx = flx;
  da->ly = fly;

  ierr = PetscOptionsHasName(PETSC_NULL,"-da_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,PETSC_VIEWER_STDOUT_(da->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-da_view_draw",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAView(da,PETSC_VIEWER_DRAW_(da->comm));CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = DAPrintHelp(da);CHKERRQ(ierr);}

  ierr = PetscPublishAll(da);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "DAPrintHelp"
/*@
   DAPrintHelp - Prints command line options for DA.

   Collective on DA

   Input Parameters:
.  da - the distributed array

   Level: intermediate

.seealso: DACreate1d(), DACreate2d(), DACreate3d()

.keywords: DA, help

@*/
int DAPrintHelp(DA da)
{
  static PetscTruth called = PETSC_FALSE;
  MPI_Comm          comm;
  int               ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  comm = da->comm;
  if (!called) {
    ierr = (*PetscHelpPrintf)(comm,"General Distributed Array (DA) options:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -da_view: print DA distribution to screen\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -da_view_draw: display DA in window\n");CHKERRQ(ierr);
    called = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARefine"
/*@C
   DARefine - Creates a new distributed array that is a refinement of a given
   distributed array.

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  comm - communicator to contain refined DA, must be either same as the da communicator or include the 
          da communicator and be 2, 4, or 8 times larger. Currently ignored

   Output Parameter:
.  daref - refined distributed array

   Level: advanced

   Note:
   Currently, refinement consists of just doubling the number of grid spaces
   in each dimension of the DA.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy()
@*/
int DARefine(DA da,MPI_Comm comm,DA *daref)
{
  int M,N,P,ierr;
  DA  da2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  if (DAXPeriodic(da->wrap) || da->interptype == DA_Q0){
    M = da->refine_x*da->M;
  } else {
    M = 1 + da->refine_x*(da->M - 1);
  }
  if (DAYPeriodic(da->wrap) || da->interptype == DA_Q0){
    N = da->refine_y*da->N;
  } else {
    N = 1 + da->refine_y*(da->N - 1);
  }
  if (DAZPeriodic(da->wrap) || da->interptype == DA_Q0){
    P = da->refine_z*da->P;
  } else {
    P = 1 + da->refine_z*(da->P - 1);
  }
  if (da->dim == 1) {
    ierr = DACreate1d(da->comm,da->wrap,M,da->w,da->s,PETSC_NULL,&da2);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    ierr = DACreate2d(da->comm,da->wrap,da->stencil_type,M,N,da->m,da->n,da->w,da->s,PETSC_NULL,PETSC_NULL,&da2);CHKERRQ(ierr);
  } else if (da->dim == 3) {
    ierr = DACreate3d(da->comm,da->wrap,da->stencil_type,M,N,P,da->m,da->n,da->p,da->w,da->s,0,0,0,&da2);CHKERRQ(ierr);
  }
  /* allow overloaded (user replaced) operations to be inherited by refinement clones */
  da2->ops->getmatrix        = da->ops->getmatrix;
  da2->ops->getinterpolation = da->ops->getinterpolation;
  da2->ops->getcoloring      = da->ops->getcoloring;
  
  /* copy fill information if given */
  if (da->dfill) {
    ierr = PetscMalloc((da->dfill[da->w]+da->w+1)*sizeof(int),&da2->dfill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->dfill,da->dfill,(da->dfill[da->w]+da->w+1)*sizeof(int));CHKERRQ(ierr);
  }
  if (da->ofill) {
    ierr = PetscMalloc((da->ofill[da->w]+da->w+1)*sizeof(int),&da2->ofill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->ofill,da->ofill,(da->ofill[da->w]+da->w+1)*sizeof(int));CHKERRQ(ierr);
  }
  *daref = da2;
  PetscFunctionReturn(0);
}

/*@C
     DASetGetMatrix - Sets the routine used by the DA to allocate a matrix.

    Collective on DA

  Input Parameters:
+    da - the DA object
-    f - the function that allocates the matrix for that specific DA

  Level: developer

   Notes: See DASetBlockFills() that provides a simple way to provide the nonzero structure for 
       the diagonal and off-diagonal blocks of the matrix

.seealso: DAGetMatrix(), DASetBlockFills()
@*/
int DASetGetMatrix(DA da,int (*f)(DA,const MatType,Mat*))
{
  PetscFunctionBegin;
  da->ops->getmatrix = f;
  PetscFunctionReturn(0);
}

/*
      M is number of grid points 
      m is number of processors

*/
#undef __FUNCT__  
#define __FUNCT__ "DASplitComm2d"
int DASplitComm2d(MPI_Comm comm,int M,int N,int sw,MPI_Comm *outcomm)
{
  int ierr,m,n = 0,csize,size,rank,x = 0,y = 0;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  csize = 4*size;
  do {
    if (csize % 4) SETERRQ4(1,"Cannot split communicator of size %d tried %d %d %d",size,csize,x,y);
    csize   = csize/4;
  
    m = (int)(0.5 + sqrt(((double)M)*((double)csize)/((double)N)));
    if (!m) m = 1;
    while (m > 0) {
      n = csize/m;
      if (m*n == csize) break;
      m--;
    }
    if (M > N && m < n) {int _m = m; m = n; n = _m;}

    x = M/m + ((M % m) > ((csize-1) % m));
    y = (N + (csize-1)/m)/n;
  } while ((x < 4 || y < 4) && csize > 1);
  if (size != csize) {
    MPI_Group entire_group,sub_group;
    int       i,*groupies;

    ierr     = MPI_Comm_group(comm,&entire_group);CHKERRQ(ierr);
    ierr = PetscMalloc(csize*sizeof(int),&groupies);CHKERRQ(ierr);
    for (i=0; i<csize; i++) {
      groupies[i] = (rank/csize)*csize + i;
    }
    ierr     = MPI_Group_incl(entire_group,csize,groupies,&sub_group);CHKERRQ(ierr);
    ierr     = PetscFree(groupies);CHKERRQ(ierr);
    ierr     = MPI_Comm_create(comm,sub_group,outcomm);CHKERRQ(ierr);
    ierr     = MPI_Group_free(&entire_group);CHKERRQ(ierr);
    ierr     = MPI_Group_free(&sub_group);CHKERRQ(ierr);
    PetscLogInfo(0,"Creating redundant coarse problems of size %d\n",csize);
  } else {
    *outcomm = comm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalFunction"
/*@C
       DASetLocalFunction - Caches in a DA a local function. 

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  lf - the local function

   Level: intermediate

   Notes: The routine SNESDAFormFunction() uses this the cached function to evaluate the user provided function.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunctioni()
@*/
int DASetLocalFunction(DA da,DALocalFunction1 lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->lf    = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalFunctioni"
/*@C
       DASetLocalFunctioni - Caches in a DA a local function that evaluates a single component

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  lfi - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction()
@*/
int DASetLocalFunctioni(DA da,int (*lfi)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->lfi = lfi;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicFunction_Private"
int DASetLocalAdicFunction_Private(DA da,DALocalFunction1 ad_lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->adic_lf = ad_lf;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicFunctioni - Caches in a DA a local functioni computed by ADIC/ADIFOR

   Collective on DA

   Synopsis:
   int int DASetLocalAdicFunctioni(DA da,int (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)
   
   Input Parameter:
+  da - initial distributed array
-  ad_lfi - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctioni()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicFunctioni_Private"
int DASetLocalAdicFunctioni_Private(DA da,int (*ad_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->adic_lfi = ad_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicMFFunctioni - Caches in a DA a local functioni computed by ADIC/ADIFOR

   Collective on DA

   Synopsis:
   int int DASetLocalAdicFunctioni(DA da,int (ad_lf*)(DALocalInfo*,MatStencil*,void*,void*,void*)
   
   Input Parameter:
+  da - initial distributed array
-  admf_lfi - the local matrix-free function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian(), DASetLocalFunctioni()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicMFFunctioni_Private"
int DASetLocalAdicMFFunctioni_Private(DA da,int (*admf_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->adicmf_lfi = admf_lfi;
  PetscFunctionReturn(0);
}

/*MC
       DASetLocalAdicMFFunction - Caches in a DA a local function computed by ADIC/ADIFOR

   Collective on DA

   Synopsis:
   int int DASetLocalAdicMFFunction(DA da,DALocalFunction1 ad_lf)
   
   Input Parameter:
+  da - initial distributed array
-  ad_lf - the local function as computed by ADIC/ADIFOR

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction(),
          DASetLocalJacobian()
M*/

#undef __FUNCT__  
#define __FUNCT__ "DASetLocalAdicMFFunction_Private"
int DASetLocalAdicMFFunction_Private(DA da,DALocalFunction1 ad_lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->adicmf_lf = ad_lf;
  PetscFunctionReturn(0);
}

/*@C
       DASetLocalJacobian - Caches in a DA a local Jacobian

   Collective on DA

   
   Input Parameter:
+  da - initial distributed array
-  lj - the local Jacobian

   Level: intermediate

   Notes: The routine SNESDAFormFunction() uses this the cached function to evaluate the user provided function.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetLocalFunction(), DASetLocalFunction()
@*/
#undef __FUNCT__  
#define __FUNCT__ "DASetLocalJacobian"
int DASetLocalJacobian(DA da,DALocalFunction1 lj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->lj    = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalFunction"
/*@C
       DAGetLocalFunction - Gets from a DA a local function and its ADIC/ADIFOR Jacobian

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameters:
.  lf - the local function

   Level: intermediate

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DASetLocalFunction()
@*/
int DAGetLocalFunction(DA da,DALocalFunction1 *lf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (lf)       *lf = da->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunction1"
/*@
    DAFormFunction1 - Evaluates a user provided function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector
.    vfu - output vector 
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
int DAFormFunction1(DA da,Vec vu,Vec vfu,void *w)
{
  int         ierr;
  void        *u,*fu;
  DALocalInfo info;
  
  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);

  ierr = (*da->lf)(&info,u,fu,w);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctioniTest1"
int DAFormFunctioniTest1(DA da,void *w)
{
  Vec         vu,fu,fui;
  int         ierr,i,n;
  PetscScalar *ui,mone = -1.0;
  PetscRandom rnd;
  PetscReal   norm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rnd);CHKERRQ(ierr);
  ierr = VecSetRandom(rnd,vu);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rnd);CHKERRQ(ierr);

  ierr = DAGetGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(da,&fui);CHKERRQ(ierr);
  
  ierr = DAFormFunction1(da,vu,fu,w);CHKERRQ(ierr);

  ierr = VecGetArray(fui,&ui);CHKERRQ(ierr);
  ierr = VecGetLocalSize(fui,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = DAFormFunctioni1(da,i,vu,ui+i,w);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(fui,&ui);CHKERRQ(ierr);

  ierr = VecAXPY(&mone,fu,fui);CHKERRQ(ierr);
  ierr = VecNorm(fui,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(da->comm,"Norm of difference in vectors %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(fu,0);CHKERRQ(ierr);
  ierr = VecView(fui,0);CHKERRQ(ierr);

  ierr = DARestoreLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&fu);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&fui);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNCT__
#define __FUNCT__ "DAFormFunctioni1"
/*@
    DAFormFunctioni1 - Evaluates a user provided function

   Input Parameters:
+    da - the DA that defines the grid
.    i - the component of the function we wish to compute (must be local)
.    vu - input vector
.    vfu - output value
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAComputeJacobian1WithAdic()

@*/
int DAFormFunctioni1(DA da,int i,Vec vu,PetscScalar *vfu,void *w)
{
  int         ierr;
  void        *u;
  DALocalInfo info;
  MatStencil  stencil;
  
  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);

  /* figure out stencil value from i */
  stencil.c = i % info.dof;
  stencil.i = (i % (info.xm*info.dof))/info.dof;
  stencil.j = (i % (info.xm*info.ym*info.dof))/(info.xm*info.dof);
  stencil.k = i/(info.xm*info.ym*info.dof);

  ierr = (*da->lfi)(&info,&stencil,u,vfu,w);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(new)
#undef __FUNCT__  
#define __FUNCT__ "DAGetDiagonal_MFFD"
/*
  DAGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix where local
    function lives on a DA

        y ~= (F(u + ha) - F(u))/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int DAGetDiagonal_MFFD(DA da,Vec U,Vec a)
{
  PetscScalar  h,*aa,*ww,v;
  PetscReal    epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  int          ierr,gI,nI;
  MatStencil   stencil;
  DALocalInfo  info;
 
  PetscFunctionBegin;
  ierr = (*ctx->func)(0,U,a,ctx->funcctx);CHKERRQ(ierr);
  ierr = (*ctx->funcisetbase)(U,ctx->funcctx);CHKERRQ(ierr);

  ierr = VecGetArray(U,&ww);CHKERRQ(ierr);
  ierr = VecGetArray(a,&aa);CHKERRQ(ierr);
  
  nI = 0;
    h  = ww[gI];
    if (h == 0.0) h = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    if (h < umin && h >= 0.0)      h = umin;
    else if (h < 0.0 && h > -umin) h = -umin;
#else
    if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0)     h = umin;
    else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
#endif
    h     *= epsilon;
    
    ww[gI += h;
    ierr          = (*ctx->funci)(i,w,&v,ctx->funcctx);CHKERRQ(ierr);
    aa[nI]  = (v - aa[nI])/h;
    ww[gI] -= h;
    nI++;
  }
  ierr = VecRestoreArray(U,&ww);CHKERRQ(ierr);
  ierr = VecRestoreArray(a,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1WithAdic"
/*@
    DAComputeJacobian1WithAdic - Evaluates a adiC provided Jacobian function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

   Level: advanced

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DAFormFunction1()

@*/
int DAComputeJacobian1WithAdic(DA da,Vec vu,Mat J,void *w)
{
  int         ierr,gtdof,tdof;
  PetscScalar *ustart;
  DALocalInfo info;
  void        *ad_u,*ad_f,*ad_ustart,*ad_fstart;
  ISColoring  iscoloring;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscADResetIndep();

  /* get space for derivative objects.  */
  ierr = DAGetAdicArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DAGetAdicArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  ierr = VecGetArray(vu,&ustart);CHKERRQ(ierr);
  ierr = DAGetColoring(da,IS_COLORING_GHOSTED,&iscoloring);CHKERRQ(ierr);

  PetscADSetValueAndColor(ad_ustart,gtdof,iscoloring->colors,ustart);

  ierr = VecRestoreArray(vu,&ustart);CHKERRQ(ierr);
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  ierr = PetscADIncrementTotalGradSize(iscoloring->n);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DALogEventBegin(DA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);
  ierr = (*da->adic_lf)(&info,ad_u,ad_f,w);CHKERRQ(ierr);
  ierr = DALogEventEnd(DA_LocalADFunction,0,0,0,0);CHKERRQ(ierr);

  /* stick the values into the matrix */
  ierr = MatSetValuesAdic(J,(PetscScalar**)ad_fstart);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = DARestoreAdicArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DARestoreAdicArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAdic"
/*@C
    DAMultiplyByJacobian1WithAdic - Applies an ADIC-provided Jacobian function to a vector on 
    each processor that shares a DA.

    Input Parameters:
+   da - the DA that defines the grid
.   vu - Jacobian is computed at this point (ghosted)
.   v - product is done on this vector (ghosted)
.   fu - output vector = J(vu)*v (not ghosted)
-   w - any user data

    Notes: 
    This routine does NOT do ghost updates on vu upon entry.

   Level: advanced

.seealso: DAFormFunction1()

@*/
int DAMultiplyByJacobian1WithAdic(DA da,Vec vu,Vec v,Vec f,void *w)
{
  int         ierr,i,gtdof,tdof;
  PetscScalar *avu,*av,*af,*ad_vustart,*ad_fstart;
  DALocalInfo info;
  void        *ad_vu,*ad_f;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* get space for derivative objects.  */
  ierr = DAGetAdicMFArray(da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DAGetAdicMFArray(da,PETSC_FALSE,(void **)&ad_f,(void**)&ad_fstart,&tdof);CHKERRQ(ierr);

  /* copy input vector into derivative object */
  ierr = VecGetArray(vu,&avu);CHKERRQ(ierr);
  ierr = VecGetArray(v,&av);CHKERRQ(ierr);
  for (i=0; i<gtdof; i++) {
    ad_vustart[2*i]   = avu[i];
    ad_vustart[2*i+1] = av[i];
  }
  ierr = VecRestoreArray(vu,&avu);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&av);CHKERRQ(ierr);

  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(1);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = (*da->adicmf_lf)(&info,ad_vu,ad_f,w);CHKERRQ(ierr);

  /* stick the values into the vector */
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);  
  for (i=0; i<tdof; i++) {
    af[i] = ad_fstart[2*i+1];
  }
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);  

  /* return space for derivative objects.  */
  ierr = DARestoreAdicMFArray(da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(da,PETSC_FALSE,(void **)&ad_f,(void**)&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#else

#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1WithAdic"
int DAComputeJacobian1WithAdic(DA da,Vec vu,Mat J,void *w)
{
  PetscFunctionBegin;
  SETERRQ(1,"Must compile with bmake/PETSC_ARCH/packages flag PETSC_HAVE_ADIC for this routine");
}

#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAdic"
int DAMultiplyByJacobian1WithAdic(DA da,Vec vu,Vec v,Vec f,void *w)
{
  PetscFunctionBegin;
  SETERRQ(1,"Must compile with bmake/PETSC_ARCH/packages flag PETSC_HAVE_ADIC for this routine");
}

#endif

#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1"
/*@
    DAComputeJacobian1 - Evaluates a local Jacobian function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

    Level: advanced

.seealso: DAFormFunction1()

@*/
int DAComputeJacobian1(DA da,Vec vu,Mat J,void *w)
{
  int         ierr;
  void        *u;
  DALocalInfo info;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = (*da->lj)(&info,u,J,w);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DAComputeJacobian1WithAdifor"
/*
    DAComputeJacobian1WithAdifor - Evaluates a ADIFOR provided Jacobian local function on each processor that 
        share a DA

   Input Parameters:
+    da - the DA that defines the grid
.    vu - input vector (ghosted)
.    J - output matrix
-    w - any user data

    Notes: Does NOT do ghost updates on vu upon entry

.seealso: DAFormFunction1()

*/
int DAComputeJacobian1WithAdifor(DA da,Vec vu,Mat J,void *w)
{
  int             i,ierr,Nc,N;
  ISColoringValue *color;
  DALocalInfo     info;
  PetscScalar     *u,*g_u,*g_f,*f,*p_u;
  ISColoring      iscoloring;
  void            (*lf)(int *,DALocalInfo*,PetscScalar*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,void*,int*) = 
                  (void (*)(int *,DALocalInfo*,PetscScalar*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,void*,int*))*da->adifor_lf;

  PetscFunctionBegin;
  ierr = DAGetColoring(da,IS_COLORING_GHOSTED,&iscoloring);CHKERRQ(ierr);
  Nc   = iscoloring->n;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  N    = info.gxm*info.gym*info.gzm*info.dof;

  /* get space for derivative objects.  */
  ierr  = PetscMalloc(Nc*info.gxm*info.gym*info.gzm*info.dof*sizeof(PetscScalar),&g_u);CHKERRQ(ierr);
  ierr  = PetscMemzero(g_u,Nc*info.gxm*info.gym*info.gzm*info.dof*sizeof(PetscScalar));CHKERRQ(ierr);
  p_u   = g_u;
  color = iscoloring->colors;
  for (i=0; i<N; i++) {
    p_u[*color++] = 1.0;
    p_u          += Nc;
  }
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  ierr = PetscMalloc(Nc*info.xm*info.ym*info.zm*info.dof*sizeof(PetscScalar),&g_f);CHKERRQ(ierr);
  ierr = PetscMalloc(info.xm*info.ym*info.zm*info.dof*sizeof(PetscScalar),&f);CHKERRQ(ierr);

  /* Seed the input array g_u with coloring information */
 
  ierr = VecGetArray(vu,&u);CHKERRQ(ierr);
  (lf)(&Nc,&info,u,g_u,&Nc,f,g_f,&Nc,w,&ierr);CHKERRQ(ierr);
  ierr = VecRestoreArray(vu,&u);CHKERRQ(ierr);

  /* stick the values into the matrix */
  /* PetscScalarView(Nc*info.xm*info.ym,g_f,0); */
  ierr = MatSetValuesAdifor(J,Nc,g_f);CHKERRQ(ierr); 

  /* return space for derivative objects.  */
  ierr = PetscFree(g_u);CHKERRQ(ierr);
  ierr = PetscFree(g_f);CHKERRQ(ierr);
  ierr = PetscFree(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAD"
/*@C
    DAMultiplyByJacobian1WithAD - Applies a Jacobian function supplied by ADIFOR or ADIC
    to a vector on each processor that shares a DA.

   Input Parameters:
+    da - the DA that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes: 
    This routine does NOT do ghost updates on vu and v upon entry.
           
    Automatically calls DAMultiplyByJacobian1WithAdifor() or DAMultiplyByJacobian1WithAdic()
    depending on whether DASetLocalAdicMFFunction() or DASetLocalAdiforMFFunction() was called.

   Level: advanced

.seealso: DAFormFunction1(), DAMultiplyByJacobian1WithAdifor(), DAMultiplyByJacobian1WithAdic()

@*/
int DAMultiplyByJacobian1WithAD(DA da,Vec u,Vec v,Vec f,void *w)
{
  int         ierr;

  PetscFunctionBegin;
  if (da->adicmf_lf) {
#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
    ierr = DAMultiplyByJacobian1WithAdic(da,u,v,f,w);CHKERRQ(ierr);
#else
    SETERRQ(1,"Requires ADIC to be installed and cannot use complex numbers");
#endif
  } else if (da->adiformf_lf) {
    ierr = DAMultiplyByJacobian1WithAdifor(da,u,v,f,w);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Must call DASetLocalAdiforMFFunction() or DASetLocalAdicMFFunction() before using");
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DAMultiplyByJacobian1WithAdifor"
/*@C
    DAMultiplyByJacobian1WithAdifor - Applies a ADIFOR provided Jacobian function on each processor that 
        share a DA to a vector

   Input Parameters:
+    da - the DA that defines the grid
.    vu - Jacobian is computed at this point (ghosted)
.    v - product is done on this vector (ghosted)
.    fu - output vector = J(vu)*v (not ghosted)
-    w - any user data

    Notes: Does NOT do ghost updates on vu and v upon entry

   Level: advanced

.seealso: DAFormFunction1()

@*/
int DAMultiplyByJacobian1WithAdifor(DA da,Vec u,Vec v,Vec f,void *w)
{
  int         ierr;
  PetscScalar *au,*av,*af,*awork;
  Vec         work;
  DALocalInfo info;
  void        (*lf)(DALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,int*) = 
              (void (*)(DALocalInfo*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*,int*))*da->adiformf_lf;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DAGetGlobalVector(da,&work);CHKERRQ(ierr); 
  ierr = VecGetArray(u,&au);CHKERRQ(ierr);
  ierr = VecGetArray(v,&av);CHKERRQ(ierr);
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);
  ierr = VecGetArray(work,&awork);CHKERRQ(ierr);
  (lf)(&info,au,av,awork,af,w,&ierr);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&au);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&av);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);
  ierr = VecRestoreArray(work,&awork);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(da,&work);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetInterpolationType"
/*@C
       DASetInterpolationType - Sets the type of interpolation that will be 
          returned by DAGetInterpolation()

   Collective on DA

   Input Parameter:
+  da - initial distributed array
.  ctype - DA_Q1 and DA_Q0 are currently the only supported forms

   Level: intermediate

   Notes: you should call this on the coarser of the two DAs you pass to DAGetInterpolation()

.keywords:  distributed array, interpolation

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DA, DAInterpolationType
@*/
int DASetInterpolationType(DA da,DAInterpolationType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  da->interptype = ctype;
  PetscFunctionReturn(0);
}

