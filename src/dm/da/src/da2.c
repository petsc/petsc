#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.48 1996/05/07 19:21:43 curfman Exp curfman $";
#endif
 
#include "daimpl.h"    /*I   "da.h"   I*/
#include "pinclude/pviewer.h"
#include "draw.h"
#include <math.h>

static int DAView_2d(PetscObject dain,Viewer viewer)
{
  DA          da = (DA) dain;
  int         rank, ierr;
  ViewerType  vtype;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  
  MPI_Comm_rank(da->comm,&rank); 

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF;
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);

  if (vtype == ASCII_FILE_VIEWER) {
    FILE *fd;
    ierr = ViewerASCIIGetPointer(viewer,&fd);  CHKERRQ(ierr);
    PetscSequentialPhaseBegin(da->comm,1);
    fprintf(fd,"Processor [%d] M %d N %d m %d n %d w %d s %d\n",rank,da->M,
                 da->N,da->m,da->n,da->w,da->s);
    fprintf(fd,"X range: %d %d, Y range: %d %d\n",da->xs,da->xe,da->ys,da->ye);
    fflush(fd);
    PetscSequentialPhaseEnd(da->comm,1);
  }
  else if (vtype == DRAW_VIEWER) {
    Draw       draw;
    double     ymin = -1*da->s-1, ymax = da->N+da->s;
    double     xmin = -1*da->s-1, xmax = da->M+da->s;
    double     x,y;
    int        base,*idx;
    char       node[10];
    PetscTruth isnull;
 
    ViewerDrawGetDraw(viewer,&draw);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
    DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);

    /* first processor draw all node lines */
    if (!rank) {
      ymin = 0.0; ymax = da->N - 1;
      for ( xmin=0; xmin<da->M; xmin++ ) {
        DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_BLACK);
      }
      xmin = 0.0; xmax = da->M - 1;
      for ( ymin=0; ymin<da->N; ymin++ ) {
        DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_BLACK);
      }
    }
    DrawSyncFlush(draw);
    DrawPause(draw);
    MPI_Barrier(da->comm);

    /* draw my box */
    ymin = da->ys; ymax = da->ye - 1; xmin = da->xs/da->w; 
    xmax =(da->xe-1)/da->w;
    DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_RED);
    DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_RED);
    DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_RED);
    DrawLine(draw,xmax,ymin,xmax,ymax,DRAW_RED);

    /* put in numbers */
    base = (da->base)/da->w;
    for ( y=ymin; y<=ymax; y++ ) {
      for ( x=xmin; x<=xmax; x++ ) {
        sprintf(node,"%d",base++);
        DrawText(draw,x,y,DRAW_BLACK,node);
      }
    }

    DrawSyncFlush(draw);
    DrawPause(draw);
    MPI_Barrier(da->comm);
    /* overlay ghost numbers, useful for error checking */
    /* put in numbers */

    base = 0; idx = da->idx;
    ymin = da->Ys; ymax = da->Ye; xmin = da->Xs; xmax = da->Xe;
    for ( y=ymin; y<ymax; y++ ) {
      for ( x=xmin; x<xmax; x++ ) {
        if ((base % da->w) == 0) {
          sprintf(node,"%d",idx[base]/da->w);
          DrawText(draw,x/da->w,y,DRAW_BLUE,node);
        }
        base++;
      }
    }        
    DrawSyncFlush(draw);
    DrawPause(draw);
  }
  return 0;
}

/*@C
    DACreate2d - Creates a two-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  comm - MPI communicator
.  wrap - type of periodicity should the array have, if any
$         DA_NONPERIODIC, DA_XPERIODIC, 
$         DA_YPERIODIC, DA_XYPERIODIC  
.  stencil_type - stencil type either DA_STENCIL_BOX or DA_STENCIL_STAR
.  M,N - global dimension in each direction of the array
.  m,n - corresponding number of processors in each dimension 
         (or PETSC_DECIDE to have calculated)
.  w - number of degress of freedom per node
.  s - stencil width

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
$  -da_view : call DAView() at the conclusion of DACreate2d()

   Notes:
   The stencil type DA_STENCIL_STAR with width 1 corresponds to the 
   standard 5-pt stencil, while DA_STENCIL_BOX with width 1 denotes
   the standard 9-pt stencil.

.keywords: distributed array, create, two-dimensional

.seealso: DADestroy(), DAView(), DACreate1d(), DACreate3d()
@*/
int DACreate2d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type,
                int M,int N,int m,int n,int w,int s,DA *inra)
{
  int           rank, size,xs,xe,ys,ye,x,y,Xs,Xe,Ys,Ye,ierr,start,end;
  int           up,down,left,i,n0,n1,n2,n3,n5,n6,n7,n8,*idx,nn;
  int           xbase,*bases,*ldims,j,x_t,y_t,s_t,base,count,flg;
  int           s_x,s_y; /* s proportionalized to w */
  int           *gA,*gB,*gAall,*gBall,ict,ldim,gdim;
  DA            da;
  Vec           local,global;
  VecScatter    ltog,gtol;
  IS            to,from;
  DF            df_local;
  *inra = 0;

  PetscHeaderCreate(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  PLogObjectMemory(da,sizeof(struct _DA));
  da->dim = 2;

  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank); 

  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int) sqrt( ((double)M)*((double)size)/((double)N) );
    if (m == 0) m = 1;
    while (m > 0) {
      n = size/m;
      if (m*n == size) break;
      m--;
    }
    if (M > N && m < n) {int _m = m; m = n; n = _m;}
    if (m*n != size)SETERRQ(1,"DaCreate2d:Internally Created Bad Partition");
  }
  else if (m*n != size) SETERRQ(1,"DACreate2d:Given Bad partition"); 

  if (M < m) SETERRQ(1,"DACreate2d:Partition in x direction is too fine!");
  if (N < n) SETERRQ(1,"DACreate2d:Partition in y direction is too fine!");

  /* determine locally owned region */
  x = M/m + ((M % m) > (rank % m));
  y = N/n + ((N % n) > (rank/m));

  if (x < s) SETERRQ(1,"DACreate2d:Column width is too thin for stencil!");
  if (y < s) SETERRQ(1,"DACreate2d:Row width is too thin for stencil!");
  if ((M % m) > (rank % m)) { xs = (rank % m)*x; }
  else { xs = (M % m)*(x+1) + ((rank % m)-(M % m))*x; }
  xe = xs + x;
  if ((N % n) > (rank/m)) { ys = (rank/m)*y; }
  else { ys = (N % n)*(y+1) + ((rank/m)-(N % n))*y; }
  ye = ys + y;

  /* determine ghost region */
  /* Assume No Periodicity */
  if (xs-s > 0) Xs = xs - s; else Xs = 0; 
  if (ys-s > 0) Ys = ys - s; else Ys = 0; 
  if (xe+s <= M) Xe = xe + s; else Xe = M; 
  if (ye+s <= N) Ye = ye + s; else Ye = N;

  /* X Periodic */
  if (wrap == DA_XPERIODIC || wrap ==  DA_XYPERIODIC) {
    Xs = xs - s; 
    Xe = xe + s; 
  }

  /* Y Periodic */
  if (wrap == DA_YPERIODIC || wrap ==  DA_XYPERIODIC) {
    Ys = ys - s;
    Ye = ye + s;
  }

  /* Resize all X parameters to reflect w */
  x   *= w;
  xs  *= w;
  xe  *= w;
  Xs  *= w;
  Xe  *= w;
  s_x = s*w;
  s_y = s;

  /* determine starting point of each processor */
  nn = x*y;
  bases = (int *) PetscMalloc( (2*size+1)*sizeof(int) ); CHKPTRQ(bases);
  ldims = (int *) (bases+size+1);
  MPI_Allgather(&nn,1,MPI_INT,ldims,1,MPI_INT,comm);
  bases[0] = 0;
  for ( i=1; i<=size; i++ ) {
    bases[i] = ldims[i-1];
  }
  for ( i=1; i<=size; i++ ) {
    bases[i] += bases[i-1];
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x*y,PETSC_DECIDE,&global); CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys),&local);CHKERRQ(ierr);

  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  ierr = VecGetOwnershipRange(global,&start,&end); CHKERRQ(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,x*y,start,1,&to);CHKERRQ(ierr);

  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  idx = (int *) PetscMalloc( x*(up - down)*sizeof(int) ); CHKPTRQ(idx);
  count = 0;
  for ( i=down; i<up; i++ ) {
    for ( j=0; j<x; j++ ) {
      idx[count++] = left + i*(Xe-Xs) + j;
    }
  }
  ierr = ISCreateSeq(MPI_COMM_SELF,count,idx,&from);CHKERRQ(ierr);
  PetscFree(idx);

  ierr = VecScatterCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX) {
    ierr = ISCreateStrideSeq(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys),0,1,&to);CHKERRQ(ierr); 
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
    idx   = (int *) PetscMalloc( count*sizeof(int) ); CHKPTRQ(idx);
    count = 0;
    for ( i=0; i<down; i++ ) {
      for ( j=0; j<xe-xs; j++ ) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    /* middle */
    for ( i=down; i<up; i++ ) {
      for ( j=0; j<Xe-Xs; j++ ) {
        idx[count++] = i*(Xe-Xs) + j;
      }
    }
    /* top */
    for ( i=up; i<Ye-Ys; i++ ) {
      for ( j=0; j<xe-xs; j++ ) {
        idx[count++] = left + i*(Xe-Xs) + j;
      }
    }
    ierr = ISCreateSeq(MPI_COMM_SELF,count,idx,&to);CHKERRQ(ierr);
    PetscFree(idx);
  }


  /* determine who lies on each side of use stored in    n6 n7 n8
                                                         n3    n5
                                                         n0 n1 n2
  */

  /* Assume the Non-Periodic Case */
  n1 = rank - m; 
  if (rank % m) {
    n0 = n1 - 1; 
  }
  else {
    n0 = -1;
  }
  if ((rank+1) % m) {
    n2 = n1 + 1;
    n5 = rank + 1;
    n8 = rank + m + 1; if (n8 >= m*n) n8 = -1;
  }
  else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (rank % m) {
    n3 = rank - 1; 
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  }
  else {
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
  } 
  else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }
  else if (wrap == DA_XYPERIODIC) {

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

  if (stencil_type == DA_STENCIL_STAR) {n0 = n2 = n6 = n8 = -1;}

  idx = (int *)PetscMalloc((x+2*s_x)*(y+2*s_y)*sizeof(int));CHKPTRQ(idx);
  PLogObjectMemory(da,(x+2*s_x)*(y+2*s_y)*sizeof(int));
  nn = 0;

  xbase = bases[rank];
  for ( i=1; i<=s_y; i++ ) {
    if (n0 >= 0) { /* left below */
      x_t = (M/m + ((M % m) > (n0 % m)))*w;
      y_t = N/n + ((N % n) > (n0/m));
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = N/n + ((N % n) > (n1/m));
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = (M/m + ((M % m) > (n2 % m)))*w;
      y_t = N/n + ((N % n) > (n2/m));
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=0; i<y; i++ ) {
    if (n3 >= 0) { /* directly left */
      x_t = (M/m + ((M % m) > (n3 % m)))*w;
      y_t = y;
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }

    for ( j=0; j<x; j++ ) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = (M/m + ((M % m) > (n5 % m)))*w;
      y_t = y;
      s_t = bases[n5] + (i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=1; i<=s_y; i++ ) {
    if (n6 >= 0) { /* left above */
      x_t = (M/m + ((M % m) > (n6 % m)))*w;
      y_t = N/n + ((N % n) > (n6/m));
      s_t = bases[n6] + (i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      y_t = N/n + ((N % n) > (n7/m));
      s_t = bases[n7] + (i-1)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = (M/m + ((M % m) > (n8 % m)))*w;
      y_t = N/n + ((N % n) > (n8/m));
      s_t = bases[n8] + (i-1)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  base = bases[rank];
  ierr = ISCreateSeq(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,gtol);
  ISDestroy(to); ISDestroy(from);

  da->M  = M;  da->N  = N;  da->m  = m;  da->n  = n;  da->w = w;  da->s = s;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye; da->zs = 0; da->ze = 0;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye; da->Zs = 0; da->Ze = 0;
  da->P  = 1;  da->p  = 1;

  PLogObjectParent(da,global);
  PLogObjectParent(da,local);

  da->global       = global; 
  da->local        = local; 
  da->gtol         = gtol;
  da->ltog         = ltog;
  da->idx          = idx;
  da->Nl           = nn;
  da->base         = base;
  da->wrap         = wrap;
  da->view         = DAView_2d;
  da->stencil_type = stencil_type;
  *inra = da;

  /* recalculate the idx including missed ghost points */
  /* Assume the Non-Periodic Case */
  n1 = rank - m; 
  if (rank % m) {
    n0 = n1 - 1; 
  }
  else {
    n0 = -1;
  }
  if ((rank+1) % m) {
    n2 = n1 + 1;
    n5 = rank + 1;
    n8 = rank + m + 1; if (n8 >= m*n) n8 = -1;
  }
  else {
    n2 = -1; n5 = -1; n8 = -1;
  }
  if (rank % m) {
    n3 = rank - 1; 
    n6 = n3 + m; if (n6 >= m*n) n6 = -1;
  }
  else {
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
  } 
  else if (wrap == DA_XPERIODIC) { /* Handle Left and Right Sides */
    if (n3 < 0) n3 = rank + (m-1);
    if (n5 < 0) n5 = rank - (m-1);
    if ((n1 >= 0) && (n0 < 0)) n0 = rank-1;
    if ((n1 >= 0) && (n2 < 0)) n2 = rank-2*m+1;
    if ((n7 >= 0) && (n6 < 0)) n6 = rank+2*m-1;
    if ((n7 >= 0) && (n8 < 0)) n8 = rank+1;
  }
  else if (wrap == DA_XYPERIODIC) {

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

  nn = 0;

  xbase = bases[rank];
  for ( i=1; i<=s_y; i++ ) {
    if (n0 >= 0) { /* left below */
      x_t = (M/m + ((M % m) > (n0 % m)))*w;
      y_t = N/n + ((N % n) > (n0/m));
      s_t = bases[n0] + x_t*y_t - (s_y-i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n1 >= 0) { /* directly below */
      x_t = x;
      y_t = N/n + ((N % n) > (n1/m));
      s_t = bases[n1] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n2 >= 0) { /* right below */
      x_t = (M/m + ((M % m) > (n2 % m)))*w;
      y_t = N/n + ((N % n) > (n2/m));
      s_t = bases[n2] + x_t*y_t - (s_y+1-i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=0; i<y; i++ ) {
    if (n3 >= 0) { /* directly left */
      x_t = (M/m + ((M % m) > (n3 % m)))*w;
      y_t = y;
      s_t = bases[n3] + (i+1)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }

    for ( j=0; j<x; j++ ) { idx[nn++] = xbase++; } /* interior */

    if (n5 >= 0) { /* directly right */
      x_t = (M/m + ((M % m) > (n5 % m)))*w;
      y_t = y;
      s_t = bases[n5] + (i)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }

  for ( i=1; i<=s_y; i++ ) {
    if (n6 >= 0) { /* left above */
      x_t = (M/m + ((M % m) > (n6 % m)))*w;
      y_t = N/n + ((N % n) > (n6/m));
      s_t = bases[n6] + (i)*x_t - s_x;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
    if (n7 >= 0) { /* directly above */
      x_t = x;
      y_t = N/n + ((N % n) > (n7/m));
      s_t = bases[n7] + (i-1)*x_t;
      for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
    }
    if (n8 >= 0) { /* right above */
      x_t = (M/m + ((M % m) > (n8 % m)))*w;
      y_t = N/n + ((N % n) > (n8/m));
      s_t = bases[n8] + (i-1)*x_t;
      for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
    }
  }
  /* keep bases for use at end of routine */
  /* PetscFree(bases); */

  /* construct the local to local scatter context */
  /* 
      We simply remap the values in the from part of 
    global to local to read from an array with the ghost values 
    rather then from the plan array.
  */
  ierr = VecScatterCopy(gtol,&da->ltol); CHKERRQ(ierr);
  PLogObjectParent(da,da->ltol);
  left  = xs - Xs; down  = ys - Ys; up    = down + y;
  idx = (int *) PetscMalloc( x*(up - down)*sizeof(int) ); CHKPTRQ(idx);
  count = 0;
  for ( i=down; i<up; i++ ) {
    for ( j=0; j<x; j++ ) {
      idx[count++] = left + i*(Xe-Xs) + j;
    }
  }
  ierr = VecScatterRemap(da->ltol,idx,PETSC_NULL); CHKERRQ(ierr); 
  PetscFree(idx);

  /* Construct the mapping from current global ordering to global
     ordering that would be used if only 1 processor were employed.
     This mapping is intended only for internal use by discrete
     function and matrix viewers.

     Note: At this point, x has already been adjusted for multiple
     degrees of freedom per node.
   */
  ldim = x*y;
  ierr = VecGetSize(global,&gdim); CHKERRQ(ierr);
  da->gtog1 = (int *)PetscMalloc(gdim*sizeof(int)); CHKPTRQ(da->gtog1);
  gA        = (int *)PetscMalloc((2*(gdim+ldim))*sizeof(int)); CHKPTRQ(gA);
  gB        = (int *)(gA + ldim);
  gAall     = (int *)(gB + ldim);
  gBall     = (int *)(gAall + gdim);
  /* Compute local parts of global orderings */
  ict = 0;
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
      /* gA = global number for 1 proc; gB = current global number */
      gA[ict] = i + j*M*w;
      gB[ict] = start + ict;
      ict++;
    }
  }
  /* Broadcast the orderings */
  MPI_Allgatherv(gA,ldim,MPI_INT,gAall,ldims,bases,MPI_INT,comm);
  MPI_Allgatherv(gB,ldim,MPI_INT,gBall,ldims,bases,MPI_INT,comm);
  for (i=0; i<gdim; i++) da->gtog1[gBall[i]] = gAall[i];
  PetscFree(gA); PetscFree(bases);

  /* Create discrete function shell and associate with vectors in DA */
  /* Eventually will pass in optional labels for each component */
  ierr = DFShellCreateDA_Private(comm,PETSC_NULL,da,&da->dfshell); CHKERRQ(ierr);
  PLogObjectParent(da,da->dfshell);
  ierr = DFShellGetLocalDFShell(da->dfshell,&df_local);
  ierr = DFVecShellAssociate(da->dfshell,global); CHKERRQ(ierr);
  ierr = DFVecShellAssociate(df_local,local); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-da_view",&flg); CHKERRQ(ierr);
  if (flg) {ierr = DAView(da,STDOUT_VIEWER_SELF); CHKERRQ(ierr);}
  return 0;
}

/*@
   DARefine - Creates a new distributed array that is a refinement of a given
   distributed array.

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  daref - refined distributed array

   Note:
   Currently, refinement consists of just doubling the number of grid spaces
   in each dimension of the DA.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy()
@*/
int DARefine(DA da, DA *daref)
{
  int M, N, P, ierr;
  DA  da2;

  PetscValidHeaderSpecific(da,DA_COOKIE);

  M = 2*da->M - 1; N = 2*da->N - 1; P = 2*da->P - 1;
  if (da->dim == 1) {
    ierr = DACreate1d(da->comm,da->wrap,M,da->w,da->s,&da2); CHKERRQ(ierr);
  }
  else if (da->dim == 2) {
    ierr = DACreate2d(da->comm,da->wrap,da->stencil_type,M,N,da->m,da->n,da->w,da->s,&da2); CHKERRQ(ierr);
  }
  else if (da->dim == 3) {
    ierr = DACreate3d(da->comm,da->wrap,da->stencil_type,M,N,P,da->m,da->n,da->p,
           da->w,da->s,&da2); CHKERRQ(ierr);
  }
  *daref = da2;
  return 0;
}

 

