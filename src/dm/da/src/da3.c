#ifndef lint
static char vcid[] = "$Id: da3.c,v 1.34 1996/05/19 15:54:21 curfman Exp balay $";
#endif

/*
   Code for manipulating distributed regular 3d arrays in parallel.
   File created by Peter Mell  7/14/95
 */

#include "daimpl.h"     /*I   "da.h"    I*/
#include "pinclude/pviewer.h"
#include <math.h>
#include "draw.h"

int DAView_3d(PetscObject dain,Viewer viewer)
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
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);

    PetscSequentialPhaseBegin(da->comm,1);
    fprintf(fd,"Processor [%d] M %d N %d P %d m %d n %d p %d w %d s %d\n",
               rank,da->M,da->N,da->P,da->m,da->n,da->p,da->w,da->s);
    fprintf(fd,"X range: %d %d, Y: range %d %d, Z range: %d %d\n",
               da->xs,da->xe,da->ys,da->ye,da->zs,da->ze);
    fflush(fd);
    PetscSequentialPhaseEnd(da->comm,1);
  }
  else if (vtype == DRAW_VIEWER) {
    Draw       draw;
    double     ymin = -1.0,ymax = (double) da->N;
    double     xmin = -1.0,xmax = (double) ((da->M+2)*da->P),x,y;
    int        k, plane;
    double     ycoord, xcoord;
    int        base,*idx;
    char       node[10];
    PetscTruth isnull;

    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;

    DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);

    /* first processor draw all node lines */
    if (!rank) {
      for (k=0; k<da->P; k++) {
        ymin = 0.0; ymax = (double) (da->N - 1);
        for ( xmin=(double)(k*(da->M+1)); xmin<(double)(da->M+(k*(da->M+1))); xmin++ ) {
          DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_BLACK);
        }
      
        xmin = (double)(k*(da->M+1)); xmax = xmin + (double)(da->M - 1);
        for ( ymin=0; ymin<(double)da->N; ymin++ ) {
          DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_BLACK);
        }
      }
    }
    DrawSyncFlush(draw);
    DrawPause(draw);
    MPI_Barrier(da->comm);


    for (k=0; k<da->P; k++) {  /*Go through and draw for each plane*/
      if ((k >= da->zs) && (k < da->ze)) {
        /* draw my box */
        ymin = da->ys;       
        ymax = da->ye - 1; 
        xmin = da->xs/da->w    + (da->M+1)*k; 
        xmax =(da->xe-1)/da->w + (da->M+1)*k;

        DrawLine(draw,xmin,ymin,xmax,ymin,DRAW_RED);
        DrawLine(draw,xmin,ymin,xmin,ymax,DRAW_RED);
        DrawLine(draw,xmin,ymax,xmax,ymax,DRAW_RED);
        DrawLine(draw,xmax,ymin,xmax,ymax,DRAW_RED);  

        xmin = da->xs/da->w; 
        xmax =(da->xe-1)/da->w;

        /* put in numbers*/
        base = (da->base+(da->xe-da->xs)*(da->ye-da->ys)*(k-da->zs))/da->w;

        /* Identify which processor owns the box */
        sprintf(node,"%d",rank);
        DrawText(draw,xmin+(da->M+1)*k+.2,ymin+.3,DRAW_RED,node);

        for ( y=ymin; y<=ymax; y++ ) {
          for ( x=xmin+(da->M+1)*k; x<=xmax+(da->M+1)*k; x++ ) {
            sprintf(node,"%d",base++);
            DrawText(draw,x,y,DRAW_BLACK,node);
          }
        } 
 
      }
    } 
    DrawSyncFlush(draw);
    DrawPause(draw);
    MPI_Barrier(da->comm);
    for (k=0-da->s; k<da->P+da->s; k++) {  
      /* Go through and draw for each plane */
      if ((k >= da->Zs) && (k < da->Ze)) {
  
        /* overlay ghost numbers, useful for error checking */
        base = (da->Xe-da->Xs)*(da->Ye-da->Ys)*(k-da->Zs); idx = da->idx;
        plane=k;  
        /* Keep z wrap around points on the dradrawg */
        if (k<0)    { plane=da->P+k; }  
        if (k>=da->P) { plane=k-da->P; }
        ymin = da->Ys; ymax = da->Ye; 
        xmin = (da->M+1)*plane*da->w; 
        xmax = (da->M+1)*plane*da->w+da->M*da->w;
        for ( y=ymin; y<ymax; y++ ) {
          for ( x=xmin+da->Xs; x<xmin+da->Xe; x+=da->w) {
            sprintf(node,"%d",idx[base]/da->w);
            ycoord = y;
            /*Keep y wrap around points on drawing */  
            if (y<0)      { ycoord = da->N+y; } 

            if (y>=da->N) { ycoord = y-da->N; }
            xcoord = x;   /* Keep x wrap points on drawing */          

            if (x<xmin)  { xcoord = xmax - (xmin-x); }
            if (x>=xmax) { xcoord = xmin + (x-xmax); }
            DrawText(draw,xcoord/da->w,ycoord,DRAW_BLUE,node);
            base+=da->w;
          }
        }
      }         
    } 
    DrawSyncFlush(draw);
    DrawPause(draw);
  }
  return 0;
}

/*@C
    DACreate3d - Creates a three-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  comm - MPI communicator
.  wrap - type of periodicity the array should have, if any. 
$      DA_NONPERIODIC, DA_XPERIODIC, 
$      DA_YPERIODIC, DA_XYPERIODIC
$      DA_XYZPERIODIC, DA_XZPERIODIC, 
$      DA_YZPERIODIC
.  stencil_type - Type of stencil (DA_STENCIL_STAR or DA_STENCIL_BOX)
.  M,N,P - global dimension in each direction of the array
.  m,n,p - corresponding number of processors in each dimension 
           (or PETSC_DECIDE to have calculated)
.  w - number of degress of freedom per node
.  s - stencil width

   Output Parameter:
.  inra - the resulting distributed array object

   Options Database Key:
$  -da_view : call DAView() at the conclusion of DACreate3d()

.keywords: distributed array, create, three-dimensional

.seealso: DADestroy(), DAView(), DACreate1d(), DACreate2d()
@*/
int DACreate3d(MPI_Comm comm,DAPeriodicType wrap,DAStencilType stencil_type, 
               int M,int N,int P,int m,int n,int p,int w,int s,DA *inra)
{
  int           rank,size,ierr,start,end,pm,flg;
  int           xs,xe,ys,ye,zs,ze,x,y,z,Xs,Xe,Ys,Ye,Zs,Ze;
  int           left,up,down,bottom,top,i,j,k,*idx,nn;
  int           n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n14;
  int           n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26;
  int           *bases,*ldims,x_t,y_t,z_t,s_t,base,count;
  int           s_x,s_y,s_z; /* s proportionalized to w */
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
  da->dim = 3;

  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank); 

  /* Partition the array among the processors */
  if (m == PETSC_DECIDE && n != PETSC_DECIDE && p != PETSC_DECIDE) {
    m = size/(n*p);
  } 
  else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    n = size/(m*p);
  }
  else if (m != PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    p = size/(m*n);
  }
  else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int) (0.5 + sqrt( ((double)M)*((double)size)/((double)N*p) ));
    if (m == 0) m = 1;
    while (m > 0) {
      n = size/(m*p);
      if (m*n*p == size) break;
      m--;
    }
    if (m == 0) SETERRQ(1,"DACreate3d:bad p value");
    if (M > N && m < n) {int _m = m; m = n; n = _m;}
  }  
  else if (m == PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int) (0.5 + sqrt( ((double)M)*((double)size)/((double)P*n) ));
    if (m == 0) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (m == 0) SETERRQ(1,"DACreate3d:bad n value");
    if (M > P && m < p) {int _m = m; m = p; p = _m;}
  } 
  else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (int) ( 0.5 + sqrt( ((double)N)*((double)size)/((double)P*m) ));
    if (n == 0) n = 1;
    while (n > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      n--;
    }
    if (n == 0) SETERRQ(1,"DACreate3d:bad m value");
    if (N > P && n < p) {int _n = n; n = p; p = _n;}
  } 
  else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (int) (0.5 + pow( ((double)N*N)*((double)size)/((double)P*M), 1./3. ));
    if (n == 0) n = 1;
    while (n > 0) {
      pm = size/n;
      if (n*pm == size) break;
      n--;
    }   
    if (n == 0) n = 1; 
    m = (int) (0.5 + sqrt( ((double)M)*((double)size)/((double)P*n) ));
    if (m == 0) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (M > P && m < p) {int _m = m; m = p; p = _m;}
  } 
  else if (m*n*p != size) SETERRQ(1,"DACreate3d:Given Bad partition"); 

  if (m*n*p != size) SETERRQ(1,"DACreate3d:Could not find good partition");  
  if (M < m) SETERRQ(1,"DACreate3d:Partition in x direction is too fine!");
  if (N < n) SETERRQ(1,"DACreate3d:Partition in y direction is too fine!");
  if (P < p) SETERRQ(1,"DACreate3d:Partition in z direction is too fine!");

  ierr = OptionsHasName(PETSC_NULL,"-da_partition_blockcomm",&flg); CHKERRQ(ierr);
  if (flg) { /* Block Comm type Distribution */
    x = (M + rank%m)/m;
    y = (N + (rank%(m*n))/m)/n;
    z = (P + rank/(m*n))/p;

    if (x < s) SETERRQ(1,"DACreate3d:Column width is too thin for stencil!");
    if (y < s) SETERRQ(1,"DACreate3d:Row width is too thin for stencil!");
    if (z < s) SETERRQ(1,"DACreate3d:Plane width is too thin for stencil!");
    
    if (M/m == x) { xs = (rank % m)*x; }
    else { xs = (rank % m)*(x-1) + (M+(rank % m))%(x*m); }
    if (N/n == y) { ys = ((rank%(m*n))/m)*y;  }
    else { ys = ((rank%(m*n))/m)*(y-1) + (N+((rank%(m*n))/m))%(y*n); }
    if (P/p == z) { zs = (rank/(m*n))*z; }
    else { zs = (rank/(m*n))*(z-1) + (P+(rank/(m*n)))%(z*p); }
  }
  else { /* Normal PETSc distribution */
    /* determine locally owned region */
    x = M/m + ((M % m) > (rank % m));
    y = N/n + ((N % n) > ((rank % (m*n)) /m)); 
    z = P/p + ((P % p) > (rank / (m*n)));
    
    if (x < s) SETERRQ(1,"DACreate3d:Column width is too thin for stencil!");
    if (y < s) SETERRQ(1,"DACreate3d:Row width is too thin for stencil!");
    if (z < s) SETERRQ(1,"DACreate3d:Plane width is too thin for stencil!");
    
    if ((M % m) > (rank % m)) { xs = (rank % m)*x; }
    else { xs = (M % m)*(x+1) + ((rank % m)-(M % m))*x; }
    
    
    if ((N % n) > ((rank % (m*n)) /m)) { ys = ((rank % (m*n))/m)*y; }
    else { ys = (N % n)*(y+1) + (((rank % (m*n))/m)-(N % n))*y; }
    
    if ((P % p) > (rank / (m*n))) { zs = (rank/(m*n))*z; }
    else { zs = (P % p)*(z+1) + ((rank/(m*n))-(P % p))*z; }
  }
  ye = ys + y;
  xe = xs + x;
  ze = zs + z;

  /* determine ghost region */
  /* Assume No Periodicity */
  if (xs-s > 0) Xs = xs - s; else Xs = 0; 
  if (ys-s > 0) Ys = ys - s; else Ys = 0;
  if (zs-s > 0) Zs = zs - s; else Zs = 0;
  if (xe+s <= M) Xe = xe + s; else Xe = M; 
  if (ye+s <= N) Ye = ye + s; else Ye = N;
  if (ze+s <= P) Ze = ze + s; else Ze = P;

  /* X Periodic */
  if ((wrap == DA_XPERIODIC)  || (wrap == DA_XYPERIODIC) || 
      (wrap == DA_XZPERIODIC) || (wrap == DA_XYZPERIODIC)) {
    Xs = xs - s; 
    Xe = xe + s; 
  }

  /* Y Periodic */
  if ((wrap == DA_YPERIODIC)  || (wrap == DA_XYPERIODIC) || 
      (wrap == DA_YZPERIODIC) || (wrap == DA_XYZPERIODIC)) {
    Ys = ys - s;
    Ye = ye + s;
  }

  /* Z Periodic */
  if ((wrap == DA_ZPERIODIC)  || (wrap == DA_XZPERIODIC) || 
      (wrap == DA_YZPERIODIC) ||(wrap == DA_XYZPERIODIC)) {
    Zs = zs - s;
    Ze = ze + s;
  }

  /* Resize all X parameters to reflect w */
  x   *= w;
  xs  *= w;
  xe  *= w;
  Xs  *= w;
  Xe  *= w;
  s_x  = s*w;
  s_y  = s;
  s_z  = s;

  /* determine starting point of each processor */
  nn = x*y*z;
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
  ierr = VecCreateMPI(comm,x*y*z,PETSC_DECIDE,&global); CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys)*(Ze-Zs),&local);CHKERRQ(ierr);

  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,x*y*z,start,1,&to);CHKERRQ(ierr);

  left   = xs - Xs; 
  bottom = ys - Ys; top = bottom + y;
  down   = zs - Zs; up  = down + z;
  count  = x*(top-bottom)*(up-down);
  idx    = (int *) PetscMalloc( count*sizeof(int) ); CHKPTRQ(idx);
  count  = 0;
  for ( i=down; i<up; i++ ) {
    for ( j=bottom; j<top; j++) {
      for ( k=0; k<x; k++ ) {
        idx[count++] = (left+j*(Xe-Xs))+i*(Xe-Xs)*(Ye-Ys) + k;
      }
    }
  }
  ierr = ISCreateSeq(MPI_COMM_SELF,count,idx,&from); CHKERRQ(ierr);
  PetscFree(idx);

  ierr = VecScatterCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX) {
    ierr = ISCreateStrideSeq(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys)*(Ze-Zs),0,1,&to); 
  }
  else {
    /* This is way ugly! We need to list the funny cross type region */
    /* the bottom chunck */
    left   = xs - Xs; 
    bottom = ys - Ys; top = bottom + y;
    down   = zs - Zs;   up  = down + z;
    count  = down*(top-bottom)*x +
             (up-down)*(bottom*x  + (top-bottom)*(Xe-Xs) + (Ye-Ys-top)*x) +
             (Ze-Zs-up)*(top-bottom)*x;
    idx    = (int *) PetscMalloc( count*sizeof(int) ); CHKPTRQ(idx);
    count  = 0;
    for ( i=0; i<down; i++ ) {
      for ( j=bottom; j<top; j++) {
        for ( k=0; k<x; k++ ) idx[count++] = left+j*(Xe-Xs)+i*(Xe-Xs)*(Ye-Ys)+k;
      }
    }
    /* the middle piece */
    for ( i=down; i<up; i++ ) {
      /* front */
      for ( j=0; j<bottom; j++) {
        for ( k=0; k<x; k++ ) idx[count++] = left+j*(Xe-Xs)+i*(Xe-Xs)*(Ye-Ys)+k;
      }
      /* middle */
      for ( j=bottom; j<top; j++) {
        for ( k=0; k<Xe-Xs; k++ ) idx[count++] = j*(Xe-Xs)+i*(Xe-Xs)*(Ye-Ys)+k;
      }
      /* back */
      for ( j=top; j<Ye-Ys; j++) {
        for ( k=0; k<x; k++ ) idx[count++] = left+j*(Xe-Xs)+i*(Xe-Xs)*(Ye-Ys)+k;
      }
    }
    /* the top piece */
    for ( i=up; i<Ze-Zs; i++ ) {
      for ( j=bottom; j<top; j++) {
        for ( k=0; k<x; k++ ) idx[count++] = left+j*(Xe-Xs)+i*(Xe-Xs)*(Ye-Ys)+k;
      }
    }
    ierr = ISCreateSeq(MPI_COMM_SELF,count,idx,&to); CHKERRQ(ierr);
    PetscFree(idx);
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
 
  n0  = rank - m*n - m - 1;
  n1  = rank - m*n - m;
  n2  = rank - m*n - m + 1;
  n3  = rank - m*n -1;
  n4  = rank - m*n;
  n5  = rank - m*n + 1;
  n6  = rank - m*n + m - 1;
  n7  = rank - m*n + m;
  n8  = rank - m*n + m + 1;

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

  if (xe == M*w) { /* First assume not corner or edge */
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
    n7 = size - (m*n) + rank + m ;
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

  if ((xe==M*w) && (zs==0)) { /* Assume an edge, not corner */
    n2 = size - (m*n-rank) - (m-1) - m;
    n5 = size - (m*n-rank) - (m-1);
    n8 = size - (m*n-rank) - (m-1) + m;
  }

  if ((xe==M*w) && (ze==P)) { /* Assume an edge, not corner */
    n20 = m*n - (size - rank) - (m-1) - m;
    n23 = m*n - (size - rank) - (m-1);
    n26 = m*n - (size - rank) - (m-1) + m;
  }

  if ((xe==M*w) && (ys==0)) { /* Assume an edge, not corner */
    n2  = rank + m*(n-1) - (m-1) - m*n;
    n11 = rank + m*(n-1) - (m-1);
    n20 = rank + m*(n-1) - (m-1) + m*n;
  }

  if ((xe==M*w) && (ye==N)) { /* Assume an edge, not corner */
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
  if ((xs==0)   && (ys==0) && (zs==0)) { n0  = size -1;}
  if ((xs==0)   && (ys==0) && (ze==P)) { n18 = m*n-1;}    
  if ((xs==0)   && (ye==N) && (zs==0)) { n6  = (size-1)-m*(n-1);}
  if ((xs==0)   && (ye==N) && (ze==P)) { n24 = m-1;}
  if ((xe==M*w) && (ys==0) && (zs==0)) { n2  = size-m;}
  if ((xe==M*w) && (ys==0) && (ze==P)) { n20 = m*n-m;}
  if ((xe==M*w) && (ye==N) && (zs==0)) { n8  = size-m*n;}
  if ((xe==M*w) && (ye==N) && (ze==P)) { n26 = 0;}

  /* Check for when not X,Y, and Z Periodic */

  /* If not X periodic */
  if ((wrap != DA_XPERIODIC)  && (wrap != DA_XYPERIODIC) && 
     (wrap != DA_XZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (xs==0)   {n0  = n3  = n6  = n9  = n12 = n15 = n18 = n21 = n24 = -2;}
    if (xe==M*w) {n2  = n5  = n8  = n11 = n14 = n17 = n20 = n23 = n26 = -2;}
  }

  /* If not Y periodic */
  if ((wrap != DA_YPERIODIC)  && (wrap != DA_XYPERIODIC) && 
      (wrap != DA_YZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (ys==0)   {n0  = n1  = n2  = n9  = n10 = n11 = n18 = n19 = n20 = -2;}
    if (ye==N)   {n6  = n7  = n8  = n15 = n16 = n17 = n24 = n25 = n26 = -2;}
  }

  /* If not Z periodic */
  if ((wrap != DA_ZPERIODIC)  && (wrap != DA_XZPERIODIC) && 
      (wrap != DA_YZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (zs==0)   {n0  = n1  = n2  = n3  = n4  = n5  = n6  = n7  = n8  = -2;}
    if (ze==P)   {n18 = n19 = n20 = n21 = n22 = n23 = n24 = n25 = n26 = -2;}
  }

  /* If star stencil then delete some of the neighbors */
  if (stencil_type == DA_STENCIL_STAR) { n0  = n1  = n2  = n3  = n5  = n6  = 
                      n7  = n8  = n9  = n11 = n15 = n17 = n18 = n19 = 
                      n20 = n21 = n23 = n24 = n25 = n26 = -1;}


  idx = (int *) PetscMalloc( (Xe-Xs)*(Ye-Ys)*(Ze-Zs)*sizeof(int) ); CHKPTRQ(idx);
  PLogObjectMemory(da,(Xe-Xs)*(Ye-Ys)*(Ze-Zs)*sizeof(int) );

  nn = 0;

  /* Bottom Level */
  for ( k=0; k<s_z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n0 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n0 % m)))*w;
        y_t = N/n + ((N % n) > ((n0 % (m*n))/m));
        z_t = P/p + ((P % p) > (n0 / (m*n)));
        s_t = bases[n0] + x_t*y_t*z_t - (s_y-i)*x_t -s_x - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n1 % (m*n)) /m));
        z_t = P/p + ((P % p) > (n1 / (m*n)));
        s_t = bases[n1] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n2 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n2 % m)))*w;
        y_t = N/n + ((N % n) > ((n2 % (m*n))/m));
        z_t = P/p + ((P % p) > (n2 / (m*n)));
        s_t = bases[n2] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n3 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n3 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n3 / (m*n)));
        s_t = bases[n3] + (i+1)*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      if (n4 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        z_t = P/p + ((P % p) > (n4 / (m*n)));
        s_t = bases[n4] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }

      if (n5 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n5 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n5 / (m*n)));
        s_t = bases[n5] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n6 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n6 % m)))*w;
        y_t = N/n + ((N % n) > ((n6 % (m*n))/m));
        z_t = P/p + ((P % p) > (n6 / (m*n)));
        s_t = bases[n6] + i*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n7 % (m*n)/m));
        z_t = P/p + ((P % p) > (n7 / (m*n)));
        s_t = bases[n7] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n8 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n8 % m)))*w;
        y_t = N/n + ((N % n) > (n8 % (m*n) /m));
        z_t = P/p + ((P % p) > (n8 / (m*n)));
        s_t = bases[n8] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }
  }

  /* Middle Level */
  for ( k=0; k<z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n9 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n9 % m)))*w;
        y_t = N/n + ((N % n) > ((n9 % (m*n))/m));
        z_t = z;
        s_t = bases[n9] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n10 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n10 % (m*n)) /m));
        z_t = z;
        s_t = bases[n10] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n11 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n11 % m)))*w;
        y_t = N/n + ((N % n) > ((n11 % (m*n))/m));
        z_t = z;
        s_t = bases[n11] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n12 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n12 % m)))*w;
        y_t = y;
        z_t = z;
        s_t = bases[n12] + (i+1)*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      /* Interior */
      s_t = bases[rank] + i*x + k*x*y;
      for ( j=0; j<x; j++ ) { idx[nn++] = s_t++;}

      if (n14 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n14 % m)))*w;
        y_t = y;
        z_t = z;
        s_t = bases[n14] + i*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n15 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n15 % m)))*w;
        y_t = N/n + ((N % n) > ((n15 % (m*n))/m));
        z_t = z;
        s_t = bases[n15] + i*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n16 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n16 % (m*n)/m));
        z_t = z;
        s_t = bases[n16] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n17 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n17 % m)))*w;
        y_t = N/n + ((N % n) > (n17 % (m*n) /m));
        z_t = z;
        s_t = bases[n17] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    } 
  }
 
  /* Upper Level */
  for ( k=0; k<s_z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n18 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n18 % m)))*w;
        y_t = N/n + ((N % n) > ((n18 % (m*n))/m));
        z_t = P/p + ((P % p) > (n18 / (m*n)));
        s_t = bases[n18] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n19 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n19 % (m*n)) /m));
        z_t = P/p + ((P % p) > (n19 / (m*n)));
        s_t = bases[n19] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n20 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n20 % m)))*w;
        y_t = N/n + ((N % n) > ((n20 % (m*n))/m));
        z_t = P/p + ((P % p) > (n20 / (m*n)));
        s_t = bases[n20] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n21 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n21 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n21 / (m*n)));
        s_t = bases[n21] + (i+1)*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      if (n22 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        z_t = P/p + ((P % p) > (n22 / (m*n)));
        s_t = bases[n22] + i*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }

      if (n23 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n23 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n23 / (m*n)));
        s_t = bases[n23] + i*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n24 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n24 % m)))*w;
        y_t = N/n + ((N % n) > ((n24 % (m*n))/m));
        z_t = P/p + ((P % p) > (n24 / (m*n)));
        s_t = bases[n24] + i*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n25 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n25 % (m*n)/m));
        z_t = P/p + ((P % p) > (n25 / (m*n)));
        s_t = bases[n25] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n26 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n26 % m)))*w;
        y_t = N/n + ((N % n) > (n26 % (m*n) /m));
        z_t = P/p + ((P % p) > (n26 / (m*n)));
        s_t = bases[n26] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }
  }  
  base = bases[rank];
  ierr = ISCreateSeq(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  PLogObjectParent(da,gtol);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  ISDestroy(to); ISDestroy(from);
  da->stencil_type = stencil_type;
  da->M  = M;  da->N  = N; da->P = P; 
  da->m  = m;  da->n  = n; da->p = p;
  da->w  = w;  da->s  = s;
  da->xs = xs; da->xe = xe; da->ys = ys; da->ye = ye; da->zs = zs; da->ze = ze;
  da->Xs = Xs; da->Xe = Xe; da->Ys = Ys; da->Ye = Ye; da->Zs = Zs; da->Ze = Ze;

  PLogObjectParent(da,global);
  PLogObjectParent(da,local);

  da->global = global; 
  da->local  = local; 
  da->gtol   = gtol;
  da->ltog   = ltog;
  da->idx    = idx;
  da->Nl     = nn;
  da->base   = base;
  da->view   = DAView_3d;
  *inra = da;

  /* redo idx to include "missing" ghost points */
  /* Solve for X,Y, and Z Periodic Case First, Then Modify Solution */
 
  /* Assume Nodes are Internal to the Cube */
 
  n0  = rank - m*n - m - 1;
  n1  = rank - m*n - m;
  n2  = rank - m*n - m + 1;
  n3  = rank - m*n -1;
  n4  = rank - m*n;
  n5  = rank - m*n + 1;
  n6  = rank - m*n + m - 1;
  n7  = rank - m*n + m;
  n8  = rank - m*n + m + 1;

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

  if (xe == M*w) { /* First assume not corner or edge */
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
    n7 = size - (m*n) + rank + m ;
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

  if ((xe==M*w) && (zs==0)) { /* Assume an edge, not corner */
    n2 = size - (m*n-rank) - (m-1) - m;
    n5 = size - (m*n-rank) - (m-1);
    n8 = size - (m*n-rank) - (m-1) + m;
  }

  if ((xe==M*w) && (ze==P)) { /* Assume an edge, not corner */
    n20 = m*n - (size - rank) - (m-1) - m;
    n23 = m*n - (size - rank) - (m-1);
    n26 = m*n - (size - rank) - (m-1) + m;
  }

  if ((xe==M*w) && (ys==0)) { /* Assume an edge, not corner */
    n2  = rank + m*(n-1) - (m-1) - m*n;
    n11 = rank + m*(n-1) - (m-1);
    n20 = rank + m*(n-1) - (m-1) + m*n;
  }

  if ((xe==M*w) && (ye==N)) { /* Assume an edge, not corner */
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
  if ((xs==0)   && (ys==0) && (zs==0)) { n0  = size -1;}
  if ((xs==0)   && (ys==0) && (ze==P)) { n18 = m*n-1;}    
  if ((xs==0)   && (ye==N) && (zs==0)) { n6  = (size-1)-m*(n-1);}
  if ((xs==0)   && (ye==N) && (ze==P)) { n24 = m-1;}
  if ((xe==M*w) && (ys==0) && (zs==0)) { n2  = size-m;}
  if ((xe==M*w) && (ys==0) && (ze==P)) { n20 = m*n-m;}
  if ((xe==M*w) && (ye==N) && (zs==0)) { n8  = size-m*n;}
  if ((xe==M*w) && (ye==N) && (ze==P)) { n26 = 0;}

  /* Check for when not X,Y, and Z Periodic */

  /* If not X periodic */
  if ((wrap != DA_XPERIODIC)  && (wrap != DA_XYPERIODIC) && 
     (wrap != DA_XZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (xs==0)   {n0  = n3  = n6  = n9  = n12 = n15 = n18 = n21 = n24 = -2;}
    if (xe==M*w) {n2  = n5  = n8  = n11 = n14 = n17 = n20 = n23 = n26 = -2;}
  }

  /* If not Y periodic */
  if ((wrap != DA_YPERIODIC)  && (wrap != DA_XYPERIODIC) && 
      (wrap != DA_YZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (ys==0)   {n0  = n1  = n2  = n9  = n10 = n11 = n18 = n19 = n20 = -2;}
    if (ye==N)   {n6  = n7  = n8  = n15 = n16 = n17 = n24 = n25 = n26 = -2;}
  }

  /* If not Z periodic */
  if ((wrap != DA_ZPERIODIC)  && (wrap != DA_XZPERIODIC) && 
      (wrap != DA_YZPERIODIC) && (wrap != DA_XYZPERIODIC)) {
    if (zs==0)   {n0  = n1  = n2  = n3  = n4  = n5  = n6  = n7  = n8  = -2;}
    if (ze==P)   {n18 = n19 = n20 = n21 = n22 = n23 = n24 = n25 = n26 = -2;}
  }

  nn = 0;

  /* Bottom Level */
  for ( k=0; k<s_z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n0 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n0 % m)))*w;
        y_t = N/n + ((N % n) > ((n0 % (m*n))/m));
        z_t = P/p + ((P % p) > (n0 / (m*n)));
        s_t = bases[n0] + x_t*y_t*z_t - (s_y-i)*x_t -s_x - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n1 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n1 % (m*n)) /m));
        z_t = P/p + ((P % p) > (n1 / (m*n)));
        s_t = bases[n1] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n2 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n2 % m)))*w;
        y_t = N/n + ((N % n) > ((n2 % (m*n))/m));
        z_t = P/p + ((P % p) > (n2 / (m*n)));
        s_t = bases[n2] + x_t*y_t*z_t - (s_y+1-i)*x_t - (s_z-k-1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n3 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n3 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n3 / (m*n)));
        s_t = bases[n3] + (i+1)*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      if (n4 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        z_t = P/p + ((P % p) > (n4 / (m*n)));
        s_t = bases[n4] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }

      if (n5 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n5 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n5 / (m*n)));
        s_t = bases[n5] + i*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n6 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n6 % m)))*w;
        y_t = N/n + ((N % n) > ((n6 % (m*n))/m));
        z_t = P/p + ((P % p) > (n6 / (m*n)));
        s_t = bases[n6] + i*x_t - s_x + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n7 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n7 % (m*n)/m));
        z_t = P/p + ((P % p) > (n7 / (m*n)));
        s_t = bases[n7] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n8 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n8 % m)))*w;
        y_t = N/n + ((N % n) > (n8 % (m*n) /m));
        z_t = P/p + ((P % p) > (n8 / (m*n)));
        s_t = bases[n8] + (i-1)*x_t + x_t*y_t*z_t - (s_z-k)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }
  }

  /* Middle Level */
  for ( k=0; k<z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n9 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n9 % m)))*w;
        y_t = N/n + ((N % n) > ((n9 % (m*n))/m));
        z_t = z;
        s_t = bases[n9] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n10 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n10 % (m*n)) /m));
        z_t = z;
        s_t = bases[n10] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n11 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n11 % m)))*w;
        y_t = N/n + ((N % n) > ((n11 % (m*n))/m));
        z_t = z;
        s_t = bases[n11] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n12 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n12 % m)))*w;
        y_t = y;
        z_t = z;
        s_t = bases[n12] + (i+1)*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      /* Interior */
      s_t = bases[rank] + i*x + k*x*y;
      for ( j=0; j<x; j++ ) { idx[nn++] = s_t++;}

      if (n14 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n14 % m)))*w;
        y_t = y;
        z_t = z;
        s_t = bases[n14] + i*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n15 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n15 % m)))*w;
        y_t = N/n + ((N % n) > ((n15 % (m*n))/m));
        z_t = z;
        s_t = bases[n15] + i*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n16 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n16 % (m*n)/m));
        z_t = z;
        s_t = bases[n16] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n17 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n17 % m)))*w;
        y_t = N/n + ((N % n) > (n17 % (m*n) /m));
        z_t = z;
        s_t = bases[n17] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    } 
  }
 
  /* Upper Level */
  for ( k=0; k<s_z; k++) {  
    for ( i=1; i<=s_y; i++ ) {
      if (n18 >= 0) { /* left below */
        x_t = (M/m + ((M % m) > (n18 % m)))*w;
        y_t = N/n + ((N % n) > ((n18 % (m*n))/m));
        z_t = P/p + ((P % p) > (n18 / (m*n)));
        s_t = bases[n18] - (s_y-i)*x_t -s_x + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n19 >= 0) { /* directly below */
        x_t = x;
        y_t = N/n + ((N % n) > ((n19 % (m*n)) /m));
        z_t = P/p + ((P % p) > (n19 / (m*n)));
        s_t = bases[n19] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n20 >= 0) { /* right below */
        x_t = (M/m + ((M % m) > (n20 % m)))*w;
        y_t = N/n + ((N % n) > ((n20 % (m*n))/m));
        z_t = P/p + ((P % p) > (n20 / (m*n)));
        s_t = bases[n20] - (s_y+1-i)*x_t + (k+1)*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=0; i<y; i++ ) {
      if (n21 >= 0) { /* directly left */
        x_t = (M/m + ((M % m) > (n21 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n21 / (m*n)));
        s_t = bases[n21] + (i+1)*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }

      if (n22 >= 0) { /* middle */
        x_t = x;
        y_t = y;
        z_t = P/p + ((P % p) > (n22 / (m*n)));
        s_t = bases[n22] + i*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }

      if (n23 >= 0) { /* directly right */
        x_t = (M/m + ((M % m) > (n23 % m)))*w;
        y_t = y;
        z_t = P/p + ((P % p) > (n23 / (m*n)));
        s_t = bases[n23] + i*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }

    for ( i=1; i<=s_y; i++ ) {
      if (n24 >= 0) { /* left above */
        x_t = (M/m + ((M % m) > (n24 % m)))*w;
        y_t = N/n + ((N % n) > ((n24 % (m*n))/m));
        z_t = P/p + ((P % p) > (n24 / (m*n)));
        s_t = bases[n24] + i*x_t - s_x + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
      if (n25 >= 0) { /* directly above */
        x_t = x;
        y_t = N/n + ((N % n) > (n25 % (m*n)/m));
        z_t = P/p + ((P % p) > (n25 / (m*n)));
        s_t = bases[n25] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<x_t; j++ ) { idx[nn++] = s_t++;}
      }
      if (n26 >= 0) { /* right above */
        x_t = (M/m + ((M % m) > (n26 % m)))*w;
        y_t = N/n + ((N % n) > (n26 % (m*n) /m));
        z_t = P/p + ((P % p) > (n26 / (m*n)));
        s_t = bases[n26] + (i-1)*x_t + k*x_t*y_t;
        for ( j=0; j<s_x; j++ ) { idx[nn++] = s_t++;}
      }
    }
  }

  /* construct the local to local scatter context */
  /* 
      We simply remap the values in the from part of 
    global to local to read from an array with the ghost values 
    rather then from the plan array.
  */
  ierr = VecScatterCopy(gtol,&da->ltol); CHKERRQ(ierr);
  PLogObjectParent(da,da->ltol);
  left   = xs - Xs; 
  bottom = ys - Ys; top = bottom + y;
  down   = zs - Zs; up  = down + z;
  count  = x*(top-bottom)*(up-down);
  idx    = (int *) PetscMalloc( count*sizeof(int) ); CHKPTRQ(idx);
  count  = 0;
  for ( i=down; i<up; i++ ) {
    for ( j=bottom; j<top; j++) {
      for ( k=0; k<x; k++ ) {
        idx[count++] = (left+j*(Xe-Xs))+i*(Xe-Xs)*(Ye-Ys) + k;
      }
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
  ldim = x*y*z;
  ierr = VecGetSize(global,&gdim); CHKERRQ(ierr);
  da->gtog1 = (int *)PetscMalloc(gdim*sizeof(int)); CHKPTRQ(da->gtog1);
  gA        = (int *)PetscMalloc((2*(gdim+ldim))*sizeof(int)); CHKPTRQ(gA);
  gB        = (int *)(gA + ldim);
  gAall     = (int *)(gB + ldim);
  gBall     = (int *)(gAall + gdim);
  /* Compute local parts of global orderings */
  ict = 0;
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
        /* gA = global number for 1 proc; gB = current global number */
        gA[ict] = i + j*M*w + k*M*N*w;
        gB[ict] = start + ict;
        ict++;
      }
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

