
/* File created by Peter Mell  7/14/95 */

/*  Code for manipulating distributed regular 3d arrays in parallel  */

#include "daimpl.h"     /*I   "da.h"    I*/
#include "pviewer.h"
#include <math.h>
#include "draw.h"

int DAView_3d(PetscObject dain,Viewer ptr)
{
  DA da = (DA) dain;
  PetscObject vobj = (PetscObject)ptr;
  int         mytid;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);

  MPI_Comm_rank(da->comm,&mytid); 

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  if (vobj->cookie == VIEWER_COOKIE) {
    FILE *fd = ViewerFileGetPointer_Private(ptr);
    if (vobj->type == FILE_VIEWER) {
      MPIU_Seq_begin(da->comm,1);
      fprintf(fd,"Processor [%d] M %d N %d P %d m %d n %d p %d w %d s %d\n",
                 mytid,da->M,da->N,da->P,da->m,da->n,da->p,da->w,da->s);
      fprintf(fd,"X range %d %d Y range %d %d\n",da->xs,da->xe,da->ys,da->ye);
      fflush(fd);
      MPIU_Seq_end(da->comm,1);
    }
    else if (vobj->type == FILES_VIEWER) {

      if (!mytid) {
      }
      else {
      }
    }
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    DrawCtx win = (DrawCtx) ptr;
    double  ymin = -1.0,ymax = (double) da->N;
    double  xmin = -1.0,xmax = (double) ((da->M+2)*da->P),x,y;
    int     k, plane;
    double  ycoord, xcoord;
    int     base,*idx;
    char    node[10];

    DrawSetCoordinates(win,xmin,ymin,xmax,ymax);

    /* first processor draw all node lines */
    if (!mytid) {
      for (k=0; k<da->P; k++) {
        ymin = 0.0; ymax = (double) (da->N - 1);
        for ( xmin=(double)(k*(da->M+1)); xmin<(double)(da->M+(k*(da->M+1))); xmin++ ) {
          DrawLine(win,xmin,ymin,xmin,ymax,DRAW_BLACK);
        }
      
        xmin = (double)(k*(da->M+1)); xmax = xmin + (double)(da->M - 1);
        for ( ymin=0; ymin<(double)da->N; ymin++ ) {
          DrawLine(win,xmin,ymin,xmax,ymin,DRAW_BLACK);
        }
      }
    }
    DrawSyncFlush(win);
    MPI_Barrier(da->comm);


    for (k=0; k<da->P; k++) {  /*Go through and draw for each plane*/
      if ((k >= da->zs) && (k < da->ze)) {
        /* draw my box */
        ymin = da->ys;       
        ymax = da->ye - 1; 
        xmin = da->xs/da->w    + (da->M+1)*k; 
        xmax =(da->xe-1)/da->w + (da->M+1)*k;

        DrawLine(win,xmin,ymin,xmax,ymin,DRAW_RED);
        DrawLine(win,xmin,ymin,xmin,ymax,DRAW_RED);
        DrawLine(win,xmin,ymax,xmax,ymax,DRAW_RED);
        DrawLine(win,xmax,ymin,xmax,ymax,DRAW_RED);  

        xmin = da->xs/da->w; 
        xmax =(da->xe-1)/da->w;

        /* put in numbers*/
        base = (da->base+(da->xe-da->xs)*(da->ye-da->ys)*(k-da->zs))/da->w;

        /* Identify which processor owns the box */
        sprintf(node,"%d",mytid);
        DrawText(win,xmin+(da->M+1)*k+.2,ymin+.3,DRAW_RED,node);

        for ( y=ymin; y<=ymax; y++ ) {
          for ( x=xmin+(da->M+1)*k; x<=xmax+(da->M+1)*k; x++ ) {
            sprintf(node,"%d",base++);
            DrawText(win,x,y,DRAW_BLACK,node);
          }
        } 
 
      }
    } 
   
    DrawSyncFlush(win);
    MPI_Barrier(da->comm);

  if (da->stencil_type == DA_STENCIL_BOX) {
    for (k=0-da->s; k<da->P+da->s; k++) {  
      /*Go through and draw for each plane*/
      if ((k >= da->Zs) && (k < da->Ze)) {
  
        /* overlay ghost numbers, useful for error checking */
        base = (da->Xe-da->Xs)*(da->Ye-da->Ys)*(k-da->Zs); idx = da->idx;
        plane=k;  
        /*Keep z wrap around points on the drawing*/
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
            xcoord = x;   /*Keep x wrap points on drawing */          

            if (x<xmin)  { xcoord = xmax - (xmin-x); }
            if (x>=xmax) { xcoord = xmin + (x-xmax); }
            DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node);
            base+=da->w;
          }
        }
      }         
    } 
  }
  else
  /* Print Ghost Points for Star Stencil */
  {
    /*Go through and draw for each plane*/
    for (k=0-da->s; k<da->P+da->s; k++) {  
  
        /* overlay ghost numbers, useful for error checking */

        /* Bottom Part */
        if ((k<da->zs) && (k>=da->Zs)){ 
          base = (da->ye-da->ys)*(da->xe-da->xs)*(k-da->Zs);
          idx = da->idx;
          plane=k;  
          /*Keep z wrap around points on the drawing*/
          if (k<0)    { plane=da->P+k; }  
          if (k>=da->P) { plane=k-da->P; }
          ymin = da->ys; ymax = da->ye; 
          xmin = (da->M+1)*plane*da->w; 
          xmax = (da->M+1)*plane*da->w+da->M*da->w;
          for ( y=ymin; y<ymax; y++ ) {
            for ( x=xmin+da->xs; x<xmin+da->xe; x+=da->w) {
              sprintf(node,"%d",idx[base]/da->w);  
              ycoord = y;
              /*Keep y wrap around points on drawing */
              if (y<0)      { ycoord = da->N+y; }   
              if (y>=da->N) { ycoord = y-da->N; }
              xcoord = x;   /*Keep x wrap points on drawing */          

              if (x<xmin)  { xcoord = xmax - (xmin-x); } 
              if (x>=xmax) { xcoord = xmin + (x-xmax); }
              DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node); 
              base+=da->w;
            }
          }
	}

        
        /* Middle Part */
        if ((k<da->ze) && (k>=da->zs)) { 
          base = (da->ye-da->ys)*(da->xe-da->xs)*(da->zs-da->Zs) + 
                 ( (da->xe-da->xs)*(da->ys-da->Ys + da->Ye-da->ye) + 
                   (da->Xe-da->Xs)*(da->ye-da->ys) ) * (k-da->zs);
          idx=da->idx;
          plane=k;  
          /*Keep z wrap around points on the drawing*/
          if (k<0)    { plane=da->P+k; }  
          if (k>=da->P) { plane=k-da->P; }

          /* below middle */
          ymin = da->Ys; ymax = da->ys; 
          xmin = (da->M+1)*plane*da->w; 
          xmax = (da->M+1)*plane*da->w+da->M*da->w;
          for ( y=ymin; y<ymax; y++ ) {
            for ( x=xmin+da->xs; x<xmin+da->xe; x+=da->w) {
              sprintf(node,"%d",idx[base]/da->w);
              ycoord = y;
              /*Keep y wrap around points on drawing */
              if (y<0)      { ycoord = da->N+y; }   
              if (y>=da->N) { ycoord = y-da->N; }
              xcoord = x;   /*Keep x wrap points on drawing */          

              if (x<xmin)  { xcoord = xmax - (xmin-x); }
              if (x>=xmax) { xcoord = xmin + (x-xmax); }
              DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node);
              base+=da->w;
            }
          }

          /* center middle */
          ymin = da->ys; ymax = da->ye; 
          xmin = (da->M+1)*plane*da->w; 
          xmax = (da->M+1)*plane*da->w+da->M*da->w;
          for ( y=ymin; y<ymax; y++ ) {
            for ( x=xmin+da->Xs; x<xmin+da->Xe; x+=da->w) {
              sprintf(node,"%d",idx[base]/da->w);
              ycoord = y;
              /*Keep y wrap around points on drawing */
              if (y<0)      { ycoord = da->N+y; }   
              if (y>=da->N) { ycoord = y-da->N; }
              xcoord = x;   /*Keep x wrap points on drawing */          

              if (x<xmin)  { xcoord = xmax - (xmin-x); }
              if (x>=xmax) { xcoord = xmin + (x-xmax); }
              DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node);
              base+=da->w;
            }
          }
          
          /* above middle */
          ymin = da->ye; ymax = da->Ye; 
          xmin = (da->M+1)*plane*da->w; 
          xmax = (da->M+1)*plane*da->w+da->M*da->w;
          for ( y=ymin; y<ymax; y++ ) {
            for ( x=xmin+da->xs; x<xmin+da->xe; x+=da->w) {
              sprintf(node,"%d",idx[base]/da->w);
              ycoord = y;
              /*Keep y wrap around points on drawing */
              if (y<0)      { ycoord = da->N+y; }   
              if (y>=da->N) { ycoord = y-da->N; }
              xcoord = x;   /*Keep x wrap points on drawing */          

              if (x<xmin)  { xcoord = xmax - (xmin-x); }
              if (x>=xmax) { xcoord = xmin + (x-xmax); }
              DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node);
              base+=da->w;
            }
          }

        }

        
        /* Top Part */
        if ((k>=da->ze) && (k<da->Ze)) { 
          base = (da->ye-da->ys)*(da->xe-da->xs)*(da->zs-da->Zs) + 
                 ( (da->xe-da->xs)*(da->ys-da->Ys + da->Ye-da->ye) + 
                   (da->Xe-da->Xs)*(da->ye-da->ys) ) * (da->ze-da->zs) +
                   (da->ye-da->ys)*(da->xe-da->xs)*(k-da->ze); 
          idx=da->idx;
          plane=k;  
          /*Keep z wrap around points on the drawing*/
          if (k<0)    { plane=da->P+k; }  
          if (k>=da->P) { plane=k-da->P; }
          ymin = da->ys; ymax = da->ye; 
          xmin = (da->M+1)*plane*da->w; 
          xmax = (da->M+1)*plane*da->w+da->M*da->w;
          for ( y=ymin; y<ymax; y++ ) {
            for ( x=xmin+da->xs; x<xmin+da->xe; x+=da->w) {
              sprintf(node,"%d",idx[base]/da->w); 
              ycoord = y;
              /*Keep y wrap around points on drawing */
              if (y<0)      { ycoord = da->N+y; }   
              if (y>=da->N) { ycoord = y-da->N; }
              xcoord = x;   /*Keep x wrap points on drawing */          

              if (x<xmin)  { xcoord = xmax - (xmin-x); } 
              if (x>=xmax) { xcoord = xmin + (x-xmax); }
              DrawText(win,xcoord/da->w,ycoord,DRAW_BLUE,node); 
              base+=da->w;
            }
          }
        }
  


    
    } 
  }
    DrawSyncFlush(win);
  }
  return 0;
}

/*@
    DACreate3d - Creates a three-dimensional regular array that is
    distributed across some processors.

   Input Parameters:
.  stencil_type - Type of stencil { DA_STENCIL_STAR or DA_STENCIL_BOX }
.  M,N,P - global dimension in each direction of the array
.  m,n,p - corresponding local dimensions (or PETSC_DECIDE to have calculated)
.  w - number of degress of freedom per node
.  s - stencil width
.  wrap - what type of periodicity should the array have, if any
$      DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC
$      DA_XYZPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC

   Output Parameter:
.  inra - the resulting array object

.keywords: distributed array, create, three-dimensional
.seealso: DADestroy(), DAView()
@*/
int DACreate3d(MPI_Comm comm, DAPeriodicType wrap, DAStencilType stencil_type, 
             int M, int N, int P, int m, int n, int p, int w, int s, DA *inra)
{
  int           mytid, numtid,ierr,start,end;
  int           xs,xe,ys,ye,zs,ze,x,y,z,Xs,Xe,Ys,Ye,Zs,Ze;
  int           left,up,down,bottom,top,i,j,k,*idx,nn;
  int           n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n14;
  int           n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26;
  int           *bases,x_t,y_t,z_t,s_t,base;
  int           s_x,s_y,s_z; /* s proportionalized to w */
  int           temp_m, temp_n, temp_p;
  int           rank = M + N + P;
  DA            da;
  Vec           local,global;
  VecScatterCtx ltog,gtol;
  IS            to,from;
  *inra = 0;

  PETSCHEADERCREATE(da,_DA,DA_COOKIE,0,comm);
  PLogObjectCreate(da);
  PLogObjectMemory(da,sizeof(struct _DA));

  MPI_Comm_size(comm,&numtid); 
  MPI_Comm_rank(comm,&mytid); 

  /* Partition the array among the processors */
  if (m == PETSC_DECIDE && n != PETSC_DECIDE && p != PETSC_DECIDE) {
    m = numtid/(n*p);
  }
  else 
  if (m != PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    n = numtid/(m*p);
  }
  else 
  if (m != PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    p = numtid/(m*n);
  }
  else
  if (m == PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    for (temp_m=1; temp_m<=numtid; temp_m++)
      for (temp_n=1; temp_n<=numtid; temp_n++)
        if ( (temp_m*temp_n*p==numtid) && 
             (rank >= (M/temp_m + N/temp_n + P/p)) &&
             (M>=temp_m) && (N>=temp_n) )
          {
            m=temp_m; n=temp_n;
            rank = M/m + N/n + P/p; 
          }       
  }  
  else
  if (m == PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    for (temp_m=1; temp_m<=numtid; temp_m++)
      for (temp_p=1; temp_p<=numtid; temp_p++)
        if ( (temp_m*n*temp_p==numtid) && 
             (rank >= (M/temp_m + N/n + P/temp_p)) &&
             (M>=temp_m) && (P>=temp_p) )
          {
            m=temp_m; p=temp_p;
            rank = M/m + N/n + P/p; 
          }       
  } 
  else
  if (m != PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    for (temp_n=1; temp_n<=numtid; temp_n++)
      for (temp_p=1; temp_p<=numtid; temp_p++)
        if ( (m*temp_n*temp_p==numtid) && 
             (rank >= (M/m + N/temp_n + P/temp_p)) &&
             (N>=temp_n) && (P>=temp_p) )
          {
            n=temp_n; p=temp_p;
            rank = M/m + N/n + P/p; 
          }       
  } 
  else
  if (m == PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* Brute force approach: Try all combinations and choose best */
    for (temp_m=1; temp_m<=numtid; temp_m++)
      for (temp_n=1; temp_n<=numtid; temp_n++)
        for (temp_p=1; temp_p<=numtid; temp_p++)
          if ( (temp_m*temp_n*temp_p==numtid) && 
               (rank >= (M/temp_m + N/temp_n + P/temp_p)) &&
               (M>=temp_m) && (N>=temp_n) && (P>=temp_p) )
            {
              m=temp_m; n=temp_n; p=temp_p;
              rank = M/m + N/n + P/p; 
            }
  }
  else if (m*n*p != numtid) SETERRQ(1,"DACreate3d: Given Bad partition"); 

  if (m*n*p != numtid) SETERRQ(1,"DACreate3d: Could not find good partition");  
  if (M < m) SETERRQ(1,"DACreate3d: Partition in x direction is too fine!");
  if (N < n) SETERRQ(1,"DACreate3d: Partition in y direction is too fine!");
  if (P < p) SETERRQ(1,"DACreate3d: Partition in z direction is too fine!");

  /* determine local owned region */
  x = M/m + ((M % m) > (mytid % m));
  y = N/n + ((N % n) > ((mytid % (m*n)) /m)); 
  z = P/p + ((P % p) > (mytid / (m*n)));

  if (x < s) SETERRQ(1,"DACreate3d: Column width is too thin for stencil!");
  if (y < s) SETERRQ(1,"DACreate3d: Row width is too thin for stencil!");
  if (z < s) SETERRQ(1,"DACreate3d: Plane width is too thin for stencil!");

  if ((M % m) > (mytid % m)) { xs = (mytid % m)*x; }
  else { xs = (M % m)*(x+1) + ((mytid % m)-(M % m))*x; }
  xe = xs + x;

  if ((N % n) > ((mytid % (m*n)) /m)) { ys = ((mytid % (m*n))/m)*y; }
  else { ys = (N % n)*(y+1) + (((mytid % (m*n))/m)-(N % n))*y; }
  ye = ys + y;

  if ((P % p) > (mytid / (m*n))) { zs = (mytid/(m*n))*z; }
  else { zs = (P % p)*(z+1) + ((mytid/(m*n))-(P % p))*z; }
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
  bases = (int *) PETSCMALLOC( (numtid+1)*sizeof(int) ); CHKPTRQ(bases);
  MPI_Allgather(&nn,1,MPI_INT,bases+1,1,MPI_INT,comm);
  bases[0] = 0;
  for ( i=1; i<=numtid; i++ ) {
    bases[i] += bases[i-1];
  }

  /* allocate the base parallel and sequential vectors */
  ierr = VecCreateMPI(comm,x*y*z,PETSC_DECIDE,&global); CHKERRQ(ierr);

  if (stencil_type == DA_STENCIL_BOX) {  /* Box Stencil */
  ierr = VecCreateSequential(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys)*(Ze-Zs),&local);
  }
  else { /* Star Stencil */
  ierr = VecCreateSequential(MPI_COMM_SELF,       (zs-Zs)*x*y +
                                                  z*x*(ys-Ys) +
				                  z*(Xe-Xs)*y +
				                  z*x*(Ye-ye) +
				                  (Ze-ze)*x*y,&local);
  }
  CHKERRQ(ierr);

  /* generate appropriate vector scatters */
  /* local to global inserts non-ghost point region into global */
  VecGetOwnershipRange(global,&start,&end);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,x*y*z,start,1,&to); CHKERRQ(ierr);

  if (stencil_type == DA_STENCIL_BOX) {  /* Regular Stencil */
    left = xs - Xs; 
    bottom = ys - Ys; top = bottom + y;
    down = zs - Zs;   up  = down + z;
    from = 0;
    for ( i=down; i<up; i++ ) {
      for ( j=bottom; j<top; j++) {
        ierr = ISAddStrideSequential(&from,x,(left+j*(Xe-Xs))+i*(Xe-Xs)*(Ye-Ys),1);
        CHKERRQ(ierr);
      }
    }
  }
  else { /* Star Stencil */

    bottom = (zs-Zs)*x*y;
    down   = x*(ys-Ys);
    up     = x*(Ye-ye);
    left   = (xs-Xs);
    from   = 0;

    for ( i=0; i<z; i++ ) {
      for ( j=0; j<y; j++) {
        ierr = ISAddStrideSequential(&from,x,
                  bottom + i*(down+up+(Xe-Xs)*y) + down 
                  + j*(Xe-Xs) + left,1);
        CHKERRQ(ierr);
      }
    }  
  }

  ierr = VecScatterCtxCreate(local,from,global,to,&ltog); CHKERRQ(ierr);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  PLogObjectParent(da,ltog);
  ISDestroy(from); ISDestroy(to);

  /* global to local must include ghost points */
  if (stencil_type == DA_STENCIL_BOX) {
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,(Xe-Xs)*(Ye-Ys)*(Ze-Zs),0,1,&to); 
  }
  else {
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,(zs-Zs)*x*y +
                                                  z*x*(ys-Ys) +
				                  z*(Xe-Xs)*y +
				                  z*x*(Ye-ye) +
				                  (Ze-ze)*x*y,0,1,&to); 
  }
  CHKERRQ(ierr); 

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
 
  n0  = mytid - m*n - m - 1;
  n1  = mytid - m*n - m;
  n2  = mytid - m*n - m + 1;
  n3  = mytid - m*n -1;
  n4  = mytid - m*n;
  n5  = mytid - m*n + 1;
  n6  = mytid - m*n + m - 1;
  n7  = mytid - m*n + m;
  n8  = mytid - m*n + m + 1;

  n9  = mytid - m - 1;
  n10 = mytid - m;
  n11 = mytid - m + 1;
  n12 = mytid - 1;
  n14 = mytid + 1;
  n15 = mytid + m - 1;
  n16 = mytid + m;
  n17 = mytid + m + 1;

  n18 = mytid + m*n - m - 1;
  n19 = mytid + m*n - m;
  n20 = mytid + m*n - m + 1;
  n21 = mytid + m*n - 1;
  n22 = mytid + m*n;
  n23 = mytid + m*n + 1;
  n24 = mytid + m*n + m - 1;
  n25 = mytid + m*n + m;
  n26 = mytid + m*n + m + 1;

  /* Assume Pieces are on Faces of Cube */

  if (xs == 0) { /* First assume not corner or edge */
    n0  = mytid       -1 - (m*n);
    n3  = mytid + m   -1 - (m*n);
    n6  = mytid + 2*m -1 - (m*n);
    n9  = mytid       -1;
    n12 = mytid + m   -1;
    n15 = mytid + 2*m -1;
    n18 = mytid       -1 + (m*n);
    n21 = mytid + m   -1 + (m*n);
    n24 = mytid + 2*m -1 + (m*n);
   }

  if (xe == M*w) { /* First assume not corner or edge */
    n2  = mytid -2*m +1 - (m*n);
    n5  = mytid - m  +1 - (m*n);
    n8  = mytid      +1 - (m*n);      
    n11 = mytid -2*m +1;
    n14 = mytid - m  +1;
    n17 = mytid      +1;
    n20 = mytid -2*m +1 + (m*n);
    n23 = mytid - m  +1 + (m*n);
    n26 = mytid      +1 + (m*n);
  }

  if (ys==0) { /* First assume not corner or edge */
    n0  = mytid + m * (n-1) -1 - (m*n);
    n1  = mytid + m * (n-1)    - (m*n);
    n2  = mytid + m * (n-1) +1 - (m*n);
    n9  = mytid + m * (n-1) -1;
    n10 = mytid + m * (n-1);
    n11 = mytid + m * (n-1) +1;
    n18 = mytid + m * (n-1) -1 + (m*n);
    n19 = mytid + m * (n-1)    + (m*n);
    n20 = mytid + m * (n-1) +1 + (m*n);
  }

  if (ye == N) { /* First assume not corner or edge */
    n6  = mytid - m * (n-1) -1 - (m*n);
    n7  = mytid - m * (n-1)    - (m*n);
    n8  = mytid - m * (n-1) +1 - (m*n);
    n15 = mytid - m * (n-1) -1;
    n16 = mytid - m * (n-1);
    n17 = mytid - m * (n-1) +1;
    n24 = mytid - m * (n-1) -1 + (m*n);
    n25 = mytid - m * (n-1)    + (m*n);
    n26 = mytid - m * (n-1) +1 + (m*n);
  }
 
  if (zs == 0) { /* First assume not corner or edge */
    n0 = numtid - (m*n) + mytid - m - 1;
    n1 = numtid - (m*n) + mytid - m;
    n2 = numtid - (m*n) + mytid - m + 1;
    n3 = numtid - (m*n) + mytid - 1;
    n4 = numtid - (m*n) + mytid;
    n5 = numtid - (m*n) + mytid + 1;
    n6 = numtid - (m*n) + mytid + m - 1;
    n7 = numtid - (m*n) + mytid + m ;
    n8 = numtid - (m*n) + mytid + m + 1;
  }

  if (ze == P) { /* First assume not corner or edge */
    n18 = (m*n) - (numtid-mytid) - m - 1;
    n19 = (m*n) - (numtid-mytid) - m;
    n20 = (m*n) - (numtid-mytid) - m + 1;
    n21 = (m*n) - (numtid-mytid) - 1;
    n22 = (m*n) - (numtid-mytid);
    n23 = (m*n) - (numtid-mytid) + 1;
    n24 = (m*n) - (numtid-mytid) + m - 1;
    n25 = (m*n) - (numtid-mytid) + m;
    n26 = (m*n) - (numtid-mytid) + m + 1; 
  }

  if ((xs==0) && (zs==0)) { /* Assume an edge, not corner */
    n0 = numtid - m*n + mytid + m-1 - m;
    n3 = numtid - m*n + mytid + m-1;
    n6 = numtid - m*n + mytid + m-1 + m;
  }
 
  if ((xs==0) && (ze==P)) { /* Assume an edge, not corner */
    n18 = m*n - (numtid - mytid) + m-1 - m;
    n21 = m*n - (numtid - mytid) + m-1;
    n24 = m*n - (numtid - mytid) + m-1 + m;
  }

  if ((xs==0) && (ys==0)) { /* Assume an edge, not corner */
    n0  = mytid + m*n -1 - m*n;
    n9  = mytid + m*n -1;
    n18 = mytid + m*n -1 + m*n;
  }

  if ((xs==0) && (ye==N)) { /* Assume an edge, not corner */
    n6  = mytid - m*(n-1) + m-1 - m*n;
    n15 = mytid - m*(n-1) + m-1;
    n24 = mytid - m*(n-1) + m-1 + m*n;
  }

  if ((xe==M*w) && (zs==0)) { /* Assume an edge, not corner */
    n2 = numtid - (m*n-mytid) - (m-1) - m;
    n5 = numtid - (m*n-mytid) - (m-1);
    n8 = numtid - (m*n-mytid) - (m-1) + m;
  }

  if ((xe==M*w) && (ze==P)) { /* Assume an edge, not corner */
    n20 = m*n - (numtid - mytid) - (m-1) - m;
    n23 = m*n - (numtid - mytid) - (m-1);
    n26 = m*n - (numtid - mytid) - (m-1) + m;
  }

  if ((xe==M*w) && (ys==0)) { /* Assume an edge, not corner */
    n2  = mytid + m*(n-1) - (m-1) - m*n;
    n11 = mytid + m*(n-1) - (m-1);
    n20 = mytid + m*(n-1) - (m-1) + m*n;
  }

  if ((xe==M*w) && (ye==N)) { /* Assume an edge, not corner */
    n8  = mytid - m*n +1 - m*n;
    n17 = mytid - m*n +1;
    n26 = mytid - m*n +1 + m*n;
  }

  if ((ys==0) && (zs==0)) { /* Assume an edge, not corner */
    n0 = numtid - m + mytid -1;
    n1 = numtid - m + mytid;
    n2 = numtid - m + mytid +1;
  }

  if ((ys==0) && (ze==P)) { /* Assume an edge, not corner */
    n18 = m*n - (numtid - mytid) + m*(n-1) -1;
    n19 = m*n - (numtid - mytid) + m*(n-1);
    n20 = m*n - (numtid - mytid) + m*(n-1) +1;
  }

  if ((ye==N) && (zs==0)) { /* Assume an edge, not corner */
    n6 = numtid - (m*n-mytid) - m * (n-1) -1;
    n7 = numtid - (m*n-mytid) - m * (n-1);
    n8 = numtid - (m*n-mytid) - m * (n-1) +1;
  }

  if ((ye==N) && (ze==P)) { /* Assume an edge, not corner */
    n24 = mytid - (numtid-m) -1;
    n25 = mytid - (numtid-m);
    n26 = mytid - (numtid-m) +1;
  }

  /* Check for Corners */
  if ((xs==0)   && (ys==0) && (zs==0)) { n0  = numtid -1;}
  if ((xs==0)   && (ys==0) && (ze==P)) { n18 = m*n-1;}    
  if ((xs==0)   && (ye==N) && (zs==0)) { n6  = (numtid-1)-m*(n-1);}
  if ((xs==0)   && (ye==N) && (ze==P)) { n24 = m-1;}
  if ((xe==M*w) && (ys==0) && (zs==0)) { n2  = numtid-m;}
  if ((xe==M*w) && (ys==0) && (ze==P)) { n20 = m*n-m;}
  if ((xe==M*w) && (ye==N) && (zs==0)) { n8  = numtid-m*n;}
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


  idx = (int *) PETSCMALLOC( (Xe-Xs)*(Ye-Ys)*(Ze-Zs)*sizeof(int) ); CHKPTRQ(idx);
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
      s_t = bases[mytid] + i*x + k*x*y;
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
  base = bases[mytid];
  PETSCFREE(bases);
  ierr = ISCreateSequential(comm,nn,idx,&from); CHKERRQ(ierr);
  ierr = VecScatterCtxCreate(global,from,local,to,&gtol); CHKERRQ(ierr);
  PLogObjectParent(da,gtol);
  PLogObjectParent(da,to);
  PLogObjectParent(da,from);
  ISDestroy(to); ISDestroy(from);
  da->stencil_type = stencil_type;
  da->M  = M;  da->N  = N; da->P = P; 
  da->m  = m;  da->n  = n; da->p = p;
  da->w = w; da->s = s;
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
  return 0;
}

