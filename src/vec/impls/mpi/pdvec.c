#ifdef PETSC_RCS_HEADER
static char vcid[] = $Id: pdvec.c,v 1.120 1999/05/12 03:28:25 bsmith Exp balay $ 
#endif

/*
     Code for some of the parallel vector primatives.
*/
#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/

#undef __FUNC__  
#define __FUNC__ "VecGetOwnershipRange_MPI"
int VecGetOwnershipRange_MPI(Vec v,int *low,int* high) 
{
  PetscFunctionBegin;
  *low  = v->map->rstart;
  *high = v->map->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDestroy_MPI"
int VecDestroy_MPI(Vec v)
{
  Vec_MPI *x = (Vec_MPI *) v->data;
  int     ierr;

  PetscFunctionBegin;

#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)v,"Length=%d",x->N);
#endif  
  if (x->array_allocated) PetscFree(x->array_allocated);

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    ierr = VecDestroy(x->localrep);CHKERRQ(ierr);
    if (x->localupdate) {ierr = VecScatterDestroy(x->localupdate);CHKERRQ(ierr);}
  }
  /* Destroy the stashes */
  ierr = VecStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = VecStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  if (x->browners) { PetscFree(x->browners);}
  PetscFree(x);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_MPI_ASCII"
int VecView_MPI_ASCII(Vec xin, Viewer viewer )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,len, work = x->n,n,j,size,ierr,format,cnt;
  MPI_Status  status;
  FILE        *fd;
  Scalar      *values;
  char        *outputname;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
  /* determine maximum message to arrive */
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Reduce(&work,&len,1,MPI_INT,MPI_MAX,0,xin->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);

  if (!rank) {
    values = (Scalar *) PetscMalloc( (len+1)*sizeof(Scalar) );CHKPTRQ(values);
    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    /*
        Matlab format and ASCII format are very similar except 
        Matlab uses %18.16e format while ASCII uses %g
    */
    if (format == VIEWER_FORMAT_ASCII_MATLAB) {
      ierr = ViewerGetOutputname(viewer,&outputname);CHKERRQ(ierr);
      fprintf(fd,"%s = [\n",outputname);
      for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginary(x->array[i]) > 0.0) {
          fprintf(fd,"%18.16e + %18.16e i\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
        } else if (PetscImaginary(x->array[i]) < 0.0) {
          fprintf(fd,"%18.16e - %18.16e i\n",PetscReal(x->array[i]),-PetscImaginary(x->array[i]));
        } else {
          fprintf(fd,"%18.16e\n",PetscReal(x->array[i]));
        }
#else
        fprintf(fd,"%18.16e\n",x->array[i]);
#endif
      }
      /* receive and print messages */
      for ( j=1; j<size; j++ ) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
        for ( i=0; i<n; i++ ) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginary(values[i]) > 0.0) {
            fprintf(fd,"%18.16e + %18.16e i\n",PetscReal(values[i]),PetscImaginary(values[i]));
          } else if (PetscImaginary(values[i]) < 0.0) {
            fprintf(fd,"%18.16e - %18.16e i\n",PetscReal(values[i]),-PetscImaginary(values[i]));
          } else {
            fprintf(fd,"%18.16e\n",PetscReal(values[i]));
          }
#else
          fprintf(fd,"%18.16e\n",values[i]);
#endif
        }
      }          
      fprintf(fd,"];\n");

    } else if (format == VIEWER_FORMAT_ASCII_SYMMODU) {
      for (i=0; i<x->n; i++ ) {
#if defined(PETSC_USE_COMPLEX)
        fprintf(fd,"%18.16e %18.16e\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
#else
        fprintf(fd,"%18.16e\n",x->array[i]);
#endif
      }
      /* receive and print messages */
      for ( j=1; j<size; j++ ) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
        for ( i=0; i<n; i++ ) {
#if defined(PETSC_USE_COMPLEX)
          fprintf(fd,"%18.16e %18.16e\n",PetscReal(values[i]),PetscImaginary(values[i]));
#else
          fprintf(fd,"%18.16e\n",values[i]);
#endif
        }
      }          

    } else {
      if (format != VIEWER_FORMAT_ASCII_COMMON) fprintf(fd,"Processor [%d]\n",rank);
      cnt = 0;
      for ( i=0; i<x->n; i++ ) {
        if (format == VIEWER_FORMAT_ASCII_INDEX) {
          fprintf(fd,"%d: ",cnt++);
        }
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginary(x->array[i]) > 0.0) {
          fprintf(fd,"%g + %g i\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
        } else if (PetscImaginary(x->array[i]) < 0.0) {
          fprintf(fd,"%g - %g i\n",PetscReal(x->array[i]),-PetscImaginary(x->array[i]));
        } else {
          fprintf(fd,"%g\n",PetscReal(x->array[i]));
        }
#else
        fprintf(fd,"%g\n",x->array[i]);
#endif
      }
      /* receive and print messages */
      for ( j=1; j<size; j++ ) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);        
        if (format != VIEWER_FORMAT_ASCII_COMMON) {
          fprintf(fd,"Processor [%d]\n",j);
        }
        for ( i=0; i<n; i++ ) {
          if (format == VIEWER_FORMAT_ASCII_INDEX) {
            fprintf(fd,"%d: ",cnt++);
          }
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginary(values[i]) > 0.0) {
            fprintf(fd,"%g + %g i\n",PetscReal(values[i]),PetscImaginary(values[i]));
          } else if (PetscImaginary(values[i]) < 0.0) {
            fprintf(fd,"%g - %g i\n",PetscReal(values[i]),-PetscImaginary(values[i]));
          } else {
            fprintf(fd,"%g\n",PetscReal(values[i]));
          }
#else
          fprintf(fd,"%g\n",values[i]);
#endif
        }          
      }
    }
    PetscFree(values);
    fflush(fd);
  } else {
    /* send values */
    ierr = MPI_Send(x->array,x->n,MPIU_SCALAR,0,47,xin->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Binary"
int VecView_MPI_Binary(Vec xin, Viewer viewer )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         rank,ierr,len, work = x->n,n,j,size, fdes;
  MPI_Status  status;
  Scalar      *values;
  FILE        *file;

  PetscFunctionBegin;
  ierr = ViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);

  /* determine maximum message to arrive */
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Reduce(&work,&len,1,MPI_INT,MPI_MAX,0,xin->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscBinaryWrite(fdes,&xin->cookie,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,&x->N,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,x->array,x->n,PETSC_SCALAR,0);CHKERRQ(ierr);

    values = (Scalar *) PetscMalloc( (len+1)*sizeof(Scalar) );CHKPTRQ(values);
    /* receive and print messages */
    for ( j=1; j<size; j++ ) {
      ierr = MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
      ierr = PetscBinaryWrite(fdes,values,n,PETSC_SCALAR,0);CHKERRQ(ierr);
    }
    PetscFree(values);
    ierr = ViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file && xin->bs > 1) {
      fprintf(file,"-vecload_block_size %d\n",xin->bs);
    }
  } else {
    /* send values */
    ierr = MPI_Send(x->array,x->n,MPIU_SCALAR,0,47,xin->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Draw_LG"
int VecView_MPI_Draw_LG(Vec xin,Viewer v  )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size, N = x->N,*lens,ierr;
  Draw        draw;
  double      *xx,*yy;
  DrawLG      lg;
  PetscTruth  isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = ViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  ierr = DrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  if (!rank) {
    DrawLGReset(lg);
    xx   = (double *) PetscMalloc( 2*(N+1)*sizeof(double) );CHKPTRQ(xx);
    for ( i=0; i<N; i++ ) {xx[i] = (double) i;}
    yy   = xx + N;
    lens = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(lens);
    for (i=0; i<size; i++ ) {
      lens[i] = xin->map->range[i+1] - xin->map->range[i];
    }
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Gatherv(x->array,x->n,MPI_DOUBLE,yy,lens,xin->map->range,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
#else
    {
      double *xr;
      xr = (double *) PetscMalloc( (x->n+1)*sizeof(double) );CHKPTRQ(xr);
      for ( i=0; i<x->n; i++ ) {
        xr[i] = PetscReal(x->array[i]);
      }
      ierr = MPI_Gatherv(xr,x->n,MPI_DOUBLE,yy,lens,xin->map->range,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
      PetscFree(xr);
    }
#endif
    PetscFree(lens);
    ierr = DrawLGAddPoints(lg,N,&xx,&yy);CHKERRQ(ierr);
    PetscFree(xx);
    ierr = DrawLGDraw(lg);CHKERRQ(ierr);
  } else {
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Gatherv(x->array,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
#else
    {
      double *xr;
      xr = (double *) PetscMalloc( (x->n+1)*sizeof(double) );CHKPTRQ(xr);
      for ( i=0; i<x->n; i++ ) {
        xr[i] = PetscReal(x->array[i]);
      }
      ierr = MPI_Gatherv(xr,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
      PetscFree(xr);
    }
#endif
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
  DrawPause(draw);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Draw"
int VecView_MPI_Draw(Vec xin, Viewer v )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size,ierr,start,end;
  MPI_Status  status;
  double      coors[4],ymin,ymax,xmin,xmax,tmp;
  Draw        draw;
  PetscTruth  isnull;
  DrawAxis    axis;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);


  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  xmin = 1.e20; xmax = -1.e20;
  for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscReal(x->array[i]) < xmin) xmin = PetscReal(x->array[i]);
    if (PetscReal(x->array[i]) > xmax) xmax = PetscReal(x->array[i]);
#else
    if (x->array[i] < xmin) xmin = x->array[i];
    if (x->array[i] > xmax) xmax = x->array[i];
#endif
  }
  if (xmin + 1.e-10 > xmax) {
    xmin -= 1.e-5;
    xmax += 1.e-5;
  }
  ierr = MPI_Reduce(&xmin,&ymin,1,MPI_DOUBLE,MPI_MIN,0,xin->comm);CHKERRQ(ierr);
  ierr = MPI_Reduce(&xmax,&ymax,1,MPI_DOUBLE,MPI_MAX,0,xin->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = DrawAxisCreate(draw,&axis);CHKERRQ(ierr);
  PLogObjectParent(draw,axis);
  if (!rank) {
    ierr = DrawClear(draw);CHKERRQ(ierr);
    ierr = DrawFlush(draw);CHKERRQ(ierr);
    ierr = DrawAxisSetLimits(axis,0.0,(double) x->N,ymin,ymax);CHKERRQ(ierr);
    ierr = DrawAxisDraw(axis);CHKERRQ(ierr);
    ierr = DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);CHKERRQ(ierr);
  }
  ierr = DrawAxisDestroy(axis);CHKERRQ(ierr);
  ierr = MPI_Bcast(coors,4,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
  if (rank) {ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);}
  /* draw local part of vector */
  ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
  if (rank < size-1) { /*send value to right */
    ierr = MPI_Send(&x->array[x->n-1],1,MPI_DOUBLE,rank+1,xin->tag,xin->comm);CHKERRQ(ierr);
  }
  for ( i=1; i<x->n; i++ ) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = DrawLine(draw,(double)(i-1+start),x->array[i-1],(double)(i+start),
                   x->array[i],DRAW_RED);CHKERRQ(ierr);
#else
    ierr = DrawLine(draw,(double)(i-1+start),PetscReal(x->array[i-1]),(double)(i+start),
                   PetscReal(x->array[i]),DRAW_RED);CHKERRQ(ierr);
#endif
  }
  if (rank) { /* receive value from right */
    ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,rank-1,xin->tag,xin->comm,&status);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = DrawLine(draw,(double)start-1,tmp,(double)start,x->array[0],DRAW_RED);CHKERRQ(ierr);
#else
    ierr = DrawLine(draw,(double)start-1,tmp,(double)start,PetscReal(x->array[0]),DRAW_RED);CHKERRQ(ierr);
#endif
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
  ierr = DrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Socket"
int VecView_MPI_Socket(Vec xin, Viewer viewer )
{
#if !defined(PETSC_USE_COMPLEX)
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size, N = x->N,*lens,ierr;
  double      *xx;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_ERR_SUP,0,"Complex not done");
#else
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  if (!rank) {
    xx = (double *) PetscMalloc( (N+1)*sizeof(double) );CHKPTRQ(xx);
    lens = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(lens);
    for (i=0; i<size; i++ ) {
      lens[i] = xin->map->range[i+1] - xin->map->range[i];
    }
    ierr = MPI_Gatherv(x->array,x->n,MPI_DOUBLE,xx,lens,xin->map->range,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
    PetscFree(lens);
    ierr = ViewerSocketPutScalar_Private(viewer,N,1,xx);CHKERRQ(ierr);
    PetscFree(xx);
  } else {
    ierr = MPI_Gatherv(x->array,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "VecView_MPI"
int VecView_MPI(Vec xin,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr,(*f)(Vec,Viewer),format;
  PetscTruth  native = PETSC_FALSE;
  char        *fname;

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)xin,"VecView_C",(void **)&f);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  /*
      VIEWER_FORMAT_NATIVE means use the standard vector viewers not (for example) 
     DA provided special ones
  */
  if (format == VIEWER_FORMAT_NATIVE) {
   f      = (int (*)(Vec,Viewer)) 0;
   ierr   = ViewerGetOutputname(viewer,&fname);CHKERRQ(ierr);
   ierr   = ViewerPopFormat(viewer);CHKERRQ(ierr);
   native = PETSC_TRUE;
  }
  if (f) {
    ierr = (*f)(xin,viewer);CHKERRQ(ierr);
  } else {
    ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
    if (PetscTypeCompare(vtype,ASCII_VIEWER)){
      ierr = VecView_MPI_ASCII(xin,viewer);CHKERRQ(ierr);
    } else if (PetscTypeCompare(vtype,SOCKET_VIEWER)) {
      ierr = VecView_MPI_Socket(xin,viewer);CHKERRQ(ierr);
    } else if (PetscTypeCompare(vtype,BINARY_VIEWER)) {
      ierr = VecView_MPI_Binary(xin,viewer);CHKERRQ(ierr);
    } else if (PetscTypeCompare(vtype,DRAW_VIEWER)) {
      ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
      if (format == VIEWER_FORMAT_DRAW_LG) {
        ierr = VecView_MPI_Draw_LG(xin, viewer );CHKERRQ(ierr);
      } else {
        SETERRQ(1,1,"Viewer Draw format not supported for this vector");
      }
    } else {
      SETERRQ(1,1,"Viewer type not supported for this object");
    }
  }
  if (native) {
    ierr   = ViewerPushFormat(viewer,VIEWER_FORMAT_NATIVE,fname);CHKERRQ(ierr);
  }   
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetSize_MPI"
int VecGetSize_MPI(Vec xin,int *N)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;

  PetscFunctionBegin;
  *N = x->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValues_MPI"
int VecSetValues_MPI(Vec xin, int ni,const int ix[],const Scalar y[],InsertMode addv)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  int      rank = x->rank, *owners = xin->map->range, start = owners[rank];
  int      end = owners[rank+1], i, row,ierr;
  Scalar   *xx = x->array;

  PetscFunctionBegin;
#if defined(PETSC_USE_BOPT_g)
  if (x->insertmode == INSERT_VALUES && addv == ADD_VALUES) { 
   SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"You have already inserted values; you cannot now add");
  } else if (x->insertmode == ADD_VALUES && addv == INSERT_VALUES) { 
   SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"You have already added values; you cannot now insert");
  }
#endif
  x->insertmode = addv;

  if (addv == INSERT_VALUES) {
    for ( i=0; i<ni; i++ ) {
      if ( (row = ix[i]) >= start && row < end) {
        xx[row-start] = y[i];
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] >= x->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],x->N);
#endif
        VecStashValue_Private(&xin->stash,row,y[i]);
      }
    }
  } else {
    for ( i=0; i<ni; i++ ) {
      if ( (row = ix[i]) >= start && row < end) {
        xx[row-start] += y[i];
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] > x->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],x->N);
#endif        
        VecStashValue_Private(&xin->stash,row,y[i]);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValuesBlocked_MPI"
int VecSetValuesBlocked_MPI(Vec xin, int ni,const int ix[],const Scalar yin[],InsertMode addv)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  int      rank = x->rank, *owners = xin->map->range, start = owners[rank];
  int      end = owners[rank+1], i, row,bs = xin->bs,j,ierr;
  Scalar   *xx = x->array,*y = (Scalar*)yin;

  PetscFunctionBegin;
#if defined(PETSC_USE_BOPT_g)
  if (x->insertmode == INSERT_VALUES && addv == ADD_VALUES) { 
   SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"You have already inserted values; you cannot now add");
  }
  else if (x->insertmode == ADD_VALUES && addv == INSERT_VALUES) { 
   SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"You have already added values; you cannot now insert");
  }
#endif
  x->insertmode = addv;

  if (addv == INSERT_VALUES) {
    for ( i=0; i<ni; i++ ) {
      if ( (row = bs*ix[i]) >= start && row < end) {
        for ( j=0; j<bs; j++ ) {
          xx[row-start+j] = y[j];
        }
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] >= x->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d max %d",ix[i],x->N);
#endif
        VecStashValuesBlocked_Private(&xin->bstash,ix[i],y);
      }
      y += bs;
    }
  } else {
    for ( i=0; i<ni; i++ ) {
      if ( (row = bs*ix[i]) >= start && row < end) {
        for ( j=0; j<bs; j++ ) {
          xx[row-start+j] += y[j];
        }
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] > x->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d max %d",ix[i],x->N);
#endif
        VecStashValuesBlocked_Private(&xin->bstash,ix[i],y);
      }
      y += bs;
    }
  }
  PetscFunctionReturn(0);
}

/*
   Since nsends or nreceives may be zero we add 1 in certain mallocs
to make sure we never malloc an empty one.      
*/
#undef __FUNC__  
#define __FUNC__ "VecAssemblyBegin_MPI"
int VecAssemblyBegin_MPI(Vec xin)
{
  Vec_MPI    *x = (Vec_MPI *)xin->data;
  int         *owners = xin->map->range,*bowners,ierr,size,i,bs,nstash,reallocs;
  InsertMode  addv;
  MPI_Comm    comm = xin->comm;

  PetscFunctionBegin;
  if (x->donotstash) {
    PetscFunctionReturn(0);
  }

  ierr = MPI_Allreduce(&x->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,0,"Some processors inserted values while others added");
  }
  x->insertmode = addv; /* in case this processor had no cache */
  
  bs = xin->bs;
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  if (!x->browners && xin->bs != -1) {
    bowners = (int*) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(bowners);
    for ( i=0; i<size+1; i++ ){ bowners[i] = owners[i]/bs;}
    x->browners = bowners;
  } else { 
    bowners = x->browners; 
  }
  ierr = VecStashScatterBegin_Private(&xin->stash,owners);CHKERRQ(ierr);
  ierr = VecStashScatterBegin_Private(&xin->bstash,bowners);CHKERRQ(ierr);
  ierr  = VecStashGetInfo_Private(&xin->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"VecAssemblyBegin_MPI:Stash has %d entries, uses %d mallocs.\n",
           nstash,reallocs);
  ierr  = VecStashGetInfo_Private(&xin->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"VecAssemblyBegin_MPI:Block-Stash has %d entries, uses %d mallocs.\n",
           nstash,reallocs);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAssemblyEnd_MPI"
int VecAssemblyEnd_MPI(Vec vec)
{
  Vec_MPI     *x = (Vec_MPI *)vec->data;
  int         ierr,base,i,j,n,*row,flg,bs;
  Scalar      *val,*vv,*array;

   PetscFunctionBegin;
  if (!x->donotstash) {
    base = vec->map->range[x->rank];
    bs   = vec->bs;

    /* Process the stash */
    while (1) {
      ierr = VecStashScatterGetMesg_Private(&vec->stash,&n,&row,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      if (x->insertmode == ADD_VALUES) {
        for (i=0; i<n; i++) { x->array[row[i] - base] += val[i]; }
      } else if (x->insertmode == INSERT_VALUES) {
        for (i=0; i<n; i++) { x->array[row[i] - base] = val[i]; }
      } else {
        SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Insert mode is not set correctly; corrupted vector");
      }
    }
    ierr = VecStashScatterEnd_Private(&vec->stash);CHKERRQ(ierr);

    /* now process the block-stash */
    while (1) {
      ierr = VecStashScatterGetMesg_Private(&vec->bstash,&n,&row,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      for (i=0; i<n; i++) { 
        array = x->array+row[i]*bs-base;
        vv    = val+i*bs;
        if (x->insertmode == ADD_VALUES) {
          for ( j=0; j<bs; j++ ) { array[j] += vv[j];}
        } else if (x->insertmode == INSERT_VALUES) {
          for ( j=0; j<bs; j++ ) { array[j] = vv[j]; }
        } else {
          SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Insert mode is not set correctly; corrupted vector");
        }
      }
    }
    ierr = VecStashScatterEnd_Private(&vec->bstash);CHKERRQ(ierr);
  }
  x->insertmode = NOT_SET_VALUES;
  PetscFunctionReturn(0);
}

