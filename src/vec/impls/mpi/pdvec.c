/* $Id: pdvec.c,v 1.134 2000/04/09 04:35:30 bsmith Exp bsmith $*/
/*
     Code for some of the parallel vector primatives.
*/
#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecGetOwnershipRange_MPI"
int VecGetOwnershipRange_MPI(Vec v,int *low,int* high) 
{
  PetscFunctionBegin;
  *low  = v->map->rstart;
  *high = v->map->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecDestroy_MPI"
int VecDestroy_MPI(Vec v)
{
  Vec_MPI *x = (Vec_MPI*)v->data;
  int     ierr;

  PetscFunctionBegin;

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)v,"Length=%d",v->N);
#endif  
  if (x->array_allocated) {ierr = PetscFree(x->array_allocated);CHKERRQ(ierr);}

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    ierr = VecDestroy(x->localrep);CHKERRQ(ierr);
    if (x->localupdate) {ierr = VecScatterDestroy(x->localupdate);CHKERRQ(ierr);}
  }
  /* Destroy the stashes */
  ierr = VecStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = VecStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  if (x->browners) {ierr = PetscFree(x->browners);CHKERRQ(ierr);}
  ierr = PetscFree(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_ASCII"
int VecView_MPI_ASCII(Vec xin,Viewer viewer)
{
  Vec_MPI     *x = (Vec_MPI*)xin->data;
  int         i,rank,len,work = xin->n,n,j,size,ierr,format,cnt,tag = ((PetscObject)viewer)->tag;
  MPI_Status  status;
  Scalar      *values;
  char        *outputname;

  PetscFunctionBegin;
  /* determine maximum message to arrive */
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Reduce(&work,&len,1,MPI_INT,MPI_MAX,0,xin->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);

  if (!rank) {
    values = (Scalar*)PetscMalloc((len+1)*sizeof(Scalar));CHKPTRQ(values);
    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    /*
        Matlab format and ASCII format are very similar except 
        Matlab uses %18.16e format while ASCII uses %g
    */
    if (format == VIEWER_FORMAT_ASCII_MATLAB) {
      ierr = ViewerGetOutputname(viewer,&outputname);CHKERRQ(ierr);
      ierr = ViewerASCIIPrintf(viewer,"%s = [\n",outputname);CHKERRQ(ierr);
      for (i=0; i<xin->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(x->array[i]) > 0.0) {
          ierr = ViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(x->array[i]) < 0.0) {
          ierr = ViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",PetscRealPart(x->array[i]),-PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
        } else {
          ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",PetscRealPart(x->array[i]));CHKERRQ(ierr);
        }
#else
        ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",x->array[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,tag,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            ierr = ViewerASCIIPrintf(viewer,"%18.16e + %18.16e i\n",PetscRealPart(values[i]),PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            ierr = ViewerASCIIPrintf(viewer,"%18.16e - %18.16e i\n",PetscRealPart(values[i]),-PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else {
            ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",PetscRealPart(values[i]));CHKERRQ(ierr);
          }
#else
          ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",values[i]);CHKERRQ(ierr);
#endif
        }
      }          
      ierr = ViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);

    } else if (format == VIEWER_FORMAT_ASCII_SYMMODU) {
      for (i=0; i<xin->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = ViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
#else
        ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",x->array[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,tag,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          ierr = ViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",PetscRealPart(values[i]),PetscImaginaryPart(values[i]));CHKERRQ(ierr);
#else
          ierr = ViewerASCIIPrintf(viewer,"%18.16e\n",values[i]);CHKERRQ(ierr);
#endif
        }
      }          

    } else {
      if (format != VIEWER_FORMAT_ASCII_COMMON) {ierr = ViewerASCIIPrintf(viewer,"Processor [%d]\n",rank);CHKERRQ(ierr);}
      cnt = 0;
      for (i=0; i<xin->n; i++) {
        if (format == VIEWER_FORMAT_ASCII_INDEX) {
          ierr = ViewerASCIIPrintf(viewer,"%d: ",cnt++);CHKERRQ(ierr);
        }
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(x->array[i]) > 0.0) {
          ierr = ViewerASCIIPrintf(viewer,"%g + %g i\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(x->array[i]) < 0.0) {
          ierr = ViewerASCIIPrintf(viewer,"%g - %g i\n",PetscRealPart(x->array[i]),-PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
        } else {
          ierr = ViewerASCIIPrintf(viewer,"%g\n",PetscRealPart(x->array[i]));CHKERRQ(ierr);
        }
#else
        ierr = ViewerASCIIPrintf(viewer,"%g\n",x->array[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,len,MPIU_SCALAR,j,tag,xin->comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);        
        if (format != VIEWER_FORMAT_ASCII_COMMON) {
          ierr = ViewerASCIIPrintf(viewer,"Processor [%d]\n",j);CHKERRQ(ierr);
        }
        for (i=0; i<n; i++) {
          if (format == VIEWER_FORMAT_ASCII_INDEX) {
            ierr = ViewerASCIIPrintf(viewer,"%d: ",cnt++);CHKERRQ(ierr);
          }
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            ierr = ViewerASCIIPrintf(viewer,"%g + %g i\n",PetscRealPart(values[i]),PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            ierr = ViewerASCIIPrintf(viewer,"%g - %g i\n",PetscRealPart(values[i]),-PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else {
            ierr = ViewerASCIIPrintf(viewer,"%g\n",PetscRealPart(values[i]));CHKERRQ(ierr);
          }
#else
          ierr = ViewerASCIIPrintf(viewer,"%g\n",values[i]);CHKERRQ(ierr);
#endif
        }          
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    /* send values */
    ierr = MPI_Send(x->array,xin->n,MPIU_SCALAR,0,tag,xin->comm);CHKERRQ(ierr);
  }
  ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_Binary"
int VecView_MPI_Binary(Vec xin,Viewer viewer)
{
  Vec_MPI     *x = (Vec_MPI*)xin->data;
  int         rank,ierr,len,work = xin->n,n,j,size,fdes,tag = ((PetscObject)viewer)->tag;
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
    ierr = PetscBinaryWrite(fdes,&xin->N,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,x->array,xin->n,PETSC_SCALAR,0);CHKERRQ(ierr);

    values = (Scalar*)PetscMalloc((len+1)*sizeof(Scalar));CHKPTRQ(values);
    /* receive and print messages */
    for (j=1; j<size; j++) {
      ierr = MPI_Recv(values,len,MPIU_SCALAR,j,tag,xin->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);         
      ierr = PetscBinaryWrite(fdes,values,n,PETSC_SCALAR,0);CHKERRQ(ierr);
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = ViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file && xin->bs > 1) {
      fprintf(file,"-vecload_block_size %d\n",xin->bs);
    }
  } else {
    /* send values */
    ierr = MPI_Send(x->array,xin->n,MPIU_SCALAR,0,tag,xin->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_Draw_LG"
int VecView_MPI_Draw_LG(Vec xin,Viewer viewer)
{
  Vec_MPI     *x = (Vec_MPI*)xin->data;
  int         i,rank,size,N = xin->N,*lens,ierr;
  Draw        draw;
  PetscReal   *xx,*yy;
  DrawLG      lg;
  PetscTruth  isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = ViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  ierr = DrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  if (!rank) {
    DrawLGReset(lg);
    xx   = (PetscReal*)PetscMalloc(2*(N+1)*sizeof(PetscReal));CHKPTRQ(xx);
    for (i=0; i<N; i++) {xx[i] = (PetscReal) i;}
    yy   = xx + N;
    lens = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(lens);
    for (i=0; i<size; i++) {
      lens[i] = xin->map->range[i+1] - xin->map->range[i];
    }
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Gatherv(x->array,xin->n,MPI_DOUBLE,yy,lens,xin->map->range,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
#else
    {
      PetscReal *xr;
      xr = (PetscReal*)PetscMalloc((xin->n+1)*sizeof(PetscReal));CHKPTRQ(xr);
      for (i=0; i<xin->n; i++) {
        xr[i] = PetscRealPart(x->array[i]);
      }
      ierr = MPI_Gatherv(xr,xin->n,MPI_DOUBLE,yy,lens,xin->map->range,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
      ierr = PetscFree(xr);CHKERRQ(ierr);
    }
#endif
    ierr = PetscFree(lens);CHKERRQ(ierr);
    ierr = DrawLGAddPoints(lg,N,&xx,&yy);CHKERRQ(ierr);
    ierr = PetscFree(xx);CHKERRQ(ierr);
    ierr = DrawLGDraw(lg);CHKERRQ(ierr);
  } else {
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Gatherv(x->array,xin->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
#else
    {
      PetscReal *xr;
      xr = (PetscReal*)PetscMalloc((xin->n+1)*sizeof(PetscReal));CHKPTRQ(xr);
      for (i=0; i<xin->n; i++) {
        xr[i] = PetscRealPart(x->array[i]);
      }
      ierr = MPI_Gatherv(xr,xin->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
      ierr = PetscFree(xr);CHKERRQ(ierr);
    }
#endif
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
  ierr = DrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_Draw"
int VecView_MPI_Draw(Vec xin,Viewer viewer)
{
  Vec_MPI     *x = (Vec_MPI*)xin->data;
  int         i,rank,size,ierr,start,end,tag = ((PetscObject)viewer)->tag;
  MPI_Status  status;
  PetscReal   coors[4],ymin,ymax,xmin,xmax,tmp;
  Draw        draw;
  PetscTruth  isnull;
  DrawAxis    axis;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  xmin = 1.e20; xmax = -1.e20;
  for (i=0; i<xin->n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscRealPart(x->array[i]) < xmin) xmin = PetscRealPart(x->array[i]);
    if (PetscRealPart(x->array[i]) > xmax) xmax = PetscRealPart(x->array[i]);
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
    ierr = DrawAxisSetLimits(axis,0.0,(double)xin->N,ymin,ymax);CHKERRQ(ierr);
    ierr = DrawAxisDraw(axis);CHKERRQ(ierr);
    ierr = DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);CHKERRQ(ierr);
  }
  ierr = DrawAxisDestroy(axis);CHKERRQ(ierr);
  ierr = MPI_Bcast(coors,4,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
  if (rank) {ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);}
  /* draw local part of vector */
  ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
  if (rank < size-1) { /*send value to right */
    ierr = MPI_Send(&x->array[xin->n-1],1,MPI_DOUBLE,rank+1,tag,xin->comm);CHKERRQ(ierr);
  }
  for (i=1; i<xin->n; i++) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = DrawLine(draw,(PetscReal)(i-1+start),x->array[i-1],(PetscReal)(i+start),
                   x->array[i],DRAW_RED);CHKERRQ(ierr);
#else
    ierr = DrawLine(draw,(PetscReal)(i-1+start),PetscRealPart(x->array[i-1]),(PetscReal)(i+start),
                   PetscRealPart(x->array[i]),DRAW_RED);CHKERRQ(ierr);
#endif
  }
  if (rank) { /* receive value from right */
    ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,rank-1,tag,xin->comm,&status);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = DrawLine(draw,(PetscReal)start-1,tmp,(PetscReal)start,x->array[0],DRAW_RED);CHKERRQ(ierr);
#else
    ierr = DrawLine(draw,(PetscReal)start-1,tmp,(PetscReal)start,PetscRealPart(x->array[0]),DRAW_RED);CHKERRQ(ierr);
#endif
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
  ierr = DrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_Socket"
int VecView_MPI_Socket(Vec xin,Viewer viewer)
{
  Vec_MPI     *x = (Vec_MPI*)xin->data;
  int         i,rank,size,N = xin->N,*lens,ierr;
  Scalar      *xx;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(xin->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xin->comm,&size);CHKERRQ(ierr);
  if (!rank) {
    xx   = (Scalar*)PetscMalloc((N+1)*sizeof(Scalar));CHKPTRQ(xx);
    lens = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(lens);
    for (i=0; i<size; i++) {
      lens[i] = xin->map->range[i+1] - xin->map->range[i];
    }
    ierr = MPI_Gatherv(x->array,xin->n,MPIU_SCALAR,xx,lens,xin->map->range,MPIU_SCALAR,0,xin->comm);CHKERRQ(ierr);
    ierr = PetscFree(lens);CHKERRQ(ierr);
    ierr = ViewerSocketPutScalar_Private(viewer,N,1,xx);CHKERRQ(ierr);
    ierr = PetscFree(xx);CHKERRQ(ierr);
  } else {
    ierr = MPI_Gatherv(x->array,xin->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI"
int VecView_MPI(Vec xin,Viewer viewer)
{
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
    PetscTruth isascii,issocket,isbinary,isdraw;

    ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)viewer,SOCKET_VIEWER,&issocket);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
    if (isascii){
      ierr = VecView_MPI_ASCII(xin,viewer);CHKERRQ(ierr);
    } else if (issocket) {
      ierr = VecView_MPI_Socket(xin,viewer);CHKERRQ(ierr);
    } else if (isbinary) {
      ierr = VecView_MPI_Binary(xin,viewer);CHKERRQ(ierr);
    } else if (isdraw) {
      ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
      if (format == VIEWER_FORMAT_DRAW_LG) {
        ierr = VecView_MPI_Draw_LG(xin,viewer);CHKERRQ(ierr);
      } else {
        SETERRQ(1,1,"Viewer Draw format not supported for this vector");
      }
    } else {
      SETERRQ1(1,1,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
    }
  }
  if (native) {
    ierr   = ViewerPushFormat(viewer,VIEWER_FORMAT_NATIVE,fname);CHKERRQ(ierr);
  }   
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecGetSize_MPI"
int VecGetSize_MPI(Vec xin,int *N)
{
  PetscFunctionBegin;
  *N = xin->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecSetValues_MPI"
int VecSetValues_MPI(Vec xin,int ni,const int ix[],const Scalar y[],InsertMode addv)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  int      rank = x->rank,*owners = xin->map->range,start = owners[rank];
  int      end = owners[rank+1],i,row,ierr;
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
    for (i=0; i<ni; i++) {
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] = y[i];
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] >= xin->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],xin->N);
#endif
        VecStashValue_Private(&xin->stash,row,y[i]);
      }
    }
  } else {
    for (i=0; i<ni; i++) {
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] += y[i];
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] > xin->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],xin->N);
#endif        
        VecStashValue_Private(&xin->stash,row,y[i]);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecSetValuesBlocked_MPI"
int VecSetValuesBlocked_MPI(Vec xin,int ni,const int ix[],const Scalar yin[],InsertMode addv)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  int      rank = x->rank,*owners = xin->map->range,start = owners[rank];
  int      end = owners[rank+1],i,row,bs = xin->bs,j,ierr;
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
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) {
          xx[row-start+j] = y[j];
        }
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] >= xin->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d max %d",ix[i],xin->N);
#endif
        VecStashValuesBlocked_Private(&xin->bstash,ix[i],y);
      }
      y += bs;
    }
  } else {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) {
          xx[row-start+j] += y[j];
        }
      } else if (!x->donotstash) {
        if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        if (ix[i] > xin->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d max %d",ix[i],xin->N);
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
#define __FUNC__ /*<a name=""></a>*/"VecAssemblyBegin_MPI"
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
    bowners = (int*)PetscMalloc((size+1)*sizeof(int));CHKPTRQ(bowners);
    for (i=0; i<size+1; i++){ bowners[i] = owners[i]/bs;}
    x->browners = bowners;
  } else { 
    bowners = x->browners; 
  }
  ierr = VecStashScatterBegin_Private(&xin->stash,owners);CHKERRQ(ierr);
  ierr = VecStashScatterBegin_Private(&xin->bstash,bowners);CHKERRQ(ierr);
  ierr  = VecStashGetInfo_Private(&xin->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"VecAssemblyBegin_MPI:Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  ierr  = VecStashGetInfo_Private(&xin->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"VecAssemblyBegin_MPI:Block-Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecAssemblyEnd_MPI"
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
          for (j=0; j<bs; j++) { array[j] += vv[j];}
        } else if (x->insertmode == INSERT_VALUES) {
          for (j=0; j<bs; j++) { array[j] = vv[j]; }
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

