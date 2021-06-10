
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <petsc/private/viewerhdf5impl.h>
#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>

PetscErrorCode VecDestroy_MPI(Vec v)
{
  Vec_MPI        *x = (Vec_MPI*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->N);
#endif
  if (!x) PetscFunctionReturn(0);
  ierr = PetscFree(x->array_allocated);CHKERRQ(ierr);

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    ierr = VecDestroy(&x->localrep);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&x->localupdate);CHKERRQ(ierr);
  }
  ierr = VecAssemblyReset_MPI(v);CHKERRQ(ierr);

  /* Destroy the stashes: note the order - so that the tags are freed properly */
  ierr = VecStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  ierr = VecStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_MPI_ASCII(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          i,work = xin->map->n,cnt,len,nLen;
  PetscMPIInt       j,n = 0,size,rank,tag = ((PetscObject)viewer)->tag;
  MPI_Status        status;
  PetscScalar       *values;
  const PetscScalar *xarray;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size);CHKERRMPI(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_LOAD_BALANCE) {
    PetscInt nmax = 0,nmin = xin->map->n,navg;
    for (i=0; i<(PetscInt)size; i++) {
      nmax = PetscMax(nmax,xin->map->range[i+1] - xin->map->range[i]);
      nmin = PetscMin(nmin,xin->map->range[i+1] - xin->map->range[i]);
    }
    navg = xin->map->N/size;
    ierr = PetscViewerASCIIPrintf(viewer,"  Load Balance - Local vector size Min %D  avg %D  max %D\n",nmin,navg,nmax);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  /* determine maximum message to arrive */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank);CHKERRMPI(ierr);
  ierr = MPI_Reduce(&work,&len,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (format == PETSC_VIEWER_ASCII_GLVIS) { rank = 0, len = 0; } /* no parallel distributed write support from GLVis */
  if (rank == 0) {
    ierr = PetscMalloc1(len,&values);CHKERRQ(ierr);
    /*
        MATLAB format and ASCII format are very similar except
        MATLAB uses %18.16e format while ASCII uses %g
    */
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(xarray[i]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(values[i]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]);CHKERRQ(ierr);
#endif
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);

    } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i]));CHKERRQ(ierr);
#else
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]);CHKERRQ(ierr);
#endif
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED) {
      /*
        state 0: No header has been output
        state 1: Only POINT_DATA has been output
        state 2: Only CELL_DATA has been output
        state 3: Output both, POINT_DATA last
        state 4: Output both, CELL_DATA last
      */
      static PetscInt stateId     = -1;
      int             outputState = 0;
      int             doOutput    = 0;
      PetscBool       hasState;
      PetscInt        bs, b;

      if (stateId < 0) {
        ierr = PetscObjectComposedDataRegister(&stateId);CHKERRQ(ierr);
      }
      ierr = PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState);CHKERRQ(ierr);
      if (!hasState) outputState = 0;

      ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
      ierr = VecGetLocalSize(xin, &nLen);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(nLen,&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED) {
        if (outputState == 0) {
          outputState = 1;
          doOutput    = 1;
        } else if (outputState == 1) doOutput = 0;
        else if (outputState == 2) {
          outputState = 3;
          doOutput    = 1;
        } else if (outputState == 3) doOutput = 0;
        else if (outputState == 4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

        if (doOutput) {
          ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", xin->map->N/bs);CHKERRQ(ierr);
        }
      } else {
        if (outputState == 0) {
          outputState = 2;
          doOutput    = 1;
        } else if (outputState == 1) {
          outputState = 4;
          doOutput    = 1;
        } else if (outputState == 2) doOutput = 0;
        else if (outputState == 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
        else if (outputState == 4) doOutput = 0;

        if (doOutput) {
          ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", xin->map->N/bs);CHKERRQ(ierr);
        }
      }
      ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
      if (name) {
        if (bs == 3) {
          ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name, bs);CHKERRQ(ierr);
        }
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %d\n", bs);CHKERRQ(ierr);
      }
      if (bs != 3) {
        ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
      }
      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b]));CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b]));CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED) {
      PetscInt bs, b;

      ierr = VecGetLocalSize(xin, &nLen);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(nLen,&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
      if ((bs < 1) || (bs > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);

      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b]));CHKERRQ(ierr);
        }
        for (b=bs; b<3; b++) {
          ierr = PetscViewerASCIIPrintf(viewer," 0.0");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b]));CHKERRQ(ierr);
          }
          for (b=bs; b<3; b++) {
            ierr = PetscViewerASCIIPrintf(viewer," 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_PCICE) {
      PetscInt bs, b, vertexCount = 1;

      ierr = VecGetLocalSize(xin, &nLen);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(nLen,&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
      if ((bs < 1) || (bs > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %d", bs);

      ierr = PetscViewerASCIIPrintf(viewer,"%D\n", xin->map->N/bs);CHKERRQ(ierr);
      for (i=0; i<n/bs; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
        for (b=0; b<bs; b++) {
          if (b > 0) {
            ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
          }
#if !defined(PETSC_USE_COMPLEX)
          ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)xarray[i*bs+b]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        for (i=0; i<n/bs; i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
          for (b=0; b<bs; b++) {
            if (b > 0) {
              ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
            }
#if !defined(PETSC_USE_COMPLEX)
            ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)values[i*bs+b]);CHKERRQ(ierr);
#endif
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
      /* GLVis ASCII visualization/dump: this function mimics mfem::GridFunction::Save() */
      const PetscScalar       *array;
      PetscInt                i,n,vdim, ordering = 1; /* mfem::FiniteElementSpace::Ordering::byVDIM */
      PetscContainer          glvis_container;
      PetscViewerGLVisVecInfo glvis_vec_info;
      PetscViewerGLVisInfo    glvis_info;
      PetscErrorCode          ierr;

      /* mfem::FiniteElementSpace::Save() */
      ierr = VecGetBlockSize(xin,&vdim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n");CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject)xin,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
      if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_PLIB,"Missing GLVis container");
      ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_vec_info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s\n",glvis_vec_info->fec_type);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"VDim: %d\n",vdim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Ordering: %d\n",ordering);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      /* mfem::Vector::Print() */
      ierr = PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
      if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_PLIB,"Missing GLVis container");
      ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_info);CHKERRQ(ierr);
      if (glvis_info->enabled) {
        ierr = VecGetLocalSize(xin,&n);CHKERRQ(ierr);
        ierr = VecGetArrayRead(xin,&array);CHKERRQ(ierr);
        for (i=0;i<n;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,glvis_info->fmt,(double)PetscRealPart(array[i]));CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(xin,&array);CHKERRQ(ierr);
      }
    } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* No info */
    } else {
      if (format != PETSC_VIEWER_ASCII_COMMON) {ierr = PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank);CHKERRQ(ierr);}
      cnt = 0;
      for (i=0; i<xin->map->n; i++) {
        if (format == PETSC_VIEWER_ASCII_INDEX) {
          ierr = PetscViewerASCIIPrintf(viewer,"%D: ",cnt++);CHKERRQ(ierr);
        }
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(xarray[i]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)xarray[i]);CHKERRQ(ierr);
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRMPI(ierr);
        if (format != PETSC_VIEWER_ASCII_COMMON) {
          ierr = PetscViewerASCIIPrintf(viewer,"Process [%d]\n",j);CHKERRQ(ierr);
        }
        for (i=0; i<n; i++) {
          if (format == PETSC_VIEWER_ASCII_INDEX) {
            ierr = PetscViewerASCIIPrintf(viewer,"%D: ",cnt++);CHKERRQ(ierr);
          }
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(values[i]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)values[i]);CHKERRQ(ierr);
#endif
        }
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* Rank 0 is not trying to receive anything, so don't send anything */
    } else {
      if (format == PETSC_VIEWER_ASCII_MATLAB || format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED) {
        /* this may be a collective operation so make sure everyone calls it */
        ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
      }
      ierr = MPI_Send((void*)xarray,xin->map->n,MPIU_SCALAR,0,tag,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_MPI_Binary(Vec xin,PetscViewer viewer)
{
  return VecView_Binary(xin,viewer);
}

#include <petscdraw.h>
PetscErrorCode VecView_MPI_Draw_LG(Vec xin,PetscViewer viewer)
{
  PetscDraw         draw;
  PetscBool         isnull;
  PetscDrawLG       lg;
  PetscMPIInt       i,size,rank,n,N,*lens = NULL,*disp = NULL;
  PetscReal         *values, *xx = NULL,*yy = NULL;
  const PetscScalar *xarray;
  int               colors[] = {PETSC_DRAW_RED};
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size);CHKERRMPI(ierr);
  ierr = PetscMPIIntCast(xin->map->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(xin->map->N,&N);CHKERRQ(ierr);

  ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(n+1,&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) values[i] = PetscRealPart(xarray[i]);
#else
  values = (PetscReal*)xarray;
#endif
  if (rank == 0) {
    ierr = PetscMalloc2(N,&xx,N,&yy);CHKERRQ(ierr);
    for (i=0; i<N; i++) xx[i] = (PetscReal)i;
    ierr = PetscMalloc2(size,&lens,size,&disp);CHKERRQ(ierr);
    for (i=0; i<size; i++) lens[i] = (PetscMPIInt)xin->map->range[i+1] - (PetscMPIInt)xin->map->range[i];
    for (i=0; i<size; i++) disp[i] = (PetscMPIInt)xin->map->range[i];
  }
  ierr = MPI_Gatherv(values,n,MPIU_REAL,yy,lens,disp,MPIU_REAL,0,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  ierr = PetscFree2(lens,disp);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(values);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
  ierr = PetscDrawLGSetColors(lg,colors);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = PetscDrawLGAddPoints(lg,N,&xx,&yy);CHKERRQ(ierr);
    ierr = PetscFree2(xx,yy);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  VecView_MPI_Draw(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size,tag = ((PetscObject)viewer)->tag;
  PetscInt          i,start,end;
  MPI_Status        status;
  PetscReal         min,max,tmp = 0.0;
  PetscDraw         draw;
  PetscBool         isnull;
  PetscDrawAxis     axis;
  const PetscScalar *xarray;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank);CHKERRMPI(ierr);

  ierr = VecMin(xin,NULL,&min);CHKERRQ(ierr);
  ierr = VecMax(xin,NULL,&max);CHKERRQ(ierr);
  if (min == max) {
    min -= 1.e-5;
    max += 1.e-5;
  }

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  ierr = PetscDrawAxisCreate(draw,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLimits(axis,0.0,(PetscReal)xin->map->N,min,max);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisDestroy(&axis);CHKERRQ(ierr);

  /* draw local part of vector */
  ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
  if (rank < size-1) { /* send value to right */
    ierr = MPI_Send((void*)&xarray[xin->map->n-1],1,MPIU_REAL,rank+1,tag,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  }
  if (rank) { /* receive value from right */
    ierr = MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,PetscObjectComm((PetscObject)xin),&status);CHKERRMPI(ierr);
  }
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank) {
    ierr = PetscDrawLine(draw,(PetscReal)start-1,tmp,(PetscReal)start,PetscRealPart(xarray[0]),PETSC_DRAW_RED);CHKERRQ(ierr);
  }
  for (i=1; i<xin->map->n; i++) {
    ierr = PetscDrawLine(draw,(PetscReal)(i-1+start),PetscRealPart(xarray[i-1]),(PetscReal)(i+start),PetscRealPart(xarray[i]),PETSC_DRAW_RED);CHKERRQ(ierr);
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
PetscErrorCode VecView_MPI_Matlab(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size,*lens;
  PetscInt          i,N = xin->map->N;
  const PetscScalar *xarray;
  PetscScalar       *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size);CHKERRMPI(ierr);
  if (rank == 0) {
    ierr = PetscMalloc1(N,&xx);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&lens);CHKERRQ(ierr);
    for (i=0; i<size; i++) lens[i] = xin->map->range[i+1] - xin->map->range[i];

    ierr = MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,xx,lens,xin->map->range,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    ierr = PetscFree(lens);CHKERRQ(ierr);

    ierr = PetscObjectName((PetscObject)xin);CHKERRQ(ierr);
    ierr = PetscViewerMatlabPutArray(viewer,N,1,xx,((PetscObject)xin)->name);CHKERRQ(ierr);

    ierr = PetscFree(xx);CHKERRQ(ierr);
  } else {
    ierr = MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,0,0,0,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_ADIOS)
#include <adios.h>
#include <adios_read.h>
#include <petsc/private/vieweradiosimpl.h>
#include <petsc/private/viewerimpl.h>

PetscErrorCode VecView_MPI_ADIOS(Vec xin, PetscViewer viewer)
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*)viewer->data;
  PetscErrorCode    ierr;
  const char        *vecname;
  int64_t           id;
  PetscInt          n,N,rstart;
  const PetscScalar *array;
  char              nglobalname[16],nlocalname[16],coffset[16];

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject) xin, &vecname);CHKERRQ(ierr);

  ierr = VecGetLocalSize(xin,&n);CHKERRQ(ierr);
  ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);

  sprintf(nlocalname,"%d",(int)n);
  sprintf(nglobalname,"%d",(int)N);
  sprintf(coffset,"%d",(int)rstart);
  id   = adios_define_var(Petsc_adios_group,vecname,"",adios_double,nlocalname,nglobalname,coffset);
  ierr = VecGetArrayRead(xin,&array);CHKERRQ(ierr);
  ierr = adios_write_byid(adios->adios_handle,id,array);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecView_MPI_HDF5(Vec xin, PetscViewer viewer)
{
  PetscViewer_HDF5  *hdf5 = (PetscViewer_HDF5*) viewer->data;
  /* TODO: It looks like we can remove the H5Sclose(filespace) and H5Dget_space(dset_id). Why do we do this? */
  hid_t             filespace; /* file dataspace identifier */
  hid_t             chunkspace; /* chunk dataset property identifier */
  hid_t             dset_id;   /* dataset identifier */
  hid_t             memspace;  /* memory dataspace identifier */
  hid_t             file_id;
  hid_t             group;
  hid_t             memscalartype; /* scalar type for mem (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  hid_t             filescalartype; /* scalar type for file (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscInt          bs = PetscAbs(xin->map->bs);
  hsize_t           dim;
  hsize_t           maxDims[4], dims[4], chunkDims[4], count[4], offset[4];
  PetscBool         timestepping, dim2, spoutput;
  PetscInt          timestep=PETSC_MIN_INT, low;
  hsize_t           chunksize;
  const PetscScalar *x;
  const char        *vecname;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5IsTimestepping(viewer, &timestepping);CHKERRQ(ierr);
  if (timestepping) {
    ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5GetBaseDimension2(viewer,&dim2);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetSPOutput(viewer,&spoutput);CHKERRQ(ierr);

  /* Create the dataspace for the dataset.
   *
   * dims - holds the current dimensions of the dataset
   *
   * maxDims - holds the maximum dimensions of the dataset (unlimited
   * for the number of time steps with the current dimensions for the
   * other dimensions; so only additional time steps can be added).
   *
   * chunkDims - holds the size of a single time step (required to
   * permit extending dataset).
   */
  dim = 0;
  chunksize = 1;
  if (timestep >= 0) {
    dims[dim]      = timestep+1;
    maxDims[dim]   = H5S_UNLIMITED;
    chunkDims[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(xin->map->N/bs,dims + dim);CHKERRQ(ierr);

  maxDims[dim]   = dims[dim];
  chunkDims[dim] = PetscMax(1, dims[dim]);
  chunksize      *= chunkDims[dim];
  ++dim;
  if (bs > 1 || dim2) {
    dims[dim]      = bs;
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = PetscMax(1, dims[dim]);
    chunksize      *= chunkDims[dim];
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  dims[dim]      = 2;
  maxDims[dim]   = dims[dim];
  chunkDims[dim] = PetscMax(1, dims[dim]);
  chunksize      *= chunkDims[dim];
  /* hdf5 chunks must be less than 4GB */
  if (chunksize > PETSC_HDF5_MAX_CHUNKSIZE/64) {
    if (bs > 1 || dim2) {
      if (chunkDims[dim-2] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/128))) {
        chunkDims[dim-2] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/128));
      } if (chunkDims[dim-1] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/128))) {
        chunkDims[dim-1] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/128));
      }
    } else {
      chunkDims[dim-1] = PETSC_HDF5_MAX_CHUNKSIZE/128;
    }
  }
  ++dim;
#else
  /* hdf5 chunks must be less than 4GB */
  if (chunksize > PETSC_HDF5_MAX_CHUNKSIZE/64) {
    if (bs > 1 || dim2) {
      if (chunkDims[dim-2] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64))) {
        chunkDims[dim-2] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64));
      } if (chunkDims[dim-1] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64))) {
        chunkDims[dim-1] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64));
      }
    } else {
      chunkDims[dim-1] = PETSC_HDF5_MAX_CHUNKSIZE/64;
    }
  }
#endif

  PetscStackCallHDF5Return(filespace,H5Screate_simple,(dim, dims, maxDims));

#if defined(PETSC_USE_REAL_SINGLE)
  memscalartype = H5T_NATIVE_FLOAT;
  filescalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  memscalartype = H5T_NATIVE_DOUBLE;
  if (spoutput == PETSC_TRUE) filescalartype = H5T_NATIVE_FLOAT;
  else filescalartype = H5T_NATIVE_DOUBLE;
#endif

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetName((PetscObject) xin, &vecname);CHKERRQ(ierr);
  if (H5Lexists(group, vecname, H5P_DEFAULT) < 1) {
    /* Create chunk */
    PetscStackCallHDF5Return(chunkspace,H5Pcreate,(H5P_DATASET_CREATE));
    PetscStackCallHDF5(H5Pset_chunk,(chunkspace, dim, chunkDims));

    PetscStackCallHDF5Return(dset_id,H5Dcreate2,(group, vecname, filescalartype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
    PetscStackCallHDF5(H5Pclose,(chunkspace));
  } else {
    PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname, H5P_DEFAULT));
    PetscStackCallHDF5(H5Dset_extent,(dset_id, dims));
  }
  PetscStackCallHDF5(H5Sclose,(filespace));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(xin->map->n/bs,count + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    count[dim] = bs;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  count[dim] = 2;
  ++dim;
#endif
  if (xin->map->n > 0 || H5_VERSION_GE(1,10,0)) {
    PetscStackCallHDF5Return(memspace,H5Screate_simple,(dim, count, NULL));
  } else {
    /* Can't create dataspace with zero for any dimension, so create null dataspace. */
    PetscStackCallHDF5Return(memspace,H5Screate,(H5S_NULL));
  }

  /* Select hyperslab in the file */
  ierr = VecGetOwnershipRange(xin, &low, NULL);CHKERRQ(ierr);
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  ierr = PetscHDF5IntCast(low/bs,offset + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    offset[dim] = 0;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  offset[dim] = 0;
  ++dim;
#endif
  if (xin->map->n > 0 || H5_VERSION_GE(1,10,0)) {
    PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
    PetscStackCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));
  } else {
    /* Create null filespace to match null memspace. */
    PetscStackCallHDF5Return(filespace,H5Screate,(H5S_NULL));
  }

  ierr   = VecGetArrayRead(xin, &x);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Dwrite,(dset_id, memscalartype, memspace, filespace, hdf5->dxpl_id, x));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  ierr   = VecRestoreArrayRead(xin, &x);CHKERRQ(ierr);

  /* Close/release resources */
  PetscStackCallHDF5(H5Gclose,(group));
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));

#if defined(PETSC_USE_COMPLEX)
  {
    PetscBool   tru = PETSC_TRUE;
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"complex",PETSC_BOOL,&tru);CHKERRQ(ierr);
  }
#endif
  if (timestepping) {
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"timestepping",PETSC_BOOL,&timestepping);CHKERRQ(ierr);
  }
  ierr = PetscInfo1(xin,"Wrote Vec object with name %s\n",vecname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscBool      ismathematica;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool      ismatlab;
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscBool      isadios;
#endif
  PetscBool      isglvis;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATHEMATICA,&ismathematica);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ADIOS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios);CHKERRQ(ierr);
#endif
  if (iascii) {
    ierr = VecView_MPI_ASCII(xin,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = VecView_MPI_Binary(xin,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_DRAW_LG) {
      ierr = VecView_MPI_Draw_LG(xin,viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView_MPI_Draw(xin,viewer);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    ierr = PetscViewerMathematicaPutVector(viewer,xin);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = VecView_MPI_HDF5(xin,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    ierr = VecView_MPI_ADIOS(xin,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = VecView_MPI_Matlab(xin,viewer);CHKERRQ(ierr);
#endif
  } else if (isglvis) {
    ierr = VecView_GLVis(xin,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetSize_MPI(Vec xin,PetscInt *N)
{
  PetscFunctionBegin;
  *N = xin->map->N;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetValues_MPI(Vec xin,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  const PetscScalar *xx;
  PetscInt          i,tmp,start = xin->map->range[xin->stash.rank];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
    tmp = ix[i] - start;
    if (PetscUnlikelyDebug(tmp < 0 || tmp >= xin->map->n)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only get local values, trying %D",ix[i]);
    y[i] = xx[tmp];
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValues_MPI(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode addv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank    = xin->stash.rank;
  PetscInt       *owners = xin->map->range,start = owners[rank];
  PetscInt       end     = owners[rank+1],i,row;
  PetscScalar    *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    if (xin->stash.insertmode == INSERT_VALUES && addv == ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else if (xin->stash.insertmode == ADD_VALUES && addv == INSERT_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      if (PetscUnlikelyDebug(ix[i] < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] = y[i];
      } else if (!xin->stash.donotstash) {
        if (PetscUnlikelyDebug(ix[i] >= xin->map->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->N);
        ierr = VecStashValue_Private(&xin->stash,row,y[i]);CHKERRQ(ierr);
      }
    }
  } else {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      if (PetscUnlikelyDebug(ix[i] < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] += y[i];
      } else if (!xin->stash.donotstash) {
        if (PetscUnlikelyDebug(ix[i] >= xin->map->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->N);
        ierr = VecStashValue_Private(&xin->stash,row,y[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesBlocked_MPI(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode addv)
{
  PetscMPIInt    rank    = xin->stash.rank;
  PetscInt       *owners = xin->map->range,start = owners[rank];
  PetscErrorCode ierr;
  PetscInt       end = owners[rank+1],i,row,bs = PetscAbs(xin->map->bs),j;
  PetscScalar    *xx,*y = (PetscScalar*)yin;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (xin->stash.insertmode == INSERT_VALUES && addv == ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else if (xin->stash.insertmode == ADD_VALUES && addv == INSERT_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) xx[row-start+j] = y[j];
      } else if (!xin->stash.donotstash) {
        if (ix[i] < 0) { y += bs; continue; }
        if (PetscUnlikelyDebug(ix[i] >= xin->map->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D max %D",ix[i],xin->map->N);
        ierr = VecStashValuesBlocked_Private(&xin->bstash,ix[i],y);CHKERRQ(ierr);
      }
      y += bs;
    }
  } else {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) xx[row-start+j] += y[j];
      } else if (!xin->stash.donotstash) {
        if (ix[i] < 0) { y += bs; continue; }
        if (PetscUnlikelyDebug(ix[i] > xin->map->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D max %D",ix[i],xin->map->N);
        ierr = VecStashValuesBlocked_Private(&xin->bstash,ix[i],y);CHKERRQ(ierr);
      }
      y += bs;
    }
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Since nsends or nreceives may be zero we add 1 in certain mallocs
to make sure we never malloc an empty one.
*/
PetscErrorCode VecAssemblyBegin_MPI(Vec xin)
{
  PetscErrorCode ierr;
  PetscInt       *owners = xin->map->range,*bowners,i,bs,nstash,reallocs;
  PetscMPIInt    size;
  InsertMode     addv;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  if (xin->stash.donotstash) PetscFunctionReturn(0);

  ierr = MPIU_Allreduce((PetscEnum*)&xin->stash.insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,comm);CHKERRMPI(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(comm,PETSC_ERR_ARG_NOTSAMETYPE,"Some processors inserted values while others added");
  xin->stash.insertmode = addv; /* in case this processor had no cache */
  xin->bstash.insertmode = addv; /* Block stash implicitly tracks InsertMode of scalar stash */

  ierr = VecGetBlockSize(xin,&bs);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size);CHKERRMPI(ierr);
  if (!xin->bstash.bowners && xin->map->bs != -1) {
    ierr = PetscMalloc1(size+1,&bowners);CHKERRQ(ierr);
    for (i=0; i<size+1; i++) bowners[i] = owners[i]/bs;
    xin->bstash.bowners = bowners;
  } else bowners = xin->bstash.bowners;

  ierr = VecStashScatterBegin_Private(&xin->stash,owners);CHKERRQ(ierr);
  ierr = VecStashScatterBegin_Private(&xin->bstash,bowners);CHKERRQ(ierr);
  ierr = VecStashGetInfo_Private(&xin->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(xin,"Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  ierr = VecStashGetInfo_Private(&xin->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(xin,"Block-Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAssemblyEnd_MPI(Vec vec)
{
  PetscErrorCode ierr;
  PetscInt       base,i,j,*row,flg,bs;
  PetscMPIInt    n;
  PetscScalar    *val,*vv,*array,*xarray;

  PetscFunctionBegin;
  if (!vec->stash.donotstash) {
    ierr = VecGetArray(vec,&xarray);CHKERRQ(ierr);
    ierr = VecGetBlockSize(vec,&bs);CHKERRQ(ierr);
    base = vec->map->range[vec->stash.rank];

    /* Process the stash */
    while (1) {
      ierr = VecStashScatterGetMesg_Private(&vec->stash,&n,&row,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      if (vec->stash.insertmode == ADD_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] += val[i];
      } else if (vec->stash.insertmode == INSERT_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] = val[i];
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Insert mode is not set correctly; corrupted vector");
    }
    ierr = VecStashScatterEnd_Private(&vec->stash);CHKERRQ(ierr);

    /* now process the block-stash */
    while (1) {
      ierr = VecStashScatterGetMesg_Private(&vec->bstash,&n,&row,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      for (i=0; i<n; i++) {
        array = xarray+row[i]*bs-base;
        vv    = val+i*bs;
        if (vec->stash.insertmode == ADD_VALUES) {
          for (j=0; j<bs; j++) array[j] += vv[j];
        } else if (vec->stash.insertmode == INSERT_VALUES) {
          for (j=0; j<bs; j++) array[j] = vv[j];
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Insert mode is not set correctly; corrupted vector");
      }
    }
    ierr = VecStashScatterEnd_Private(&vec->bstash);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&xarray);CHKERRQ(ierr);
  }
  vec->stash.insertmode = NOT_SET_VALUES;
  PetscFunctionReturn(0);
}
