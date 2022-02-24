
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

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->N);
#endif
  if (!x) PetscFunctionReturn(0);
  CHKERRQ(PetscFree(x->array_allocated));

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    CHKERRQ(VecDestroy(&x->localrep));
    CHKERRQ(VecScatterDestroy(&x->localupdate));
  }
  CHKERRQ(VecAssemblyReset_MPI(v));

  /* Destroy the stashes: note the order - so that the tags are freed properly */
  CHKERRQ(VecStashDestroy_Private(&v->bstash));
  CHKERRQ(VecStashDestroy_Private(&v->stash));
  CHKERRQ(PetscFree(v->data));
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_MPI_ASCII(Vec xin,PetscViewer viewer)
{
  PetscInt          i,work = xin->map->n,cnt,len,nLen;
  PetscMPIInt       j,n = 0,size,rank,tag = ((PetscObject)viewer)->tag;
  MPI_Status        status;
  PetscScalar       *values;
  const PetscScalar *xarray;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_LOAD_BALANCE) {
    PetscInt nmax = 0,nmin = xin->map->n,navg;
    for (i=0; i<(PetscInt)size; i++) {
      nmax = PetscMax(nmax,xin->map->range[i+1] - xin->map->range[i]);
      nmin = PetscMin(nmin,xin->map->range[i+1] - xin->map->range[i]);
    }
    navg = xin->map->N/size;
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Load Balance - Local vector size Min %" PetscInt_FMT "  avg %" PetscInt_FMT "  max %" PetscInt_FMT "\n",nmin,navg,nmax));
    PetscFunctionReturn(0);
  }

  CHKERRQ(VecGetArrayRead(xin,&xarray));
  /* determine maximum message to arrive */
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  CHKERRMPI(MPI_Reduce(&work,&len,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)xin)));
  if (format == PETSC_VIEWER_ASCII_GLVIS) { rank = 0, len = 0; } /* no parallel distributed write support from GLVis */
  if (rank == 0) {
    CHKERRQ(PetscMalloc1(len,&values));
    /*
        MATLAB format and ASCII format are very similar except
        MATLAB uses %18.16e format while ASCII uses %g
    */
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      CHKERRQ(PetscObjectGetName((PetscObject)xin,&name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s = [\n",name));
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i])));
        } else {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(xarray[i])));
        }
#else
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i])));
          } else {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(values[i])));
          }
#else
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]));
#endif
        }
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));

    } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
#else
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
#else
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]));
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
        CHKERRQ(PetscObjectComposedDataRegister(&stateId));
      }
      CHKERRQ(PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState));
      if (!hasState) outputState = 0;

      CHKERRQ(PetscObjectGetName((PetscObject)xin,&name));
      CHKERRQ(VecGetLocalSize(xin, &nLen));
      CHKERRQ(PetscMPIIntCast(nLen,&n));
      CHKERRQ(VecGetBlockSize(xin, &bs));
      if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED) {
        if (outputState == 0) {
          outputState = 1;
          doOutput    = 1;
        } else if (outputState == 1) doOutput = 0;
        else if (outputState == 2) {
          outputState = 3;
          doOutput    = 1;
        } else if (outputState == 3) doOutput = 0;
        else PetscCheckFalse(outputState == 4,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

        if (doOutput) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "POINT_DATA %" PetscInt_FMT "\n", xin->map->N/bs));
        }
      } else {
        if (outputState == 0) {
          outputState = 2;
          doOutput    = 1;
        } else if (outputState == 1) {
          outputState = 4;
          doOutput    = 1;
        } else if (outputState == 2) doOutput = 0;
        else PetscCheckFalse(outputState == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
        else if (outputState == 4) doOutput = 0;

        if (doOutput) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "CELL_DATA %" PetscInt_FMT "\n", xin->map->N/bs));
        }
      }
      CHKERRQ(PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState));
      if (name) {
        if (bs == 3) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name));
        } else {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "SCALARS %s double %" PetscInt_FMT "\n", name, bs));
        }
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %" PetscInt_FMT "\n", bs));
      }
      if (bs != 3) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"));
      }
      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b])));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
            }
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b])));
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED) {
      PetscInt bs, b;

      CHKERRQ(VecGetLocalSize(xin, &nLen));
      CHKERRQ(PetscMPIIntCast(nLen,&n));
      CHKERRQ(VecGetBlockSize(xin, &bs));
      PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);

      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b])));
        }
        for (b=bs; b<3; b++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer," 0.0"));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
            }
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b])));
          }
          for (b=bs; b<3; b++) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer," 0.0"));
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_PCICE) {
      PetscInt bs, b, vertexCount = 1;

      CHKERRQ(VecGetLocalSize(xin, &nLen));
      CHKERRQ(PetscMPIIntCast(nLen,&n));
      CHKERRQ(VecGetBlockSize(xin, &bs));
      PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %" PetscInt_FMT, bs);

      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT "\n", xin->map->N/bs));
      for (i=0; i<n/bs; i++) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%7" PetscInt_FMT "   ", vertexCount++));
        for (b=0; b<bs; b++) {
          if (b > 0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
          }
#if !defined(PETSC_USE_COMPLEX)
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)xarray[i*bs+b]));
#endif
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%7" PetscInt_FMT "   ", vertexCount++));
          for (b=0; b<bs; b++) {
            if (b > 0) {
              CHKERRQ(PetscViewerASCIIPrintf(viewer," "));
            }
#if !defined(PETSC_USE_COMPLEX)
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)values[i*bs+b]));
#endif
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
      /* GLVis ASCII visualization/dump: this function mimics mfem::GridFunction::Save() */
      const PetscScalar       *array;
      PetscInt                i,n,vdim, ordering = 1; /* mfem::FiniteElementSpace::Ordering::byVDIM */
      PetscContainer          glvis_container;
      PetscViewerGLVisVecInfo glvis_vec_info;
      PetscViewerGLVisInfo    glvis_info;

      /* mfem::FiniteElementSpace::Save() */
      CHKERRQ(VecGetBlockSize(xin,&vdim));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n"));
      CHKERRQ(PetscObjectQuery((PetscObject)xin,"_glvis_info_container",(PetscObject*)&glvis_container));
      PetscCheckFalse(!glvis_container,PetscObjectComm((PetscObject)xin),PETSC_ERR_PLIB,"Missing GLVis container");
      CHKERRQ(PetscContainerGetPointer(glvis_container,(void**)&glvis_vec_info));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",glvis_vec_info->fec_type));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"VDim: %" PetscInt_FMT "\n",vdim));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Ordering: %" PetscInt_FMT "\n",ordering));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      /* mfem::Vector::Print() */
      CHKERRQ(PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container));
      PetscCheckFalse(!glvis_container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_PLIB,"Missing GLVis container");
      CHKERRQ(PetscContainerGetPointer(glvis_container,(void**)&glvis_info));
      if (glvis_info->enabled) {
        CHKERRQ(VecGetLocalSize(xin,&n));
        CHKERRQ(VecGetArrayRead(xin,&array));
        for (i=0;i<n;i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,glvis_info->fmt,(double)PetscRealPart(array[i])));
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
        }
        CHKERRQ(VecRestoreArrayRead(xin,&array));
      }
    } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* No info */
    } else {
      if (format != PETSC_VIEWER_ASCII_COMMON) CHKERRQ(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank));
      cnt = 0;
      for (i=0; i<xin->map->n; i++) {
        if (format == PETSC_VIEWER_ASCII_INDEX) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",cnt++));
        }
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i])));
        } else {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(xarray[i])));
        }
#else
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        CHKERRMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        CHKERRMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        if (format != PETSC_VIEWER_ASCII_COMMON) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",j));
        }
        for (i=0; i<n; i++) {
          if (format == PETSC_VIEWER_ASCII_INDEX) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",cnt++));
          }
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i])));
          } else {
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(values[i])));
          }
#else
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g\n",(double)values[i]));
#endif
        }
      }
    }
    CHKERRQ(PetscFree(values));
  } else {
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* Rank 0 is not trying to receive anything, so don't send anything */
    } else {
      if (format == PETSC_VIEWER_ASCII_MATLAB || format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED) {
        /* this may be a collective operation so make sure everyone calls it */
        CHKERRQ(PetscObjectGetName((PetscObject)xin,&name));
      }
      CHKERRMPI(MPI_Send((void*)xarray,xin->map->n,MPIU_SCALAR,0,tag,PetscObjectComm((PetscObject)xin)));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(VecRestoreArrayRead(xin,&xarray));
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

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  CHKERRQ(PetscMPIIntCast(xin->map->n,&n));
  CHKERRQ(PetscMPIIntCast(xin->map->N,&N));

  CHKERRQ(VecGetArrayRead(xin,&xarray));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscMalloc1(n+1,&values));
  for (i=0; i<n; i++) values[i] = PetscRealPart(xarray[i]);
#else
  values = (PetscReal*)xarray;
#endif
  if (rank == 0) {
    CHKERRQ(PetscMalloc2(N,&xx,N,&yy));
    for (i=0; i<N; i++) xx[i] = (PetscReal)i;
    CHKERRQ(PetscMalloc2(size,&lens,size,&disp));
    for (i=0; i<size; i++) lens[i] = (PetscMPIInt)xin->map->range[i+1] - (PetscMPIInt)xin->map->range[i];
    for (i=0; i<size; i++) disp[i] = (PetscMPIInt)xin->map->range[i];
  }
  CHKERRMPI(MPI_Gatherv(values,n,MPIU_REAL,yy,lens,disp,MPIU_REAL,0,PetscObjectComm((PetscObject)xin)));
  CHKERRQ(PetscFree2(lens,disp));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscFree(values));
#endif
  CHKERRQ(VecRestoreArrayRead(xin,&xarray));

  CHKERRQ(PetscViewerDrawGetDrawLG(viewer,0,&lg));
  CHKERRQ(PetscDrawLGReset(lg));
  CHKERRQ(PetscDrawLGSetDimension(lg,1));
  CHKERRQ(PetscDrawLGSetColors(lg,colors));
  if (rank == 0) {
    CHKERRQ(PetscDrawLGAddPoints(lg,N,&xx,&yy));
    CHKERRQ(PetscFree2(xx,yy));
  }
  CHKERRQ(PetscDrawLGDraw(lg));
  CHKERRQ(PetscDrawLGSave(lg));
  PetscFunctionReturn(0);
}

PetscErrorCode  VecView_MPI_Draw(Vec xin,PetscViewer viewer)
{
  PetscMPIInt       rank,size,tag = ((PetscObject)viewer)->tag;
  PetscInt          i,start,end;
  MPI_Status        status;
  PetscReal         min,max,tmp = 0.0;
  PetscDraw         draw;
  PetscBool         isnull;
  PetscDrawAxis     axis;
  const PetscScalar *xarray;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));

  CHKERRQ(VecMin(xin,NULL,&min));
  CHKERRQ(VecMax(xin,NULL,&max));
  if (min == max) {
    min -= 1.e-5;
    max += 1.e-5;
  }

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawClear(draw));

  CHKERRQ(PetscDrawAxisCreate(draw,&axis));
  CHKERRQ(PetscDrawAxisSetLimits(axis,0.0,(PetscReal)xin->map->N,min,max));
  CHKERRQ(PetscDrawAxisDraw(axis));
  CHKERRQ(PetscDrawAxisDestroy(&axis));

  /* draw local part of vector */
  CHKERRQ(VecGetArrayRead(xin,&xarray));
  CHKERRQ(VecGetOwnershipRange(xin,&start,&end));
  if (rank < size-1) { /* send value to right */
    CHKERRMPI(MPI_Send((void*)&xarray[xin->map->n-1],1,MPIU_REAL,rank+1,tag,PetscObjectComm((PetscObject)xin)));
  }
  if (rank) { /* receive value from right */
    CHKERRMPI(MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,PetscObjectComm((PetscObject)xin),&status));
  }
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank) {
    CHKERRQ(PetscDrawLine(draw,(PetscReal)start-1,tmp,(PetscReal)start,PetscRealPart(xarray[0]),PETSC_DRAW_RED));
  }
  for (i=1; i<xin->map->n; i++) {
    CHKERRQ(PetscDrawLine(draw,(PetscReal)(i-1+start),PetscRealPart(xarray[i-1]),(PetscReal)(i+start),PetscRealPart(xarray[i]),PETSC_DRAW_RED));
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  CHKERRQ(VecRestoreArrayRead(xin,&xarray));

  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
PetscErrorCode VecView_MPI_Matlab(Vec xin,PetscViewer viewer)
{
  PetscMPIInt       rank,size,*lens;
  PetscInt          i,N = xin->map->N;
  const PetscScalar *xarray;
  PetscScalar       *xx;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(xin,&xarray));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  if (rank == 0) {
    CHKERRQ(PetscMalloc1(N,&xx));
    CHKERRQ(PetscMalloc1(size,&lens));
    for (i=0; i<size; i++) lens[i] = xin->map->range[i+1] - xin->map->range[i];

    CHKERRMPI(MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,xx,lens,xin->map->range,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin)));
    CHKERRQ(PetscFree(lens));

    CHKERRQ(PetscObjectName((PetscObject)xin));
    CHKERRQ(PetscViewerMatlabPutArray(viewer,N,1,xx,((PetscObject)xin)->name));

    CHKERRQ(PetscFree(xx));
  } else {
    CHKERRMPI(MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,0,0,0,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin)));
  }
  CHKERRQ(VecRestoreArrayRead(xin,&xarray));
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
  const char        *vecname;
  int64_t           id;
  PetscInt          n,N,rstart;
  const PetscScalar *array;
  char              nglobalname[16],nlocalname[16],coffset[16];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject) xin, &vecname));

  CHKERRQ(VecGetLocalSize(xin,&n));
  CHKERRQ(VecGetSize(xin,&N));
  CHKERRQ(VecGetOwnershipRange(xin,&rstart,NULL));

  sprintf(nlocalname,"%d",(int)n);
  sprintf(nglobalname,"%d",(int)N);
  sprintf(coffset,"%d",(int)rstart);
  id   = adios_define_var(Petsc_adios_group,vecname,"",adios_double,nlocalname,nglobalname,coffset);
  CHKERRQ(VecGetArrayRead(xin,&array));
  CHKERRQ(adios_write_byid(adios->adios_handle,id,array));
  CHKERRQ(VecRestoreArrayRead(xin,&array));
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

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5OpenGroup(viewer, &file_id, &group));
  CHKERRQ(PetscViewerHDF5IsTimestepping(viewer, &timestepping));
  if (timestepping) {
    CHKERRQ(PetscViewerHDF5GetTimestep(viewer, &timestep));
  }
  CHKERRQ(PetscViewerHDF5GetBaseDimension2(viewer,&dim2));
  CHKERRQ(PetscViewerHDF5GetSPOutput(viewer,&spoutput));

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
  CHKERRQ(PetscHDF5IntCast(xin->map->N/bs,dims + dim));

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
  CHKERRQ(PetscObjectGetName((PetscObject) xin, &vecname));
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
  CHKERRQ(PetscHDF5IntCast(xin->map->n/bs,count + dim));
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
  CHKERRQ(VecGetOwnershipRange(xin, &low, NULL));
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  CHKERRQ(PetscHDF5IntCast(low/bs,offset + dim));
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

  CHKERRQ(VecGetArrayRead(xin, &x));
  PetscStackCallHDF5(H5Dwrite,(dset_id, memscalartype, memspace, filespace, hdf5->dxpl_id, x));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  CHKERRQ(VecRestoreArrayRead(xin, &x));

  /* Close/release resources */
  PetscStackCallHDF5(H5Gclose,(group));
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));

#if defined(PETSC_USE_COMPLEX)
  {
    PetscBool   tru = PETSC_TRUE;
    CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"complex",PETSC_BOOL,&tru));
  }
#endif
  if (timestepping) {
    CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"timestepping",PETSC_BOOL,&timestepping));
  }
  CHKERRQ(PetscInfo(xin,"Wrote Vec object with name %s\n",vecname));
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec xin,PetscViewer viewer)
{
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
#if defined(PETSC_HAVE_MATHEMATICA)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATHEMATICA,&ismathematica));
#endif
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab));
#endif
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis));
#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
#endif
  if (iascii) {
    CHKERRQ(VecView_MPI_ASCII(xin,viewer));
  } else if (isbinary) {
    CHKERRQ(VecView_MPI_Binary(xin,viewer));
  } else if (isdraw) {
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_DRAW_LG) {
      CHKERRQ(VecView_MPI_Draw_LG(xin,viewer));
    } else {
      CHKERRQ(VecView_MPI_Draw(xin,viewer));
    }
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    CHKERRQ(PetscViewerMathematicaPutVector(viewer,xin));
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(VecView_MPI_HDF5(xin,viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    CHKERRQ(VecView_MPI_ADIOS(xin,viewer));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    CHKERRQ(VecView_MPI_Matlab(xin,viewer));
#endif
  } else if (isglvis) {
    CHKERRQ(VecView_GLVis(xin,viewer));
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

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(xin,&xx));
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
    tmp = ix[i] - start;
    PetscCheck(tmp >= 0 && tmp < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only get local values, trying %" PetscInt_FMT,ix[i]);
    y[i] = xx[tmp];
  }
  CHKERRQ(VecRestoreArrayRead(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValues_MPI(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode addv)
{
  PetscMPIInt    rank    = xin->stash.rank;
  PetscInt       *owners = xin->map->range,start = owners[rank];
  PetscInt       end     = owners[rank+1],i,row;
  PetscScalar    *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCheckFalse(xin->stash.insertmode == INSERT_VALUES && addv == ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else PetscCheckFalse(xin->stash.insertmode == ADD_VALUES && addv == INSERT_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  CHKERRQ(VecGetArray(xin,&xx));
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] = y[i];
      } else if (!xin->stash.donotstash) {
        PetscCheck(ix[i] < xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,ix[i],xin->map->N);
        CHKERRQ(VecStashValue_Private(&xin->stash,row,y[i]));
      }
    }
  } else {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] += y[i];
      } else if (!xin->stash.donotstash) {
        PetscCheck(ix[i] < xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,ix[i],xin->map->N);
        CHKERRQ(VecStashValue_Private(&xin->stash,row,y[i]));
      }
    }
  }
  CHKERRQ(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesBlocked_MPI(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode addv)
{
  PetscMPIInt    rank    = xin->stash.rank;
  PetscInt       *owners = xin->map->range,start = owners[rank];
  PetscInt       end = owners[rank+1],i,row,bs = PetscAbs(xin->map->bs),j;
  PetscScalar    *xx,*y = (PetscScalar*)yin;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(xin,&xx));
  if (PetscDefined(USE_DEBUG)) {
    PetscCheckFalse(xin->stash.insertmode == INSERT_VALUES && addv == ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else PetscCheckFalse(xin->stash.insertmode == ADD_VALUES && addv == INSERT_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) xx[row-start+j] = y[j];
      } else if (!xin->stash.donotstash) {
        if (ix[i] < 0) { y += bs; continue; }
        PetscCheck(ix[i] < xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " max %" PetscInt_FMT,ix[i],xin->map->N);
        CHKERRQ(VecStashValuesBlocked_Private(&xin->bstash,ix[i],y));
      }
      y += bs;
    }
  } else {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) xx[row-start+j] += y[j];
      } else if (!xin->stash.donotstash) {
        if (ix[i] < 0) { y += bs; continue; }
        PetscCheck(ix[i] <= xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " max %" PetscInt_FMT,ix[i],xin->map->N);
        CHKERRQ(VecStashValuesBlocked_Private(&xin->bstash,ix[i],y));
      }
      y += bs;
    }
  }
  CHKERRQ(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

/*
   Since nsends or nreceives may be zero we add 1 in certain mallocs
to make sure we never malloc an empty one.
*/
PetscErrorCode VecAssemblyBegin_MPI(Vec xin)
{
  PetscInt       *owners = xin->map->range,*bowners,i,bs,nstash,reallocs;
  PetscMPIInt    size;
  InsertMode     addv;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)xin,&comm));
  if (xin->stash.donotstash) PetscFunctionReturn(0);

  CHKERRMPI(MPIU_Allreduce((PetscEnum*)&xin->stash.insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,comm));
  PetscCheckFalse(addv == (ADD_VALUES|INSERT_VALUES),comm,PETSC_ERR_ARG_NOTSAMETYPE,"Some processors inserted values while others added");
  xin->stash.insertmode = addv; /* in case this processor had no cache */
  xin->bstash.insertmode = addv; /* Block stash implicitly tracks InsertMode of scalar stash */

  CHKERRQ(VecGetBlockSize(xin,&bs));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  if (!xin->bstash.bowners && xin->map->bs != -1) {
    CHKERRQ(PetscMalloc1(size+1,&bowners));
    for (i=0; i<size+1; i++) bowners[i] = owners[i]/bs;
    xin->bstash.bowners = bowners;
  } else bowners = xin->bstash.bowners;

  CHKERRQ(VecStashScatterBegin_Private(&xin->stash,owners));
  CHKERRQ(VecStashScatterBegin_Private(&xin->bstash,bowners));
  CHKERRQ(VecStashGetInfo_Private(&xin->stash,&nstash,&reallocs));
  CHKERRQ(PetscInfo(xin,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  CHKERRQ(VecStashGetInfo_Private(&xin->bstash,&nstash,&reallocs));
  CHKERRQ(PetscInfo(xin,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAssemblyEnd_MPI(Vec vec)
{
  PetscInt       base,i,j,*row,flg,bs;
  PetscMPIInt    n;
  PetscScalar    *val,*vv,*array,*xarray;

  PetscFunctionBegin;
  if (!vec->stash.donotstash) {
    CHKERRQ(VecGetArray(vec,&xarray));
    CHKERRQ(VecGetBlockSize(vec,&bs));
    base = vec->map->range[vec->stash.rank];

    /* Process the stash */
    while (1) {
      CHKERRQ(VecStashScatterGetMesg_Private(&vec->stash,&n,&row,&val,&flg));
      if (!flg) break;
      if (vec->stash.insertmode == ADD_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] += val[i];
      } else if (vec->stash.insertmode == INSERT_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] = val[i];
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Insert mode is not set correctly; corrupted vector");
    }
    CHKERRQ(VecStashScatterEnd_Private(&vec->stash));

    /* now process the block-stash */
    while (1) {
      CHKERRQ(VecStashScatterGetMesg_Private(&vec->bstash,&n,&row,&val,&flg));
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
    CHKERRQ(VecStashScatterEnd_Private(&vec->bstash));
    CHKERRQ(VecRestoreArray(vec,&xarray));
  }
  vec->stash.insertmode = NOT_SET_VALUES;
  PetscFunctionReturn(0);
}
