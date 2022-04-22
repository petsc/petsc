
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <petsc/private/viewerhdf5impl.h>
#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscsf.h>

static PetscErrorCode VecResetPreallocationCOO_MPI(Vec v)
{
  Vec_MPI        *vmpi = (Vec_MPI*)v->data;

  PetscFunctionBegin;
  if (vmpi) {
    PetscCall(PetscFree(vmpi->jmap1));
    PetscCall(PetscFree(vmpi->perm1));
    PetscCall(PetscFree(vmpi->Cperm));
    PetscCall(PetscFree4(vmpi->imap2,vmpi->jmap2,vmpi->sendbuf,vmpi->recvbuf));
    PetscCall(PetscFree(vmpi->perm2));
    PetscCall(PetscSFDestroy(&vmpi->coo_sf));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_MPI(Vec v)
{
  Vec_MPI        *x = (Vec_MPI*)v->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->N);
#endif
  if (!x) PetscFunctionReturn(0);
  PetscCall(PetscFree(x->array_allocated));

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    PetscCall(VecDestroy(&x->localrep));
    PetscCall(VecScatterDestroy(&x->localupdate));
  }
  PetscCall(VecAssemblyReset_MPI(v));

  /* Destroy the stashes: note the order - so that the tags are freed properly */
  PetscCall(VecStashDestroy_Private(&v->bstash));
  PetscCall(VecStashDestroy_Private(&v->stash));

  PetscCall(VecResetPreallocationCOO_MPI(v));
  PetscCall(PetscFree(v->data));
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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_LOAD_BALANCE) {
    PetscInt nmax = 0,nmin = xin->map->n,navg;
    for (i=0; i<(PetscInt)size; i++) {
      nmax = PetscMax(nmax,xin->map->range[i+1] - xin->map->range[i]);
      nmin = PetscMin(nmin,xin->map->range[i+1] - xin->map->range[i]);
    }
    navg = xin->map->N/size;
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Load Balance - Local vector size Min %" PetscInt_FMT "  avg %" PetscInt_FMT "  max %" PetscInt_FMT "\n",nmin,navg,nmax));
    PetscFunctionReturn(0);
  }

  PetscCall(VecGetArrayRead(xin,&xarray));
  /* determine maximum message to arrive */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  PetscCallMPI(MPI_Reduce(&work,&len,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)xin)));
  if (format == PETSC_VIEWER_ASCII_GLVIS) { rank = 0, len = 0; } /* no parallel distributed write support from GLVis */
  if (rank == 0) {
    PetscCall(PetscMalloc1(len,&values));
    /*
        MATLAB format and ASCII format are very similar except
        MATLAB uses %18.16e format while ASCII uses %g
    */
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      PetscCall(PetscObjectGetName((PetscObject)xin,&name));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s = [\n",name));
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i])));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(xarray[i])));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i])));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(values[i])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]));
#endif
        }
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));

    } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
      for (i=0; i<xin->map->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
#else
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)values[i]));
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
        PetscCall(PetscObjectComposedDataRegister(&stateId));
      }
      PetscCall(PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState));
      if (!hasState) outputState = 0;

      PetscCall(PetscObjectGetName((PetscObject)xin,&name));
      PetscCall(VecGetLocalSize(xin, &nLen));
      PetscCall(PetscMPIIntCast(nLen,&n));
      PetscCall(VecGetBlockSize(xin, &bs));
      if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED) {
        if (outputState == 0) {
          outputState = 1;
          doOutput    = 1;
        } else if (outputState == 1) doOutput = 0;
        else if (outputState == 2) {
          outputState = 3;
          doOutput    = 1;
        } else if (outputState == 3) doOutput = 0;
        else PetscCheck(outputState != 4,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

        if (doOutput) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "POINT_DATA %" PetscInt_FMT "\n", xin->map->N/bs));
        }
      } else {
        if (outputState == 0) {
          outputState = 2;
          doOutput    = 1;
        } else if (outputState == 1) {
          outputState = 4;
          doOutput    = 1;
        } else if (outputState == 2) doOutput = 0;
        else PetscCheck(outputState != 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
        else if (outputState == 4) doOutput = 0;

        if (doOutput) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "CELL_DATA %" PetscInt_FMT "\n", xin->map->N/bs));
        }
      }
      PetscCall(PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState));
      if (name) {
        if (bs == 3) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS %s double %" PetscInt_FMT "\n", name, bs));
        }
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %" PetscInt_FMT "\n", bs));
      }
      if (bs != 3) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"));
      }
      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," "));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b])));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," "));
            }
            PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b])));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED) {
      PetscInt bs, b;

      PetscCall(VecGetLocalSize(xin, &nLen));
      PetscCall(PetscMPIIntCast(nLen,&n));
      PetscCall(VecGetBlockSize(xin, &bs));
      PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);

      for (i=0; i<n/bs; i++) {
        for (b=0; b<bs; b++) {
          if (b > 0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," "));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(xarray[i*bs+b])));
        }
        for (b=bs; b<3; b++) {
          PetscCall(PetscViewerASCIIPrintf(viewer," 0.0"));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          for (b=0; b<bs; b++) {
            if (b > 0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," "));
            }
            PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)PetscRealPart(values[i*bs+b])));
          }
          for (b=bs; b<3; b++) {
            PetscCall(PetscViewerASCIIPrintf(viewer," 0.0"));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        }
      }
    } else if (format == PETSC_VIEWER_ASCII_PCICE) {
      PetscInt bs, b, vertexCount = 1;

      PetscCall(VecGetLocalSize(xin, &nLen));
      PetscCall(PetscMPIIntCast(nLen,&n));
      PetscCall(VecGetBlockSize(xin, &bs));
      PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %" PetscInt_FMT, bs);

      PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT "\n", xin->map->N/bs));
      for (i=0; i<n/bs; i++) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%7" PetscInt_FMT "   ", vertexCount++));
        for (b=0; b<bs; b++) {
          if (b > 0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," "));
          }
#if !defined(PETSC_USE_COMPLEX)
          PetscCall(PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)xarray[i*bs+b]));
#endif
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        for (i=0; i<n/bs; i++) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%7" PetscInt_FMT "   ", vertexCount++));
          for (b=0; b<bs; b++) {
            if (b > 0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," "));
            }
#if !defined(PETSC_USE_COMPLEX)
            PetscCall(PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)values[i*bs+b]));
#endif
          }
          PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
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
      PetscCall(VecGetBlockSize(xin,&vdim));
      PetscCall(PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n"));
      PetscCall(PetscObjectQuery((PetscObject)xin,"_glvis_info_container",(PetscObject*)&glvis_container));
      PetscCheck(glvis_container,PetscObjectComm((PetscObject)xin),PETSC_ERR_PLIB,"Missing GLVis container");
      PetscCall(PetscContainerGetPointer(glvis_container,(void**)&glvis_vec_info));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",glvis_vec_info->fec_type));
      PetscCall(PetscViewerASCIIPrintf(viewer,"VDim: %" PetscInt_FMT "\n",vdim));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Ordering: %" PetscInt_FMT "\n",ordering));
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      /* mfem::Vector::Print() */
      PetscCall(PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container));
      PetscCheck(glvis_container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_PLIB,"Missing GLVis container");
      PetscCall(PetscContainerGetPointer(glvis_container,(void**)&glvis_info));
      if (glvis_info->enabled) {
        PetscCall(VecGetLocalSize(xin,&n));
        PetscCall(VecGetArrayRead(xin,&array));
        for (i=0;i<n;i++) {
          PetscCall(PetscViewerASCIIPrintf(viewer,glvis_info->fmt,(double)PetscRealPart(array[i])));
          PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        }
        PetscCall(VecRestoreArrayRead(xin,&array));
      }
    } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* No info */
    } else {
      if (format != PETSC_VIEWER_ASCII_COMMON) PetscCall(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank));
      cnt = 0;
      for (i=0; i<xin->map->n; i++) {
        if (format == PETSC_VIEWER_ASCII_INDEX) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",cnt++));
        }
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(xarray[i]) > 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(xarray[i]),(double)PetscImaginaryPart(xarray[i])));
        } else if (PetscImaginaryPart(xarray[i]) < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(xarray[i]),-(double)PetscImaginaryPart(xarray[i])));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(xarray[i])));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)xarray[i]));
#endif
      }
      /* receive and print messages */
      for (j=1; j<size; j++) {
        PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,PetscObjectComm((PetscObject)xin),&status));
        PetscCallMPI(MPI_Get_count(&status,MPIU_SCALAR,&n));
        if (format != PETSC_VIEWER_ASCII_COMMON) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",j));
        }
        for (i=0; i<n; i++) {
          if (format == PETSC_VIEWER_ASCII_INDEX) {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",cnt++));
          }
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(values[i]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(values[i]),(double)PetscImaginaryPart(values[i])));
          } else if (PetscImaginaryPart(values[i]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(values[i]),-(double)PetscImaginaryPart(values[i])));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(values[i])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)values[i]));
#endif
        }
      }
    }
    PetscCall(PetscFree(values));
  } else {
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      /* Rank 0 is not trying to receive anything, so don't send anything */
    } else {
      if (format == PETSC_VIEWER_ASCII_MATLAB || format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED) {
        /* this may be a collective operation so make sure everyone calls it */
        PetscCall(PetscObjectGetName((PetscObject)xin,&name));
      }
      PetscCallMPI(MPI_Send((void*)xarray,xin->map->n,MPIU_SCALAR,0,tag,PetscObjectComm((PetscObject)xin)));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecRestoreArrayRead(xin,&xarray));
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
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  PetscCall(PetscMPIIntCast(xin->map->n,&n));
  PetscCall(PetscMPIIntCast(xin->map->N,&N));

  PetscCall(VecGetArrayRead(xin,&xarray));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(n+1,&values));
  for (i=0; i<n; i++) values[i] = PetscRealPart(xarray[i]);
#else
  values = (PetscReal*)xarray;
#endif
  if (rank == 0) {
    PetscCall(PetscMalloc2(N,&xx,N,&yy));
    for (i=0; i<N; i++) xx[i] = (PetscReal)i;
    PetscCall(PetscMalloc2(size,&lens,size,&disp));
    for (i=0; i<size; i++) lens[i] = (PetscMPIInt)xin->map->range[i+1] - (PetscMPIInt)xin->map->range[i];
    for (i=0; i<size; i++) disp[i] = (PetscMPIInt)xin->map->range[i];
  }
  PetscCallMPI(MPI_Gatherv(values,n,MPIU_REAL,yy,lens,disp,MPIU_REAL,0,PetscObjectComm((PetscObject)xin)));
  PetscCall(PetscFree2(lens,disp));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(values));
#endif
  PetscCall(VecRestoreArrayRead(xin,&xarray));

  PetscCall(PetscViewerDrawGetDrawLG(viewer,0,&lg));
  PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGSetDimension(lg,1));
  PetscCall(PetscDrawLGSetColors(lg,colors));
  if (rank == 0) {
    PetscCall(PetscDrawLGAddPoints(lg,N,&xx,&yy));
    PetscCall(PetscFree2(xx,yy));
  }
  PetscCall(PetscDrawLGDraw(lg));
  PetscCall(PetscDrawLGSave(lg));
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

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));

  PetscCall(VecMin(xin,NULL,&min));
  PetscCall(VecMax(xin,NULL,&max));
  if (min == max) {
    min -= 1.e-5;
    max += 1.e-5;
  }

  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawClear(draw));

  PetscCall(PetscDrawAxisCreate(draw,&axis));
  PetscCall(PetscDrawAxisSetLimits(axis,0.0,(PetscReal)xin->map->N,min,max));
  PetscCall(PetscDrawAxisDraw(axis));
  PetscCall(PetscDrawAxisDestroy(&axis));

  /* draw local part of vector */
  PetscCall(VecGetArrayRead(xin,&xarray));
  PetscCall(VecGetOwnershipRange(xin,&start,&end));
  if (rank < size-1) { /* send value to right */
    PetscCallMPI(MPI_Send((void*)&xarray[xin->map->n-1],1,MPIU_REAL,rank+1,tag,PetscObjectComm((PetscObject)xin)));
  }
  if (rank) { /* receive value from right */
    PetscCallMPI(MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,PetscObjectComm((PetscObject)xin),&status));
  }
  PetscDrawCollectiveBegin(draw);
  if (rank) {
    PetscCall(PetscDrawLine(draw,(PetscReal)start-1,tmp,(PetscReal)start,PetscRealPart(xarray[0]),PETSC_DRAW_RED));
  }
  for (i=1; i<xin->map->n; i++) {
    PetscCall(PetscDrawLine(draw,(PetscReal)(i-1+start),PetscRealPart(xarray[i-1]),(PetscReal)(i+start),PetscRealPart(xarray[i]),PETSC_DRAW_RED));
  }
  PetscDrawCollectiveEnd(draw);
  PetscCall(VecRestoreArrayRead(xin,&xarray));

  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawSave(draw));
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
  PetscCall(VecGetArrayRead(xin,&xarray));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)xin),&rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  if (rank == 0) {
    PetscCall(PetscMalloc1(N,&xx));
    PetscCall(PetscMalloc1(size,&lens));
    for (i=0; i<size; i++) lens[i] = xin->map->range[i+1] - xin->map->range[i];

    PetscCallMPI(MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,xx,lens,xin->map->range,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin)));
    PetscCall(PetscFree(lens));

    PetscCall(PetscObjectName((PetscObject)xin));
    PetscCall(PetscViewerMatlabPutArray(viewer,N,1,xx,((PetscObject)xin)->name));

    PetscCall(PetscFree(xx));
  } else {
    PetscCallMPI(MPI_Gatherv((void*)xarray,xin->map->n,MPIU_SCALAR,0,0,0,MPIU_SCALAR,0,PetscObjectComm((PetscObject)xin)));
  }
  PetscCall(VecRestoreArrayRead(xin,&xarray));
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
  PetscCall(PetscObjectGetName((PetscObject) xin, &vecname));

  PetscCall(VecGetLocalSize(xin,&n));
  PetscCall(VecGetSize(xin,&N));
  PetscCall(VecGetOwnershipRange(xin,&rstart,NULL));

  sprintf(nlocalname,"%d",(int)n);
  sprintf(nglobalname,"%d",(int)N);
  sprintf(coffset,"%d",(int)rstart);
  id   = adios_define_var(Petsc_adios_group,vecname,"",adios_double,nlocalname,nglobalname,coffset);
  PetscCall(VecGetArrayRead(xin,&array));
  PetscCall(adios_write_byid(adios->adios_handle,id,array));
  PetscCall(VecRestoreArrayRead(xin,&array));
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
  PetscCall(PetscViewerHDF5OpenGroup(viewer, &file_id, &group));
  PetscCall(PetscViewerHDF5IsTimestepping(viewer, &timestepping));
  if (timestepping) {
    PetscCall(PetscViewerHDF5GetTimestep(viewer, &timestep));
  }
  PetscCall(PetscViewerHDF5GetBaseDimension2(viewer,&dim2));
  PetscCall(PetscViewerHDF5GetSPOutput(viewer,&spoutput));

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
  PetscCall(PetscHDF5IntCast(xin->map->N/bs,dims + dim));

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
  PetscCall(PetscObjectGetName((PetscObject) xin, &vecname));
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
  PetscCall(PetscHDF5IntCast(xin->map->n/bs,count + dim));
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
  PetscCall(VecGetOwnershipRange(xin, &low, NULL));
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  PetscCall(PetscHDF5IntCast(low/bs,offset + dim));
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

  PetscCall(VecGetArrayRead(xin, &x));
  PetscStackCallHDF5(H5Dwrite,(dset_id, memscalartype, memspace, filespace, hdf5->dxpl_id, x));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  PetscCall(VecRestoreArrayRead(xin, &x));

  /* Close/release resources */
  PetscStackCallHDF5(H5Gclose,(group));
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));

#if defined(PETSC_USE_COMPLEX)
  {
    PetscBool   tru = PETSC_TRUE;
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"complex",PETSC_BOOL,&tru));
  }
#endif
  if (timestepping) {
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)xin,"timestepping",PETSC_BOOL,&timestepping));
  }
  PetscCall(PetscInfo(xin,"Wrote Vec object with name %s\n",vecname));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATHEMATICA,&ismathematica));
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis));
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
#endif
  if (iascii) {
    PetscCall(VecView_MPI_ASCII(xin,viewer));
  } else if (isbinary) {
    PetscCall(VecView_MPI_Binary(xin,viewer));
  } else if (isdraw) {
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_DRAW_LG) {
      PetscCall(VecView_MPI_Draw_LG(xin,viewer));
    } else {
      PetscCall(VecView_MPI_Draw(xin,viewer));
    }
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    PetscCall(PetscViewerMathematicaPutVector(viewer,xin));
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(VecView_MPI_HDF5(xin,viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    PetscCall(VecView_MPI_ADIOS(xin,viewer));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    PetscCall(VecView_MPI_Matlab(xin,viewer));
#endif
  } else if (isglvis) {
    PetscCall(VecView_GLVis(xin,viewer));
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
  PetscCall(VecGetArrayRead(xin,&xx));
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
    tmp = ix[i] - start;
    PetscCheck(tmp >= 0 && tmp < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only get local values, trying %" PetscInt_FMT,ix[i]);
    y[i] = xx[tmp];
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
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
    PetscCheck(xin->stash.insertmode != INSERT_VALUES || addv != ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else PetscCheck(xin->stash.insertmode != ADD_VALUES || addv != INSERT_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  PetscCall(VecGetArray(xin,&xx));
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
      if ((row = ix[i]) >= start && row < end) {
        xx[row-start] = y[i];
      } else if (!xin->stash.donotstash) {
        PetscCheck(ix[i] < xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,ix[i],xin->map->N);
        PetscCall(VecStashValue_Private(&xin->stash,row,y[i]));
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
        PetscCall(VecStashValue_Private(&xin->stash,row,y[i]));
      }
    }
  }
  PetscCall(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesBlocked_MPI(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode addv)
{
  PetscMPIInt    rank    = xin->stash.rank;
  PetscInt       *owners = xin->map->range,start = owners[rank];
  PetscInt       end = owners[rank+1],i,row,bs = PetscAbs(xin->map->bs),j;
  PetscScalar    *xx,*y = (PetscScalar*)yin;

  PetscFunctionBegin;
  PetscCall(VecGetArray(xin,&xx));
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(xin->stash.insertmode != INSERT_VALUES || addv != ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already inserted values; you cannot now add");
    else PetscCheck(xin->stash.insertmode != ADD_VALUES || addv != INSERT_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You have already added values; you cannot now insert");
  }
  xin->stash.insertmode = addv;

  if (addv == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if ((row = bs*ix[i]) >= start && row < end) {
        for (j=0; j<bs; j++) xx[row-start+j] = y[j];
      } else if (!xin->stash.donotstash) {
        if (ix[i] < 0) { y += bs; continue; }
        PetscCheck(ix[i] < xin->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " max %" PetscInt_FMT,ix[i],xin->map->N);
        PetscCall(VecStashValuesBlocked_Private(&xin->bstash,ix[i],y));
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
        PetscCall(VecStashValuesBlocked_Private(&xin->bstash,ix[i],y));
      }
      y += bs;
    }
  }
  PetscCall(VecRestoreArray(xin,&xx));
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
  PetscCall(PetscObjectGetComm((PetscObject)xin,&comm));
  if (xin->stash.donotstash) PetscFunctionReturn(0);

  PetscCall(MPIU_Allreduce((PetscEnum*)&xin->stash.insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,comm));
  PetscCheck(addv != (ADD_VALUES|INSERT_VALUES),comm,PETSC_ERR_ARG_NOTSAMETYPE,"Some processors inserted values while others added");
  xin->stash.insertmode = addv; /* in case this processor had no cache */
  xin->bstash.insertmode = addv; /* Block stash implicitly tracks InsertMode of scalar stash */

  PetscCall(VecGetBlockSize(xin,&bs));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)xin),&size));
  if (!xin->bstash.bowners && xin->map->bs != -1) {
    PetscCall(PetscMalloc1(size+1,&bowners));
    for (i=0; i<size+1; i++) bowners[i] = owners[i]/bs;
    xin->bstash.bowners = bowners;
  } else bowners = xin->bstash.bowners;

  PetscCall(VecStashScatterBegin_Private(&xin->stash,owners));
  PetscCall(VecStashScatterBegin_Private(&xin->bstash,bowners));
  PetscCall(VecStashGetInfo_Private(&xin->stash,&nstash,&reallocs));
  PetscCall(PetscInfo(xin,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscCall(VecStashGetInfo_Private(&xin->bstash,&nstash,&reallocs));
  PetscCall(PetscInfo(xin,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAssemblyEnd_MPI(Vec vec)
{
  PetscInt       base,i,j,*row,flg,bs;
  PetscMPIInt    n;
  PetscScalar    *val,*vv,*array,*xarray;

  PetscFunctionBegin;
  if (!vec->stash.donotstash) {
    PetscCall(VecGetArray(vec,&xarray));
    PetscCall(VecGetBlockSize(vec,&bs));
    base = vec->map->range[vec->stash.rank];

    /* Process the stash */
    while (1) {
      PetscCall(VecStashScatterGetMesg_Private(&vec->stash,&n,&row,&val,&flg));
      if (!flg) break;
      if (vec->stash.insertmode == ADD_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] += val[i];
      } else if (vec->stash.insertmode == INSERT_VALUES) {
        for (i=0; i<n; i++) xarray[row[i] - base] = val[i];
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Insert mode is not set correctly; corrupted vector");
    }
    PetscCall(VecStashScatterEnd_Private(&vec->stash));

    /* now process the block-stash */
    while (1) {
      PetscCall(VecStashScatterGetMesg_Private(&vec->bstash,&n,&row,&val,&flg));
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
    PetscCall(VecStashScatterEnd_Private(&vec->bstash));
    PetscCall(VecRestoreArray(vec,&xarray));
  }
  vec->stash.insertmode = NOT_SET_VALUES;
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetPreallocationCOO_MPI(Vec x,PetscCount coo_n,const PetscInt coo_i[])
{
  PetscInt       m,M,rstart,rend;
  Vec_MPI        *vmpi = (Vec_MPI*)x->data;
  PetscCount     k,p,q,rem; /* Loop variables over coo arrays */
  PetscMPIInt    size;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)x,&comm));
  PetscCall(MPI_Comm_size(comm,&size));
  PetscCall(VecResetPreallocationCOO_MPI(x));

  PetscCall(PetscLayoutSetUp(x->map));
  PetscCall(VecGetOwnershipRange(x,&rstart,&rend));
  PetscCall(VecGetLocalSize(x,&m));
  PetscCall(VecGetSize(x,&M));

  /* ---------------------------------------------------------------------------*/
  /* Sort COOs along with a permuation array, so that negative indices come     */
  /* first, then local ones, then remote ones.                                  */
  /* ---------------------------------------------------------------------------*/
  PetscCount n1 = coo_n,nneg,*perm;
  PetscInt   *i1; /* Copy of input COOs along with a permutation array */
  PetscCall(PetscMalloc1(n1,&i1));
  PetscCall(PetscMalloc1(n1,&perm));
  PetscCall(PetscArraycpy(i1,coo_i,n1)); /* Make a copy since we'll modify it */
  for (k=0; k<n1; k++) perm[k] = k;

  /* Manipulate i1[] so that entries with negative indices will have the smallest
     index, local entries will have greater but negative indices, and remote entries
     will have positive indices.
  */
  for (k=0; k<n1; k++) {
    if (i1[k] < 0) {
      if (x->stash.ignorenegidx) i1[k] = PETSC_MIN_INT; /* e.g., -2^31, minimal to move them ahead */
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Found a negative index in VecSetPreallocateCOO() but VEC_IGNORE_NEGATIVE_INDICES was not set");
    } else if (i1[k] >= rstart && i1[k] < rend) {
      i1[k] -= PETSC_MAX_INT; /* e.g., minus 2^31-1 to shift local rows to range of [-PETSC_MAX_INT, -1] */
    } else {
      PetscCheck(i1[k]<M,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Found index %" PetscInt_FMT " in VecSetPreallocateCOO() larger than the global size %" PetscInt_FMT "",i1[k],M);
      if (x->stash.donotstash) i1[k] = PETSC_MIN_INT; /* Ignore off-proc indices as if they were negative */
    }
  }

  /* Sort the indices, after that, [0,nneg) have ignored entires, [nneg,rem) have local entries and [rem,n1) have remote entries */
  PetscCall(PetscSortIntWithCountArray(n1,i1,perm));
  for (k=0; k<n1; k++) {if (i1[k] > PETSC_MIN_INT) break;} /* Advance k to the first entry we need to take care of */
  nneg = k;
  PetscCall(PetscSortedIntUpperBound(i1,nneg,n1,rend-1-PETSC_MAX_INT,&rem)); /* rem is upper bound of the last local row */
  for (k=nneg; k<rem; k++) i1[k] += PETSC_MAX_INT; /* Revert indices of local entries */

  /* ---------------------------------------------------------------------------*/
  /*           Build stuff for local entries                                    */
  /* ---------------------------------------------------------------------------*/
  PetscCount tot1,*jmap1,*perm1;
  PetscCall(PetscCalloc1(m+1,&jmap1));
  for (k=nneg; k<rem; k++) jmap1[i1[k]-rstart+1]++; /* Count repeats of each local entry */
  for (k=0; k<m; k++) jmap1[k+1] += jmap1[k]; /* Transform jmap1[] to CSR-like data structure */
  tot1 = jmap1[m];
  PetscAssert(tot1 == rem-nneg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected errors in VecSetPreallocationCOO_MPI");
  PetscCall(PetscMalloc1(tot1,&perm1));
  PetscCall(PetscArraycpy(perm1,perm+nneg,tot1));

  /* ---------------------------------------------------------------------------*/
  /*        Record the permutation array for filling the send buffer            */
  /* ---------------------------------------------------------------------------*/
  PetscCount *Cperm;
  PetscCall(PetscMalloc1(n1-rem,&Cperm));
  PetscCall(PetscArraycpy(Cperm,perm+rem,n1-rem));
  PetscCall(PetscFree(perm));

  /* ---------------------------------------------------------------------------*/
  /*           Send remote entries to their owner                                  */
  /* ---------------------------------------------------------------------------*/
  /* Find which entries should be sent to which remote ranks*/
  PetscInt       nsend = 0; /* Number of MPI ranks to send data to */
  PetscMPIInt    *sendto;   /* [nsend], storing remote ranks */
  PetscInt       *nentries; /* [nsend], storing number of entries sent to remote ranks; Assume PetscInt is big enough for this count, and error if not */
  const PetscInt *ranges;
  PetscInt       maxNsend = size >= 128? 128 : size; /* Assume max 128 neighbors; realloc when needed */

  PetscCall(PetscLayoutGetRanges(x->map,&ranges));
  PetscCall(PetscMalloc2(maxNsend,&sendto,maxNsend,&nentries));
  for (k=rem; k<n1;) {
    PetscMPIInt  owner;
    PetscInt     firstRow,lastRow;

    /* Locate a row range */
    firstRow = i1[k]; /* first row of this owner */
    PetscCall(PetscLayoutFindOwner(x->map,firstRow,&owner));
    lastRow  = ranges[owner+1]-1; /* last row of this owner */

    /* Find the first index 'p' in [k,n) with i[p] belonging to next owner */
    PetscCall(PetscSortedIntUpperBound(i1,k,n1,lastRow,&p));

    /* All entries in [k,p) belong to this remote owner */
    if (nsend >= maxNsend) { /* Double the remote ranks arrays if not long enough */
      PetscMPIInt *sendto2;
      PetscInt    *nentries2;
      PetscInt    maxNsend2 = (maxNsend <= size/2) ? maxNsend*2 : size;

      PetscCall(PetscMalloc2(maxNsend2,&sendto2,maxNsend2,&nentries2));
      PetscCall(PetscArraycpy(sendto2,sendto,maxNsend));
      PetscCall(PetscArraycpy(nentries2,nentries2,maxNsend+1));
      PetscCall(PetscFree2(sendto,nentries2));
      sendto      = sendto2;
      nentries    = nentries2;
      maxNsend    = maxNsend2;
    }
    sendto[nsend]   = owner;
    nentries[nsend] = p - k;
    PetscCall(PetscCountCast(p-k,&nentries[nsend]));
    nsend++;
    k = p;
  }

  /* Build 1st SF to know offsets on remote to send data */
  PetscSF     sf1;
  PetscInt    nroots = 1,nroots2 = 0;
  PetscInt    nleaves = nsend,nleaves2 = 0;
  PetscInt    *offsets;
  PetscSFNode *iremote;

  PetscCall(PetscSFCreate(comm,&sf1));
  PetscCall(PetscMalloc1(nsend,&iremote));
  PetscCall(PetscMalloc1(nsend,&offsets));
  for (k=0; k<nsend; k++) {
    iremote[k].rank  = sendto[k];
    iremote[k].index = 0;
    nleaves2        += nentries[k];
    PetscCheck(nleaves2 >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of SF leaves is too large for PetscInt");
  }
  PetscCall(PetscSFSetGraph(sf1,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  PetscCall(PetscSFFetchAndOpWithMemTypeBegin(sf1,MPIU_INT,PETSC_MEMTYPE_HOST,&nroots2/*rootdata*/,PETSC_MEMTYPE_HOST,nentries/*leafdata*/,PETSC_MEMTYPE_HOST,offsets/*leafupdate*/,MPI_SUM));
  PetscCall(PetscSFFetchAndOpEnd(sf1,MPIU_INT,&nroots2,nentries,offsets,MPI_SUM)); /* Would nroots2 overflow, we check offsets[] below */
  PetscCall(PetscSFDestroy(&sf1));
  PetscAssert(nleaves2 == n1-rem,PETSC_COMM_SELF,PETSC_ERR_PLIB,"nleaves2 %" PetscInt_FMT " != number of remote entries %" PetscCount_FMT "",nleaves2,n1-rem);

  /* Build 2nd SF to send remote COOs to their owner */
  PetscSF sf2;
  nroots  = nroots2;
  nleaves = nleaves2;
  PetscCall(PetscSFCreate(comm,&sf2));
  PetscCall(PetscSFSetFromOptions(sf2));
  PetscCall(PetscMalloc1(nleaves,&iremote));
  p       = 0;
  for (k=0; k<nsend; k++) {
    PetscCheck(offsets[k] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of SF roots is too large for PetscInt");
    for (q=0; q<nentries[k]; q++,p++) {
      iremote[p].rank  = sendto[k];
      iremote[p].index = offsets[k] + q;
    }
  }
  PetscCall(PetscSFSetGraph(sf2,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));

  /* Send the remote COOs to their owner */
  PetscInt   n2 = nroots,*i2; /* Buffers for received COOs from other ranks, along with a permutation array */
  PetscCount *perm2;
  PetscCall(PetscMalloc1(n2,&i2));
  PetscCall(PetscMalloc1(n2,&perm2));
  PetscCall(PetscSFReduceWithMemTypeBegin(sf2,MPIU_INT,PETSC_MEMTYPE_HOST,i1+rem,PETSC_MEMTYPE_HOST,i2,MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf2,MPIU_INT,i1+rem,i2,MPI_REPLACE));

  PetscCall(PetscFree(i1));
  PetscCall(PetscFree(offsets));
  PetscCall(PetscFree2(sendto,nentries));

  /* ---------------------------------------------------------------*/
  /* Sort received COOs along with a permutation array            */
  /* ---------------------------------------------------------------*/
  PetscCount  *imap2;
  PetscCount  *jmap2,nnz2;
  PetscScalar *sendbuf,*recvbuf;
  PetscInt    old;
  PetscCount  sendlen = n1-rem,recvlen = n2;

  for (k=0; k<n2; k++) perm2[k] = k;
  PetscCall(PetscSortIntWithCountArray(n2,i2,perm2));

  /* nnz2 will be # of unique entries in the recvbuf */
  nnz2 = n2;
  for (k=1; k<n2; k++) {if (i2[k] == i2[k-1]) nnz2--;}

  /* Build imap2[] and jmap2[] for each unique entry */
  PetscCall(PetscMalloc4(nnz2,&imap2,nnz2+1,&jmap2,sendlen,&sendbuf,recvlen,&recvbuf));
  p   = -1;
  old = -1;
  jmap2[0] = 0;
  jmap2++;
  for (k=0; k<n2; k++) {
    if (i2[k] != old) { /* Meet a new entry */
      p++;
      imap2[p] = i2[k] - rstart;
      jmap2[p] = 1;
      old      = i2[k];
    } else {
      jmap2[p]++;
    }
  }
  jmap2--;
  for (k=0; k<nnz2; k++) jmap2[k+1] += jmap2[k];

  PetscCall(PetscFree(i2));

  vmpi->coo_n   = coo_n;
  vmpi->tot1    = tot1;
  vmpi->jmap1   = jmap1;
  vmpi->perm1   = perm1;
  vmpi->nnz2    = nnz2;
  vmpi->imap2   = imap2;
  vmpi->jmap2   = jmap2;
  vmpi->perm2   = perm2;

  vmpi->Cperm   = Cperm;
  vmpi->sendbuf = sendbuf;
  vmpi->recvbuf = recvbuf;
  vmpi->sendlen = sendlen;
  vmpi->recvlen = recvlen;
  vmpi->coo_sf  = sf2;
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesCOO_MPI(Vec x,const PetscScalar v[],InsertMode imode)
{
  Vec_MPI              *vmpi = (Vec_MPI*)x->data;
  PetscInt             m;
  PetscScalar          *a,*sendbuf = vmpi->sendbuf,*recvbuf = vmpi->recvbuf;
  const PetscCount     *jmap1 = vmpi->jmap1;
  const PetscCount     *perm1 = vmpi->perm1;
  const PetscCount     *imap2 = vmpi->imap2;
  const PetscCount     *jmap2 = vmpi->jmap2;
  const PetscCount     *perm2 = vmpi->perm2;
  const PetscCount     *Cperm = vmpi->Cperm;
  const PetscCount     nnz2 = vmpi->nnz2;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x,&m));
  PetscCall(VecGetArray(x,&a));

  /* Pack entries to be sent to remote */
  for (PetscInt i=0; i<vmpi->sendlen; i++) sendbuf[i] = v[Cperm[i]];

  /* Send remote entries to their owner and overlap the communication with local computation */
  PetscCall(PetscSFReduceWithMemTypeBegin(vmpi->coo_sf,MPIU_SCALAR,PETSC_MEMTYPE_HOST,sendbuf,PETSC_MEMTYPE_HOST,recvbuf,MPI_REPLACE));
  /* Add local entries to A and B */
  for (PetscInt i=0; i<m; i++) { /* All entries in a[] are either zero'ed or added with a value (i.e., initialized) */
    PetscScalar sum = 0.0; /* Do partial summation first to improve numerical stablility */
    for (PetscCount k=jmap1[i]; k<jmap1[i+1]; k++) sum += v[perm1[k]];
    a[i] = (imode == INSERT_VALUES? 0.0 : a[i]) + sum;
  }
  PetscCall(PetscSFReduceEnd(vmpi->coo_sf,MPIU_SCALAR,sendbuf,recvbuf,MPI_REPLACE));

  /* Add received remote entries to A and B */
  for (PetscInt i=0; i<nnz2; i++) {
    for (PetscCount k=jmap2[i]; k<jmap2[i+1]; k++) a[imap2[i]] += recvbuf[perm2[k]];
  }

  PetscCall(VecRestoreArray(x,&a));
  PetscFunctionReturn(0);
}

