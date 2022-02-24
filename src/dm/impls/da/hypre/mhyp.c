
/*
    Creates hypre ijmatrix from PETSc matrix
*/
#include <petscsys.h>
#include <petsc/private/petschypre.h>
#include <petsc/private/matimpl.h>
#include <petscdmda.h>                /*I "petscdmda.h" I*/
#include <../src/dm/impls/da/hypre/mhyp.h>

/* -----------------------------------------------------------------------------------------------------------------*/

/*MC
   MATHYPRESTRUCT - MATHYPRESTRUCT = "hyprestruct" - A matrix type to be used for parallel sparse matrices
          based on the hypre HYPRE_StructMatrix.

   Level: intermediate

   Notes:
    Unlike the more general support for blocks in hypre this allows only one block per process and requires the block
          be defined by a DMDA.

          The matrix needs a DMDA associated with it by either a call to MatSetDM() or if the matrix is obtained from DMCreateMatrix()

.seealso: MatCreate(), PCPFMG, MatSetDM(), DMCreateMatrix()
M*/

PetscErrorCode  MatSetValuesLocal_HYPREStruct_3d(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  HYPRE_Int       index[3],entries[7];
  PetscInt        i,j,stencil,row;
  HYPRE_Complex   *values = (HYPRE_Complex*)y;
  Mat_HYPREStruct *ex     = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  PetscCheckFalse(ncol > 7,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ncol %D > 7 too large",ncol);
  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      stencil = icol[j] - irow[i];
      if (!stencil) {
        entries[j] = 3;
      } else if (stencil == -1) {
        entries[j] = 2;
      } else if (stencil == 1) {
        entries[j] = 4;
      } else if (stencil == -ex->gnx) {
        entries[j] = 1;
      } else if (stencil == ex->gnx) {
        entries[j] = 5;
      } else if (stencil == -ex->gnxgny) {
        entries[j] = 0;
      } else if (stencil == ex->gnxgny) {
        entries[j] = 6;
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row %D local column %D have bad stencil %D",irow[i],icol[j],stencil);
    }
    row      = ex->gindices[irow[i]] - ex->rstart;
    index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
    index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
    index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));
    if (addv == ADD_VALUES) {
      PetscStackCallStandard(HYPRE_StructMatrixAddToValues,ex->hmat,index,ncol,entries,values);
    } else {
      PetscStackCallStandard(HYPRE_StructMatrixSetValues,ex->hmat,index,ncol,entries,values);
    }
    values += ncol;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatZeroRowsLocal_HYPREStruct_3d(Mat mat,PetscInt nrow,const PetscInt irow[],PetscScalar d,Vec x,Vec b)
{
  HYPRE_Int       index[3],entries[7] = {0,1,2,3,4,5,6};
  PetscInt        row,i;
  HYPRE_Complex   values[7];
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  PetscCheckFalse(x && b,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"No support");
  CHKERRQ(PetscArrayzero(values,7));
  CHKERRQ(PetscHYPREScalarCast(d,&values[3]));
  for (i=0; i<nrow; i++) {
    row      = ex->gindices[irow[i]] - ex->rstart;
    index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
    index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
    index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));
    PetscStackCallStandard(HYPRE_StructMatrixSetValues,ex->hmat,index,7,entries,values);
  }
  PetscStackCallStandard(HYPRE_StructMatrixAssemble,ex->hmat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_HYPREStruct_3d(Mat mat)
{
  HYPRE_Int       indices[7] = {0,1,2,3,4,5,6};
  Mat_HYPREStruct *ex        = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  /* hypre has no public interface to do this */
  PetscStackCallStandard(hypre_StructMatrixClearBoxValues,ex->hmat,&ex->hbox,7,indices,0,1);
  PetscStackCallStandard(HYPRE_StructMatrixAssemble,ex->hmat);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetUp_HYPREStruct(Mat mat)
{
  Mat_HYPREStruct        *ex = (Mat_HYPREStruct*) mat->data;
  HYPRE_Int              sw[6];
  HYPRE_Int              hlower[3],hupper[3];
  PetscInt               dim,dof,psw,nx,ny,nz,ilower[3],iupper[3],ssize,i;
  DMBoundaryType         px,py,pz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;
  DM                     da;

  PetscFunctionBegin;
  CHKERRQ(MatGetDM(mat,(DM*)&da));
  ex->da = da;
  CHKERRQ(PetscObjectReference((PetscObject)da));

  CHKERRQ(DMDAGetInfo(ex->da,&dim,0,0,0,0,0,0,&dof,&psw,&px,&py,&pz,&st));
  CHKERRQ(DMDAGetCorners(ex->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]));

  /* when HYPRE_MIXEDINT is defined, sizeof(HYPRE_Int) == 32 */
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;
  hlower[0]  = (HYPRE_Int)ilower[0];
  hlower[1]  = (HYPRE_Int)ilower[1];
  hlower[2]  = (HYPRE_Int)ilower[2];
  hupper[0]  = (HYPRE_Int)iupper[0];
  hupper[1]  = (HYPRE_Int)iupper[1];
  hupper[2]  = (HYPRE_Int)iupper[2];

  /* the hypre_Box is used to zero out the matrix entries in MatZeroValues() */
  ex->hbox.imin[0] = hlower[0];
  ex->hbox.imin[1] = hlower[1];
  ex->hbox.imin[2] = hlower[2];
  ex->hbox.imax[0] = hupper[0];
  ex->hbox.imax[1] = hupper[1];
  ex->hbox.imax[2] = hupper[2];

  /* create the hypre grid object and set its information */
  PetscCheckFalse(dof > 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Currently only support for scalar problems");
  PetscCheckFalse(px || py || pz,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add periodic support by calling HYPRE_StructGridSetPeriodic()");
  PetscStackCallStandard(HYPRE_StructGridCreate,ex->hcomm,dim,&ex->hgrid);
  PetscStackCallStandard(HYPRE_StructGridSetExtents,ex->hgrid,hlower,hupper);
  PetscStackCallStandard(HYPRE_StructGridAssemble,ex->hgrid);

  sw[5] = sw[4] = sw[3] = sw[2] = sw[1] = sw[0] = psw;
  PetscStackCallStandard(HYPRE_StructGridSetNumGhost,ex->hgrid,sw);

  /* create the hypre stencil object and set its information */
  PetscCheckFalse(sw[0] > 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add support for wider stencils");
  PetscCheckFalse(st == DMDA_STENCIL_BOX,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add support for box stencils");
  if (dim == 1) {
    HYPRE_Int offsets[3][1] = {{-1},{0},{1}};
    ssize = 3;
    PetscStackCallStandard(HYPRE_StructStencilCreate,dim,ssize,&ex->hstencil);
    for (i=0; i<ssize; i++) {
      PetscStackCallStandard(HYPRE_StructStencilSetElement,ex->hstencil,i,offsets[i]);
    }
  } else if (dim == 2) {
    HYPRE_Int offsets[5][2] = {{0,-1},{-1,0},{0,0},{1,0},{0,1}};
    ssize = 5;
    PetscStackCallStandard(HYPRE_StructStencilCreate,dim,ssize,&ex->hstencil);
    for (i=0; i<ssize; i++) {
      PetscStackCallStandard(HYPRE_StructStencilSetElement,ex->hstencil,i,offsets[i]);
    }
  } else if (dim == 3) {
    HYPRE_Int offsets[7][3] = {{0,0,-1},{0,-1,0},{-1,0,0},{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    ssize = 7;
    PetscStackCallStandard(HYPRE_StructStencilCreate,dim,ssize,&ex->hstencil);
    for (i=0; i<ssize; i++) {
      PetscStackCallStandard(HYPRE_StructStencilSetElement,ex->hstencil,i,offsets[i]);
    }
  }

  /* create the HYPRE vector for rhs and solution */
  PetscStackCallStandard(HYPRE_StructVectorCreate,ex->hcomm,ex->hgrid,&ex->hb);
  PetscStackCallStandard(HYPRE_StructVectorCreate,ex->hcomm,ex->hgrid,&ex->hx);
  PetscStackCallStandard(HYPRE_StructVectorInitialize,ex->hb);
  PetscStackCallStandard(HYPRE_StructVectorInitialize,ex->hx);
  PetscStackCallStandard(HYPRE_StructVectorAssemble,ex->hb);
  PetscStackCallStandard(HYPRE_StructVectorAssemble,ex->hx);

  /* create the hypre matrix object and set its information */
  PetscStackCallStandard(HYPRE_StructMatrixCreate,ex->hcomm,ex->hgrid,ex->hstencil,&ex->hmat);
  PetscStackCallStandard(HYPRE_StructGridDestroy,ex->hgrid);
  PetscStackCallStandard(HYPRE_StructStencilDestroy,ex->hstencil);
  if (ex->needsinitialization) {
    PetscStackCallStandard(HYPRE_StructMatrixInitialize,ex->hmat);
    ex->needsinitialization = PETSC_FALSE;
  }

  /* set the global and local sizes of the matrix */
  CHKERRQ(DMDAGetCorners(da,0,0,0,&nx,&ny,&nz));
  CHKERRQ(MatSetSizes(mat,dof*nx*ny*nz,dof*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(PetscLayoutSetBlockSize(mat->rmap,1));
  CHKERRQ(PetscLayoutSetBlockSize(mat->cmap,1));
  CHKERRQ(PetscLayoutSetUp(mat->rmap));
  CHKERRQ(PetscLayoutSetUp(mat->cmap));
  mat->preallocated = PETSC_TRUE;

  if (dim == 3) {
    mat->ops->setvalueslocal = MatSetValuesLocal_HYPREStruct_3d;
    mat->ops->zerorowslocal  = MatZeroRowsLocal_HYPREStruct_3d;
    mat->ops->zeroentries    = MatZeroEntries_HYPREStruct_3d;

    /*        CHKERRQ(MatZeroEntries_HYPREStruct_3d(mat)); */
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only support for 3d DMDA currently");

  /* get values that will be used repeatedly in MatSetValuesLocal() and MatZeroRowsLocal() repeatedly */
  CHKERRQ(MatGetOwnershipRange(mat,&ex->rstart,NULL));
  CHKERRQ(DMGetLocalToGlobalMapping(ex->da,&ltog));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(ltog, (const PetscInt **) &ex->gindices));
  CHKERRQ(DMDAGetGhostCorners(ex->da,0,0,0,&ex->gnx,&ex->gnxgny,0));
  ex->gnxgny *= ex->gnx;
  CHKERRQ(DMDAGetCorners(ex->da,&ex->xs,&ex->ys,&ex->zs,&ex->nx,&ex->ny,0));
  ex->nxny    = ex->nx*ex->ny;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_HYPREStruct(Mat A,Vec x,Vec y)
{
  const PetscScalar *xx;
  PetscScalar       *yy;
  PetscInt          ilower[3],iupper[3];
  HYPRE_Int         hlower[3],hupper[3];
  Mat_HYPREStruct   *mx = (Mat_HYPREStruct*)(A->data);

  PetscFunctionBegin;
  CHKERRQ(DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]));
  /* when HYPRE_MIXEDINT is defined, sizeof(HYPRE_Int) == 32 */
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;
  hlower[0]  = (HYPRE_Int)ilower[0];
  hlower[1]  = (HYPRE_Int)ilower[1];
  hlower[2]  = (HYPRE_Int)ilower[2];
  hupper[0]  = (HYPRE_Int)iupper[0];
  hupper[1]  = (HYPRE_Int)iupper[1];
  hupper[2]  = (HYPRE_Int)iupper[2];

  /* copy x values over to hypre */
  PetscStackCallStandard(HYPRE_StructVectorSetConstantValues,mx->hb,0.0);
  CHKERRQ(VecGetArrayRead(x,&xx));
  PetscStackCallStandard(HYPRE_StructVectorSetBoxValues,mx->hb,hlower,hupper,(HYPRE_Complex*)xx);
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  PetscStackCallStandard(HYPRE_StructVectorAssemble,mx->hb);
  PetscStackCallStandard(HYPRE_StructMatrixMatvec,1.0,mx->hmat,mx->hb,0.0,mx->hx);

  /* copy solution values back to PETSc */
  CHKERRQ(VecGetArray(y,&yy));
  PetscStackCallStandard(HYPRE_StructVectorGetBoxValues,mx->hx,hlower,hupper,(HYPRE_Complex*)yy);
  CHKERRQ(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_HYPREStruct(Mat mat,MatAssemblyType mode)
{
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_StructMatrixAssemble,ex->hmat);
  /* PetscStackCallStandard(HYPRE_StructMatrixPrint,"dummy",ex->hmat,0); */
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_HYPREStruct(Mat mat)
{
  PetscFunctionBegin;
  /* before the DMDA is set to the matrix the zero doesn't need to do anything */
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_HYPREStruct(Mat mat)
{
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_StructMatrixDestroy,ex->hmat);
  PetscStackCallStandard(HYPRE_StructVectorDestroy,ex->hx);
  PetscStackCallStandard(HYPRE_StructVectorDestroy,ex->hb);
  CHKERRQ(PetscObjectDereference((PetscObject)ex->da));
  CHKERRMPI(MPI_Comm_free(&(ex->hcomm)));
  CHKERRQ(PetscFree(ex));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_HYPREStruct(Mat B)
{
  Mat_HYPREStruct *ex;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(B,&ex));
  B->data      = (void*)ex;
  B->rmap->bs  = 1;
  B->assembled = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;

  B->ops->assemblyend = MatAssemblyEnd_HYPREStruct;
  B->ops->mult        = MatMult_HYPREStruct;
  B->ops->zeroentries = MatZeroEntries_HYPREStruct;
  B->ops->destroy     = MatDestroy_HYPREStruct;
  B->ops->setup       = MatSetUp_HYPREStruct;

  ex->needsinitialization = PETSC_TRUE;

  CHKERRMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)B),&(ex->hcomm)));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATHYPRESTRUCT));
  PetscFunctionReturn(0);
}

/*MC
   MATHYPRESSTRUCT - MATHYPRESSTRUCT = "hypresstruct" - A matrix type to be used for parallel sparse matrices
          based on the hypre HYPRE_SStructMatrix.

   Level: intermediate

   Notes:
    Unlike hypre's general semi-struct object consisting of a collection of structured-grid objects and unstructured
          grid objects, we restrict the semi-struct objects to consist of only structured-grid components.

          Unlike the more general support for parts and blocks in hypre this allows only one part, and one block per process and requires the block
          be defined by a DMDA.

          The matrix needs a DMDA associated with it by either a call to MatSetDM() or if the matrix is obtained from DMCreateMatrix()

M*/

PetscErrorCode  MatSetValuesLocal_HYPRESStruct_3d(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  HYPRE_Int         index[3],*entries;
  PetscInt          i,j,stencil;
  HYPRE_Complex     *values = (HYPRE_Complex*)y;
  Mat_HYPRESStruct  *ex     = (Mat_HYPRESStruct*) mat->data;

  PetscInt          part = 0;          /* Petsc sstruct interface only allows 1 part */
  PetscInt          ordering;
  PetscInt          grid_rank, to_grid_rank;
  PetscInt          var_type, to_var_type;
  PetscInt          to_var_entry = 0;
  PetscInt          nvars= ex->nvars;
  PetscInt          row;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(7*nvars,&entries));
  ordering = ex->dofs_order;  /* ordering= 0   nodal ordering
                                           1   variable ordering */
  /* stencil entries are ordered by variables: var0_stencil0, var0_stencil1, ..., var0_stencil6, var1_stencil0, var1_stencil1, ...  */
  if (!ordering) {
    for (i=0; i<nrow; i++) {
      grid_rank= irow[i]/nvars;
      var_type = (irow[i] % nvars);

      for (j=0; j<ncol; j++) {
        to_grid_rank= icol[j]/nvars;
        to_var_type = (icol[j] % nvars);

        to_var_entry= to_var_entry*7;
        entries[j]  = to_var_entry;

        stencil = to_grid_rank-grid_rank;
        if (!stencil) {
          entries[j] += 3;
        } else if (stencil == -1) {
          entries[j] += 2;
        } else if (stencil == 1) {
          entries[j] += 4;
        } else if (stencil == -ex->gnx) {
          entries[j] += 1;
        } else if (stencil == ex->gnx) {
          entries[j] += 5;
        } else if (stencil == -ex->gnxgny) {
          entries[j] += 0;
        } else if (stencil == ex->gnxgny) {
          entries[j] += 6;
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row %D local column %D have bad stencil %D",irow[i],icol[j],stencil);
      }

      row      = ex->gindices[grid_rank] - ex->rstart;
      index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
      index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
      index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));

      if (addv == ADD_VALUES) {
        PetscStackCallStandard(HYPRE_SStructMatrixAddToValues,ex->ss_mat,part,index,var_type,ncol,entries,values);
      } else {
        PetscStackCallStandard(HYPRE_SStructMatrixSetValues,ex->ss_mat,part,index,var_type,ncol,entries,values);
      }
      values += ncol;
    }
  } else {
    for (i=0; i<nrow; i++) {
      var_type = irow[i]/(ex->gnxgnygnz);
      grid_rank= irow[i] - var_type*(ex->gnxgnygnz);

      for (j=0; j<ncol; j++) {
        to_var_type = icol[j]/(ex->gnxgnygnz);
        to_grid_rank= icol[j] - to_var_type*(ex->gnxgnygnz);

        to_var_entry= to_var_entry*7;
        entries[j]  = to_var_entry;

        stencil = to_grid_rank-grid_rank;
        if (!stencil) {
          entries[j] += 3;
        } else if (stencil == -1) {
          entries[j] += 2;
        } else if (stencil == 1) {
          entries[j] += 4;
        } else if (stencil == -ex->gnx) {
          entries[j] += 1;
        } else if (stencil == ex->gnx) {
          entries[j] += 5;
        } else if (stencil == -ex->gnxgny) {
          entries[j] += 0;
        } else if (stencil == ex->gnxgny) {
          entries[j] += 6;
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row %D local column %D have bad stencil %D",irow[i],icol[j],stencil);
      }

      row      = ex->gindices[grid_rank] - ex->rstart;
      index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
      index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
      index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));

      if (addv == ADD_VALUES) {
        PetscStackCallStandard(HYPRE_SStructMatrixAddToValues,ex->ss_mat,part,index,var_type,ncol,entries,values);
      } else {
        PetscStackCallStandard(HYPRE_SStructMatrixSetValues,ex->ss_mat,part,index,var_type,ncol,entries,values);
      }
      values += ncol;
    }
  }
  CHKERRQ(PetscFree(entries));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatZeroRowsLocal_HYPRESStruct_3d(Mat mat,PetscInt nrow,const PetscInt irow[],PetscScalar d,Vec x,Vec b)
{
  HYPRE_Int        index[3],*entries;
  PetscInt         i;
  HYPRE_Complex    **values;
  Mat_HYPRESStruct *ex = (Mat_HYPRESStruct*) mat->data;

  PetscInt         part     = 0;      /* Petsc sstruct interface only allows 1 part */
  PetscInt         ordering = ex->dofs_order;
  PetscInt         grid_rank;
  PetscInt         var_type;
  PetscInt         nvars= ex->nvars;
  PetscInt         row;

  PetscFunctionBegin;
  PetscCheckFalse(x && b,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"No support");
  CHKERRQ(PetscMalloc1(7*nvars,&entries));

  CHKERRQ(PetscMalloc1(nvars,&values));
  CHKERRQ(PetscMalloc1(7*nvars*nvars,&values[0]));
  for (i=1; i<nvars; i++) {
    values[i] = values[i-1] + nvars*7;
  }

  for (i=0; i< nvars; i++) {
    CHKERRQ(PetscArrayzero(values[i],nvars*7*sizeof(HYPRE_Complex)));
    CHKERRQ(PetscHYPREScalarCast(d,values[i]+3));
  }

  for (i=0; i< nvars*7; i++) entries[i] = i;

  if (!ordering) {
    for (i=0; i<nrow; i++) {
      grid_rank = irow[i] / nvars;
      var_type  = (irow[i] % nvars);

      row      = ex->gindices[grid_rank] - ex->rstart;
      index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
      index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
      index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));
      PetscStackCallStandard(HYPRE_SStructMatrixSetValues,ex->ss_mat,part,index,var_type,7*nvars,entries,values[var_type]);
    }
  } else {
    for (i=0; i<nrow; i++) {
      var_type = irow[i]/(ex->gnxgnygnz);
      grid_rank= irow[i] - var_type*(ex->gnxgnygnz);

      row      = ex->gindices[grid_rank] - ex->rstart;
      index[0] = (HYPRE_Int)(ex->xs + (row % ex->nx));
      index[1] = (HYPRE_Int)(ex->ys + ((row/ex->nx) % ex->ny));
      index[2] = (HYPRE_Int)(ex->zs + (row/(ex->nxny)));
      PetscStackCallStandard(HYPRE_SStructMatrixSetValues,ex->ss_mat,part,index,var_type,7*nvars,entries,values[var_type]);
    }
  }
  PetscStackCallStandard(HYPRE_SStructMatrixAssemble,ex->ss_mat);
  CHKERRQ(PetscFree(values[0]));
  CHKERRQ(PetscFree(values));
  CHKERRQ(PetscFree(entries));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_HYPRESStruct_3d(Mat mat)
{
  Mat_HYPRESStruct *ex  = (Mat_HYPRESStruct*) mat->data;
  PetscInt         nvars= ex->nvars;
  PetscInt         size;
  PetscInt         part= 0;   /* only one part */

  PetscFunctionBegin;
  size = ((ex->hbox.imax[0])-(ex->hbox.imin[0])+1)*((ex->hbox.imax[1])-(ex->hbox.imin[1])+1)*((ex->hbox.imax[2])-(ex->hbox.imin[2])+1);
  {
    HYPRE_Int     i,*entries,iupper[3],ilower[3];
    HYPRE_Complex *values;

    for (i= 0; i< 3; i++) {
      ilower[i] = ex->hbox.imin[i];
      iupper[i] = ex->hbox.imax[i];
    }

    CHKERRQ(PetscMalloc2(nvars*7,&entries,nvars*7*size,&values));
    for (i= 0; i< nvars*7; i++) entries[i] = i;
    CHKERRQ(PetscArrayzero(values,nvars*7*size));

    for (i= 0; i< nvars; i++) {
      PetscStackCallStandard(HYPRE_SStructMatrixSetBoxValues,ex->ss_mat,part,ilower,iupper,i,nvars*7,entries,values);
    }
    CHKERRQ(PetscFree2(entries,values));
  }
  PetscStackCallStandard(HYPRE_SStructMatrixAssemble,ex->ss_mat);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetUp_HYPRESStruct(Mat mat)
{
  Mat_HYPRESStruct       *ex = (Mat_HYPRESStruct*) mat->data;
  PetscInt               dim,dof,sw[3],nx,ny,nz;
  PetscInt               ilower[3],iupper[3],ssize,i;
  DMBoundaryType         px,py,pz;
  DMDAStencilType        st;
  PetscInt               nparts= 1;  /* assuming only one part */
  PetscInt               part  = 0;
  ISLocalToGlobalMapping ltog;
  DM                     da;

  PetscFunctionBegin;
  CHKERRQ(MatGetDM(mat,(DM*)&da));
  ex->da = da;
  CHKERRQ(PetscObjectReference((PetscObject)da));

  CHKERRQ(DMDAGetInfo(ex->da,&dim,0,0,0,0,0,0,&dof,&sw[0],&px,&py,&pz,&st));
  CHKERRQ(DMDAGetCorners(ex->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]));
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;
  /* the hypre_Box is used to zero out the matrix entries in MatZeroValues() */
  ex->hbox.imin[0] = ilower[0];
  ex->hbox.imin[1] = ilower[1];
  ex->hbox.imin[2] = ilower[2];
  ex->hbox.imax[0] = iupper[0];
  ex->hbox.imax[1] = iupper[1];
  ex->hbox.imax[2] = iupper[2];

  ex->dofs_order = 0;

  /* assuming that the same number of dofs on each gridpoint. Also assume all cell-centred based */
  ex->nvars= dof;

  /* create the hypre grid object and set its information */
  PetscCheckFalse(px || py || pz,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add periodic support by calling HYPRE_SStructGridSetPeriodic()");
  PetscStackCallStandard(HYPRE_SStructGridCreate,ex->hcomm,dim,nparts,&ex->ss_grid);
  PetscStackCallStandard(HYPRE_SStructGridSetExtents,ex->ss_grid,part,ex->hbox.imin,ex->hbox.imax);
  {
    HYPRE_SStructVariable *vartypes;
    CHKERRQ(PetscMalloc1(ex->nvars,&vartypes));
    for (i= 0; i< ex->nvars; i++) vartypes[i]= HYPRE_SSTRUCT_VARIABLE_CELL;
    PetscStackCallStandard(HYPRE_SStructGridSetVariables,ex->ss_grid, part, ex->nvars,vartypes);
    CHKERRQ(PetscFree(vartypes));
  }
  PetscStackCallStandard(HYPRE_SStructGridAssemble,ex->ss_grid);

  sw[1] = sw[0];
  sw[2] = sw[1];
  /* PetscStackCallStandard(HYPRE_SStructGridSetNumGhost,ex->ss_grid,sw); */

  /* create the hypre stencil object and set its information */
  PetscCheckFalse(sw[0] > 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add support for wider stencils");
  PetscCheckFalse(st == DMDA_STENCIL_BOX,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Ask us to add support for box stencils");

  if (dim == 1) {
    HYPRE_Int offsets[3][1] = {{-1},{0},{1}};
    PetscInt  j, cnt;

    ssize = 3*(ex->nvars);
    PetscStackCallStandard(HYPRE_SStructStencilCreate,dim,ssize,&ex->ss_stencil);
    cnt= 0;
    for (i = 0; i < (ex->nvars); i++) {
      for (j = 0; j < 3; j++) {
        PetscStackCallStandard(HYPRE_SStructStencilSetEntry,ex->ss_stencil, cnt, offsets[j], i);
        cnt++;
      }
    }
  } else if (dim == 2) {
    HYPRE_Int offsets[5][2] = {{0,-1},{-1,0},{0,0},{1,0},{0,1}};
    PetscInt  j, cnt;

    ssize = 5*(ex->nvars);
    PetscStackCallStandard(HYPRE_SStructStencilCreate,dim,ssize,&ex->ss_stencil);
    cnt= 0;
    for (i= 0; i< (ex->nvars); i++) {
      for (j= 0; j< 5; j++) {
        PetscStackCallStandard(HYPRE_SStructStencilSetEntry,ex->ss_stencil, cnt, offsets[j], i);
        cnt++;
      }
    }
  } else if (dim == 3) {
    HYPRE_Int offsets[7][3] = {{0,0,-1},{0,-1,0},{-1,0,0},{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    PetscInt  j, cnt;

    ssize = 7*(ex->nvars);
    PetscStackCallStandard(HYPRE_SStructStencilCreate,dim,ssize,&ex->ss_stencil);
    cnt= 0;
    for (i= 0; i< (ex->nvars); i++) {
      for (j= 0; j< 7; j++) {
        PetscStackCallStandard(HYPRE_SStructStencilSetEntry,ex->ss_stencil, cnt, offsets[j], i);
        cnt++;
      }
    }
  }

  /* create the HYPRE graph */
  PetscStackCallStandard(HYPRE_SStructGraphCreate,ex->hcomm, ex->ss_grid, &(ex->ss_graph));

  /* set the stencil graph. Note that each variable has the same graph. This means that each
     variable couples to all the other variable and with the same stencil pattern. */
  for (i= 0; i< (ex->nvars); i++) {
    PetscStackCallStandard(HYPRE_SStructGraphSetStencil,ex->ss_graph,part,i,ex->ss_stencil);
  }
  PetscStackCallStandard(HYPRE_SStructGraphAssemble,ex->ss_graph);

  /* create the HYPRE sstruct vectors for rhs and solution */
  PetscStackCallStandard(HYPRE_SStructVectorCreate,ex->hcomm,ex->ss_grid,&ex->ss_b);
  PetscStackCallStandard(HYPRE_SStructVectorCreate,ex->hcomm,ex->ss_grid,&ex->ss_x);
  PetscStackCallStandard(HYPRE_SStructVectorInitialize,ex->ss_b);
  PetscStackCallStandard(HYPRE_SStructVectorInitialize,ex->ss_x);
  PetscStackCallStandard(HYPRE_SStructVectorAssemble,ex->ss_b);
  PetscStackCallStandard(HYPRE_SStructVectorAssemble,ex->ss_x);

  /* create the hypre matrix object and set its information */
  PetscStackCallStandard(HYPRE_SStructMatrixCreate,ex->hcomm,ex->ss_graph,&ex->ss_mat);
  PetscStackCallStandard(HYPRE_SStructGridDestroy,ex->ss_grid);
  PetscStackCallStandard(HYPRE_SStructStencilDestroy,ex->ss_stencil);
  if (ex->needsinitialization) {
    PetscStackCallStandard(HYPRE_SStructMatrixInitialize,ex->ss_mat);
    ex->needsinitialization = PETSC_FALSE;
  }

  /* set the global and local sizes of the matrix */
  CHKERRQ(DMDAGetCorners(da,0,0,0,&nx,&ny,&nz));
  CHKERRQ(MatSetSizes(mat,dof*nx*ny*nz,dof*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(PetscLayoutSetBlockSize(mat->rmap,dof));
  CHKERRQ(PetscLayoutSetBlockSize(mat->cmap,dof));
  CHKERRQ(PetscLayoutSetUp(mat->rmap));
  CHKERRQ(PetscLayoutSetUp(mat->cmap));
  mat->preallocated = PETSC_TRUE;

  if (dim == 3) {
    mat->ops->setvalueslocal = MatSetValuesLocal_HYPRESStruct_3d;
    mat->ops->zerorowslocal  = MatZeroRowsLocal_HYPRESStruct_3d;
    mat->ops->zeroentries    = MatZeroEntries_HYPRESStruct_3d;

    /* CHKERRQ(MatZeroEntries_HYPRESStruct_3d(mat)); */
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only support for 3d DMDA currently");

  /* get values that will be used repeatedly in MatSetValuesLocal() and MatZeroRowsLocal() repeatedly */
  CHKERRQ(MatGetOwnershipRange(mat,&ex->rstart,NULL));
  CHKERRQ(DMGetLocalToGlobalMapping(ex->da,&ltog));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(ltog, (const PetscInt **) &ex->gindices));
  CHKERRQ(DMDAGetGhostCorners(ex->da,0,0,0,&ex->gnx,&ex->gnxgny,&ex->gnxgnygnz));

  ex->gnxgny    *= ex->gnx;
  ex->gnxgnygnz *= ex->gnxgny;

  CHKERRQ(DMDAGetCorners(ex->da,&ex->xs,&ex->ys,&ex->zs,&ex->nx,&ex->ny,&ex->nz));

  ex->nxny   = ex->nx*ex->ny;
  ex->nxnynz = ex->nz*ex->nxny;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_HYPRESStruct(Mat A,Vec x,Vec y)
{
  const PetscScalar *xx;
  PetscScalar       *yy;
  HYPRE_Int         hlower[3],hupper[3];
  PetscInt          ilower[3],iupper[3];
  Mat_HYPRESStruct  *mx     = (Mat_HYPRESStruct*)(A->data);
  PetscInt          ordering= mx->dofs_order;
  PetscInt          nvars   = mx->nvars;
  PetscInt          part    = 0;
  PetscInt          size;
  PetscInt          i;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]));

  /* when HYPRE_MIXEDINT is defined, sizeof(HYPRE_Int) == 32 */
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;
  hlower[0]  = (HYPRE_Int)ilower[0];
  hlower[1]  = (HYPRE_Int)ilower[1];
  hlower[2]  = (HYPRE_Int)ilower[2];
  hupper[0]  = (HYPRE_Int)iupper[0];
  hupper[1]  = (HYPRE_Int)iupper[1];
  hupper[2]  = (HYPRE_Int)iupper[2];

  size= 1;
  for (i = 0; i < 3; i++) size *= (iupper[i]-ilower[i]+1);

  /* copy x values over to hypre for variable ordering */
  if (ordering) {
    PetscStackCallStandard(HYPRE_SStructVectorSetConstantValues,mx->ss_b,0.0);
    CHKERRQ(VecGetArrayRead(x,&xx));
    for (i= 0; i< nvars; i++) {
      PetscStackCallStandard(HYPRE_SStructVectorSetBoxValues,mx->ss_b,part,hlower,hupper,i,(HYPRE_Complex*)(xx+(size*i)));
    }
    CHKERRQ(VecRestoreArrayRead(x,&xx));
    PetscStackCallStandard(HYPRE_SStructVectorAssemble,mx->ss_b);
    PetscStackCallStandard(HYPRE_SStructMatrixMatvec,1.0,mx->ss_mat,mx->ss_b,0.0,mx->ss_x);

    /* copy solution values back to PETSc */
    CHKERRQ(VecGetArray(y,&yy));
    for (i= 0; i< nvars; i++) {
      PetscStackCallStandard(HYPRE_SStructVectorGetBoxValues,mx->ss_x,part,hlower,hupper,i,(HYPRE_Complex*)(yy+(size*i)));
    }
    CHKERRQ(VecRestoreArray(y,&yy));
  } else {      /* nodal ordering must be mapped to variable ordering for sys_pfmg */
    PetscScalar *z;
    PetscInt    j, k;

    CHKERRQ(PetscMalloc1(nvars*size,&z));
    PetscStackCallStandard(HYPRE_SStructVectorSetConstantValues,mx->ss_b,0.0);
    CHKERRQ(VecGetArrayRead(x,&xx));

    /* transform nodal to hypre's variable ordering for sys_pfmg */
    for (i= 0; i< size; i++) {
      k= i*nvars;
      for (j= 0; j< nvars; j++) z[j*size+i]= xx[k+j];
    }
    for (i= 0; i< nvars; i++) {
      PetscStackCallStandard(HYPRE_SStructVectorSetBoxValues,mx->ss_b,part,hlower,hupper,i,(HYPRE_Complex*)(z+(size*i)));
    }
    CHKERRQ(VecRestoreArrayRead(x,&xx));
    PetscStackCallStandard(HYPRE_SStructVectorAssemble,mx->ss_b);
    PetscStackCallStandard(HYPRE_SStructMatrixMatvec,1.0,mx->ss_mat,mx->ss_b,0.0,mx->ss_x);

    /* copy solution values back to PETSc */
    CHKERRQ(VecGetArray(y,&yy));
    for (i= 0; i< nvars; i++) {
      PetscStackCallStandard(HYPRE_SStructVectorGetBoxValues,mx->ss_x,part,hlower,hupper,i,(HYPRE_Complex*)(z+(size*i)));
    }
    /* transform hypre's variable ordering for sys_pfmg to nodal ordering */
    for (i= 0; i< size; i++) {
      k= i*nvars;
      for (j= 0; j< nvars; j++) yy[k+j]= z[j*size+i];
    }
    CHKERRQ(VecRestoreArray(y,&yy));
    CHKERRQ(PetscFree(z));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_HYPRESStruct(Mat mat,MatAssemblyType mode)
{
  Mat_HYPRESStruct *ex = (Mat_HYPRESStruct*) mat->data;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_SStructMatrixAssemble,ex->ss_mat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_HYPRESStruct(Mat mat)
{
  PetscFunctionBegin;
  /* before the DMDA is set to the matrix the zero doesn't need to do anything */
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_HYPRESStruct(Mat mat)
{
  Mat_HYPRESStruct       *ex = (Mat_HYPRESStruct*) mat->data;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalToGlobalMapping(ex->da,&ltog));
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(ltog, (const PetscInt **) &ex->gindices));
  PetscStackCallStandard(HYPRE_SStructGraphDestroy,ex->ss_graph);
  PetscStackCallStandard(HYPRE_SStructMatrixDestroy,ex->ss_mat);
  PetscStackCallStandard(HYPRE_SStructVectorDestroy,ex->ss_x);
  PetscStackCallStandard(HYPRE_SStructVectorDestroy,ex->ss_b);
  CHKERRQ(PetscObjectDereference((PetscObject)ex->da));
  CHKERRMPI(MPI_Comm_free(&(ex->hcomm)));
  CHKERRQ(PetscFree(ex));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_HYPRESStruct(Mat B)
{
  Mat_HYPRESStruct *ex;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(B,&ex));
  B->data      = (void*)ex;
  B->rmap->bs  = 1;
  B->assembled = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;

  B->ops->assemblyend = MatAssemblyEnd_HYPRESStruct;
  B->ops->mult        = MatMult_HYPRESStruct;
  B->ops->zeroentries = MatZeroEntries_HYPRESStruct;
  B->ops->destroy     = MatDestroy_HYPRESStruct;
  B->ops->setup       = MatSetUp_HYPRESStruct;

  ex->needsinitialization = PETSC_TRUE;

  CHKERRMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)B),&(ex->hcomm)));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATHYPRESSTRUCT));
  PetscFunctionReturn(0);
}
