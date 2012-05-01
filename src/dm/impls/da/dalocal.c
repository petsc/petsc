
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

/*
   This allows the DMDA vectors to properly tell MATLAB their dimensions
*/
#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEnginePut_DA2d"
PetscErrorCode  VecMatlabEnginePut_DA2d(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  PetscInt       n,m;
  Vec            vec = (Vec)obj;
  PetscScalar    *array;
  mxArray        *mat;
  DM             da;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)vec,"DM",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(((PetscObject)vec)->comm,PETSC_ERR_ARG_WRONGSTATE,"Vector not associated with a DMDA");
  ierr = DMDAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRQ(ierr);

  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*m*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine *)mengine,obj->name,mat);
  
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif


#undef __FUNCT__  
#define __FUNCT__ "DMCreateLocalVector_DA"
PetscErrorCode  DMCreateLocalVector_DA(DM da,Vec* g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  ierr = VecCreate(PETSC_COMM_SELF,g);CHKERRQ(ierr);
  ierr = VecSetSizes(*g,dd->nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,dd->w);CHKERRQ(ierr);
  ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DM",(PetscObject)da);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  if (dd->w == 1  && dd->dim == 2) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)*g,"PetscMatlabEnginePut_C","VecMatlabEnginePut_DA2d",VecMatlabEnginePut_DA2d);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDACreateSection"
/*@C
  DMDACreateSection - Create a PetscSection inside the DMDA that describes data layout. This allows multiple fields with
  different numbers of dofs on vertices, cells, and faces in each direction.

  Input Parameters:
+ dm- The DMDA
. numFields - The number of fields
. numComp - The number of components in each field, or PETSC_NULL for 1
. numVertexDof - The number of dofs per vertex for each field, or PETSC_NULL
. numFaceDof - The number of dofs per face for each field and direction, or PETSC_NULL
- numCellDof - The number of dofs per cell for each field, or PETSC_NULL

  Level: developer

  Note:
  The default DMDA numbering is as follows:

    - Cells:    [0,             nC)
    - Vertices: [nC,            nC+nV)
    - X-Faces:  [nC+nV,         nC+nV+nXF)         normal is +- x-dir
    - Y-Faces:  [nC+nV+nXF,     nC+nV+nXF+nYF)     normal is +- y-dir
    - Z-Faces:  [nC+nV+nXF+nYF, nC+nV+nXF+nYF+nZF) normal is +- z-dir

  We interpret the default DMDA partition as a cell partition, and the data assignment as a cell assignment.
@*/
PetscErrorCode DMDACreateSection(DM dm, PetscInt numComp[], PetscInt numVertexDof[], PetscInt numFaceDof[], PetscInt numCellDof[])
{
  DM_DA         *da  = (DM_DA *) dm->data;
  const PetscInt dim = da->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nC  = (mx  )*(dim > 1 ? (my  )*(dim > 2 ? (mz  ) : 1) : 1);
  const PetscInt nVx = mx+1;
  const PetscInt nVy = dim > 1 ? (my+1) : 1;
  const PetscInt nVz = dim > 2 ? (mz+1) : 1;
  const PetscInt nV  = nVx*nVy*nVz;
  const PetscInt nxF = (dim > 1 ? (my  )*(dim > 2 ? (mz  ) : 1) : 1);
  const PetscInt nXF = (mx+1)*nxF;
  const PetscInt nyF = mx*(dim > 2 ? mz : 1);
  const PetscInt nYF = dim > 1 ? (my+1)*nyF : 0;
  const PetscInt nzF = mx*(dim > 1 ? my : 0);
  const PetscInt nZF = dim > 2 ? (mz+1)*nzF : 0;
  const PetscInt cStart  = 0,     cEnd  = cStart+nC;
  const PetscInt vStart  = cEnd,  vEnd  = vStart+nV;
  const PetscInt xfStart = vEnd,  xfEnd = xfStart+nXF;
  const PetscInt yfStart = xfEnd, yfEnd = yfStart+nYF;
  const PetscInt zfStart = yfEnd, zfEnd = zfStart+nZF;
  const PetscInt pStart  = 0,     pEnd  = zfEnd;
  PetscInt       numFields, numVertexTotDof = 0, numCellTotDof = 0, numFaceTotDof[3] = {0, 0, 0};
  PetscSF        sf;
  PetscMPIInt    rank;
  const PetscMPIInt *neighbors;
  PetscInt      *localPoints;
  PetscSFNode   *remotePoints;
  PetscInt       nleaves = 0,  nleavesCheck = 0;
  PetscInt       f, v, c, xf, yf, zf, xn, yn, zn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  /* Create local section */
  ierr = DMDAGetInfo(dm, 0,0,0,0,0,0,0, &numFields, 0,0,0,0,0);CHKERRQ(ierr);
  for(f = 0; f < numFields; ++f) {
    if (numVertexDof) {numVertexTotDof  += numVertexDof[f];}
    if (numCellDof)   {numCellTotDof    += numCellDof[f];}
    if (numFaceDof)   {numFaceTotDof[0] += numFaceDof[f*dim+0];
                       numFaceTotDof[1] += dim > 1 ? numFaceDof[f*dim+1] : 0;
                       numFaceTotDof[2] += dim > 2 ? numFaceDof[f*dim+2] : 0;}
  }
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &dm->defaultSection);CHKERRQ(ierr);
  if (numFields > 1) {
    ierr = PetscSectionSetNumFields(dm->defaultSection, numFields);CHKERRQ(ierr);
    for(f = 0; f < numFields; ++f) {
      const char *name;

      ierr = DMDAGetFieldName(dm, f, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(dm->defaultSection, f, name);CHKERRQ(ierr);
      if (numComp) {
        ierr = PetscSectionSetFieldComponents(dm->defaultSection, f, numComp[f]);CHKERRQ(ierr);
      }
    }
  } else {
    numFields = 0;
  }
  ierr = PetscSectionSetChart(dm->defaultSection, pStart, pEnd);CHKERRQ(ierr);
  if (numVertexDof) {
    for(v = vStart; v < vEnd; ++v) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(dm->defaultSection, v, f, numVertexDof[f]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(dm->defaultSection, v, numVertexTotDof);CHKERRQ(ierr);
    }
  }
  if (numFaceDof) {
    for(xf = xfStart; xf < xfEnd; ++xf) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(dm->defaultSection, xf, f, numFaceDof[f*dim+0]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(dm->defaultSection, xf, numFaceTotDof[0]);CHKERRQ(ierr);
    }
    for(yf = yfStart; yf < yfEnd; ++yf) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(dm->defaultSection, yf, f, numFaceDof[f*dim+1]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(dm->defaultSection, yf, numFaceTotDof[1]);CHKERRQ(ierr);
    }
    for(zf = zfStart; zf < zfEnd; ++zf) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(dm->defaultSection, zf, f, numFaceDof[f*dim+2]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(dm->defaultSection, zf, numFaceTotDof[2]);CHKERRQ(ierr);
    }
  }
  if (numCellDof) {
    for(c = cStart; c < cEnd; ++c) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(dm->defaultSection, c, f, numCellDof[f]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(dm->defaultSection, c, numCellTotDof);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(dm->defaultSection);CHKERRQ(ierr);
  /* Create mesh point SF */
  ierr = DMDAGetNeighbors(dm, &neighbors);CHKERRQ(ierr);
  for(zn = 0; zn < (dim > 2 ? 3 : 1); ++zn) {
    for(yn = 0; yn < (dim > 1 ? 3 : 1); ++yn) {
      for(xn = 0; xn < 3; ++xn) {
        const PetscInt xp = xn-1, yp = dim > 1 ? yn-1 : 0, zp = dim > 2 ? zn-1 : 0;
        const PetscInt neighbor = neighbors[(zn*3+yn)*3+xn];

        if (neighbor >= 0 && neighbor != rank) {
          nleaves += (!xp ? nVx : 1) * (!yp ? nVy : 1) * (!zp ? nVz : 1); /* vertices */
          if (xp && !yp && !zp) {
            nleaves += nxF; /* x faces */
          } else if (yp && !zp && !xp) {
            nleaves += nyF; /* y faces */
          } else if (zp && !xp && !yp) {
            nleaves += nzF; /* z faces */
          }
        }
      }
    }
  }
  ierr = PetscMalloc2(nleaves,PetscInt,&localPoints,nleaves,PetscSFNode,&remotePoints);CHKERRQ(ierr);
  for(zn = 0; zn < (dim > 2 ? 3 : 1); ++zn) {
    for(yn = 0; yn < (dim > 1 ? 3 : 1); ++yn) {
      for(xn = 0; xn < 3; ++xn) {
        const PetscInt xp = xn-1, yp = dim > 1 ? yn-1 : 0, zp = dim > 2 ? zn-1 : 0;
        const PetscInt neighbor = neighbors[(zn*3+yn)*3+xn];

        if (neighbor >= 0 && neighbor != rank) {
          if (xp < 0) { /* left */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                nleavesCheck += 1; /* left bottom back vertex */
              } else if (zp > 0) { /* front */
                nleavesCheck += 1; /* left bottom front vertex */
              } else {
                nleavesCheck += nVz; /* left bottom vertices */
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                nleavesCheck += 1; /* left top back vertex */
              } else if (zp > 0) { /* front */
                nleavesCheck += 1; /* left top front vertex */
              } else {
                nleavesCheck += nVz; /* left top vertices */
              }
            } else {
              if (zp < 0) { /* back */
                nleavesCheck += nVy; /* left back vertices */
              } else if (zp > 0) { /* front */
                nleavesCheck += nVy; /* left front vertices */
              } else {
                nleavesCheck += nVy*nVz; /* left vertices */
                nleavesCheck += nxF;     /* left faces */
              }
            }
          } else if (xp > 0) { /* right */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                nleavesCheck += 1; /* right bottom back vertex */
              } else if (zp > 0) { /* front */
                nleavesCheck += 1; /* right bottom front vertex */
              } else {
                nleavesCheck += nVz; /* right bottom vertices */
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                nleavesCheck += 1; /* right top back vertex */
              } else if (zp > 0) { /* front */
                nleavesCheck += 1; /* right top front vertex */
              } else {
                nleavesCheck += nVz; /* right top vertices */
              }
            } else {
              if (zp < 0) { /* back */
                nleavesCheck += nVy; /* right back vertices */
              } else if (zp > 0) { /* front */
                nleavesCheck += nVy; /* right front vertices */
              } else {
                nleavesCheck += nVy*nVz; /* right vertices */
                nleavesCheck += nxF;     /* right faces */
              }
            }
          } else {
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                nleavesCheck += nVx; /* bottom back vertices */
              } else if (zp > 0) { /* front */
                nleavesCheck += nVx; /* bottom front vertices */
              } else {
                nleavesCheck += nVx*nVz; /* bottom vertices */
                nleavesCheck += nyF;     /* bottom faces */
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                nleavesCheck += nVx; /* top back vertices */
              } else if (zp > 0) { /* front */
                nleavesCheck += nVx; /* top front vertices */
              } else {
                nleavesCheck += nVx*nVz; /* top vertices */
                nleavesCheck += nyF;     /* top faces */
              }
            } else {
              if (zp < 0) { /* back */
                nleavesCheck += nVx*nVy; /* back vertices */
                nleavesCheck += nzF;     /* back faces */
              } else if (zp > 0) { /* front */
                nleavesCheck += nVx*nVy; /* front vertices */
                nleavesCheck += nzF;     /* front faces */
              } else {
                /* Nothing is shared from the interior */
              }
            }
          }
        }
      }
    }
  }
  if (nleaves != nleavesCheck) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "The number of leaves %d did not match the number of remote leaves %d", nleaves, nleavesCheck);
  ierr = PetscSFCreate(((PetscObject) dm)->comm, &sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf, pEnd, nleaves, localPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Create global section */
  ierr = PetscSectionCreateGlobalSection(dm->defaultSection, sf, &dm->defaultGlobalSection);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* Create default SF */
  ierr = DMCreateDefaultSF(dm, dm->defaultSection, dm->defaultGlobalSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#if defined(PETSC_HAVE_ADIC)

EXTERN_C_BEGIN
#include <adic/ad_utils.h>
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAdicArray"
/*@C
     DMDAGetAdicArray - Gets an array of derivative types for a DMDA

    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
+    vptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes:
       The vector values are NOT initialized and may have garbage in them, so you may need
       to zero them.

       Returns the same type of object as the DMDAVecGetArray() except its elements are 
           derivative types instead of PetscScalars

     Level: advanced

.seealso: DMDARestoreAdicArray()

@*/
PetscErrorCode  DMDAGetAdicArray(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,deriv_type_size,xs,ys,xm,ym,zs,zm,itdof;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->adarrayghostedin[i]) {
        *iptr                   = dd->adarrayghostedin[i];
        iarray_start            = (char*)dd->adstartghostedin[i];
        itdof                   = dd->ghostedtdof;
        dd->adarrayghostedin[i] = PETSC_NULL;
        dd->adstartghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
    zm = dd->Ze-dd->Zs;
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->adarrayin[i]) {
        *iptr            = dd->adarrayin[i];
        iarray_start     = (char*)dd->adstartin[i];
        itdof            = dd->tdof;
        dd->adarrayin[i] = PETSC_NULL;
        dd->adstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
    zm = dd->ze-dd->zs;
  }
  deriv_type_size = PetscADGetDerivTypeSize();

  switch (dd->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*deriv_type_size);
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*deriv_type_size - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + deriv_type_size*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*deriv_type_size - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*deriv_type_size + zm*sizeof(void**));

      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym - ys);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + deriv_type_size*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->adarrayghostedout[i]) {
        dd->adarrayghostedout[i] = *iptr ;
        dd->adstartghostedout[i] = iarray_start;
        dd->ghostedtdof          = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->adarrayout[i]) {
        dd->adarrayout[i] = *iptr ;
        dd->adstartout[i] = iarray_start;
        dd->tdof          = itdof;
        break;
      }
    }
  }
  if (i == DMDA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Too many DMDA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *(void**)array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreAdicArray"
/*@C
     DMDARestoreAdicArray - Restores an array of derivative types for a DMDA
          
    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
+    ptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly
-    tdof - total number of degrees of freedom represented in array_start

     Level: advanced

.seealso: DMDAGetAdicArray()

@*/
PetscErrorCode  DMDARestoreAdicArray(DM da,PetscBool  ghosted,void *ptr,void *array_start,PetscInt *tdof)
{
  PetscInt  i;
  void      **iptr = (void**)ptr,iarray_start = 0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->adarrayghostedout[i] == *iptr) {
        iarray_start             = dd->adstartghostedout[i];
        dd->adarrayghostedout[i] = PETSC_NULL;
        dd->adstartghostedout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->adarrayghostedin[i]){
        dd->adarrayghostedin[i] = *iptr;
        dd->adstartghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->adarrayout[i] == *iptr) {
        iarray_start      = dd->adstartout[i];
        dd->adarrayout[i] = PETSC_NULL;
        dd->adstartout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->adarrayin[i]){
        dd->adarrayin[i]   = *iptr;
        dd->adstartin[i]   = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ad_DAGetArray"
PetscErrorCode  ad_DAGetArray(DM da,PetscBool  ghosted,void *iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDAGetAdicArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ad_DARestoreArray"
PetscErrorCode  ad_DARestoreArray(DM da,PetscBool  ghosted,void *iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDARestoreAdicArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "DMDAGetArray"
/*@C
     DMDAGetArray - Gets a work array for a DMDA

    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
.    vptr - array data structured

    Note:  The vector values are NOT initialized and may have garbage in them, so you may need
           to zero them.

  Level: advanced

.seealso: DMDARestoreArray(), DMDAGetAdicArray()

@*/
PetscErrorCode  DMDAGetArray(DM da,PetscBool  ghosted,void *vptr)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedin[i]) {
        *iptr                 = dd->arrayghostedin[i];
        iarray_start          = (char*)dd->startghostedin[i];
        dd->arrayghostedin[i] = PETSC_NULL;
        dd->startghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
    zm = dd->Ze-dd->Zs;
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayin[i]) {
        *iptr          = dd->arrayin[i];
        iarray_start   = (char*)dd->startin[i];
        dd->arrayin[i] = PETSC_NULL;
        dd->startin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
    zm = dd->ze-dd->zs;
  }

  switch (dd->dim) {
    case 1: {
      void *ptr;

      ierr  = PetscMalloc(xm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym - ys);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedout[i]) {
        dd->arrayghostedout[i] = *iptr ;
        dd->startghostedout[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayout[i]) {
        dd->arrayout[i] = *iptr ;
        dd->startout[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreArray"
/*@C
     DMDARestoreArray - Restores an array of derivative types for a DMDA
          
    Input Parameter:
+    da - information about my local patch
.    ghosted - do you want arrays for the ghosted or nonghosted patch
-    vptr - array data structured to be passed to ad_FormFunctionLocal()

     Level: advanced

.seealso: DMDAGetArray(), DMDAGetAdicArray()

@*/
PetscErrorCode  DMDARestoreArray(DM da,PetscBool  ghosted,void *vptr)
{
  PetscInt  i;
  void      **iptr = (void**)vptr,*iarray_start = 0;
  DM_DA     *dd = (DM_DA*)da->data;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedout[i] == *iptr) {
        iarray_start           = dd->startghostedout[i];
        dd->arrayghostedout[i] = PETSC_NULL;
        dd->startghostedout[i] = PETSC_NULL;
        break;
      }
    }
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedin[i]){
        dd->arrayghostedin[i] = *iptr;
        dd->startghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayout[i] == *iptr) {
        iarray_start    = dd->startout[i];
        dd->arrayout[i] = PETSC_NULL;
        dd->startout[i] = PETSC_NULL;
        break;
      }
    }
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayin[i]){
        dd->arrayin[i]  = *iptr;
        dd->startin[i]  = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAdicMFArray"
/*@C
     DMDAGetAdicMFArray - Gets an array of derivative types for a DMDA for matrix-free ADIC.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    vptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes: 
     The vector values are NOT initialized and may have garbage in them, so you may need
     to zero them.

     This routine returns the same type of object as the DMDAVecGetArray(), except its
     elements are derivative types instead of PetscScalars.

     Level: advanced

.seealso: DMDARestoreAdicMFArray(), DMDAGetArray(), DMDAGetAdicArray()

@*/
PetscErrorCode  DMDAGetAdicMFArray(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayghostedin[i]) {
        *iptr                     = dd->admfarrayghostedin[i];
        iarray_start              = (char*)dd->admfstartghostedin[i];
        itdof                     = dd->ghostedtdof;
        dd->admfarrayghostedin[i] = PETSC_NULL;
        dd->admfstartghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
    zm = dd->Ze-dd->Zs;
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayin[i]) {
        *iptr              = dd->admfarrayin[i];
        iarray_start       = (char*)dd->admfstartin[i];
        itdof              = dd->tdof;
        dd->admfarrayin[i] = PETSC_NULL;
        dd->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
    zm = dd->ze-dd->zs;
  }

  switch (dd->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*2*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*2*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 2*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym* - ys)*sizeof(void*);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + 2*sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayghostedout[i]) {
        dd->admfarrayghostedout[i] = *iptr ;
        dd->admfstartghostedout[i] = iarray_start;
        dd->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayout[i]) {
        dd->admfarrayout[i] = *iptr ;
        dd->admfstartout[i] = iarray_start;
        dd->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DMDA_MAX_AD_ARRAYS+1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Too many DMDA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *(void**)array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAdicMFArray4"
PetscErrorCode  DMDAGetAdicMFArray4(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,itdof = 0;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayghostedin[i]) {
        *iptr                     = dd->admfarrayghostedin[i];
        iarray_start              = (char*)dd->admfstartghostedin[i];
        itdof                     = dd->ghostedtdof;
        dd->admfarrayghostedin[i] = PETSC_NULL;
        dd->admfstartghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayin[i]) {
        *iptr              = dd->admfarrayin[i];
        iarray_start       = (char*)dd->admfstartin[i];
        itdof              = dd->tdof;
        dd->admfarrayin[i] = PETSC_NULL;
        dd->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
  }

  switch (dd->dim) {
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*5*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*5*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 5*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayghostedout[i]) {
        dd->admfarrayghostedout[i] = *iptr ;
        dd->admfstartghostedout[i] = iarray_start;
        dd->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayout[i]) {
        dd->admfarrayout[i] = *iptr ;
        dd->admfstartout[i] = iarray_start;
        dd->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DMDA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Too many DMDA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *(void**)array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAdicMFArray9"
PetscErrorCode  DMDAGetAdicMFArray9(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,itdof = 0;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayghostedin[i]) {
        *iptr                     = dd->admfarrayghostedin[i];
        iarray_start              = (char*)dd->admfstartghostedin[i];
        itdof                     = dd->ghostedtdof;
        dd->admfarrayghostedin[i] = PETSC_NULL;
        dd->admfstartghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayin[i]) {
        *iptr              = dd->admfarrayin[i];
        iarray_start       = (char*)dd->admfstartin[i];
        itdof              = dd->tdof;
        dd->admfarrayin[i] = PETSC_NULL;
        dd->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
  }

  switch (dd->dim) {
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*10*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*10*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 10*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayghostedout[i]) {
        dd->admfarrayghostedout[i] = *iptr ;
        dd->admfstartghostedout[i] = iarray_start;
        dd->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayout[i]) {
        dd->admfarrayout[i] = *iptr ;
        dd->admfstartout[i] = iarray_start;
        dd->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DMDA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Too many DMDA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *(void**)array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAdicMFArrayb"
/*@C
     DMDAGetAdicMFArrayb - Gets an array of derivative types for a DMDA for matrix-free ADIC.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    vptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes: 
     The vector values are NOT initialized and may have garbage in them, so you may need
     to zero them.

     This routine returns the same type of object as the DMDAVecGetArray(), except its
     elements are derivative types instead of PetscScalars.

     Level: advanced

.seealso: DMDARestoreAdicMFArray(), DMDAGetArray(), DMDAGetAdicArray()

@*/
PetscErrorCode  DMDAGetAdicMFArrayb(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd = (DM_DA*)da->data;
  PetscInt       bs = dd->w,bs1 = bs+1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayghostedin[i]) {
        *iptr                     = dd->admfarrayghostedin[i];
        iarray_start              = (char*)dd->admfstartghostedin[i];
        itdof                     = dd->ghostedtdof;
        dd->admfarrayghostedin[i] = PETSC_NULL;
        dd->admfstartghostedin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
    zm = dd->Ze-dd->Zs;
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayin[i]) {
        *iptr              = dd->admfarrayin[i];
        iarray_start       = (char*)dd->admfstartin[i];
        itdof              = dd->tdof;
        dd->admfarrayin[i] = PETSC_NULL;
        dd->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
    zm = dd->ze-dd->zs;
  }

  switch (dd->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*bs1*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*bs1*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + bs1*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym* - ys)*sizeof(void*);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + bs1*sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayghostedout[i]) {
        dd->admfarrayghostedout[i] = *iptr ;
        dd->admfstartghostedout[i] = iarray_start;
        dd->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayout[i]) {
        dd->admfarrayout[i] = *iptr ;
        dd->admfstartout[i] = iarray_start;
        dd->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DMDA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Too many DMDA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *(void**)array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreAdicMFArray"
/*@C
     DMDARestoreAdicMFArray - Restores an array of derivative types for a DMDA.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    ptr - array data structure to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly
-    tdof - total number of degrees of freedom represented in array_start

     Level: advanced

.seealso: DMDAGetAdicArray()

@*/
PetscErrorCode  DMDARestoreAdicMFArray(DM da,PetscBool  ghosted,void *vptr,void *array_start,PetscInt *tdof)
{
  PetscInt  i;
  void      **iptr = (void**)vptr,*iarray_start = 0;
  DM_DA     *dd = (DM_DA*)da->data;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayghostedout[i] == *iptr) {
        iarray_start               = dd->admfstartghostedout[i];
        dd->admfarrayghostedout[i] = PETSC_NULL;
        dd->admfstartghostedout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayghostedin[i]){
        dd->admfarrayghostedin[i] = *iptr;
        dd->admfstartghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (dd->admfarrayout[i] == *iptr) {
        iarray_start        = dd->admfstartout[i];
        dd->admfarrayout[i] = PETSC_NULL;
        dd->admfstartout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
      if (!dd->admfarrayin[i]){
        dd->admfarrayin[i] = *iptr;
        dd->admfstartin[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "admf_DAGetArray"
PetscErrorCode  admf_DAGetArray(DM da,PetscBool  ghosted,void *iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDAGetAdicMFArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "admf_DARestoreArray"
PetscErrorCode  admf_DARestoreArray(DM da,PetscBool  ghosted,void *iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDARestoreAdicMFArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

