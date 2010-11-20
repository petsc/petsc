#define PETSCDM_DLL
 
#include "private/daimpl.h" /*I      "petscda.h"     I*/
#include "petscmat.h"         /*I      "petscmat.h"    I*/


EXTERN PetscErrorCode DAGetColoring1d_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN PetscErrorCode DAGetColoring2d_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN PetscErrorCode DAGetColoring2d_5pt_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN PetscErrorCode DAGetColoring3d_MPIAIJ(DA,ISColoringType,ISColoring *);

/*
   For ghost i that may be negative or greater than the upper bound this
  maps it into the 0:m-1 range using periodicity
*/
#define SetInRange(i,m) ((i < 0) ? m+i:((i >= m) ? i-m:i))

#undef __FUNCT__  
#define __FUNCT__ "DASetBlockFills_Private"
static PetscErrorCode DASetBlockFills_Private(PetscInt *dfill,PetscInt w,PetscInt **rfill)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nz,*fill;

  PetscFunctionBegin;
  if (!dfill) PetscFunctionReturn(0);

  /* count number nonzeros */
  nz = 0; 
  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      if (dfill[w*i+j]) nz++;
    }
  }
  ierr = PetscMalloc((nz + w + 1)*sizeof(PetscInt),&fill);CHKERRQ(ierr);
  /* construct modified CSR storage of nonzero structure */
  nz = w + 1;
  for (i=0; i<w; i++) {
    fill[i] = nz;
    for (j=0; j<w; j++) {
      if (dfill[w*i+j]) {
	fill[nz] = j;
	nz++;
      }
    }
  }
  fill[w] = nz;
   
  *rfill = fill;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetMatPreallocateOnly"
/*@
    DASetMatPreallocateOnly - When DAGetMatrix() is called the matrix will be properly
       preallocated but the nonzero structure and zero values will not be set.

    Collective on DA

    Input Parameter:
+   da - the distributed array
-   only - PETSC_TRUE if only want preallocation


    Level: developer

.seealso DAGetMatrix(), DASetGetMatrix(), DASetBlockSize(), DASetBlockFills()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetMatPreallocateOnly(DA da,PetscTruth only)
{
  PetscFunctionBegin;
  da->prealloc_only = only;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetBlockFills"
/*@
    DASetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DAGetMatrix().

    Collective on DA

    Input Parameter:
+   da - the distributed array
.   dfill - the fill pattern in the diagonal block (may be PETSC_NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks


    Level: developer

    Notes: This only makes sense when you are doing multicomponent problems but using the
       MPIAIJ matrix format

           The format for dfill and ofill is a 2 dimensional dof by dof matrix with 1 entries
       representing coupling and 0 entries for missing coupling. For example 
$             dfill[9] = {1, 0, 0,
$                         1, 1, 0,
$                         0, 1, 1} 
       means that row 0 is coupled with only itself in the diagonal block, row 1 is coupled with 
       itself and row 0 (in the diagonal block) and row 2 is coupled with itself and row 1 (in the 
       diagonal block).

     DASetGetMatrix() allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Glenn Hammond

.seealso DAGetMatrix(), DASetGetMatrix(), DASetMatPreallocateOnly()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetBlockFills(DA da,PetscInt *dfill,PetscInt *ofill)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DASetBlockFills_Private(dfill,da->w,&da->dfill);CHKERRQ(ierr);
  ierr = DASetBlockFills_Private(ofill,da->w,&da->ofill);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring" 
/*@
    DAGetColoring - Gets the coloring required for computing the Jacobian via
    finite differences on a function defined using a stencil on the DA.

    Collective on DA

    Input Parameter:
+   da - the distributed array
.   ctype - IS_COLORING_GLOBAL or IS_COLORING_GHOSTED
-   mtype - either MATAIJ or MATBAIJ

    Output Parameters:
.   coloring - matrix coloring for use in computing Jacobians (or PETSC_NULL if not needed)

    Level: advanced

    Notes: These compute the graph coloring of the graph of A^{T}A. The coloring used 
   for efficient (parallel or thread based) triangular solves etc is NOT
   available. 

        For BAIJ matrices this colors the graph for the blocks, not for the individual matrix elements;
    the same as MatGetColoring().

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), ISColoringType, ISColoring

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetColoring(DA da,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode ierr;
  PetscInt       dim,m,n,p,nc;
  DAPeriodicType wrap;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscTruth     isBAIJ;

  PetscFunctionBegin;
  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           yn  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     xn                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */  
  ierr = DAGetInfo(da,&dim,0,0,0,&m,&n,&p,&nc,0,&wrap,0);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GHOSTED){
    if (size == 1) {
      ctype = IS_COLORING_GLOBAL;
    } else if (dim > 1){
      if ((m==1 && DAXPeriodic(wrap)) || (n==1 && DAYPeriodic(wrap)) || (p==1 && DAZPeriodic(wrap))){
        SETERRQ(PETSC_ERR_SUP,"IS_COLORING_GHOSTED cannot be used for periodic boundary condition having both ends of the domain  on the same process");
      }
    }
  }

  /* Tell the DA it has 1 degree of freedom per grid point so that the coloring for BAIJ 
     matrices is for the blocks, not the individual matrix elements  */
  ierr = PetscStrcmp(mtype,MATBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (!isBAIJ) {ierr = PetscStrcmp(mtype,MATMPIBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (!isBAIJ) {ierr = PetscStrcmp(mtype,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (isBAIJ) {
    da->w = 1;
    da->xs = da->xs/nc;
    da->xe = da->xe/nc;
    da->Xs = da->Xs/nc;
    da->Xe = da->Xe/nc;
  }

  /*
     We do not provide a getcoloring function in the DA operations because 
   the basic DA does not know about matrices. We think of DA as being more 
   more low-level then matrices.
  */
  if (dim == 1) {
    ierr = DAGetColoring1d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr =  DAGetColoring2d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr =  DAGetColoring3d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Not done for %D dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
  }
  if (isBAIJ) {
    da->w = nc;
    da->xs = da->xs*nc;
    da->xe = da->xe*nc;
    da->Xs = da->Xs*nc;
    da->Xe = da->Xe*nc;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring2d_MPIAIJ" 
PetscErrorCode DAGetColoring2d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,M,N,dim,s,k,nc,col;
  PetscInt               ncolors;
  MPI_Comm               comm;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISColoringValue        *colors;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,&M,&N,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* special case as taught to us by Paul Hovland */
  if (st == DA_STENCIL_STAR && s == 1) {
    ierr = DAGetColoring2d_5pt_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else {

    if (DAXPeriodic(wrap) && (m % col)){ 
      SETERRQ2(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X (%d) is divisible\n\
                 by 2*stencil_width + 1 (%d)\n", m, col);
    }
    if (DAYPeriodic(wrap) && (n % col)){ 
      SETERRQ2(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y (%d) is divisible\n\
                 by 2*stencil_width + 1 (%d)\n", n, col);
    }
    if (ctype == IS_COLORING_GLOBAL) {
      if (!da->localcoloring) {
	ierr = PetscMalloc(nc*nx*ny*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
	ii = 0;
	for (j=ys; j<ys+ny; j++) {
	  for (i=xs; i<xs+nx; i++) {
	    for (k=0; k<nc; k++) {
	      colors[ii++] = k + nc*((i % col) + col*(j % col));
	    }
	  }
	}
        ncolors = nc + nc*(col-1 + col*(col-1));
	ierr = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,&da->localcoloring);CHKERRQ(ierr);
      }
      *coloring = da->localcoloring;
    } else if (ctype == IS_COLORING_GHOSTED) {
      if (!da->ghostedcoloring) {
	ierr = PetscMalloc(nc*gnx*gny*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
	ii = 0;
	for (j=gys; j<gys+gny; j++) {
	  for (i=gxs; i<gxs+gnx; i++) {
	    for (k=0; k<nc; k++) {
	      /* the complicated stuff is to handle periodic boundaries */
	      colors[ii++] = k + nc*((SetInRange(i,m) % col) + col*(SetInRange(j,n) % col));
	    }
	  }
	}
        ncolors = nc + nc*(col - 1 + col*(col-1));
	ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,&da->ghostedcoloring);CHKERRQ(ierr);
        /* PetscIntView(ncolors,(PetscInt *)colors,0); */

	ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
      }
      *coloring = da->ghostedcoloring;
    } else SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring3d_MPIAIJ" 
PetscErrorCode DAGetColoring3d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,ys,nx,ny,i,j,gxs,gys,gnx,gny,m,n,p,dim,s,k,nc,col,zs,gzs,ii,l,nz,gnz,M,N,P;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DAPeriodicType  wrap;
  DAStencilType   st;
  ISColoringValue *colors;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  if (DAXPeriodic(wrap) && (m % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAYPeriodic(wrap) && (n % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAZPeriodic(wrap) && (p % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");
  }

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*ny*nz*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      ii = 0;
      for (k=zs; k<zs+nz; k++) {
        for (j=ys; j<ys+ny; j++) {
          for (i=xs; i<xs+nx; i++) {
            for (l=0; l<nc; l++) {
              colors[ii++] = l + nc*((i % col) + col*(j % col) + col*col*(k % col));
            }
          }
        }
      }
      ncolors = nc + nc*(col-1 + col*(col-1)+ col*col*(col-1));
      ierr = ISColoringCreate(comm,ncolors,nc*nx*ny*nz,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*gny*gnz*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      ii = 0;
      for (k=gzs; k<gzs+gnz; k++) {
        for (j=gys; j<gys+gny; j++) {
          for (i=gxs; i<gxs+gnx; i++) {
            for (l=0; l<nc; l++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = l + nc*((SetInRange(i,m) % col) + col*(SetInRange(j,n) % col) + col*col*(SetInRange(k,p) % col));
            }
          }
        }
      }
      ncolors = nc + nc*(col-1 + col*(col-1)+ col*col*(col-1));
      ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny*gnz,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring1d_MPIAIJ" 
PetscErrorCode DAGetColoring1d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,nx,i,i1,gxs,gnx,l,m,M,dim,s,nc,col;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DAPeriodicType  wrap;
  ISColoringValue *colors;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,0,0,&M,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  if (DAXPeriodic(wrap) && (m % col)) {
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points is divisible\n\
                 by 2*stencil_width + 1\n");
  }

  ierr = DAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=xs; i<xs+nx; i++) {
        for (l=0; l<nc; l++) {
          colors[i1++] = l + nc*(i % col);
        }
      }
      ncolors = nc + nc*(col-1);
      ierr = ISColoringCreate(comm,ncolors,nc*nx,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=gxs; i<gxs+gnx; i++) {
        for (l=0; l<nc; l++) {
          /* the complicated stuff is to handle periodic boundaries */
          colors[i1++] = l + nc*(SetInRange(i,m) % col);
        }
      }
      ncolors = nc + nc*(col-1);
      ierr = ISColoringCreate(comm,ncolors,nc*gnx,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring2d_5pt_MPIAIJ" 
PetscErrorCode DAGetColoring2d_5pt_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,dim,s,k,nc;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DAPeriodicType  wrap;
  ISColoringValue *colors;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr   = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  ierr   = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr   = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr   = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  if (DAXPeriodic(wrap) && (m % 5)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 5\n");
  }
  if (DAYPeriodic(wrap) && (n % 5)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 5\n");
  }

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*ny*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=ys; j<ys+ny; j++) {
	for (i=xs; i<xs+nx; i++) {
	  for (k=0; k<nc; k++) {
	    colors[ii++] = k + nc*((3*j+i) % 5);
	  }
	}
      }
      ncolors = 5*nc;
      ierr = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*gny*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=gys; j<gys+gny; j++) {
	for (i=gxs; i<gxs+gnx; i++) {
	  for (k=0; k<nc; k++) {
	    colors[ii++] = k + nc*((3*SetInRange(j,n) + SetInRange(i,m)) % 5);
	  }
	}
      }
      ncolors = 5*nc;
      ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  PetscFunctionReturn(0);
}

/* =========================================================================== */
EXTERN PetscErrorCode DAGetMatrix1d_MPIAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix2d_MPIAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix2d_MPIAIJ_Fill(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix3d_MPIAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix3d_MPIAIJ_Fill(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix2d_MPIBAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix3d_MPIBAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix2d_MPISBAIJ(DA,Mat);
EXTERN PetscErrorCode DAGetMatrix3d_MPISBAIJ(DA,Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatSetDA"
/*@
   MatSetDA - Sets the DA that is to be used by the HYPRE_StructMatrix PETSc matrix

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  da - the da

   Level: intermediate

@*/
PetscErrorCode PETSCKSP_DLLEXPORT MatSetDA(Mat mat,DA da)
{
  PetscErrorCode ierr,(*f)(Mat,DA);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSetDA_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,da);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix" 
/*@C
    DAGetMatrix - Creates a matrix with the correct parallel layout and nonzero structure required for 
      computing the Jacobian on a function defined using the stencil set in the DA.

    Collective on DA

    Input Parameter:
+   da - the distributed array
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATBAIJ, MATSBAIJ)

    Output Parameters:
.   J  - matrix with the correct nonzero structure
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself. 

       By default it also sets the nonzero structure and puts in the zero entries. To prevent setting 
       the nonzero pattern call DASetMatPreallocateOnly()

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DASetBlockFills(), DASetMatPreallocateOnly()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetMatrix(DA da, const MatType mtype,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       dim,dof,nx,ny,nz,dims[3],starts[3];
  Mat            A;
  MPI_Comm       comm;
  const MatType  Atype;
  void           (*aij)(void)=PETSC_NULL,(*baij)(void)=PETSC_NULL,(*sbaij)(void)=PETSC_NULL;
  MatType        ttype[256];
  PetscTruth     flg;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscStrcpy((char*)ttype,mtype);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA options","Mat");CHKERRQ(ierr); 
  ierr = PetscOptionsList("-da_mat_type","Matrix type","MatSetType",MatList,mtype,(char*)ttype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           ny  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     nx                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,0,0,0,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,dof*nx*ny*nz,dof*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(A,(const MatType)ttype);CHKERRQ(ierr); 
  ierr = MatSetDA(A,da);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetType(A,&Atype);CHKERRQ(ierr);
  /*
     We do not provide a getmatrix function in the DA operations because 
   the basic DA does not know about matrices. We think of DA as being more 
   more low-level than matrices. This is kind of cheating but, cause sometimes 
   we think of DA has higher level than matrices.

     We could switch based on Atype (or mtype), but we do not since the
   specialized setting routines depend only the particular preallocation
   details of the matrix, not the type itself.
  */
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  if (!aij) {
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  }
  if (!aij) {
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    if (!baij) {
      ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    }
    if (!baij){
      ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
      if (!sbaij) {
        ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
      }
      if (!sbaij) {
        PetscTruth flg, flg2;
        ierr = PetscTypeCompare((PetscObject)A,MATHYPRESTRUCT,&flg);CHKERRQ(ierr);
        ierr = PetscTypeCompare((PetscObject)A,MATHYPRESSTRUCT,&flg2);CHKERRQ(ierr);
        if (!flg && !flg2) SETERRQ2(PETSC_ERR_SUP,"Not implemented for the matrix type: %s in %D dimension!\n" \
                           "Send mail to petsc-maint@mcs.anl.gov for code",Atype,dim);
      }
    }
  }
  if (aij) {
    if (dim == 1) {
      ierr = DAGetMatrix1d_MPIAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 2) {
      if (da->ofill) {
        ierr = DAGetMatrix2d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DAGetMatrix2d_MPIAIJ(da,A);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      if (da->ofill) {
        ierr = DAGetMatrix3d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DAGetMatrix3d_MPIAIJ(da,A);CHKERRQ(ierr);
      }
    }
  } else if (baij) {
    if (dim == 2) {
      ierr = DAGetMatrix2d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DAGetMatrix3d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else {
      SETERRQ3(PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension!\n" \
	       "Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
    }
  } else if (sbaij) {
    if (dim == 2) {
      ierr = DAGetMatrix2d_MPISBAIJ(da,A);CHKERRQ(ierr); 
    } else if (dim == 3) {
      ierr = DAGetMatrix3d_MPISBAIJ(da,A);CHKERRQ(ierr);
    } else {
      SETERRQ3(PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension!\n" \
	       "Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
    }
  } 
  ierr = DAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(A,dim,dims,starts,dof);CHKERRQ(ierr);
  *J = A;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix2d_MPIAIJ" 
PetscErrorCode DAGetMatrix2d_MPIAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny,m,n,dim,s,*cols = PETSC_NULL,k,nc,*rows = PETSC_NULL,col,cnt,l,p;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog,ltogb;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,PetscInt,&rows,col*col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);
  
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    pend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      lend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));

      cnt  = 0;
      for (k=0; k<nc; k++) {
	for (l=lstart; l<lend+1; l++) {
	  for (p=pstart; p<pend+1; p++) {
	    if ((st == DA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
	      cols[cnt++]  = k + nc*(slot + gnx*l + p);
	    }
	  }
	}
	rows[k] = k + nc*(slot);
      }
      ierr = MatPreallocateSetLocal(ltog,nc,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
    
      pstart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      pend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      
      for (j=ys; j<ys+ny; j++) {
	slot = i - gxs + gnx*(j - gys);
      
	lstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	lend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1)); 

	cnt  = 0;
	for (k=0; k<nc; k++) {
	  for (l=lstart; l<lend+1; l++) {
	    for (p=pstart; p<pend+1; p++) {
	      if ((st == DA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
		cols[cnt++]  = k + nc*(slot + gnx*l + p);
	      }
	    }
	  }
	  rows[k]      = k + nc*(slot);
	}
	ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix2d_MPIAIJ_Fill" 
PetscErrorCode DAGetMatrix2d_MPIAIJ_Fill(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt,l,p;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  PetscInt               ifill_col,*ofill = da->ofill, *dfill = da->dfill;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog,ltogb;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);
  
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    pend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      lend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));

      for (k=0; k<nc; k++) {
        cnt  = 0;
	for (l=lstart; l<lend+1; l++) {
	  for (p=pstart; p<pend+1; p++) {
            if (l || p) {
	      if ((st == DA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
                for (ifill_col=ofill[k]; ifill_col<ofill[k+1]; ifill_col++)
		  cols[cnt++]  = ofill[ifill_col] + nc*(slot + gnx*l + p);
	      }
            } else {
	      if (dfill) {
		for (ifill_col=dfill[k]; ifill_col<dfill[k+1]; ifill_col++)
		  cols[cnt++]  = dfill[ifill_col] + nc*(slot + gnx*l + p);
	      } else {
		for (ifill_col=0; ifill_col<nc; ifill_col++)
		  cols[cnt++]  = ifill_col + nc*(slot + gnx*l + p);
	      }
            }
	  }
	}
	row = k + nc*(slot);
        ierr = MatPreallocateSetLocal(ltog,1,&row,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
    
      pstart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      pend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      
      for (j=ys; j<ys+ny; j++) {
	slot = i - gxs + gnx*(j - gys);
      
	lstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	lend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1)); 

	for (k=0; k<nc; k++) {
	  cnt  = 0;
	  for (l=lstart; l<lend+1; l++) {
	    for (p=pstart; p<pend+1; p++) {
	      if (l || p) {
		if ((st == DA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
		  for (ifill_col=ofill[k]; ifill_col<ofill[k+1]; ifill_col++)
		    cols[cnt++]  = ofill[ifill_col] + nc*(slot + gnx*l + p);
		}
	      } else {
		if (dfill) {
		  for (ifill_col=dfill[k]; ifill_col<dfill[k+1]; ifill_col++)
		    cols[cnt++]  = dfill[ifill_col] + nc*(slot + gnx*l + p);
		} else {
		  for (ifill_col=0; ifill_col<nc; ifill_col++)
		    cols[cnt++]  = ifill_col + nc*(slot + gnx*l + p);
		}
	      }
	    }
	  }
	  row  = k + nc*(slot);
	  ierr = MatSetValuesLocal(J,1,&row,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPIAIJ" 
PetscErrorCode DAGetMatrix3d_MPIAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols = PETSC_NULL,k,nc,*rows = PETSC_NULL,col,cnt,l,p,*dnz = PETSC_NULL,*onz = PETSC_NULL;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog,ltogb;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,PetscInt,&rows,col*col*col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
	kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
	kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	
	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	cnt  = 0;
	for (l=0; l<nc; l++) {
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		  cols[cnt++]  = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		}
	      }
	    }
	  }
	  rows[l] = l + nc*(slot);
	}
	ierr = MatPreallocateSetLocal(ltog,nc,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr)
  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
	  kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	  cnt  = 0;
	  for (l=0; l<nc; l++) {
	    for (ii=istart; ii<iend+1; ii++) {
	      for (jj=jstart; jj<jend+1; jj++) {
		for (kk=kstart; kk<kend+1; kk++) {
		  if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		    cols[cnt++]  = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		  }
		}
	      }
	    }
	    rows[l]      = l + nc*(slot);
	  }
	  ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix1d_MPIAIJ" 
PetscErrorCode DAGetMatrix1d_MPIAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,nx,i,i1,slot,gxs,gnx;           
  PetscInt               m,dim,s,*cols = PETSC_NULL,nc,*rows = PETSC_NULL,col,cnt,l;
  PetscInt               istart,iend;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,0,0,0,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);

  ierr = MatSeqAIJSetPreallocation(J,col*nc,0);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,col*nc,0,col*nc,0);CHKERRQ(ierr);
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = PetscMalloc2(nc,PetscInt,&rows,col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);
  
  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = PetscMax(-s,gxs - i);
      iend   = PetscMin(s,gxs + gnx - i - 1);
      slot   = i - gxs;
    
      cnt  = 0;
      for (l=0; l<nc; l++) {
	for (i1=istart; i1<iend+1; i1++) {
	  cols[cnt++] = l + nc*(slot + i1);
	}
	rows[l]      = l + nc*(slot);
      }
      ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix2d_MPIBAIJ" 
PetscErrorCode DAGetMatrix2d_MPIBAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      slot = i - gxs + gnx*(j - gys); 

      /* Find block columns in block row */
      cnt  = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
            cols[cnt++]  = slot + ii + gnx*jj;
          }
        }
      }
      ierr = MatPreallocateSetLocal(ltogb,1,&slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
    } 
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
	slot = i - gxs + gnx*(j - gys); 
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
              cols[cnt++]  = slot + ii + gnx*jj;
            }
          }
        }
	ierr = MatSetValuesBlockedLocal(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPIBAIJ" 
PetscErrorCode DAGetMatrix3d_MPIBAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr  = PetscMalloc(col*col*col*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
	kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
	kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));

	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

	/* Find block columns in block row */
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	}
	ierr = MatPreallocateSetLocal(ltogb,1,&slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr  = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr  = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
	  kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	  cnt  = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                  cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
                }
              }
            }
          }
	  ierr = MatSetValuesBlockedLocal(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "L2GFilterUpperTriangular"
/*
  This helper is for of SBAIJ preallocation, to discard the lower-triangular values which are difficult to
  identify in the local ordering with periodic domain.
*/
static PetscErrorCode L2GFilterUpperTriangular(ISLocalToGlobalMapping ltog,PetscInt *row,PetscInt *cnt,PetscInt col[])
{
  PetscErrorCode ierr;
  PetscInt i,n;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingApply(ltog,1,row,row);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltog,*cnt,col,col);CHKERRQ(ierr);
  for (i=0,n=0; i<*cnt; i++) {
    if (col[i] >= *row) col[n++] = col[i];
  }
  *cnt = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix2d_MPISBAIJ" 
PetscErrorCode DAGetMatrix2d_MPISBAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateSymmetricInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j));
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      slot = i - gxs + gnx*(j - gys);

      /* Find block columns in block row */
      cnt  = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DA_STENCIL_BOX || !ii || !jj) {
            cols[cnt++]  = slot + ii + gnx*jj;
          }
        }
      }
      ierr = L2GFilterUpperTriangular(ltogb,&slot,&cnt,cols);CHKERRQ(ierr);
      ierr = MatPreallocateSymmetricSet(slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
    } 
  }
  ierr = MatSeqSBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPISBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j));
        jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
        slot = i - gxs + gnx*(j - gys);

        /* Find block columns in block row */
        cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DA_STENCIL_BOX || !ii || !jj) {
              cols[cnt++]  = slot + ii + gnx*jj;
            }
          }
        }
        ierr = L2GFilterUpperTriangular(ltogb,&slot,&cnt,cols);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPISBAIJ" 
PetscErrorCode DAGetMatrix3d_MPISBAIJ(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem 
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the matrix */
  ierr = PetscMalloc(col*col*col*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateSymmetricInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j));
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k));
	kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));

	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

	/* Find block columns in block row */
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
                cols[cnt++] = slot + ii + gnx*jj + gnx*gny*kk;
              }
            }
          }
        }
        ierr = L2GFilterUpperTriangular(ltogb,&slot,&cnt,cols);CHKERRQ(ierr);
        ierr = MatPreallocateSymmetricSet(slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqSBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPISBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j));
	jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
          kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k));
	  kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	  
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	  
	  cnt  = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
		  cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
		}
	      }
	    }
	  }
          ierr = L2GFilterUpperTriangular(ltogb,&slot,&cnt,cols);CHKERRQ(ierr);
          ierr = MatSetValuesBlocked(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPIAIJ_Fill" 
PetscErrorCode DAGetMatrix3d_MPIAIJ_Fill(DA da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt,l,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  PetscInt               ifill_col,*dfill = da->dfill,*ofill = da->ofill;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog,ltogb;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  if (DAXPeriodic(wrap) && (m % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAYPeriodic(wrap) && (n % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAZPeriodic(wrap) && (p % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");
  }

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*col*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr); 


  for (i=xs; i<xs+nx; i++) {
    istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
    iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
      jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
        kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	
        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	for (l=0; l<nc; l++) {
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		if (ii || jj || kk) {
		  if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		    for (ifill_col=ofill[l]; ifill_col<ofill[l+1]; ifill_col++)
		      cols[cnt++]  = ofill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		  }
		} else {
		  if (dfill) {
		    for (ifill_col=dfill[l]; ifill_col<dfill[l+1]; ifill_col++)
		      cols[cnt++]  = dfill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		  } else {
		    for (ifill_col=0; ifill_col<nc; ifill_col++)
		      cols[cnt++]  = ifill_col + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		  }
		}
	      }
	    }
	  }
	  row  = l + nc*(slot);
	  ierr = MatPreallocateSetLocal(ltog,1,&row,cnt,cols,dnz,onz);CHKERRQ(ierr); 
	}
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr); 
  ierr = MatSetLocalToGlobalMapping(J,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = DAXPeriodic(wrap) ? -s : (PetscMax(-s,-i));
      iend   = DAXPeriodic(wrap) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = DAYPeriodic(wrap) ? -s : (PetscMax(-s,-j)); 
	jend   = DAYPeriodic(wrap) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = DAZPeriodic(wrap) ? -s : (PetscMax(-s,-k)); 
	  kend   = DAZPeriodic(wrap) ?  s : (PetscMin(s,p-k-1));
	  
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	  
	  for (l=0; l<nc; l++) {
	    cnt  = 0;
	    for (ii=istart; ii<iend+1; ii++) {
	      for (jj=jstart; jj<jend+1; jj++) {
		for (kk=kstart; kk<kend+1; kk++) {
		  if (ii || jj || kk) {
		    if ((st == DA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		      for (ifill_col=ofill[l]; ifill_col<ofill[l+1]; ifill_col++)
			cols[cnt++]  = ofill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		    }
		  } else {
		    if (dfill) {
		      for (ifill_col=dfill[l]; ifill_col<dfill[l+1]; ifill_col++)
			cols[cnt++]  = dfill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		    } else {
		      for (ifill_col=0; ifill_col<nc; ifill_col++)
			cols[cnt++]  = ifill_col + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		    }
		  }
		}
	      }
	    }
	    row  = l + nc*(slot);
	    ierr = MatSetValuesLocal(J,1,&row,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
	  }
	}
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
