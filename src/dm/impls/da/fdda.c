 
#include <petsc-private/daimpl.h> /*I      "petscdmda.h"     I*/
#include <petscmat.h>         /*I      "petscmat.h"    I*/
#include <petsc-private/matimpl.h>

extern PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM,ISColoringType,ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM,ISColoringType,ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM,ISColoringType,ISColoring *);
extern PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM,ISColoringType,ISColoring *);

/*
   For ghost i that may be negative or greater than the upper bound this
  maps it into the 0:m-1 range using periodicity
*/
#define SetInRange(i,m) ((i < 0) ? m+i:((i >= m) ? i-m:i))

#undef __FUNCT__  
#define __FUNCT__ "DMDASetBlockFills_Private"
static PetscErrorCode DMDASetBlockFills_Private(PetscInt *dfill,PetscInt w,PetscInt **rfill)
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
#define __FUNCT__ "DMDASetBlockFills"
/*@
    DMDASetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMCreateMatrix().

    Logically Collective on DMDA

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

     DMDASetGetMatrix() allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Glenn Hammond

.seealso DMCreateMatrix(), DMDASetGetMatrix(), DMSetMatrixPreallocateOnly()

@*/
PetscErrorCode  DMDASetBlockFills(DM da,PetscInt *dfill,PetscInt *ofill)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDASetBlockFills_Private(dfill,dd->w,&dd->dfill);CHKERRQ(ierr);
  ierr = DMDASetBlockFills_Private(ofill,dd->w,&dd->ofill);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring_DA" 
PetscErrorCode  DMCreateColoring_DA(DM da,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode   ierr;
  PetscInt         dim,m,n,p,nc;
  DMDABoundaryType bx,by,bz;
  MPI_Comm         comm;
  PetscMPIInt      size;
  PetscBool        isBAIJ;
  DM_DA            *dd = (DM_DA*)da->data;

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
  ierr = DMDAGetInfo(da,&dim,0,0,0,&m,&n,&p,&nc,0,&bx,&by,&bz,0);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GHOSTED){
    if (size == 1) {
      ctype = IS_COLORING_GLOBAL;
    } else if (dim > 1){
      if ((m==1 && bx == DMDA_BOUNDARY_PERIODIC) || (n==1 && by == DMDA_BOUNDARY_PERIODIC) || (p==1 && bz == DMDA_BOUNDARY_PERIODIC)){
        SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"IS_COLORING_GHOSTED cannot be used for periodic boundary condition having both ends of the domain  on the same process");
      }
    }
  }

  /* Tell the DMDA it has 1 degree of freedom per grid point so that the coloring for BAIJ 
     matrices is for the blocks, not the individual matrix elements  */
  ierr = PetscStrcmp(mtype,MATBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (!isBAIJ) {ierr = PetscStrcmp(mtype,MATMPIBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (!isBAIJ) {ierr = PetscStrcmp(mtype,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (isBAIJ) {
    dd->w = 1;
    dd->xs = dd->xs/nc;
    dd->xe = dd->xe/nc;
    dd->Xs = dd->Xs/nc;
    dd->Xe = dd->Xe/nc;
  }

  /*
     We do not provide a getcoloring function in the DMDA operations because 
   the basic DMDA does not know about matrices. We think of DMDA as being more 
   more low-level then matrices.
  */
  if (dim == 1) {
    ierr = DMCreateColoring_DA_1d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr =  DMCreateColoring_DA_2d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr =  DMCreateColoring_DA_3d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_SUP,"Not done for %D dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
  if (isBAIJ) {
    dd->w = nc;
    dd->xs = dd->xs*nc;
    dd->xe = dd->xe*nc;
    dd->Xs = dd->Xs*nc;
    dd->Xe = dd->Xe*nc;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring_DA_2d_MPIAIJ" 
PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,M,N,dim,s,k,nc,col;
  PetscInt               ncolors;
  MPI_Comm               comm;
  DMDABoundaryType       bx,by;
  DMDAStencilType        st;
  ISColoringValue        *colors;
  DM_DA                  *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,0,&M,&N,0,&nc,&s,&bx,&by,0,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* special case as taught to us by Paul Hovland */
  if (st == DMDA_STENCIL_STAR && s == 1) {
    ierr = DMCreateColoring_DA_2d_5pt_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else {

    if (bx == DMDA_BOUNDARY_PERIODIC && (m % col)){ 
      SETERRQ2(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X (%d) is divisible\n\
                 by 2*stencil_width + 1 (%d)\n", m, col);
    }
    if (by == DMDA_BOUNDARY_PERIODIC && (n % col)){ 
      SETERRQ2(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y (%d) is divisible\n\
                 by 2*stencil_width + 1 (%d)\n", n, col);
    }
    if (ctype == IS_COLORING_GLOBAL) {
      if (!dd->localcoloring) {
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
	ierr = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,&dd->localcoloring);CHKERRQ(ierr);
      }
      *coloring = dd->localcoloring;
    } else if (ctype == IS_COLORING_GHOSTED) {
      if (!dd->ghostedcoloring) {
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
	ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,&dd->ghostedcoloring);CHKERRQ(ierr);
        /* PetscIntView(ncolors,(PetscInt *)colors,0); */

	ierr = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
      }
      *coloring = dd->ghostedcoloring;
    } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring_DA_3d_MPIAIJ" 
PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode    ierr;
  PetscInt          xs,ys,nx,ny,i,j,gxs,gys,gnx,gny,m,n,p,dim,s,k,nc,col,zs,gzs,ii,l,nz,gnz,M,N,P;
  PetscInt          ncolors;
  MPI_Comm          comm;
  DMDABoundaryType  bx,by,bz;
  DMDAStencilType   st;
  ISColoringValue   *colors;
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  if (bx == DMDA_BOUNDARY_PERIODIC && (m % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (by == DMDA_BOUNDARY_PERIODIC && (n % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (bz == DMDA_BOUNDARY_PERIODIC && (p % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");
  }

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
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
      ierr = ISColoringCreate(comm,ncolors,nc*nx*ny*nz,colors,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!dd->ghostedcoloring) {
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
      ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny*gnz,colors,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring_DA_1d_MPIAIJ" 
PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode    ierr;
  PetscInt          xs,nx,i,i1,gxs,gnx,l,m,M,dim,s,nc,col;
  PetscInt          ncolors;
  MPI_Comm          comm;
  DMDABoundaryType  bx;
  ISColoringValue   *colors;
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,0,0,&M,0,0,&nc,&s,&bx,0,0,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  if (bx == DMDA_BOUNDARY_PERIODIC && (m % col)) {
    SETERRQ2(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points %d is divisible\n\
                 by 2*stencil_width + 1 %d\n",(int)m,(int)col);
  }

  ierr = DMDAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      ierr = PetscMalloc(nc*nx*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=xs; i<xs+nx; i++) {
        for (l=0; l<nc; l++) {
          colors[i1++] = l + nc*(i % col);
        }
      }
      ncolors = nc + nc*(col-1);
      ierr = ISColoringCreate(comm,ncolors,nc*nx,colors,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!dd->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=gxs; i<gxs+gnx; i++) {
        for (l=0; l<nc; l++) {
          /* the complicated stuff is to handle periodic boundaries */
          colors[i1++] = l + nc*(SetInRange(i,m) % col);
        }
      }
      ncolors = nc + nc*(col-1);
      ierr = ISColoringCreate(comm,ncolors,nc*gnx,colors,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateColoring_DA_2d_5pt_MPIAIJ" 
PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode    ierr;
  PetscInt          xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,dim,s,k,nc;
  PetscInt          ncolors;
  MPI_Comm          comm;
  DMDABoundaryType  bx,by;
  ISColoringValue   *colors;
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr   = DMDAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&bx,&by,0,0);CHKERRQ(ierr);
  ierr   = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr   = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr   = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  if (bx == DMDA_BOUNDARY_PERIODIC && (m % 5)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 5\n");
  }
  if (by == DMDA_BOUNDARY_PERIODIC && (n % 5)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 5\n");
  }

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
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
      ierr = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!dd->ghostedcoloring) {
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
      ierr = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  PetscFunctionReturn(0);
}

/* =========================================================================== */
extern PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM,Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatSetDM"
/*@
   MatSetDM - Sets the DMDA that is to be used by the HYPRE_StructMatrix PETSc matrix

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  da - the da

   Level: intermediate

@*/
PetscErrorCode  MatSetDM(Mat mat,DM da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = PetscTryMethod(mat,"MatSetDM_C",(Mat,DM),(mat,da));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatView_MPI_DA"
PetscErrorCode  MatView_MPI_DA(Mat A,PetscViewer viewer)
{
  DM             da;
  PetscErrorCode ierr;
  const char     *prefix;
  Mat            Anatural;
  AO             ao;
  PetscInt       rstart,rend,*petsc,i;
  IS             is;
  MPI_Comm       comm;
  PetscViewerFormat format;

  PetscFunctionBegin;
  /* Check whether we are just printing info, in which case MatView() already viewed everything we wanted to view */
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"DM",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from a DMDA");

  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc((rend-rstart)*sizeof(PetscInt),&petsc);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) petsc[i-rstart] = i;
  ierr = AOApplicationToPetsc(ao,rend-rstart,petsc);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,rend-rstart,petsc,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);

  /* call viewer on natural ordering */
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&Anatural);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)A,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)Anatural,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Anatural,((PetscObject)A)->name);CHKERRQ(ierr);
  ierr = MatView(Anatural,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPI_DA"
PetscErrorCode  MatLoad_MPI_DA(Mat A,PetscViewer viewer)
{
  DM             da;
  PetscErrorCode ierr;
  Mat            Anatural,Aapp;
  AO             ao;
  PetscInt       rstart,rend,*app,i;
  IS             is;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"DM",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from a DMDA");

  /* Load the matrix in natural ordering */
  ierr = MatCreate(((PetscObject)A)->comm,&Anatural);CHKERRQ(ierr);
  ierr = MatSetType(Anatural,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetSizes(Anatural,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatLoad(Anatural,viewer);CHKERRQ(ierr);

  /* Map natural ordering to application ordering and create IS */
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Anatural,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc((rend-rstart)*sizeof(PetscInt),&app);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) app[i-rstart] = i;
  ierr = AOPetscToApplication(ao,rend-rstart,app);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,rend-rstart,app,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);

  /* Do permutation and replace header */
  ierr = MatGetSubMatrix(Anatural,is,is,MAT_INITIAL_MATRIX,&Aapp);CHKERRQ(ierr);
  ierr = MatHeaderReplace(A,Aapp);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix_DA" 
PetscErrorCode DMCreateMatrix_DA(DM da, const MatType mtype,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       dim,dof,nx,ny,nz,dims[3],starts[3],M,N,P;
  Mat            A;
  MPI_Comm       comm;
  const MatType  Atype;
  PetscSection   section, sectionGlobal;
  void           (*aij)(void)=PETSC_NULL,(*baij)(void)=PETSC_NULL,(*sbaij)(void)=PETSC_NULL;
  MatType        ttype[256];
  PetscBool      flg;
  PetscMPIInt    size;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  if (!mtype) mtype = MATAIJ;
  ierr = PetscStrcpy((char*)ttype,mtype);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DMDA options","Mat");CHKERRQ(ierr); 
  ierr = PetscOptionsList("-dm_mat_type","Matrix type","MatSetType",MatList,mtype,(char*)ttype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = DMGetDefaultSection(da, &section);CHKERRQ(ierr);
  if (section) {
    PetscInt  bs = -1;
    PetscInt  localSize;
    PetscBool isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock, isSymmetric;

    ierr = DMGetDefaultGlobalSection(da, &sectionGlobal);CHKERRQ(ierr);
    ierr = PetscSectionGetConstrainedStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr);
    ierr = MatCreate(((PetscObject) da)->comm, J);CHKERRQ(ierr);
    ierr = MatSetSizes(*J, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSHELL, &isShell);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATBAIJ, &isBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSBAIJ, &isSymBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock);CHKERRQ(ierr);
    /* Check for symmetric storage */
    isSymmetric = (PetscBool) (isSymBlock || isSymSeqBlock || isSymMPIBlock);
    if (isSymmetric) {
      ierr = MatSetOption(*J, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE);CHKERRQ(ierr);
    }
    if (!isShell) {
      /* PetscBool fillMatrix = (PetscBool) !da->prealloc_only; */
      PetscInt *dnz, *onz, *dnzu, *onzu, bsLocal;

      if (bs < 0) {
        if (isBlock || isSeqBlock || isMPIBlock || isSymBlock || isSymSeqBlock || isSymMPIBlock) {
          PetscInt pStart, pEnd, p, dof;

          ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
          for(p = pStart; p < pEnd; ++p) {
            ierr = PetscSectionGetDof(sectionGlobal, p, &dof);CHKERRQ(ierr);
            if (dof) {
              bs = dof;
              break;
            }
          }
        } else {
          bs = 1;
        }
        /* Must have same blocksize on all procs (some might have no points) */
        bsLocal = bs;
        ierr = MPI_Allreduce(&bsLocal, &bs, 1, MPIU_INT, MPI_MAX, ((PetscObject) da)->comm);CHKERRQ(ierr);
      }
      ierr = PetscMalloc4(localSize/bs, PetscInt, &dnz, localSize/bs, PetscInt, &onz, localSize/bs, PetscInt, &dnzu, localSize/bs, PetscInt, &onzu);CHKERRQ(ierr);
      ierr = PetscMemzero(dnz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(onz,  localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(dnzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(onzu, localSize/bs * sizeof(PetscInt));CHKERRQ(ierr);
      /* ierr = DMComplexPreallocateOperator(dm, bs, section, sectionGlobal, dnz, onz, dnzu, onzu, *J, fillMatrix);CHKERRQ(ierr); */
      ierr = PetscFree4(dnz, onz, dnzu, onzu);CHKERRQ(ierr);
    }
  }
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
  ierr = DMDAGetInfo(da,&dim,&M,&N,&P,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,0,0,0,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,dof*nx*ny*nz,dof*nx*ny*nz,dof*M*N*P,dof*M*N*P);CHKERRQ(ierr);
  ierr = MatSetType(A,(const MatType)ttype);CHKERRQ(ierr); 
  ierr = MatSetDM(A,da);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetType(A,&Atype);CHKERRQ(ierr);
  /*
     We do not provide a getmatrix function in the DMDA operations because 
   the basic DMDA does not know about matrices. We think of DMDA as being more 
   more low-level than matrices. This is kind of cheating but, cause sometimes 
   we think of DMDA has higher level than matrices.

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
    }
  }
  if (aij) {
    if (dim == 1) {
      ierr = DMCreateMatrix_DA_1d_MPIAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 2) {
      if (dd->ofill) {
        ierr = DMCreateMatrix_DA_2d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DMCreateMatrix_DA_2d_MPIAIJ(da,A);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      if (dd->ofill) {
        ierr = DMCreateMatrix_DA_3d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DMCreateMatrix_DA_3d_MPIAIJ(da,A);CHKERRQ(ierr);
      }
    }
  } else if (baij) {
    if (dim == 2) {
      ierr = DMCreateMatrix_DA_2d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DMCreateMatrix_DA_3d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else {
      SETERRQ3(((PetscObject)da)->comm,PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension!\n" \
	       "Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
    }
  } else if (sbaij) {
    if (dim == 2) {
      ierr = DMCreateMatrix_DA_2d_MPISBAIJ(da,A);CHKERRQ(ierr); 
    } else if (dim == 3) {
      ierr = DMCreateMatrix_DA_3d_MPISBAIJ(da,A);CHKERRQ(ierr);
    } else {
      SETERRQ3(((PetscObject)da)->comm,PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension!\n" \
	       "Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
    }
  } else {
    ISLocalToGlobalMapping ltog,ltogb;
    ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
    ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(A,ltog,ltog);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMappingBlock(A,ltogb,ltogb);CHKERRQ(ierr);
  }
  ierr = DMDAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(A,dim,dims,starts,dof);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"DM",(PetscObject)da);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) { 
    /* change viewer to display matrix in natural ordering */
    ierr = MatShellSetOperation(A, MATOP_VIEW, (void (*)(void)) MatView_MPI_DA);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_LOAD, (void (*)(void)) MatLoad_MPI_DA);CHKERRQ(ierr);
  }
  *J = A;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCreateMatrix_DA_2d_MPIAIJ" 
PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny,m,n,dim,s,*cols = PETSC_NULL,k,nc,*rows = PETSC_NULL,col,cnt,l,p;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by;
  ISLocalToGlobalMapping ltog,ltogb;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&bx,&by,0,&st);CHKERRQ(ierr);
  col = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,PetscInt,&rows,col*col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);
  
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    pend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      lend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

      cnt  = 0;
      for (k=0; k<nc; k++) {
	for (l=lstart; l<lend+1; l++) {
	  for (p=pstart; p<pend+1; p++) {
	    if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
	      cols[cnt++]  = k + nc*(slot + gnx*l + p);
	    }
	  }
	}
	rows[k] = k + nc*(slot);
      }
      ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
    
      pstart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      pend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      
      for (j=ys; j<ys+ny; j++) {
	slot = i - gxs + gnx*(j - gys);
      
	lstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	lend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1)); 

	cnt  = 0;
	for (k=0; k<nc; k++) {
	  for (l=lstart; l<lend+1; l++) {
	    for (p=pstart; p<pend+1; p++) {
	      if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
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
#define __FUNCT__ "DMCreateMatrix_DA_2d_MPIAIJ_Fill" 
PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt,l,p;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  DM_DA                  *dd = (DM_DA*)da->data;
  PetscInt               ifill_col,*ofill = dd->ofill, *dfill = dd->dfill;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by;
  ISLocalToGlobalMapping ltog,ltogb;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&bx,&by,0,&st);CHKERRQ(ierr);
  col = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);
  
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    pend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      lend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

      for (k=0; k<nc; k++) {
        cnt  = 0;
	for (l=lstart; l<lend+1; l++) {
	  for (p=pstart; p<pend+1; p++) {
            if (l || p) {
	      if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
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
        ierr = MatPreallocateSetLocal(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
    
      pstart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      pend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      
      for (j=ys; j<ys+ny; j++) {
	slot = i - gxs + gnx*(j - gys);
      
	lstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	lend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1)); 

	for (k=0; k<nc; k++) {
	  cnt  = 0;
	  for (l=lstart; l<lend+1; l++) {
	    for (p=pstart; p<pend+1; p++) {
	      if (l || p) {
		if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
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
#define __FUNCT__ "DMCreateMatrix_DA_3d_MPIAIJ" 
PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols = PETSC_NULL,k,nc,*rows = PETSC_NULL,col,cnt,l,p,*dnz = PETSC_NULL,*onz = PETSC_NULL;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by,bz;
  ISLocalToGlobalMapping ltog,ltogb;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,PetscInt,&rows,col*col*col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
	kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
	kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	
	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	cnt  = 0;
	for (l=0; l<nc; l++) {
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		  cols[cnt++]  = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
		}
	      }
	    }
	  }
	  rows[l] = l + nc*(slot);
	}
	ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
	  kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	  cnt  = 0;
	  for (l=0; l<nc; l++) {
	    for (ii=istart; ii<iend+1; ii++) {
	      for (jj=jstart; jj<jend+1; jj++) {
		for (kk=kstart; kk<kend+1; kk++) {
		  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
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
#define __FUNCT__ "DMCreateMatrix_DA_1d_MPIAIJ" 
PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,nx,i,i1,slot,gxs,gnx;           
  PetscInt               m,dim,s,*cols = PETSC_NULL,nc,*rows = PETSC_NULL,col,cnt,l;
  PetscInt               istart,iend;
  PetscScalar            *values;
  DMDABoundaryType       bx;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,0,0,0,0,0,&nc,&s,&bx,0,0,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,col*nc,0);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,col*nc,0,col*nc,0);CHKERRQ(ierr);
  ierr = PetscMalloc2(nc,PetscInt,&rows,col*nc*nc,PetscInt,&cols);CHKERRQ(ierr);
  
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);
  
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
#define __FUNCT__ "DMCreateMatrix_DA_2d_MPIBAIJ" 
PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&bx,&by,0,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      slot = i - gxs + gnx*(j - gys); 

      /* Find block columns in block row */
      cnt  = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
            cols[cnt++]  = slot + ii + gnx*jj;
          }
        }
      }
      ierr = MatPreallocateSetLocal(ltogb,1,&slot,ltogb,cnt,cols,dnz,onz);CHKERRQ(ierr);
    } 
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
	slot = i - gxs + gnx*(j - gys); 
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
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
#define __FUNCT__ "DMCreateMatrix_DA_3d_MPIBAIJ" 
PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by,bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col    = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr  = PetscMalloc(col*col*col*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
	kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
	kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

	/* Find block columns in block row */
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	}
	ierr = MatPreallocateSetLocal(ltogb,1,&slot,ltogb,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr  = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr  = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
	  kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	  cnt  = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
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
  PetscInt       i,n;

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
#define __FUNCT__ "DMCreateMatrix_DA_2d_MPISBAIJ" 
PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&bx,&by,0,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      slot = i - gxs + gnx*(j - gys);

      /* Find block columns in block row */
      cnt  = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) {
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

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        slot = i - gxs + gnx*(j - gys);

        /* Find block columns in block row */
        cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) {
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
#define __FUNCT__ "DMCreateMatrix_DA_3d_MPISBAIJ" 
PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by,bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  /*     
     nc - number of components per grid point 
     col - number of colors needed in one direction for single component problem 
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the matrix */
  ierr = PetscMalloc(col*col*col*sizeof(PetscInt),&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
	kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

	slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

	/* Find block columns in block row */
	cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
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

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
	jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
	  kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	  
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	  
	  cnt  = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
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
#define __FUNCT__ "DMCreateMatrix_DA_3d_MPIAIJ_Fill" 
PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt,l,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  DM_DA                  *dd = (DM_DA*)da->data;
  PetscInt               ifill_col,*dfill = dd->dfill,*ofill = dd->ofill;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMDABoundaryType       bx,by,bz;
  ISLocalToGlobalMapping ltog,ltogb;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col    = 2*s + 1;
  if (bx == DMDA_BOUNDARY_PERIODIC && (m % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (by == DMDA_BOUNDARY_PERIODIC && (n % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (bz == DMDA_BOUNDARY_PERIODIC && (p % col)){ 
    SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");
  }

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc(col*col*col*nc*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(da,&ltogb);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr); 


  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
      jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
        kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	
        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	
	for (l=0; l<nc; l++) {
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		if (ii || jj || kk) {
		  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
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
	  ierr = MatPreallocateSetLocal(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr); 
	}
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr); 
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(J,ltogb,ltogb);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc(col*col*col*nc*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
	jstart = (by == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j)); 
	jend   = (by == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
	for (k=zs; k<zs+nz; k++) {
	  kstart = (bz == DMDA_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k)); 
	  kend   = (bz == DMDA_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));
	  
	  slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);
	  
	  for (l=0; l<nc; l++) {
	    cnt  = 0;
	    for (ii=istart; ii<iend+1; ii++) {
	      for (jj=jstart; jj<jend+1; jj++) {
		for (kk=kstart; kk<kend+1; kk++) {
		  if (ii || jj || kk) {
		    if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
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
