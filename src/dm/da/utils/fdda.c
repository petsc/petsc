/*$Id: fdda.c,v 1.75 2001/08/07 21:31:51 bsmith Exp $*/
 
#include "src/dm/da/daimpl.h" /*I      "petscda.h"     I*/
#include "petscmat.h"         /*I      "petscmat.h"    I*/


EXTERN int DAGetColoring1d_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN int DAGetColoring2d_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN int DAGetColoring2d_5pt_MPIAIJ(DA,ISColoringType,ISColoring *);
EXTERN int DAGetColoring3d_MPIAIJ(DA,ISColoringType,ISColoring *);

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring" 
/*@C
    DAGetColoring - Gets the coloring required for computing the Jacobian via
    finite differences on a function defined using a stencil on the DA.

    Collective on DA

    Input Parameter:
+   da - the distributed array
-   ctype - IS_COLORING_LOCAL or IS_COLORING_GHOSTED

    Output Parameters:
.   coloring - matrix coloring for use in computing Jacobians (or PETSC_NULL if not needed)

    Level: advanced

    Notes: These compute the graph coloring of the graph of A^{T}A. The coloring used 
   for efficient (parallel or thread based) triangular solves etc is NOT yet 
   available. 


.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), ISColoringType, ISColoring

@*/
int DAGetColoring(DA da,ISColoringType ctype,ISColoring *coloring)
{
  int        ierr,dim;

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
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

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
      SETERRQ1(1,"Not done for %d dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring2d_MPIAIJ" 
int DAGetColoring2d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  int                    ierr,xs,ys,nx,ny,*colors,i,j,ii,gxs,gys,gnx,gny;           
  int                    m,n,M,N,dim,w,s,k,nc,col,size;
  MPI_Comm               comm;
  DAPeriodicType         wrap;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,&M,&N,0,&w,&s,&wrap,&st);CHKERRQ(ierr);
  nc     = w;
  col    = 2*s + 1;
  if (DAXPeriodic(wrap) && (m % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAYPeriodic(wrap) && (n % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create the coloring */
  if (st == DA_STENCIL_STAR && s == 1 && !DAXPeriodic(wrap) && !DAYPeriodic(wrap)) {
    ierr = DAGetColoring2d_5pt_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!da->localcoloring) {
    ierr = PetscMalloc(nc*nx*ny*sizeof(int),&colors);CHKERRQ(ierr);
    ii = 0;
    for (j=ys; j<ys+ny; j++) {
        for (i=xs; i<xs+nx; i++) {
          for (k=0; k<nc; k++) {
            colors[ii++] = k + nc*((i % col) + col*(j % col));
          }
        }
      }
      ierr = ISColoringCreate(comm,nc*nx*ny,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*gny*sizeof(int),&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=gys; j<gys+gny; j++) {
        for (i=gxs; i<gxs+gnx; i++) {
          for (k=0; k<nc; k++) {
            /* the complicated stuff is to handle periodic boundaries */
            colors[ii++] = k + nc*(    (((i < 0) ? m+i:((i >= m) ? i-m:i)) % col) +
  			           col*(((j < 0) ? n+j:((j >= n) ? j-n:j)) % col));
          }
        }
      }
      ierr = ISColoringCreate(comm,nc*gnx*gny,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(1,"Unknown ISColoringType %d",ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring3d_MPIAIJ" 
int DAGetColoring3d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  int                    ierr,xs,ys,nx,ny,*colors,i,j,gxs,gys,gnx,gny;           
  int                    m,n,p,dim,s,k,nc,col,size,zs,gzs,ii,l,nz,gnz,M,N,P;
  MPI_Comm               comm;
  DAPeriodicType         wrap;
  DAStencilType          st;

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
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_LOCAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*ny*nz*sizeof(int),&colors);CHKERRQ(ierr);
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
      ierr = ISColoringCreate(comm,nc*nx*ny*nz,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*gny*gnz*sizeof(int),&colors);CHKERRQ(ierr);
      ii = 0;
      for (k=gzs; k<gzs+gnz; k++) {
        for (j=gys; j<gys+gny; j++) {
          for (i=gxs; i<gxs+gnx; i++) {
            for (l=0; l<nc; l++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = l + nc*(        (((i < 0) ? m+i:((i >= m) ? i-m:i)) % col) +
  			                 col*(((j < 0) ? n+j:((j >= n) ? j-n:j)) % col) +
                                     col*col*(((k < 0) ? p+k:((k >= p) ? k-p:k)) % col));
            }
          }
        }
      }
      ierr = ISColoringCreate(comm,nc*gnx*gny*gnz,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(1,"Unknown ISColoringType %d",ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring1d_MPIAIJ" 
int DAGetColoring1d_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  int                    ierr,xs,nx,*colors,i,i1,gxs,gnx,l;           
  int                    m,M,dim,s,nc,col,size;
  MPI_Comm               comm;
  DAPeriodicType         wrap;

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
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_LOCAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*sizeof(int),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=xs; i<xs+nx; i++) {
        for (l=0; l<nc; l++) {
          colors[i1++] = l + nc*(i % col);
        }
      }
      ierr = ISColoringCreate(comm,nc*nx,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*sizeof(int),&colors);CHKERRQ(ierr);
      i1 = 0;
      for (i=gxs; i<gxs+gnx; i++) {
        for (l=0; l<nc; l++) {
          /* the complicated stuff is to handle periodic boundaries */
          colors[i1++] = l + nc*((((i < 0) ? m+i:((i >= m) ? i-m:i)) % col));
        }
      }
      ierr = ISColoringCreate(comm,nc*gnx,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(1,"Unknown ISColoringType %d",ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetColoring2d_5pt_MPIAIJ" 
int DAGetColoring2d_5pt_MPIAIJ(DA da,ISColoringType ctype,ISColoring *coloring)
{
  int      ierr,xs,ys,nx,ny,*colors,i,j,ii,gxs,gys,gnx,gny;           
  int      m,n,dim,w,s,k,nc;
  MPI_Comm comm;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr   = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,0,0);CHKERRQ(ierr);
  nc     = w;
  ierr   = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr   = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr   = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_LOCAL) {
    if (!da->localcoloring) {
      ierr = PetscMalloc(nc*nx*ny*sizeof(int),&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=ys; j<ys+ny; j++) {
	for (i=xs; i<xs+nx; i++) {
	  for (k=0; k<nc; k++) {
	    colors[ii++] = k + nc*((3*j+i) % 5);
	  }
	}
      }
      ierr = ISColoringCreate(comm,nc*nx*ny,colors,&da->localcoloring);CHKERRQ(ierr);
    }
    *coloring = da->localcoloring;
  } else if (ctype == IS_COLORING_GHOSTED) {
    if (!da->ghostedcoloring) {
      ierr = PetscMalloc(nc*gnx*gny*sizeof(int),&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=gys; j<gys+gny; j++) {
	for (i=gxs; i<gxs+gnx; i++) {
	  for (k=0; k<nc; k++) {
	    colors[ii++] = k + nc*((3*j+i) % 5);
	  }
	}
      }
      ierr = ISColoringCreate(comm,nc*gnx*gny,colors,&da->ghostedcoloring);CHKERRQ(ierr);
      ierr = ISColoringSetType(da->ghostedcoloring,IS_COLORING_GHOSTED);CHKERRQ(ierr);
    }
    *coloring = da->ghostedcoloring;
  } else SETERRQ1(1,"Unknown ISColoringType %d",ctype);
  PetscFunctionReturn(0);
}

/* =========================================================================== */
EXTERN int DAGetMatrix1d_MPIAIJ(DA,Mat *);
EXTERN int DAGetMatrix2d_MPIAIJ(DA,Mat *);
EXTERN int DAGetMatrix3d_MPIAIJ(DA,Mat *);
EXTERN int DAGetMatrix3d_MPIBAIJ(DA,Mat *);
EXTERN int DAGetMatrix3d_MPISBAIJ(DA,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix" 
/*@C
    DAGetMatrix - Creates a matrix with the correct parallel layout and nonzero structure required for 
      computing the Jacobian on a function defined using the stencil set in the DA.

    Collective on DA

    Input Parameter:
+   da - the distributed array
-   mtype - either MATMPIAIJ or MATMPIBAIJ

    Output Parameters:
.   J  - matrix with the correct nonzero structure
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate()

@*/
int DAGetMatrix(DA da,MatType mtype,Mat *J)
{
  int        ierr,dim;
  PetscTruth aij,baij,sbaij;

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
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  /*
     We do not provide a getcoloring function in the DA operations because 
   the basic DA does not know about matrices. We think of DA as being more 
   more low-level then matrices.
  */
  ierr = PetscStrcmp(MATMPIAIJ,mtype,&aij);CHKERRQ(ierr);
  ierr = PetscStrcmp(MATMPIBAIJ,mtype,&baij);CHKERRQ(ierr);
  ierr = PetscStrcmp(MATMPISBAIJ,mtype,&sbaij);CHKERRQ(ierr);
  if (aij) {
    if (dim == 1) {
      ierr = DAGetMatrix1d_MPIAIJ(da,J);CHKERRQ(ierr);
    } else if (dim == 2) {
      ierr =  DAGetMatrix2d_MPIAIJ(da,J);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr =  DAGetMatrix3d_MPIAIJ(da,J);CHKERRQ(ierr);
    }
  } else if(baij) {
    if (dim == 3) {
      ierr =  DAGetMatrix3d_MPIBAIJ(da,J);CHKERRQ(ierr);
    } else {
      SETERRQ1(1,"Not done for %d dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
    }
  } else if(sbaij) {
    if (dim == 3) {
      ierr =  DAGetMatrix3d_MPISBAIJ(da,J);CHKERRQ(ierr);
    } else {
      SETERRQ1(1,"Not done for %d dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
    }
  } else {
    SETERRQ1(1,"Not done for %% matrix type, send us mail petsc-maint@mcs.anl.gov for code",mtype);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix2d_MPIAIJ" 
int DAGetMatrix2d_MPIAIJ(DA da,Mat *J)
{
  int                    ierr,xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,k,nc,*rows,col,cnt,l,p;
  int                    lstart,lend,pstart,pend,*dnz,*onz,size;
  int                    dims[2],starts[2];
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  col = 2*s + 1;
  if (DAXPeriodic(wrap) && (m % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  if (DAYPeriodic(wrap) && (n % col)){ 
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  }
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreate(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,J);CHKERRQ(ierr);  

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(int),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(col*col*nc*nc*sizeof(int),&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  
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
  /* set matrix type and preallocation information */
  if (size > 1) {
    ierr = MatSetType(*J,MATMPIAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*J,MATSEQAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);  
  ierr = MatSeqBAIJSetPreallocation(*J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(*J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&starts[0],&starts[1],PETSC_IGNORE,&dims[0],&dims[1],PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatSetStencil(*J,2,dims,starts,nc);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
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
      ierr = MatSetValuesLocal(*J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPIAIJ" 
int DAGetMatrix3d_MPIAIJ(DA da,Mat *J)
{
  int                    ierr,xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,k,nc,*rows,col,cnt,l,p,*dnz,*onz;
  int                    istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk,size;
  int                    dims[3],starts[3];
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;
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
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);


  /* create the matrix */
  /* create empty Jacobian matrix */
  ierr = MatCreate(comm,nc*nx*ny*nz,nc*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE,J);CHKERRQ(ierr);  
  ierr = PetscMalloc(col*col*col*nc*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(int),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(col*col*col*nc*sizeof(int),&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

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
  /* set matrix type and preallocation */
  if (size > 1) {
    ierr = MatSetType(*J,MATMPIAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*J,MATSEQAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);  
  ierr = MatSeqBAIJSetPreallocation(*J,nc,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatMPIBAIJSetPreallocation(*J,nc,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(*J,3,dims,starts,nc);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
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
	ierr = MatSetValuesLocal(*J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix1d_MPIAIJ" 
int DAGetMatrix1d_MPIAIJ(DA da,Mat *J)
{
  int                    ierr,xs,nx,i,i1,slot,gxs,gnx;           
  int                    m,dim,s,*cols,nc,*rows,col,cnt,l;
  int                    istart,iend,size;
  int                    dims[1],starts[1];
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,0,0,0,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  if (DAXPeriodic(wrap) && (m % col)) {
    SETERRQ(PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points is divisible\n\
                 by 2*stencil_width + 1\n");
  }


  ierr = DAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  
  ierr    = MatCreate(comm,nc*nx,nc*nx,PETSC_DECIDE,PETSC_DECIDE,J);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*J,MATMPIAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*J,MATSEQAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(*J,col*nc,0);CHKERRQ(ierr);  
  ierr = MatSeqBAIJSetPreallocation(*J,nc,col,0);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(*J,col*nc,0,0,0);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,nc,col,0,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&starts[0],PETSC_IGNORE,PETSC_IGNORE,&dims[0],PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatSetStencil(*J,1,dims,starts,nc);CHKERRQ(ierr);
  
  ierr = PetscMalloc(col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr = PetscMemzero(values,col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(int),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(col*nc*sizeof(int),&cols);CHKERRQ(ierr);
  
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);
  
  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
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
    ierr = MatSetValuesLocal(*J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPIBAIJ" 
int DAGetMatrix3d_MPIBAIJ(DA da,Mat *J)
{
  int                    ierr,xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  int                    istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  if (wrap != DA_NONPERIODIC) SETERRQ(PETSC_ERR_SUP,"Currently no support for periodic");
  col    = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the matrix */
  ierr  = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr  = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr  = PetscMalloc(col*col*col*sizeof(int),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltog);CHKERRQ(ierr);

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
	if (st == DA_STENCIL_BOX) {   /* if using BOX stencil */
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	} else {  /* Star stencil */
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    if (ii) {
	      /* jj and kk must be zero */
	      /* cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk; */
	      cols[cnt++]  = slot + ii;
	    } else {
	      for (jj=jstart; jj<jend+1; jj++) {
		if (jj) {
                  /* ii and kk must be zero */
		  cols[cnt++]  = slot + gnx*jj;
		} else {
		  /* ii and jj must be zero */
		  for (kk=kstart; kk<kend+1; kk++) {
		    cols[cnt++]  = slot + gnx*gny*kk;
		  }
		}
	      }
	    }
	  }
	}
	ierr = MatPreallocateSetLocal(ltog,1,&slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIBAIJ(comm,nc,nc*nx*ny*nz,nc*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE,0,dnz,0,onz,J);CHKERRQ(ierr);

  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */

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
	if (st == DA_STENCIL_BOX) {   /* if using BOX stencil */
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	} else {  /* Star stencil */
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    if (ii) {
	      /* jj and kk must be zero */
	      cols[cnt++]  = slot + ii;
	    } else {
	      for (jj=jstart; jj<jend+1; jj++) {
		if (jj) {
                  /* ii and kk must be zero */
		  cols[cnt++]  = slot + gnx*jj;
		} else {
		  /* ii and jj must be zero */
		  for (kk=kstart; kk<kend+1; kk++) {
		    cols[cnt++]  = slot + gnx*gny*kk;
		  }
                }
              }
            }
	  }
	}
	ierr = MatSetValuesBlockedLocal(*J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/* BAD! Almost identical to the BAIJ one */
#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix3d_MPISBAIJ" 
int DAGetMatrix3d_MPISBAIJ(DA da,Mat *J)
{
  int                    ierr,xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  int                    istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DAPeriodicType         wrap;
  DAStencilType          st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,&st);CHKERRQ(ierr);
  if (wrap != DA_NONPERIODIC) SETERRQ(PETSC_ERR_SUP,"Currently no support for periodic");
  col    = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the matrix */
  ierr  = PetscMalloc(col*col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr  = PetscMemzero(values,col*col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr  = PetscMalloc(col*col*col*sizeof(int),&cols);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltog);CHKERRQ(ierr);

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
	if (st == DA_STENCIL_BOX) {   /* if using BOX stencil */
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	} else {  /* Star stencil */
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    if (ii) {
	      /* jj and kk must be zero */
	      /* cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk; */
	      cols[cnt++]  = slot + ii;
	    } else {
	      for (jj=jstart; jj<jend+1; jj++) {
		if (jj) {
                  /* ii and kk must be zero */
		  cols[cnt++]  = slot + gnx*jj;
		} else {
		  /* ii and jj must be zero */
		  for (kk=kstart; kk<kend+1; kk++) {
		    cols[cnt++]  = slot + gnx*gny*kk;
		  }
		}
	      }
	    }
	  }
	}
	ierr = MatPreallocateSymmetricSetLocal(ltog,1,&slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }

  /* create empty Jacobian matrix */
  ierr = MatCreateMPISBAIJ(comm,nc,nc*nx*ny*nz,nc*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE,0,dnz,0,onz,J);CHKERRQ(ierr);

  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */

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
	if (st == DA_STENCIL_BOX) {   /* if using BOX stencil */
	  for (ii=istart; ii<iend+1; ii++) {
	    for (jj=jstart; jj<jend+1; jj++) {
	      for (kk=kstart; kk<kend+1; kk++) {
		cols[cnt++]  = slot + ii + gnx*jj + gnx*gny*kk;
	      }
	    }
	  }
	} else {  /* Star stencil */
	  cnt  = 0;
	  for (ii=istart; ii<iend+1; ii++) {
	    if (ii) {
	      /* jj and kk must be zero */
	      cols[cnt++]  = slot + ii;
	    } else {
	      for (jj=jstart; jj<jend+1; jj++) {
		if (jj) {
                  /* ii and kk must be zero */
		  cols[cnt++]  = slot + gnx*jj;
		} else {
		  /* ii and jj must be zero */
		  for (kk=kstart; kk<kend+1; kk++) {
		    cols[cnt++]  = slot + gnx*gny*kk;
		  }
                }
              }
            }
	  }
	}
	ierr = MatSetValuesBlockedLocal(*J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}
