#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdda.c,v 1.39 1999/06/30 23:55:25 balay Exp bsmith $";
#endif
 
#include "da.h"     /*I      "da.h"     I*/
#include "mat.h"    /*I      "mat.h"    I*/
#include "src/dm/da/daimpl.h" 

extern int DAGetColoring1d(DA,ISColoring *,Mat *);
extern int DAGetColoring2d_1(DA,ISColoring *,Mat *);
extern int DAGetColoring2d(DA,ISColoring *,Mat *);
extern int DAGetColoring3d(DA,ISColoring *,Mat *);

#undef __FUNC__  
#define __FUNC__ "DAGetColoring" 
/*@C
    DAGetColoring - Gets the coloring required for computing the Jacobian via
    finite differences on a function defined using a stencil on the DA.

    Collective on DA

    Input Parameter:
.   da - the distributed array

    Output Parameters:
+   coloring - matrix coloring for use in computing Jacobians
-   J  - matrix with the correct nonzero structure
        (obviously without the correct Jacobian values)

    Level: advanced

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate()

@*/
int DAGetColoring(DA da,ISColoring *coloring,Mat *J)
{
  int            ierr,dim;

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
    ierr = DAGetColoring1d(da,coloring,J);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr =  DAGetColoring2d(da,coloring,J);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr =  DAGetColoring3d(da,coloring,J);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "DAGetColoring2d" 
int DAGetColoring2d(DA da,ISColoring *coloring,Mat *J)
{
  int                    ierr, xs,ys,nx,ny,*colors,i,j,ii,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,w,s,*cols,k,nc,*rows,col,cnt,l,p;
  int                    lstart,lend,pstart,pend;
  MPI_Comm               comm;
  Scalar                 *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,&wrap,0);CHKERRQ(ierr);
  if (wrap != DA_NONPERIODIC) SETERRQ(PETSC_ERR_SUP,0,"Currently no support for periodic");

  nc     = w;
  col    = 2*s + 1;

  /*
      Faster code for stencil width of 1 
  */
  if (s == 1) {
    ierr = DAGetColoring2d_1(da,coloring,J);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  values  = (Scalar *) PetscMalloc( col*col*nc*nc*sizeof(Scalar) );CHKPTRQ(values);
  ierr    = PetscMemzero(values,col*col*nc*nc*sizeof(Scalar));CHKERRQ(ierr);
  rows    = (int *) PetscMalloc( nc*sizeof(int) );CHKPTRQ(rows);
  cols    = (int *) PetscMalloc( col*col*nc*nc*sizeof(int) );CHKPTRQ(cols);

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*ny*sizeof(int) );CHKPTRQ(colors);
  ii = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      for ( k=0; k<nc; k++ ) {
        colors[ii++] = k + nc*((i % col) + col*(j % col));
      }
    }
  }
  ierr = ISColoringCreate(comm,nc*nx*ny,colors,coloring);CHKERRQ(ierr);
  ierr = PetscFree(colors);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,col*col*nc,0,0,0,J);CHKERRQ(ierr);  

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);

  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  for ( i=xs; i<xs+nx; i++ ) {

    pstart = PetscMax(-s,-i);
    pend   = PetscMin(s,m-i-1);

    for ( j=ys; j<ys+ny; j++ ) {
      slot = i - gxs + gnx*(j - gys);

      lstart = PetscMax(-s,-j); 
      lend   = PetscMin(s,n-j-1);

      cnt  = 0;
      for ( k=0; k<nc; k++ ) {
        for ( l=lstart; l<lend+1; l++ ) {
          for ( p=pstart; p<pend+1; p++ ) {
            cols[cnt++]  = k + nc*(slot + gnx*l + p);
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

#undef __FUNC__  
#define __FUNC__ "DAGetColoring3d" 
int DAGetColoring3d(DA da,ISColoring *coloring,Mat *J)
{
  int                    ierr, xs,ys,nx,ny,*colors,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,k,nc,*rows,col,cnt,l,p;
  int                    istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  Scalar                 *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  if (wrap != DA_NONPERIODIC) SETERRQ(PETSC_ERR_SUP,0,"Currently no support for periodic");
  col    = 2*s + 1;

  values  = (Scalar *) PetscMalloc( col*col*col*nc*nc*nc*sizeof(Scalar) );CHKPTRQ(values);
  ierr    = PetscMemzero(values,col*col*col*nc*nc*nc*sizeof(Scalar));CHKERRQ(ierr);
  rows    = (int *) PetscMalloc( nc*sizeof(int) );CHKPTRQ(rows);
  cols    = (int *) PetscMalloc( col*col*col*nc*sizeof(int) );CHKPTRQ(cols);

  ierr = DAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*ny*nz*sizeof(int) );CHKPTRQ(colors);
  ii = 0;
  for ( k=zs; k<zs+nz; k++ ) {
    for ( j=ys; j<ys+ny; j++ ) {
      for ( i=xs; i<xs+nx; i++ ) {
        for ( l=0; l<nc; l++ ) {
          colors[ii++] = l + nc*((i % col) + col*(j % col) + col*col*(k % col));
        }
      }
    }
  }
  ierr = ISColoringCreate(comm,nc*nx*ny*nz,colors,coloring);CHKERRQ(ierr);
  ierr = PetscFree(colors);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx*ny*nz,nc*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE,col*col*col*nc,0,0,0,J);CHKERRQ(ierr);  

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);

  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  for ( i=xs; i<xs+nx; i++ ) {
    istart = PetscMax(-s,-i);
    iend   = PetscMin(s,m-i-1);
    for ( j=ys; j<ys+ny; j++ ) {
      jstart = PetscMax(-s,-j); 
      jend   = PetscMin(s,n-j-1);
      for ( k=zs; k<zs+nz; k++ ) {
        kstart = PetscMax(-s,-k); 
        kend   = PetscMin(s,p-k-1);

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        cnt  = 0;
        for ( l=0; l<nc; l++ ) {
          for ( ii=istart; ii<iend+1; ii++ ) {
            for ( jj=jstart; jj<jend+1; jj++ ) {
              for ( kk=kstart; kk<kend+1; kk++ ) {
                cols[cnt++]  = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
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

/*  ------------------------------------------------------------------------------------*/
/*
      Special optimized code with a stencil width of 1

      The difference is that we don't have max and min inside the loops
*/
#undef __FUNC__  
#define __FUNC__ "DAGetColoring2d_1" 
int DAGetColoring2d_1(DA da,ISColoring *coloring,Mat *J)
{
  int                    ierr, xs,ys,nx,ny,*colors,i,j,i1,slot,gxs,gys,ys1,ny1;
  int                    m,n,dim,w,s,*indices,k,xs1,nc,*cols;
  int                    nx1,gnx,gny;           
  MPI_Comm               comm;
  Scalar                 *values;
  ISLocalToGlobalMapping ltog;

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
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,0,0);CHKERRQ(ierr);

  nc     = w;
  values = (Scalar *) PetscMalloc( 9*nc*nc*sizeof(Scalar) );CHKPTRQ(values);
  ierr   = PetscMemzero(values,9*nc*nc*sizeof(Scalar));CHKERRQ(ierr);
  cols    = (int *) PetscMalloc( nc*sizeof(int) );CHKPTRQ(cols);
  indices = (int *) PetscMalloc( 9*nc*nc*sizeof(int) );CHKPTRQ(cols);

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*ny*sizeof(int) );CHKPTRQ(colors);
  i1 = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      for ( k=0; k<nc; k++ ) {
	/*        if (da->stencil_type == DA_STENCIL_STAR) {
          colors[i1++] = k + nc*((i + 3*j)) % 5);
        } else { */
          colors[i1++] = k + nc*((i % 3) + 3*(j % 3));
	  /*        } */
      }
    }
  }
  ierr = ISColoringCreate(comm,nc*nx*ny,colors,coloring);CHKERRQ(ierr);
  ierr = PetscFree(colors);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,9*nc,0,0,0,J);CHKERRQ(ierr);  

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);

  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  
  /* fill up Jacobian for left edge */
  if (xs == 0) {
    ys1 = ys;
    ny1 = ny;
    /* lower left corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot);         
        indices[1*nc+k] = k + nc*(slot + 1); 
        indices[2*nc+k] = k + nc*(slot + gnx);  
        indices[3*nc+k] = k + nc*(slot + gnx + 1);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* upper left corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - gnx);  
        indices[1*nc+k] = k + nc*(slot - gnx + 1);
        indices[2*nc+k] = k + nc*(slot);         
        indices[3*nc+k] = k + nc*(slot + 1);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - gnx);  
        indices[1*nc+k] = k + nc*(slot - gnx + 1);
        indices[2*nc+k] = k + nc*(slot);         
        indices[3*nc+k] = k + nc*(slot + 1);
        indices[4*nc+k] = k + nc*(slot + gnx);   
        indices[5*nc+k] = k + nc*(slot + gnx + 1);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);  
    }
  }      

  /* fill up Jacobian for right edge */
  if (xs + nx == m) {
    ys1 = ys;
    ny1 = ny;
    /* lower right corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - 1);        
        indices[1*nc+k] = k + nc*(slot); 
        indices[2*nc+k] = k + nc*(slot + gnx - 1);  
        indices[3*nc+k] = k + nc*(slot + gnx);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* upper right corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - gnx - 1);  
        indices[1*nc+k] = k + nc*(slot - gnx);
        indices[2*nc+k] = k + nc*(slot-1);         
        indices[3*nc+k] = k + nc*(slot);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs + (nx - 1) - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - gnx - 1);  
        indices[1*nc+k] = k + nc*(slot - gnx);
        indices[2*nc+k] = k + nc*(slot - 1);       
        indices[3*nc+k] = k + nc*(slot);
        indices[4*nc+k] = k + nc*(slot + gnx - 1); 
        indices[5*nc+k] = k + nc*(slot + gnx);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);   
    }
  }   

  /* fill up Jacobian for bottom */
  if (ys == 0) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = k + nc*(slot - 1);      
        indices[1*nc+k] = k + nc*(slot);       
        indices[2*nc+k] = k + nc*(slot + 1);
        indices[3*nc+k] = k + nc*(slot + gnx - 1); 
        indices[4*nc+k] = k + nc*(slot + gnx);
        indices[5*nc+k] = k + nc*(slot + gnx + 1);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for top */
  if (ys + ny == n) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {    
        indices[0+k]    = k + nc*(slot - gnx - 1);
        indices[1*nc+k] = k + nc*(slot - gnx);  
        indices[2*nc+k] = k + nc*(slot - gnx + 1);
        indices[3*nc+k] = k + nc*(slot - 1);      
        indices[4*nc+k] = k + nc*(slot);       
        indices[5*nc+k] = k + nc*(slot  + 1);
        cols[k]         = k + nc*(slot);
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for interior of grid */
  nx1 = nx;
  ny1 = ny;
  if (xs == 0)    {xs++; nx1--;}
  if (ys == 0)    {ys++; ny1--;}
  if (xs+nx1 == m) {nx1--;}
  if (ys+ny1 == n) {ny1--;}
  for ( i=xs; i<xs+nx1; i++ ) {
    for ( j=ys; j<ys+ny1; j++ ) {
      slot = i - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {    
        indices[0+k]    = k + nc*(slot - gnx - 1);
        indices[1*nc+k] = k + nc*(slot - gnx);
        indices[2*nc+k] = k + nc*(slot - gnx + 1);
        indices[3*nc+k] = k + nc*(slot - 1);  
        indices[4*nc+k] = k + nc*(slot);      
        indices[5*nc+k] = k + nc*(slot + 1);
        indices[6*nc+k] = k + nc*(slot + gnx - 1);
        indices[7*nc+k] = k + nc*(slot + gnx); 
        indices[8*nc+k] = k + nc*(slot + gnx + 1);
        cols[k]         = k + nc*(slot);
    }
      ierr = MatSetValuesLocal(*J,nc,cols,9*nc,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "DAGetColoring1d" 
int DAGetColoring1d(DA da,ISColoring *coloring,Mat *J)
{
  int                    ierr, xs,nx,*colors,i,i1,slot,gxs,gnx;           
  int                    m,dim,s,*cols,nc,*rows,col,cnt,l;
  int                    istart,iend;
  MPI_Comm               comm;
  Scalar                 *values;
  DAPeriodicType         wrap;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,0,0,0,0,0,&nc,&s,&wrap,0);CHKERRQ(ierr);
  col    = 2*s + 1;

  if (wrap && (m % col)) {
    SETERRQ(PETSC_ERR_SUP,1,"For coloring efficiency ensure number of grid points is divisible\n\
                 by 2*stencil_width + 1\n");
  }


  values  = (Scalar *) PetscMalloc( col*nc*nc*sizeof(Scalar) );CHKPTRQ(values);
  ierr    = PetscMemzero(values,col*nc*nc*sizeof(Scalar));CHKERRQ(ierr);
  rows    = (int *) PetscMalloc( nc*sizeof(int) );CHKPTRQ(rows);
  cols    = (int *) PetscMalloc( col*nc*sizeof(int) );CHKPTRQ(cols);

  ierr = DAGetCorners(da,&xs,0,0,&nx,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,0,0,&gnx,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*sizeof(int) );CHKPTRQ(colors);
  i1 = 0;
  for ( i=xs; i<xs+nx; i++ ) {
    for ( l=0; l<nc; l++ ) {
      colors[i1++] = l + nc*(i % col);
    }
  }
  ierr = ISColoringCreate(comm,nc*nx,colors,coloring);CHKERRQ(ierr);
  ierr = PetscFree(colors);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx,nc*nx,PETSC_DECIDE,PETSC_DECIDE,col*nc,0,0,0,J);CHKERRQ(ierr);

  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);

  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  for ( i=xs; i<xs+nx; i++ ) {
    istart = PetscMax(-s,gxs - i);
    iend   = PetscMin(s,gxs + gnx - i - 1);
    slot   = i - gxs;

    cnt  = 0;
    for ( l=0; l<nc; l++ ) {
      for ( i1=istart; i1<iend+1; i1++ ) {
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

