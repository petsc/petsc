

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdda.c,v 1.12 1997/08/13 22:26:44 bsmith Exp bsmith $";
#endif
 
#include "da.h"     /*I      "da.h"     I*/
#include "mat.h"    /*I      "mat.h"    I*/

#undef __FUNC__  
#define __FUNC__ "DAGetColoring2dBox" 
/*@C
      DAGetColoring2dBox - Gets the coloring required for computing the Jacobian via
          finite differences on a function defined using the nine point stencil
          on a two dimensional grid. 

          This is a utility routine that will change over time, not part of the 
          core PETSc package.

  Input Parameter:
.    da - the distributed array

  Output Parameters:
.    coloring - matrix coloring for compute Jacobians
.    J  - matrix with the correct nonzero structured 
            (obviously without the correct Jacobian values)

.seealso: DAGetColoring2dBox()

@*/
int DAGetColoring2dBox(DA da,ISColoring *coloring,Mat *J)
{
  int      ierr, xs,ys,nx,ny,*colors,i,j,ii,slot,gxs,gys,ys1,ny1,nx1,gnx,gny;           
  int      m,n,dim,w,s,N,*indices,*gindices,k,xs1,nc,ng,*cols;
  MPI_Comm comm;
  Scalar   *values;
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
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,0); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic boundary conditions */
  nc     = w;
  values = (Scalar *) PetscMalloc( 9*nc*nc*sizeof(Scalar) ); CHKPTRQ(values);
  PetscMemzero(values,9*nc*nc*sizeof(Scalar));
  cols    = (int *) PetscMalloc( nc*sizeof(int) ); CHKPTRQ(cols);
  indices = (int *) PetscMalloc( 9*nc*nc*sizeof(int) ); CHKPTRQ(cols);
  N       = m*n;

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*ny*sizeof(int) ); CHKPTRQ(colors);
  ii = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      for ( k=0; k<nc; k++ ) {
        colors[ii++] = k + nc*((i % 3) + 3*(j % 3));
      }
    }
  }
  ierr = ISColoringCreate(comm,nc*nx*ny,colors,coloring); CHKERRQ(ierr);
  PetscFree(colors);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,9*nc,0,0,0,J);CHKERRQ(ierr);  

  ierr = DAGetGlobalIndices(da,&ng,&gindices);
  {
    ISLocalToGlobalMapping ltog;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,ng,gindices,&ltog);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(*J,ltog); CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(ltog); CHKERRQ(ierr);
  }

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
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);  
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
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);   
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
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
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
      ierr = MatSetValuesLocal(*J,nc,cols,9*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  PetscFree(values); 
  PetscFree(cols);
  PetscFree(indices);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  return 0;
}


