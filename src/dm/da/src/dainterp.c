/*$Id: dainterp.c,v 1.1 1999/11/09 01:02:28 bsmith Exp bsmith $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGetInterpolation_2D"
int DAGetInterpolation_2D(DA dac, DA daf, Mat *A)
{
  int      ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx;
  int      m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c;
  int      row,col,i_start_ghost,j_start_ghost,cols[4],mx, m_c,my, nc,ratio;
  int      i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  Scalar   v[4],x,y, one = 1.0;
  Mat      mat;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ratio = mx/Mx;
  if (ratio != my/My) SETERRQ(1,1,"Grid spacing ratio must be same in X and Y direction");

  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx); CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c); CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,m_f*n_f,m_c*n_c,Mx*My,mx*my,5,0,3,0,&mat);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for ( j=j_start; j<j_start+n_f; j++ ) {
    for ( i=i_start; i<i_start+m_f; i++ ) {
      /* convert to local "natural" numbering and 
         then to PETSc global numbering */
      row    = idx[m_ghost*(j-j_start_ghost) + (i-i_start_ghost)];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratio);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      x  = ((double)(i - i_c*ratio))/((double)ratio);
      y  = ((double)(j - j_c*ratio))/((double)ratio);
      /* printf("i j %d %d %g %g\n",i,j,x,y); */
      nc = 0;
      /* one left and below; or we are right on it */
      if (j_c < j_start_ghost_c || j_c > j_start_ghost_c+n_ghost_c) {
        SETERRQ3(1,1,"Sorry j %d %d %d",j_c,j_start_ghost_c,j_start_ghost_c+n_ghost_c);
      }
      if (i_c < i_start_ghost_c || i_c > i_start_ghost_c+m_ghost_c) {
        SETERRQ3(1,1,"Sorry i %d %d %d",i_c,i_start_ghost_c,i_start_ghost_c+m_ghost_c);
      }
      col      = m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c);
      cols[nc] = idx_c[col]; 
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratio != i) { 
        cols[nc] = idx_c[col+1];
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratio != j) { 
        cols[nc] = idx_c[col+m_ghost_c];
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratio != j && i_c*ratio != i) { 
        cols[nc] = idx_c[col+m_ghost_c+1];
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES); CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  *A = mat;
  PLogFlops(13*m_f*n_f);
  PetscFunctionReturn(0);;
}

#undef __FUNC__  
#define __FUNC__ "DAGetInterpolation"
/*@C
   DAGetInterpolation - Gets and interpolation matrix that maps between 
   grids associated with two DAs.

   Collective on DA

   Input Parameters:
+  dac - the coarse grid DA
-  daf - the fine grid DA

   Output Parameters:
.  A - the interpolation matrix

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: DARefine()
@*/
int DAGetInterpolation(DA dac,DA daf, Mat *A)
{
  int            ierr,dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE);
  PetscValidHeaderSpecific(daf,DA_COOKIE);
  PetscCheckSameComm(dac,daf);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(1,1,"Dimensions of DA do not match %d %d",dimc,dimf);CHKERRQ(ierr);
  if (mc != mf) SETERRQ2(1,1,"Processor dimensions of DA in X %d %d",mc,mf);CHKERRQ(ierr);
  if (nc != pf) SETERRQ2(1,1,"Processor dimensions of DA in Y %d %d",nc,nf);CHKERRQ(ierr);
  if (pc != pf) SETERRQ2(1,1,"Processor dimensions of DA in Z %d %d",pc,pf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(1,1,"DOF of DA do not match %d %d",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(1,1,"Stencil width of DA do not match %d %d",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(1,1,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(1,1,"Stencil type different in two DAs");CHKERRQ(ierr);

  if (dimc == 2 && dofc == 1 && sc == 1 && wrapc == DA_NONPERIODIC) {
    ierr = DAGetInterpolation_2D(dac,daf,A);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"No support for this DA yet");
  }
  PetscFunctionReturn(0);
} 
