#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.6 1999/10/01 21:21:04 bsmith Exp bsmith $";
#endif

static char help[]= "Scatters from a parallel vector to a sequential vector.\n\
uses block index sets\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           bs = 1, n = 5, ierr, ix0[3] = {5, 7, 9}, ix1[3] = {2,3,4};
  int           size,rank,i, iy0[3] = {1,2,4}, iy1[3] = {0,1,3},flg;
  Scalar        value;
  Vec           x,y;
  IS            isx,isy;
  VecScatter    ctx = 0, newctx;

  PetscInitialize(&argc,&argv,(char*)0,help); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  if (size != 2) SETERRQ(1,1,"Must run with 2 processors");

  ierr = OptionsGetInt(PETSC_NULL,"-bs",&bs,&flg);CHKERRA(ierr);
  n = bs*n;

  /* create two vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*n,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRA(ierr);

  /* create two index sets */
  for (i=0; i<3; i++ ) {
    ix0[i] *= bs; ix1[i] *= bs; 
    iy0[i] *= bs; iy1[i] *= bs; 
  }

  if (!rank) {
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,ix0,&isx);CHKERRA(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,iy0,&isy);CHKERRA(ierr);
  } else {
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,ix1,&isx);CHKERRA(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,iy1,&isy);CHKERRA(ierr);
  }

  /* fill local part of parallel vector */
  for ( i=n*rank; i<n*(rank+1); i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* fill local part of parallel vector */
  for ( i=0; i<n; i++ ) {
    value = -(Scalar) (i + 100*rank);
    ierr = VecSetValues(y,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRA(ierr);
  ierr = VecAssemblyEnd(y);CHKERRA(ierr);


  ierr = VecScatterCreate(x,isx,y,isy,&ctx);CHKERRA(ierr);
  ierr = VecScatterCopy(ctx,&newctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);

  ierr = VecScatterBegin(y,x,INSERT_VALUES,SCATTER_REVERSE,newctx);CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,INSERT_VALUES,SCATTER_REVERSE,newctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(newctx);CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = ISDestroy(isx);CHKERRA(ierr);
  ierr = ISDestroy(isy);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
