#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex9.c,v 1.5 1999/03/19 21:24:17 bsmith Exp balay $";
#endif
      
static char help[] = "Tests DAGetColoring() in 3d.\n\n";

#include "mat.h"
#include "da.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            i,test_order,M = 3, N = 5, P=3, s=1, w=2, flg;
  int            m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, ierr;
  int            *lx = PETSC_NULL, *ly = PETSC_NULL, *lz = PETSC_NULL;
  DA             da;
  ISColoring     coloring;
  Mat            mat;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  Vec            lvec,dvec;
  MatFDColoring  fdcoloring;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */  
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-P",&P,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg);CHKERRA(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = OptionsHasName(PETSC_NULL,"-test_order",&test_order);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-distribute",&flg);CHKERRA(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRA(1,1,"Must set -m option with -distribute option");
    lx = (int *) PetscMalloc( m*sizeof(int) );CHKPTRQ(lx);
    for ( i=0; i<m-1; i++ ) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRA(1,1,"Must set -n option with -distribute option");
    ly = (int *) PetscMalloc( n*sizeof(int) );CHKPTRQ(ly);
    for ( i=0; i<n-1; i++ ) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRA(1,1,"Must set -p option with -distribute option");
    lz = (int *) PetscMalloc( p*sizeof(int) );CHKPTRQ(lz);
    for ( i=0; i<p-1; i++ ) { lz[i] = 2;}
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,stencil_type,M,N,P,m,n,p,w,s,
                    lx,ly,lz,&da);CHKERRA(ierr);
  if (lx) {PetscFree(lx); PetscFree(ly); PetscFree(lz);}

  ierr = DAGetColoring(da,&coloring,&mat);CHKERRA(ierr);
  ierr = MatFDColoringCreate(mat,coloring,&fdcoloring);CHKERRA(ierr); 

  ierr = DACreateGlobalVector(da,&dvec);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&lvec);CHKERRA(ierr);

  /* Free memory */
  ierr = MatFDColoringDestroy(fdcoloring);CHKERRA(ierr);
  ierr = VecDestroy(dvec);CHKERRA(ierr);
  ierr = VecDestroy(lvec);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr); 
  ierr = ISColoringDestroy(coloring);CHKERRA(ierr); 
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















