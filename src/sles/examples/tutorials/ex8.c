#ifndef lint
static char vcid[] = "$Id: ex14.c,v 1.1 1995/11/01 00:18:45 bsmith Exp bsmith $";
#endif

static char help[] = "Tests the preconditioner ASM\n\n";

#include "mat.h"
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat     C;
  int     i, j, m = 15, n = 17, its, I, J, ierr, Istart, Iend, N = 1, M = 2;
  int     overlap = 1, Nsub, *idx, nidx, width, height,xstart,ystart;
  int     xleft,xright,yleft,yright;
  Scalar  v,  one = 1.0;
  Vec     u,b,x;
  SLES    sles;
  PC      pc;
  IS      *is;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(0,"-m",&m);   /* mesh lines in x */
  OptionsGetInt(0,"-n",&n);   /* mesh lines in y */
  OptionsGetInt(0,"-M",&M);   /* subdomains in x */
  OptionsGetInt(0,"-N",&N);   /* subdomains in y */
  OptionsGetInt(0,"-overlap",&overlap);
  Nsub = N*M;

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create and set vectors */
  ierr = VecCreate(MPI_COMM_WORLD,m*n,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  /* Create index sets defining subdomains */
  is = (IS *) PETSCMALLOC( Nsub*sizeof(IS **) ); CHKPTRQ(is);
  height = (n+1)/N; /* height of subdomain */
  if (height < 2) SETERRA(1,"Too many M subdomains for m mesh");
  ystart = 0;
  for ( i=0; i<N; i++ ) {
    if (ystart+height >= n) height += (n - ystart - height); 
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    width = (m+1)/M; /* width of subdomain */
    if (width < 2) SETERRA(1,"Too many M subdomains for m mesh");
    xstart = 0;
    for ( j=0; j<M; j++ ) {
      if (xstart+width >= m) width += (m - xstart - width); 
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;

      /*      
       printf("subdomain %d %d xstart %d end %d ystart %d end %d\n",xleft,xright,
              yleft,yright);
      */

      nidx   = (xleft - xright)*(yleft - yright);

      xstart += width;
    }
    ystart += height;
  }

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCASM); CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,C,C,ALLMAT_DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
