/*$Id: ex4.c,v 1.41 1999/11/05 14:47:57 bsmith Exp bsmith $*/
  
static char help[] = "Tests various 2-dimensional DA routines.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            rank,M = 10,N = 8,m = PETSC_DECIDE,ierr;
  int            s=2,w=2,n = PETSC_DECIDE,nloc,l,i,j,kk;
  int            Xs,Xm,Ys,Ym,iloc,*iglobal,*ltog;
  int            *lx = PETSC_NULL,*ly = PETSC_NULL;
  PetscTruth     testorder,flg;
  DAPeriodicType wrap = DA_NONPERIODIC;
  DA             da;
  Viewer         viewer;
  Vec            local,global;
  Scalar         value;
  DAStencilType  st = DA_STENCIL_BOX;
  AO             ao;
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,400,&viewer);CHKERRA(ierr);
 
  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-xwrap",&flg);CHKERRA(ierr); if (flg)  wrap = DA_XPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-ywrap",&flg);CHKERRA(ierr); if (flg)  wrap = DA_YPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-xywrap",&flg);CHKERRA(ierr); if (flg) wrap = DA_XYPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg);CHKERRA(ierr); if (flg)   st = DA_STENCIL_STAR;
  ierr = OptionsHasName(PETSC_NULL,"-testorder",&testorder);CHKERRA(ierr);
  /*
      Test putting two nodes in x and y on each processor, exact last processor 
      in x and y gets the rest.
  */
  ierr = OptionsHasName(PETSC_NULL,"-distribute",&flg);CHKERRA(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRA(1,1,"Must set -m option with -distribute option");
    lx = (int*)PetscMalloc(m*sizeof(int));CHKPTRQ(lx);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRA(1,1,"Must set -n option with -distribute option");
    ly = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(lx);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
  }


  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,wrap,st,M,N,m,n,w,s,lx,ly,&da);CHKERRA(ierr);
  if (lx) { 
    ierr = PetscFree(lx);CHKERRA(ierr);
    ierr = PetscFree(ly);CHKERRA(ierr);
  }

  ierr = DAView(da,viewer);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global);CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  value = rank;
  ierr = VecScale(&value,local);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  if (!testorder) { /* turn off printing when testing ordering mappings */
    ierr = PetscPrintf (PETSC_COMM_WORLD,"\nGlobal Vectors:\n");CHKERRA(ierr);
    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_NATIVE,0);CHKERRA(ierr);
    ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
    ierr = PetscPrintf (PETSC_COMM_WORLD,"\n\n");CHKERRA(ierr);
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg);CHKERRA(ierr);
  if (flg) {
    Viewer sviewer;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRA(ierr);
    ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = VecView(local,sviewer);CHKERRA(ierr); 
    ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (testorder) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,PETSC_NULL,&Xm,&Ym,PETSC_NULL);CHKERRA(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog);CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao);CHKERRA(ierr);
    iglobal = (int*)PetscMalloc(nloc*sizeof(int));CHKPTRA(iglobal);

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DA ordering of grid points */
    kk = 0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs); 
        for (l=0; l<w; l++) {
          iglobal[kk++] = ltog[iloc+l];
        }
      }
    } 

    /* Map this to the application ordering (which for DAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y) */
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRA(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRA(ierr); 

    /* Verify the mappings */
    kk=0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs); 
        for (l=0; l<w; l++) {
          if (iglobal[kk] != ltog[iloc+l]) {fprintf(stdout,
            "[%d] Problem with mapping: j=%d, i=%d, l=%d, petsc1=%d, petsc2=%d\n",
             rank,j,i,l,ltog[iloc+l],iglobal[kk]);}
          kk++;
        }
      }
    }
    ierr = PetscFree(iglobal);CHKERRA(ierr);
  } 

  /* Free memory */
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
