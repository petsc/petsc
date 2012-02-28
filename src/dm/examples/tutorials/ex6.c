
static char help[] = "Demonstrates using 3 DMDA's to manage a slightly non-trivial grid";

#include <petscdmda.h>

struct _p_FA {
  MPI_Comm   comm[3];
  PetscInt   xl[3],yl[3],ml[3],nl[3];    /* corners and sizes of local vector in DMDA */
  PetscInt   xg[3],yg[3],mg[3],ng[3];    /* corners and sizes of global vector in DMDA */
  PetscInt   offl[3],offg[3];            /* offset in local and global vector of region 1, 2 and 3 portions */
  Vec        g,l;
  VecScatter vscat;
  PetscInt   p1,p2,r1,r2,r1g,r2g,sw;     
};
typedef struct _p_FA *FA;

typedef struct {
  PetscScalar X;
  PetscScalar Y;
} Field;

PetscErrorCode FAGetLocalCorners(FA fa,PetscInt j,PetscInt *x,PetscInt *y,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  if (fa->comm[j]) {
    *x = fa->xl[j];
    *y = fa->yl[j];
    *m = fa->ml[j];
    *n = fa->nl[j];
  } else {
    *x = *y = *m = *n = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FAGetGlobalCorners(FA fa,PetscInt j,PetscInt *x,PetscInt *y,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  if (fa->comm[j]) {
    *x = fa->xg[j];
    *y = fa->yg[j];
    *m = fa->mg[j];
    *n = fa->ng[j];
  } else {
    *x = *y = *m = *n = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FAGetLocalArray(FA fa,Vec v,PetscInt j,Field ***f)
{
  PetscErrorCode ierr;
  PetscScalar    *va;
  PetscInt       i;
  Field          **a;

  PetscFunctionBegin;
  if (fa->comm[j]) {
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    ierr = PetscMalloc(fa->nl[j]*sizeof(Field*),&a);CHKERRQ(ierr);
    for (i=0; i<fa->nl[j]; i++) (a)[i] = (Field*) (va + 2*fa->offl[j] + i*2*fa->ml[j] - 2*fa->xl[j]);
    *f = a - fa->yl[j];
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);    
  } else {
    *f = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FARestoreLocalArray(FA fa,Vec v,PetscInt j,Field ***f)
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  if (fa->comm[j]) {
    dummy = *f + fa->yl[j];
    ierr  = PetscFree(dummy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FAGetGlobalArray(FA fa,Vec v,PetscInt j,Field ***f)
{
  PetscErrorCode ierr;
  PetscScalar    *va;
  PetscInt       i;
  Field          **a;

  PetscFunctionBegin;
  if (fa->comm[j]) {
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    ierr = PetscMalloc(fa->ng[j]*sizeof(Field*),&a);CHKERRQ(ierr);
    for (i=0; i<fa->ng[j]; i++) (a)[i] = (Field*) (va + 2*fa->offg[j] + i*2*fa->mg[j] - 2*fa->xg[j]);
    *f = a - fa->yg[j];
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);    
  } else {
    *f = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FARestoreGlobalArray(FA fa,Vec v,PetscInt j,Field ***f)
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  if (fa->comm[j]) {
    dummy = *f + fa->yg[j];
    ierr  = PetscFree(dummy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FAGetGlobalVector(FA fa,Vec *v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDuplicate(fa->g,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FAGetLocalVector(FA fa,Vec *v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDuplicate(fa->l,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FAGlobalToLocal(FA fa,Vec g,Vec l)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(fa->vscat,g,l,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fa->vscat,g,l,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FADestroy(FA *fa)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&(*fa)->g);CHKERRQ(ierr);
  ierr = VecDestroy(&(*fa)->l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&(*fa)->vscat);CHKERRQ(ierr);
  ierr = PetscFree(*fa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FACreate(FA *infa)
{
  FA             fa;
  PetscMPIInt    rank;
  PetscInt       tonglobal,globalrstart,x,nx,y,ny,*tonatural,i,j,*to,*from,offt[3];
  PetscInt       *fromnatural,fromnglobal,nscat,nlocal,cntl1,cntl2,cntl3,*indices;
  PetscErrorCode ierr;

  /* Each DMDA manages the local vector for the portion of region 1, 2, and 3 for that processor
     Each DMDA can belong on any subset (overlapping between DMDA's or not) of processors
     For processes that a particular DMDA does not exist on, the corresponding comm should be set to zero
  */
  DM             da1 = 0,da2 = 0,da3 = 0;
  /* 
      v1, v2, v3 represent the local vector for a single DMDA
  */
  Vec            vl1 = 0,vl2 = 0,vl3 = 0, vg1 = 0, vg2 = 0,vg3 = 0;

  /*
     globalvec and friends represent the global vectors that are used for the PETSc solvers 
     localvec represents the concatenation of the (up to) 3 local vectors; vl1, vl2, vl3

     tovec and friends represent intermediate vectors that are ONLY used for setting up the 
     final communication patterns. Once this setup routine is complete they are destroyed.
     The tovec  is like the globalvec EXCEPT it has redundant locations for the ghost points
     between regions 2+3 and 1.
  */
  AO             toao,globalao;
  IS             tois,globalis,is;
  Vec            tovec,globalvec,localvec;
  VecScatter     vscat;
  PetscScalar    *globalarray,*localarray,*toarray;

  ierr = PetscNew(struct _p_FA,&fa);CHKERRQ(ierr);
  /*
      fa->sw is the stencil width  

      fa->p1 is the width of region 1, fa->p2 the width of region 2 (must be the same)
      fa->r1 height of region 1 
      fa->r2 height of region 2
 
      fa->r2 is also the height of region 3-4
      (fa->p1 - fa->p2)/2 is the width of both region 3 and region 4
  */
  fa->p1 = 24;
  fa->p2 = 15;
  fa->r1 = 6; 
  fa->r2 = 6;
  fa->sw = 1;
  fa->r1g = fa->r1 + fa->sw;
  fa->r2g = fa->r2 + fa->sw;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  fa->comm[0] = PETSC_COMM_WORLD;
  fa->comm[1] = PETSC_COMM_WORLD;
  fa->comm[2] = PETSC_COMM_WORLD;
  /* Test case with different communicators */
  /* Normally one would use MPI_Comm routines to build MPI communicators on which you wish to partition the DMDAs*/
  /*
  if (rank == 0) {
    fa->comm[0] = PETSC_COMM_SELF;
    fa->comm[1] = 0;
    fa->comm[2] = 0;
  } else if (rank == 1) {
    fa->comm[0] = 0;  
    fa->comm[1] = PETSC_COMM_SELF;
    fa->comm[2] = 0;  
  } else {
    fa->comm[0] = 0;  
    fa->comm[1] = 0;
    fa->comm[2] = PETSC_COMM_SELF;
  } */

  if (fa->p2 > fa->p1 - 3)   SETERRQ(PETSC_COMM_SELF,1,"Width of region fa->p2 must be at least 3 less then width of region 1");
  if (!((fa->p2 - fa->p1) % 2)) SETERRQ(PETSC_COMM_SELF,1,"width of region 3 must NOT be divisible by 2");

  if (fa->comm[1]) {
    ierr = DMDACreate2d(fa->comm[1],DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,fa->p2,fa->r2g,PETSC_DECIDE,PETSC_DECIDE,1,fa->sw,PETSC_NULL,PETSC_NULL,&da2);CHKERRQ(ierr);
    ierr = DMGetLocalVector(da2,&vl2);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da2,&vg2);CHKERRQ(ierr);
  }
  if (fa->comm[2]) {
    ierr = DMDACreate2d(fa->comm[2],DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,fa->p1-fa->p2,fa->r2g,PETSC_DECIDE,PETSC_DECIDE,1,fa->sw,PETSC_NULL,PETSC_NULL,&da3);CHKERRQ(ierr);
    ierr = DMGetLocalVector(da3,&vl3);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da3,&vg3);CHKERRQ(ierr);
  }
  if (fa->comm[0]) {
    ierr = DMDACreate2d(fa->comm[0],DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,fa->p1,fa->r1g,PETSC_DECIDE,PETSC_DECIDE,1,fa->sw,PETSC_NULL,PETSC_NULL,&da1);CHKERRQ(ierr);
    ierr = DMGetLocalVector(da1,&vl1);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da1,&vg1);CHKERRQ(ierr);
  }

  /* count the number of unknowns owned on each processor and determine the starting point of each processors ownership 
     for global vector with redundancy */
  tonglobal = 0;
  if (fa->comm[1]) {
    ierr = DMDAGetCorners(da2,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    tonglobal += nx*ny;
  }
  if (fa->comm[2]) {
    ierr = DMDAGetCorners(da3,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    tonglobal += nx*ny;
  }
  if (fa->comm[0]) {
    ierr = DMDAGetCorners(da1,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    tonglobal += nx*ny;
  }
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Number of unknowns owned %d\n",rank,tonglobal);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  /* Get tonatural number for each node */
  ierr = PetscMalloc((tonglobal+1)*sizeof(PetscInt),&tonatural);CHKERRQ(ierr);
  tonglobal = 0;
  if (fa->comm[1]) {
    ierr = DMDAGetCorners(da2,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        tonatural[tonglobal++] = (fa->p1 - fa->p2)/2 + x + i + fa->p1*(y + j);
      }
    }
  }
  if (fa->comm[2]) {
    ierr = DMDAGetCorners(da3,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        if (x + i < (fa->p1 - fa->p2)/2) tonatural[tonglobal++] = x + i + fa->p1*(y + j);
        else tonatural[tonglobal++] = fa->p2 + x + i + fa->p1*(y + j);
      }
    }
  }
  if (fa->comm[0]) {
    ierr = DMDAGetCorners(da1,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        tonatural[tonglobal++] = fa->p1*fa->r2g + x + i + fa->p1*(y + j);
      }
    }
  }
  /*  ierr = PetscIntView(tonglobal,tonatural,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = AOCreateBasic(PETSC_COMM_WORLD,tonglobal,tonatural,0,&toao);CHKERRQ(ierr);
  ierr = PetscFree(tonatural);CHKERRQ(ierr);

  /* count the number of unknowns owned on each processor and determine the starting point of each processors ownership 
     for global vector without redundancy */
  fromnglobal = 0;
  fa->offg[1] = 0;
  offt[1]     = 0;
  if (fa->comm[1]) {
    ierr = DMDAGetCorners(da2,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    offt[2] = nx*ny;
    if (y+ny == fa->r2g) {ny--;}  /* includes the ghost points on the upper side */
    fromnglobal += nx*ny;
    fa->offg[2] = fromnglobal;
  } else {
    offt[2] = 0;
    fa->offg[2] = 0;
  }
  if (fa->comm[2]) {
    ierr = DMDAGetCorners(da3,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    offt[0] = offt[2] + nx*ny;
    if (y+ny == fa->r2g) {ny--;}  /* includes the ghost points on the upper side */
    fromnglobal += nx*ny;
    fa->offg[0] = fromnglobal;
  } else {
    offt[0]     = offt[2];
    fa->offg[0] = fromnglobal;
  }
  if (fa->comm[0]) {
    ierr = DMDAGetCorners(da1,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    if (y == 0) {ny--;}  /* includes the ghost points on the lower side */
    fromnglobal += nx*ny;
  }
  ierr        = MPI_Scan(&fromnglobal,&globalrstart,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  globalrstart -= fromnglobal;
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Number of unknowns owned %d\n",rank,fromnglobal);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* Get fromnatural number for each node */
  ierr = PetscMalloc((fromnglobal+1)*sizeof(PetscInt),&fromnatural);CHKERRQ(ierr);
  fromnglobal = 0;
  if (fa->comm[1]) {
    ierr = DMDAGetCorners(da2,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    if (y+ny == fa->r2g) {ny--;}  /* includes the ghost points on the upper side */
    fa->xg[1] = x; fa->yg[1] = y; fa->mg[1] = nx; fa->ng[1] = ny;
    ierr = DMDAGetGhostCorners(da2,&fa->xl[1],&fa->yl[1],0,&fa->ml[1],&fa->nl[1],0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        fromnatural[fromnglobal++] = (fa->p1 - fa->p2)/2 + x + i + fa->p1*(y + j);
      }
    }
  }
  if (fa->comm[2]) {
    ierr = DMDAGetCorners(da3,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    if (y+ny == fa->r2g) {ny--;}  /* includes the ghost points on the upper side */
    fa->xg[2] = x; fa->yg[2] = y; fa->mg[2] = nx; fa->ng[2] = ny;
    ierr = DMDAGetGhostCorners(da3,&fa->xl[2],&fa->yl[2],0,&fa->ml[2],&fa->nl[2],0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        if (x + i < (fa->p1 - fa->p2)/2) fromnatural[fromnglobal++] = x + i + fa->p1*(y + j);
        else fromnatural[fromnglobal++] = fa->p2 + x + i + fa->p1*(y + j);
      }
    }
  }
  if (fa->comm[0]) {
    ierr = DMDAGetCorners(da1,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    if (y == 0) {ny--;}  /* includes the ghost points on the lower side */
    else y--;
    fa->xg[0] = x; fa->yg[0] = y; fa->mg[0] = nx; fa->ng[0] = ny;
    ierr = DMDAGetGhostCorners(da1,&fa->xl[0],&fa->yl[0],0,&fa->ml[0],&fa->nl[0],0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        fromnatural[fromnglobal++] = fa->p1*fa->r2 + x + i + fa->p1*(y + j);
      }
    }
  }
  /*ierr = PetscIntView(fromnglobal,fromnatural,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/
  ierr = AOCreateBasic(PETSC_COMM_WORLD,fromnglobal,fromnatural,0,&globalao);CHKERRQ(ierr);
  ierr = PetscFree(fromnatural);CHKERRQ(ierr);

  /* ---------------------------------------------------*/
  /* Create the scatter that updates 1 from 2 and 3 and 3 and 2 from 1 */
  /* currently handles stencil width of 1 ONLY */
  ierr = PetscMalloc(tonglobal*sizeof(PetscInt),&to);CHKERRQ(ierr);
  ierr = PetscMalloc(tonglobal*sizeof(PetscInt),&from);CHKERRQ(ierr);
  nscat = 0;
  if (fa->comm[1]) {
    ierr = DMDAGetCorners(da2,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        to[nscat] = from[nscat] = (fa->p1 - fa->p2)/2 + x + i + fa->p1*(y + j);nscat++;
      }
    }
  }
  if (fa->comm[2]) {
    ierr = DMDAGetCorners(da3,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        if (x + i < (fa->p1 - fa->p2)/2) {
          to[nscat]   = from[nscat] = x + i + fa->p1*(y + j);nscat++;
        } else {
          to[nscat]   = from[nscat] = fa->p2 + x + i + fa->p1*(y + j);nscat++;
        }
      }
    }
  }
  if (fa->comm[0]) {
    ierr = DMDAGetCorners(da1,&x,&y,0,&nx,&ny,0);CHKERRQ(ierr);
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        to[nscat]     = fa->p1*fa->r2g + x + i + fa->p1*(y + j);
        from[nscat++] = fa->p1*(fa->r2 - 1) + x + i + fa->p1*(y + j);
      }
    }
  }
  ierr = AOApplicationToPetsc(toao,nscat,to);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(globalao,nscat,from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,nscat,to,PETSC_COPY_VALUES,&tois);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,nscat,from,PETSC_COPY_VALUES,&globalis);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,tonglobal,PETSC_DETERMINE,&tovec);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,fromnglobal,PETSC_DETERMINE,&globalvec);CHKERRQ(ierr);
  ierr = VecScatterCreate(globalvec,globalis,tovec,tois,&vscat);CHKERRQ(ierr);
  ierr = ISDestroy(&tois);CHKERRQ(ierr);
  ierr = ISDestroy(&globalis);CHKERRQ(ierr);
  ierr = AODestroy(&globalao);CHKERRQ(ierr);
  ierr = AODestroy(&toao);CHKERRQ(ierr);

  /* fill up global vector without redundant values with PETSc global numbering */
  ierr = VecGetArray(globalvec,&globalarray);CHKERRQ(ierr);
  for (i=0; i<fromnglobal; i++) {
    globalarray[i] = globalrstart + i;
  }
  ierr = VecRestoreArray(globalvec,&globalarray);CHKERRQ(ierr);
  
  /* scatter PETSc global indices to redundant valueed array */
  ierr = VecScatterBegin(vscat,globalvec,tovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,globalvec,tovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  /* Create local vector that is the concatenation of the local vectors */
  nlocal = 0;
  cntl1  = cntl2 = cntl3 = 0;
  if (fa->comm[1]) {
    ierr = VecGetSize(vl2,&cntl2);CHKERRQ(ierr);
    nlocal += cntl2;
  }  
  if (fa->comm[2]) {
    ierr = VecGetSize(vl3,&cntl3);CHKERRQ(ierr);
    nlocal += cntl3;
  }  
  if (fa->comm[0]) {
    ierr = VecGetSize(vl1,&cntl1);CHKERRQ(ierr);
    nlocal += cntl1;
  }
  fa->offl[0] = cntl2 + cntl3;
  fa->offl[1] = 0;
  fa->offl[2] = cntl2;
  ierr = VecCreateSeq(PETSC_COMM_SELF,nlocal,&localvec);CHKERRQ(ierr);
  
  /* cheat so that  vl1, vl2, vl3 shared array memory with localvec */
  ierr = VecGetArray(localvec,&localarray);CHKERRQ(ierr);
  ierr = VecGetArray(tovec,&toarray);CHKERRQ(ierr);
  if (fa->comm[1]) {
    ierr = VecPlaceArray(vl2,localarray+fa->offl[1]);CHKERRQ(ierr);
    ierr = VecPlaceArray(vg2,toarray+offt[1]);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da2,vg2,INSERT_VALUES,vl2);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da2,vg2,INSERT_VALUES,vl2);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da2,&vg2);CHKERRQ(ierr);
  }  
  if (fa->comm[2]) {
    ierr = VecPlaceArray(vl3,localarray+fa->offl[2]);CHKERRQ(ierr);
    ierr = VecPlaceArray(vg3,toarray+offt[2]);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da3,vg3,INSERT_VALUES,vl3);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da3,vg3,INSERT_VALUES,vl3);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da3,&vg3);CHKERRQ(ierr);
  }  
  if (fa->comm[0]) {
    ierr = VecPlaceArray(vl1,localarray+fa->offl[0]);CHKERRQ(ierr);
    ierr = VecPlaceArray(vg1,toarray+offt[0]);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da1,vg1,INSERT_VALUES,vl1);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da1,vg1,INSERT_VALUES,vl1);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da1,&vg1);CHKERRQ(ierr);
  }  
  ierr = VecRestoreArray(localvec,&localarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(tovec,&toarray);CHKERRQ(ierr);

  /* no longer need the redundant vector and VecScatter to it */
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  ierr = VecDestroy(&tovec);CHKERRQ(ierr);

  /* Create final scatter that goes directly from globalvec to localvec */
  /* this is the one to be used in the application code */
  ierr = PetscMalloc(nlocal*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = VecGetArray(localvec,&localarray);CHKERRQ(ierr);
  for (i=0; i<nlocal; i++) {
    indices[i] = (PetscInt) (localarray[i]);
  }
  ierr = VecRestoreArray(localvec,&localarray);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_WORLD,2,nlocal,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,2*nlocal,&fa->l);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,2*fromnglobal,PETSC_DETERMINE,&fa->g);CHKERRQ(ierr);

  ierr = VecScatterCreate(fa->g,is,fa->l,PETSC_NULL,&fa->vscat);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = VecDestroy(&globalvec);CHKERRQ(ierr);
  ierr = VecDestroy(&localvec);CHKERRQ(ierr);
  if (fa->comm[0]) {
    ierr = DMRestoreLocalVector(da1,&vl1);CHKERRQ(ierr);
    ierr = DMDestroy(&da1);CHKERRQ(ierr);
  }
  if (fa->comm[1]) {
    ierr = DMRestoreLocalVector(da2,&vl2);CHKERRQ(ierr);
    ierr = DMDestroy(&da2);CHKERRQ(ierr);
  }
  if (fa->comm[2]) {
    ierr = DMRestoreLocalVector(da3,&vl3);CHKERRQ(ierr);
    ierr = DMDestroy(&da3);CHKERRQ(ierr);
  }
  *infa = fa;
  PetscFunctionReturn(0);
}

/* Crude graphics to test that the ghost points are properly updated */
#include <petscdraw.h>

typedef struct {
  PetscInt     m[3],n[3];
  PetscScalar  *xy[3];
} ZoomCtx;

PetscErrorCode DrawPatch(PetscDraw draw,void *ctx)
{
  ZoomCtx        *zctx = (ZoomCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       m,n,i,j,id,k;
  PetscReal      x1,x2,x3,x4,y_1,y2,y3,y4;
  PetscScalar    *xy;

  PetscFunctionBegin;
  for (k=0; k<3; k++) {
    m    = zctx->m[k];
    n    = zctx->n[k];
    xy   = zctx->xy[k];

    for (j=0; j<n-1; j++) {
      for (i=0; i<m-1; i++) {
	id = i+j*m;    x1 = xy[2*id];y_1 = xy[2*id+1];
	id = i+j*m+1;  x2 = xy[2*id];y2  = xy[2*id+1];
	id = i+j*m+1+m;x3 = xy[2*id];y3  = xy[2*id+1];
	id = i+j*m+m;  x4 = xy[2*id];y4  = xy[2*id+1];
	ierr = PetscDrawLine(draw,x1,y_1,x2,y2,PETSC_DRAW_BLACK);CHKERRQ(ierr);
	ierr = PetscDrawLine(draw,x2,y2,x3,y3,PETSC_DRAW_BLACK);CHKERRQ(ierr);
	ierr = PetscDrawLine(draw,x3,y3,x4,y4,PETSC_DRAW_BLACK);CHKERRQ(ierr);
	ierr = PetscDrawLine(draw,x4,y4,x1,y_1,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DrawFA(FA fa,Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *va;
  ZoomCtx        zctx;
  PetscDraw      draw;
  PetscReal      xmint = 10000.0,xmaxt = -10000.0,ymint = 100000.0,ymaxt = -10000.0;
  PetscReal      xmin,xmax,ymin,ymax;
  PetscInt       i,vn,ln,j;

  PetscFunctionBegin;
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);
  ierr = VecGetSize(v,&vn);CHKERRQ(ierr);
  ierr = VecGetSize(fa->l,&ln);CHKERRQ(ierr);
  for (j=0; j<3; j++) {
    if (vn == ln) {
      zctx.xy[j] = va + 2*fa->offl[j];
      zctx.m[j]  = fa->ml[j];
      zctx.n[j]  = fa->nl[j];
    } else {
      zctx.xy[j] = va + 2*fa->offg[j];
      zctx.m[j]  = fa->mg[j];
      zctx.n[j]  = fa->ng[j];
    }
    for (i=0; i<zctx.m[j]*zctx.n[j]; i++) {
      if (zctx.xy[j][2*i] > xmax) xmax = zctx.xy[j][2*i];
      if (zctx.xy[j][2*i] < xmin) xmin = zctx.xy[j][2*i];
      if (zctx.xy[j][2*i+1] > ymax) ymax = zctx.xy[j][2*i+1];
      if (zctx.xy[j][2*i+1] < ymin) ymin = zctx.xy[j][2*i+1];
    }   
  }
  ierr = MPI_Allreduce(&xmin,&xmint,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&xmax,&xmaxt,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&ymin,&ymint,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&ymax,&ymaxt,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  xmin = xmint - .2*(xmaxt - xmint);
  xmax = xmaxt + .2*(xmaxt - xmint);
  ymin = ymint - .2*(ymaxt - ymint);
  ymax = ymaxt + .2*(ymaxt - ymint);
#if defined(PETSC_HAVE_X)
  ierr = PetscDrawOpenX(PETSC_COMM_WORLD,0,"meshes",PETSC_DECIDE,PETSC_DECIDE,700,700,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,DrawPatch,&zctx);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* crude mappings from rectangular arrays to the true geometry. These are ONLY for testing!
   they will not be used the actual code */
PetscErrorCode FAMapRegion3(FA fa,Vec g)
{
  PetscErrorCode ierr;
  PetscReal      R = 1.0,Rscale,Ascale;
  PetscInt       i,k,x,y,m,n;
  Field          **ga;

  PetscFunctionBegin; 
  Rscale = R/(fa->r2-1);
  Ascale = 2.0*PETSC_PI/(3.0*(fa->p1 - fa->p2 - 1));

  ierr = FAGetGlobalCorners(fa,2,&x,&y,&m,&n);CHKERRQ(ierr);
  ierr = FAGetGlobalArray(fa,g,2,&ga);CHKERRQ(ierr);
  for (k=y; k<y+n; k++) {
    for (i=x; i<x+m; i++) {
      ga[k][i].X = (R + k*Rscale)*PetscCosScalar(1.*PETSC_PI/6. + i*Ascale);
      ga[k][i].Y = (R + k*Rscale)*PetscSinScalar(1.*PETSC_PI/6. + i*Ascale) - 4.*R;
    }
  }
  ierr = FARestoreGlobalArray(fa,g,2,&ga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FAMapRegion2(FA fa,Vec g)
{
  PetscErrorCode ierr;
  PetscReal      R = 1.0,Rscale,Ascale;
  PetscInt       i,k,x,y,m,n;
  Field          **ga;

  PetscFunctionBegin; 
  Rscale = R/(fa->r2-1);
  Ascale = 2.0*PETSC_PI/fa->p2;

  ierr = FAGetGlobalCorners(fa,1,&x,&y,&m,&n);CHKERRQ(ierr);
  ierr = FAGetGlobalArray(fa,g,1,&ga);CHKERRQ(ierr);
  for (k=y; k<y+n; k++) {
    for (i=x; i<x+m; i++) {
      ga[k][i].X = (R + k*Rscale)*PetscCosScalar(i*Ascale - PETSC_PI/2.0);
      ga[k][i].Y = (R + k*Rscale)*PetscSinScalar(i*Ascale - PETSC_PI/2.0);
    }
  }
  ierr = FARestoreGlobalArray(fa,g,1,&ga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FAMapRegion1(FA fa,Vec g)
{
  PetscErrorCode ierr;
  PetscReal      R = 1.0,Rscale,Ascale1,Ascale3;
  PetscInt       i,k,x,y,m,n;
  Field          **ga;

  PetscFunctionBegin; 
  Rscale  = R/(fa->r1-1);
  Ascale1 = 2.0*PETSC_PI/fa->p2;
  Ascale3 = 2.0*PETSC_PI/(3.0*(fa->p1 - fa->p2 - 1));

  ierr = FAGetGlobalCorners(fa,0,&x,&y,&m,&n);CHKERRQ(ierr);
  ierr = FAGetGlobalArray(fa,g,0,&ga);CHKERRQ(ierr);

  /* This mapping is WRONG! Not sure how to do it so I've done a poor job of
     it. You can see that the grid connections are correct. */
  for (k=y; k<y+n; k++) {
    for (i=x; i<x+m; i++) {
      if (i < (fa->p1-fa->p2)/2) {
	ga[k][i].X = (2.0*R + k*Rscale)*PetscCosScalar(i*Ascale3);
	ga[k][i].Y = (2.0*R + k*Rscale)*PetscSinScalar(i*Ascale3) - 4.*R;
      } else if (i > fa->p2 + (fa->p1 - fa->p2)/2) {
	ga[k][i].X = (2.0*R + k*Rscale)*PetscCosScalar(PETSC_PI+i*Ascale3);
	ga[k][i].Y = (2.0*R + k*Rscale)*PetscSinScalar(PETSC_PI+i*Ascale3) - 4.*R;
      } else {
        ga[k][i].X = (2.*R + k*Rscale)*PetscCosScalar((i-(fa->p1-fa->p2)/2)*Ascale1 - PETSC_PI/2.0);
        ga[k][i].Y = (2.*R + k*Rscale)*PetscSinScalar((i-(fa->p1-fa->p2)/2)*Ascale1 - PETSC_PI/2.0);
      }
    }
  }
  ierr = FARestoreGlobalArray(fa,g,0,&ga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Simple test to check that the ghost points are properly updated */
PetscErrorCode FATest(FA fa)
{
  PetscErrorCode ierr;
  Vec            l,g;
  Field          **la;
  PetscInt       x,y,m,n,j,i,k,p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = FAGetGlobalVector(fa,&g);CHKERRQ(ierr);
  ierr = FAGetLocalVector(fa,&l);CHKERRQ(ierr);

  /* fill up global vector of one region at a time with ITS logical coordinates, then update LOCAL
     vector; print local vectors to confirm they are correctly filled */
  for (j=0; j<3; j++) {
    ierr = VecSet(g,0.0);CHKERRQ(ierr);
    ierr = FAGetGlobalCorners(fa,j,&x,&y,&m,&n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nFilling global region %d, showing local results \n",j+1);CHKERRQ(ierr);
    ierr = FAGetGlobalArray(fa,g,j,&la);CHKERRQ(ierr);
    for (k=y; k<y+n; k++) {
      for (i=x; i<x+m; i++) {
        la[k][i].X = i;
        la[k][i].Y = k;
      }
    }
    ierr = FARestoreGlobalArray(fa,g,j,&la);CHKERRQ(ierr);

    ierr = FAGlobalToLocal(fa,g,l);CHKERRQ(ierr);
    ierr = DrawFA(fa,g);CHKERRQ(ierr);
    ierr = DrawFA(fa,l);CHKERRQ(ierr);

    for (p=0; p<3; p++) {
      ierr = FAGetLocalCorners(fa,p,&x,&y,&m,&n);CHKERRQ(ierr);
      ierr = FAGetLocalArray(fa,l,p,&la);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n[%d] Local array for region %d \n",rank,p+1);CHKERRQ(ierr);
      for (k=y+n-1; k>=y; k--) { /* print in reverse order to match diagram in paper */
        for (i=x; i<x+m; i++) {
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"(%G,%G) ",la[k][i].X,la[k][i].Y);CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
      }
      ierr = FARestoreLocalArray(fa,l,p,&la);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  FA             fa;
  PetscErrorCode ierr;
  Vec            g,l;

  PetscInitialize(&argc,&argv,0,help);

  ierr = FACreate(&fa);CHKERRQ(ierr);
  /* ierr = FATest(fa);CHKERRQ(ierr);*/

  ierr = FAGetGlobalVector(fa,&g);CHKERRQ(ierr);
  ierr = FAGetLocalVector(fa,&l);CHKERRQ(ierr);

  ierr = FAMapRegion1(fa,g);CHKERRQ(ierr);
  ierr = FAMapRegion2(fa,g);CHKERRQ(ierr);
  ierr = FAMapRegion3(fa,g);CHKERRQ(ierr);

  ierr = FAGlobalToLocal(fa,g,l);CHKERRQ(ierr);
  ierr = DrawFA(fa,g);CHKERRQ(ierr);
  ierr = DrawFA(fa,l);CHKERRQ(ierr);

  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);
  ierr = FADestroy(&fa);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

