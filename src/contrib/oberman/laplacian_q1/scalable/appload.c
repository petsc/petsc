
#include "appctx.h"

/*
     Loads the quadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#undef __FUNCT__
#define __FUNCT__ "AppCxtCreate"
/*
    AppCtxCreate - Fills in the data structures using the grid information from 
  a AOData file.
*/
int AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  int        ierr;
  int        Nelx, Nely, nsdx, nsdy;
  double     xmin, xmax, ymin, ymax;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(AppCtx),appctx);CHKERRQ(ierr);
  (*appctx)->comm = comm;

  ierr = PetscOptionsGetInt   (PETSC_NULL,"-nx"  ,&Nelx,&flg);CHKERRQ(ierr); if (!flg) { Nelx = 5;   }
  ierr = PetscOptionsGetInt   (PETSC_NULL,"-ny"  ,&Nely,&flg);CHKERRQ(ierr); if (!flg) { Nely = 5;   }
  ierr = PetscOptionsGetInt   (PETSC_NULL,"-nsdx",&nsdx,&flg);CHKERRQ(ierr); if (!flg) { nsdx = 1;   }
  ierr = PetscOptionsGetInt   (PETSC_NULL,"-nsdy",&nsdy,&flg);CHKERRQ(ierr); if (!flg) { nsdy = 1;   }
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&xmin,&flg);CHKERRQ(ierr); if (!flg) { xmin = 0.0; }
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&xmax,&flg);CHKERRQ(ierr); if (!flg) { xmax = 1.0; }
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymin",&ymin,&flg);CHKERRQ(ierr); if (!flg) { ymin = 0.0; }
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymax",&ymax,&flg);CHKERRQ(ierr); if (!flg) { ymax = 1.0; }

  ierr = AppPartitionSetUp(&((*appctx)->part),comm,Nelx,Nely,nsdx,nsdy,xmin,xmax,ymin,ymax);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCxtDestroy"
/*
          Frees the all the data structures in the program
*/
int AppCtxDestroy(AppCtx *appctx)
{
  int        ierr;

  PetscFunctionBegin;

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
 
  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





