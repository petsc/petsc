
C      Include file for for Fortran use of the Vec package
C
#define Vec           integer
#define VecScatterCtx integer 
C
      integer INSERTVALUES, ADDVALUES
      integer SCATTERREVERSE,SCATTERDOWN,SCATTERUP,SCATTERALL,
     *        SCATTERALLREVERSE
      integer PIPELINEDOWN,PIPELINEUP

      parameter (INSERTVALUES = 1, ADDVALUES = 2)
      parameter (SCATTERREVERSE = 1,SCATTERDOWN = 2,SCATTERUP = 4,
     *           SCATTERALL = 8, SCATTERALLREVERSE = 9)
      parameter (PIPELINEDOWN = 0,PIPELINEUP = 1)
C
C      End of Fortran include file for the Vec package

