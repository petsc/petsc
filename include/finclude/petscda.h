C
C  $Id: da.h,v 1.6 1996/02/12 20:31:16 bsmith Exp balay $;
C
C  Include file for Fortran use of the DA (distributed array) package in PETSc
C
#define DA             integer
#define DAPeriodicType integer
#define DAStencilType  integer
#define DADirection    integer

C
C  Types of stencils
C
      integer DA_STENCIL_STAR, DA_STENCIL_BOX

      parameter (DA_STENCIL_STAR = 0, DA_STENCIL_BOX = 1)
C
C  Types of periodicity
C
      integer DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC,
     *        DA_XYZPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC,DA_ZPERIODIC

      parameter (DA_NONPERIODIC = 0, DA_XPERIODIC = 1, DA_YPERIODIC = 2,
     *         DA_XYPERIODIC = 3, DA_XYZPERIODIC = 4, DA_XZPERIODIC = 5,
     *         DA_YZPERIODIC = 6, DA_ZPERIODIC = 7)
C
C DA Directions
C
      integer DA_X, DA_Y, DA_Z

      parameter (DA_X = 0, DA_Y = 1, DA_Z = 2)
C
C  End of Fortran include file for the DA package in PETSc

