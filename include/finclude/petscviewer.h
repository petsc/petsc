!
!  $Id: viewer.h,v 1.17 1998/03/27 21:17:38 balay Exp balay $;
!
!  Include file for Fortran use of the Viewer package in PETSc
!
#define Viewer           PetscFortranAddr
#define ViewerType       integer
#define ViewerBinaryType integer

      integer MATLAB_VIEWER, ASCII_FILE_VIEWER,ASCII_FILES_VIEWER
      integer BINARY_FILE_VIEWER, STRING_VIEWER,DRAW_VIEWER

      parameter (MATLAB_VIEWER = 0, ASCII_FILE_VIEWER = 1)
      parameter (ASCII_FILES_VIEWER = 2, BINARY_FILE_VIEWER = 3)
      parameter (STRING_VIEWER = 4, DRAW_VIEWER = 5) 
!
!  Flags for binary I/O
!
      integer  BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

      parameter (BINARY_RDONLY = 0, BINARY_WRONLY = 1)
      parameter (BINARY_CREATE = 2)
!
!  Viewer formats
!
      integer  VIEWER_FORMAT_ASCII_DEFAULT
      integer  VIEWER_FORMAT_ASCII_MATLAB
      integer  VIEWER_FORMAT_ASCII_IMPL
      integer  VIEWER_FORMAT_ASCII_INFO
      integer  VIEWER_FORMAT_ASCII_INFO_LONG
      integer  VIEWER_FORMAT_ASCII_COMMON
      integer  VIEWER_FORMAT_ASCII_SYMMODU

      parameter (VIEWER_FORMAT_ASCII_DEFAULT = 0)
      parameter (VIEWER_FORMAT_ASCII_MATLAB = 1)
      parameter (VIEWER_FORMAT_ASCII_IMPL = 2)
      parameter (VIEWER_FORMAT_ASCII_INFO = 3)
      parameter (VIEWER_FORMAT_ASCII_INFO_LONG = 4)
      parameter (VIEWER_FORMAT_ASCII_COMMON = 5)
      parameter (VIEWER_FORMAT_ASCII_SYMMODU = 6)

      integer  VIEWER_FORMAT_BINARY_DEFAULT
      integer  VIEWER_FORMAT_BINARY_NATIVE

      parameter (VIEWER_FORMAT_BINARY_DEFAULT = 0) 
      parameter (VIEWER_FORMAT_BINARY_NATIVE = 1)

      integer VIEWER_FORMAT_DRAW_BASIC, VIEWER_FORMAT_DRAW_LG
      integer VIEWER_FORMAT_DRAW_CONTOUR

      parameter (VIEWER_FORMAT_DRAW_BASIC=0, VIEWER_FORMAT_DRAW_LG=1)
      parameter (VIEWER_FORMAT_DRAW_CONTOUR=2)


!
!  End of Fortran include file for the Viewer package in PETSc



