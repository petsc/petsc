!
!  $Id: viewer.h,v 1.23 1999/04/01 19:28:23 balay Exp balay $;
!
!  Include file for Fortran use of the Viewer package in PETSc
!
#if !defined (__VIEWER_H)
#define __VIEWER_H

#define Viewer           PetscFortranAddr
#define ViewerBinaryType integer
#define ViewerType       character*(80)

#define SOCKET_VIEWER       'socket'
#define ASCII_VIEWER        'ascii'
#define BINARY_VIEWER       'binary'
#define STRING_VIEWER       'string'
#define DRAW_VIEWER         'draw'
#define AMS_VIEWER          'ams'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)

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
      integer  VIEWER_FORMAT_ASCII_INDEX
      integer  VIEWER_FORMAT_ASCII_DENSE

      parameter (VIEWER_FORMAT_ASCII_DEFAULT = 0)
      parameter (VIEWER_FORMAT_ASCII_MATLAB = 1)
      parameter (VIEWER_FORMAT_ASCII_IMPL = 2)
      parameter (VIEWER_FORMAT_ASCII_INFO = 3)
      parameter (VIEWER_FORMAT_ASCII_INFO_LONG = 4)
      parameter (VIEWER_FORMAT_ASCII_COMMON = 5)
      parameter (VIEWER_FORMAT_ASCII_SYMMODU = 6)
      parameter (VIEWER_FORMAT_ASCII_INDEX = 7)
      parameter (VIEWER_FORMAT_ASCII_DENSE = 8)

      integer  VIEWER_FORMAT_BINARY_DEFAULT
      integer  VIEWER_FORMAT_BINARY_NATIVE

      parameter (VIEWER_FORMAT_BINARY_DEFAULT = 9) 
      parameter (VIEWER_FORMAT_BINARY_NATIVE = 10)

      integer VIEWER_FORMAT_DRAW_BASIC
      integer VIEWER_FORMAT_DRAW_LG
      integer VIEWER_FORMAT_DRAW_CONTOUR

      parameter (VIEWER_FORMAT_DRAW_BASIC = 11)
      parameter (VIEWER_FORMAT_DRAW_LG = 12)
      parameter (VIEWER_FORMAT_DRAW_CONTOUR = 13)

      integer VIEWER_FORMAT_NATIVE
      parameter (VIEWER_FORMAT_NATIVE = 14)
!
!  End of Fortran include file for the Viewer package in PETSc

#endif
