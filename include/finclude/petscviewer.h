C
C  $Id: viewer.h,v 1.13 1997/11/13 19:56:02 balay Exp bsmith $;
C
C  Include file for Fortran use of the Viewer package in PETSc
C
#define Viewer           integer
#define ViewerType       integer
#define ViewerBinaryType integer

      integer MATLAB_VIEWER, ASCII_FILE_VIEWER,ASCII_FILES_VIEWER,
     *        BINARY_FILE_VIEWER, STRING_VIEWER,DRAW_VIEWER 
      parameter (MATLAB_VIEWER = 0, ASCII_FILE_VIEWER = 1,
     *           ASCII_FILES_VIEWER = 2, BINARY_FILE_VIEWER = 3,
     *           STRING_VIEWER = 4, DRAW_VIEWER = 5) 
C
C  Flags for binary I/O
C
      integer  BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

      parameter (BINARY_RDONLY = 0, BINARY_WRONLY = 1, 
     *           BINARY_CREATE = 2)
C
C  Viewer formats
C
      integer  VIEWER_FORMAT_ASCII_DEFAULT,
     *         VIEWER_FORMAT_ASCII_MATLAB,
     *         VIEWER_FORMAT_ASCII_IMPL, 
     *         VIEWER_FORMAT_ASCII_INFO,
     *         VIEWER_FORMAT_ASCII_INFO_LONG, 
     *         VIEWER_FORMAT_ASCII_COMMON,
     *         VIEWER_FORMAT_ASCII_SYMMODU

      parameter ( VIEWER_FORMAT_ASCII_DEFAULT = 0,
     *            VIEWER_FORMAT_ASCII_MATLAB = 1,
     *            VIEWER_FORMAT_ASCII_IMPL = 2,
     *            VIEWER_FORMAT_ASCII_INFO = 3,
     *            VIEWER_FORMAT_ASCII_INFO_LONG = 4, 
     *            VIEWER_FORMAT_ASCII_COMMON = 5,
     *            VIEWER_FORMAT_ASCII_SYMMODU = 6)

      integer  VIEWER_FORMAT_BINARY_DEFAULT,
     *         VIEWER_FORMAT_BINARY_NATIVE
      parameter (VIEWER_FORMAT_BINARY_DEFAULT = 0, 
     *           VIEWER_FORMAT_BINARY_NATIVE = 1)

      integer VIEWER_FORMAT_DRAW_BASIC, VIEWER_FORMAT_DRAW_LG,
     *        VIEWER_FORMAT_DRAW_CONTOUR
      parameter (VIEWER_FORMAT_DRAW_BASIC=0, VIEWER_FORMAT_DRAW_LG=1,
     *           VIEWER_FORMAT_DRAW_CONTOUR=2)


C
C  End of Fortran include file for the Viewer package in PETSc



