c
c  Parallel array sizes, including ghost points, for the
c  mesh coordinates.  Space is allocated in UserSetLocalMesh().
c
      Double x(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      Double y(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      Double z(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)

c
c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
cc      Double X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      Double x,y,z
