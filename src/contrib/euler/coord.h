c
c  Parallel array sizes, including ghost points, for the
c  mesh coordinates.  Space is allocated in UserSetLocalMesh().
c
      double precision x(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      double precision y(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      double precision z(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)

c
c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
cc      double precision X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      double precision x,y,z
