function write(sreader,data,datatype)
%
%   write(sreader,data,datatype) - writes data to a socket opened with sopen(socketnumber)
%
%   See $PETSC_DIR/share/petsc/matlab/@PetscOpenSocket/PetscOpenSocket.m
%
  switch (datatype)
  case 'int32'
    datatype = 16;
  case 'double'
    datatype = 1;
  case 'float64'
    datatype = 1;
  case 'uchar'
    datatype = 6;
    data     = data';
  otherwise
    disp(['Unknow datatype ' datatype])
    return
  end
swrite(sreader.fd,data,datatype);




