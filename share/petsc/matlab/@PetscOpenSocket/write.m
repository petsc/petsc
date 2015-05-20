function write(sreader,data,datatype)
%
%   write(sreader,data,datatype) - writes data to a socket opened with sreader(socketnumber)
%
  switch (datatype)
  case 'int32'
    datatype = 0;
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




