function d = read(sreader,count,datatype)
%
%   O = read(sreader,count,datatype) - reads data from a socket opened with sreader(socketnumber)
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
  otherwise
    disp(['Unknow datatype ' datatype])
    return
  end
d = sread(sreader.fd,count,datatype);
if datatype == 6
  d = d';
end



