function d = read(sreader,count,datatype)
%
%   O = read(sreader,count,datatype) - reads data from a socket opened with sreader(socketnumber)
%
d = sread(sreader.fd,count,datatype);

