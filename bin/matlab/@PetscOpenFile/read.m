function d = read(freader,count,datatype)
%
%   O = read(freader,count,datatype) - reads data from a binary file opened with freader('filename')
%
if strcmp(datatype,'float128')
freader.fd
  d = longdoubleread(freader.fd,count,datatype);
else
  d = fread(freader.fd,count,datatype);
end

