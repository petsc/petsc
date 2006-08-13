function d = read(freader,count,datatype)
%
%   O = read(freader,count,datatype) - reads data from a binary file opened with freader('filename')
%
d = fread(freader.fd,count,datatype);

