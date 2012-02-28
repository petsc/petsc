function  write(freader,data,datatype)
%
%    write(freader,data,datatype) - writes data to a binary file opened with freader('filename')
%
fwrite(freader.fd,data,datatype);

