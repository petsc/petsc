function close(sreader)
%
%   O = close(sreader) - closes the socket connection
%
sclose(sreader.fd);
sreader.fd = 0;

