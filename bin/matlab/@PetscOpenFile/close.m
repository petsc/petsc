function close(freader)
%
%   O = close(freader) - closes the binary file
%
fclose(freader.fd);
freader.fd = 0;

