function ex1(np)
#
#   ex1(np) 
# creates a series of vectors in PETSc and displays them in Matlab
#
err = launch('ex1 ',np);
if (err != 0) return;

p = openport;
v = receive(p);
plot(v); pause 0;
