/*$Id: PETScRund.java,v 1.1 2000/10/24 21:12:45 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;

/*
    This is the class that this file implements (must always be the same as
  the filename.
*/
public class PETScRund 
{
  public PETScRund() {
    ;
  }

  
  public void start() throws java.io.IOException
  {
    ServerSocket serve = new ServerSocket(2000);
    while (true) {
      Socket sock  = serve.accept();
      (new ThreadRunProgram(sock)).start();
    }
  }

  class ThreadRunProgram extends Thread
  {
    Socket sock;
    public ThreadRunProgram(Socket s) {sock = s;}
    public void run() {
      try {
        OutputStream out   = sock.getOutputStream();
        Runtime      rtime = Runtime.getRuntime();
 
        System.out.println("petscrsh make fire solaris /sandbox/bsmith/petsc/ src/vec/examples/tutorials ex1");

        Process make = rtime.exec("petscrsh make fire solaris /sandbox/bsmith/petsc/ src/vec/examples/tutorials ex1");
        PumpStream pump = new PumpStream(make.getInputStream(),out);
        pump.Pump();
      } catch (java.io.IOException ex) {;}
    }
  }

  public static void main(String s[])
  {
    PETScRund prun = new PETScRund();
    try {
      prun.start();
    } catch (java.io.IOException ex) {;}
  }
}




