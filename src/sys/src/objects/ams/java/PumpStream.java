/*$Id: PumpStream.java,v 1.2 2000/10/25 19:11:52 bsmith Exp bsmith $*/
/*
     Connects two streams together. Simply takes all input from first and
  passes it into second. This allows us to move stdout etc. from the server 
  back to the client
*/

import java.lang.Thread;
import java.net.*;
import java.io.*;

public class PumpStream {
  InputStream  in,err;
  OutputStream out;
  
  /*
     The constructor for the class.
  */
  public PumpStream(Process make,OutputStream o)
  {
    in      = make.getInputStream();
    err     = make.getErrorStream();
    out     = o;
  }

  public int Pump() throws java.io.IOException
  {

    /* thread to monitor stderr and forward any of it */
    (new Thread() {
       public void run() {
         System.out.println("PumpStream: Monitoring stderr");
         try {
           ServerSocket serve = new ServerSocket(2002);
           Socket sock  = serve.accept();
           sock.setSoLinger(true,5);
           byte[] data = new byte[1024];
           int    fd   = 0;
           System.out.println("PumpStream: Waiting for stderr");
           while (true) {
             fd = err.read(data);
             if (fd == -1) break;
             System.out.println("PumpStream: Found stderr");
             sock.getOutputStream().write(data,0,fd);
           }
           sock.getOutputStream().close();
           sock.close();
           serve.close();
         } catch (java.io.IOException oops) {System.out.println("PumpStream: Problem handling stderr");}
         System.out.println("PumpStream: Done monitoring stderr");
       }
    }).start();

    byte[] data = new byte[1024];
    int    fd   = 0,cnt = 0;
    while (true) {
      fd = in.read(data);
      if (fd == -1) break;
      cnt += fd;
      out.write(data,0,fd);
    }
    out.close();
    return cnt;
  }
}




