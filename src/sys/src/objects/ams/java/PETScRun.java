/*$Id: PETScRun.java,v 1.6 2000/11/03 22:23:16 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;
import java.util.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class PETScRun extends java.applet.Applet
{
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3;
  Hashtable        systems[];

  JPanel      tpanel;

    Choice    arch;
    Choice    dir;
    Choice    example;
    Choice    np;

  JTextArea   opanel;

  public void init() {

    try {
      Socket sock = new Socket(this.getParameter("server"),2000);
      sock.setSoLinger(true,5);

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("Loadsystems","Yes");
      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      ObjectInputStream os = new ObjectInputStream(sock.getInputStream());
      systems            = new Hashtable[4];
      systems[MACHINE]   = (Hashtable) os.readObject();
      systems[DIRECTORY] = (Hashtable) os.readObject();
      systems[MAXNP]     = (Hashtable) os.readObject();
      systems[EXAMPLES]  = (Hashtable) os.readObject();
      sock.close();
    } catch (java.io.IOException ex) {;}
      catch (ClassNotFoundException ex) {    System.out.println("no class");}
 
    this.setLayout(new FlowLayout());

    tpanel = new JPanel(new GridLayout(2,4));
    this.add(tpanel, BorderLayout.NORTH);
      
      arch = new Choice();
      Enumeration keys = systems[MAXNP].keys();
      while (keys.hasMoreElements()) {
        arch.add((String)keys.nextElement());
      }
      tpanel.add(arch);
        
      dir = new Choice();
      dir.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                            System.out.println("Called choie");
                            Choice choice = (Choice) e.getItemSelectable();
                            setexamples(example,choice.getSelectedItem());
		            System.out.println("Called ch");}
	                  });
      keys = systems[EXAMPLES].keys();
      while (keys.hasMoreElements()) {
        dir.add((String)keys.nextElement());
      }
      tpanel.add(dir);

      example = new Choice();
      setexamples(example,(String)systems[EXAMPLES].keys().nextElement());
      tpanel.add(example);


      np = new Choice();
      np.add("1");
      np.add("2");
      tpanel.add(np);

      JButton rbutton = new JButton("Run");
      tpanel.add(rbutton);
      rbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("mpirun");
        }
      }); 

      JButton mbutton = new JButton("Make");
      tpanel.add(mbutton);
      mbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("make");
        }
      }); 

    opanel = new JTextArea(30,60);
    this.add(new JScrollPane(opanel), BorderLayout.NORTH); 
    opanel.setLineWrap(true);
    opanel.setWrapStyleWord(true);
  }

  /*
      Fills in the examples pull down menu based on the directory selected
  */
  public void setexamples(Choice example,String exin) {
    ArrayList ex = (ArrayList)systems[EXAMPLES].get(exin);
    Iterator its = ex.iterator();
    example.removeAll();
    while (its.hasNext()) {
      example.add((String)its.next());
    }
  }

  public void stop() {
    System.out.println("Called stop");
  }

  public void runprogram(String what)
  {

    try {
      Socket sock = new Socket(this.getParameter("server"),2000);
      sock.setSoLinger(true,5);
      InputStream sstream = sock.getInputStream();

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("PETSC_ARCH",arch.getSelectedItem());
      properties.setProperty("DIRECTORY",dir.getSelectedItem());
      properties.setProperty("EXAMPLE",example.getSelectedItem());
      properties.setProperty("NUMBERPROCESSORS",np.getSelectedItem());
      properties.setProperty("COMMAND",what);

      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      /* get output and print to screen */
      InputStreamReader stream = new InputStreamReader(sstream);
      char[] results = new char[128];
      int    fd      = 0,cnt = 0;

      opanel.setText(null);
      while (true) {
        fd  = stream.read(results);
        if (fd == -1) {break;}
        opanel.append(new String(results,0,fd));
	cnt += fd;
      }
    } catch (java.io.IOException ex) {;}
  }
}




