/*$Id: PETScRun.java,v 1.9 2000/11/07 21:35:01 bsmith Exp bsmith $*/
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

  java.applet.AppletContext appletcontext;
  java.applet.Applet applet;

  JPanel      tpanel;
  JTextField options;

  Checkbox toptions;
  Checkbox tbopt;
  Checkbox snoop;

    Choice    arch;
    Choice    dir;
    Choice    example;
    Choice    np;

  JTextArea   opanel;

  boolean isviewsource = false;

  public void init() {
    appletcontext = getAppletContext();
    applet = this;
    try {
      System.out.println("parameter"+this.getParameter("server"));
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
    } catch (java.io.IOException ex) {ex.printStackTrace(); System.out.println("no systems");}
      catch (ClassNotFoundException ex) {    System.out.println("no class");}
 
    this.setLayout(new FlowLayout());

    tpanel = new JPanel(new GridLayout(3,4));
    this.add(tpanel, BorderLayout.NORTH);
      
      arch = new Choice();
      arch.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              Choice choice = (Choice) e.getItemSelectable();
                              setnp(np,choice.getSelectedItem());}
	                  });
      Enumeration keys = systems[MAXNP].keys();
      while (keys.hasMoreElements()) {
        arch.add((String)keys.nextElement());
      }
      tpanel.add(arch);
        
      dir = new Choice();
      dir.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              Choice choice = (Choice) e.getItemSelectable();
                              setexamples(example,choice.getSelectedItem());}
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
      setnp(np,(String)systems[MAXNP].keys().nextElement());
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

      JButton tbutton = new JButton("Test");
      tpanel.add(tbutton);
      tbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("maketest");
        }
      }); 

      JButton sbutton = new JButton("View source");
      tpanel.add(sbutton);
      sbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          isviewsource = true;
          runprogram("makehtml");
        }
      }); 

      toptions = new Checkbox("Set options graphically");
      tpanel.add(toptions);

      snoop = new Checkbox("Snoop on running program");
      tpanel.add(snoop);

      tbopt = new Checkbox("Compile debug version");
      tpanel.add(tbopt);

    options = new JTextField(50);
    this.add(options, BorderLayout.NORTH);

    opanel = new JTextArea(30,60);
    this.add(new JScrollPane(opanel), BorderLayout.NORTH); 
    opanel.setLineWrap(true);
    opanel.setWrapStyleWord(true);
  }

  /*
      Fills in the examples pull down menu based on the directory selected
  */
  public void setexamples(Choice example,String arch) {
    ArrayList ex = (ArrayList)systems[EXAMPLES].get(arch);
    Iterator its = ex.iterator();
    example.removeAll();
    while (its.hasNext()) {
      example.add((String)its.next());
    }
  }

  /*
     Fills in the number processors pull down menu based on the arch selected
  */
  public void setnp(Choice np,String arch){
    int onp = Integer.parseInt((String)systems[MAXNP].get(arch)),i;
    np.removeAll();
    for (i=1; i<=onp;i++) {
      np.add(""+i);
    }
  }

  public void stop() {
    System.out.println("Called stop");
  }

  public void runprogram(String what)
  {
    try {
      final Socket sock = new Socket(this.getParameter("server"),2000);
      sock.setSoLinger(true,5);
      final InputStream sstream = sock.getInputStream();

      /* construct properties to send to server */
      final Properties   properties = new Properties();
      properties.setProperty("PETSC_ARCH",arch.getSelectedItem());
      properties.setProperty("DIRECTORY",dir.getSelectedItem());
      properties.setProperty("EXAMPLE",example.getSelectedItem());
      properties.setProperty("NUMBERPROCESSORS",np.getSelectedItem());
      properties.setProperty("COMMAND",what);
      if (tbopt.getState()) {
        properties.setProperty("BOPT","g");
      } else {
        properties.setProperty("BOPT","O");
      }
      String soptions = options.getText();
      if (toptions.getState() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_options";
	try {
	  final URL url = new URL(""+urlb+"AMSPETScOptions.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"AMSOptions");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      if (snoop.getState() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_objects";
	try {
	  final URL url = new URL(""+urlb+"PETScView.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"PETScView");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      properties.setProperty("OPTIONS",soptions);
      System.out.println("getting back");

      (new Thread() {
        public void run() {
          try {
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
          opanel.append("----DONE-----");
          donerun();
        }
      }).start();
    } catch (java.io.IOException ex) {;}
  }

  public void donerun() 
  {
    System.out.println("done run");
    if (isviewsource) {
      String ex = example.getSelectedItem();
      URL    urlb = applet.getDocumentBase();
  
      try {
        String s = null;
        if (ex.endsWith("f90") || ex.endsWith("f")) s = "_F";
        else s = "_c";
        final URL url = new URL(""+urlb+"../"+dir.getSelectedItem()+"/"+ex+s+".html");  
        System.out.println("showing"+url);
        appletcontext.showDocument(url,"Source");
      } catch (MalformedURLException oops) { System.out.println("bad:showing"+urlb);;} 
    }
    isviewsource = false;
  }
}




