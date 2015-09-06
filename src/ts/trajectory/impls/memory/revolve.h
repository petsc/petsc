/***************************************************************************
 *   Copyright (C) 2006 by Philipp Stumm   *
 *   stumm@NBTW13   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#ifndef _REVOLVE_H
#define _REVOLVE_H

#include <vector>

using namespace std;

#define checkup 2000
/*#define checkup 400*/
#define repsup 1000
/*#define repsup 64*/
#define MAXINT 2147483647

namespace ACTION
{
	enum action { advance, takeshot, restore, firsturn, youturn, terminate, error} ;
}



/** \enum action
Through an encoding of its return value REVOLVE asks the calling program to perform one of these 'actions', which we will 
   refer to as 'advance', 'takeshot', 'restore', 'firsturn' and 'youturn'  .
   There are two other return values, namely 'terminate'   and     'error'
   which indicate a regular or faulty termination of the calls 
   to REVOLVE.
   The action 'firsturn' includes a 'youturn', in that it requires  
     -advancing through the last time-step with recording 
      of intermediates                                              
     -initializing the adjoint values (possibly after               
      performing some IO)                                           
     -reversing the last time step using the record just written    
   The action 'firsturn' is obtained when the difference FINE-CAPO  
   has been reduced to 1 for the first time. 
*/

/** \class Checkpoint
The class Checkpoint contains the two vectors ch and ord_ch. All checkpoints are stored in ch and the sequence of inizies can be found in ord_ch.
\brief Basic class for all Checkpointing schedules
*/


class Checkpoint
{
	public:

	Checkpoint(int s) { snaps=s; ch.reserve(snaps); number_of_writes.reserve(snaps); }
	
	vector <int> ch;
	vector <int> ord_ch;
	vector <int> number_of_writes;
	void init_ord_ch() { ord_ch.reserve(snaps); }
	
	
	~Checkpoint() { ch.clear(); ord_ch.clear(); number_of_writes.clear(); };

	private:

	int snaps;
};



/** \class Schedule
Schedule is the basic class of all Checkpointing schedules. This class stores the number of snaps and the checkpoints that have to be stored.
\brief Basic class for all Checkpointing schedules
*/

class Schedule
{
	public:

	/** This constructor needs a given array. This array is the initialisation of ch.
	*/
	Schedule(int sn,Checkpoint *c);
	Schedule(int sn) { snaps=sn; }
	
	/** This function does not do anything but must be derived
	*/
	virtual ACTION::action revolve() {};  
	/** The necessary number of forward steps without recording is calculated by the function 
                      NUMFORW(STEPS,SNAPS)                          
	STEPS denotes the total number of time steps, i.e. FINE-CAPO     
   	during the first call of REVOLVE. When SNAPS is less than 1 an    
   	error message will be given and -1 is returned as value.  */
	
	int numforw(int steps, int snaps);
	/** This function is virtual.*/
	virtual int get_capo(){};	
	/** This function is virtual.*/
	virtual int get_fine(){};	
	/** This function is virtual.*/
	virtual int get_check() {};	
	/** This function is virtual.*/
	virtual void set_fine(int f){};  
	/** This function is virtual.*/
	virtual void set_capo(int c) {}; 
	/** This function is virtual.*/
	int get_snaps() { return snaps; }
	/** This function returns the pointer of a Checkpoint class.
//	*/
	Checkpoint *get_CP() { return checkpoint; }
//	vector <int> get_ch()   { return ch; }
	/** This function returns the number of advances. */
	int get_advances() { return advances; }
	/** This function returns the number of takeshots. */
	int get_shots() { return takeshots; }
	/** This function returns the number of commands. */
	int get_commands() { return commands; }
	/** This function returns the info.*/
	virtual int get_info(){ return info;}	
	
	
	~Schedule() { /*delete[] ch;*/ }

	protected:
	Checkpoint *checkpoint;   
	/** Number of snaps used are stored in snaps*/
	int snaps;   
	int advances, takeshots, commands;
	int info;
};


/** \class Online
Online inherates the basic class Schedule. This class is the basic class for all Online Checkpointing schedules. 
\brief Basic class for all Online Checkpointing Schedules
*/

class Online : public Schedule
{
	public:

	/** This constructor needs the number of snaps for the Schedule Constructor and a bool parameter. */
	Online(int sn,Checkpoint *c,bool o);
	/** This is the Copy Constructor*/
	Online(Online &o);

	virtual ACTION::action revolve() {};
	/** This function returns the index of the last stored checkpoint.*/
	int get_check() { return check; }
	int get_capo()  { return capo; }
	virtual int get_fine() {};
	/** This function returns the variable output.*/
	bool get_output()      { return output; }
	void set_capo(int c) { capo=c; }

	~Online();

	protected:

	/** check is last stored checkpoint*/
	int check;
	/** output=true means that special information is printed on the screen */
	bool output;
	/** capo is the temporary fine */
	int capo;

};



/**
\class Online_r2 
This class creates a checkpoint schedule that is optimal over the range [0,(snaps+2)*(snaps+1)/2].
\brief class for Online Checkpointing Schedules for r=2
*/

class Online_r2 : public Online
{
	public:

	/** This constructor does not do anything. It only calls the basic constructor of Online.*/
	Online_r2(int sn,Checkpoint *c,bool o);
	/** Copy Constructor*/
	Online_r2(Online &o);

	/** The function revolve always returns the action the user has to do. During the forward integration only the actions advance and takeshot are returned. If the capo exceeds the maximal number of time steps, i.e. (snaps+2)*(snaps+1)/2, this function returns terminate.*/
	ACTION::action revolve();
	int get_check() { return check; }
	int get_capo()  { return capo; }
	int get_fine()  { return -1; }
	bool get_output()      { return output; }

	~Online_r2();

	private:

	int oldcapo_o,offset,incr,iter,t,oldind,old_f,ind;
	/**num_rep[i] is the repetion number*/
	vector <int> num_rep;

};


/**
\class Online_r3 
This class creates a checkpoint schedule that is quasi optimal over the range [(snaps+2)*(snaps+1)/2+1,(snaps+3)*(snaps+2)*(snaps+1)/6].
\brief class for Online Checkpointing Schedules for r=3
*/


class Online_r3 : public Online 
{
	public:

	/** This constructor needs an Online class. Usually this constructor is called after revolve of the class Online_r2 has returned terminate. */
	Online_r3(int sn,Checkpoint *c);
	Online_r3(Online_r3 &o) ;

	/** This function returns advance or takeshot during the forward integration. */
	ACTION::action revolve();

	/** This function returns the capo.*/
	int get_capo()  { return capo; }
	/** This function returns -1 because the end is not still reached.*/
	int get_fine()  { return -1; }

	/** This function returns the index of the checkpoint that can be replaced (Replacement condition). The argument number means that all checkpoints that fulfill the Replacement condition before number cannot be used.*/ 
	int choose_cp(int number);
	/** This function renews the array tdiff after a checkpoint was replaced*/
	void tdiff_akt();
	/** This function renews the array ord_ch of the class Online after a checkpoint was replaced*/
	void akt_cp();
	
	
	int get_tdiff_end(int i) { return tdiff_end[i]; }

	~Online_r3();

	protected:

	
	/** forward is number of time steps to advance*/
	int forward;
	/** ind_now is the temporary index of the checkpoints of the array ch3 that have to be taken*/ 
	int ind_now;
	/** cp is the Index of the checkpoint that can be replaced. To find the checkpoint out that can be replaced cp must be used in connection with ord_ch*/
	int cp;
	/** ch3 is the array of the final checkpoint distribution for r=3. These checkpoints must be taken!*/
	vector <int> ch3;
	/** tdiff[i] is the difference between the i.th and the i-1.th checkpoint*/
	vector <int> tdiff;
	/** tdiff_end is the array of differences for the final checkpoint distribution for r=3.*/
	vector <int> tdiff_end;
	/** cp_fest[i] defines if the i.th checkpoint can be overwritten or not.*/
	vector <bool> cp_fest;
};




/**
\class Arevolve
This class creates a checkpoint schedule that uses an heuristic approach for a checkpoint to be replaced. This class is usually called after another Onlie Checkpointing class has exceeds its maximum number of time steps.
\brief class for Online Checkpointing Schedules 
*/

class Arevolve : public Online
{
	public:

	/** Constructor that is called after an Online Checkpointing class has exceeded the max. number of time steps. */
	Arevolve(int sn,Checkpoint *c);
	/** Copy Constructor */
	Arevolve(Arevolve &o);

	/** This function returns the number of advance steps. This number depends on the the number of steps and snaps.*/
	int tmin(int steps, int snaps);
	/** This function calculates the number of advance steps for a given Checkpoint distribution. This distribution is stored in Schedule. */
	int sumtmin();
	/** This function will look for a checkpoint to be replaced for the condition of arevolve.*/
	int mintmin();
	/** This function returns the momental fine. */
	int get_fine() { return fine; }
	/** This function renews the array ord_ch of the class Online after a checkpoint was replaced.*/
	void akt_cp(int cp);
	
	void set_fine(int f) { fine=f; }

	/** This function returns advance or takeshot during the forward integration. */
	enum ACTION::action revolve();

	~Arevolve() { }
	
	private:

	int checkmax;
	int fine,oldfine,newcapo,oldcheck,oldcapo;

};



/** \class Offline
Offline inherates the basic class Schedule. This class manages all Offline Checkpointing schedules. This class allows optimal Offline Checkpointing strategies if the number of time steps is a-priori known. This class also manages the optimal reversal of schedules resulting from Online Checkpointing
\brief Class for all Offline Checkpointing schedules
*/

class Offline : public Schedule
{
	public:

	/** This is the standard constructor that will be called if the number of time steps is a-priori known. */
	Offline(int st,int sn,Checkpoint *c);  
	/** This constructor will be called for the optimal Reversal of a schedule resulting from Online Checkpointing. */
	Offline(int sn,Checkpoint *c,Online *o,int f);	
	/** This constructor does not do anything and will usually not be called. */
	Offline(Schedule *o) ;    
	/** CopyConstructor */
	Offline(Offline &o);	 

	/**  Since REVOLVE involves only a few integer operations its run-time is truly negligible within any nontrivial application. The parameter SNAPS is selected by the user (possibly with the help of the routines EXPENSE and ADJUST described below ) and remains unchanged throughout. The pair (CAPO,FINE) always represents the initial and final state of the subsequence of time steps currently being traversed backwards. The conditions CHECK >= -1 and CAPO <= FINE are necessary and sufficient for a regular response of REVOLVE. If either condition is violated the value 'error' is returned. When CHECK =-1 and CAPO = FINE  then 'terminate' is returned as action value. This combination necessarily arises after a sufficiently large number of calls to REVOLVE, which depends only on the initial difference FINE-CAPO. The last parameter INFO determines how much information about the actions performed will be printed. When INFO =0 no  information is sent to standard output. When INFO > 0 REVOLVE produces an output that contains a prediction of the number of forward steps and of the factor by which the execution will slow down. When an error occurs, the return value of INFO contains information about the reason: INFO = 10: number of checkpoints stored exceeds CHECKUP, increase constant CHECKUP and recompile. INFO = 11: number of checkpoints stored exceeds SNAPS, ensure SNAPS greater than 0 and increase initial FINE. INFO = 12: error occurs in NUMFORW. INFO = 13: enhancement of FINE, SNAPS checkpoints stored,SNAPS must be increased. INFO = 14: number of SNAPS exceeds CHECKUP, increase constant CHECKUP and recompile. INFO = 15: number of REPS exceeds REPSUP, increase constant REPSUP and recompile. */

	ACTION::action revolve();
	
	int get_check() { return check; }
	int get_capo() { return capo; }
	int get_fine() { return fine; }
	int get_snaps() { return snaps; }
	int get_commands() { return commands; }
	int get_steps()    { return steps; }
	bool get_online()  { return online; }
	vector <int> get_num_ch() { return num_ch; }
	int get_num_ch(int i) { return num_ch[i]; }

	void set_fine(int f) { fine=f;}
	void set_capo(int c) { capo=c; }
	
	
	~Offline() { };

	private:

	int check, steps, oldsnaps, oldfine, capo, fine, turn,ind	;
	vector <int> num_ch;
	bool online,output;
};


/**
\class Revolve 
This class manages to create Schedules for Online or Offline Checkpointing. The user only needs to tell which Checkpointing Procedure he wants to use
\brief class to create Checkpoint Schedules
*/



class Revolve
{
	public:
	/** Constructor for Offline-Checkpointing*/
	Revolve(int st,int sn,int inf, int rank);
	/** Constructor for Online-Checkpointing */
	Revolve(int sn,int inf);

	/**The calling sequence is REVOLVE(CHECK,CAPO,FINE,SNAPS,INFO) with the return value being one of the actions to be taken. The calling parameters are all integers with the following meaning: CHECK - number of checkpoint being written or retrieved. CAPO - beginning of subrange currently being processed. FINE - end of subrange currently being processed.SNAPS - upper bound on number of checkpoints taken. INFO - determines how much information will be printed and contains information about an error occured  */

 
	ACTION::action revolve(int* check,int* capo,int* fine,int snaps,int* info);
	//ACTION::action revolve();

	/**The function ADJUST(STEPS) is provided. It can be used to determine a value of SNAPS so that the increase in spatial complexity equals approximately the increase in temporal complexity. For that ADJUST computes a return value satisfying SNAPS ~= log_4 (STEPS) because of the theory developed in the paper mentioned above. */
	int adjust(int steps);
	/** The auxiliary function MAXRANGE(SNAPS,REPS) returns the integer (SNAPS+REPS)!/(SNAPS!REPS!) provided SNAPS >=0, REPS >= 0. Otherwise there will be appropriate error messages and the value -1 will be returned. If the binomial expression is not representable as a  signed 4 byte integer, greater than 2^31-1, this maximal value is returned and a warning message printed.*/
	int maxrange(int ss, int tt);
	/** To choose an appropriated value of SNAPS the function EXPENSE(STEPS,SNAPS) estimates the run-time factor incurred by REVOLVE for a particular value of SNAPS. The ratio NUMFORW(STEPS,SNAPS)/STEPS is returned. This ratio corresponds to the run-time factor of the execution relative to the run-time of one forward time step.*/
	double expense(int steps, int snaps);
	/** The necessary number of forward steps without recording is calculated by the function NUMFORW(STEPS,SNAPS). STEPS denotes the total number of time steps, i.e. FINE-CAPO during the first call of REVOLVE. When SNAPS is less than 1 an error message will be given and -1 is returned as value.*/
	int numforw(int steps, int snaps) { return f->numforw(steps,snaps); }
	/** Turn starts the reversal of the schedule. This means that Online Checkpointing is finished. */
	void turn(int fine);
	void print_number_of_writes();
	void write_number_of_writes();
	int getadvances() { return f->get_advances(); }  
	int getcheck() { return f->get_check(); }
	int getcapo()  { return f->get_capo(); }
	int getfine()  { return f->get_fine(); }
	void setcapo(int c) {f->set_capo(c); }
	void setfine(int ff) { f->set_fine(ff); }
	
	int get_r();
	
	~Revolve() { delete f, delete checkpoint; }

	private:

	int check,capo,fine,snaps,info,steps,r;
	Schedule *f;
	bool online;
	Checkpoint *checkpoint;
};

#endif
