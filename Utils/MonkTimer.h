/**
 * MonkTimer.h
 *
 *   Created on: Apr 23, 2012
 * 		Author: jmonk
 */

#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>

class MonkTimer {
public:
	MonkTimer(const char* name) {
		this->name = name;
		runningTime = 0;
		count = 0;
		displayed = false;
	}

	~MonkTimer() {
		if (!displayed) {
			display();
		}
	}

	void display() {
		double ave = runningTime / count;
		printf("%s MonkTimer Report: \n\tCalled %g Times\n\tTook %gs on Ave\n\tTook %gs Total\n", name, (double)count, ave, runningTime);
		displayed = true;
	}

	void start() {
		clock_gettime(CLOCK_REALTIME, &t);
    	time1 = t.tv_sec + (t.tv_nsec / 1000000000.0);
	}

	void stop() {
		clock_gettime(CLOCK_REALTIME, &t);
    	time2 = t.tv_sec + (t.tv_nsec / 1000000000.0);
		count++;
		runningTime += time2 - time1;
	}

private:
	bool displayed;
	long long count;
	double time1;
	double time2;
	double runningTime;
	struct timespec t;
	const char* name;
	
};

#endif //__TIMER_H__
