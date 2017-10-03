#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double P_last_x = 0;
double P_last_y = 0;
double v_x_est_last = 0;
double v_y_est_last = 0;
int x_direction = 0;
int y_direction = 0;

double frand() {
    return 2*((rand()/(double)RAND_MAX) - 0.5);
}

double get_distance(int x, int y, int w, int h,
				 int x2, int y2, int w2, int h2)
{
	double center_x;
	double center_y;
	double center_x2;
	double center_y2;
	double distance;

	center_x = x + w / 2;
	center_y = y + y / 2;
	center_x2 = x2 + w2 / 2;
	center_y2 = y2 + y2 / 2;
	distance = sqrt((center_x2 - center_x) * (center_x2 - center_x) + (center_y2 - center_y) * (center_y2 - center_y));
	return distance;
}

// return 1 if the bb go where the x increases
// return -1 if go where the x decreases
// return 0 if the bb does not go anywhere
int get_x_direction(int x, int y, int w, int h,
				 int x2, int y2, int w2, int h2)
{
	double center_x;
	double center_x2;

	center_x = x + w / 2;
	center_x2 = x2 + w2 / 2;

	if (center_x2 - center_x > 0)
	{
		x_direction = 1;
	}
	else if (center_x2 - center_x < 0)
	{
		x_direction = -1;
	}
	else{
		x_direction = 0;
	}
}

int get_y_direction(int x, int y, int w, int h,
				 int x2, int y2, int w2, int h2)
{
	double center_y;
	double center_y2;

	center_y = y + w / 2;
	center_y2 = y2 + w2 / 2;

	if (center_y2 - center_y > 0)
	{
		y_direction = 1;
	}
	else if (center_y2 - center_y < 0)
	{
		y_direction = -1;
	}
	else{
		y_direction = 0;
	}
}

double X_kalman_filter(double x_distance) 
{
	// double x_est_last = 0;
    // double P_last = 0;
    //the noise in the system
    double Q = 0.022;
    double R = 0.617;
    
    double K;
    double P;
    double P_temp;
    double x_temp_est;
    double x_est;
    double z_measured; //the 'noisy' value we measured 
    double z_real = x_distance; //the ideal value we wish to measure
    
    srand(0);
    
    //initialize with a measurement
    if (x_est_last == 0)
    {
    	x_est_last = z_real + frand()*0.09;
    }
    // x_est_last = z_real + frand()*0.09;
    
    //do a prediction
    x_temp_est = x_est_last;
    P_temp = P_last + Q;
    //calculate the Kalman gain
    K = P_temp * (1.0/(P_temp + R));
    //measure
    z_measured = z_real + frand()*0.09; //the real measurement plus noise
    //correct
    x_est = x_temp_est + K * (z_measured - x_temp_est); 
    P = (1- K) * P_temp;
    //we have our new system
    
    // printf("Ideal    position: %6.3f \n",z_real);
    // printf("Mesaured position: %6.3f [diff:%.3f]\n",z_measured,fabs(z_real-z_measured));
    // printf("Kalman   position: %6.3f [diff:%.3f]\n",x_est,fabs(z_real - x_est));
    
    //update our last's
    P_last_x = P;
    v_x_est_last = x_est;
}


double Y_kalman_filter(double y_distance)  
{
	// double x_est_last = 0;
    // double P_last = 0;
    //the noise in the system
    double Q = 0.022;
    double R = 0.617;
    
    double K;
    double P;
    double P_temp;
    double y_temp_est;
    double y_est;
    double z_measured; //the 'noisy' value we measured 
    double z_real = y_distance; //the ideal value we wish to measure
    
    srand(0);
    
    //initialize with a measurement
    if (y_est_last == 0)
    {
    	y_est_last = z_real + frand()*0.09;
    }
    // x_est_last = z_real + frand()*0.09;
    
    //do a prediction
    y_temp_est = y_est_last;
    P_temp = P_last + Q;
    //calculate the Kalman gain
    K = P_temp * (1.0/(P_temp + R));
    //measure
    z_measured = z_real + frand()*0.09; //the real measurement plus noise
    //correct
    y_est = y_temp_est + K * (z_measured - y_temp_est); 
    P = (1 - K) * P_temp;
    //we have our new system
    
    // printf("Ideal    position: %6.3f \n",z_real);
    // printf("Mesaured position: %6.3f [diff:%.3f]\n",z_measured,fabs(z_real-z_measured));
    // printf("Kalman   position: %6.3f [diff:%.3f]\n",x_est,fabs(z_real - x_est));
    
    //update our last's
    P_last_y = P;
    v_y_est_last = y_est;
}

double get_x_prediction(int frame_number, int x)
{
	if (v_x_est_last == 0)
	{
		return;
	}

	return x + v_x_est_last*frame_number*x_direction
}

double get_x_prediction(int frame_number, int y)
{
	if (v_y_est_last == 0)
	{
		return;
	}

	return y + v_y_est_last*frame_number*y_direction
}
