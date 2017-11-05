/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
        //

    num_particles = 100;
    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
	Particle p;
        p.id = i;
	p.x = dist_x(gen);
	p.y = dist_y(gen);
	p.theta = dist_theta(gen);	 
	p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
    default_random_engine gen;
    for (int i = 0; i < num_particles; ++i) {
        double x_m, y_m, theta_m; //new mean value
        double x_p, y_p, theta_p; //previous value
        x_p = particles[i].x;
        y_p = particles[i].y;
        theta_p = particles[i].theta;
        
        x_m = x_p + velocity/yaw_rate*(sin(theta_p + yaw_rate*delta_t)
                - sin(theta_p));
        y_m = y_p + velocity/yaw_rate*(cos(theta_p)
                - cos(theta_p + yaw_rate*delta_t));
        theta_m = theta_p + yaw_rate*delta_t;

        normal_distribution<double> dist_x(x_m, std_pos[0]);
        normal_distribution<double> dist_y(y_m, std_pos[1]);
        normal_distribution<double> dist_theta(theta_m, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < predicted.size(); ++i) {
        LandmarkObs lm_p = predicted[i];
        double dist = 10000.0; //very large number
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs lm_o = observations[j];
            double dist_new = sqrt(pow((lm_p.x-lm_o.x),2)+pow((lm_p.y-lm_o.y),2));
            if (dist > dist_new) {
                dist = dist_new;
                predicted[i].x = observations[j].x;
                predicted[i].y = observations[j].y;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
        //
        //
    double x_part, y_part, x_obs, y_obs, theta, x_map, y_map;
    for (int i = 0; i < num_particles; ++i) {
        x_part = particles[i].x;
        y_part = particles[i].y;
        theta = particles[i].theta;

        for (int j = 0; j < observations.size(); ++j) {
            //Transformation
            LandmarkObs pred;
            x_obs = observations[j].x;
            y_obs = observations[j].y;
            pred.x= x_part+(cos(theta)*x_obs)-(sin(theta)*y_obs);
            pred.y= y_part+(sin(theta)*x_obs)+(cos(theta)*y_obs);


            //Association
            int nearest_lm_id;
            double dist = 10000.0; //very large number
            for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
                Map::single_landmark_s map_lm = map_landmarks.landmark_list[k];
                double dist_new = sqrt(pow((pred.x-map_lm.x_f),2)+pow((pred.y-map_lm.y_f),2));
                if (dist > dist_new) {
                    dist = dist_new;
                    nearest_lm_id = map_lm.id_i;
                }
            }

            //Update weights
            double mu_x = map_landmarks.landmark_list[nearest_lm_id].x_f;
            double mu_y = map_landmarks.landmark_list[nearest_lm_id].y_f;

            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];
            double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));
            double exponent = (pow((pred.x - mu_x),2))/(2 * pow(sig_x,2)) + 
                (pow((pred.y - mu_y),2))/(2 * pow(sig_y,2));

            particles[i].weight *= gauss_norm * exp(-exponent);
        }
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        std::default_random_engine gen;
        std::vector<double> p_wts;
        for (int i = 0; i < num_particles; ++i) {
            p_wts.push_back(particles[i].weight);
        }
        std::discrete_distribution<> dd(p_wts.begin(),p_wts.end());

        std::vector<Particle> new_particles;

        for (int i = 0; i < num_particles; ++i) {
            int picked_id = dd(gen);
            new_particles.push_back(particles[picked_id]);
        }
        particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
