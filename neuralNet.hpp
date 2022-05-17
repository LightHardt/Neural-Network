#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include "neuron.hpp"
#include <iostream>
#include <cassert>

using std::cout;
using std::endl;

typedef std::vector<Neuron> Layer;

    class NeuralNet{
        private:
            std::vector<Layer> layers;
            double error;
            double recentAverageError;
            double recentAverageSmoothingFactor = 100;
            
        public:
            NeuralNet(std::vector<unsigned> &topology);
            void feedForward(std::vector<double> &inputVals);
            void backProp(std::vector<double> &targetVals);
            void getResults(std::vector<double> &resultVals);
            double getRecentAverageError() { return recentAverageError; }
    };

#endif