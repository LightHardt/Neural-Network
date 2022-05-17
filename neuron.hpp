#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cstdlib>
#include <cmath>

    class Neuron;
    typedef std::vector<Neuron> Layer;

    struct Connection{
        double weight;
        double deltaWeight;
    };

    class Neuron{
        private:
            double outputVal;
            unsigned int index;
            double gradient;
            double eta = 0.15; //Training Rate
            double alpha = 0.5; //multiplier of last weight change (momentum)
            std::vector<Connection> outputWeights;
            double randomWeight() { return rand() / double(RAND_MAX); }
            double activationFunction(double x);
            double activationFunctionDerivative(double x);
            double sumDOW(Layer &nextLayer);

        public:
            Neuron(unsigned int numOutputs, unsigned int index);
            void feedForward(Layer &prevLayer);
            void setOutputVal(double val) { outputVal = val; }
            double getOutputVal() { return outputVal; }
            void calcOutputGradients(double targetVal);
            void calcHiddenGradients(Layer &nextLayer);
            void updateInputWeights(Layer &prevLayer);

    };
#endif