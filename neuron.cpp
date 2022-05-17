#include "neuron.hpp"

Neuron::Neuron(unsigned int numOutputs, unsigned int index)
{
    for(unsigned int i = 0; i < numOutputs; i++)
    {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }

    this->index = index;
}

void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;

    for(unsigned int i = 0; i < prevLayer.size(); i++)
    {
        sum += prevLayer[i].getOutputVal() *
                prevLayer[i].outputWeights[index].weight;
    }

    outputVal = activationFunction(sum);
}

double Neuron::activationFunction(double x)
{
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - outputVal;
    gradient = delta * activationFunction(outputVal);
}

void Neuron::calcHiddenGradients(Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * activationFunction(outputVal);
}

double Neuron::sumDOW(Layer &nextLayer)
{
    double sum = 0.0;

    for(unsigned int n = 0; n < nextLayer.size() - 1; n++)
    {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for(unsigned int n = 0; n < prevLayer.size(); n++)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;

        double newDeltaWeight = 
                // Individual input, magnified by the gradient and train rate
                eta
                * neuron.getOutputVal()
                * gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;
        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}