#ifndef Neuron_hpp
#define Neuron_hpp
#include <vector>
#include <cstdlib>
#include <cmath>
#include "type.hpp"
using namespace std;




class Neuron{
public:
    Neuron(unsigned int numOutputs, unsigned int myIndex);
    void setOutputData(double data);
    double getOutputData(void)const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGrad(double targetData);
    void calcHiddenGrad(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    
private:
    static double eta;
    static double alpha;
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void){ return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputData;
    vector<Connection> m_outputWeights;
    unsigned int m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15; //general used:0.15
double Neuron::alpha = 0.5;//general used:0.5

//Body of constructor
Neuron::Neuron(unsigned int numOutputs, unsigned int myIndex){
    for (unsigned int i = 0; i < numOutputs; ++i){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

//Body of Set output data
void Neuron::setOutputData(double data){
    m_outputData = data;}

//Body of Get output data
double Neuron::getOutputData(void)const{
    return m_outputData;}

//Body of feedForward
void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    
    //Sum the previous layer's outputs (which are our inputs)
    //Include the bias node from the previous layer
    
    for (unsigned int i = 0; i < prevLayer.size(); ++i){
        sum += prevLayer[i].getOutputData() * prevLayer[i].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputData = Neuron::activationFunction(sum);
    
}

//Body of Calculation of output gradient
void Neuron::calcOutputGrad(double targetData){
    double delta = targetData - m_outputData;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputData);
}

//Body of Calculation of hidden gradient
void Neuron::calcHiddenGrad(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputData);
    
}

//Body of updating input weights
void Neuron::updateInputWeights(Layer &prevLayer){
    //The weights to be updated are in the Connection Container
    //In the neurons in the preceding layer
    for (unsigned int i = 0; i < prevLayer.size(); ++i){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;
        double finalWeight;
        //Individual input, magnified by the gradient and train rate
        //eta = learning rate, alpha = momentum
        double newDeltaWeight = eta * neuron.getOutputData() * m_gradient + alpha * oldDeltaWeight;
        finalWeight += newDeltaWeight;
        //if (i ==prevLayer.size()-1){
            cout<<"Weights of variable: "<<finalWeight<<endl;
        //}
        neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}



//Body of activation function and its derivative
double Neuron::activationFunction(double x){
    // using bipolar sigmoid function - range from -1 to 1
    return (2/(1+ exp(-x))-1);
}

double Neuron::activationFunctionDerivative(double x){
    double a;
    a = 2/(1+exp(-x))-1;
    return 0.5*(1+a)*(1-a);
}

//Body of sum dow
double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    
    //Sum our contributions of the errors at the nodes we feed
    for (unsigned int i = 0; i < nextLayer.size() - 1; ++i){
        sum += m_outputWeights[i].weight * nextLayer[i].m_gradient;
    }
    return sum;
}




#endif /* Neuron_hpp */
