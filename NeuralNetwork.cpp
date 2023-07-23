#include "NeuralNetwork.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

Neuron::Neuron(int n_weights) {
    this->initWeights(n_weights);
    m_nWeights = n_weights;
    m_activation = 0;
    m_output = 0;
    m_delta = 0;
}

Neuron::~Neuron() {}

void Neuron::initWeights(int n_weights) {
    for(int w=0; w<n_weights; ++w)
        m_weights.push_back(
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)
        );
}

void Neuron::activate(std::vector<float> inputs) {
    m_activation = m_weights[m_nWeights - 1]; // bias

    for(size_t i=0; i < m_nWeights-1; ++i)
        m_activation += m_weights[i] * inputs[i];
}

void Neuron::transfer() {
    m_output = 1.0f / (1.0f + std::exp(-m_activation));
}


