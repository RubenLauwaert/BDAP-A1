#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
private:
    int num_hashes_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<int> hashSeeds_;
    std::map<int, std::vector<double>> weightsMatrix_;

    std::vector<int> initHashSeeds(int num_hashes){
        std::vector<int> hash_seeds = std::vector<int>(num_hashes,0);
        // Generate a random 16-bit number
        for(int i = 0 ; i < num_hashes ; i++){
            int rand = std::rand() % (65536 - 1);
            hash_seeds[i] = rand;
        }
        return hash_seeds;
    }

    std::map<int,std::vector<double>> initWeightsMatrix(std::vector<int> hash_seeds){
        std::map<int,std::vector<double>> newBucketMatrix;
        std::vector<double> initCountVector = std::vector<double>((1 << log_num_buckets_),0.0);
        for(auto hash_seed : hash_seeds){
            newBucketMatrix[hash_seed] = initCountVector;
        }
        return newBucketMatrix;
    }

    double getMedianWeight_(std::string_view n_gram) const {
        std::vector<double> weights;
        // Iterate over map
        for(auto hash_seed : hashSeeds_){
            int bucketIndex = this->hash(n_gram,hash_seed) % (1 << log_num_buckets_);
            weights.push_back(weightsMatrix_.at(hash_seed)[bucketIndex]);
        }

        // Return Median of weights
        return getMedian(weights);
    }

    double getMedian(std::vector<double> weights) const{
        int length = weights.size();
        int median;

        // Sort Vector
        std::sort(weights.begin(), weights.end());
        if (length % 2 == 0)
        {
            // vector has even length, so we need to average the middle two elements
            median = (weights[length / 2 - 1] + weights[length / 2]) / 2;
            // median is now the average of the two middle elements
        }
        else
        {
            // vector has odd length, so the median is the middle element
            median = weights[length / 2];
            // median is now the middle element
        }

        return median;

    }

    int transformTargetValue(bool isSpam) const{
        int targetValue;
        (isSpam) ? (targetValue = 1) : (targetValue = -1);
        return targetValue;
    }

public:

    PerceptronCountMin(int num_hashes, int log_num_buckets, double learning_rate) :
    num_hashes_(num_hashes),
    log_num_buckets_(log_num_buckets),
    learning_rate_(learning_rate),
    bias_(0.0),
    hashSeeds_(initHashSeeds(num_hashes)),
    weightsMatrix_(initWeightsMatrix(hashSeeds_))
    {}

    void update_(const Email& email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        int o_d = transformTargetValue(this->classify(email));
        int t_d = transformTargetValue(email.is_spam());
        while(!iter.is_done()){
            std::string_view n_gram = iter.next();
            for(auto hash_seed : hashSeeds_){
                int bucketIndex = this->hash(n_gram,hash_seed) % (1 << log_num_buckets_);
                // Calculate Delta w_i
                double x_id = t_d;
                double delta_w_i = learning_rate_ * (t_d - o_d) * x_id;
                // Update weight in weightsMatrix
                weightsMatrix_[hash_seed][bucketIndex] += delta_w_i;
            }
        }
    }

    double predict_(const Email& email) const
    {
        double activationValue = 0;

        EmailIter iter = EmailIter(email, this->ngram_k);
        int targetValue = transformTargetValue(email.is_spam());
        while(!iter.is_done()){
            // Parse n_gram
            std::string_view n_gram = iter.next();
            // Get Median Weight
            double w_i = getMedianWeight_(n_gram);

            // Update Activation Value
            activationValue += w_i * targetValue;

        }


        return activationValue - bias_;
    }
};

} // namespace bdap
