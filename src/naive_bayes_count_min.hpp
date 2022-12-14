#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include <map>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {

private:

    int num_hashes_;
    double threshold_;
    std::vector<int> hashSeeds_;
    int amt_buckets_;
    std::map<int,std::vector<int>> bucketMatrixSpam_;
    std::map<int,std::vector<int>> bucketMatrixHam_;


    std::vector<int> initHashSeeds(int num_hashes){
        std::cout << "Init Hash Seeds" << '\n';
        std::vector<int> hash_seeds = std::vector<int>(num_hashes,0);
        // Generate a random 16-bit number
        for(int i = 0 ; i < num_hashes ; i++){
            hash_seeds[i] = i;
            std::cout << hash_seeds[0];
            std::cout << "i : ";
        }
        //std::cout << "hashSeeds : " << hash_seeds[0];
        return hash_seeds;
    }


    std::map<int,std::vector<int>> initBucketMatrix(std::vector<int> hash_seeds){
        std::map<int,std::vector<int>> newBucketMatrix;
        std::vector<int> initCountVector = std::vector<int>(amt_buckets_,1);
        for(auto hash_seed : hash_seeds){
            newBucketMatrix[hash_seed] = initCountVector;
        }
        return newBucketMatrix;
    }


public:
    // TODO initialize your fields in the constructor
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold) :
        num_hashes_(num_hashes),
        amt_buckets_(std::pow(2,log_num_buckets)),
        // hashSeeds_(initHashSeeds(num_hashes)),
        bucketMatrixHam_(initBucketMatrix(hashSeeds_)),
        bucketMatrixSpam_(initBucketMatrix(hashSeeds_))
     {
        // Set basic arguments
        // num_hashes_ = num_hashes;
        // amt_buckets_ = std::pow(2,log_num_buckets);
        this->hashSeeds_ = initHashSeeds(num_hashes);
        this->threshold = threshold;
        // Generate vector of different hash seeds (hashSeeds_.length() == num_hashes)
        // hashSeeds_ = initHashSeeds(num_hashes);
        // Initialize bucket Matrices
        // bucketMatrixHam_ = initBucketMatrix(hashSeeds_);
        // bucketMatrixSpam_ = initBucketMatrix(hashSeeds_);

    }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            for(auto hash_seed : hashSeeds_){
                int bucketIndex = this->hash(iter.next(), hash_seed) % amt_buckets_;
                (email.is_spam()) ? ({bucketMatrixSpam_[hash_seed][bucketIndex] += 1;}) :
                ({bucketMatrixHam_[hash_seed][bucketIndex] += 1;}); 
            }        
        }
        //int hash_seed = hashSeeds_[0];
        //std::cout << "Bucket Matrix : Seed 1 : " << hashSeeds_.size();
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }
};

} // namespace bdap
