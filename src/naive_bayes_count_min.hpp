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

    std::vector<int> hashSeeds_;
    int amt_buckets_;
    int amt_ngrams_spam_;
    int amt_ngrams_ham_;
    std::map<int,std::vector<int>> bucketMatrixSpam_;
    std::map<int,std::vector<int>> bucketMatrixHam_;


    std::vector<int> initHashSeeds(int num_hashes){
        std::vector<int> hash_seeds = std::vector<int>(num_hashes,0);
        // Generate a random 16-bit number
        for(int i = 0 ; i < num_hashes ; i++){
            int rand = std::rand() % (65536 - 1);
            hash_seeds[i] = rand;
        }
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
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold) :
            amt_buckets_(std::pow(2,log_num_buckets)),
            amt_ngrams_spam_(std::pow(2, log_num_buckets)),
            amt_ngrams_ham_(std::pow(2, log_num_buckets)),
            hashSeeds_(initHashSeeds(num_hashes)),
            bucketMatrixHam_(initBucketMatrix(hashSeeds_)),
            bucketMatrixSpam_(initBucketMatrix(hashSeeds_))
     {
        this->threshold = threshold;
    }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            std::string_view n_gram = iter.next();
            for(auto hash_seed : hashSeeds_){
                int bucketIndex = this->hash(n_gram, hash_seed) % amt_buckets_;
                (email.is_spam()) ? ({bucketMatrixSpam_[hash_seed][bucketIndex] += 1;}) :
                ({bucketMatrixHam_[hash_seed][bucketIndex] += 1;}); 
            }
            // Increment ham/spam counters
            (email.is_spam()) ? (amt_ngrams_spam_++) : (amt_ngrams_ham_++);
        }
        //std::cout << "Bucket Matrix : " << bucketMatrixHam_[hashSeeds_[0]][10];
    }

    double predict_(const Email& email) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        bool isSpam = email.is_spam();

        double p_S = (double) amt_ngrams_spam_ / (amt_ngrams_ham_ + amt_ngrams_spam_);
        double p_H = 1 - p_S;

        std::vector<double> vec_p_spam_given_n_gram = std::vector<double>();
        double sum_vec = 0;

        while(!iter.is_done()){
            std::string_view n_gram = iter.next();
            int minCountSpam = getMinCount_(1,n_gram);
            int minCountHam = getMinCount_(0,n_gram);
            // Calculate P(Spam | n_gram)
            double p_ngram_given_S = (double) minCountSpam / amt_ngrams_spam_;
            double p_ngram_given_H = (double) minCountHam / amt_ngrams_ham_;
            double p_spam_given_n_gram = (p_S * p_ngram_given_S) / (p_S * p_ngram_given_S + p_H * p_ngram_given_H);
            //std::cout << "P(Spam | n_gram) = " << p_spam_given_n_gram << '\n';
            vec_p_spam_given_n_gram.push_back(p_spam_given_n_gram);
            sum_vec += p_spam_given_n_gram;


        }

        double average = (double) sum_vec / vec_p_spam_given_n_gram.size();
        return average;
    }

    int getMinCount_(bool spam, std::string_view n_gram) const {
        std::vector<int> counts;


        // Iterate over map
        for(auto hash_seed : hashSeeds_){
            int bucketIndex = this->hash(n_gram,hash_seed) % amt_buckets_;
            (spam) ? (counts.push_back(bucketMatrixSpam_.at(hash_seed)[bucketIndex])) :
                (counts.push_back((bucketMatrixHam_.at(hash_seed)[bucketIndex])));
        }

        // Return Min of counts
        return *std::min_element(counts.begin(),counts.end());
    }
};

} // namespace bdap
