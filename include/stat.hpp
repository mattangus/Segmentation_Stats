#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Abstract class for tracking statistics
 * 
 * @tparam T object over which to track
 */
template <typename T>
class stat
{
protected:
    std::string name;
    int numThreads;
public:
    virtual ~stat() { }
    virtual void accumulate(T obj, int thread) = 0;
    void accumulate(char* d_obj)
    {
        accumulate(d_obj, 0);
    }
    virtual void finalize(int thread) = 0;
    virtual void merge() = 0;
};