#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Abstract class for tracking statistics
 *
 */
class stat
{
protected:
    std::string name;
public:
    virtual ~stat() { }
    virtual void accumulate(unsigned char* gpuObj, int h, int w, int d) = 0;
    // void accumulate(unsigned char* d_obj, int h, int w, int d)
    // {
    //     accumulate(d_obj, int h, int w, int d, 0);
    // }
    virtual void finalize() = 0;
    virtual void merge() = 0;
    virtual void viz() = 0;
};