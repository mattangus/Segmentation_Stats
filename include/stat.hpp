#pragma once
#include <opencv2/opencv.hpp>
#include <cudnn.h>

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
    virtual void accumulate(cudnnHandle_t& cudnn, unsigned char* gpuObj, int h, int w, int d) = 0;
    // void accumulate(unsigned char* d_obj, int h, int w, int d)
    // {
    //     accumulate(d_obj, int h, int w, int d, 0);
    // }
    virtual void finalize(cudnnHandle_t& cudnn) = 0;
    virtual void merge(cudnnHandle_t& cudnn) = 0;
    virtual void save(std::string outputFolder) = 0;
};