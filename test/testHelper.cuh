#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>
#include <cudnn.h>
#include <string>

#include "helpers.cuh"
#include "cudaThreadCtx.cuh"
#include "threadCtxManager.cuh"

static const std::string pass = "Test passed";
static const std::string fail = "Test failed: ";

template<typename T>
void expect(std::vector<T> ans, std::initializer_list<T> expected, std::string name, T eps = 0)
{
    int failCount = 0;
    std::cout << "============= " << name << " =============" << std::endl;
    if(ans.size() != expected.size())
    {
        std::cout << fail << " size missmactch -- expected length " << expected.size() << " answer length " << ans.size() << std::endl;
        return;
    }
    for(size_t i = 0; i < ans.size(); i++)
    {
        if(std::abs(ans[i] - expected.begin()[i]) > eps)
        {
            std::cout << fail << " incorrect response at " << i << " -- expected " << expected.begin()[i] << " answer " << ans[i] << std::endl;
            failCount++;
        }
    }
    if(failCount == 0)
        std::cout << pass << std::endl;
}

cudaThreadCtx* ctx;
threadCtxManager* manager;

void globalSetup()
{
    std::vector<int> availDevice;
    availDevice.push_back(1);

    manager = new threadCtxManager(3, availDevice);
    ctx = (*manager)[0];
}