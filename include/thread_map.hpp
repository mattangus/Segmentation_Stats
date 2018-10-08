#pragma once

#include "RWLock.hpp"

#include <unordered_map>
#include <thread>

template<typename T>
class thread_map
{
private:
    std::unordered_map<std::thread::id, T> perThreadMap;
    RWLock lock;

    std::thread::id id()
    {
        return std::this_thread::get_id();
    }
public:
    thread_map() { }

    void set(T value)
    {
        //only one can write at a time
        lock.WriteLock();
        perThreadMap[id()] = value;
        lock.WriteUnlock();
    }

    T& get()
    {
        //many threads can read at a time
        lock.ReadLock();
        auto& ret = perThreadMap[id()];
        lock.ReadUnlock();
        return ret;
    }

    bool hasData()
    {
        lock.ReadLock();
        bool ret = perThreadMap.count(id()) > 0;
        lock.ReadUnlock();
        return ret;
    }

    std::vector<T> toList()
    {
        std::vector<T> ret;
        lock.ReadLock();
        for(auto& v : perThreadMap)
        {
            ret.push_back(v.second);
        }
        lock.ReadUnlock();
        return ret;
    }

};
