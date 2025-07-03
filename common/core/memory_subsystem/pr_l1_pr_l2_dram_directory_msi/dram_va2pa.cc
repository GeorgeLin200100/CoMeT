#include <vector>
#include "dram_cntlr.h"
#include "dram_trace_collect.h"
#include "memory_manager.h"
#include "core.h"
#include "log.h"
#include "subsecond_time.h"
#include "stats.h"
#include "fault_injection.h"
#include "shmem_perf.h"
#include "simulator.h"
#include "magic_server.h"
#include <stdint.h>
#include <inttypes.h>
#include "config.hpp"
#include "config.h"
#include "math.h"

UInt32 specifiedBank = 1;

UInt32 dram_va2pa (UInt32 virtualBank, UInt32 passThrough) {
    if (passThrough == 1) {
        return virtualBank;
    } else {
        return specifiedBank; // return a specific bank
    }
    
}