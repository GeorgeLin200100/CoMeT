#pragma once

#include <unordered_map>
#include <vector>
#include "dram_cntlr.h"
#include "dram_perf_model.h"
#include "shmem_msg.h"
#include "shmem_perf.h"
#include "fixed_types.h"
#include "memory_manager_base.h"
#include "dram_cntlr_interface.h"
#include "subsecond_time.h"
#include "core.h"
#include "log.h"
#include "subsecond_time.h"
#include "stats.h"
#include "fault_injection.h"
#include "shmem_perf.h"

using namespace std;

UInt32 dram_va2pa(UInt32 virtualBank, UInt32 passThrough);