#include "simulator.h"
#include "system/core_manager.h"
#include "performance_model/shmem_perf_model.h"
#include "core/core.h"
//#include "config.hpp"
#include "config_file.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config.cfg> <trace.txt>" << std::endl;
        return 1;
    }

    // Initialize simulator with config
    config::Config *cfg = new config::ConfigFile(argv[1]);
    Simulator::setConfig(cfg, Config::STANDALONE);
    Simulator::allocate();
    Sim()->start();

    // Get the core to inject accesses (use core 0 for single-core)
    Core *core = Sim()->getCoreManager()->getCoreFromID(0);
    if (!core) {
        std::cerr << "Failed to get core 0" << std::endl;
        Simulator::release();
        delete cfg;
        return 1;
    }

    // Open the trace file
    std::ifstream tracefile(argv[2]);
    if (!tracefile) {
        std::cerr << "Failed to open trace file: " << argv[2] << std::endl;
        Simulator::release();
        delete cfg;
        return 1;
    }

    std::string line;
    uint64_t last_time = 0;
    while (std::getline(tracefile, line)) {
        if (line.empty() || line[0] == '#') continue; // skip comments/empty
        std::istringstream iss(line);
        uint64_t time;
        std::string op;
        uint64_t addr;
        if (!(iss >> time >> op >> std::hex >> addr)) {
            std::cerr << "Malformed trace line: " << line << std::endl;
            continue;
        }

        // Advance simulated time if needed
        SubsecondTime sim_time = SubsecondTime::FS() * time;
        core->getShmemPerfModel()->setElapsedTime(ShmemPerfModel::_USER_THREAD, sim_time);

        // Inject memory access (assuming 64 bytes per access; adjust as needed)
        Core::mem_op_t mem_op = (op == "READ") ? Core::READ : Core::WRITE;
        core->accessMemory(Core::NONE, mem_op, addr, nullptr, 64, Core::MEM_MODELED_TIME);

        last_time = time;
    }

    // Finalize simulation
    Simulator::release();
    delete cfg;
    return 0;
} 