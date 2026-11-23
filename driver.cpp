#include <cstdio>
#include <exception>
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
using namespace quantcuda;
int main() {
    OptionSpec bo; bo.type = OptionType::BarrierUpOutCall; bo.strike = 105.0;
    bo.maturity = 1.0; bo.barrier = 140.0;
    for (int prec = 0; prec < 3; ++prec)
    for (std::size_t paths : {1000ul, 100000ul})
    for (int steps : {1, 2, 52})
    for (int mode : {0, 1, 2}) {
        SimulationConfig cfg; cfg.paths = paths; cfg.seed = 42; cfg.time_steps = steps;
        cfg.precision = prec == 0 ? Precision::Double : prec == 1 ? Precision::Float : Precision::Mixed;
        try {
            McResult r = gpu::price_barrier(bo, 100.0, 0.05, 0.0, 0.25, cfg, mode);
            McResult c = CpuMonteCarloPricer::price(bo, 100.0, 0.05, 0.0, 0.25, cfg);
            printf("prec=%d paths=%zu steps=%d mode=%d gpu=%.5f se=%.5f cpu=%.5f\n", prec, paths, steps, mode, r.price, r.std_error, c.price);
        } catch (const std::exception& e) {
            printf("prec=%d paths=%zu steps=%d mode=%d THROW %s\n", prec, paths, steps, mode, e.what());
        }
    }
}
