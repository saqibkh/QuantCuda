#include <cstdio>
#include <cmath>
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
using namespace quantcuda;
int main() {
    std::printf("DRV available=%d\n", (int)gpu::available());
    OptionSpec bo; bo.type = OptionType::BarrierUpOutCall; bo.strike = 105.0; bo.maturity = 1.0; bo.barrier = 140.0;
    for (Precision p : {Precision::Float, Precision::Double, Precision::Mixed})
      for (std::size_t n : {1000ul, 100000ul})
        for (int steps : {1, 2, 52})
          for (int mode : {0, 1, 2}) {
            SimulationConfig cfg; cfg.paths = n; cfg.seed = 42; cfg.time_steps = steps; cfg.precision = p;
            try { McResult r = gpu::price_barrier(bo, 100.0, 0.05, 0.0, 0.25, cfg, mode);
              std::printf("DRV barrier p=%d n=%zu steps=%d mode=%d price=%.6f se=%.6f\n",(int)p,n,steps,mode,r.price,r.std_error);
            } catch (const std::exception& e) { std::printf("DRV barrier p=%d n=%zu steps=%d mode=%d THROW %s\n",(int)p,n,steps,mode,e.what()); }
          }
    OptionSpec o; o.type = OptionType::EuropeanCall; o.strike = 105.0; o.maturity = 1.0;
    SimulationConfig c; c.paths = 200000; c.seed = 42; c.precision = Precision::Double;
    McResult g = gpu::price_european(o, 100.0, 0.0425, 0.01, 0.2, c);
    std::printf("DRV european price=%.6f se=%.6f\n", g.price, g.std_error);
    OptionSpec as; as.type = OptionType::AsianCall; as.strike = 100.0; as.maturity = 1.0;
    SimulationConfig b = c; b.paths = 100000; b.time_steps = 52;
    McResult a = gpu::price_asian(as, 100.0, 0.05, 0.0, 0.25, b);
    std::printf("DRV asian price=%.6f se=%.6f\n", a.price, a.std_error);
    McGreeksResult gr = gpu::price_greeks_european(o, 100.0, 0.04, 0.0, 0.2, c);
    std::printf("DRV greeks price=%.6f delta=%.6f vega=%.6f\n", gr.price, gr.greeks.delta, gr.greeks.vega);
    return 0;
}
