#ifndef FAULT_SIM_H
#define FAULT_SIM_H

#ifndef SS_VER
struct fault_dat
{
    int do_fault;
    unsigned int polyvec_i;
    unsigned int poly_i;
    int num_rejections;
};

extern struct fault_dat fault_data;
#endif //SS_VER

#endif //FAULT_SIM_H
