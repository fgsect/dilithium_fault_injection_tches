/***********************************************************************************************************************
 * This file was generated by the MCUXpresso Config Tools. Any manual edits made to this file
 * will be overwritten if the respective MCUXpresso Config Tools is used to update this file.
 **********************************************************************************************************************/

#ifndef _PERIPHERALS_H_
#define _PERIPHERALS_H_

/***********************************************************************************************************************
 * Included files
 **********************************************************************************************************************/
#include "fsl_common.h"
#include "fsl_lpuart.h"
#include "fsl_clock.h"
#include "fsl_dcp.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/***********************************************************************************************************************
 * Definitions
 **********************************************************************************************************************/
/* Definitions for BOARD_InitPeripherals functional group */
/* Definition of peripheral ID */
#define LPUART1_PERIPHERAL LPUART1
/* Definition of the clock source frequency */
#define LPUART1_CLOCK_SOURCE 4000000UL
/* Definition of peripheral ID */
#define DCP_PERIPHERAL DCP

/***********************************************************************************************************************
 * Global variables
 **********************************************************************************************************************/
extern const lpuart_config_t LPUART1_config;
/* Common initialization settings */
extern const dcp_config_t DCP_config;

/***********************************************************************************************************************
 * Initialization functions
 **********************************************************************************************************************/
void BOARD_InitPeripherals(void);

/***********************************************************************************************************************
 * BOARD_InitBootPeripherals function
 **********************************************************************************************************************/
void BOARD_InitBootPeripherals(void);

#if defined(__cplusplus)
}
#endif

#endif /* _PERIPHERALS_H_ */
