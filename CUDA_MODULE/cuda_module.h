#ifndef CUDA_MODULE_H
# define CUDA_MODULE_H
/* @brief
 *
 * Copyright (c)  2016 AZZAHRA Consulting France
 * The computer program contained herein contains proprietary
 * information which is the property of AZZAHRA Consulting France.
 * The program may be used and/or copied only with the written
 * permission of AZZAHRA Consulting France or in accordance with the
 * terms and conditions stipulated in the agreement/contract under
 * which the programs have been supplied.
 *
 * @author Mohamed Jaafar <mohamet.jaafar@gmail.com>
 */


# ifdef __cplusplus
extern "C" {
# endif
#include "sample_types.h"
/**
** \brief defines the cuda task holder.
*/
#define CUDA_MODULE_WORKER_PRIORITY     99
/**
 *
 * @brief   server's main.
 *
 * \param   void.
 * \return  error code
 * \author  mohamet.jaafar\@gmail.com
 * \date    2016
 */

void * CUDA_MODULE_entry_point (void * params);

/**
 *
 * @brief   cuda init.
 *
 * \param   void.
 * \return  error code
 * \author  mohamet.jaafar\@gmail.com
 * \date    2016
 */
SAMPLE_Error_t CUDA_MODULE_Init_Cuda_GPU(void);
# ifdef __cplusplus
}
# endif

#endif /* !CUDA_MODULE_H */
