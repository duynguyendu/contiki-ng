/**
 * \addtogroup rpl-lite
 * @{
 *
 * \file
 *         The Minimum Rank with Hysteresis Objective Function (MRHOF), RFC6719
 *
 *         This implementation uses the estimated number of
 *         transmissions (ETX) as the additive routing metric,
 *         and also provides stubs for the energy metric.
 *
 * \author Joakim Eriksson <joakime@sics.se>, Nicolas Tsiftes <nvt@sics.se>
 *  Simon Duquennoy <simon.duquennoy@inria.fr>
 */

#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "net/routing/rpl-lite/rpl.h"

/* Log configuration */
#include "sys/log.h"
#define LOG_MODULE "RPL"
#define LOG_LEVEL LOG_LEVEL_RPL

/* Configuration parameters of RFC6719. Reject parents that have a higher
 * link metric than the following. The default value is 512. */
#ifdef RPL_MRHOF_CONF_MAX_LINK_METRIC
#define MAX_LINK_METRIC RPL_MRHOF_CONF_MAX_LINK_METRIC
#else                       /* RPL_MRHOF_CONF_MAX_LINK_METRIC */
#define MAX_LINK_METRIC 512 /* Eq ETX of 4 */
#endif                      /* RPL_MRHOF_CONF_MAX_LINK_METRIC */

/* Reject parents that have a higher path cost than the following. */
#ifdef RPL_MRHOF_CONF_MAX_PATH_COST
#define MAX_PATH_COST RPL_MRHOF_CONF_MAX_PATH_COST
#else                       /*  RPL_MRHOF_CONF_MAX_PATH_COST */
#define MAX_PATH_COST 32768 /* Eq path ETX of 256 */
#endif                      /* RPL_MRHOF_CONF_MAX_PATH_COST */

#define RANK_THRESHOLD 192 /* Eq ETX of 1.5 */

/* Additional, custom hysteresis based on time. If a neighbor was consistently
 * better than our preferred parent for at least TIME_THRESHOLD, switch to
 * this neighbor regardless of RANK_THRESHOLD. */
#define TIME_THRESHOLD (10 * 60 * CLOCK_SECOND)

/*---------------------------------------------------------------------------*/
static void reset(void) { LOG_INFO("reset MLOF\n"); }
/*---------------------------------------------------------------------------*/
static uint16_t nbr_link_metric(rpl_nbr_t *nbr) {
  const struct link_stats *stats = rpl_neighbor_get_link_stats(nbr);
  return stats != NULL ? stats->etx : 0xffff;
}
/*---------------------------------------------------------------------------*/
static uint16_t link_metric_to_rank(uint16_t etx) { return etx; }
/*---------------------------------------------------------------------------*/
static uint16_t nbr_path_cost(rpl_nbr_t *nbr) {
  uint16_t base;

  if (nbr == NULL) {
    return 0xffff;
  }

#if RPL_WITH_MC
  /* Handle the different MC types */
  switch (curr_instance.mc.type) {
  case RPL_DAG_MC_ETX:
    base = nbr->mc.obj.etx;
    break;
  case RPL_DAG_MC_ENERGY:
    base = nbr->mc.obj.energy.energy_est << 8;
    break;
  default:
    base = nbr->rank;
    break;
  }
#else  /* RPL_WITH_MC */
  base = nbr->rank;
#endif /* RPL_WITH_MC */

  /* path cost upper bound: 0xffff */
  return MIN((uint32_t)base + link_metric_to_rank(nbr_link_metric(nbr)),
             0xffff);
}
/*---------------------------------------------------------------------------*/
static rpl_rank_t rank_via_nbr(rpl_nbr_t *nbr) {
  uint16_t min_hoprankinc;
  uint16_t path_cost;

  if (nbr == NULL) {
    return RPL_INFINITE_RANK;
  }

  min_hoprankinc = curr_instance.min_hoprankinc;
  path_cost = nbr_path_cost(nbr);

  /* Rank lower-bound: nbr rank + min_hoprankinc */
  return MAX(MIN((uint32_t)nbr->rank + min_hoprankinc, RPL_INFINITE_RANK),
             path_cost);
}
/*---------------------------------------------------------------------------*/
static int nbr_has_usable_link(rpl_nbr_t *nbr) {
  uint16_t link_metric = nbr_link_metric(nbr);
  /* Exclude links with too high link metrics  */
  return link_metric <= MAX_LINK_METRIC;
}
/*---------------------------------------------------------------------------*/
static int nbr_is_acceptable_parent(rpl_nbr_t *nbr) {
  uint16_t path_cost = nbr_path_cost(nbr);
  /* Exclude links with too high link metrics or path cost (RFC6719, 3.2.2) */
  return nbr_has_usable_link(nbr) && path_cost <= MAX_PATH_COST;
}
/*---------------------------------------------------------------------------*/
static int within_hysteresis(rpl_nbr_t *nbr) {
  uint16_t path_cost = nbr_path_cost(nbr);
  uint16_t parent_path_cost = nbr_path_cost(curr_instance.dag.preferred_parent);

  int within_rank_hysteresis = path_cost + RANK_THRESHOLD > parent_path_cost;
  int within_time_hysteresis =
      nbr->better_parent_since == 0 ||
      (clock_time() - nbr->better_parent_since) <= TIME_THRESHOLD;

  /* As we want to consider neighbors that are either beyond the rank or time
  hystereses, return 1 here iff the neighbor is within both hystereses. */
  return within_rank_hysteresis && within_time_hysteresis;
}
/*---------------------------------------------------------------------------*/
static rpl_nbr_t *best_parent(rpl_nbr_t *nbr1, rpl_nbr_t *nbr2) {
  int nbr1_is_acceptable;
  int nbr2_is_acceptable;

  nbr1_is_acceptable = nbr1 != NULL && nbr_is_acceptable_parent(nbr1);
  nbr2_is_acceptable = nbr2 != NULL && nbr_is_acceptable_parent(nbr2);

  if (!nbr1_is_acceptable) {
    return nbr2_is_acceptable ? nbr2 : NULL;
  }
  if (!nbr2_is_acceptable) {
    return nbr1_is_acceptable ? nbr1 : NULL;
  }

  /* Maintain stability of the preferred parent. Switch only if the gain
  is greater than RANK_THRESHOLD, or if the neighbor has been better than the
  current parent for at more than TIME_THRESHOLD. */
  if (nbr1 == curr_instance.dag.preferred_parent && within_hysteresis(nbr2)) {
    return nbr1;
  }
  if (nbr2 == curr_instance.dag.preferred_parent && within_hysteresis(nbr1)) {
    return nbr2;
  }

  return nbr_path_cost(nbr1) < nbr_path_cost(nbr2) ? nbr1 : nbr2;
}
/*---------------------------------------------------------------------------*/
#if !RPL_WITH_MC
static void update_metric_container(void) {
  curr_instance.mc.type = RPL_DAG_MC_NONE;
}
#else  /* RPL_WITH_MC */
static void update_metric_container(void) {
  uint16_t path_cost;
  uint8_t type;

  if (!curr_instance.used) {
    LOG_WARN("cannot update the metric container when not joined\n");
    return;
  }

  if (curr_instance.dag.rank == ROOT_RANK) {
    /* Configure MC at root only, other nodes are auto-configured when joining
     */
    curr_instance.mc.type = RPL_DAG_MC;
    curr_instance.mc.flags = 0;
    curr_instance.mc.aggr = RPL_DAG_MC_AGGR_ADDITIVE;
    curr_instance.mc.prec = 0;
    path_cost = curr_instance.dag.rank;
  } else {
    path_cost = nbr_path_cost(curr_instance.dag.preferred_parent);
  }

  /* Handle the different MC types */
  switch (curr_instance.mc.type) {
  case RPL_DAG_MC_NONE:
    break;
  case RPL_DAG_MC_ETX:
    curr_instance.mc.length = sizeof(curr_instance.mc.obj.etx);
    curr_instance.mc.obj.etx = path_cost;
    break;
  case RPL_DAG_MC_ENERGY:
    curr_instance.mc.length = sizeof(curr_instance.mc.obj.energy);
    if (curr_instance.dag.rank == ROOT_RANK) {
      type = RPL_DAG_MC_ENERGY_TYPE_MAINS;
    } else {
      type = RPL_DAG_MC_ENERGY_TYPE_BATTERY;
    }
    curr_instance.mc.obj.energy.flags = type << RPL_DAG_MC_ENERGY_TYPE;
    /* Energy_est is only one byte, use the least significant byte of the path
     * metric. */
    curr_instance.mc.obj.energy.energy_est = path_cost >> 8;
    break;
  default:
    LOG_WARN("MRHOF, non-supported MC %u\n", curr_instance.mc.type);
    break;
  }
}
#endif /* RPL_WITH_MC */
/*---------------------------------------------------------------------------*/
rpl_of_t rpl_mlof = {reset,
                     nbr_link_metric,
                     nbr_has_usable_link,
                     nbr_is_acceptable_parent,
                     nbr_path_cost,
                     rank_via_nbr,
                     best_parent,
                     update_metric_container,
                     RPL_OCP_MLOF};

/** @}*/
