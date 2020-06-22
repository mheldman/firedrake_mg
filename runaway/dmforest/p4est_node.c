/*   create a dmforest and use it as a mesh
*/


  
#include <petscds.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/dmforestimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/viewerimpl.h>
#include <petscsys.h>
#include <p4est_base.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>
#include "petsc.h"
#include <../src/dm/impls/forest/p4est/pforest.h>
// #include </Users/maxheldman/petsc/src/dm/impls/forest/p4est/pforest.h>

/* we need two levels of macros to stringify the results of macro expansion */
#define _pforest_string(a) _pforest_string_internal(a)
#define _pforest_string_internal(a) #a

#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#include <p4est_vtk.h>
#include <p4est_plex.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>

/*
study the implementation of dmforest in petsc and use it to discretize a problem. implements example 3, solving a Poisson problem using DMForest, from the DMForest documentation.

dm options:

numbers of cells in each direction controlled with  -dm_forest_initial_refinement N
topology of the dmforest (not sure yet how this works): -dm_forest_topology brick
number of blocks in each direction: -dm_p4est_brick_size 1,2
-dm_p4est_brick_periodicity 0,1 <- not sure this does anything without dmplex (maybe it gets passed to p4est though)
-dm_view vtk:amr.vtu:vtk_vtu
-dm_p4est_refine_pattern center
 */
 

static char help[] = "";

typedef struct
{
/* blank for now */
} ctx;


/** We had 1. / 0. here to create a NaN but that is not portable. */
static const double step3_invalid = -1.;

/* In this example we store data with each quadrant/octant. */

/** Per-quadrant data for this example.
 *
 * In this problem, we keep track of the state variable u, its
 * derivatives in space, and in time.
 */
typedef struct step3_data
{
  double              u;             /**< the state variable */
  double              du[2]; /**< the spatial derivatives */
  double              dudt;          /**< the time derivative */
}
step3_data_t;

/** The example parameters.
 *
 * This describes the advection problem and time-stepping used in this
 * example.
 */
typedef struct step3_ctx
{
  double              center[2];  /**< coordinates of the center of
                                               the initial condition Gaussian
                                               bump */
  double              bump_width;         /**< width of the initial condition
                                               Gaussian bump */
  double              max_err;            /**< maximum allowed global
                                               interpolation error */
  double              v[2];       /**< the advection velocity */
  int                 refine_period;      /**< the number of time steps
                                               between mesh refinement */
  int                 repartition_period; /**< the number of time steps
                                               between repartitioning */
  int                 write_period;       /**< the number of time steps
                                               between writing vtk files */
}
step3_ctx_t;

/** Compute the value and derivatives of the initial condition.
 *
 * \param [in]  x   the coordinates
 * \param [out] du  the derivative at \a x
 * \param [in]  ctx the example parameters
 *
 * \return the initial condition at \a x
 */
static double
step3_initial_condition (double x[], double du[], step3_ctx_t * ctx)
{
  int                 i;
  double             *c = ctx->center;
  double              bump_width = ctx->bump_width;
  double              r2, d[2];
  double              arg, retval;

  r2 = 0.;
  for (i = 0; i < 2; i++) {
    d[i] = x[i] - c[i];
    r2 += d[i] * d[i];
  }

  arg = -(1. / 2.) * r2 / bump_width / bump_width;
  retval = exp (arg);

  if (du) {
    for (i = 0; i < 2; i++) {
      du[i] = -(1. / bump_width / bump_width) * d[i] * retval;
    }
  }

  return retval;
}

/** Get the coordinates of the midpoint of a quadrant.
 *
 * \param [in]  p4est      the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q          the quadrant
 * \param [out] xyz        the coordinates of the midpoint of \a q
 */
static void
step3_get_midpoint (p4est_t * p4est, p4est_topidx_t which_tree,
                    p4est_quadrant_t * q, double xyz[3])
{
  p4est_qcoord_t      half_length = P4EST_QUADRANT_LEN (q->level) / 2;

  p4est_qcoord_to_vertex (p4est->connectivity, which_tree,
                          q->x + half_length, q->y + half_length,
#ifdef P4_TO_P8
                          q->z + half_length,
#endif
                          xyz);
}

/** Initialize the initial condition data of a quadrant.
 *
 * This function matches the p4est_init_t prototype that is used by
 * p4est_new(), p4est_refine(), p4est_coarsen(), and p4est_balance().
 *
 * \param [in] p4est          the forest
 * \param [in] which_tree     the tree in the forest containing \a q
 * \param [in,out] q          the quadrant whose data gets initialized
 */
 
static void
step3_init_initial_condition (p4est_t * p4est, p4est_topidx_t which_tree,
                              p4est_quadrant_t * q)
{
  /* the data associated with a forest is accessible by user_pointer */
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  /* the data associated with a quadrant is accessible by p.user_data */
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  double              midpoint[3];

  step3_get_midpoint (p4est, which_tree, q, midpoint);
  /* initialize the data */
  data->u = step3_initial_condition (midpoint, data->du, ctx);
}

/** Estimate the square of the approximation error on a quadrant.
 *
 * We compute our estimate by integrating the difference of a constant
 * approximation at the midpoint and a linear approximation that interpolates
 * at the midpoint.
 *
 * \param [in] q a quadrant
 *
 * \return the square of the error estimate for the state variables contained
 * in \a q's data.
 */
static double
step3_error_sqr_estimate (p4est_quadrant_t * q)
{
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  int                 i;
  double              diff2;
  double             *du = data->du;
  double              h =
    (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
  double              vol;

#ifdef P4_TO_P8
  vol = h * h * h;
#else
  vol = h * h;
#endif

  diff2 = 0.;
  /* use the approximate derivative to estimate the L2 error */
  for (i = 0; i < 2; i++) {
    diff2 += du[i] * du[i] * (1. / 12.) * h * h * vol;
  }

  return diff2;
}

/** Refine by the L2 error estimate.
 *
 * Given the maximum global error, we enforce that each quadrant's portion of
 * the error must not exceed is fraction of the total volume of the domain
 * (which is 1).
 *
 * This function matches the p4est_refine_t prototype that is used by
 * p4est_refine() and p4est_refine_ext().
 *
 * \param [in] p4est          the forest
 * \param [in] which_tree     the tree in the forest containing \a q
 * \param [in] q              the quadrant
 *
 * \return 1 if \a q should be refined, 0 otherwise.
 */
static int
step3_refine_err_estimate (p4est_t * p4est, p4est_topidx_t which_tree,
                           p4est_quadrant_t * q)
{
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  double              global_err = ctx->max_err;
  double              global_err2 = global_err * global_err;
  double              h =
    (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
  double              vol, err2;

  /* the quadrant's volume is also its volume fraction */
#ifdef P4_TO_P8
  vol = h * h * h;
#else
  vol = h * h;
#endif

  err2 = step3_error_sqr_estimate (q);
  if (err2 > global_err2 * vol) {
    return 1;
  }
  else {
    return 0;
  }
}

/** Coarsen by the L2 error estimate of the initial condition.
 *
 * Given the maximum global error, we enforce that each quadrant's portion of
 * the error must not exceed is fraction of the total volume of the domain
 * (which is 1).
 *
 * \param [in] p4est          the forest
 * \param [in] which_tree     the tree in the forest containing \a children
 * \param [in] children       a family of quadrants
 *
 * \return 1 if \a children should be coarsened, 0 otherwise.
 */
static int
step3_coarsen_initial_condition (p4est_t * p4est,
                                 p4est_topidx_t which_tree,
                                 p4est_quadrant_t * children[])
{
  p4est_quadrant_t    parent;
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  double              global_err = ctx->max_err;
  double              global_err2 = global_err * global_err;
  double              h;
  step3_data_t        parentdata;
  double              parentmidpoint[3];
  double              vol, err2;

  /* get the parent of the first child (the parent of all children) */
  p4est_quadrant_parent (children[0], &parent);
  step3_get_midpoint (p4est, which_tree, &parent, parentmidpoint);
  parentdata.u = step3_initial_condition (parentmidpoint, parentdata.du, ctx);
  h = (double) P4EST_QUADRANT_LEN (parent.level) / (double) P4EST_ROOT_LEN;
  /* the quadrant's volume is also its volume fraction */
#ifdef P4_TO_P8
  vol = h * h * h;
#else
  vol = h * h;
#endif
  parent.p.user_data = (void *) (&parentdata);

  err2 = step3_error_sqr_estimate (&parent);
  if (err2 < global_err2 * vol) {
    return 1;
  }
  else {
    return 0;
  }
}

/** Coarsen by the L2 error estimate of the current state approximation.
 *
 * Given the maximum global error, we enforce that each quadrant's portion of
 * the error must not exceed its fraction of the total volume of the domain
 * (which is 1).
 *
 * This function matches the p4est_coarsen_t prototype that is used by
 * p4est_coarsen() and p4est_coarsen_ext().
 *
 * \param [in] p4est          the forest
 * \param [in] which_tree     the tree in the forest containing \a children
 * \param [in] children       a family of quadrants
 *
 * \return 1 if \a children should be coarsened, 0 otherwise.
 */
static int
step3_coarsen_err_estimate (p4est_t * p4est,
                            p4est_topidx_t which_tree,
                            p4est_quadrant_t * children[])
{
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  double              global_err = ctx->max_err;
  double              global_err2 = global_err * global_err;
  double              h;
  step3_data_t       *data;
  double              vol, err2, childerr2;
  double              parentu;
  double              diff;
  int                 i;

  h =
    (double) P4EST_QUADRANT_LEN (children[0]->level) /
    (double) P4EST_ROOT_LEN;
  /* the quadrant's volume is also its volume fraction */
#ifdef P4_TO_P8
  vol = h * h * h;
#else
  vol = h * h;
#endif

  /* compute the average */
  parentu = 0.;
  for (i = 0; i < P4EST_CHILDREN; i++) {
    data = (step3_data_t *) children[i]->p.user_data;
    parentu += data->u / P4EST_CHILDREN;
  }

  err2 = 0.;
  for (i = 0; i < P4EST_CHILDREN; i++) {
    childerr2 = step3_error_sqr_estimate (children[i]);

    if (childerr2 > global_err2 * vol) {
      return 0;
    }
    err2 += step3_error_sqr_estimate (children[i]);
    diff = (parentu - data->u) * (parentu - data->u);
    err2 += diff * vol;
  }
  if (err2 < global_err2 * (vol * P4EST_CHILDREN)) {
    return 1;
  }
  else {
    return 0;
  }
}

/** Initialize the state variables of incoming quadrants from outgoing
 * quadrants.
 *
 * The functions p4est_refine_ext(), p4est_coarsen_ext(), and
 * p4est_balance_ext() take as an argument a p4est_replace_t callback function,
 * which allows one to setup the quadrant data of incoming quadrants from the
 * data of outgoing quadrants, before the outgoing data is destroyed.  This
 * function matches the p4est_replace_t prototype.
 *
 * In this example, we linearly interpolate the state variable of a quadrant
 * that is refined to its children, and we average the midpoints of children
 * that are being coarsened to the parent.
 *
 * \param [in] p4est          the forest
 * \param [in] which_tree     the tree in the forest containing \a children
 * \param [in] num_outgoing   the number of quadrants that are being replaced:
 *                            either 1 if a quadrant is being refined, or
 *                            P4EST_CHILDREN if a family of children are being
 *                            coarsened.
 * \param [in] outgoing       the outgoing quadrants
 * \param [in] num_incoming   the number of quadrants that are being added:
 *                            either P4EST_CHILDREN if a quadrant is being refined, or
 *                            1 if a family of children are being
 *                            coarsened.
 * \param [in,out] incoming   quadrants whose data are initialized.
 */
static void
step3_replace_quads (p4est_t * p4est, p4est_topidx_t which_tree,
                     int num_outgoing,
                     p4est_quadrant_t * outgoing[],
                     int num_incoming, p4est_quadrant_t * incoming[])
{
  step3_data_t       *parent_data, *child_data;
  int                 i, j;
  double              h;
  double              du_old, du_est;

  if (num_outgoing > 1) {
    /* this is coarsening */
    parent_data = (step3_data_t *) incoming[0]->p.user_data;
    parent_data->u = 0.;
    for (j = 0; j < 2; j++) {
      parent_data->du[j] = step3_invalid;

    }
    for (i = 0; i < P4EST_CHILDREN; i++) {
      child_data = (step3_data_t *) outgoing[i]->p.user_data;
      parent_data->u += child_data->u / P4EST_CHILDREN;
      for (j = 0; j < 2; j++) {
        du_old = parent_data->du[j];
        du_est = child_data->du[j];

        if (du_old == du_old) {
          if (du_est * du_old >= 0.) {
            if (fabs (du_est) < fabs (du_old)) {
              parent_data->du[j] = du_est;
            }
          }
          else {
            parent_data->du[j] = 0.;
          }
        }
        else {
          parent_data->du[j] = du_est;
        }
      }
    }
  }
  else {
    /* this is refinement */
    parent_data = (step3_data_t *) outgoing[0]->p.user_data;
    h =
      (double) P4EST_QUADRANT_LEN (outgoing[0]->level) /
      (double) P4EST_ROOT_LEN;

    for (i = 0; i < P4EST_CHILDREN; i++) {
      child_data = (step3_data_t *) incoming[i]->p.user_data;
      child_data->u = parent_data->u;
      for (j = 0; j < 2; j++) {
        child_data->du[j] = parent_data->du[j];
        child_data->u +=
          (h / 4.) * parent_data->du[j] * ((i & (1 << j)) ? 1. : -1);
      }
    }
  }
}

/** Callback function for interpolating the solution from quadrant midpoints to
 * corners.
 *
 * The function p4est_iterate() takes as an argument a p4est_iter_volume_t
 * callback function, which it executes at every local quadrant (see
 * p4est_iterate.h).  This function matches the p4est_iter_volume_t prototype.
 *
 * In this example, we use the callback function to interpolate the state
 * variable to the corners, and write those corners into an array so that they
 * can be written out.
 *
 * \param [in] info          the information about this quadrant that has been
 *                           populated by p4est_iterate()
 * \param [in,out] user_data the user_data that was given as an argument to
 *                           p4est_iterate: in this case, it points to the
 *                           array of corner values that we want to write.
 *                           The values for the corner of the quadrant
 *                           described by \a info are written during the
 *                           execution of the callback.
 */
static void
step3_interpolate_solution (p4est_iter_volume_info_t * info, void *user_data)
{
  sc_array_t         *u_interp = (sc_array_t *) user_data;      /* we passed the array of values to fill as the user_data in the call to p4est_iterate */
  p4est_t            *p4est = info->p4est;
  p4est_quadrant_t   *q = info->quad;
  p4est_topidx_t      which_tree = info->treeid;
  p4est_locidx_t      local_id = info->quadid;  /* this is the index of q *within its tree's numbering*.  We want to convert it its index for all the quadrants on this process, which we do below */
  p4est_tree_t       *tree;
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  double              h;
  p4est_locidx_t      arrayoffset;
  double              this_u;
  double             *this_u_ptr;
  int                 i, j;

  tree = p4est_tree_array_index (p4est->trees, which_tree);
  local_id += tree->quadrants_offset;   /* now the id is relative to the MPI process */
  arrayoffset = P4EST_CHILDREN * local_id;      /* each local quadrant has 2^d (P4EST_CHILDREN) values in u_interp */
  h = (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;

  for (i = 0; i < P4EST_CHILDREN; i++) {
    this_u = data->u;
    /* loop over the derivative components and linearly interpolate from the
     * midpoint to the corners */
    for (j = 0; j < 2; j++) {
      /* In order to know whether the direction from the midpoint to the corner is
       * negative or positive, we take advantage of the fact that the corners
       * are in z-order.  If i is an odd number, it is on the +x side; if it
       * is even, it is on the -x side.  If (i / 2) is an odd number, it is on
       * the +y side, etc. */
      this_u += (h / 2) * data->du[j] * ((i & (1 << j)) ? 1. : -1.);
    }
    this_u_ptr = (double *) sc_array_index (u_interp, arrayoffset + i);
    this_u_ptr[0] = this_u;
  }

}

/** Write the state variable to vtk format, one file per process.
 *
 * \param [in] p4est    the forest, whose quadrant data contains the state
 * \param [in] timestep the timestep number, used to name the output files
 */
static void
step3_write_solution (p4est_t * p4est, int timestep)
{
  char                filename[BUFSIZ] = "";
  int                 retval;
  sc_array_t         *u_interp;
  p4est_locidx_t      numquads;
  p4est_vtk_context_t *context;

  snprintf (filename, BUFSIZ, P4EST_STRING "_step3_%04d", timestep);

  numquads = p4est->local_num_quadrants;

  /* create a vector with one value for the corner of every local quadrant
   * (the number of children is always the same as the number of corners) */
  u_interp = sc_array_new_size (sizeof (double), numquads * P4EST_CHILDREN);

  /* Use the iterator to visit every cell and fill in the solution values.
   * Using the iterator is not absolutely necessary in this case: we could
   * also loop over every tree (there is only one tree in this case) and loop
   * over every quadrant within every tree, but we are trying to demonstrate
   * the usage of p4est_iterate in this example */
  p4est_iterate (p4est, NULL,   /* we don't need any ghost quadrants for this loop */
                 (void *) u_interp,     /* pass in u_interp so that we can fill it */
                 step3_interpolate_solution,    /* callback function that interpolates from the cell center to the cell corners, defined above */
                 NULL,          /* there is no callback for the faces between quadrants */
#ifdef P4_TO_P8
                 NULL,          /* there is no callback for the edges between quadrants */
#endif
                 NULL);         /* there is no callback for the corners between quadrants */

  /* create VTK output context and set its parameters */
  context = p4est_vtk_context_new (p4est, filename);
  p4est_vtk_context_set_scale (context, 0.99);  /* quadrant at almost full scale */

  /* begin writing the output files */
  context = p4est_vtk_write_header (context);
  SC_CHECK_ABORT (context != NULL,
                  P4EST_STRING "_vtk: Error writing vtk header");

  /* do not write the tree id's of each quadrant
   * (there is only one tree in this example) */
  context = p4est_vtk_write_cell_dataf (context, 0, 1,  /* do write the refinement level of each quadrant */
                                        1,      /* do write the mpi process id of each quadrant */
                                        0,      /* do not wrap the mpi rank (if this were > 0, the modulus of the rank relative to this number would be written instead of the rank) */
                                        0,      /* there is no custom cell scalar data. */
                                        0,      /* there is no custom cell vector data. */
                                        context);       /* mark the end of the variable cell data. */
  SC_CHECK_ABORT (context != NULL,
                  P4EST_STRING "_vtk: Error writing cell data");

  /* write one scalar field: the solution value */
  context = p4est_vtk_write_point_dataf (context, 1, 0, /* write no vector fields */
                                         "solution", u_interp, context);        /* mark the end of the variable cell data. */
  SC_CHECK_ABORT (context != NULL,
                  P4EST_STRING "_vtk: Error writing cell data");

  retval = p4est_vtk_write_footer (context);
  SC_CHECK_ABORT (!retval, P4EST_STRING "_vtk: Error writing footer");

  sc_array_destroy (u_interp);
}

/** Approximate the divergence of (vu) on each quadrant
 *
 * We use piecewise constant approximations on each quadrant, so the value is
 * always 0.
 *
 * Like step3_interpolate_solution(), this function matches the
 * p4est_iter_volume_t prototype used by p4est_iterate().
 *
 * \param [in] info          the information about the quadrant populated by
 *                           p4est_iterate()
 * \param [in] user_data     not used
 */
static void
step3_quad_divergence (p4est_iter_volume_info_t * info, void *user_data)
{
  p4est_quadrant_t   *q = info->quad;
  step3_data_t       *data = (step3_data_t *) q->p.user_data;

  data->dudt = 0.;
}

/** Approximate the flux across a boundary between quadrants.
 *
 * We use a very simple upwind numerical flux.
 *
 * This function matches the p4est_iter_face_t prototype used by
 * p4est_iterate().
 *
 * \param [in] info the information about the quadrants on either side of the
 *                  interface, populated by p4est_iterate()
 * \param [in] user_data the user_data given to p4est_iterate(): in this case,
 *                       it points to the ghost_data array, which contains the
 *                       step3_data_t data for all of the ghost cells, which
 *                       was populated by p4est_ghost_exchange_data()
 */
static void
step3_upwind_flux (p4est_iter_face_info_t * info, void *user_data)
{
  int                 i, j;
  p4est_t            *p4est = info->p4est;
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  step3_data_t       *ghost_data = (step3_data_t *) user_data;
  step3_data_t       *udata;
  p4est_quadrant_t   *quad;
  double              vdotn = 0.;
  double              uavg;
  double              q;
  double              h, facearea;
  int                 which_face;
  int                 upwindside;
  p4est_iter_face_side_t *side[2];
  sc_array_t         *sides = &(info->sides);

  /* because there are no boundaries, every face has two sides */
  P4EST_ASSERT (sides->elem_count == 2);

  side[0] = p4est_iter_fside_array_index_int (sides, 0);
  side[1] = p4est_iter_fside_array_index_int (sides, 1);

  /* which of the quadrant's faces the interface touches */
  which_face = side[0]->face;

  switch (which_face) {
  case 0:                      /* -x side */
    vdotn = -ctx->v[0];
    break;
  case 1:                      /* +x side */
    vdotn = ctx->v[0];
    break;
  case 2:                      /* -y side */
    vdotn = -ctx->v[1];
    break;
  case 3:                      /* +y side */
    vdotn = ctx->v[1];
    break;
#ifdef P4_TO_P8
  case 4:                      /* -z side */
    vdotn = -ctx->v[2];
    break;
  case 5:                      /* +z side */
    vdotn = ctx->v[2];
    break;
#endif
  }
  upwindside = vdotn >= 0. ? 0 : 1;

  /* Because we have non-conforming boundaries, one side of an interface can
   * either have one large ("full") quadrant or 2^(d-1) small ("hanging")
   * quadrants: we have to compute the average differently in each case.  The
   * info populated by p4est_iterate() gives us the context we need to
   * proceed. */
  uavg = 0;
  if (side[upwindside]->is_hanging) {
    /* there are 2^(d-1) (P4EST_HALF) subfaces */
    for (j = 0; j < P4EST_HALF; j++) {
      if (side[upwindside]->is.hanging.is_ghost[j]) {
        /* *INDENT-OFF* */
        udata =
          (step3_data_t *) &ghost_data[side[upwindside]->is.hanging.quadid[j]];
        /* *INDENT-ON* */
      }
      else {
        udata =
          (step3_data_t *) side[upwindside]->is.hanging.quad[j]->p.user_data;
      }
      uavg += udata->u;
    }
    uavg /= P4EST_HALF;
  }
  else {
    if (side[upwindside]->is.full.is_ghost) {
      udata = (step3_data_t *) & ghost_data[side[upwindside]->is.full.quadid];
    }
    else {
      udata = (step3_data_t *) side[upwindside]->is.full.quad->p.user_data;
    }
    uavg = udata->u;
  }
  /* flux from side 0 to side 1 */
  q = vdotn * uavg;
  for (i = 0; i < 2; i++) {
    if (side[i]->is_hanging) {
      /* there are 2^(d-1) (P4EST_HALF) subfaces */
      for (j = 0; j < P4EST_HALF; j++) {
        quad = side[i]->is.hanging.quad[j];
        h =
          (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
#ifndef P4_TO_P8
        facearea = h;
#else
        facearea = h * h;
#endif
        if (!side[i]->is.hanging.is_ghost[j]) {
          udata = (step3_data_t *) quad->p.user_data;
          if (i == upwindside) {
            udata->dudt += vdotn * udata->u * facearea * (i ? 1. : -1.);
          }
          else {
            udata->dudt += q * facearea * (i ? 1. : -1.);
          }
        }
      }
    }
    else {
      quad = side[i]->is.full.quad;
      h = (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
#ifndef P4_TO_P8
      facearea = h;
#else
      facearea = h * h;
#endif
      if (!side[i]->is.full.is_ghost) {
        udata = (step3_data_t *) quad->p.user_data;
        udata->dudt += q * facearea * (i ? 1. : -1.);
      }
    }
  }
}

/** Compute the new value of the state from the computed time derivative.
 *
 * We use a simple forward Euler scheme.
 *
 * The derivative was computed by a p4est_iterate() loop by the callbacks
 * step3_quad_divergence() and step3_upwind_flux(). Now we multiply this by
 * the timestep and add to the current solution.
 *
 * This function matches the p4est_iter_volume_t prototype used by
 * p4est_iterate().
 *
 * \param [in] info          the information about this quadrant that has been
 *                           populated by p4est_iterate()
 * \param [in] user_data the user_data given to p4est_iterate(): in this case,
 *                       it points to the timestep.
 */
static void
step3_timestep_update (p4est_iter_volume_info_t * info, void *user_data)
{
  p4est_quadrant_t   *q = info->quad;
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  double              dt = *((double *) user_data);
  double              vol;
  double              h =
    (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;

#ifdef P4_TO_P8
  vol = h * h * h;
#else
  vol = h * h;
#endif

  data->u += dt * data->dudt / vol;
}

/** Reset the approximate derivatives.
 *
 * p4est_iterate() has an invariant to the order of callback execution: the
 * p4est_iter_volume_t callback will be executed on a quadrant before the
 * p4est_iter_face_t callbacks are executed on its faces.  This function
 * resets the derivative stored in the quadrant's data before
 * step3_minmod_estimate() updates the derivative based on the face neighbors.
 *
 * This function matches the p4est_iter_volume_t prototype used by
 * p4est_iterate().
 *
 * \param [in] info          the information about this quadrant that has been
 *                           populated by p4est_iterate()
 * \param [in] user_data     not used
 */
static void
step3_reset_derivatives (p4est_iter_volume_info_t * info, void *user_data)
{
  p4est_quadrant_t   *q = info->quad;
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  int                 j;

  for (j = 0; j < 2; j++) {
    data->du[j] = step3_invalid;
  }
}

/** For two quadrants on either side of a face, estimate the derivative normal
 * to the face.
 *
 * This function matches the p4est_iter_face_t prototype used by
 * p4est_iterate().
 *
 * \param [in] info          the information about this quadrant that has been
 *                           populated by p4est_iterate()
 * \param [in] user_data the user_data given to p4est_iterate(): in this case,
 *                       it points to the ghost_data array, which contains the
 *                       step3_data_t data for all of the ghost cells, which
 *                       was populated by p4est_ghost_exchange_data()
 */
static void
step3_minmod_estimate (p4est_iter_face_info_t * info, void *user_data)
{
  int                 i, j;
  p4est_iter_face_side_t *side[2];
  sc_array_t         *sides = &(info->sides);
  step3_data_t       *ghost_data = (step3_data_t *) user_data;
  step3_data_t       *udata;
  p4est_quadrant_t   *quad;
  double              uavg[2];
  double              h[2];
  double              du_est, du_old;
  int                 which_dir;

  /* because there are no boundaries, every face has two sides */
  P4EST_ASSERT (sides->elem_count == 2);

  side[0] = p4est_iter_fside_array_index_int (sides, 0);
  side[1] = p4est_iter_fside_array_index_int (sides, 1);

  which_dir = side[0]->face / 2;        /* 0 == x, 1 == y, 2 == z */

  for (i = 0; i < 2; i++) {
    uavg[i] = 0;
    if (side[i]->is_hanging) {
      /* there are 2^(d-1) (P4EST_HALF) subfaces */
      for (j = 0; j < P4EST_HALF; j++) {
        quad = side[i]->is.hanging.quad[j];
        h[i] =
          (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
        if (side[i]->is.hanging.is_ghost[j]) {
          udata = &ghost_data[side[i]->is.hanging.quadid[j]];
        }
        else {
          udata = (step3_data_t *) side[i]->is.hanging.quad[j]->p.user_data;
        }
        uavg[i] += udata->u;
      }
      uavg[i] /= P4EST_HALF;
    }
    else {
      quad = side[i]->is.full.quad;
      h[i] =
        (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
      if (side[i]->is.full.is_ghost) {
        udata = &ghost_data[side[i]->is.full.quadid];
      }
      else {
        udata = (step3_data_t *) side[i]->is.full.quad->p.user_data;
      }
      uavg[i] = udata->u;
    }
  }
  du_est = (uavg[1] - uavg[0]) / ((h[0] + h[1]) / 2.);
  for (i = 0; i < 2; i++) {
    if (side[i]->is_hanging) {
      /* there are 2^(d-1) (P4EST_HALF) subfaces */
      for (j = 0; j < P4EST_HALF; j++) {
        quad = side[i]->is.hanging.quad[j];
        if (!side[i]->is.hanging.is_ghost[j]) {
          udata = (step3_data_t *) quad->p.user_data;
          du_old = udata->du[which_dir];
          if (du_old == du_old) {
            /* there has already been an update */
            if (du_est * du_old >= 0.) {
              if (fabs (du_est) < fabs (du_old)) {
                udata->du[which_dir] = du_est;
              }
            }
            else {
              udata->du[which_dir] = 0.;
            }
          }
          else {
            udata->du[which_dir] = du_est;
          }
        }
      }
    }
    else {
      quad = side[i]->is.full.quad;
      if (!side[i]->is.full.is_ghost) {
        udata = (step3_data_t *) quad->p.user_data;
        du_old = udata->du[which_dir];
        if (du_old == du_old) {
          /* there has already been an update */
          if (du_est * du_old >= 0.) {
            if (fabs (du_est) < fabs (du_old)) {
              udata->du[which_dir] = du_est;
            }
          }
          else {
            udata->du[which_dir] = 0.;
          }
        }
        else {
          udata->du[which_dir] = du_est;
        }
      }
    }
  }
}

/** Compute the maximum state value.
 *
 * This function updates the maximum value from the value of a single cell.
 *
 * This function matches the p4est_iter_volume_t prototype used by
 * p4est_iterate().
 *
 * \param [in] info              the information about this quadrant that has been
 *                               populated by p4est_iterate()
 * \param [in,out] user_data     the user_data given to p4est_iterate(): in this case,
 *                               it points to the maximum value that will be updated
 */
static void
step3_compute_max (p4est_iter_volume_info_t * info, void *user_data)
{
  p4est_quadrant_t   *q = info->quad;
  step3_data_t       *data = (step3_data_t *) q->p.user_data;
  double              umax = *((double *) user_data);

  umax = SC_MAX (data->u, umax);

  *((double *) user_data) = umax;
}

/** Compute the timestep.
 *
 * Find the smallest quadrant and scale the timestep based on that length and
 * the advection velocity.
 *
 * \param [in] p4est the forest
 * \return the timestep.
 */
static double
step3_get_timestep (p4est_t * p4est)
{
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  p4est_topidx_t      t, flt, llt;
  p4est_tree_t       *tree;
  int                 max_level, global_max_level;
  int                 mpiret, i;
  double              min_h, vnorm;
  double              dt;

  /* compute the timestep by finding the smallest quadrant */
  flt = p4est->first_local_tree;
  llt = p4est->last_local_tree;

  max_level = 0;
  for (t = flt; t <= llt; t++) {
    tree = p4est_tree_array_index (p4est->trees, t);
    max_level = SC_MAX (max_level, tree->maxlevel);

  }
  mpiret =
    sc_MPI_Allreduce (&max_level, &global_max_level, 1, sc_MPI_INT,
                      sc_MPI_MAX, p4est->mpicomm);
  SC_CHECK_MPI (mpiret);

  min_h =
    (double) P4EST_QUADRANT_LEN (global_max_level) / (double) P4EST_ROOT_LEN;

  vnorm = 0;
  for (i = 0; i < 2; i++) {
    vnorm += ctx->v[i] * ctx->v[i];
  }
  vnorm = sqrt (vnorm);

  dt = min_h / 2. / vnorm;

  return dt;
}

/** Timestep the advection problem.
 *
 * Update the state, refine, repartition, and write the solution to file.
 *
 * \param [in,out] p4est the forest, whose state is updated
 * \param [in] time      the end time
 */
static void
step3_timestep (p4est_t * p4est, double time)
{
  double              t = 0.;
  double              dt = 0.;
  int                 i;
  step3_data_t       *ghost_data;
  step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
  int                 refine_period = ctx->refine_period;
  int                 repartition_period = ctx->repartition_period;
  int                 write_period = ctx->write_period;
  int                 recursive = 0;
  int                 allowed_level = P4EST_QMAXLEVEL;
  int                 allowcoarsening = 1;
  int                 callbackorphans = 0;
  int                 mpiret;
  double              orig_max_err = ctx->max_err;
  double              umax, global_umax;
  p4est_ghost_t      *ghost;

  /* create the ghost quadrants */
  ghost = p4est_ghost_new (p4est, P4EST_CONNECT_FULL);
  /* create space for storing the ghost data */
  ghost_data = P4EST_ALLOC (step3_data_t, ghost->ghosts.elem_count);
  /* synchronize the ghost data */
  p4est_ghost_exchange_data (p4est, ghost, ghost_data);

  /* initialize du/dx estimates */
  p4est_iterate (p4est, ghost, (void *) ghost_data,     /* pass in ghost data that we just exchanged */
                 step3_reset_derivatives,       /* blank the previously calculated derivatives */
                 step3_minmod_estimate, /* compute the minmod estimate of each cell's derivative */
#ifdef P4_TO_P8
                 NULL,          /* there is no callback for the edges between quadrants */
#endif
                 NULL);         /* there is no callback for the corners between quadrants */

  for (t = 0., i = 0; t < time; t += dt, i++) {
    P4EST_GLOBAL_PRODUCTIONF ("time %f\n", t);

    /* refine */
    if (!(i % refine_period)) {
      if (i) {
        /* compute umax */
        umax = 0.;
        /* initialize derivative estimates */
        p4est_iterate (p4est, NULL, (void *) &umax,     /* pass in ghost data that we just exchanged */
                       step3_compute_max,       /* blank the previously calculated derivatives */
                       NULL,    /* there is no callback for the faces between quadrants */
#ifdef P4_TO_P8
                       NULL,    /* there is no callback for the edges between quadrants */
#endif
                       NULL);   /* there is no callback for the corners between quadrants */

        mpiret =
          sc_MPI_Allreduce (&umax, &global_umax, 1, sc_MPI_DOUBLE, sc_MPI_MAX,
                            p4est->mpicomm);
        SC_CHECK_MPI (mpiret);
        ctx->max_err = orig_max_err * global_umax;
        P4EST_GLOBAL_PRODUCTIONF ("u_max %f\n", global_umax);

        /* adapt */
        p4est_refine_ext (p4est, recursive, allowed_level,
                          step3_refine_err_estimate, NULL,
                          step3_replace_quads);
        p4est_coarsen_ext (p4est, recursive, callbackorphans,
                           step3_coarsen_err_estimate, NULL,
                           step3_replace_quads);
        p4est_balance_ext (p4est, P4EST_CONNECT_FACE, NULL,
                           step3_replace_quads);

        p4est_ghost_destroy (ghost);
        P4EST_FREE (ghost_data);
        ghost = NULL;
        ghost_data = NULL;
      }
      dt = step3_get_timestep (p4est);
    }

    /* repartition */
    if (i && !(i % repartition_period)) {
      p4est_partition (p4est, allowcoarsening, NULL);

      if (ghost) {
        p4est_ghost_destroy (ghost);
        P4EST_FREE (ghost_data);
        ghost = NULL;
        ghost_data = NULL;
      }
    }

    /* write out solution */
    if (!(i % write_period)) {
      step3_write_solution (p4est, i);
    }

    /* synchronize the ghost data */
    if (!ghost) {
      ghost = p4est_ghost_new (p4est, P4EST_CONNECT_FULL);
      ghost_data = P4EST_ALLOC (step3_data_t, ghost->ghosts.elem_count);
      p4est_ghost_exchange_data (p4est, ghost, ghost_data);
    }

    /* compute du/dt */
    /* *INDENT-OFF* */
    p4est_iterate (p4est,                 /* the forest */
                   ghost,                 /* the ghost layer */
                   (void *) ghost_data,   /* the synchronized ghost data */
                   step3_quad_divergence, /* callback to compute each quad's
                                             interior contribution to du/dt */
                   step3_upwind_flux,     /* callback to compute each quads'
                                             faces' contributions to du/du */
#ifdef P4_TO_P8
                   NULL,                  /* there is no callback for the
                                             edges between quadrants */
#endif
                   NULL);                 /* there is no callback for the
                                             corners between quadrants */
    /* *INDENT-ON* */

    /* update u */
    p4est_iterate (p4est, NULL, /* ghosts are not needed for this loop */
                   (void *) &dt,        /* pass in dt */
                   step3_timestep_update,       /* update each cell */
                   NULL,        /* there is no callback for the faces between quadrants */
#ifdef P4_TO_P8
                   NULL,        /* there is no callback for the edges between quadrants */
#endif
                   NULL);       /* there is no callback for the corners between quadrants */

    /* synchronize the ghost data */
    p4est_ghost_exchange_data (p4est, ghost, ghost_data);

    /* update du/dx estimate */
    p4est_iterate (p4est, ghost, (void *) ghost_data,   /* pass in ghost data that we just exchanged */
                   step3_reset_derivatives,     /* blank the previously calculated derivatives */
                   step3_minmod_estimate,       /* compute the minmod estimate of each cell's derivative */
#ifdef P4_TO_P8
                   NULL,        /* there is no callback for the edges between quadrants */
#endif
                   NULL);       /* there is no callback for the corners between quadrants */
  }

  P4EST_FREE (ghost_data);
  p4est_ghost_destroy (ghost);
}

/*
this function gets the exact solution on a particular quadrant
 */
 
void get_midpoint(p4est_t * p4est, p4est_topidx_t which_tree,
                    p4est_quadrant_t * q, double xyz[3]){
  p4est_qcoord_t      half_length = P4EST_QUADRANT_LEN (q->level) / 2;

  p4est_qcoord_to_vertex(p4est->connectivity, which_tree,
                          q->x + half_length, q->y + half_length,
#ifdef P4_TO_P8
                          q->z + half_length,
#endif
                          xyz);
}

void build_rhs(p4est_iter_volume_info_t * info,
                                            Vec user_data){
        Vec            exact = user_data;
        p4est_quadrant_t   *q = info->quad;
        p4est_t            *p = info->p4est;
        p4est_tree_t       *tree;
        p4est_locidx_t      arrayoffset;
        p4est_topidx_t      which_tree = info->treeid;
        p4est_locidx_t      local_id = info->quadid; // local id (within tree's numbering). making this the petsc ordering as well makes sense
        PetscInt            mpirank = p->mpirank;
        p4est_locidx_t id[1];
        double              h;
        double       xyz[3]; //coordinates to evaluate the function at
        PetscReal val;
        PetscErrorCode ierr;
        
        tree = p4est_tree_array_index (p->trees, which_tree);
        local_id += tree->quadrants_offset;
        h = (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
        
        p4est_qcoord_to_vertex(p->connectivity, which_tree, q->x, q->y, xyz);
        xyz[0] += h/2.0; xyz[1] += h/2.0; //want the midpoint
        val = -h*h*PetscSinReal(xyz[0])*PetscSinReal(xyz[1]);
        // printf("%i (%f, %f), %f\n", local_id, xyz[0], xyz[1], val);
        VecSetValue(user_data,local_id*mpirank,val,INSERT_VALUES); //local to global mapping is local*mpirank = global
}

// Construct bilinear stiffness matrix
// Non-hanging nodes will get the stencil -1  -1  -1
//                                        -1   8  -1
//					  -1  -1  -1
// Hanging nodes will get the stencil     1/2  0  1/2
// Or                                         1/2
//                                             0
//                                            1/2
/*
void fd(p4est_iter_face_side * info, Mat K){
  int                 i, j;
  p4est_t            *p = info->p4est;
  step3_data_t       *ghost_data = (step3_data_t *) user_data;
  p4est_quadrant_t   *quad;
  double              uavg;
  double              q;
  double              h, facearea;
  double              flux;
  double               val;
  int                 which_face;
  int                 upwindside;
  p4est_iter_face_side_t *side[2]; //p4est_iter_face_side_t is a struct containing face side data (one or two quadrants, with ids, and ghost true/false)
  sc_array_t         *sides = &(info->sides);

  
  side[0] = p4est_iter_fside_array_index_int(sides, 0); // this gives me face on one side i guess
  if(sides->elem_count == 2){
    side[1] = p4est_iter_fside_array_index_int(sides, 1);
  

  /* want to: loop over the edges. Find the elements i,j on either side of the edge, and then update the i,j entry of the jacobian according to the fd scheme. 
  /* which of the quadrant's faces the interface touches
  /* which_face = side[0]->face;
  switch (which_face) {
  case 0:                      /* -x side
    vdotn = -ctx->v[0];
    break;
  case 1:                      /* +x side
    vdotn = ctx->v[0];
    break;
  case 2:                      /* -y side
    vdotn = -ctx->v[1];
    break;
  case 3:                      /* +y side
    vdotn = ctx->v[1];
    break;  }
  upwindside = vdotn >= 0. ? 0 : 1; could be useful later 

  /* Because we have non-conforming boundaries, one side of an interface can
   * either have one large ("full") quadrant or 2^(d-1) small ("hanging")
   * quadrants: we have to compute the average differently in each case.  The
   * info populated by p4est_iterate() gives us the context we need to
   * proceed. 
  uavg = 0;
  /*
  if (side[0]->is_hanging) {
    /* there are 2^(d-1) (P4EST_HALF) subfaces
    for (j = 0; j < P4EST_HALF; j++) {
      if (side[0]->is.hanging.is_ghost[j]) {
        /* *INDENT-OFF*         // later we'll add in hanging nodes. for now just do non-hanging
        /* *INDENT-ON*
      }
  
  }
  else {
    if (side[upwindside]->is.full.is_ghost) {
      udata = (step3_data_t *) & ghost_data[side[upwindside]->is.full.quadid];
    }
    else {
      udata = (step3_data_t *) side[upwindside]->is.full.quad->p.user_data;
    }
    uavg = udata->u;
  }
  /* flux from side 0 to side 1 
  q = vdotn * uavg;
  for (i = 0; i < 2; i++) {
    if (side[i]->is_hanging) {
      /* there are 2^(d-1) (P4EST_HALF) subfaces 
      for (j = 0; j < P4EST_HALF; j++) {
        quad = side[i]->is.hanging.quad[j];
        h =
          (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
#ifndef P4_TO_P8
        facearea = h;
#else
        facearea = h * h;
#endif
        if (!side[i]->is.hanging.is_ghost[j]) {
          udata = (step3_data_t *) quad->p.user_data;
          if (i == upwindside) {
            udata->dudt += vdotn * udata->u * facearea * (i ? 1. : -1.);
          }
          else {
            udata->dudt += q * facearea * (i ? 1. : -1.);
          }
        }
      }
    }
    else {
      quad = side[i]->is.full.quad;
      h = (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
#ifndef P4_TO_P8
      facearea = h;
#else
      facearea = h * h;
#endif
      if (!side[i]->is.full.is_ghost) {
        udata = (step3_data_t *) quad->p.user_data;
        udata->dudt += q * facearea * (i ? 1. : -1.);
      }
    }
  }
}

}
  */



void exact_solution(p4est_iter_volume_info_t * info,
                                            Vec user_data){
        Vec            exact = user_data;
        p4est_quadrant_t   *q = info->quad;
        p4est_t            *p = info->p4est;
        p4est_tree_t       *tree;
        p4est_locidx_t      arrayoffset;
        p4est_topidx_t      which_tree = info->treeid;
        p4est_locidx_t      local_id = info->quadid; // local id (within tree's numbering). making this the petsc ordering as well makes sense
        PetscInt            mpirank = p->mpirank;
        p4est_locidx_t id[1];
        double              h;
        double       xyz[3]; //coordinates to evaluate the function at
        PetscReal val;
        PetscErrorCode ierr;
        
	for (i = 0; i < P4EST_CHILDREN; ++i) { /* Loop over 2**D corners. */
	  lni = lnodes->element_nodes[P4EST_CHILDREN * k + i]; /* Element index k. */
	  if (bc[lni] < 0) {
	    if (anyhang && hanging_corner[i] >= 0) {
	      /* This node is hanging; access the referenced node instead. */
	      p4est_quadrant_corner_node (parent, i, &node);
	    }
	    else {
	      p4est_quadrant_corner_node (quad, i, &node);
	    }
	    /* Determine boundary status of independent node. */
	    bc[lni] = is_boundary_unitsquare (p4est, tt, &node);
	    /* Transform per-tree reference coordinates into physical space. */
	    p4est_qcoord_to_vertex (p4est->connectivity, tt,
				    node.x, node.y, vxyz);
	    /* Use physical space coordinates to evaluate functions */
	    rhs[lni] = func_rhs (vxyz);
	    uexact[lni] = func_uexact (vxyz);        tree = p4est_tree_array_index (p->trees, which_tree);
	    local_id += tree->quadrants_offset;
	    h = (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
        
	    p4est_qcoord_to_vertex(p->connectivity, which_tree, q->x, q->y, xyz);
	    xyz[0] += h/2.0; xyz[1] += h/2.0; //want the midpoint
	    val = PetscSinReal(xyz[0])*PetscSinReal(xyz[1]);
	    // printf("%i (%f, %f), %f\n", local_id, xyz[0], xyz[1], val);
	    VecSetValue(exact,local_id*mpirank,val,INSERT_VALUES); //local to global mapping is local*mpirank = global
	  }
	  
interpolate_functions (p4est_t * p4est, p4est_lnodes_t * lnodes,
                       double **rhs_eval, double **uexact_eval, int8_t ** pbc)
{
  const p4est_locidx_t nloc = lnodes->num_local_nodes;
  int                 anyhang, hanging_corner[P4EST_CHILDREN];
  int                 i;        /* We use plain int for small loops. */
  double             *rhs, *uexact;
  double              vxyz[3];  /* We embed the 2D vertices into 3D space. */
  int8_t             *bc;
  p4est_topidx_t      tt;       /* Connectivity variables have this type. */
  p4est_locidx_t      k, q, Q;  /* Process-local counters have this type. */
  p4est_locidx_t      lni;      /* Node index relative to this processor. */
  p4est_tree_t       *tree;     /* Pointer to one octree */
  p4est_quadrant_t   *quad, *parent, sp, node;
  sc_array_t         *tquadrants;       /* Quadrant array for one tree */

  bc = *pbc = P4EST_ALLOC (int8_t, nloc);
  memset (bc, -1, sizeof (int8_t) * nloc);      /* Indicator for visiting. */

  /* We need to compute the xyz locations of non-hanging nodes to evaluate the
   * given functions.  For hanging nodes, we have to look at the corresponding
   * independent nodes.  Usually we would cache this information, here we only
   * need it once and throw it away again.
   * We also compute boundary status of independent nodes. */
  for (tt = p4est->first_local_tree, k = 0;
       tt <= p4est->last_local_tree; ++tt) {
    tree = p4est_tree_array_index (p4est->trees, tt);   /* Current tree */
    tquadrants = &tree->quadrants;
    Q = (p4est_locidx_t) tquadrants->elem_count;
    for (q = 0; q < Q; ++q, ++k) {
      /* This is now a loop over all local elements.
       * Users might aggregate the above code into a more compact iterator. */
      quad = p4est_quadrant_array_index (tquadrants, q);

      /* We need to determine whether any node on this element is hanging. */
      anyhang = lnodes_decode2 (lnodes->face_code[q], hanging_corner);
      if (!anyhang) {
        parent = NULL;          /* Defensive programming. */
      }
      else {
        /* At least one node is hanging.  We need the parent quadrant to
         * find the location of the corresponding non-hanging node. */
        parent = &sp;
        p4est_quadrant_parent (quad, parent);
      }

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dmf;
  PetscInt       cStart, cEnd, c, n; // may need this for a local section?
  Vec exact;
  Vec exact_loc;
  p4est_t           *p4est;
  p4est_locidx_t     numquads;
  p4est_gloidx_t    *Nloc;
  PetscInt size;
  p4est_ghost_t      *ghost;
  p4est_lnodes_t     *lnodes;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  DMCreate(PETSC_COMM_WORLD, &dmf);
  DMSetType(dmf, "p4est");
  
  /*
  get the pforest
   */
  
  ierr = DMSetFromOptions(dmf);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmf, NULL, "-dm_view");CHKERRQ(ierr);
  p4est = ((DM_Forest_pforest*) ((DM_Forest*) dmf->data)->data)->forest;
  
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  lnodes = p4est_lnodes_new(p4est, ghost, 1);
  Nloc  = lnodes->global_owned_count;

  p4est_ghost_destroy (ghost);
  ghost = NULL;

  PetscPrintf(PETSC_COMM_WORLD, "Local node count: %i\n", *Nloc);

  
  VecCreate(PETSC_COMM_WORLD, &exact);
  VecSetSizes(exact, *Nloc, PETSC_DECIDE); //change from petsc_decide to the number of on each processor?
    
  VecSetFromOptions(exact);
  PetscObjectSetName((PetscObject)exact,"Approx. Solution");
  VecSet(exact, 0.0);
  VecGetSize(exact, &size);
  PetscPrintf(PETSC_COMM_WORLD, "Global node count: %i\n", size);
  //ierr = VecGetLocalVector(exact, exact_loc);
  
  /* p4est_iterate(p4est, NULL,
               exact,
             exact_solution,
          NULL,
#ifdef P4_TO_P8
         NULL,
#endif
       NULL);*/ // i want to pass the Vec directly to p4est_iterate, but for some reason this causes a segfault
  /* p4est_iterate(p4est,  NULL, NULL,
                 NULL,
                 NULL,  NULL);*/
  ierr = DMDestroy(&dmf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



