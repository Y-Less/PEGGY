#if !defined _TYPEDEF_H
#define _TYPEDEF_H

// Define a type and all relevant associated types (pointers, consts and
// references to the type).
/*#define TYPEDEF(m, n)                                                          \
	typedef                                                                    \
		m                              n##_t,                                  \
		const                          n##_tc,                                 \
		*                              n##_tp,                                 \
		const *                        n##_tcp,                                \
		* const                        n##_tpc,                                \
		const * const                  n##_tcpc,                               \
		&                              n##_tr,                                 \
		const &                        n##_tcr

// Define and typedef a struct and all relevant associated types (pointers,
// consts and references to the type).
#define STRUCT(n, m)                                                           \
	typedef                                                                    \
		struct _##n##_s m              n##_s,                                  \
		const                          n##_sc,                                 \
		*                              n##_sp,                                 \
		const *                        n##_scp,                                \
		* const                        n##_spc,                                \
		const * const                  n##_scpc,                               \
		&                              n##_sr,                                 \
		const &                        n##_scr

// Define and typedef a union and all relevant associated types (pointers,
// consts and references to the type).
#define UNION(n, m)                                                            \
	typedef                                                                    \
		union _##n##_u m               n##_u,                                  \
		const                          n##_uc,                                 \
		*                              n##_up,                                 \
		const *                        n##_ucp,                                \
		* const                        n##_upc,                                \
		const * const                  n##_ucpc,                               \
		&                              n##_ur,                                 \
		const &                        n##_ucr

// Define and typedef a union and all relevant associated types (pointers,
// consts and references to the type).
#define CLASS(n, m)                                                            \
	typedef                                                                    \
		union _##n##_u m               n##_u,                                  \
		const                          n##_uc,                                 \
		*                              n##_up,                                 \
		const *                        n##_ucp,                                \
		* const                        n##_upc,                                \
		const * const                  n##_ucpc,                               \
		&                              n##_ur,                                 \
		const &                        n##_ucr*/

// Define all relevant associated types to a regular type (pointers, consts and
// references to the type).
#define TYPES(n)                                                               \
	typedef n const                    n##c;                                   \
	typedef n *                        n##p;                                   \
	typedef n const *                  n##cp;                                  \
	typedef n * const                  n##pc;                                  \
	typedef n const * const            n##cpc;                                 \
	typedef n &                        n##r;                                   \
	typedef n const &                  n##cr;

#endif
